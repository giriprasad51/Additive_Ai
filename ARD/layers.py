import torch
import torch.nn as nn
from typing import Callable, Optional, Union

import functools
import operator

from transformers.activations import ACT2FN
from transformers.pytorch_utils import Conv1D
from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS
from transformers.cache_utils import Cache, EncoderDecoderCache
from transformers.models.gpt2.modeling_gpt2 import eager_attention_forward

from transformers.models.gpt2.modeling_gpt2 import GPT2MLP, GPT2Attention, GPT2Block
from transformers.activations import ACT2FN

class Conv1DARD(nn.Module):
    """
    1D-convolutional layer with ARD (Automatic Relevance Determination) capability.
    Based on the Conv1D implementation from OpenAI GPT but with ARD functionality.

    Args:
        nf (int): The number of output features.
        nx (int): The number of input features.
        thresh (float): Threshold for pruning weights (default: 3)
        ard_init (float): Initial value for log_sigma2 (default: -10)
    """

    def __init__(self, nf, nx, thresh=3, ard_init=-10):
        super().__init__()
        self.nf = nf
        self.nx = nx
        self.thresh = thresh
        self.ard_init = ard_init
        
        # Weight parameters
        self.weight = nn.Parameter(torch.empty(nx, nf))
        self.bias = nn.Parameter(torch.zeros(nf))
        
        # ARD parameters
        self.log_sigma2 = nn.Parameter(torch.empty(nx, nf))
        
        # Initialize parameters
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.normal_(self.weight, std=0.02)
        nn.init.constant_(self.bias, 0)
        nn.init.constant_(self.log_sigma2, self.ard_init)

    def forward(self, x):
        size_out = x.size()[:-1] + (self.nf,)
        
        if self.training:
            # Bayesian mode during training
            W_mu = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
            
            # Calculate variance term
            log_alpha = self.log_alpha
            W_var = torch.mm(x.view(-1, x.size(-1)).pow(2), 
                           torch.exp(log_alpha) * self.weight.pow(2))
            W_std = torch.sqrt(W_var + 1e-15)
            
            # Reparameterization trick
            epsilon = torch.randn_like(W_mu)
            x = W_mu + W_std * epsilon
        else:
            # Use clipped weights during inference
            W = self.weights_clipped
            x = torch.addmm(self.bias, x.view(-1, x.size(-1)), W)
        
        x = x.view(size_out)
        return x

    @property
    def weights_clipped(self):
        clip_mask = self.get_clip_mask()
        return torch.where(clip_mask, torch.zeros_like(self.weight), self.weight)

    def get_clip_mask(self):
        log_alpha = self.log_alpha
        return torch.ge(log_alpha, self.thresh)

    def get_reg(self):
        """
        Get weights regularization (KL(q(w)||p(w)) approximation)
        """
        k1, k2, k3 = 0.63576, 1.8732, 1.48695
        C = -k1
        log_alpha = self.log_alpha
        mdkl = k1 * torch.sigmoid(k2 + k3 * log_alpha) - \
            0.5 * torch.log1p(torch.exp(-log_alpha)) + C
        return -torch.sum(mdkl)

    @property
    def log_alpha(self):
        log_alpha = self.log_sigma2 - 2 * \
            torch.log(torch.abs(self.weight) + 1e-15)
        return torch.clamp(log_alpha, -10, 10)

    def get_dropped_params_cnt(self):
        """
        Get number of dropped weights (with log alpha greater than "thresh" parameter)
        :returns number of dropped weights
        """
        return self.get_clip_mask().sum().item()

    def __repr__(self) -> str:
        return f"Conv1DARD(nf={self.nf}, nx={self.nx}, thresh={self.thresh})"


class GPT2MLPARD(GPT2MLP):
    def __init__(self, intermediate_size, config):
        super().__init__(intermediate_size, config)
        self.c_fc = Conv1DARD(intermediate_size, config.hidden_size,)
        self.c_proj = Conv1DARD(config.hidden_size, intermediate_size,)

    def get_reg(self):
        return self.c_fc.get_reg() + self.c_proj.get_reg()

    def get_dropped_params_cnt(self):
        return self.c_fc.get_dropped_params_cnt() + self.c_proj.get_dropped_params_cnt()


class GPT2AttentionARD(GPT2Attention):
    def __init__(self, config, is_cross_attention=False, layer_idx=None):
        super().__init__(config, is_cross_attention=is_cross_attention, layer_idx=layer_idx)
        if is_cross_attention:
            self.c_attn = Conv1DARD(2 * self.embed_dim, self.embed_dim,)
            self.q_attn = Conv1DARD(self.embed_dim, self.embed_dim,)
        else:
            self.c_attn = Conv1DARD(3 * self.embed_dim, self.embed_dim,)
        self.c_proj = Conv1DARD(self.embed_dim, self.embed_dim,)

    def get_reg(self):
        reg = self.c_attn.get_reg() + self.c_proj.get_reg()
        if hasattr(self, 'q_attn'):
            reg += self.q_attn.get_reg()
        return reg

    def get_dropped_params_cnt(self):
        cnt = self.c_attn.get_dropped_params_cnt() + self.c_proj.get_dropped_params_cnt()
        if hasattr(self, 'q_attn'):
            cnt += self.q_attn.get_dropped_params_cnt()
        return cnt
    
class LayerNormARD(nn.Module):
    def __init__(self, normalized_shape, eps=1e-5, thresh=3, ard_init=-10):
        super().__init__()
        self.normalized_shape = (normalized_shape,) if isinstance(normalized_shape, int) else normalized_shape
        self.eps = eps
        self.thresh = thresh
        self.ard_init = ard_init
        
        # Standard LayerNorm parameters
        self.weight = nn.Parameter(torch.Tensor(*self.normalized_shape))
        self.bias = nn.Parameter(torch.Tensor(*self.normalized_shape))
        
        # ARD parameters (log variance for weight and bias)
        self.log_sigma2_weight = nn.Parameter(torch.Tensor(*self.normalized_shape))
        self.log_sigma2_bias = nn.Parameter(torch.Tensor(*self.normalized_shape))
        
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.ones_(self.weight)
        nn.init.zeros_(self.bias)
        nn.init.constant_(self.log_sigma2_weight, self.ard_init)
        nn.init.constant_(self.log_sigma2_bias, self.ard_init)

    @property
    def log_alpha_weight(self):
        log_alpha = self.log_sigma2_weight - 2 * torch.log(abs(self.weight) + 1e-15)
        return torch.clamp(log_alpha, -10, 10)

    @property
    def log_alpha_bias(self):
        log_alpha = self.log_sigma2_bias - 2 * torch.log(abs(self.bias) + 1e-15)
        return torch.clamp(log_alpha, -10, 10)

    def get_clip_mask(self, log_alpha):
        return torch.ge(log_alpha, self.thresh)
    
    def get_dropped_params_cnt(self):
        """
        Get number of dropped weights (with log alpha greater than "thresh" parameter)
        :returns number of dropped weights
        """
        return self.get_clip_mask(self.log_alpha_weight).sum().item() +  self.get_clip_mask(self.log_alpha_bias).sum().item()

    def forward(self, x):
        # Compute mean/variance (same as standard LayerNorm)
        mean = x.mean(-1, keepdim=True)
        var = x.var(-1, keepdim=True, unbiased=False)
        x_normalized = (x - mean) / torch.sqrt(var + self.eps)

        if self.training:
            # Bayesian mode: Sample weight/bias
            std_weight = torch.exp(0.5 * self.log_sigma2_weight)
            std_bias = torch.exp(0.5 * self.log_sigma2_bias)
            weight_sample = self.weight + std_weight * torch.randn_like(self.weight)
            bias_sample = self.bias + std_bias * torch.randn_like(self.bias)
            return x_normalized * weight_sample + bias_sample
        else:
            # Inference: Use clipped weights (prune where log_alpha > thresh)
            mask_weight = self.get_clip_mask(self.log_alpha_weight)
            mask_bias = self.get_clip_mask(self.log_alpha_bias)
            weight = torch.where(mask_weight, torch.zeros_like(self.weight), self.weight)
            bias = torch.where(mask_bias, torch.zeros_like(self.bias), self.bias)
            return x_normalized * weight + bias

    def get_reg(self):
        """KL divergence regularization for weight and bias."""
        k1, k2, k3 = 0.63576, 1.8732, 1.48695
        C = -k1
        
        def kl(log_alpha):
            return k1 * torch.sigmoid(k2 + k3 * log_alpha) - 0.5 * torch.log1p(torch.exp(-log_alpha)) + C
        
        reg_weight = -torch.sum(kl(self.log_alpha_weight))
        reg_bias = -torch.sum(kl(self.log_alpha_bias))
        return reg_weight + reg_bias


class GPT2BlockARD(GPT2Block):
    def __init__(self,  config, layer_idx=None):
        super().__init__(config, layer_idx=layer_idx)
        hidden_size = config.hidden_size
        inner_dim = config.n_inner if config.n_inner is not None else 4 * hidden_size

        self.ln_1 = LayerNormARD(hidden_size, eps=config.layer_norm_epsilon)
        self.attn = GPT2AttentionARD(config, layer_idx=layer_idx)
        self.ln_2 = LayerNormARD(hidden_size, eps=config.layer_norm_epsilon)
        if config.add_cross_attention:
            self.crossattention = GPT2Attention(config=config, is_cross_attention=True, layer_idx=layer_idx)
            self.ln_cross_attn = nn.LayerNorm(hidden_size, eps=config.layer_norm_epsilon)

        self.mlp = GPT2MLPARD(4 * config.n_embd, config)



class LinearARD(torch.nn.Module):
    def __init__(self, in_features, out_features, bias=True, thresh=3, ard_init=-10):
        super(LinearARD, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = torch.nn.Parameter(torch.Tensor(out_features, in_features))
        self.thresh = thresh
        if bias:
            self.bias = torch.nn.Parameter(torch.Tensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.ard_init = ard_init
        self.log_sigma2 = torch.nn.Parameter(torch.Tensor(out_features, in_features))
        self.reset_parameters()

    def forward(self, input):
        if self.training:
            W_mu = torch.nn.functional.linear(input, self.weight)
            std_w = torch.exp(self.log_alpha).permute(1,0)
            W_std = torch.sqrt((input.pow(2)).matmul(std_w*(self.weight.permute(1,0)**2)) + 1e-15)

            epsilon = W_std.new(W_std.shape).normal_()
            output = W_mu + W_std * epsilon
            if self.bias is not None : output += self.bias
        else:
            W = self.weights_clipped
            output = torch.nn.functional.linear(input, W) + self.bias
        return output

    @property
    def weights_clipped(self):
        clip_mask = self.get_clip_mask()
        return torch.where(clip_mask, torch.zeros_like(self.weight), self.weight)

    def reset_parameters(self):
        self.weight.data.normal_(0, 0.02)
        if self.bias is not None:
            self.bias.data.zero_()
        self.log_sigma2.data.fill_(self.ard_init)

    def get_clip_mask(self):
        log_alpha = self.log_alpha
        return torch.ge(log_alpha, self.thresh)

    def get_reg(self, **kwargs):
        """
        Get weights regularization (KL(q(w)||p(w)) approximation)
        """
        k1, k2, k3 = 0.63576, 1.8732, 1.48695
        C = -k1
        mdkl = k1 * torch.sigmoid(k2 + k3 * self.log_alpha) - \
            0.5 * torch.log1p(torch.exp(-self.log_alpha)) + C
        return -torch.sum(mdkl)

    def extra_repr(self):
        return 'in_features={}, out_features={}, bias={}'.format(
            self.in_features, self.out_features, self.bias is not None
        )

    def get_dropped_params_cnt(self):
        """
        Get number of dropped weights (with log alpha greater than "thresh" parameter)
        :returns (number of dropped weights, number of all weight)
        """
        return self.get_clip_mask().sum().cpu().numpy()

    @property
    def log_alpha(self):
        log_alpha = self.log_sigma2 - 2 * \
            torch.log(torch.abs(self.weight) + 1e-15)
        return torch.clamp(log_alpha, -10, 10)


class Conv2dARD(torch.nn.Conv2d):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=0, dilation=1, groups=1, ard_init=-10, thresh=3):
        bias = False  # Goes to nan if bias = True
        super(Conv2dARD, self).__init__(in_channels, out_channels, kernel_size, stride,
                                        padding, dilation, groups, bias)
        self.bias = None
        self.thresh = thresh
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.ard_init = ard_init
        self.log_sigma2 = torch.nn.Parameter(ard_init * torch.ones_like(self.weight))
        # self.log_sigma2 = Parameter(2 * torch.log(torch.abs(self.weight) + eps).clone().detach()+ard_init*torch.ones_like(self.weight))

    def forward(self, input):
        """
        Forward with all regularized connections and random activations (Beyesian mode). Typically used for train
        """
        if self.training == False:
            return torch.nn.functional.conv2d(input, self.weights_clipped,
                            self.bias, self.stride,
                            self.padding, self.dilation, self.groups)
        W = self.weight

        conved_mu = torch.nn.functional.conv2d(input, W, self.bias, self.stride,
                             self.padding, self.dilation, self.groups)
        log_alpha = self.log_alpha
        conved_si = torch.sqrt(1e-15 + torch.nn.functional.conv2d(input * input,
                                                torch.exp(log_alpha) * W *
                                                W, self.bias, self.stride,
                                                self.padding, self.dilation, self.groups))
        conved = conved_mu + \
            conved_si * \
            torch.normal(torch.zeros_like(conved_mu),
                         torch.ones_like(conved_mu))
        return conved

    @property
    def weights_clipped(self):
        clip_mask = self.get_clip_mask()
        return torch.where(clip_mask, torch.zeros_like(self.weight), self.weight)

    def get_clip_mask(self):
        log_alpha = self.log_alpha
        return torch.ge(log_alpha, self.thresh)

    def get_reg(self, **kwargs):
        """
        Get weights regularization (KL(q(w)||p(w)) approximation)
        """
        k1, k2, k3 = 0.63576, 1.8732, 1.48695
        C = -k1
        log_alpha = self.log_alpha
        mdkl = k1 * torch.sigmoid(k2 + k3 * log_alpha) - \
            0.5 * torch.log1p(torch.exp(-log_alpha)) + C
        return -torch.sum(mdkl)

    def extra_repr(self):
        return 'in_features={}, out_features={}, bias={}'.format(
            self.in_channels, self.out_channels, self.bias is not None
        )

    def get_dropped_params_cnt(self):
        """
        Get number of dropped weights (greater than "thresh" parameter)

        :returns (number of dropped weights, number of all weight)
        """
        return self.get_clip_mask().sum().cpu().numpy()

    @property
    def log_alpha(self):
        log_alpha = self.log_sigma2 - 2 * \
            torch.log(torch.abs(self.weight) + 1e-15)
        return torch.clamp(log_alpha, -8, 8)


class ELBOLoss(torch.nn.Module):
    def __init__(self, net, loss_fn):
        super(ELBOLoss, self).__init__()
        self.loss_fn = loss_fn
        self.net = net

    def forward(self, input, target, loss_weight=1., kl_weight=1.):
        assert not target.requires_grad
        # Estimate ELBO
        return loss_weight * self.loss_fn(input, target)  \
            + kl_weight * get_ard_reg(self.net)


def get_ard_reg(module):
    """
    :param module: model to evaluate ard regularization for
    :param reg: auxilary cumulative variable for recursion
    :return: total regularization for module
    """
    if hasattr(module, 'get_reg'):
        # Case 1: Module is an ARD layer (LinearARD, Conv1DARD, LayerNormARD, etc.)
        return module.get_reg()
    elif hasattr(module, 'children'):
        return sum([get_ard_reg(submodule) for submodule in module.children()])
    return 0


def _get_dropped_params_cnt(module):
    if hasattr(module, 'get_dropped_params_cnt'):
        return module.get_dropped_params_cnt()
    elif hasattr(module, 'children'):
        return sum([_get_dropped_params_cnt(submodule) for submodule in module.children()])
    return 0


def _get_params_cnt(module):
    if any([isinstance(module, l) for l in [LinearARD, Conv2dARD]]):
        return functools.reduce(operator.mul, module.weight.shape, 1)
    elif hasattr(module, 'children'):
        return sum(
            [_get_params_cnt(submodule) for submodule in module.children()])
    return sum(p.numel() for p in module.parameters())


def get_dropped_params_ratio(model):
    return _get_dropped_params_cnt(model) * 1.0 / _get_params_cnt(model)

