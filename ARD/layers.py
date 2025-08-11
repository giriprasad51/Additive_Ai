import torch
import torch.nn as nn
from typing import Callable, Optional, Union

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
        self.c_fc = Conv1DARD(intermediate_size, config.hidden_size,
                              thresh=config.ard_thresh, ard_init=config.ard_init)
        self.c_proj = Conv1DARD(config.hidden_size, intermediate_size,
                                thresh=config.ard_thresh, ard_init=config.ard_init)

    def get_reg(self):
        return self.c_fc.get_reg() + self.c_proj.get_reg()

    def get_dropped_params_cnt(self):
        return self.c_fc.get_dropped_params_cnt() + self.c_proj.get_dropped_params_cnt()


class GPT2AttentionARD(GPT2Attention):
    def __init__(self, config, is_cross_attention=False, layer_idx=None):
        super().__init__(config, is_cross_attention=is_cross_attention, layer_idx=layer_idx)
        if is_cross_attention:
            self.c_attn = Conv1DARD(2 * self.embed_dim, self.embed_dim,
                                    thresh=config.ard_thresh, ard_init=config.ard_init)
            self.q_attn = Conv1DARD(self.embed_dim, self.embed_dim,
                                    thresh=config.ard_thresh, ard_init=config.ard_init)
        else:
            self.c_attn = Conv1DARD(3 * self.embed_dim, self.embed_dim,
                                    thresh=config.ard_thresh, ard_init=config.ard_init)
        self.c_proj = Conv1DARD(self.embed_dim, self.embed_dim,
                                thresh=config.ard_thresh, ard_init=config.ard_init)

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


class GPT2BlockARD(GPT2Block):
    def __init__(self, n_ctx, config, scale=False):
        super().__init__(n_ctx, config, scale=scale)
        self.attn = GPT2AttentionARD(config, layer_idx=getattr(config, "layer_idx", None))
        self.mlp = GPT2MLPARD(4 * config.n_embd, config)

