import torch
import torch.nn as nn
from .maths import sum_random_nums_n, moe_masked
from .layers import ParallelActivations, OutputChannelSplitLinear, InputChannelSplitLinear
from transformers.pytorch_utils import Conv1D
from transformers.models.gpt2.modeling_gpt2 import GPT2MLP, GPT2Attention, GPT2Block
from transformers.activations import ACT2FN


class OutputChannelSplitConv1DGPT(nn.Module):
    def __init__(self, conv1d_layer, num_splits=4, split_channels=None, combine=True):
        super().__init__()
        self.weight = conv1d_layer.weight
        self.bias = conv1d_layer.bias
        self.device = self.weight.device

        self.in_features = self.weight.shape[0]
        self.out_features = self.weight.shape[1]

        self.combine = combine
        self.num_splits = num_splits
        self.rem = self.out_features % num_splits

        if split_channels:
            self.split_sizes = split_channels
        else:
            self.split_sizes = [self.out_features // num_splits + (1 if i < self.rem else 0) for i in range(num_splits)]

        self.split_layers = nn.ModuleList([
            nn.Linear(self.in_features, self.split_sizes[i]).to(self.device)
            for i in range(len(self.split_sizes))
        ])

        self.copy_weights_from()

    def copy_weights_from(self):
        with torch.no_grad():
            start = 0
            for i, layer in enumerate(self.split_layers):
                end = start + self.split_sizes[i]
                layer.weight.copy_(self.weight[:, start:end].T)
                layer.bias.copy_(self.bias[start:end])
                start = end

    def forward(self, x):
        if isinstance(x, list):
            out = [layer(x[i]) for i, layer in enumerate(self.split_layers)]
        else:
            out = [layer(x) for layer in self.split_layers]
        return torch.cat(out, dim=-1) if self.combine else out

class InputChannelSplitConv1DGPT(nn.Module):
    def __init__(self, conv1d_layer, num_splits=4, split_channels=None, combine=True, skipconnections=False):
        super().__init__()
        self.weight = conv1d_layer.weight
        self.bias = conv1d_layer.bias
        self.device = self.weight.device

        self.in_features = self.weight.shape[0]
        self.out_features = self.weight.shape[1]

        self.combine = combine
        self.skipconnections = skipconnections
        self.num_splits = num_splits
        self.rem = self.in_features % num_splits

        if split_channels:
            self.split_sizes = split_channels
        else:
            self.split_sizes = [self.in_features // num_splits + (1 if i < self.rem else 0) for i in range(num_splits)]

        self.split_layers = nn.ModuleList([
            nn.Linear(self.split_sizes[i], self.out_features).to(self.device)
            for i in range(len(self.split_sizes))
        ])

        self.copy_weights_from()

    def copy_weights_from(self):
        with torch.no_grad():
            start = 0
            for i, layer in enumerate(self.split_layers):
                end = start + self.split_sizes[i]
                layer.weight.copy_(self.weight[start:end, :].T)
                layer.bias.copy_(self.bias / len(self.split_layers))
                start = end

    def forward(self, x):
        if isinstance(x, list):
            outputs = [layer(x[i]) for i, layer in enumerate(self.split_layers)]
        else:
            splits = torch.split(x, self.split_sizes, dim=-1)
            outputs = [layer(split) for layer, split in zip(self.split_layers, splits)]

        

        return sum(outputs) if self.combine else outputs


class ParallelGPT2MLP(nn.Module):
    def __init__(self, gpt2mlp, combine=True, num_splits=4,split_channels=None):
        gpt2mlp.c_fc = OutputChannelSplitConv1DGPT(gpt2mlp.c_fc,combine=False, num_splits=num_splits, split_channels=split_channels)
        gpt2mlp.act = ParallelActivations(gpt2mlp.act, combine=False)
        gpt2mlp.c_proj = InputChannelSplitConv1DGPT(gpt2mlp.c_fc,combine=combine, num_splits=num_splits, split_channels=split_channels)
        gpt2mlp.dropout = ParallelActivations(gpt2mlp.dropout, combine=False)

class ParallelGPT2MLP(nn.Module):
    def __init__(self, gpt2mlp, num_splits=4, split_channels=None, combine=True):
        super().__init__()
        self.num_splits = num_splits
        self.split_channels = split_channels
        self.combine = combine
        
        # Parallelize the first dense layer (output channel split)
        self.c_fc = OutputChannelSplitConv1DGPT(
            gpt2mlp.c_fc, 
            num_splits=num_splits,
            split_channels=split_channels,
            combine=False  # Keep outputs separate for parallel processing
        )
        
        # Parallel activation function
        self.act = ParallelActivations(
            gpt2mlp.act,
            combine=False
        )
        
        # Parallelize the second dense layer (input channel split)
        self.c_proj = InputChannelSplitConv1DGPT(
            gpt2mlp.c_proj,
            num_splits=num_splits,
            split_channels=split_channels,
            combine=combine  # Combine at the end unless specified otherwise
        )
        
        # Parallel dropout
        self.dropout = ParallelActivations(
            gpt2mlp.dropout,
            combine=combine
        )
        
        
    def forward(self, hidden_states):
        # First dense layer (split output channels)
        hidden_states = self.c_fc(hidden_states)
        
        # Activation function (processed in parallel)
        hidden_states = self.act(hidden_states)
        
        # Second dense layer (split input channels)
        hidden_states = self.c_proj(hidden_states)
        
        # Dropout (processed in parallel)
        hidden_states = self.dropout(hidden_states)
        
        return hidden_states
    
class GPT2AttentionSplit(GPT2Attention):
    def __init__(self, config, is_cross_attention=False, layer_idx=None, dropout=1):
        super().__init__(config, is_cross_attention=is_cross_attention, layer_idx=layer_idx)
        
        # Original dimensions
        self.embed_dim = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = int(self.embed_dim * dropout)// self.num_heads
        
        # Calculate reduced dimensions with dropout
        self.reduced_embed_dim = int(self.embed_dim * dropout)
        # Ensure the reduced dimension is divisible by num_heads
        self.reduced_head_dim = self.reduced_embed_dim // self.num_heads
        self.reduced_embed_dim = self.reduced_head_dim * self.num_heads  # Adjust to maintain divisibility
        
        # Update projection layers with reduced dimensions
        if self.is_cross_attention:
            self.c_attn = Conv1D(2 * self.reduced_embed_dim, self.embed_dim)
            self.q_attn = Conv1D(self.reduced_embed_dim, self.embed_dim)
        else:
            self.c_attn = Conv1D(3 * self.reduced_embed_dim, self.embed_dim)
        self.c_proj = Conv1D(self.embed_dim, self.reduced_embed_dim)
        
        # Update split size for the reduced dimension
        self.split_size = self.reduced_embed_dim

class DeepseekV2MLP1(nn.Module):
    def __init__(self, config, hidden_size=None, intermediate_size=None):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size if hidden_size is None else hidden_size
        self.intermediate_size = (
            config.intermediate_size if intermediate_size is None else intermediate_size
        )

        self.gate_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.up_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.down_proj = nn.Linear(self.intermediate_size, self.hidden_size, bias=False)
        self.act_fn = ACT2FN[config.hidden_act]

        self.gate_proj = OutputChannelSplitLinear(self.gate_proj, num_splits=6, combine = False)
        self.up_proj = OutputChannelSplitLinear(self.up_proj, num_splits=6, combine = False)
        self.act_fn = ParallelActivations(self.act_fn, combine=False)
        self.down_proj =  InputChannelSplitLinear(self.down_proj, num_splits=6, combine = False)

    def forward(self, x):
        x1 = self.act_fn(self.gate_proj(x)[0])
        x2 = self.up_proj(x)[0]
        down_proj = self.down_proj([x1[i] * x2[i] for i in range(len(x1))])
        return down_proj
