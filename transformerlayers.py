import torch
import torch.nn as nn
from .maths import sum_random_nums_n, moe_masked
from .layers import ParallelActivations


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
        splits = torch.split(x, self.split_sizes, dim=-1)
        outputs = [layer(split) for layer, split in zip(self.split_layers, splits)]

        if self.skipconnections:
            outputs = [sum(outputs) + o for o in outputs]

        return sum(outputs) if self.combine else outputs


class ParallelGPT2MLP(nn.Module):
    def __init__(self, gpt2mlp, combine=True, num_splits=4,split_channels=None):
        gpt2mlp.c_fc = OutputChannelSplitConv1DGPT(gpt2mlp.c_fc,combine=False, num_splits=num_splits, split_channels=split_channels)
        gpt2mlp.act = ParallelActivations(gpt2mlp.act, combine=False)
        gpt2mlp.c_proj = InputChannelSplitConv1DGPT(gpt2mlp.c_fc,combine=combine, num_splits=num_splits, split_channels=split_channels)
        gpt2mlp.dropout = ParallelActivations(gpt2mlp.dropout, combine=False)