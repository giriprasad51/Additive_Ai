import torch
import torch.nn as nn
import torch.nn.functional as F

import math
from typing import Optional, List

class LoRALayer():
    def __init__(
        self, 
        r: int, 
        lora_alpha: int, 
        lora_dropout: float,
        merge_weights: bool,
    ):
        self.r = r
        self.lora_alpha = lora_alpha
        # Optional dropout
        if lora_dropout > 0.:
            self.lora_dropout = nn.Dropout(p=lora_dropout)
        else:
            self.lora_dropout = lambda x: x
        # Mark the weight as unmerged
        self.merged = False
        self.merge_weights = merge_weights


class OutputChannelSplitLinear(nn.Module):
    def __init__(self, linear_layer: nn.Linear, dropout_per=0.5, mode='both'):
        super().__init__()
        self.weight = linear_layer.weight
        self.bias = linear_layer.bias

        self.in_features = linear_layer.in_features
        self.out_features = linear_layer.out_features

        self.device = self.weight.device 
        self.mode = mode
        # freeze original
        self.weight.requires_grad = False
        if self.bias is not None:
            self.bias.requires_grad = False
        
        # learnable split
        split_out = int(self.out_features * dropout_per)
        self.weight_s = nn.Parameter(torch.zeros(split_out, self.in_features, device=self.device))
        self.bias_s = nn.Parameter(torch.zeros(split_out, device=self.device)) if self.bias is not None else None

    def forward(self, x):
        
        output = []
        if self.mode in ["large", "both"]:
            result = F.linear(x, self.weight, self.bias)
            output.append(result)
        if self.mode in ["large", "both"]:
            result_s = F.linear(x, self.weight_s, self.bias_s)
            output.append(result_s)

        return output
    
    def removeweight(self):
        del self.weight,self.bias


class InputChannelSplitLinear(nn.Module):
    def __init__(self, linear_layer: nn.Linear, dropout_per=0.5, mode='both'):
        super().__init__()
        self.weight = linear_layer.weight
        self.bias = linear_layer.bias

        self.in_features = linear_layer.in_features
        self.out_features = linear_layer.out_features

        self.device = self.weight.device 
        self.mode = mode

        # freeze original
        self.weight.requires_grad = False
        if self.bias is not None:
            self.bias.requires_grad = False

        # learnable split
        split_in = int(self.in_features * dropout_per)
        self.weight_s = nn.Parameter(torch.zeros(self.out_features, split_in, device=self.device))
        self.bias_s = nn.Parameter(torch.zeros(self.out_features, device=self.device)) if self.bias is not None else None

    def forward(self, x_pair):
        x_main, x_split = x_pair
        
        result = 0
        if self.mode in ["large", "both"]:
            result = F.linear(x_main, self.weight, self.bias)
        if self.mode in ["large", "both"]:
            result += F.linear(x_split, self.weight_s, self.bias_s)
        return result /2  # additive correction
    
    def removeweight(self):
        del self.weight,self.bias
    


class OutputChannelSplitConv2d(nn.Module):
    def __init__(self, conv_layer: nn.Conv2d, dropout_per=0.5, mode='both'):
        super().__init__()
        # frozen pretrained conv
        self.weight = conv_layer.weight
        self.bias = conv_layer.bias

        self.in_channels = conv_layer.in_channels
        self.out_channels = conv_layer.out_channels
        self.kernel_size = conv_layer.kernel_size
        self.stride = conv_layer.stride
        self.padding = conv_layer.padding
        self.dilation = conv_layer.dilation
        self.groups = conv_layer.groups

        self.device = self.weight.device 
        self.mode = mode

        self.weight.requires_grad = False
        if self.bias is not None:
            self.bias.requires_grad = False

        # learnable split (extra out channels)
        split_out = int(self.out_channels * dropout_per)
        self.weight_s = nn.Parameter(torch.zeros(
            split_out, self.in_channels // self.groups, *self.kernel_size, device=self.device
        ))
        self.bias_s = nn.Parameter(torch.zeros(split_out, device=self.device)) if self.bias is not None else None

    def forward(self, x):
        
        output = []
        if self.mode in ["large", "both"]:
            result = F.conv2d(
                x, self.weight, self.bias,
                stride=self.stride, padding=self.padding,
                dilation=self.dilation, groups=self.groups
            )
            output.append(result)

        if self.mode in ["large", "both"]:
            result_s = F.conv2d(
                x, self.weight_s, self.bias_s,
                stride=self.stride, padding=self.padding,
                dilation=self.dilation, groups=self.groups
            )
            output.append(result_s)
        return output
    
    def removeweight(self):
        del self.weight,self.bias

    
class InputChannelSplitConv2d(nn.Module):
    def __init__(self, conv_layer: nn.Conv2d, dropout_per=0.5, mode='both'):
        super().__init__()
        # frozen pretrained conv
        self.weight = conv_layer.weight
        self.bias = conv_layer.bias

        self.in_channels = conv_layer.in_channels
        self.out_channels = conv_layer.out_channels
        self.kernel_size = conv_layer.kernel_size
        self.stride = conv_layer.stride
        self.padding = conv_layer.padding
        self.dilation = conv_layer.dilation
        self.groups = conv_layer.groups

        self.device = self.weight.device
        self.mode = mode 

        self.weight.requires_grad = False
        if self.bias is not None:
            self.bias.requires_grad = False

        # learnable split (extra in channels)
        split_in = int(self.in_channels * dropout_per)
        self.weight_s = nn.Parameter(torch.zeros(
            self.out_channels, split_in // self.groups, *self.kernel_size, device=self.device
        ))
        self.bias_s = nn.Parameter(torch.zeros(self.out_channels, device=self.device)) if self.bias is not None else None

    def forward(self, x_pair):
        print(self.in_channels, self.out_channels)
        x_main, x_split = x_pair  # (main channels, split channels)
        
        result = 0
        if self.mode in ["large", "both"]:
            result = F.conv2d(
                x_main, self.weight, self.bias,
                stride=self.stride, padding=self.padding,
                dilation=self.dilation, groups=self.groups
            )
        if self.mode in ["large", "both"]:
            result += F.conv2d(
                x_split, self.weight_s, self.bias_s,
                stride=self.stride, padding=self.padding,
                dilation=self.dilation, groups=self.groups
            )
        return result*0.5
    
    def removeweight(self):
        del self.weight,self.bias

class ParallelActivations(nn.Module):
    def __init__(self, activation =nn.ReLU(), dropout = False):
        super(ParallelActivations, self).__init__()
        self.activation = activation
        self.droupout = dropout

    def forward(self, x):
       
        if isinstance(x, list):
            
            activation_outputs = [self.activation(tensor) for tensor in x]
            if (not self.droupout ) and  self.activation == nn.Dropout:
                activation_outputs[1] = x[1]
            
            return activation_outputs, 
                
        elif isinstance(x, torch.Tensor):
            return self.activation(x)
        else:
            raise TypeError("Input must be a Tensor or a list of Tensors")

