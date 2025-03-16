import torch
import torch.nn as nn
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP


class DistributedOutputChannelSplitConv2d(nn.Module):
    def __init__(self, conv_layer: nn.Conv2d, world_size, rank, combine=True):
        super().__init__()
        assert conv_layer.out_channels % world_size == 0, "Output channels must be divisible by world_size"

        self.device = torch.device(f"cuda:{rank}")
        self.world_size = world_size
        self.rank = rank
        self.combine = combine
        self.split_channels = conv_layer.out_channels // world_size

        # Define Conv layer for the split
        self.conv = nn.Conv2d(conv_layer.in_channels, self.split_channels, 
                              kernel_size=conv_layer.kernel_size,
                              stride=conv_layer.stride, padding=conv_layer.padding).to(self.device)
        
        # Synchronize weights
        self.copy_weights_from(conv_layer)

        

    def forward(self, x):
        x = x.to(self.device)
        # print(self.conv.weight.data)
        output = self.conv(x)

        if self.combine:
            # Create a tensor to hold all gathered outputs
            gathered_output = torch.zeros(
                (output.shape[0], self.split_channels * self.world_size, *output.shape[2:]), device=self.device
            )

            # Gather results
            dist.all_gather_into_tensor(gathered_output, output)

            if self.rank == 0:
                # print("---------",gathered_output)
                return gathered_output
            
        # print("output",output)
        return output

    def copy_weights_from(self, original_layer: nn.Conv2d):
        with torch.no_grad():
            weights = original_layer.weight[self.rank * self.split_channels:(self.rank + 1) * self.split_channels].clone()
            biases = original_layer.bias[self.rank * self.split_channels:(self.rank + 1) * self.split_channels].clone()

            self.conv.weight.data.copy_(weights.to(self.device))
            self.conv.bias.data.copy_(biases.to(self.device))


            print(f"Rank {self.rank} - Weights Copied: Mean {self.conv.weight.mean()}, Bias Mean {self.conv.bias.mean()}")
