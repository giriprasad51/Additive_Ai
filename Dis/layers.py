import torch
import torch.nn as nn
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP


class DistributedOutputChannelSplitConv2d(nn.Module):
    def __init__(self, conv_layer: nn.Conv2d, world_size, rank, combine=True):
        super().__init__()
        # assert conv_layer.out_channels % world_size == 0, "Output channels must be divisible by world_size"

        self.device = torch.device(f"cuda:{rank}")
        self.world_size = world_size
        self.rank = rank
        self.combine = combine
        self.out_channels = conv_layer.out_channels
        self.in_channels = conv_layer.in_channels
        self.kernel_size = conv_layer.kernel_size
        self.stride = conv_layer.stride
        self.padding = conv_layer.padding
        
        # Compute split sizes
        self.split_channels = self.out_channels // world_size
        self.rem = self.out_channels % world_size
        
        # Assign extra channels to some ranks to distribute remainder
        self.local_out_channels = self.split_channels + (1 if rank < self.rem else 0)

        # Define Conv layer for the split
        self.conv = nn.Conv2d(self.in_channels, self.local_out_channels, 
                              kernel_size=self.kernel_size, stride=self.stride, 
                              padding=self.padding).to(self.device)
        # print(self.conv)

        # Copy weights
        self.copy_weights_from(conv_layer)

    def forward(self, x):
        x = x.to(self.device)
        output = self.conv(x)
        
        torch.cuda.synchronize(self.device)
        # print(f"Rank {self.rank}: Local output shape {output.shape}")

        if self.combine:
            if self.rank == 0:
                # Rank 0 gathers outputs from all other ranks
                gathered_outputs = [output]
                for src in range(1, self.world_size):
                    recv_shape = list(output.shape)
                    recv_shape[1] = self.split_channels + (1 if src < self.rem else 0)
                    recv_tensor = torch.zeros(recv_shape, device=self.device)
                    dist.recv(recv_tensor, src=src)
                    gathered_outputs.append(recv_tensor)

                gathered_output = torch.cat(gathered_outputs, dim=1)
                return gathered_output

            else:
                # Other ranks send their outputs to rank 0
                dist.send(output, dst=0)
            
        return output

    def copy_weights_from(self, original_layer: nn.Conv2d):
        """Distributes weights and biases from the original Conv2d layer"""
        with torch.no_grad():
            start_idx = sum(self.split_channels + (1 if i < self.rem else 0) for i in range(self.rank))
            end_idx = start_idx + self.local_out_channels

            self.conv.weight.copy_(original_layer.weight[start_idx:end_idx])
            self.conv.bias.copy_(original_layer.bias[start_idx:end_idx])

            # print(f"Rank {self.rank} - Weights Copied: Mean {self.conv.weight.mean()}, Bias Mean {self.conv.bias.mean()}")