import torch
import torch.nn as nn
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.autograd import Function

class AllGatherWithGrad(Function):
    @staticmethod
    def forward(ctx, tensor, gathered_tensors, split_sizes, world_size, rank):
        ctx.world_size = world_size
        ctx.rank = rank
        ctx.split_sizes = split_sizes
        dist.all_gather(gathered_tensors, tensor)
        
        return torch.cat(gathered_tensors, dim=1)

    @staticmethod
    def backward(ctx, grad_output):
        world_size = ctx.world_size
        rank = ctx.rank
        
        all_grad = torch.split(grad_output, ctx.split_sizes, dim=1)
        local_grad = all_grad[rank].contiguous()
    
        return local_grad, None, None, None, None # `None` for `world_size` argument


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
        self.rem = self.out_channels % world_size
        self.split_channels = [self.out_channels // world_size + (1 if i < self.rem else 0) for i in range(world_size) ]
        
        
        # Assign extra channels to some ranks to distribute remainder
        self.local_out_channels = self.split_channels[self.rank]

        # Define Conv layer for the split
        self.conv = nn.Conv2d(self.in_channels, self.local_out_channels, 
                              kernel_size=self.kernel_size, stride=self.stride, 
                              padding=self.padding).to(self.device)
        # print(self.conv)

        # Copy weights
        self.copy_weights_from(conv_layer)

    def forward(self, x):
        x = x.to(self.device)
        self.output = self.conv(x)

        torch.cuda.synchronize(self.device)
        if self.combine:
            gathered_outputs = []
            for i in range(self.world_size):
                recv_shape = list(self.output.shape)
                recv_shape[1] = self.split_channels[i]
                gathered_outputs.append(torch.zeros(recv_shape,  device=self.device))
            

            gathered_output = AllGatherWithGrad.apply(self.output, gathered_outputs, self.split_channels, self.world_size, self.rank)
            return gathered_output
            
            
        return self.output
    
    
    def copy_weights_from(self, original_layer: nn.Conv2d):
        """Distributes weights and biases from the original Conv2d layer"""
        with torch.no_grad():
            start_idx = sum(self.split_channels[:self.rank])
            end_idx = start_idx + self.local_out_channels

            self.conv.weight.copy_(original_layer.weight[start_idx:end_idx])
            self.conv.bias.copy_(original_layer.bias[start_idx:end_idx])

        self.conv.weight.requires_grad = True  # Ensure gradients flow
        self.conv.bias.requires_grad = True

