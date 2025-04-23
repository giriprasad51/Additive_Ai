import torch
import torch.nn as nn
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.autograd import Function

class AllGatherOutputSplit(Function):
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

class AllReduceInputSplit(torch.autograd.Function):
    @staticmethod
    def forward(ctx, local_output):
        # No need to keep anything extra in ctx unless needed in backward
        ctx.world_size = dist.get_world_size()
        ctx.rank = dist.get_rank()

        # Create tensor to store sum across ranks
        summed_output = torch.zeros_like(local_output)
        dist.all_reduce(local_output, op=dist.ReduceOp.SUM, async_op=False)
        summed_output = local_output  # after all_reduce, it contains the sum

        return summed_output

    @staticmethod
    def backward(ctx, grad_output):
        # Gradient is already summed; distribute gradient as-is
        # Each rank receives full gradient
        return grad_output



class DistributedOutputChannelSplitConv2d(nn.Module):
    def __init__(self, conv_layer: nn.Conv2d, world_size, rank, device, combine=True):
        super().__init__()
        # assert conv_layer.out_channels % world_size == 0, "Output channels must be divisible by world_size"

        self.device = device
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
            

            gathered_output = AllGatherOutputSplit.apply(self.output, gathered_outputs, self.split_channels, self.world_size, self.rank)
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

class DistributedInputChannelSplitConv2d(nn.Module):
    def __init__(self, conv_layer: nn.Conv2d, world_size, rank, device, combine=True):
        super().__init__()
        self.device = device
        self.world_size = world_size
        self.rank = rank
        self.combine = combine

        self.in_channels = conv_layer.in_channels
        self.out_channels = conv_layer.out_channels
        self.kernel_size = conv_layer.kernel_size
        self.stride = conv_layer.stride
        self.padding = conv_layer.padding

        # Compute how to split input channels
        self.rem = self.in_channels % world_size
        self.split_channels = [
            self.in_channels // world_size + (1 if i < self.rem else 0)
            for i in range(world_size)
        ]
        self.local_in_channels = self.split_channels[self.rank]

        # Local Conv2d layer
        self.conv = nn.Conv2d(self.local_in_channels, self.out_channels,
                              kernel_size=self.kernel_size,
                              stride=self.stride,
                              padding=self.padding).to(self.device)

        self.copy_weights_from(conv_layer)

    def forward(self, x):
        x = x.to(self.device)

        # Forward through local conv
        local_output = self.conv(x)
        # print("----local-output---")
        # print(local_output)

        if self.combine:
            # AllReduce across all ranks (sum outputs)
            return AllReduceInputSplit.apply(local_output)
        else:
            return local_output

    def copy_weights_from(self, original_layer: nn.Conv2d):
        """Copies the correct input channel split of the weights to each GPU"""
        with torch.no_grad():
            start_idx = sum(self.split_channels[:self.rank])
            end_idx = start_idx + self.local_in_channels

            self.conv.weight.copy_(original_layer.weight[:, start_idx:end_idx, :, :])
            self.conv.bias.copy_(original_layer.bias/self.world_size)

        self.conv.weight.requires_grad = True
        self.conv.bias.requires_grad = True


class DistributedOutputChannelSplitLinear(nn.Module):
    def __init__(self, linear_layer: nn.Linear, world_size, rank, device, combine=True):
        super().__init__()
        self.device = device
        self.world_size = world_size
        self.rank = rank
        self.combine = combine
        
        self.in_features = linear_layer.in_features
        self.out_features = linear_layer.out_features
        
        # Compute split sizes
        self.rem = self.out_features % world_size
        self.split_sizes = [self.out_features // world_size + (1 if i < self.rem else 0) 
                           for i in range(world_size)]
        self.local_out_features = self.split_sizes[self.rank]
        
        # Create local linear layer
        self.linear = nn.Linear(self.in_features, self.local_out_features).to(self.device)
        
        # Copy weights
        self.copy_weights_from(linear_layer)

    def forward(self, x):
        x = x.to(self.device)
        local_output = self.linear(x)
        
        if self.combine:
            gathered_outputs = []
            for i in range(self.world_size):
                recv_shape = list(local_output.shape)
                recv_shape[-1] = self.split_sizes[i]
                gathered_outputs.append(torch.zeros(recv_shape, device=self.device))
            
            gathered_output = AllGatherOutputSplit.apply(
                local_output, gathered_outputs, self.split_sizes, self.world_size, self.rank
            )
            return gathered_output
        
        return local_output

    def copy_weights_from(self, original_layer: nn.Linear):
        with torch.no_grad():
            start_idx = sum(self.split_sizes[:self.rank])
            end_idx = start_idx + self.local_out_features
            
            self.linear.weight.copy_(original_layer.weight[start_idx:end_idx])
            self.linear.bias.copy_(original_layer.bias[start_idx:end_idx])
            
        self.linear.weight.requires_grad = True
        self.linear.bias.requires_grad = True

class DistributedInputChannelSplitLinear(nn.Module):
    def __init__(self, linear_layer: nn.Linear, world_size, rank, device, combine=True):
        super().__init__()
        self.device = device
        self.world_size = world_size
        self.rank = rank
        self.combine = combine
        
        self.in_features = linear_layer.in_features
        self.out_features = linear_layer.out_features
        
        # Compute split sizes
        self.rem = self.in_features % world_size
        self.split_sizes = [self.in_features // world_size + (1 if i < self.rem else 0) 
                           for i in range(world_size)]
        self.local_in_features = self.split_sizes[self.rank]
        
        # Create local linear layer
        self.linear = nn.Linear(self.local_in_features, self.out_features).to(self.device)
        
        # Copy weights
        self.copy_weights_from(linear_layer)

    def forward(self, x):
        x = x.to(self.device)
        
        # Split input along feature dimension
        split_x = torch.split(x, self.split_sizes, dim=1)
        x_local = split_x[self.rank]
        
        local_output = self.linear(x_local)
        # print("-----------local-----------------")
        # print(local_output)
        
        if self.combine:
            return AllReduceInputSplit.apply(local_output)
        return local_output

    def copy_weights_from(self, original_layer: nn.Linear):
        with torch.no_grad():
            start_idx = sum(self.split_sizes[:self.rank])
            end_idx = start_idx + self.local_in_features
            
            self.linear.weight.copy_(original_layer.weight[:, start_idx:end_idx])
            self.linear.bias.copy_(original_layer.bias / self.world_size)
            
        self.linear.weight.requires_grad = True
        self.linear.bias.requires_grad = True

class DistributedParallelActivations(nn.Module):
    def __init__(self, activation=nn.ReLU(), world_size=None, rank=None, device=None, combine=False):
        super().__init__()
        self.activation = activation
        self.combine = combine
        self.world_size = world_size
        self.rank = rank
        self.device = device


    def forward(self, x):
        
        
        # For distributed case, we expect x to be local tensor
        local_activated = self.activation(x)
        if self.combine:
            # AllReduce across all ranks (sum outputs)
            gathered_outputs= [torch.zeros(local_activated.shape, device=self.device)for _ in range(self.world_size)] 
            dist.all_gather(gathered_outputs, local_activated)
            return torch.cat(gathered_outputs, dim=1)
            
        else:
            return local_activated
        
        


