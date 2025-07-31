import torch
import torch.nn as nn
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.autograd import Function
from Dis.custom_grade import Customgrade_OutputChannelSplitConv2d

class AllGatherOutputSplit(Function):
    @staticmethod
    def forward(ctx, tensor, gathered_tensors, split_sizes, world_size, rank, device):
        ctx.world_size = world_size
        ctx.rank = rank
        ctx.split_sizes = split_sizes
        # print(rank,  [t.shape for t in gathered_tensors], tensor.shape)
        # dist.all_gather_object(gathered_tensors, tensor)
        print()
        dist.all_gather(gathered_tensors, tensor)
        gathered_tensors = [t.to(device) for t in gathered_tensors]
        # print(rank)
        
        return torch.cat(gathered_tensors, dim=1)

    @staticmethod
    def backward(ctx, grad_output):
        world_size = ctx.world_size
        rank = ctx.rank
        
        # print(ctx.split_sizes)
        all_grad = torch.split(grad_output, ctx.split_sizes, dim=1)
        local_grad = all_grad[rank].contiguous()
        # print(local_grad.shape)
    
        return local_grad, None, None, None, None, None # `None` for `world_size` argument

class AllReduceInputSplit(torch.autograd.Function):
    @staticmethod
    def forward(ctx, local_output):
        # No need to keep anything extra in ctx unless needed in backward
        ctx.world_size = dist.get_world_size()
        ctx.rank = dist.get_rank()

        device = local_output.device
        # local_dtype = local_output.dtype
        # dtype = torch.float64
        # print(device,dtype)

        # Create tensor to store sum across ranks
        # summed_output = torch.zeros_like(local_output, device=device)
        # print(summed_output.dtype)

        # local_output = local_output#.to(device=device, dtype=dtype)

        # Only the rank 0 process will receive gathered tensors
        # gather_list = [torch.zeros_like(local_output, device=device) for _ in range(ctx.world_size)]

        # dist.all_gather(gather_list, local_output)


        dist.all_reduce(local_output, op=dist.ReduceOp.SUM, async_op=False)
        # for output in gather_list:
        #     summed_output += output   # after all_reduce, it contains the 
        # summed_output = local_output

        return local_output#.to(device=device, dtype=local_dtype)

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
        # print(self.device)
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
        # self.output = nn.BatchNorm2d(self.local_out_channels)(self.output)

        torch.cuda.synchronize(self.device)
        if self.combine:
            gathered_outputs = []
            for i in range(self.world_size):
                recv_shape = list(self.output.shape)
                recv_shape[1] = self.split_channels[i]
                gathered_outputs.append(torch.zeros(recv_shape,  device=self.device))
            

            # gathered_output = Customgrade_OutputChannelSplitConv2d.apply(self.output, gathered_outputs, self.split_channels, self.world_size, self.rank, self.device)
            gathered_output = Customgrade_OutputChannelSplitConv2d.apply(self.output, gathered_outputs, x,  self.conv.weight, self.conv.bias, self.stride, self.padding, self.split_channels, self.world_size, self.rank, self.device )
            if torch.isnan(gathered_output).any():
                print("Any NaNs in gathered_output?", torch.isnan(gathered_output).sum().item(), (~torch.isnan(gathered_output)).sum().item(), gathered_output.shape)
            
            return gathered_output
        
        if torch.isnan(self.output).any():
            print("Any NaNs in gathered_output?", torch.isnan(self.output).sum().item(), (~torch.isnan(self.output)).sum().item(), self.output.shape)
               
            
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
        # self.conv.to(self.device)

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
        x = x.to(self.device)#, dtype = torch.float64)

        # self.conv.weight = self.conv.weight.to(dtype=torch.float64)
        # self.conv.bias = self.conv.weight.to(dtype=torch.float64)

        # Forward through local conv
        local_output = self.conv(x)
        # print("----local-output---")
        # print(id(local_output))

        if self.combine:
            # AllReduce across all ranks (sum outputs)
            gathered_output = AllReduceInputSplit.apply(local_output)

        local_output = local_output + gathered_output - local_output
        # print(id(local_output))
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
        # self.conv.to(self.device)
        


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
        self.local_output = self.linear(x)
        # self.local_output = nn.BatchNorm2d(self.local_out_channels)(self.local_output)
        
        if self.combine:
            gathered_outputs = []
            for i in range(self.world_size):
                recv_shape = list(self.local_output.shape)
                recv_shape[-1] = self.split_sizes[i]
                gathered_outputs.append(torch.zeros(recv_shape, device=self.device))
            
            gathered_output = AllGatherOutputSplit.apply(
                self.local_output, gathered_outputs, self.split_sizes, self.world_size, self.rank, self.device
            )

            if torch.isnan(gathered_output).any():
                print("Any NaNs in gathered_output?", torch.isnan(gathered_output).sum().item(), (~torch.isnan(gathered_output)).sum().item(), gathered_output.shape)
            

            return gathered_output
        
        if torch.isnan(self.local_output).any():
                print("Any NaNs in gathered_output?", torch.isnan(self.local_output).sum().item(), (~torch.isnan(self.local_output)).sum().item(), self.local_output.shape)
        
        return self.local_output

    def copy_weights_from(self, original_layer: nn.Linear):
        with torch.no_grad():
            start_idx = sum(self.split_sizes[:self.rank])
            end_idx = start_idx + self.local_out_features
            
            self.linear.weight.copy_(original_layer.weight[start_idx:end_idx])
            self.linear.bias.copy_(original_layer.bias[start_idx:end_idx])
            
        self.linear.weight.requires_grad = True
        self.linear.bias.requires_grad = True
        # self.linear.to(self.device)

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
        # split_x = torch.split(x, self.split_sizes, dim=1)
        # x_local = split_x[self.rank]
        
        local_output = self.linear(x)
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
        # self.linear.to(self.device)

class DistributedParallelActivations(nn.Module):
    def __init__(self, activation=nn.ReLU(), world_size=None, rank=None, device=None, combine=False, previous= "OS"):
        super().__init__()
        if isinstance(activation, nn.ReLU):
            activation = nn.ReLU(inplace=False)
        self.activation = activation
        self.combine = combine
        self.world_size = world_size
        self.rank = rank
        self.device = device
        self.split_sizes = [None for i in range(world_size)]
        self.previous = previous

    def forward(self, x):
        
        
        # For distributed case, we expect x to be local tensor
        local_activated = self.activation(x)
        # if self.previous == "IS":
        #     dist.broadcast(local_activated, src=0)

        local_size = torch.tensor([local_activated.shape[1]], device=self.device)
        gathered_sizes = [torch.zeros_like(local_size) for _ in range(self.world_size)]

        dist.all_gather(gathered_sizes, local_size)
        self.split_sizes = [int(t.item()) for t in gathered_sizes]
        dist.barrier()
        # print(self.split_sizes)
        if self.combine:
            # AllReduce across all ranks (sum outputs)
            gathered_outputs= [torch.zeros(local_activated.shape, device=self.device)for _ in range(self.world_size)] 
            # dist.all_gather(gathered_outputs, local_activated)
            gathered_output = AllGatherOutputSplit.apply(
                local_activated, gathered_outputs, self.split_sizes, self.world_size, self.rank, self.device
            )
            return gathered_output
            
        else:
            return local_activated
        
class DistributedParallelDropout(nn.Module):
    def __init__(self, dropout=nn.Dropout(p=0.5, inplace=False), world_size=None, rank=None, device=None, combine=False):
        super().__init__()
        if isinstance(dropout, nn.Dropout):
            dropout = nn.Dropout(p=dropout.p, inplace=False)  # Make sure inplace=False for safety in distributed
        self.dropout = dropout
        self.combine = combine
        self.world_size = world_size
        self.rank = rank
        self.device = device
        self.split_sizes = [None for _ in range(world_size)]

    def forward(self, x):
        # Apply local dropout
        local_output = self.dropout(x)
        # mask = torch.ones_like(x)
        # if self.rank == 0:
        #     mask = (local_output == 0).float()
        #     # print(mask)
        #     dist.broadcast(mask, src=0)

        # else:
        #     dist.broadcast(mask, src=0)
        #     # print(mask)
        #     p = self.dropout.p
        #     local_output = x*(1-mask)/(1-p)
        # print(local_output)
        return local_output

        
        


