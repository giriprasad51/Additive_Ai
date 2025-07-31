import torch
import torch.nn as nn
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.autograd import Function
from Dis.custom_grade import Customgrade_OutputChannelSplitConv2d
import torch.nn.functional as F

# class DistributedOutputChannelSplitConv2d(nn.Module):
#     def __init__(self, conv_layer: nn.Conv2d, world_size, rank, device, combine=True):
#         super().__init__()
#         self.device = device
#         self.world_size = world_size
#         self.rank = rank
#         self.combine = combine
        
#         # Original layer parameters
#         self.out_channels = conv_layer.out_channels
#         self.in_channels = conv_layer.in_channels
#         self.kernel_size = conv_layer.kernel_size
#         self.stride = conv_layer.stride
#         self.padding = conv_layer.padding
        
#         # Compute split sizes
#         self.rem = self.out_channels % world_size
#         self.split_channels = [self.out_channels // world_size + (1 if i < self.rem else 0) 
#                              for i in range(world_size)]
#         self.local_out_channels = self.split_channels[self.rank]
        
#         # Local convolution
#         self.conv = nn.Conv2d(self.in_channels, self.local_out_channels,
#                              kernel_size=self.kernel_size, stride=self.stride,
#                              padding=self.padding).to(self.device)
        
#         # Copy weights
#         self.copy_weights_from(conv_layer)
        
#         # Register hooks
#         self.register_forward_hook(self._forward_hook)
#         self.register_backward_hook(self._backward_hook)
        
#         # For storing tensors needed in backward pass
#         self.saved_tensors = {}

#     def forward(self, x):
#         x = x.to(self.device)
#         local_output = self.conv(x)
        
#         if self.combine:
#             # Prepare tensors for gathering
#             gathered_outputs = []
#             for i in range(self.world_size):
#                 recv_shape = list(local_output.shape)
#                 recv_shape[1] = self.split_channels[i]
#                 gathered_outputs.append(torch.zeros(recv_shape, device=self.device))
            
#             dist.all_gather(gathered_outputs, local_output)
#             # Perform the gather operation
#             output = Customgrade_OutputChannelSplitConv2d.apply(
#                 local_output, gathered_outputs, x,
#                 self.conv.weight, self.conv.bias,
#                 self.stride, self.padding,
#                 self.split_channels, self.world_size,
#                 self.rank, self.device
#             )
#         else:
#             output = local_output
            
#         return output

#     def copy_weights_from(self, original_layer):
#         """Distributes weights and biases from the original Conv2d layer"""
#         with torch.no_grad():
#             start_idx = sum(self.split_channels[:self.rank])
#             end_idx = start_idx + self.local_out_channels

#             self.conv.weight.copy_(original_layer.weight[start_idx:end_idx])
#             if original_layer.bias is not None:
#                 self.conv.bias.copy_(original_layer.bias[start_idx:end_idx])

#         self.conv.weight.requires_grad = True
#         if self.conv.bias is not None:
#             self.conv.bias.requires_grad = True

#     def _forward_hook(self, module, input, output):
#         """Store necessary tensors for backward pass"""
#         self.saved_tensors['input'] = input[0].detach()
#         if self.combine:
#             self.saved_tensors['local_output'] = output.detach()
#         return output

#     def _backward_hook(self, module, grad_input, grad_output):
#         """Handle gradient computation for distributed case"""
#         if not self.combine:
#             return grad_output
            
#         # For combined case, we need to handle the gradient splitting
#         input_tensor = self.saved_tensors['input']
#         local_output = self.saved_tensors['local_output']
        
#         # Split the incoming gradient according to our rank's portion
#         start_idx = sum(self.split_channels[:self.rank])
#         end_idx = start_idx + self.local_out_channels
#         grad_output_local = grad_output[0][:, start_idx:end_idx, :, :]
        
#         # Recompute the local operation with gradient tracking
#         with torch.enable_grad():
#             recomputed = F.conv2d(
#                 input_tensor, 
#                 self.conv.weight, 
#                 self.conv.bias,
#                 stride=self.stride, 
#                 padding=self.padding
#             )
            
#             # Backpropagate through the local operation
#             torch.autograd.backward(
#                 recomputed,
#                 grad_tensors=grad_output_local,
#                 retain_graph=True
#             )
        
#         # The gradients are now properly accumulated in conv.weight.grad and conv.bias.grad
#         # Input gradient is handled by the OutputChannelGather backward
#         return (None,)
    
class DistributedOutputChannelSplitConv2d(nn.Module):
    def __init__(self, conv_layer: nn.Conv2d, world_size, rank, device, combine=True):
        super().__init__()
        self.device = device
        self.world_size = world_size
        self.rank = rank
        self.combine = combine
        
        # Original layer parameters
        self.out_channels = conv_layer.out_channels
        self.in_channels = conv_layer.in_channels
        self.kernel_size = conv_layer.kernel_size
        self.stride = conv_layer.stride
        self.padding = conv_layer.padding
        self.dilation = conv_layer.dilation
        self.groups = conv_layer.groups
        
        # Compute split sizes
        self.rem = self.out_channels % world_size
        self.split_channels = [self.out_channels // world_size + (1 if i < self.rem else 0) 
                             for i in range(world_size)]
        self.local_out_channels = self.split_channels[self.rank]
        
        # Local convolution
        self.conv = nn.Conv2d(
            self.in_channels, self.local_out_channels,
            kernel_size=self.kernel_size, stride=self.stride,
            padding=self.padding, dilation=self.dilation,
            groups=self.groups, bias=conv_layer.bias is not None
        ).to(self.device)
        
        # Copy weights
        self.copy_weights_from(conv_layer)
        
        # For storing tensors needed in backward pass
        self.saved_for_backward = {}

    def forward(self, x):
        # Ensure input is on correct device
        x = x.to(self.device)
        
        # Compute local convolution
        local_output = self.conv(x)
        
        if not self.combine:
            return local_output
            
        # Prepare for all_gather - each rank needs same-sized tensors
        output_size = list(local_output.size())
        output_size[1] = self.split_channels[self.rank]  # Channel dimension
        
        # Create list to store gathered outputs
        gathered_outputs = [torch.empty(output_size, device=self.device) 
                          for _ in range(self.world_size)]
        
        # Perform all_gather operation
        dist.all_gather(gathered_outputs, local_output)
        
        # Concatenate along channel dimension
        full_output = torch.cat(gathered_outputs, dim=1)
        
        # Save for backward if needed
        if self.training:
            self.saved_for_backward['x'] = x.detach()
            self.saved_for_backward['local_output'] = local_output.detach()
        
        return full_output

    def copy_weights_from(self, original_layer):
        """Distributes weights and biases from the original Conv2d layer"""
        with torch.no_grad():
            start_idx = sum(self.split_channels[:self.rank])
            end_idx = start_idx + self.local_out_channels

            self.conv.weight.copy_(original_layer.weight[start_idx:end_idx])
            if original_layer.bias is not None:
                self.conv.bias.copy_(original_layer.bias[start_idx:end_idx])

        self.conv.weight.requires_grad = True
        if self.conv.bias is not None:
            self.conv.bias.requires_grad = True

    def backward(self, grad_output):
        """Custom backward pass to handle distributed gradients"""
        if not self.combine:
            return super().backward(grad_output)
            
        # Get saved tensors
        x = self.saved_for_backward['x']
        local_output = self.saved_for_backward['local_output']
        
        # Split the gradient according to original split
        grad_outputs = torch.split(grad_output, self.split_channels, dim=1)
        grad_local = grad_outputs[self.rank]
        
        # Compute gradients for local convolution
        x.requires_grad_(True)
        local_output_recompute = self.conv(x)
        
        # Manually compute gradients
        torch.autograd.backward(
            local_output_recompute,
            grad_tensors=grad_local,
            retain_graph=True
        )
        
        # Get input gradient
        grad_input = x.grad
        
        # Clean up
        x.requires_grad_(False)
        
        return grad_input