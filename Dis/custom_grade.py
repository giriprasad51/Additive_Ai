import torch
import torch.distributed as dist
from torch.autograd import Function
import torch.nn.functional as F



# class Customgrade_OutputChannelSplitConv2d(Function):
#     @staticmethod
#     def forward(ctx, output,gathered_outputs, input, weight, bias, stride, padding, split_channels, world_size, rank, device):
#         ctx.save_for_backward(input, weight, bias)
#         ctx.stride = stride
#         ctx.padding = padding
#         ctx.split_channels = split_channels
#         ctx.world_size = world_size
#         ctx.rank = rank
#         ctx.device = device
        
        
#         dist.all_gather_object(gathered_outputs, output)
#         gathered_outputs = [t.to(device) for t in gathered_outputs]
        
#         # Concatenate along channel dimension
#         full_output = torch.cat(gathered_outputs, dim=1)
#         return full_output

#     @staticmethod
#     def backward(ctx, grad_output):
#         input, weight, bias = ctx.saved_tensors
#         split_channels = ctx.split_channels
#         world_size = ctx.world_size
#         rank = ctx.rank
        
#         # Split the gradient according to original channel division
#         grad_outputs = torch.split(grad_output, split_channels, dim=1)
#         local_grad_output = grad_outputs[rank].contiguous()
        
#         # Compute gradients
#         grad_input = grad_weight = grad_bias = None
        
#         if ctx.needs_input_grad[0]:
#             # Fix: Use proper input padding and output_padding for conv_transpose2d
#             grad_input = F.conv_transpose2d(
#                 local_grad_output, 
#                 weight, 
#                 None,
#                 stride=ctx.stride,
#                 padding=ctx.padding,
#                 # Important: output_padding should be 0 in this case
#                 output_padding=0
#             )
#             # The all_reduce should be done only if world_size > 1
#             if world_size > 1:
#                 dist.all_reduce(grad_input, op=dist.ReduceOp.SUM)
        
#         if ctx.needs_input_grad[1]:
#             grad_weight = F.conv2d(
#                 input, 
#                 local_grad_output, 
#                 None, 
#                 stride=ctx.stride, 
#                 padding=ctx.padding
#             )
        
#         if bias is not None and ctx.needs_input_grad[2]:
#             grad_bias = local_grad_output.sum((0, 2, 3))

#         print(local_grad_output.shape)
        
#         grad_bias = torch.zeros(local_grad_output.shape,  device=ctx.device)
#         return (
#             local_grad_output,
#             None,  # For gathered_outputs
#             None,  # For input
#             grad_weight if ctx.needs_input_grad[3] else None,
#             grad_bias if (bias is not None and ctx.needs_input_grad[4]) else None,
#             None, None, None, None, None, None, None
#         )

class Customgrade_OutputChannelSplitConv2d(Function):
    @staticmethod
    def forward(ctx, local_output, gathered_outputs, input, weight, bias, 
               stride, padding, split_channels, world_size, rank, device):
        # Perform the gather operation
        # This would typically use all_gather or similar collective operation
        # For simplicity, we'll show a placeholder implementation
        
        ctx.save_for_backward(input, weight, bias, local_output)
        ctx.stride = stride
        ctx.padding = padding
        ctx.split_channels = split_channels
        ctx.world_size = world_size
        ctx.rank = rank
        ctx.device = device
        
        # In a real implementation, you would:
        # 1. All-gather the local outputs from all ranks
        # 2. Concatenate them along the channel dimension
        # Here we just return the local output for demonstration
        return torch.cat(gathered_outputs, dim=1)

    @staticmethod
    def backward(ctx, grad_output):
        input, weight, bias, local_output = ctx.saved_tensors
        
        # Split the gradient according to the original split
        grad_outputs = torch.split(grad_output, ctx.split_channels, dim=1)
        grad_local = grad_outputs[ctx.rank]
        
        # Compute gradients for input
        grad_input = None
        if ctx.needs_input_grad[0]:
            grad_input = F.grad.conv2d_input(
                input.shape, weight, grad_local,
                stride=ctx.stride, padding=ctx.padding
            )
        
        # The weight and bias gradients are handled by the local convolution's backward
        return grad_local, None, grad_input, None, None, None, None, None, None, None, None
    
# class OutputChannelGather(Function):
#     @staticmethod
#     def forward(ctx, local_output, split_channels, rank):
#         ctx.split_channels = split_channels
#         ctx.rank = rank
        
#         # Prepare for all_gather
#         output_size = list(local_output.size())
#         gathered_outputs = [torch.empty(output_size, device=local_output.device) 
#                           for _ in range(len(split_channels))]
        
#         # Perform all_gather
#         dist.all_gather(gathered_outputs, local_output)
        
#         # Concatenate along channel dimension
#         full_output = torch.cat(gathered_outputs, dim=1)
        
#         # Save for backward
#         ctx.save_for_backward(local_output)
#         return full_output

#     @staticmethod
#     def backward(ctx, grad_output):
#         local_output, = ctx.saved_tensors
        
#         # Split the gradient according to original split
#         grad_outputs = torch.split(grad_output, ctx.split_channels, dim=1)
#         grad_local = grad_outputs[ctx.rank]
        
#         return grad_local, None, None
    