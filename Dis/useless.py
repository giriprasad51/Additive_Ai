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

        print("Rank:",self.rank,"-----", self.output.shape)
        # Register backward hook
        # self.output.register_hook(self.backward_hook)
        
        torch.cuda.synchronize(self.device)
        # print(f"Rank {self.rank}: Local output shape {output.shape}")

        if self.combine:
            # print("--------------check-point-1------------------")
            # gathered_outputs = []
            # for i in range(self.world_size):
            #     recv_shape = list(self.output.shape)
            #     recv_shape[1] = self.split_channels[i]
            #     gathered_outputs.append(torch.zeros(recv_shape,  device=self.device))
            # print([t.shape for t in gathered_outputs])
            # # Each rank sends its output to all other ranks
            # for dst in range(self.world_size):
                
            #     if dst == self.rank:
            #         gathered_outputs[self.rank] = self.output  # Store own output directly
            #         print("dst",self.rank,"----->", dst,"replaced")
            #     else:
            #         dist.send(self.output, dst=dst)
            #         print("dst",self.rank,"----->", dst,"sended")
            # # Ensure synchronization before validation
            # dist.barrier()
            # print("--------------check-point-2------------------")
            # # Each rank receives outputs from all other ranks
            # for src in range(self.world_size):
            #     if src != self.rank:
            #         recv_tensor = torch.zeros_like(self.output, device=self.device)
            #         print("src",self.rank,"<-----",src,recv_tensor.shape)
            #         dist.recv(recv_tensor, src=src)
            #         gathered_outputs[src] = recv_tensor
            #         print("src",self.rank,"<------" ,src,"recived")

            # dist.barrier()
            # print("--------------check-point-3------------------")
            
            # if self.rank == 0:
            #     # Rank 0 gathers outputs from all other ranks
            #     gathered_outputs = [self.output]
            #     for src in range(1, self.world_size):
            #         recv_shape = list(self.output.shape)
            #         recv_shape[1] = self.split_channels[src]
            #         recv_tensor = torch.zeros(recv_shape, device=self.device)
            #         dist.recv(recv_tensor, src=src)
            #         gathered_outputs.append(recv_tensor)

            # gathered_output = torch.cat(gathered_outputs, dim=1)
            # return gathered_output

            # else:
            #     # Other ranks send their outputs to rank 0
            #     dist.send(self.output, dst=0)

            gathered_outputs = []
            for i in range(self.world_size):
                recv_shape = list(self.output.shape)
                recv_shape[1] = self.split_channels[i]
                gathered_outputs.append(torch.zeros(recv_shape,  device=self.device))
            dist.all_reduce(gathered_outputs, self.output)
            gathered_output = torch.cat(gathered_outputs, dim=1)
            # print(gathered_output)
            return gathered_output
            
        return self.output
    
    def backward_hook(self, grad_output):
        pass
        print(f"---------Rank-{self.rank}---{grad_output}--------------")
        """Handles gradient communication in the backward pass."""
        
        return torch.zeros(grad_output)

    def copy_weights_from(self, original_layer: nn.Conv2d):
        """Distributes weights and biases from the original Conv2d layer"""
        with torch.no_grad():
            start_idx = sum(self.split_channels[:self.rank])
            end_idx = start_idx + self.local_out_channels

            self.conv.weight.copy_(original_layer.weight[start_idx:end_idx])
            self.conv.bias.copy_(original_layer.bias[start_idx:end_idx])

            # print(f"Rank {self.rank} - Weights Copied: Mean {self.conv.weight.mean()}, Bias Mean {self.conv.bias.mean()}")

        self.conv.weight.requires_grad = True  # Ensure gradients flow
        self.conv.bias.requires_grad = True

