import torch
import torch.nn as nn
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
import os, datetime


def cleanup():
    dist.destroy_process_group()


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
        print(self.conv.weight.data)
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


def test_DistributedOutputChannelSplitConv2d(world_size, rank):
    ddp_setup(rank, world_size)
    original_conv = nn.Conv2d(in_channels=3, out_channels=4, kernel_size=3, stride=1, padding=1).to(f"cuda:{rank}")  
    x = torch.randn(1, 3, 3, 3).to(f"cuda:{rank}")   
    
    # Broadcast original model's weights and input to all ranks
    for param in original_conv.parameters():
        dist.broadcast(param.data, src=0)
    dist.broadcast(x, src=0)
    # Initialize parallel convolution
    parallel_conv = DistributedOutputChannelSplitConv2d(original_conv, world_size, rank, combine=True)
    dist.barrier()
    if rank ==0:
        print(parallel_conv.rank)
    else:
        print(parallel_conv.rank)
        
    parallel_output = parallel_conv(x)
    original_output = original_conv(x)
    # print(f"rank {rank}","parallel_output",parallel_output)
    # print(f"rank {rank}","original_output",original_output)


    # Ensure synchronization before validation
    dist.barrier()

    if rank == 0:
        print(f"Original Output Mean: {original_output.mean()}, Std: {original_output.std()}")
        print(f"Parallel Output Mean: {parallel_output.mean()}, Std: {parallel_output.std()}")

        # Ensure shape matches
        assert original_output.shape == parallel_output.shape, f"Shape mismatch: {original_output.shape} vs {parallel_output.shape}"

        # Ensure values match closely
        num_mismatched = torch.sum(~torch.isclose(original_output, parallel_output, atol=1e-5)).item()
        assert num_mismatched == 0, f"Number of mismatched elements: {num_mismatched}"
        assert torch.allclose(original_output, parallel_output, atol=1e-5), "Outputs do not match!"

        print("Test passed: DistributedOutputChannelSplitConv2d matches standard Conv2d!")

    cleanup()


def ddp_setup(rank: int, world_size: int):
    dist.init_process_group("nccl", timeout=datetime.timedelta(seconds=7200), rank=rank, world_size=world_size)


def main():
    rank = int(os.environ['RANK'])
    world_size = int(os.environ['WORLD_SIZE'])

    # Run the test
    test_DistributedOutputChannelSplitConv2d(world_size, rank)


if __name__ == '__main__':
    total_epochs = 200
    FULL_START = datetime.datetime.now()
    main()
    print("Total time taken: ", datetime.datetime.now() - FULL_START)
    exit(0)
