import torch
import torch.nn as nn
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
import os, datetime
from Dis.layers import DistributedOutputChannelSplitConv2d, DistributedInputChannelSplitConv2d, DistributedOutputChannelSplitLinear,DistributedInputChannelSplitLinear,DistributedParallelActivations





def test_DistributedOutputChannelSplitConv2d(world_size, rank, device):
    
    original_conv = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, stride=1, padding=1).to(device)  
       
    # Broadcast original model's weights and input to all ranks
    for param in original_conv.parameters():
        dist.broadcast(param.data, src=0)
    
    # Initialize parallel convolution
    parallel_conv = DistributedOutputChannelSplitConv2d(original_conv, world_size, rank, device, combine=True)
    dist.barrier()
    
    for _ in range(2):

        x = torch.randn(1, 3, 32, 32, requires_grad=True).to(device)
        dist.broadcast(x, src=0)

        optimizer1 = torch.optim.SGD(original_conv.parameters(), lr=0.1)
        optimizer2 = torch.optim.SGD(parallel_conv.conv.parameters(), lr=0.1)
        
        parallel_output = parallel_conv(x)
        original_output = original_conv(x)

        target = torch.randn_like(parallel_output, requires_grad=False).to(device)
        dist.broadcast(target, src=0)
        
        # Ensure synchronization before validation
        dist.barrier()

        # Create a dummy loss
        loss_fn = nn.MSELoss()

        if rank == 0:
            
            # Ensure shape matches
            assert original_output.shape == parallel_output.shape, f"Shape mismatch: {original_output.shape} vs {parallel_output.shape}"

            # Ensure values match closely
            num_mismatched = torch.sum(~torch.isclose(original_output, parallel_output, atol=1e-5)).item()
            assert num_mismatched == 0, f"Number of mismatched elements: {num_mismatched}/ {original_output.numel()}"
            assert torch.allclose(original_output, parallel_output, atol=1e-5), "Outputs do not match!"
            # print("original", rank, original_conv.weight.shape, original_conv.weight[22:,:,:,:])
            
            original_loss = loss_fn(original_output, target)
            # Backward pass
            original_loss.backward()
            optimizer1.step()

            parallel_loss = loss_fn(parallel_output, target)
            parallel_loss.backward()
            optimizer2.step()
        
        else:
            parallel_loss = loss_fn(parallel_output, target)
            parallel_loss.backward()
            optimizer2.step()
            

        # Ensure synchronization before checking gradients
        dist.barrier()

    if rank == 0:
        print("Test passed: Forward and Backward outputs match standard Conv2d!")            
    

def test_DistributedInputChannelSplitConv2d(world_size, rank, device):
    
    original_conv = nn.Conv2d(in_channels=64, out_channels=32, kernel_size=3, stride=1, padding=1).to(device)

    # Broadcast weights
    for param in original_conv.parameters():
        dist.broadcast(param.data, src=0)

    # Input channel split version
    parallel_conv = DistributedInputChannelSplitConv2d(original_conv, world_size, rank, device, combine=True)
    dist.barrier()

    for _ in range(1):
        # Input with more channels (split across ranks)
        x = torch.randn(1, 64, 32, 32, requires_grad=True).to(device)
        dist.broadcast(x, src=0)

        optimizer1 = torch.optim.SGD(original_conv.parameters(), lr=0.1)
        optimizer2 = torch.optim.SGD(parallel_conv.conv.parameters(), lr=0.1)
        # Split the input tensor along channel dim
        split_x = torch.split(x, [64 // world_size + (1 if i < 64%world_size else 0) for i in range(world_size) ], dim=1)
        x_local = split_x[rank]
        parallel_output = parallel_conv(x_local)
        original_output = original_conv(x)
        # print("---------original-output------------")
        # print(original_output)

        target = torch.randn_like(parallel_output).to(device)
        dist.broadcast(target, src=0)
        dist.barrier()

        loss_fn = nn.MSELoss()

        if rank == 0:
            assert original_output.shape == parallel_output.shape, f"Shape mismatch: {original_output.shape} vs {parallel_output.shape}"
            num_mismatched = torch.sum(~torch.isclose(original_output, parallel_output, atol=1e-5)).item()
            assert num_mismatched == 0, f"Mismatch: {num_mismatched}/{original_output.numel()}"
            assert torch.allclose(original_output, parallel_output, atol=1e-5), "Outputs do not match!"

            original_loss = loss_fn(original_output, target)
            original_loss.backward()
            optimizer1.step()

            parallel_loss = loss_fn(parallel_output, target)
            parallel_loss.backward()
            optimizer2.step()

        else:
            parallel_loss = loss_fn(parallel_output, target)
            parallel_loss.backward()
            optimizer2.step()

        dist.barrier()

    if rank == 0:
        print("Test passed: DistributedInputChannelSplitConv2d matches standard Conv2d!")

    
def test_DistributedOutputChannelSplitLinear(world_size, rank, device):
    original_linear = nn.Linear(128, 64).to(device)

    for param in original_linear.parameters():
        dist.broadcast(param.data, src=0)

    parallel_linear = DistributedOutputChannelSplitLinear(original_linear, world_size, rank, device, combine=True)
    dist.barrier()

    for _ in range(2):
        x = torch.randn(4, 128, requires_grad=True).to(device)
        dist.broadcast(x, src=0)

        optimizer1 = torch.optim.SGD(original_linear.parameters(), lr=0.1)
        optimizer2 = torch.optim.SGD(parallel_linear.linear.parameters(), lr=0.1)

        original_output = original_linear(x)
        parallel_output = parallel_linear(x)

        target = torch.randn_like(parallel_output).to(device)
        dist.broadcast(target, src=0)
        dist.barrier()

        loss_fn = nn.MSELoss()

        if rank == 0:
            assert original_output.shape == parallel_output.shape
            assert torch.allclose(original_output, parallel_output, atol=1e-5)

            original_loss = loss_fn(original_output, target)
            original_loss.backward()
            optimizer1.step()

            parallel_loss = loss_fn(parallel_output, target)
            parallel_loss.backward()
            optimizer2.step()
        else:
            parallel_loss = loss_fn(parallel_output, target)
            parallel_loss.backward()
            optimizer2.step()

        dist.barrier()

    if rank == 0:
        print("Test passed: DistributedOutputChannelSplitLinear matches standard Linear!")


def test_DistributedInputChannelSplitLinear(world_size, rank, device):
    original_linear = nn.Linear(128, 64).to(device)

    for param in original_linear.parameters():
        dist.broadcast(param.data, src=0)

    parallel_linear = DistributedInputChannelSplitLinear(original_linear, world_size, rank, device, combine=True)
    dist.barrier()

    for _ in range(1):
        x = torch.randn(4, 128, requires_grad=True).to(device)
        dist.broadcast(x, src=0)

        optimizer1 = torch.optim.SGD(original_linear.parameters(), lr=0.1)
        optimizer2 = torch.optim.SGD(parallel_linear.linear.parameters(), lr=0.1)

        original_output = original_linear(x)
        # print("-------original-out--------------")
        # print(original_output)
        split_x = torch.split(x, [128 // world_size + (1 if i < 64%world_size else 0) for i in range(world_size) ], dim=1)
        x_local = split_x[rank]
        parallel_output = parallel_linear(x)

        target = torch.randn_like(parallel_output).to(device)
        dist.broadcast(target, src=0)
        dist.barrier()

        loss_fn = nn.MSELoss()

        if rank == 0:
            assert original_output.shape == parallel_output.shape
            assert torch.allclose(original_output, parallel_output, atol=1e-5)

            original_loss = loss_fn(original_output, target)
            original_loss.backward()
            optimizer1.step()

            parallel_loss = loss_fn(parallel_output, target)
            parallel_loss.backward()
            optimizer2.step()
        else:
            parallel_loss = loss_fn(parallel_output, target)
            parallel_loss.backward()
            optimizer2.step()

        dist.barrier()

    if rank == 0:
        print("Test passed: DistributedInputChannelSplitLinear matches standard Linear!")

def test_DistributedParallelActivations(world_size, rank, device):
    relu = nn.ReLU()
    parallel_activation = DistributedParallelActivations(activation=relu, world_size=world_size, rank=rank, device=device, combine=True)
    dist.barrier()

    x = torch.randn(6, 8, requires_grad=True).to(device)
    dist.broadcast(x, src=0)
    split_x = torch.split(x, [8 // world_size + (1 if i < 64%world_size else 0) for i in range(world_size) ], dim=1)
    x_local = split_x[rank]
    parallel_output = parallel_activation(x_local)

    if rank == 0:
        original_output = relu(x)
        assert parallel_output.shape == original_output.shape
        # print(original_output)
        assert torch.allclose(parallel_output, original_output, atol=1e-5)
        num_mismatched = torch.sum(~torch.isclose(original_output, parallel_output, atol=1e-5)).item()
        assert num_mismatched == 0, f"Mismatch: {num_mismatched}/{original_output.numel()}"
        print("Test passed: DistributedParallelActivations works as expected.")



def main():
    rank = int(os.environ['RANK'])
    world_size = int(os.environ['WORLD_SIZE'])
    gpu_id = int(os.environ['LOCAL_RANK'])
    device = 'cuda:'+str(gpu_id)
    

    print(torch.cuda.device_count())
    dist.init_process_group(backend='gloo', timeout=datetime.timedelta(seconds=7200))
    # Run the test
    test_DistributedOutputChannelSplitConv2d(world_size, rank, device)
    test_DistributedInputChannelSplitConv2d(world_size, rank, device)
    test_DistributedOutputChannelSplitLinear(world_size, rank, device)
    test_DistributedInputChannelSplitLinear(world_size, rank, device)
    test_DistributedParallelActivations(world_size, rank, device)
    print(f"Rank {device} Total time taken: ", datetime.datetime.now() - FULL_START)
    


if __name__ == '__main__':
    total_epochs = 200
    FULL_START = datetime.datetime.now()
    main()
    exit(0)
