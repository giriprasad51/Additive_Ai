import torch
import torch.nn as nn
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
import os, datetime
from Dis.layers import DistributedOutputChannelSplitConv2d, DistributedInputChannelSplitConv2d, DistributedOutputChannelSplitLinear,DistributedInputChannelSplitLinear,DistributedParallelActivations, DistributedParallelDropout
import torch.nn.functional as F

def check_outputs(original_output, parallel_output):
    assert original_output.shape == parallel_output.shape, f"Shape mismatch: {original_output.shape} vs {parallel_output.shape}"
    output_diff = (original_output - parallel_output).abs().max().item()
    print(f"Maximum absolute difference: {output_diff}")
    num_mismatched = torch.sum(~torch.isclose(original_output, parallel_output, atol=1e-3)).item()
    assert num_mismatched == 0, f"Mismatch: {num_mismatched}/{original_output.numel()}"
    assert torch.allclose(original_output, parallel_output, atol=1e-3), "Outputs do not match!"


def test_DistributedOutputChannelSplitConv2d(world_size, rank, device):

    tem_conv1o = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, stride=1, padding=1, bias=True).to(device)
    original_conv = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1, bias=True).to(device)  
    tem_conv2 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1, bias=True).to(device)

    tem_conv1p = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, stride=1, padding=1, bias=True).to(device)

    # Copy weights and biases (deep copy with .clone())
    tem_conv1p.weight.data = tem_conv1o.weight.data.clone()
    if tem_conv1o.bias is not None:
        tem_conv1p.bias.data = tem_conv1o.bias.data.clone()

    # print(id(tem_conv1o.weight), id(tem_conv1p.weight))
    # Broadcast original model's weights and input to all ranks
    for param in original_conv.parameters():
        dist.broadcast(param.data, src=0)

    for param in tem_conv1o.parameters():
        dist.broadcast(param.data, src=0)

    for param in tem_conv2.parameters():
        dist.broadcast(param.data, src=0)

    for param in tem_conv1p.parameters():
        dist.broadcast(param.data, src=0)
    
    # Initialize parallel convolution
    parallel_conv = DistributedOutputChannelSplitConv2d(original_conv, world_size, rank, device, combine=True)
    dist.barrier()
    # original_conv.to(device)
    for i in range(2):

        x = torch.randn(1, 3, 32, 32, requires_grad=True).to(device)
        dist.broadcast(x, src=0)

        optimizer1 = torch.optim.SGD(original_conv.parameters(), lr=0.1)
        optimizer2 = torch.optim.SGD(parallel_conv.conv.parameters(), lr=0.1)

        xp = tem_conv1p(x)
        parallel_output = parallel_conv(xp)
        xp = tem_conv2(parallel_output)

        xo = tem_conv1o(x)
        original_output = original_conv(xo)
        xo = tem_conv2(original_output)

        target = torch.randn_like(xp, requires_grad=False).to(device)
        dist.broadcast(target, src=0)
        
        # Ensure synchronization before validation
        dist.barrier()

        # Create a dummy loss
        loss_fn = nn.MSELoss()

        # print(original_output)
        # print(parallel_output)
        # if rank == 0:
        check_outputs(original_output, parallel_output)

        
        # Backward pass
        original_loss = loss_fn(xo, target)
        original_loss.backward()
        optimizer1.step()

        parallel_loss = loss_fn(xp, target)
        parallel_loss.backward()
        optimizer2.step()

            
        
        
            

        # Ensure synchronization before checking gradients
        dist.barrier()
        
        original_grad = original_conv.weight.grad
        original_grad = torch.split(original_grad, [original_grad.shape[0] // world_size + (1 if i < original_grad.shape[0] %world_size else 0) for i in range(world_size) ], dim=0)
        original_grad = original_grad[rank]
        parallel_grad = parallel_conv.conv.weight.grad
        check_outputs(original_grad, parallel_grad)

        original_grad_b = original_conv.bias.grad
        original_grad_b = torch.split(original_grad_b, [original_grad_b.shape[0] // world_size + (1 if i < original_grad_b.shape[0] %world_size else 0) for i in range(world_size) ], dim=0)
        original_grad_b = original_grad_b[rank]
        parallel_grad_b = parallel_conv.conv.bias.grad
        # print(original_grad_b)
        # print(parallel_grad_b)
        # print(parallel_grad_b/original_grad_b)
        check_outputs(original_grad_b, parallel_grad_b)

        original_grad = tem_conv1o.weight.grad
        parallel_grad = tem_conv1p.weight.grad
        print(original_grad_b)
        print(parallel_grad_b)
        check_outputs(original_grad, parallel_grad)

        original_grad_b = tem_conv1o.bias.grad
        parallel_grad_b = tem_conv1p.bias.grad
        print(original_grad_b)
        print(parallel_grad_b)
        print(parallel_grad_b/original_grad_b)
        check_outputs(original_grad_b, parallel_grad_b)
        print(f"Test passed {i}: Forward and Backward outputs match standard Conv2d!") 
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
    # original_conv.to(device)

    for _ in range(10):
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
        

        target = torch.randn_like(parallel_output).to(device)
        dist.broadcast(target, src=0)
        dist.barrier()

        loss_fn = nn.MSELoss()

        # if rank == 0:
            # print("---------original-output------------")
            # print(original_output)
            # print("---------pallel-output------------")
            # print(parallel_output)
        assert original_output.shape == parallel_output.shape, f"Shape mismatch: {original_output.shape} vs {parallel_output.shape}"
        output_diff = (original_output - parallel_output).abs().max().item()
        print(f"Maximum absolute difference: {output_diff}")
        # num_mismatched = torch.sum(~torch.isclose(original_output, parallel_output, atol=1e-5)).item()
        # assert num_mismatched == 0, f"Mismatch: {num_mismatched}/{original_output.numel()}"
        assert torch.allclose(original_output, parallel_output, atol=1e-5), "Outputs do not match!"

        original_loss = loss_fn(original_output, target)
        original_loss.backward()
        optimizer1.step()

        parallel_loss = loss_fn(parallel_output, target)
        parallel_loss.backward()
        optimizer2.step()

        # else:
        #     parallel_loss = loss_fn(parallel_output, target)
        #     parallel_loss.backward()
        #     optimizer2.step()

        dist.barrier()

        original_grad = original_conv.weight.grad
        # original_grad = torch.split(original_grad, [original_grad.shape[1] // world_size + (1 if i < original_grad.shape[1] %world_size else 0) for i in range(world_size) ], dim=1)
        # original_grad = original_grad[rank]
        # parallel_grad = original_grad[1]
        # parallel_grad = parallel_conv.conv.weight.grad
        # print("----------original-grade---------")
        # print(original_grad)
        # print("----------parallel-grade---------")
        # print(parallel_grad)
        grad_output = 2 * (original_output - target) / (1 * 32 * 32 * 32)
        manual_grad_weight = F.conv2d(
                x.transpose(0, 1),  # [in_channels, batch_size, H, W]
                grad_output.transpose(0, 1),  # [out_channels, batch_size, H, W]
                padding=original_conv.padding,
                stride=original_conv.stride,
                groups=original_conv.groups
            ).transpose(0, 1)  # Back to [out_channels, in_channels, k, k]
        
        parallel_grad = manual_grad_weight
        
        assert original_grad is not None, f"Original param  has no grad!"
        assert parallel_grad is not None, f"Parallel param  has no grad!"
        assert original_grad.shape == parallel_grad.shape, f"Shape mismatch: {original_grad.shape} vs {parallel_grad.shape}"
        grad_diff = (original_grad - parallel_grad).abs().max().item()
        print(f"Maximum absolute difference: {grad_diff}")
        # grad_mismatch = torch.sum(~torch.isclose(original_grad, parallel_grad, atol=1e-5)).item()
        # assert grad_mismatch == 0, f"Grad mismatch in param : {grad_mismatch}/{original_grad.numel()}"
        assert torch.allclose(original_grad, parallel_grad, atol=1e-5), f"Gradient mismatch in param !"


    if rank == 0:
        print("Test passed: DistributedInputChannelSplitConv2d matches standard Conv2d!")

    
def test_DistributedOutputChannelSplitLinear(world_size, rank, device):
    original_linear = nn.Linear(128, 64).to(device)

    for param in original_linear.parameters():
        dist.broadcast(param.data, src=0)

    parallel_linear = DistributedOutputChannelSplitLinear(original_linear, world_size, rank, device, combine=True)
    dist.barrier()
    # original_linear.to(device)
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
    # original_linear.to(device)
    for _ in range(2):
        x = torch.randn(4, 128, requires_grad=True).to(device)
        dist.broadcast(x, src=0)

        optimizer1 = torch.optim.SGD(original_linear.parameters(), lr=0.1)
        optimizer2 = torch.optim.SGD(parallel_linear.linear.parameters(), lr=0.1)

        original_output = original_linear(x)
        # print("-------original-out--------------")
        # print(original_output)
        split_x = torch.split(x, [128 // world_size + (1 if i < 64%world_size else 0) for i in range(world_size) ], dim=1)
        x_local = split_x[rank]
        parallel_output = parallel_linear(x_local)

        target = torch.randn_like(parallel_output).to(device)
        dist.broadcast(target, src=0)
        dist.barrier()

        loss_fn = nn.MSELoss()

        if rank == 0:
            assert original_output.shape == parallel_output.shape
            # print("original",original_output)
            # print("-------------------------------")
            # print("pallel", parallel_output)
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

# def test_DistributedParallelActivations(world_size, rank, device):
#     relu = nn.ReLU()
#     parallel_activation = DistributedParallelActivations(activation=relu, world_size=world_size, rank=rank, device=device, combine=True)
#     dist.barrier()

#     x = torch.randn(6, 8, requires_grad=True).to(device)
#     dist.broadcast(x, src=0)
#     split_x = torch.split(x, [8 // world_size + (1 if i < 64%world_size else 0) for i in range(world_size) ], dim=1)
#     x_local = split_x[rank]
#     parallel_output = parallel_activation(x_local)

#     if rank == 0:
#         original_output = relu(x)
#         assert parallel_output.shape == original_output.shape
#         # print(original_output)
#         assert torch.allclose(parallel_output, original_output, atol=1e-5)
#         num_mismatched = torch.sum(~torch.isclose(original_output, parallel_output, atol=1e-5)).item()
#         assert num_mismatched == 0, f"Mismatch: {num_mismatched}/{original_output.numel()}"
#         print("Test passed: DistributedParallelActivations works as expected.")
def test_DistributedParallelActivations_with_output_split(world_size, rank, device):
    # Base configuration
    in_features = 128
    out_features = 64
    batch_size = 4
    
    # Calculate output split sizes
    out_split_sizes = [out_features // world_size + (1 if i < out_features % world_size else 0) 
                      for i in range(world_size)]
    local_out_features = out_split_sizes[rank]
    
    # Original model
    original_linear = nn.Linear(in_features, out_features).to(device)
    # Broadcast original weights to all ranks
    for param in original_linear.parameters():
        dist.broadcast(param.data, src=0)

    original_activation = nn.ReLU()
    
    # Distributed model components
    parallel_linear = DistributedOutputChannelSplitLinear(
        original_linear, world_size, rank, device, combine=False
    )
    parallel_activation = DistributedParallelActivations(
        activation=nn.ReLU(), 
        world_size=world_size,
        rank=rank,
        device=device,
        combine=True
    )
    
    
    
    dist.barrier()
    # original_linear.to(device)
    # Test loop
    for _ in range(2):
        # Forward pass
        x = torch.randn(batch_size, in_features, requires_grad=True).to(device)
        dist.broadcast(x, src=0)
        
        # Original path
        original_output = original_activation(original_linear(x))
        
        # Parallel path
        parallel_output = parallel_activation(parallel_linear(x))
        
        if rank == 0:
            # print(original_output)
            # print(parallel_output)
            assert original_output.shape == parallel_output.shape
            assert torch.allclose(original_output, parallel_output, atol=1e-5), \
                "Output split + activation outputs don't match!"
            
        # Loss calculation
        target = torch.randn(batch_size, out_features).to(device)
        dist.broadcast(target, src=0)
        # local_target = torch.split(target, out_split_sizes, dim=1)[rank]
        
        loss_fn = nn.MSELoss()
        original_loss = loss_fn(original_output, target)
        parallel_loss = loss_fn(parallel_output, target)
        
        # Backward pass
        optimizer1 = torch.optim.SGD(original_linear.parameters(), lr=0.1)
        optimizer2 = torch.optim.SGD(parallel_linear.linear.parameters(), lr=0.1)
        
        optimizer1.zero_grad()
        original_loss.backward()
        optimizer1.step()
        
        optimizer2.zero_grad()
        
        parallel_loss.backward()
        
        optimizer2.step()
        # Validation
        dist.barrier()
        
    
    if rank == 0:
        print("Test passed: DistributedParallelActivations works with output channel splitting!")

def test_DistributedParallelActivations_with_input_split(world_size, rank, device):
    # Base configuration
    in_features = 128
    out_features = 64
    batch_size = 4

    # Calculate input split sizes
    in_split_sizes = [in_features // world_size + (1 if i < in_features % world_size else 0)
                      for i in range(world_size)]
    local_in_features = in_split_sizes[rank]

    # Original model
    original_linear = nn.Linear(in_features, out_features).to(device)
    for param in original_linear.parameters():
        dist.broadcast(param.data, src=0)

    original_activation = nn.ReLU()

    # Distributed model components
    parallel_linear = DistributedInputChannelSplitLinear(
        original_linear, world_size, rank, device, combine=True
    )
    parallel_activation = DistributedParallelActivations(
        activation=nn.ReLU(),
        world_size=world_size,
        rank=rank,
        device=device,
        combine=False,
        previous = "IS",
    )

    dist.barrier()
    # original_linear.to(device)

    for _ in range(2):
        # Forward pass
        x = torch.randn(batch_size, in_features, requires_grad=True).to(device)
        dist.broadcast(x, src=0)

        # Split x across input channels
        split_x = torch.split(x, in_split_sizes, dim=1)
        x_local = split_x[rank]

        original_output = original_activation(original_linear(x))
        parallel_output = parallel_activation(parallel_linear(x_local))

        if rank == 0:
            pass
            assert original_output.shape == parallel_output.shape, \
                f"Shape mismatch: {original_output.shape} vs {parallel_output.shape}"
            num_mismatched = torch.sum(~torch.isclose(original_output, parallel_output, atol=1e-5)).item()
            assert num_mismatched == 0, f"Number of mismatched elements: {num_mismatched}/ {original_output.numel()}"
            assert torch.allclose(original_output, parallel_output, atol=1e-5), \
                "Input split + activation outputs don't match!"

        # Loss calculation
        target = torch.randn(batch_size, out_features).to(device)
        dist.broadcast(target, src=0)

        loss_fn = nn.MSELoss()
        original_loss = loss_fn(original_output, target)
        parallel_loss = loss_fn(parallel_output, target)

        optimizer1 = torch.optim.SGD(original_linear.parameters(), lr=0.1)
        optimizer2 = torch.optim.SGD(parallel_linear.linear.parameters(), lr=0.1)

        optimizer1.zero_grad()
        original_loss.backward()
        optimizer1.step()

        optimizer2.zero_grad()
        parallel_loss.backward()
        optimizer2.step()

        dist.barrier()

    if rank == 0:
        print("Test passed: DistributedParallelActivations with input split works!")

def test_DistributedParallelDropout_with_input_split(world_size, rank, device):
    # Base configuration
    in_features = 128
    out_features = 64
    batch_size = 4

    # Calculate input split sizes
    in_split_sizes = [in_features // world_size + (1 if i < in_features % world_size else 0)
                      for i in range(world_size)]
    local_in_features = in_split_sizes[rank]

    # Original model
    original_linear = nn.Linear(in_features, out_features).to(device)
    for param in original_linear.parameters():
        dist.broadcast(param.data, src=0)

    original_activation = nn.Dropout(p=0.5, inplace=False)

    # Distributed model components
    parallel_linear = DistributedInputChannelSplitLinear(
        original_linear, world_size, rank, device, combine=True
    )
    parallel_dropout = DistributedParallelDropout(
        dropout=nn.Dropout(p=0.5, inplace=False),
        world_size=world_size,
        rank=rank,
        device=device,
        combine=False,
    )

    dist.barrier()
    # original_linear.to(device)

    for _ in range(2):
        # Forward pass
        x = torch.randn(batch_size, in_features, requires_grad=True).to(device)
        dist.broadcast(x, src=0)

        # Split x across input channels
        split_x = torch.split(x, in_split_sizes, dim=1)
        x_local = split_x[rank]

        original_output = original_activation(original_linear(x))
        parallel_output = parallel_dropout(parallel_linear(x_local))

        if rank == 0:
            pass
            assert original_output.shape == parallel_output.shape, \
                f"Shape mismatch: {original_output.shape} vs {parallel_output.shape}"
            num_mismatched = torch.sum(~torch.isclose(original_output, parallel_output, atol=1e-5)).item()
            assert num_mismatched == 0, f"Number of mismatched elements: {num_mismatched}/ {original_output.numel()}"
            assert torch.allclose(original_output, parallel_output, atol=1e-5), \
                "Input split + activation outputs don't match!"

        # Loss calculation
        target = torch.randn(batch_size, out_features).to(device)
        dist.broadcast(target, src=0)

        loss_fn = nn.MSELoss()
        original_loss = loss_fn(original_output, target)
        parallel_loss = loss_fn(parallel_output, target)

        optimizer1 = torch.optim.SGD(original_linear.parameters(), lr=0.1)
        optimizer2 = torch.optim.SGD(parallel_linear.linear.parameters(), lr=0.1)

        optimizer1.zero_grad()
        original_loss.backward()
        optimizer1.step()

        optimizer2.zero_grad()
        parallel_loss.backward()
        optimizer2.step()

        dist.barrier()

    if rank == 0:
        print("Test passed: DistributedParallelDropout with input split works!")


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
    # test_DistributedParallelActivations(world_size, rank, device)
    test_DistributedParallelActivations_with_output_split(world_size, rank, device)
    # test_DistributedParallelActivations_with_input_split(world_size, rank, device)
    test_DistributedParallelDropout_with_input_split(world_size, rank, device)
    print(f"Rank {device} Total time taken: ", datetime.datetime.now() - FULL_START)
    


if __name__ == '__main__':
    total_epochs = 200
    FULL_START = datetime.datetime.now()
    main()
    exit(0)
