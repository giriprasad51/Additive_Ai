import torch
import torch.nn as nn
import torch.distributed as dist
import torch.multiprocessing as mp
# from torch.nn.parallel import DistributedDataParallel as DDP
import os, datetime
from Dis.layers1 import DistributedOutputChannelSplitConv2d
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
        
        # original_grad = original_conv.weight.grad
        # original_grad = torch.split(original_grad, [original_grad.shape[0] // world_size + (1 if i < original_grad.shape[0] %world_size else 0) for i in range(world_size) ], dim=0)
        # original_grad = original_grad[rank]
        # parallel_grad = parallel_conv.conv.weight.grad
        # check_outputs(original_grad, parallel_grad)

        # original_grad_b = original_conv.bias.grad
        # original_grad_b = torch.split(original_grad_b, [original_grad_b.shape[0] // world_size + (1 if i < original_grad_b.shape[0] %world_size else 0) for i in range(world_size) ], dim=0)
        # original_grad_b = original_grad_b[rank]
        # parallel_grad_b = parallel_conv.conv.bias.grad
        # print(original_grad_b)
        # print(parallel_grad_b)
        # print(parallel_grad_b/original_grad_b)
        # check_outputs(original_grad_b, parallel_grad_b)

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
    


def main():
    rank = int(os.environ['RANK'])
    world_size = int(os.environ['WORLD_SIZE'])
    gpu_id = int(os.environ['LOCAL_RANK'])
    device = 'cuda:'+str(gpu_id)
    

    print(torch.cuda.device_count())
    dist.init_process_group(backend='gloo', timeout=datetime.timedelta(seconds=7200))
    # Run the test
    test_DistributedOutputChannelSplitConv2d(world_size, rank, device)
    
    print(f"Rank {device} Total time taken: ", datetime.datetime.now() - FULL_START)
    


if __name__ == '__main__':
    total_epochs = 200
    FULL_START = datetime.datetime.now()
    main()
    exit(0)
