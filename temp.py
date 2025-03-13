import torch
import torch.nn as nn
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
import os, datetime


def setup(rank, world_size):
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)


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
        
        self.conv = nn.Conv2d(conv_layer.in_channels, self.split_channels, kernel_size=conv_layer.kernel_size,
                              stride=conv_layer.stride, padding=conv_layer.padding).to(self.device)
        self.copy_weights_from(conv_layer)
        self.conv = DDP(self.conv, device_ids=[rank])

    def forward(self, x):
        x = x.to(self.device)
        output = self.conv(x)
        if self.combine:
            gathered_output = [torch.zeros_like(output) for _ in range(self.world_size)]
            dist.all_gather(gathered_output, output)
            if self.rank == 0:
                return torch.cat(gathered_output, dim=1)
            else:
                return None
        return output

    def copy_weights_from(self, original_layer: nn.Conv2d):
        with torch.no_grad():
            self.conv.weight.copy_(original_layer.weight[self.rank * self.split_channels:(self.rank + 1) * self.split_channels])
            self.conv.bias.copy_(original_layer.bias[self.rank * self.split_channels:(self.rank + 1) * self.split_channels])

def run(rank, world_size, model):
    setup(rank, world_size)
    model = model.to(rank)
    model = DDP(model, device_ids=[rank])
    
    x = torch.randn(1, 3, 32, 32).to(rank)
    output = model(x)
    if rank == 0 and output is not None:
        print("Output shape:", output.shape)
    
    cleanup()


def ddp_setup(rank:int, world_size:int):

	#torchrun will handle this
	'''
	os.environ['MASTER_ADDR'] = 'localhost'
	os.environ['MASTER_PORT'] = '25000'
	'''
	#os.environ['CUDA_VISIBLE_DEVICES'] = os.environ['LOCAL_RANK']
	dist.init_process_group("nccl", timeout=datetime.timedelta(seconds=7200), rank=rank, world_size=world_size)


def main():
    ddp_setup(int(os.environ['RANK']), int(os.environ['WORLD_SIZE']))
    rank = int(os.environ['RANK'])
    world_size = int(os.environ['WORLD_SIZE'])
    conv = nn.Conv2d(3, 8, 3, 1, 1)
    model = DistributedOutputChannelSplitConv2d(conv, world_size, rank=rank, combine=True)

    x = torch.randn(1, 3, 32, 32).to(rank)
    output = model(x)
    if rank == 0 and output is not None:
        print("Output shape:", output.shape)
    
    cleanup()

if __name__ == '__main__':
	total_epochs = 200
	FULL_START = datetime.datetime.now()
	main()
	print("Total time taken: ", datetime.datetime.now()-FULL_START)
	exit(0)
