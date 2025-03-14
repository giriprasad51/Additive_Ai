import torch
import torch.nn as nn
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP

class OutputChannelSplitConv2d(nn.Module):
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