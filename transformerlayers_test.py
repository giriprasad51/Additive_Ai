import torch
import torch.nn as nn
from .transformerlayers import OutputChannelSplitConv1DGPT,InputChannelSplitConv1DGPT
from transformers.pytorch_utils import Conv1D

from maths import sum_random_nums_n
import pytest

class TestGPTStyleConvSplits:
    def test_output_channel_split(self):
        for n in range(2, 32):
            conv = Conv1D(nf=64, nx=32)
            split_conv = OutputChannelSplitConv1DGPT(conv, num_splits=n)

            x = torch.randn(4, 32, requires_grad=True)

            out1 = conv(x)
            out2 = split_conv(x)

            assert out1.shape == out2.shape
            assert torch.allclose(out1, out2, atol=1e-5), f"Output mismatch at split {n}"

            loss_fn = nn.MSELoss()
            target = torch.randn_like(out1)
            loss1 = loss_fn(out1, target)
            loss1.backward()
            grad1 = x.grad.clone()
            x.grad.zero_()
            loss2 = loss_fn(out2, target)
            loss2.backward()
            grad2 = x.grad.clone()

            assert torch.allclose(grad1, grad2, atol=1e-5), f"Grad mismatch at split {n}"

            print(f"[✓] OutputChannelSplitConv1DGPT passed for split={n}")

    def test_input_channel_split(self):
        for n in range(2, 32):
            conv = Conv1D(nf=64, nx=32)
            split_conv = InputChannelSplitConv1DGPT(conv, num_splits=n)

            x = torch.randn(4, 32, requires_grad=True)

            out1 = conv(x)
            out2 = split_conv(x)

            assert out1.shape == out2.shape
            assert torch.allclose(out1, out2, atol=1e-5), f"Output mismatch at split {n}"

            loss_fn = nn.MSELoss()
            target = torch.randn_like(out1)
            loss1 = loss_fn(out1, target)
            loss1.backward()
            grad1 = x.grad.clone()
            x.grad.zero_()
            loss2 = loss_fn(out2, target)
            loss2.backward()
            grad2 = x.grad.clone()

            assert torch.allclose(grad1, grad2, atol=1e-5), f"Grad mismatch at split {n}"

            print(f"[✓] InputChannelSplitConv1DGPT passed for split={n}")
