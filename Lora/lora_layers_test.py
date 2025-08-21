import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from .compress import (OutputChannelSplitConv2d, InputChannelSplitConv2d, 
                    OutputChannelSplitLinear, InputChannelSplitLinear, 
                    )
from maths import sum_random_nums_n
import pytest
import copy

class TestOutputChannelSplitConv2d:
    def test_output_channel_split_conv2d():
        torch.manual_seed(42)

        original_conv = nn.Conv2d(in_channels=3, out_channels=8, kernel_size=3, padding=1)
        parallel_conv = OutputChannelSplitConv2d(copy.deepcopy(original_conv), dropout_per=0.5)

        # Frozen check
        assert parallel_conv.weight.requires_grad == False
        if parallel_conv.bias is not None:
            assert parallel_conv.bias.requires_grad == False

        optimizer = torch.optim.SGD(parallel_conv.parameters(), lr=0.1)

        # Dummy input
        x = torch.randn(2, 3, 16, 16, requires_grad=True)

        # Forward
        orig_out = original_conv(x)
        split_out_main, split_out_s = parallel_conv(x)

        # Check consistency
        assert torch.allclose(orig_out, split_out_main, atol=1e-6), "Main conv forward mismatch!"
        assert torch.allclose(split_out_s, torch.zeros_like(split_out_s)), "Split branch should start at zero!"

        # Train step
        target = torch.randn_like(orig_out)
        loss = F.mse_loss(split_out_main , target)
        loss.backward()
        optimizer.step()

        # Frozen check
        assert torch.allclose(original_conv.weight, parallel_conv.weight), "Frozen weight changed!"
        if original_conv.bias is not None:
            assert torch.allclose(original_conv.bias, parallel_conv.bias), "Frozen bias changed!"

        # Split params updated
        # assert not torch.allclose(parallel_conv.weight_s, torch.zeros_like(parallel_conv.weight_s)), "Split weight did not update!"
        # if parallel_conv.bias_s is not None:
        #     assert not torch.allclose(parallel_conv.bias_s, torch.zeros_like(parallel_conv.bias_s)), "Split bias did not update!"

    print("✅ Test passed: OutputChannelSplitConv2d works correctly.")

    # test_output_channel_split_conv2d()

class TestInputChannelSplitConv2d:
    def test_input_channel_split_conv2d():
        torch.manual_seed(42)

        original_conv = nn.Conv2d(in_channels=6, out_channels=8, kernel_size=3, padding=1)
        parallel_conv = InputChannelSplitConv2d(copy.deepcopy(original_conv), dropout_per=0.5)

        # Frozen check
        assert parallel_conv.weight.requires_grad == False
        if parallel_conv.bias is not None:
            assert parallel_conv.bias.requires_grad == False

        optimizer = torch.optim.SGD(parallel_conv.parameters(), lr=0.1)

        # Dummy input: main & split parts
        x_main = torch.randn(2, 6, 16, 16, requires_grad=True)
        x_split = torch.randn(2, int(6*0.5), 16, 16, requires_grad=True)

        # Forward
        orig_out = original_conv(x_main)
        split_out = parallel_conv((x_main, x_split))

        # Check consistency
        assert torch.allclose(orig_out, split_out*2, atol=1e-6), "InputSplitConv forward mismatch!"

        # Train step
        target = torch.randn_like(orig_out)
        loss = F.mse_loss(split_out, target)
        loss.backward()
        optimizer.step()

        # Frozen check
        assert torch.allclose(original_conv.weight, parallel_conv.weight), "Frozen weight changed!"
        if original_conv.bias is not None:
            assert torch.allclose(original_conv.bias, parallel_conv.bias), "Frozen bias changed!"

        # Split params updated
        assert not torch.allclose(parallel_conv.weight_s, torch.zeros_like(parallel_conv.weight_s)), "Split weight did not update!"
        if parallel_conv.bias_s is not None:
            assert not torch.allclose(parallel_conv.bias_s, torch.zeros_like(parallel_conv.bias_s)), "Split bias did not update!"

        print("✅ Test passed: InputChannelSplitConv2d works correctly.")

    # test_input_channel_split_conv2d()


class TestOutputChannelSplitLinear:
    def test_output_channel_split_linear():
        torch.manual_seed(42)

        # Create original Linear layer
        original_linear = nn.Linear(in_features=16, out_features=64)

        # Clone for parallel split layer
        parallel_linear = OutputChannelSplitLinear(copy.deepcopy(original_linear), dropout_per=0.5)

        # Check frozen weights
        assert parallel_linear.weight.requires_grad == False, "Original weight should be frozen!"
        if parallel_linear.bias is not None:
            assert parallel_linear.bias.requires_grad == False, "Original bias should be frozen!"

        # Optimizer should only see split params
        optimizer = torch.optim.SGD(parallel_linear.parameters(), lr=0.1)
        param_names = [name for name, _ in parallel_linear.named_parameters()]
        assert "weight_s" in param_names, "Split weight not found in parameters!"
        if parallel_linear.bias is not None:
            assert "bias_s" in param_names, "Split bias not found in parameters!"

        # Dummy input
        x = torch.randn(4, 16, requires_grad=True)

        # Forward pass comparison (main branch)
        orig_out = original_linear(x)
        split_out_main, split_out_s = parallel_linear(x)

        assert torch.allclose(orig_out, split_out_main, atol=1e-6), "Main branch output should match original Linear!"
        assert torch.allclose(split_out_s, torch.zeros_like(split_out_s)), "Split output should start at zero!"

        # Dummy loss
        target = torch.randn_like(orig_out)
        loss = F.mse_loss(split_out_main + split_out_s, target)
        loss.backward()
        optimizer.step()

        # After one step: frozen weights unchanged
        assert torch.allclose(original_linear.weight, parallel_linear.weight), "Frozen weight changed!"
        if original_linear.bias is not None:
            assert torch.allclose(original_linear.bias, parallel_linear.bias), "Frozen bias changed!"

        # Split params should move away from zero
        assert not torch.allclose(parallel_linear.weight_s, torch.zeros_like(parallel_linear.weight_s)), "Split weight did not update!"
        if parallel_linear.bias_s is not None:
            assert not torch.allclose(parallel_linear.bias_s, torch.zeros_like(parallel_linear.bias_s)), "Split bias did not update!"

        print("✅ Test passed: Frozen weights, split branch learns, outputs consistent.")


    # Run the test
    # test_output_channel_split_linear()

class TestInputChannelSplitLinear:
    

    def test_input_channel_split_linear():
        torch.manual_seed(42)

        original_linear = nn.Linear(in_features=16, out_features=32)
        parallel_linear = InputChannelSplitLinear(copy.deepcopy(original_linear), dropout_per=0.5)

        # Frozen check
        assert parallel_linear.weight.requires_grad == False
        if parallel_linear.bias is not None:
            assert parallel_linear.bias.requires_grad == False

        optimizer = torch.optim.SGD(parallel_linear.parameters(), lr=0.1)

        # Dummy input: main & split parts
        x_main = torch.randn(4, 16, requires_grad=True)
        x_split = torch.randn(4, int(16*0.5), requires_grad=True)

        # Forward
        orig_out = original_linear(x_main)
        split_out = parallel_linear((x_main, x_split))

        # Check consistency
        assert torch.allclose(orig_out, split_out*2, atol=1e-6), "InputSplit forward mismatch before training!"

        # Train step
        target = torch.randn_like(orig_out)
        loss = F.mse_loss(split_out, target)
        loss.backward()
        optimizer.step()

        # Frozen check
        assert torch.allclose(original_linear.weight, parallel_linear.weight), "Frozen weight changed!"
        if original_linear.bias is not None:
            assert torch.allclose(original_linear.bias, parallel_linear.bias), "Frozen bias changed!"

        # print(parallel_linear.weight_s)
        # Split params updated
        assert not torch.allclose(parallel_linear.weight_s, torch.zeros_like(parallel_linear.weight_s)), "Split weight did not update!"
        if parallel_linear.bias_s is not None:
            assert not torch.allclose(parallel_linear.bias_s, torch.zeros_like(parallel_linear.bias_s)), "Split bias did not update!"

        print("✅ Test passed: InputChannelSplitLinear works correctly.")

    # test_input_channel_split_linear()


