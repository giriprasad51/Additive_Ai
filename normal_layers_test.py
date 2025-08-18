import torch
import torch.nn as nn
from .layers import (OutputChannelSplitConv2d, InputChannelSplitConv2d, 
                    OutputChannelSplitLinear, InputChannelSplitLinear, 
                    ParallelMaxPool2d, InputChannelSplitConv1D, OutputChannelSplitConv1D)
from .maths import sum_random_nums_n
import pytest


class TestOutputChannelSplitConv2d:

    def test_OutputChannelSplitConv2d(self):
        # Create original Conv2d layer
        original_conv = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, stride=1, padding=1)
        
        print(original_conv.weight.shape)
        for n in range(2, 64):
            # Assume OutputChannelSplitConv2d is correctly implemented
            parallel_conv = OutputChannelSplitConv2d(original_conv, num_splits=n)
            
            # Create a dummy input tensor with requires_grad=True for gradient testing
            x = torch.randn(1, 3, 32, 32, requires_grad=True)
            
            # Get outputs from both layers
            original_output = original_conv(x)
            parallel_output = parallel_conv(x)
            
            # Check shape consistency
            assert original_output.shape == parallel_output.shape, f"Shape mismatch: {original_output.shape} vs {parallel_output.shape}"
            
            # Check number of elements mismatch
            num_mismatched = torch.sum(~torch.isclose(original_output, parallel_output, atol=1e-5)).item()
            assert num_mismatched == 0, f"Number of mismatched elements: {num_mismatched}/ {original_output.numel()}"

            # Check output similarity
            assert torch.allclose(original_output, parallel_output, atol=1e-5), "Outputs do not match!"

            # Create a dummy loss function
            loss_fn = nn.MSELoss()
            
            # Create dummy target tensor
            target = torch.randn_like(original_output)

            # Compute loss and backward for original Conv2d
            original_loss = loss_fn(original_output, target)
            original_loss.backward()
            original_grad = x.grad.clone()  # Save original gradients

            # Reset gradients
            x.grad.zero_()

            # Compute loss and backward for OutputChannelSplitConv2d
            parallel_loss = loss_fn(parallel_output, target)
            parallel_loss.backward()
            parallel_grad = x.grad.clone()  # Save parallel gradients

            # Check shape consistency
            assert original_grad.shape == parallel_grad.shape, f"Shape mismatch: {original_grad.shape} vs {parallel_grad.shape}"

            # Check number of elements mismatch
            num_mismatched = torch.sum(~torch.isclose(original_grad, parallel_grad, atol=1e-5)).item()
            assert num_mismatched == 0, f"Number of mismatched elements: {num_mismatched} / {original_grad.numel()}"

            # Check gradient consistency
            assert torch.allclose(original_grad, parallel_grad, atol=1e-5), "Gradient mismatch detected!"


            print("Test passed: Forward and backward pass of OutputChannelSplitConv2d match standard Conv2d!")

    def test_split_channels(self):

        # Create original Conv2d layer
        original_conv = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, stride=1, padding=1)
        

        parallel_conv = OutputChannelSplitConv2d(original_conv, num_splits=2, split_channels=[32,32])
        # Create a dummy input tensor with requires_grad=True for gradient testing
        x = torch.randn(1, 3, 32, 32, requires_grad=True)
        
        # Get outputs from both layers
        original_output = original_conv(x)
        parallel_output = parallel_conv(x)
        
        # Check shape consistency
        assert original_output.shape == parallel_output.shape, f"Shape mismatch: {original_output.shape} vs {parallel_output.shape}"
        
        # Check number of elements mismatch
        num_mismatched = torch.sum(~torch.isclose(original_output, parallel_output, atol=1e-5)).item()
        assert num_mismatched == 0, f"Number of mismatched elements: {num_mismatched}/ {original_output.numel()}"

        # Check output similarity
        assert torch.allclose(original_output, parallel_output, atol=1e-5), "Outputs do not match!"

        # Create a dummy loss function
        loss_fn = nn.MSELoss()
        
        # Create dummy target tensor
        target = torch.randn_like(original_output)

        # Compute loss and backward for original Conv2d
        original_loss = loss_fn(original_output, target)
        original_loss.backward()
        original_grad = x.grad.clone()  # Save original gradients

        # Reset gradients
        x.grad.zero_()

        # Compute loss and backward for OutputChannelSplitConv2d
        parallel_loss = loss_fn(parallel_output, target)
        parallel_loss.backward()
        parallel_grad = x.grad.clone()  # Save parallel gradients

        # Check shape consistency
        assert original_grad.shape == parallel_grad.shape, f"Shape mismatch: {original_grad.shape} vs {parallel_grad.shape}"

        # Check number of elements mismatch
        num_mismatched = torch.sum(~torch.isclose(original_grad, parallel_grad, atol=1e-5)).item()
        assert num_mismatched == 0, f"Number of mismatched elements: {num_mismatched} / {original_grad.numel()}"

        # Check gradient consistency
        assert torch.allclose(original_grad, parallel_grad, atol=1e-5), "Gradient mismatch detected!"


        print("Test passed: Forward and backward pass of OutputChannelSplitConv2d match standard Conv2d!")

    def test_change_split_channels(self):
        # Create original Conv2d layer
        original_conv = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, stride=1, padding=1)
        

        parallel_conv = OutputChannelSplitConv2d(original_conv, num_splits=2)
        # Create a dummy input tensor with requires_grad=True for gradient testing
        
        
        for _ in range(20):
            x = torch.randn(1, 3, 32, 32, requires_grad=True)
            split_channels = sum_random_nums_n(64)
            print(split_channels)
            parallel_conv.change_split_channels(split_channels)
            # Get outputs from both layers
            original_output = original_conv(x)
            parallel_output = parallel_conv(x)
            
            # Check shape consistency
            assert original_output.shape == parallel_output.shape, f"Shape mismatch: {original_output.shape} vs {parallel_output.shape}"
            
            # Check number of elements mismatch
            num_mismatched = torch.sum(~torch.isclose(original_output, parallel_output, atol=1e-5)).item()
            assert num_mismatched == 0, f"Number of mismatched elements: {num_mismatched}/ {original_output.numel()}"

            # Check output similarity
            assert torch.allclose(original_output, parallel_output, atol=1e-5), "Outputs do not match!"

            # Create a dummy loss function
            loss_fn = nn.MSELoss()
            
            # Create dummy target tensor
            target = torch.randn_like(original_output)

            # Compute loss and backward for original Conv2d
            original_loss = loss_fn(original_output, target)
            original_loss.backward()
            original_grad = x.grad.clone()  # Save original gradients

            # Reset gradients
            x.grad.zero_()

            # Compute loss and backward for OutputChannelSplitConv2d
            parallel_loss = loss_fn(parallel_output, target)
            parallel_loss.backward()
            parallel_grad = x.grad.clone()  # Save parallel gradients

            # Check shape consistency
            assert original_grad.shape == parallel_grad.shape, f"Shape mismatch: {original_grad.shape} vs {parallel_grad.shape}"

            # Check number of elements mismatch
            num_mismatched = torch.sum(~torch.isclose(original_grad, parallel_grad, atol=1e-5)).item()
            assert num_mismatched == 0, f"Number of mismatched elements: {num_mismatched} / {original_grad.numel()}"

            # Check gradient consistency
            assert torch.allclose(original_grad, parallel_grad, atol=1e-5), "Gradient mismatch detected!"


            print("Test passed: Forward and backward pass of OutputChannelSplitConv2d match standard Conv2d!")


    # Run the test
    # test_OutputChannelSplitConv2d()

class TestInputChannelSplitConv2d:

    def test_InputChannelSplitConv2d(self):
            
        for n in range(4,64):
            # Create original Conv2d layer
            original_conv = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1)
            # Create InputChannelSplitConv2d layer
            split_conv = InputChannelSplitConv2d(original_conv, num_splits=n)

            # Create a dummy input tensor
            x = torch.randn(8, 64, 32, 32, requires_grad=True)

            # Get outputs from both layers
            original_output = original_conv(x)
            split_output = split_conv(x)

            # Check shape consistency
            assert original_output.shape == split_output.shape, f"Shape mismatch: {original_output.shape} vs {split_output.shape}"

            # Check number of elements mismatch
            num_mismatched = torch.sum(~torch.isclose(original_output, split_output, atol=1e-5)).item()
            assert num_mismatched == 0, f"Number of mismatched elements: {num_mismatched} / {original_output.numel()}"
            
            # Check output similarity
            assert torch.allclose(original_output, split_output, atol=1e-5), "Outputs do not match!"

            # Create a dummy loss function
            loss_fn = nn.MSELoss()
            
            # Create dummy target tensor
            target = torch.randn_like(original_output)

            # Compute loss and backward for original Conv2d
            original_loss = loss_fn(original_output, target)
            original_loss.backward()
            original_grad = x.grad.clone()  # Save original gradients

            # Reset gradients
            x.grad.zero_()

            # Compute loss and backward for OutputChannelSplitConv2d
            parallel_loss = loss_fn(split_output, target)
            parallel_loss.backward()
            parallel_grad = x.grad.clone()  # Save parallel gradients

            # Check shape consistency
            assert original_grad.shape == parallel_grad.shape, f"Shape mismatch: {original_grad.shape} vs {parallel_grad.shape}"

            # Check number of elements mismatch
            num_mismatched = torch.sum(~torch.isclose(original_grad, parallel_grad, atol=1e-5)).item()
            assert num_mismatched == 0, f"Number of mismatched elements: {num_mismatched}/ {original_grad.numel()}"

            # Check gradient consistency
            assert torch.allclose(original_grad, parallel_grad, atol=1e-5), "Gradient mismatch detected!"


            print("Test passed: Forward and backward pass of InputChannelSplitConv2d matches standard Conv2d!")

    def test_change_split_channels(self):

    
        # Create original Conv2d layer
        original_conv = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1)
        # Create InputChannelSplitConv2d layer
        split_conv = InputChannelSplitConv2d(original_conv, split_channels=[32,32])

        for _ in range(20):
            
            split_channels = sum_random_nums_n(64)
            print(split_channels)
            split_conv.change_split_channels(split_channels)
            # Create a dummy input tensor
            x = torch.randn(1, 64, 32, 32, requires_grad=True)

            # Get outputs from both layers
            original_output = original_conv(x)
            split_output = split_conv(x)

            # Check shape consistency
            assert original_output.shape == split_output.shape, f"Shape mismatch: {original_output.shape} vs {split_output.shape}"

            # Check number of elements mismatch
            num_mismatched = torch.sum(~torch.isclose(original_output, split_output, atol=1e-5)).item()
            assert num_mismatched == 0, f"Number of mismatched elements: {num_mismatched} / {original_output.numel()}"
            
            # Check output similarity
            assert torch.allclose(original_output, split_output, atol=1e-5), "Outputs do not match!"

            # Create a dummy loss function
            loss_fn = nn.MSELoss()
            
            # Create dummy target tensor
            target = torch.randn_like(original_output)

            # Compute loss and backward for original Conv2d
            original_loss = loss_fn(original_output, target)
            original_loss.backward()
            original_grad = x.grad.clone()  # Save original gradients

            # Reset gradients
            x.grad.zero_()

            # Compute loss and backward for OutputChannelSplitConv2d
            parallel_loss = loss_fn(split_output, target)
            parallel_loss.backward()
            parallel_grad = x.grad.clone()  # Save parallel gradients

            # Check shape consistency
            assert original_grad.shape == parallel_grad.shape, f"Shape mismatch: {original_grad.shape} vs {parallel_grad.shape}"

            # Check number of elements mismatch
            num_mismatched = torch.sum(~torch.isclose(original_grad, parallel_grad, atol=1e-5)).item()
            assert num_mismatched == 0, f"Number of mismatched elements: {num_mismatched}/ {original_grad.numel()}"

            # Check gradient consistency
            assert torch.allclose(original_grad, parallel_grad, atol=1e-5), "Gradient mismatch detected!"


            print("Test passed: Forward and backward pass of InputChannelSplitConv2d matches standard Conv2d!")


    # Run the test
    # test_InputChannelSplitConv2d()

class TestOutputChannelSplitConv1D:
    def test_output_channel_split_conv1d(self):
        for n in range(2, 32):  # output splits
            conv = nn.Conv1d(in_channels=8, out_channels=32, kernel_size=3, padding=1)
            split_conv = OutputChannelSplitConv1D(conv, num_splits=n)

            x = torch.randn(1, 8, 16, requires_grad=True)

            original_out = conv(x)
            split_out = split_conv(x)

            # Output match
            assert original_out.shape == split_out.shape, f"Shape mismatch: {original_out.shape} vs {split_out.shape}"
            assert torch.allclose(original_out, split_out, atol=1e-5), "Forward outputs don't match"

            # Gradient check
            loss_fn = nn.MSELoss()
            target = torch.randn_like(original_out)

            loss_original = loss_fn(original_out, target)
            loss_original.backward()
            grad_original = x.grad.clone()

            x.grad.zero_()
            loss_split = loss_fn(split_out, target)
            loss_split.backward()
            grad_split = x.grad.clone()

            assert grad_original.shape == grad_split.shape, f"Grad shape mismatch: {grad_original.shape} vs {grad_split.shape}"
            assert torch.allclose(grad_original, grad_split, atol=1e-5), "Gradient mismatch"

            print(f"Passed OutputChannelSplitConv1D test with num_splits={n}")


class TestInputChannelSplitConv1D:
    def test_input_channel_split_conv1d(self):
        for n in range(2, 32):  # input splits
            conv = nn.Conv1d(in_channels=32, out_channels=16, kernel_size=3, padding=1)
            split_conv = InputChannelSplitConv1D(conv, num_splits=n)

            x = torch.randn(1, 32, 16, requires_grad=True)

            original_out = conv(x)
            split_out = split_conv(x)

            # Output match
            assert original_out.shape == split_out.shape, f"Shape mismatch: {original_out.shape} vs {split_out.shape}"
            assert torch.allclose(original_out, split_out, atol=1e-5), "Forward outputs don't match"

            # Gradient check
            loss_fn = nn.MSELoss()
            target = torch.randn_like(original_out)

            loss_original = loss_fn(original_out, target)
            loss_original.backward()
            grad_original = x.grad.clone()

            x.grad.zero_()
            loss_split = loss_fn(split_out, target)
            loss_split.backward()
            grad_split = x.grad.clone()

            assert grad_original.shape == grad_split.shape, f"Grad shape mismatch: {grad_original.shape} vs {grad_split.shape}"
            assert torch.allclose(grad_original, grad_split, atol=1e-5), "Gradient mismatch"

            print(f"Passed InputChannelSplitConv1D test with num_splits={n}")


class TestOutputChannelSplitLinear:

    def test_OutputChannelSplitLinear(self):

        for n in range(2,64):        
            # Create original Linear layer
            original_linear = nn.Linear(in_features=16, out_features=64)
            
            # Create OutputChannelSplitLinear layer
            parallel_linear = OutputChannelSplitLinear(original_linear, num_splits=n)
            
            
            # Create a dummy input tensor
            x = torch.randn(1, 16, requires_grad=True)
            
            # Get outputs from both layers
            original_output = original_linear(x)
            parallel_output = parallel_linear(x)
            
            # Check shape consistency
            assert original_output.shape == parallel_output.shape, f"Shape mismatch: {original_output.shape} vs {parallel_output.shape}"
            
            # Check number of elements mismatch
            num_mismatched = torch.sum(~torch.isclose(original_output, parallel_output, atol=1e-5)).item()
            assert num_mismatched == 0, f"Number of mismatched elements: {num_mismatched} / {original_output.numel()}"

            # Check output similarity
            assert torch.allclose(original_output, parallel_output, atol=1e-5), "Outputs do not match!"
            

            # Create a dummy loss function
            loss_fn = nn.MSELoss()
            
            # Create dummy target tensor
            target = torch.randn_like(original_output)

            # Compute loss and backward for original Conv2d
            original_loss = loss_fn(original_output, target)
            original_loss.backward()
            original_grad = x.grad.clone()  # Save original gradients

            # Reset gradients
            x.grad.zero_()

            # Compute loss and backward for OutputChannelSplitConv2d
            parallel_loss = loss_fn(parallel_output, target)
            parallel_loss.backward()
            parallel_grad = x.grad.clone()  # Save parallel gradients

            # Check shape consistency
            assert original_grad.shape == parallel_grad.shape, f"Shape mismatch: {original_grad.shape} vs {parallel_grad.shape}"

            # Check number of elements mismatch
            num_mismatched = torch.sum(~torch.isclose(original_grad, parallel_grad, atol=1e-5)).item()
            assert num_mismatched == 0, f"Number of mismatched elements: {num_mismatched}/ {original_grad.numel()}"

            # Check gradient consistency
            assert torch.allclose(original_grad, parallel_grad, atol=1e-5), "Gradient mismatch detected!"


            print("Test passed: Forward and backward pass of OutputChannelSplitLinear matches standard Linear!")

    def test_change_split_channels(self):
        # Create original Linear layer
        original_linear = nn.Linear(in_features=16, out_features=64)
        
        # Create OutputChannelSplitLinear layer
        parallel_linear = OutputChannelSplitLinear(original_linear, split_channels=[32,32])
        
        for _ in range(20):
            
            split_channels = sum_random_nums_n(64)
            print(split_channels)

            parallel_linear.change_split_channels(split_channels=split_channels)
            # Create a dummy input tensor
            x = torch.randn(1, 16, requires_grad=True)
            
            # Get outputs from both layers
            original_output = original_linear(x)
            parallel_output = parallel_linear(x)
            
            # Check shape consistency
            assert original_output.shape == parallel_output.shape, f"Shape mismatch: {original_output.shape} vs {parallel_output.shape}"
            
            # Check number of elements mismatch
            num_mismatched = torch.sum(~torch.isclose(original_output, parallel_output, atol=1e-5)).item()
            assert num_mismatched == 0, f"Number of mismatched elements: {num_mismatched} / {original_output.numel()}"

            # Check output similarity
            assert torch.allclose(original_output, parallel_output, atol=1e-5), "Outputs do not match!"
            

            # Create a dummy loss function
            loss_fn = nn.MSELoss()
            
            # Create dummy target tensor
            target = torch.randn_like(original_output)

            # Compute loss and backward for original Conv2d
            original_loss = loss_fn(original_output, target)
            original_loss.backward()
            original_grad = x.grad.clone()  # Save original gradients

            # Reset gradients
            x.grad.zero_()

            # Compute loss and backward for OutputChannelSplitConv2d
            parallel_loss = loss_fn(parallel_output, target)
            parallel_loss.backward()
            parallel_grad = x.grad.clone()  # Save parallel gradients

            # Check shape consistency
            assert original_grad.shape == parallel_grad.shape, f"Shape mismatch: {original_grad.shape} vs {parallel_grad.shape}"

            # Check number of elements mismatch
            num_mismatched = torch.sum(~torch.isclose(original_grad, parallel_grad, atol=1e-5)).item()
            assert num_mismatched == 0, f"Number of mismatched elements: {num_mismatched}/ {original_grad.numel()}"

            # Check gradient consistency
            assert torch.allclose(original_grad, parallel_grad, atol=1e-5), "Gradient mismatch detected!"


            print("Test passed: Forward and backward pass of OutputChannelSplitLinear matches standard Linear!")

    # Run the test
    # test_OutputChannelSplitLinear()

class TestInputChannelSplitLinear:

    def test_InputChannelSplitLinear(self):
        
        for n in range(2,64):
            # Create original Linear layer
            original_linear = nn.Linear(in_features=64, out_features=64)
            
            # Create InputChannelSplitLinear layer
            split_linear = InputChannelSplitLinear(original_linear, num_splits=n)
            
            # Create a dummy input tensor
            x = torch.randn(1, 64, requires_grad=True)
            
            # Get outputs from both layers
            original_output = original_linear(x)
            split_output = split_linear(x)
            
            # Check shape consistency
            assert original_output.shape == split_output.shape, f"Shape mismatch: {original_output.shape} vs {split_output.shape}"
            
            # Check number of elements mismatch
            num_mismatched = torch.sum(~torch.isclose(original_output, split_output, atol=1e-5)).item()
            assert num_mismatched == 0, f"Number of mismatched elements: {num_mismatched} / {original_output.numel()}"
            
            # Check output similarity
            assert torch.allclose(original_output, split_output, atol=1e-5), "Outputs do not match!"
            

            # Create a dummy loss function
            loss_fn = nn.MSELoss()
            
            # Create dummy target tensor
            target = torch.randn_like(original_output)

            # Compute loss and backward for original Conv2d
            original_loss = loss_fn(original_output, target)
            original_loss.backward()
            original_grad = x.grad.clone()  # Save original gradients

            # Reset gradients
            x.grad.zero_()

            # Compute loss and backward for OutputChannelSplitConv2d
            parallel_loss = loss_fn(split_output, target)
            parallel_loss.backward()
            parallel_grad = x.grad.clone()  # Save parallel gradients

            # Check shape consistency
            assert original_grad.shape == parallel_grad.shape, f"Shape mismatch: {original_grad.shape} vs {parallel_grad.shape}"

            # Check number of elements mismatch
            num_mismatched = torch.sum(~torch.isclose(original_grad, parallel_grad, atol=1e-5)).item()
            assert num_mismatched == 0, f"Number of mismatched elements: {num_mismatched}/ {original_grad.numel()}"

            # Check gradient consistency
            assert torch.allclose(original_grad, parallel_grad, atol=1e-5), "Gradient mismatch detected!"


            print("Test passed: Forward and backward pass of InputChannelSplitLinear matches standard Linear!")

    def test_change_split_channels(self):

        # Create original Linear layer
        original_linear = nn.Linear(in_features=64, out_features=64)
        
        # Create InputChannelSplitLinear layer
        split_linear = InputChannelSplitLinear(original_linear, split_channels=[32,32])

        for _ in range(20):
            
            split_channels = sum_random_nums_n(64)
            print(split_channels)

            split_linear.change_split_channels(split_channels=split_channels)

            # Create a dummy input tensor
            x = torch.randn(1, 64, requires_grad=True)
            
            # Get outputs from both layers
            original_output = original_linear(x)
            split_output = split_linear(x)
            
            # Check shape consistency
            assert original_output.shape == split_output.shape, f"Shape mismatch: {original_output.shape} vs {split_output.shape}"
            
            # Check number of elements mismatch
            num_mismatched = torch.sum(~torch.isclose(original_output, split_output, atol=1e-5)).item()
            assert num_mismatched == 0, f"Number of mismatched elements: {num_mismatched} / {original_output.numel()}"
            
            # Check output similarity
            assert torch.allclose(original_output, split_output, atol=1e-5), "Outputs do not match!"
            

            # Create a dummy loss function
            loss_fn = nn.MSELoss()
            
            # Create dummy target tensor
            target = torch.randn_like(original_output)

            # Compute loss and backward for original Conv2d
            original_loss = loss_fn(original_output, target)
            original_loss.backward()
            original_grad = x.grad.clone()  # Save original gradients

            # Reset gradients
            x.grad.zero_()

            # Compute loss and backward for OutputChannelSplitConv2d
            parallel_loss = loss_fn(split_output, target)
            parallel_loss.backward()
            parallel_grad = x.grad.clone()  # Save parallel gradients

            # Check shape consistency
            assert original_grad.shape == parallel_grad.shape, f"Shape mismatch: {original_grad.shape} vs {parallel_grad.shape}"

            # Check number of elements mismatch
            num_mismatched = torch.sum(~torch.isclose(original_grad, parallel_grad, atol=1e-5)).item()
            assert num_mismatched == 0, f"Number of mismatched elements: {num_mismatched}/ {original_grad.numel()}"

            # Check gradient consistency
            assert torch.allclose(original_grad, parallel_grad, atol=1e-5), "Gradient mismatch detected!"


            print("Test passed: Forward and backward pass of InputChannelSplitLinear matches standard Linear!")


    # Run the test
    # test_InputChannelSplitLinear()

def test_ParallelMaxPool2d():
    # Define a max pooling layer
    pool_layer = nn.MaxPool2d(kernel_size=2, stride=2)
    parallel_pool = ParallelMaxPool2d(pool_layer, combine="cat")
    
    # Create dummy input tensor
    x = torch.randn(16, 16, 32, 32)
    x1 = list(torch.split(x, [8,5,3], dim=0))
    
    
    # Test with single tensor input
    expected_single = pool_layer(x)
    pooled_single = parallel_pool(x1)
    
    
    # Check shape consistency
    assert pooled_single.shape == expected_single.shape, f"Shape mismatch: {pooled_single.shape} vs {expected_single.shape}"
    
    # Check output correctness
    assert torch.allclose(pooled_single, expected_single, atol=1e-5), "Outputs do not match!"
    
    
    
    print("All tests passed for ParallelMaxPool2d!")

# Run the test
# test_ParallelMaxPool2d()
