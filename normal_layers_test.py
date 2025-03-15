import torch
import torch.nn as nn
from layers import OutputChannelSplitConv2d, InputChannelSplitConv2d, OutputChannelSplitLinear, InputChannelSplitLinear

import torch
import torch.nn as nn

def test_OutputChannelSplitConv2d():
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

# Run the test
# test_OutputChannelSplitConv2d()


def test_InputChannelSplitConv2d():
        
    for n in [2, 4, 8, 16, 32, 64]:
        # Create original Conv2d layer
        original_conv = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1)
        # Create InputChannelSplitConv2d layer
        split_conv = InputChannelSplitConv2d(original_conv, num_splits=n)

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

def test_OutputChannelSplitLinear():

    for n in [2, 4, 8, 16, 32, 64]:        
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


# Run the test
# test_OutputChannelSplitLinear()

def test_InputChannelSplitLinear():
    
    for n in [2, 4, 8, 16, 32, 64]:
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


# Run the test
# test_InputChannelSplitLinear()
