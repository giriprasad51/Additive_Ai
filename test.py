import torch
import torch.nn as nn
from .layers import OutputChannelSplitConv2d, InputChannelSplitConv2d

def test_OutputChannelSplitConv2d():
   
    
    # Create original Conv2d layer
    original_conv = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, stride=1, padding=1)
    
    # Create OutputChannelSplitConv2d layer
    parallel_conv = OutputChannelSplitConv2d(original_conv)
    
    # Copy weights from original_conv to parallel_conv
    parallel_conv.copy_weights_from(original_conv)
    
    # Create a dummy input tensor
    x = torch.randn(1, 3, 32, 32)
    
    # Get outputs from both layers
    original_output = original_conv(x)
    parallel_output = parallel_conv(x)
    
    # Check shape consistency
    assert original_output.shape == parallel_output.shape, f"Shape mismatch: {original_output.shape} vs {parallel_output.shape}"
    
    # Check output similarity
    assert torch.allclose(original_output, parallel_output, atol=1e-5), "Outputs do not match!"
    
    print("Test passed: OutputChannelSplitConv2d matches standard Conv2d!")

# Run the test
test_OutputChannelSplitConv2d()

def test_InputChannelSplitConv2d():
    

    # Create original Conv2d layer
    original_conv = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1)
    # Create InputChannelSplitConv2d layer
    split_conv = InputChannelSplitConv2d(original_conv)

    # Create a dummy input tensor
    x = torch.randn(1, 64, 32, 32)

    # Get outputs from both layers
    original_output = original_conv(x)
    split_output = split_conv(x)

    # Check shape consistency
    assert original_output.shape == split_output.shape, f"Shape mismatch: {original_output.shape} vs {split_output.shape}"

    # Check number of elements mismatch
    num_mismatched = torch.sum(~torch.isclose(original_output, split_output, atol=1e-5)).item()
    assert num_mismatched == 0, f"Number of mismatched elements: {num_mismatched}"
    
    # Check output similarity
    assert torch.allclose(original_output, split_output, atol=1e-5), "Outputs do not match!"

    print("Test passed: InputChannelSplitConv2d matches standard Conv2d!")

# Run the test
test_InputChannelSplitConv2d()

def test_OutputChannelSplitLinear():
    # Create original Linear layer
    original_linear = nn.Linear(in_features=16, out_features=64)
    
    # Create OutputChannelSplitLinear layer
    parallel_linear = OutputChannelSplitLinear(original_linear)
    
    # Copy weights from original to parallel
    parallel_linear.copy_weights_from(original_linear)
    
    # Create a dummy input tensor
    x = torch.randn(1, 16)
    
    # Get outputs from both layers
    original_output = original_linear(x)
    parallel_output = parallel_linear(x)
    
    # Check shape consistency
    assert original_output.shape == parallel_output.shape, f"Shape mismatch: {original_output.shape} vs {parallel_output.shape}"
    
    # Check output similarity
    assert torch.allclose(original_output, parallel_output, atol=1e-5), "Outputs do not match!"
    
    print("Test passed: OutputChannelSplitLinear matches standard Linear!")

# Run the test
test_OutputChannelSplitLinear()

def test_InputChannelSplitLinear():
    # Create original Linear layer
    original_linear = nn.Linear(in_features=64, out_features=64)
    
    # Create InputChannelSplitLinear layer
    split_linear = InputChannelSplitLinear(original_linear)
    
    # Create a dummy input tensor
    x = torch.randn(1, 64)
    
    # Get outputs from both layers
    original_output = original_linear(x)
    split_output = split_linear(x)
    
    # Check shape consistency
    assert original_output.shape == split_output.shape, f"Shape mismatch: {original_output.shape} vs {split_output.shape}"
    
    # Check number of elements mismatch
    num_mismatched = torch.sum(~torch.isclose(original_output, split_output, atol=1e-5)).item()
    assert num_mismatched == 0, f"Number of mismatched elements: {num_mismatched}"
    
    # Check output similarity
    assert torch.allclose(original_output, split_output, atol=1e-5), "Outputs do not match!"
    
    print("Test passed: InputChannelSplitLinear matches standard Linear!")

# Run the test
test_InputChannelSplitLinear()
