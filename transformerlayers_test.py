import torch
import torch.nn as nn
from transformerlayers import OutputChannelSplitConv1DGPT,InputChannelSplitConv1DGPT, ParallelGPT2MLP
from transformers.pytorch_utils import Conv1D
from transformers import GPT2Model, GPT2Config, GPT2Tokenizer

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

class TestParallelGPT2MLP:
    @classmethod
    def setup_class(cls):
        # Load a pretrained GPT-2 model
        cls.model = GPT2Model.from_pretrained('gpt2')
        cls.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        cls.model.to(cls.device)
        
        # Get the first MLP layer from the model
        cls.original_mlp = cls.model.h[0].mlp
        cls.hidden_size = cls.original_mlp.c_fc.weight.shape[0]
        
    def test_output_equivalence(self):
        """Test that ParallelGPT2MLP produces same output as original MLP"""
        for num_splits in [2, 4, 8]:
            # Create parallel version
            parallel_mlp = ParallelGPT2MLP(
                self.original_mlp,
                num_splits=num_splits,
                combine=True
            ).to(self.device)
            
            # Create test input
            batch_size = 4
            seq_length = 16
            x = torch.randn(batch_size, seq_length, self.hidden_size).to(self.device)
            x.requires_grad = True
            
            # Forward pass
            original_out = self.original_mlp(x)
            parallel_out = parallel_mlp([x for _ in range(num_splits)])
            
            # Check output shape and values
            assert original_out.shape == parallel_out.shape
            assert torch.allclose(original_out, parallel_out, atol=1e-5), \
                f"Output mismatch with {num_splits} splits"
            
            # Backward pass check
            target = torch.randn_like(original_out)
            loss_fn = nn.MSELoss()
            
            # Original backward
            original_loss = loss_fn(original_out, target)
            original_loss.backward()
            original_grad = x.grad.clone()
            x.grad = None
            
            # Parallel backward
            parallel_loss = loss_fn(parallel_out, target)
            parallel_loss.backward()
            parallel_grad = x.grad.clone()
            x.grad = None
            
            # Check gradients
            assert torch.allclose(original_grad, parallel_grad, atol=1e-5), \
                f"Gradient mismatch with {num_splits} splits"
            
            print(f"[✓] Passed equivalence test with {num_splits} splits")

    def test_uncombined_output(self):
        """Test that uncombined output works correctly"""
        num_splits = 4
        parallel_mlp = ParallelGPT2MLP(
            self.original_mlp,
            num_splits=num_splits,
            combine=False
        ).to(self.device)
        
        x = torch.randn(2, 8, self.hidden_size).to(self.device)
        outputs = parallel_mlp(x)
        
        # Should return list of outputs
        assert isinstance(outputs, list)
        assert len(outputs) == num_splits
        
        # Combined output should match original
        original_out = self.original_mlp(x)
        combined_out = sum(outputs)  # For linear layers, sum is correct combination
        assert torch.allclose(original_out, combined_out, atol=1e-5)

    def test_custom_split_channels(self):
        """Test with custom split channel sizes"""
        split_channels = [256, 256, 512]  # Must sum to hidden_size
        num_splits = len(split_channels)
        
        parallel_mlp = ParallelGPT2MLP(
            self.original_mlp,
            num_splits=num_splits,
            split_channels=split_channels,
            combine=True
        ).to(self.device)
        
        x = torch.randn(3, 10, self.hidden_size).to(self.device)
        original_out = self.original_mlp(x)
        parallel_out = parallel_mlp(x)
        
        assert original_out.shape == parallel_out.shape
        assert torch.allclose(original_out, parallel_out, atol=1e-5)
        
        print("[✓] Passed custom split channels test")

    
