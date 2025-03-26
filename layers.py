import torch
import torch.nn as nn


# For Conv2d

class OutputChannelSplitConv2d(nn.Module):
    """
        # Example Usage
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        conv = nn.Conv2d(in_channels=3, out_channels=8, kernel_size=3, stride=1, padding=1).to(device)
        parallel_conv = OutputChannelSplitConv2d(conv, num_splits=4).to(device)  # Ensure it's on the same device

        # Test with a dummy input
        x = torch.randn(1, 3, 32, 32).to(device)
        output = parallel_conv(x)
        print("Output shape:", output.shape)
    """
    def __init__(self, conv_layer: nn.Conv2d, num_splits=4, combine = True):
        super(OutputChannelSplitConv2d, self).__init__()
        # assert conv_layer.out_channels % num_splits == 0, "Output channels must be divisible by num_splits"

        self.device = conv_layer.weight.device  # Ensure the same device
        self.num_splits = num_splits
        self.in_channels = conv_layer.in_channels
        self.out_channels = conv_layer.out_channels
        self.kernel_size = conv_layer.kernel_size
        self.stride = conv_layer.stride
        self.padding = conv_layer.padding
        self.rem = self.out_channels % num_splits
        self.split_channels = [self.out_channels // num_splits + (1 if i < self.rem else 0) for i in range(num_splits) ]
        self.combine = combine

        self.split_layers = nn.ModuleList([
            nn.Conv2d(self.in_channels,self.split_channels[i] , kernel_size=self.kernel_size,
                      stride=self.stride, padding=self.padding).to(self.device)  # Move to the same device
            for i in range(num_splits)
        ])

        self.copy_weights_from(conv_layer)

    def forward(self, x):
        if isinstance(x, list):
            split_outputs = [layer(x1) for x1,layer in zip(x, self.split_layers)]
        else:
            split_outputs = [layer(x) for layer in self.split_layers]
            # for layer in self.split_layers:
            #     print(layer.weight.shape)
        return torch.cat(split_outputs, dim=1) if self.combine else split_outputs

    def copy_weights_from(self, original_layer: nn.Conv2d):
        """Copies weights and biases from an original Conv2d layer"""
        with torch.no_grad():
            start_idx = 0
            end_idx =  0
            for i in range(self.num_splits):
                end_idx +=  self.split_channels[i]
                self.split_layers[i].weight.copy_(original_layer.weight[start_idx:end_idx])
                self.split_layers[i].bias.copy_(original_layer.bias[start_idx:end_idx]) 
                start_idx = end_idx 
                

class InputChannelSplitConv2d(nn.Module):
    def __init__(self, conv_layer: nn.Conv2d, num_splits=4, combine = True):
        super(InputChannelSplitConv2d, self).__init__()
        # assert conv_layer.in_channels % num_splits == 0, "Input channels must be divisible by num_splits"

        self.device = conv_layer.weight.device
        self.num_splits = num_splits
        self.in_channels = conv_layer.in_channels
        self.out_channels = conv_layer.out_channels
        self.kernel_size = conv_layer.kernel_size
        self.stride = conv_layer.stride
        self.padding = conv_layer.padding
        self.num_splits = num_splits
        self.rem = self.in_channels % num_splits
        self.split_channels = [self.out_channels // num_splits + (1 if i < self.rem else 0) for i in range(num_splits) ]
        self.combine = combine
        
        self.split_layers = nn.ModuleList([
            nn.Conv2d(self.split_channels[i], self.out_channels, kernel_size=self.kernel_size,
                      stride=self.stride, padding=self.padding).to(self.device)
            for i in range(num_splits)
        ])

        self.copy_weights_from(conv_layer)

    def forward(self, x):
        split_sizes = self.split_channels 
        split_inputs = x if  isinstance(x, list) else torch.split(x, split_sizes, dim=1)  # Corrected to ensure proper channel allocation
        split_outputs = [layer(split_inputs[i]) for i, layer in enumerate(self.split_layers)]
        return sum(split_outputs) if self.combine else split_outputs  # Element-wise sum of outputs

    def copy_weights_from(self, original_layer: nn.Conv2d):
        with torch.no_grad():
            rem = self.rem
            start_idx = 0
            end_idx =  0
            for i in range(self.num_splits):
                end_idx +=  self.split_channels[i]
                self.split_layers[i].weight.copy_(original_layer.weight[:, start_idx:end_idx, :, :])
                self.split_layers[i].bias.copy_(original_layer.bias / self.num_splits)
                start_idx = end_idx 
                


class OutputChannelSplitLinear(nn.Module):
    def __init__(self, linear_layer: nn.Linear, num_splits=4, combine = True):
        super(OutputChannelSplitLinear, self).__init__()
        # assert linear_layer.out_features % num_splits == 0, "Output features must be divisible by num_splits"

        self.device = linear_layer.weight.device
        self.num_splits = num_splits
        self.in_features = linear_layer.in_features
        self.out_features = linear_layer.out_features
        self.rem = self.out_features % num_splits
        self.split_sizes = [self.out_features // num_splits + (1 if i < self.rem else 0) for i in range(num_splits)]
        self.combine = combine

        # Create multiple smaller linear layers
        self.split_layers = nn.ModuleList([
            nn.Linear(self.in_features, self.split_sizes[i]).to(self.device)
            for i in range(num_splits)
        ])

        self.copy_weights_from(linear_layer)

    def forward(self, x):
        split_outputs = [layer(x) for layer in self.split_layers]
        return torch.cat(split_outputs, dim=1)  if self.combine else split_outputs # Concatenate outputs

    def copy_weights_from(self, original_layer: nn.Linear):
        """Copies weights and biases from the original linear layer"""
        # with torch.no_grad():
        #     for i in range(self.num_splits):
        #         self.split_layers[i].weight.copy_(original_layer.weight[i * self.split_size: (i + 1) * self.split_size])
        #         self.split_layers[i].bias.copy_(original_layer.bias[i * self.split_size: (i + 1) * self.split_size])

        with torch.no_grad():
            start_idx = 0
            end_idx =  0
            for i in range(self.num_splits):
                end_idx +=  self.split_sizes[i]
                self.split_layers[i].weight.copy_(original_layer.weight[start_idx:end_idx])
                self.split_layers[i].bias.copy_(original_layer.bias[start_idx:end_idx]) 
                start_idx = end_idx 
                
                
class InputChannelSplitLinear(nn.Module):
    def __init__(self, linear_layer: nn.Linear, num_splits=4, combine=True):
        super(InputChannelSplitLinear, self).__init__()
        # assert linear_layer.in_features % num_splits == 0, "Input features must be divisible by num_splits"

        self.device = linear_layer.weight.device
        self.num_splits = num_splits
        self.in_features = linear_layer.in_features
        self.out_features = linear_layer.out_features
        self.rem = self.in_features % num_splits
        self.split_sizes = [self.in_features // num_splits + (1 if i < self.rem else 0) for i in range(num_splits)]
        self.combine = combine

        # Create multiple smaller linear layers
        self.split_layers = nn.ModuleList([
            nn.Linear(self.split_sizes[i], self.out_features).to(self.device)
            for i in range(num_splits)
        ])

        # print(self.split_layers)

        self.copy_weights_from(linear_layer)

    def forward(self, x):

        split_inputs = x if type(x) is list else torch.split(x, self.split_sizes, dim=1)  # Corrected to ensure proper channel allocation
        split_outputs = [layer(split_inputs[i]) for i, layer in enumerate(self.split_layers)]
        return sum(split_outputs) if self.combine else split_outputs  # Element-wise sum if combining

    def copy_weights_from(self, original_layer: nn.Linear):
        """Copies weights and biases from the original linear layer"""
        # with torch.no_grad():
        #     for i in range(self.num_splits):
        #         self.split_layers[i].weight.copy_(original_layer.weight[:, i * self.split_size: (i + 1) * self.split_size])
        #         self.split_layers[i].bias.copy_(original_layer.bias / self.num_splits)

        with torch.no_grad():
            start_idx = 0
            end_idx =  0
            for i in range(self.num_splits):
                end_idx +=  self.split_sizes[i]
                self.split_layers[i].weight.copy_(original_layer.weight[:, start_idx:end_idx])
                self.split_layers[i].bias.copy_(original_layer.bias / self.num_splits)
                start_idx = end_idx 


class ParallelReLU(nn.Module):
    def __init__(self, combine=False):
        super(ParallelReLU, self).__init__()
        self.relu = nn.ReLU()
        self.combine = combine

    def forward(self, x):
        if isinstance(x, list):
            relu_outputs = [self.relu(tensor) for tensor in x]
            return torch.cat(relu_outputs, dim=1) if self.combine else relu_outputs
        elif isinstance(x, torch.Tensor):
            return self.relu(x)
        else:
            raise TypeError("Input must be a Tensor or a list of Tensors")

class ParallelMaxPool2d(nn.Module):
    def __init__(self, pool_layer: nn.MaxPool2d, combine=None):
        super(ParallelMaxPool2d, self).__init__()
        self.pool = pool_layer
        self.combine = combine

    def forward(self, x):
        if isinstance(x, list):
            # print("-----------------------------")
            # print([tensor.shape for tensor in x])
            pooled_outputs = [self.pool(tensor) for tensor in x]
            # print([tensor.shape for tensor in pooled_outputs])
            # return sum(pooled_outputs)
            # return torch.cat(pooled_outputs, dim=0) if self.combine else pooled_outputs
            if self.combine:
                return torch.cat(pooled_outputs, dim=0) if self.combine == "cat" else sum(pooled_outputs)
            else:
                return pooled_outputs
        elif isinstance(x, torch.Tensor):
            print(self.pool(x).shape)
            return self.pool(x)
        else:
            raise TypeError("Input must be a Tensor or a list of Tensors")