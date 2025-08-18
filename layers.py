import torch
import torch.nn as nn
from .maths import sum_random_nums_n, moe_masked


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
    def __init__(self, conv_layer: nn.Conv2d, num_splits=4, combine=True, split_channels=None, random_split=False):
        super(OutputChannelSplitConv2d, self).__init__()
        # assert conv_layer.out_channels % num_splits == 0, "Output channels must be divisible by num_splits"

        self.weight = conv_layer.weight
        self.bias = conv_layer.bias
        self.device = conv_layer.weight.device  # Ensure the same device
        self.in_channels = conv_layer.in_channels
        self.out_channels = conv_layer.out_channels
        self.kernel_size = conv_layer.kernel_size
        self.stride = conv_layer.stride
        self.padding = conv_layer.padding
        self.rem = self.out_channels % num_splits
        if split_channels:
            self.split_channels = split_channels
        else:
            self.split_channels = [self.out_channels // num_splits + (1 if i < self.rem else 0) for i in range(num_splits) ]
        self.combine = combine
        self.random_split = random_split

        self.split_layers = nn.ModuleList([
            nn.Conv2d(self.in_channels,self.split_channels[i] , kernel_size=self.kernel_size,
                      stride=self.stride, padding=self.padding).to(self.device)  # Move to the same device
            for i in range(num_splits)
        ])

        self.copy_weights_from()

    def forward(self, x):
        if isinstance(x, list):
            if isinstance(x[0], list):
                # self.change_split_channels(split_channels=x[1], num_splits=len(x[1]))
                split_outputs = [layer(x1) for x1,layer in zip(x[0], self.split_layers)]
            else:
                split_outputs = [layer(x1) for x1,layer in zip(x, self.split_layers)]
        else:
            split_outputs = [layer(x) for layer in self.split_layers]
            # for layer in self.split_layers:
            #     print(layer.weight.shape)

        split_sizes = self.split_channels
        if self.random_split:
            split_outputs = torch.cat(split_outputs, dim=1)
            split_sizes = sum_random_nums_n(split_outputs.shape[1])
            split_outputs = torch.split(split_outputs, split_sizes, dim=1)

        return torch.cat(split_outputs, dim=1) if self.combine else [split_outputs, torch.tensor(split_sizes)]
    
    def change_split_channels(self, split_channels, num_splits=None):
        if num_splits:
            split_channels = [self.out_channels // num_splits + (1 if i < self.rem else 0) for i in range(num_splits) ]
        assert sum(split_channels) == self.out_channels, \
            "Sum of new split_channels must equal total out_channels"
        

        self.split_channels = split_channels

        self.split_layers = nn.ModuleList([
            nn.Conv2d(self.in_channels,self.split_channels[i] , kernel_size=self.kernel_size,
                      stride=self.stride, padding=self.padding).to(self.device)  # Move to the same device
            for i in range(len(self.split_channels))
        ])

        self.copy_weights_from()


    def copy_weights_from(self, split_channels=None):
        """Copies weights and biases from an original Conv2d layer"""
        if split_channels:
            self.split_channels = split_channels
        with torch.no_grad():
            start_idx = 0
            end_idx =  0
            for i in range(len(self.split_channels)):
                end_idx +=  self.split_channels[i]
                self.split_layers[i].weight.copy_(self.weight[start_idx:end_idx])
                self.split_layers[i].bias.copy_(self.bias[start_idx:end_idx]) 
                start_idx = end_idx 
                

class InputChannelSplitConv2d(nn.Module):
    def __init__(self, conv_layer: nn.Conv2d, num_splits=4, combine=True, split_channels=None, skipconnections = False):
        super(InputChannelSplitConv2d, self).__init__()
        # assert conv_layer.in_channels % num_splits == 0, "Input channels must be divisible by num_splits"

        self.weight = conv_layer.weight
        self.bias = conv_layer.bias
        self.device = conv_layer.weight.device
        self.num_splits = num_splits
        self.in_channels = conv_layer.in_channels
        self.out_channels = conv_layer.out_channels
        self.kernel_size = conv_layer.kernel_size
        self.stride = conv_layer.stride
        self.padding = conv_layer.padding
        self.rem = self.in_channels % num_splits
        if split_channels:
            self.split_channels = split_channels
        else:
            self.split_channels = [self.in_channels // num_splits + (1 if i < self.rem else 0) for i in range(num_splits) ]
        self.combine = combine
        self.skipconnections = skipconnections
        
        self.split_layers = nn.ModuleList([
            nn.Conv2d(self.split_channels[i], self.out_channels, kernel_size=self.kernel_size,
                      stride=self.stride, padding=self.padding).to(self.device)
            for i in range(len(self.split_channels))
        ])

        # MoE Gate components
        self.gate_input_dim = self.in_channels  # Input features for the gate
        
        self.gate = nn.Sequential(
            nn.Linear(self.gate_input_dim, self.num_splits - 1),
            nn.Sigmoid()  # Output probabilities for each potential merge point
        )

        self.copy_weights_from()

    def forward(self, x):
        
        if isinstance(x[0], list):
            # print("check-point2",x[1])
            split_sizes = x[1].tolist()
            self.change_split_channels(split_channels=split_sizes)
            split_inputs = x[0] if  isinstance(x[0], list) else torch.split(x[0], split_sizes, dim=1)  # Corrected to ensure proper channel allocation
            split_outputs = [layer(split_inputs[i]) for i, layer in enumerate(self.split_layers)]
        else:
            split_sizes = self.split_channels 
            split_inputs = x if  isinstance(x, list) else torch.split(x, split_sizes, dim=1)  # Corrected to ensure proper channel allocation
            split_outputs = [layer(split_inputs[i]) for i, layer in enumerate(self.split_layers)]
        # self.mask = torch.randint(0, 2, (split_outputs[0].shape[0], len(split_outputs)-1)).bool() 
        print([s.shape for s in split_inputs])
        gate_input = torch.mean(x[0] if isinstance(x[0], list) else x, dim=[2, 3])  # [B, C]
        gate_probs = self.gate(gate_input)  # [B, num_splits-1]
        
        # Generate mask from gate probabilities
        # self.mask = gate_probs > 0.5  # Threshold at 0.5
        # if self.mask:
        #     split_outputs =  moe_masked(split_outputs)

        if self.skipconnections:
            split_outputs = [sum(split_outputs)+split_output for split_output in split_outputs]
            return [split_outputs,torch.tensor(split_sizes)]
            
        return sum(split_outputs) if self.combine else [split_outputs,torch.tensor(split_sizes)]   # Element-wise sum of outputs

    def change_split_channels(self, split_channels, num_splits=None):
        assert sum(split_channels) == self.in_channels, \
            "Sum of new split_channels must equal total in_channels"

        self.split_channels = split_channels

        self.split_layers = nn.ModuleList([
            nn.Conv2d(self.split_channels[i], self.out_channels,
                      kernel_size=self.kernel_size, stride=self.stride,
                      padding=self.padding).to(self.device)
            for i in range(len(self.split_channels))
        ])

        self.copy_weights_from()
    
    def copy_weights_from(self):
        with torch.no_grad():
            rem = self.rem
            start_idx = 0
            end_idx =  0
            for i in range(len(self.split_channels)):
                end_idx +=  self.split_channels[i]
                # print(i, start_idx, end_idx, self.split_channels,self.weight.shape)
                self.split_layers[i].weight.copy_(self.weight[:, start_idx:end_idx, :, :])
                self.split_layers[i].bias.copy_(self.bias / len(self.split_channels))
                start_idx = end_idx 

class OutputChannelSplitConv1D(nn.Module):
    def __init__(self, conv_layer: nn.Conv1d, num_splits=4, combine=True, split_channels=None):
        super(OutputChannelSplitConv1D, self).__init__()
        self.device = conv_layer.weight.device
        self.num_splits = num_splits
        self.combine = combine

        self.in_channels = conv_layer.in_channels
        self.out_channels = conv_layer.out_channels
        self.kernel_size = conv_layer.kernel_size
        self.stride = conv_layer.stride
        self.padding = conv_layer.padding
        self.dilation = conv_layer.dilation
        self.groups = conv_layer.groups
        self.bias = conv_layer.bias is not None
        self.rem = self.out_channels % num_splits

        if split_channels:
            self.split_sizes = split_channels
        else:
            self.split_sizes = [self.out_channels // num_splits + (1 if i < self.rem else 0) for i in range(num_splits)]

        # Create split Conv1d layers
        self.split_layers = nn.ModuleList([
            nn.Conv1d(self.in_channels, self.split_sizes[i],
                      kernel_size=self.kernel_size, stride=self.stride,
                      padding=self.padding, dilation=self.dilation,
                      groups=self.groups, bias=self.bias).to(self.device)
            for i in range(len(self.split_sizes))
        ])

        self.copy_weights_from(conv_layer)

    def forward(self, x):
        outputs = [layer(x) for layer in self.split_layers]
        return torch.cat(outputs, dim=1) if self.combine else outputs

    def copy_weights_from(self, conv_layer):
        with torch.no_grad():
            start = 0
            for i, split_layer in enumerate(self.split_layers):
                end = start + self.split_sizes[i]
                split_layer.weight.copy_(conv_layer.weight[start:end])
                if self.bias:
                    split_layer.bias.copy_(conv_layer.bias[start:end])
                start = end

class InputChannelSplitConv1D(nn.Module):
    def __init__(self, conv_layer: nn.Conv1d, num_splits=4, combine=True, split_channels=None, skipconnections=False):
        super(InputChannelSplitConv1D, self).__init__()
        self.device = conv_layer.weight.device
        self.num_splits = num_splits
        self.combine = combine
        self.skipconnections = skipconnections

        self.in_channels = conv_layer.in_channels
        self.out_channels = conv_layer.out_channels
        self.kernel_size = conv_layer.kernel_size
        self.stride = conv_layer.stride
        self.padding = conv_layer.padding
        self.dilation = conv_layer.dilation
        self.groups = conv_layer.groups
        self.bias = conv_layer.bias is not None
        self.rem = self.in_channels % num_splits

        if split_channels:
            self.split_sizes = split_channels
        else:
            self.split_sizes = [self.in_channels // num_splits + (1 if i < self.rem else 0) for i in range(num_splits)]

        # Create split Conv1d layers
        self.split_layers = nn.ModuleList([
            nn.Conv1d(self.split_sizes[i], self.out_channels,
                      kernel_size=self.kernel_size, stride=self.stride,
                      padding=self.padding, dilation=self.dilation,
                      groups=1, bias=self.bias).to(self.device)
            for i in range(len(self.split_sizes))
        ])

        self.copy_weights_from(conv_layer)

    def forward(self, x):
        # x: (batch, channels, width)
        split_inputs = torch.split(x, self.split_sizes, dim=1)
        outputs = [layer(split_inputs[i]) for i, layer in enumerate(self.split_layers)]
        if self.skipconnections:
            outputs = [sum(outputs) + o for o in outputs]
        return sum(outputs) if self.combine else outputs

    def copy_weights_from(self, conv_layer):
        with torch.no_grad():
            start = 0
            for i, split_layer in enumerate(self.split_layers):
                end = start + self.split_sizes[i]
                split_layer.weight.copy_(conv_layer.weight[:, start:end])
                if self.bias:
                    split_layer.bias.copy_(conv_layer.bias / len(self.split_layers))
                start = end

                


class OutputChannelSplitLinear(nn.Module):
    def __init__(self, linear_layer: nn.Linear, num_splits=4, combine = True, split_channels=None, random_split=False):
        super(OutputChannelSplitLinear, self).__init__()
        # assert linear_layer.out_features % num_splits == 0, "Output features must be divisible by num_splits"

        self.weight = linear_layer.weight
        self.bias = linear_layer.bias
        self.device = linear_layer.weight.device
        self.num_splits = num_splits
        self.in_features = linear_layer.in_features
        self.out_features = linear_layer.out_features
        self.rem = self.out_features % num_splits
        if split_channels:
            self.split_sizes = split_channels
        else:
            self.split_sizes = [self.out_features // num_splits + (1 if i < self.rem else 0) for i in range(num_splits)]
        self.combine = combine
        self.random_split = random_split

        # Create multiple smaller linear layers
        self.split_layers = nn.ModuleList([
            nn.Linear(self.in_features, self.split_sizes[i]).to(self.device)
            for i in range(len(self.split_sizes))
        ])

        self.copy_weights_from()

    def forward(self, x):
        if isinstance(x, list):
            if isinstance(x[0], list):
                # self.change_split_channels(x[1].tolist(), num_splits=len(x[1].tolist()))
                split_outputs = [layer(x1) for x1,layer in zip(x[0], self.split_layers)]
            else:
                split_outputs = [layer(x1) for x1,layer in zip(x, self.split_layers)]
        else:
            split_outputs = [layer(x) for layer in self.split_layers]
        split_sizes = self.split_sizes
        if self.random_split:
            split_outputs = torch.cat(split_outputs, dim=1)
            split_sizes = sum_random_nums_n(split_outputs.shape[1])
            split_outputs = torch.split(split_outputs, split_sizes, dim=1)

        return torch.cat(split_outputs, dim=1)  if self.combine else [split_outputs, torch.tensor(split_sizes)] # Concatenate outputs

    def change_split_channels(self, split_sizes, num_splits=None):
        if num_splits:
            split_sizes = [self.out_features // num_splits + (1 if i < self.rem else 0) for i in range(num_splits)]
        assert sum(split_sizes) == self.out_features, \
            "Sum of new split_channels must equal total out_features"

        self.split_sizes = split_sizes
        self.split_layers = nn.ModuleList([
            nn.Linear(self.in_features, self.split_sizes[i]).to(self.device)
            for i in range(len(self.split_sizes))
        ])
        self.copy_weights_from()

    def copy_weights_from(self):
        """Copies weights and biases from the original linear layer"""
        
        with torch.no_grad():
            start_idx = 0
            end_idx =  0
            for i in range(len(self.split_sizes)):
                end_idx +=  self.split_sizes[i]
                self.split_layers[i].weight.copy_(self.weight[start_idx:end_idx])
                self.split_layers[i].bias.copy_(self.bias[start_idx:end_idx]) 
                start_idx = end_idx 
                
                
class InputChannelSplitLinear(nn.Module):
    def __init__(self, linear_layer: nn.Linear, num_splits=4, combine=True, split_channels=None, structs=None, skipconnections=False):
        super(InputChannelSplitLinear, self).__init__()
        # assert linear_layer.in_features % num_splits == 0, "Input features must be divisible by num_splits"

        self.weight = linear_layer.weight
        self.bias = linear_layer.bias
        self.device = linear_layer.weight.device
        self.num_splits = num_splits
        self.in_features = linear_layer.in_features
        self.out_features = linear_layer.out_features
        self.rem = self.in_features % num_splits
        if split_channels:
            self.split_sizes = split_channels
        else:
            self.split_sizes = [self.in_features // num_splits + (1 if i < self.rem else 0) for i in range(num_splits)]
        if structs:
            self.structs = structs
        else:
            self.structs = [True for i in range(len(self.split_sizes)-1)]

        self.combine = combine
        self.skipconnections =skipconnections

        # Create multiple smaller linear layers
        self.split_layers = nn.ModuleList([
            nn.Linear(self.split_sizes[i], self.out_features).to(self.device)
            for i in range(len(self.split_sizes))
        ])

        # print(self.split_layers)

        self.copy_weights_from()

    def forward(self, x):
        if isinstance(x[0], list):
            # print("check-point2",x[1])
            split_sizes = x[1].tolist()
            self.change_split_channels(split_channels=split_sizes)
            split_inputs = x[0] if type(x[0]) is list else torch.split(x[0], self.split_sizes, dim=1)  # Corrected to ensure proper channel allocation
            split_outputs = [layer(split_inputs[i]) for i, layer in enumerate(self.split_layers)]
        else:
            split_sizes = self.split_sizes
            split_inputs = x if type(x) is list else torch.split(x, self.split_sizes, dim=1)  # Corrected to ensure proper channel allocation
            split_outputs = [layer(split_inputs[i]) for i, layer in enumerate(self.split_layers)]
        if self.skipconnections:
            split_outputs = [sum(split_outputs)+split_output for split_output in split_outputs]
            # return [split_outputs,torch.tensor(split_sizes)]

        if self.combine: 
            return sum(split_outputs)
        
        else:
            tem_outputs = [split_outputs[0]]
            for i in range(len(self.structs)):
                if self.structs[i]:
                    tem_outputs[-1] += split_outputs[i+1]
                else:
                    tem_outputs = tem_outputs.append(split_outputs[i])
            split_outputs = tem_outputs

            return [split_outputs, torch.tensor(self.structs)] # Element-wise sum if combining

    def change_split_channels(self, split_channels):
        assert sum(split_channels) == self.in_features, \
            "Sum of new split_channels must equal total in_features"

        self.split_sizes = split_channels
        self.split_layers = nn.ModuleList([
            nn.Linear(self.split_sizes[i], self.out_features).to(self.device)
            for i in range(len(self.split_sizes))
        ])
        self.copy_weights_from()

    def copy_weights_from(self):
        """Copies weights and biases from the original linear layer"""
        

        with torch.no_grad():
            start_idx = 0
            end_idx =  0
            for i in range(len(self.split_layers)):
                end_idx +=  self.split_sizes[i]
                # print(i, start_idx, end_idx, self.split_layers[i],self.split_layers[i].weight.shape ,self.weight.shape)
                self.split_layers[i].weight.copy_(self.weight[:, start_idx:end_idx])
                self.split_layers[i].bias.copy_(self.bias / len(self.split_layers))
                start_idx = end_idx 


class ParallelActivations(nn.Module):
    def __init__(self, activation =nn.ReLU(),  combine=False):
        super(ParallelActivations, self).__init__()
        self.activation = activation
        self.combine = combine

    def forward(self, x):
       
        if isinstance(x, list):
            # print("checkpoint-1",self.activation,x[1])
            if isinstance(x[0], list):
                # print("---------------")
                # print(self.activation, type(x[0][0]))
                activation_outputs = [self.activation(tensor) for tensor in x[0]]
                # print("--------end------------")
                return torch.cat(activation_outputs, dim=1) if self.combine else [activation_outputs,x[1]]
                
            else:
                print("---------check-point-1----------")
                activation_outputs = [self.activation(tensor) for tensor in x]
            return torch.cat(activation_outputs, dim=1) if self.combine else activation_outputs
        elif isinstance(x, torch.Tensor):
            return self.activation(x)
        else:
            raise TypeError("Input must be a Tensor or a list of Tensors")
        
class ParallelDropout(nn.Module):
    def __init__(self, layer=None, combine=False):
        super(ParallelDropout, self).__init__()
        self.dropout = layer if layer is not None else nn.Dropout(0.5)
        self.combine = combine

    def forward(self, x):
        if isinstance(x, list):
            dropout_outputs = [self.dropout(tensor) for tensor in x]
            return torch.cat(dropout_outputs, dim=1) if self.combine else dropout_outputs
        elif isinstance(x, torch.Tensor):
            return self.dropout(x)
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
            # print(self.pool(x).shape)
            return self.pool(x)
        else:
            raise TypeError("Input must be a Tensor or a list of Tensors")
        
