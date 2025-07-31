
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
from torch.distributed.device_mesh import DeviceMesh
from torch.utils.data import DataLoader, DistributedSampler
from torchvision import datasets, transforms
from torch.distributed._tensor import Shard, Replicate
from torch.distributed.tensor.parallel import (
    parallelize_module,
    ColwiseParallel,
    RowwiseParallel,
)

class ParallelAlexNet(nn.Module):
    def __init__(self, device_mesh, num_classes=1000):
        super().__init__()
        self.device_mesh = device_mesh
        self.features = self._make_features()
        self.avgpool = nn.AdaptiveAvgPool2d((6, 6))
        self.classifier = self._make_classifier(num_classes)
        self._parallelize_layers()

    def _make_features(self):
        return nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=11, stride=4, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(64, 192, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(192, 384, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
        )

    def _make_classifier(self, num_classes):
        return nn.Sequential(
            nn.Dropout(),
            nn.Linear(256 * 6 * 6, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096, num_classes),
        )

    def _parallelize_layers(self):
        # Parallelize conv layers - alternating splits
        conv_counter = 0
        for module in self.features.modules():
            if isinstance(module, nn.Conv2d):
                if conv_counter % 2 == 0:
                    parallelize_module(
                        module,
                        self.device_mesh,
                        {"weight": ColwiseParallel(), "bias": ColwiseParallel()},
                    )
                else:
                    parallelize_module(
                        module,
                        self.device_mesh,
                        {"weight": RowwiseParallel(), "bias": Replicate()},
                    )
                conv_counter += 1

        # Parallelize classifier
        parallelize_module(
            self.classifier[1],  # Linear(256*6*6, 4096)
            self.device_mesh,
            {"weight": RowwiseParallel(), "bias": ColwiseParallel()},
        )
        parallelize_module(
            self.classifier[4],  # Linear(4096, 4096)
            self.device_mesh,
            {"weight": RowwiseParallel(), "bias": ColwiseParallel()},
        )
        parallelize_module(
            self.classifier[6],  # Linear(4096, num_classes)
            self.device_mesh,
            {"weight": ColwiseParallel(), "bias": Replicate()},
        )

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x


class ParallelDenseNet(nn.Module):
    def __init__(self, device_mesh, growth_rate=32, block_config=(6, 12, 24, 16),
                 num_init_features=64, bn_size=4, drop_rate=0):
        super().__init__()
        self.device_mesh = device_mesh
        
        # First convolution
        self.features = nn.Sequential(
            nn.Conv2d(3, num_init_features, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(num_init_features),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
        )
        
        # Each denseblock
        num_features = num_init_features
        for i, num_layers in enumerate(block_config):
            block = _DenseBlock(
                num_layers=num_layers,
                num_input_features=num_features,
                bn_size=bn_size,
                growth_rate=growth_rate,
                drop_rate=drop_rate,
                device_mesh=device_mesh
            )
            self.features.add_module('denseblock%d' % (i + 1), block)
            num_features = num_features + num_layers * growth_rate
            
            if i != len(block_config) - 1:
                trans = _Transition(num_input_features=num_features,
                                    num_output_features=num_features // 2,
                                    device_mesh=device_mesh)
                self.features.add_module('transition%d' % (i + 1), trans)
                num_features = num_features // 2
        
        # Final batch norm
        self.features.add_module('norm5', nn.BatchNorm2d(num_features))
        
        # Linear layer
        self.classifier = nn.Linear(num_features, 10)
        
        # Parallelize
        self._parallelize_layers()

    def _parallelize_layers(self):
        # First convolution
        parallelize_module(
            self.features[0],
            self.device_mesh,
            {"weight": ColwiseParallel()},
        )
        
        # Classifier
        parallelize_module(
            self.classifier,
            self.device_mesh,
            {"weight": ColwiseParallel(), "bias": Replicate()},
        )

    def forward(self, x):
        features = self.features(x)
        out = F.relu(features, inplace=True)
        out = F.adaptive_avg_pool2d(out, (1, 1))
        out = torch.flatten(out, 1)
        out = self.classifier(out)
        return out

class _DenseLayer(nn.Module):
    def __init__(self, num_input_features, growth_rate, bn_size, drop_rate, device_mesh):
        super().__init__()
        self.device_mesh = device_mesh
        self.norm1 = nn.BatchNorm2d(num_input_features)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(num_input_features, bn_size * growth_rate,
                              kernel_size=1, stride=1, bias=False)
        self.norm2 = nn.BatchNorm2d(bn_size * growth_rate)
        self.relu2 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(bn_size * growth_rate, growth_rate,
                              kernel_size=3, stride=1, padding=1, bias=False)
        self.drop_rate = float(drop_rate)
        self._parallelize()

    def _parallelize(self):
        parallelize_module(
            self.conv1,
            self.device_mesh,
            {"weight": ColwiseParallel()},
        )
        parallelize_module(
            self.conv2,
            self.device_mesh,
            {"weight": ColwiseParallel()},
        )

    def forward(self, x):
        out = self.conv1(self.relu1(self.norm1(x)))
        out = self.conv2(self.relu2(self.norm2(out)))
        if self.drop_rate > 0:
            out = F.dropout(out, p=self.drop_rate, training=self.training)
        return torch.cat([x, out], 1)

class _DenseBlock(nn.Module):
    def __init__(self, num_layers, num_input_features, bn_size, growth_rate, drop_rate, device_mesh):
        super().__init__()
        self.device_mesh = device_mesh
        layers = []
        for i in range(num_layers):
            layer = _DenseLayer(
                num_input_features + i * growth_rate,
                growth_rate=growth_rate,
                bn_size=bn_size,
                drop_rate=drop_rate,
                device_mesh=device_mesh
            )
            layers.append(layer)
        self.layers = nn.ModuleList(layers)

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

class _Transition(nn.Module):
    def __init__(self, num_input_features, num_output_features, device_mesh):
        super().__init__()
        self.device_mesh = device_mesh
        self.norm = nn.BatchNorm2d(num_input_features)
        self.relu = nn.ReLU(inplace=True)
        self.conv = nn.Conv2d(num_input_features, num_output_features,
                             kernel_size=1, stride=1, bias=False)
        self.pool = nn.AvgPool2d(kernel_size=2, stride=2)
        self._parallelize()

    def _parallelize(self):
        parallelize_module(
            self.conv,
            self.device_mesh,
            {"weight": ColwiseParallel()},
        )

    def forward(self, x):
        out = self.conv(self.relu(self.norm(x)))
        out = self.pool(out)
        return out

class Bottleneck(nn.Module):
    expansion = 4  # Output channels = planes * expansion

    def __init__(self, inplanes, planes, stride=1, downsample=None, device_mesh=None):
        super().__init__()
        self.device_mesh = device_mesh
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, stride=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * self.expansion, kernel_size=1, stride=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride
        self._parallelize()

    def _parallelize(self):
        # Parallelize all conv layers with ColwiseParallel (shard on input channels)
        parallelize_module(
            self.conv1,
            self.device_mesh,
            {"weight": ColwiseParallel()},
        )
        parallelize_module(
            self.conv2,
            self.device_mesh,
            {"weight": ColwiseParallel()},
        )
        parallelize_module(
            self.conv3,
            self.device_mesh,
            {"weight": ColwiseParallel()},
        )
        if self.downsample is not None:
            parallelize_module(
                self.downsample[0],  # Conv layer in downsample path
                self.device_mesh,
                {"weight": ColwiseParallel()},
            )

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class ParallelResNet(nn.Module):
    def __init__(self, device_mesh, block, layers, num_classes=1000):
        super().__init__()
        self.device_mesh = device_mesh
        self.inplanes = 64  # Initial number of channels
        
        # Initial layers
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        
        # Residual layers
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        
        # Final layers
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block.expansion, num_classes)
        
        # Apply Tensor Parallelism
        self._parallelize_layers()

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                         kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, self.device_mesh))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, device_mesh=self.device_mesh))

        return nn.Sequential(*layers)

    def _parallelize_layers(self):
        # Parallelize initial conv layer (shard on input channels)
        parallelize_module(
            self.conv1,
            self.device_mesh,
            {"weight": ColwiseParallel()},
        )
        
        # Parallelize final fc layer (shard on output features)
        parallelize_module(
            self.fc,
            self.device_mesh,
            {"weight": RowwiseParallel(), "bias": Replicate()},
        )

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)

        return x


class ParallelResNet50(ParallelResNet):
    def __init__(self, device_mesh, num_classes=1000):
        super().__init__(
            device_mesh=device_mesh,
            block=Bottleneck,
            layers=[3, 4, 6, 3],  # ResNet-50 layer configuration
            num_classes=num_classes,
        )


class ParallelVGG19(nn.Module):
    def __init__(self, device_mesh):
        super().__init__()
        self.device_mesh = device_mesh
        self.features = self._make_layers()
        self.classifier = self._make_classifier()
        self._parallelize_layers()

    def _make_layers(self):
        layers = []
        in_channels = 3
        cfg = [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 
               512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M']
        
        conv_counter = 0  # Track Conv2d layers for alternating splits
        for v in cfg:
            if v == 'M':
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            else:
                layers += [
                    nn.Conv2d(in_channels, v, kernel_size=3, padding=1),
                    nn.SyncBatchNorm(v),  # Critical for multi-GPU
                    nn.ReLU(inplace=True),
                ]
                in_channels = v
                conv_counter += 1
        
        layers += [nn.AdaptiveAvgPool2d((1, 1))]
        return nn.Sequential(*layers)

    def _make_classifier(self):
        return nn.Sequential(
            nn.Flatten(),
            nn.Dropout(),
            nn.Linear(512, 512),  # Will be RowwiseParallel
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(512, 10),   # Will be ColwiseParallel
        )

    def _parallelize_layers(self):
        conv_counter = 0
        for module in self.features.modules():
            if isinstance(module, nn.Conv2d):
                if conv_counter % 2 == 0:
                    # Even layers: Split output channels (Colwise)
                    parallelize_module(
                        module,
                        self.device_mesh,
                        {"weight": ColwiseParallel(), "bias": ColwiseParallel()},
                    )
                else:
                    # Odd layers: Split input channels (Rowwise)
                    parallelize_module(
                        module,
                        self.device_mesh,
                        {"weight": RowwiseParallel(), "bias": Replicate()},
                    )
                conv_counter += 1

        # Megatron-style Linear splits (Rowwise -> Colwise)
        parallelize_module(
            self.classifier[2],  # Linear(512, 512)
            self.device_mesh,
            {"weight": RowwiseParallel(), "bias": ColwiseParallel()},
        )
        parallelize_module(
            self.classifier[5],  # Linear(512, 10)
            self.device_mesh,
            {"weight": ColwiseParallel(), "bias": Replicate()},
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x

class ParallelVGG16(nn.Module):
    def __init__(self, device_mesh):
        super().__init__()
        self.device_mesh = device_mesh
        self.features = self._make_layers()
        self.classifier = self._make_classifier()
        self._parallelize_layers()

    def _make_layers(self):
        layers = []
        in_channels = 3
        cfg = [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 
               512, 512, 512, 'M', 512, 512, 512, 'M']
        
        conv_counter = 0
        for v in cfg:
            if v == 'M':
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            else:
                layers += [
                    nn.Conv2d(in_channels, v, kernel_size=3, padding=1),
                    nn.SyncBatchNorm(v),
                    nn.ReLU(inplace=True),
                ]
                in_channels = v
                conv_counter += 1
        
        layers += [nn.AdaptiveAvgPool2d((1, 1))]
        return nn.Sequential(*layers)

    def _make_classifier(self):
        return nn.Sequential(
            nn.Flatten(),
            nn.Dropout(),
            nn.Linear(512, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(512, 10),
        )

    def _parallelize_layers(self):
        conv_counter = 0
        for module in self.features.modules():
            if isinstance(module, nn.Conv2d):
                if conv_counter % 2 == 0:
                    parallelize_module(
                        module,
                        self.device_mesh,
                        {"weight": ColwiseParallel(), "bias": ColwiseParallel()},
                    )
                else:
                    parallelize_module(
                        module,
                        self.device_mesh,
                        {"weight": RowwiseParallel(), "bias": Replicate()},
                    )
                conv_counter += 1

        parallelize_module(
            self.classifier[2],
            self.device_mesh,
            {"weight": RowwiseParallel(), "bias": ColwiseParallel()},
        )
        parallelize_module(
            self.classifier[5],
            self.device_mesh,
            {"weight": ColwiseParallel(), "bias": Replicate()},
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x


# class ParallelVGG19(nn.Module):
#     def __init__(self, device_mesh):
#         super().__init__()
#         self.device_mesh = device_mesh
#         self.features = self._make_layers()
#         self.classifier = self._make_classifier()

#         # Apply tensor parallelism
#         self._parallelize_layers()

#     def _make_layers(self):
#         layers = []
#         in_channels = 3
#         cfg = [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 
#                512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M']
#         for v in cfg:
#             if v == 'M':
#                 layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
#             else:
#                 layers += [
#                     nn.Conv2d(in_channels, v, kernel_size=3, padding=1),
#                     nn.BatchNorm2d(v),
#                     nn.ReLU(inplace=True)
#                 ]
#                 in_channels = v

#         # Add AdaptiveAvgPool to handle CIFAR10 spatial dimensions
#         layers += [nn.AdaptiveAvgPool2d((1, 1))]
#         return nn.Sequential(*layers)

#     def _make_classifier(self):
#         return nn.Sequential(
#             nn.Flatten(),
#             nn.Dropout(),
#             nn.Linear(512, 512),
#             nn.ReLU(inplace=True),
#             nn.Dropout(),
#             nn.Linear(512, 10)  # CIFAR-10 (10 classes)
#         )

#     def _parallelize_layers(self):
#         # Parallelize conv layers (shard output channels)
#         for module in self.features.modules():
#             if isinstance(module, nn.Conv2d):
#                 parallelize_module(module, self.device_mesh,
#                                    {'weight': ColwiseParallel(),
#                                     'bias': ColwiseParallel()})

#         # Parallelize classifier linear layers
#         for module in self.classifier.modules():
#             if isinstance(module, nn.Linear):
#                 parallelize_module(module, self.device_mesh,
#                                    {'weight': ColwiseParallel(),
#                                     'bias': ColwiseParallel()})

#     def forward(self, x):
#         x = self.features(x)
#         x = self.classifier(x)
#         return x