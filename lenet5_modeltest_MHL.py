import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from layers import OutputChannelSplitConv2d, InputChannelSplitConv2d, OutputChannelSplitLinear, InputChannelSplitLinear, ParallelActivations, ParallelMaxPool2d, ParallelDropout
torch.autograd.set_detect_anomaly(True)

# Device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Data transformations
transform = transforms.Compose([
    transforms.Resize((32, 32)),  # LeNet-5 expects 32x32 inputs
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])

# Load CIFAR-10 datasets
train_dataset = torchvision.datasets.CIFAR10(
    root='/scratch/pusunuru/data', 
    train=True, 
    download=False, 
    transform=transform
)
test_dataset = torchvision.datasets.CIFAR10(
    root='/scratch/pusunuru/data', 
    train=False, 
    download=False, 
    transform=transform
)

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
train_loader_MHL = DataLoader(train_dataset, batch_size=256, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

# LeNet-5 Model Definition
class LeNet5(nn.Module):
    def __init__(self, num_classes=10):
        super(LeNet5, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 6, kernel_size=5),     # CIFAR-10 has 3 channels (RGB)
            nn.Tanh(),
            nn.AvgPool2d(kernel_size=2, stride=2),
            nn.Conv2d(6, 16, kernel_size=5),
            nn.Tanh(),
            nn.AvgPool2d(kernel_size=2, stride=2),
        )
        
        self.classifier = nn.Sequential(
            nn.Linear(16 * 5 * 5, 120),         # Flattened size after conv/pool
            nn.Tanh(),
            nn.Linear(120, 84),
            nn.Tanh(),
            nn.Linear(84, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        if isinstance(x, list):
            x = [torch.flatten(xi, 1) for xi in x[0]]

        else:
            x = torch.flatten(x, 1)
        # print(x)
        x = self.classifier(x)
        return x

def modify_lenet5(model):
    model.features[0] = OutputChannelSplitConv2d(model.features[0], num_splits=4, combine=False)
    model.features[1] = ParallelActivations(activation = model.features[1])
    model.features[2] = ParallelActivations(activation = model.features[2])
    model.features[3] = InputChannelSplitConv2d(model.features[3], num_splits=4, combine=True)
    model.features[4] = ParallelActivations(activation = model.features[4])
    model.features[5] = ParallelActivations(activation = model.features[5])

    

    # model.classifier[0] = OutputChannelSplitLinear(model.classifier[0], num_splits=4, combine=False)
    # model.classifier[1] = ParallelActivations(activation = model.classifier[1])
    # model.classifier[2] = InputChannelSplitLinear(model.classifier[2], num_splits=4, combine=False)
    # model.classifier[3] = ParallelActivations(activation = model.classifier[3])
    # model.classifier[4] = OutputChannelSplitLinear(model.classifier[4], num_splits=4, combine=True)
    

    # model.classifier[0] = model.classifier[0], 
    model.classifier[1] = ParallelActivations(activation = model.classifier[1])
    model.classifier[2] = OutputChannelSplitLinear(model.classifier[2], num_splits=4, combine=False)
    model.classifier[3] = ParallelActivations(activation = model.classifier[3])
    model.classifier[4] = InputChannelSplitLinear(model.classifier[4], num_splits=4, combine=True)

    return model

# Initialize model, loss, and optimizer
model = LeNet5(num_classes=10).to(device)
model = modify_lenet5(model)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)


# Test function (similar to your VGG19 example)
def test_lenet5_on_cifar():
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    accuracy = 100 * correct / total
    print(f'Accuracy of LeNet-5 on CIFAR-10: {accuracy:.2f}%')

# Training loop
def train_model(model, train_loader, criterion, optimizer, epochs=2):
    model.train()
    for epoch in range(epochs):
        running_loss = 0.0
        
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        print(f'Epoch {epoch+1}, Loss: {running_loss/len(train_loader):.4f}')
        test_lenet5_on_cifar()

# Training loop
def train_model_MHL(model, train_loader, criterion, optimizer, epochs=2):
    model.train()
    for epoch in range(epochs):
        running_loss = 0.0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)

            # Handle splitting safely
            split_sizes = [len(images) // 4] * 4
            # split_sizes[-1] += len(images) % 4

            image_splits = list(torch.split(images, split_sizes, dim=0))
            label_splits = list(torch.split(labels, split_sizes, dim=0))
            optimizer.zero_grad()
            # print(image_splits)
            # print(label_splits)
            outputs = model(image_splits)
            total_loss = 0.0
            for outputi, labeli in zip(outputs[0], label_splits):
                
                loss = criterion(outputi, labeli)
                total_loss+= loss
            total_loss.backward()
            optimizer.step()
            running_loss += loss.item()
                
        print(f'Epoch {epoch+1}, Loss: {running_loss/len(train_loader):.4f}')

        
set =0 
while set<30:
    # train_model(model, train_loader, criterion, optimizer, epochs=1)

    model.classifier[4].combine = False
    train_model_MHL(model, train_loader_MHL, criterion, optimizer, epochs=1)
    model.classifier[4].combine = True
    test_lenet5_on_cifar()

    set+=1




