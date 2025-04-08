import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models as models
from torch.utils.data import DataLoader
import torch.optim as optim
import time

from layers import OutputChannelSplitConv2d, InputChannelSplitConv2d, OutputChannelSplitLinear, InputChannelSplitLinear, ParallelReLU, ParallelMaxPool2d, ParallelDropout

def test_vgg19_on_cifar():
    device = torch.device("cuda" if torch.cuda.is_available() else True)
    
    # Define data transformations
    transform = transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])
    
    # Load CIFAR-10 test dataset
    test_dataset = datasets.CIFAR10(root='/scratch/pusunuru/data', train=False, download=False, transform=transform)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)
    
    # Load pretrained VGG19 model
    model = models.vgg19(num_classes=10)
    model.load_state_dict(torch.load('./vgg19_cifar10.pth'))
    
    # Modify classifier for CIFAR-10 (10 classes)
    model = model.to(device)
    model.eval()
    # print(model)
    
    # Test the model
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    
    accuracy = 100 * correct / total
    print(f'Accuracy of VGG19 on CIFAR-10 test set: {accuracy:.2f}%')

def modify_vgg19(model):
    """Manually modify all layers without loops"""
    model.features[0] = OutputChannelSplitConv2d(model.features[0], num_splits=4, combine=False)
    model.features[1] = ParallelReLU()
    model.features[2] = InputChannelSplitConv2d(model.features[2], num_splits=4, combine=True)
    model.features[3] = ParallelReLU()
    model.features[4] = ParallelMaxPool2d(model.features[4], combine=True)
    model.features[5] = OutputChannelSplitConv2d(model.features[5], num_splits=4, combine=False)
    model.features[6] = ParallelReLU()
    model.features[7] = InputChannelSplitConv2d(model.features[7], num_splits=4, combine=True)
    model.features[8] = ParallelReLU()
    model.features[9] = ParallelMaxPool2d(model.features[9], combine=True)
    model.features[10] = OutputChannelSplitConv2d(model.features[10], num_splits=4, combine=False)
    model.features[11] = ParallelReLU()
    model.features[12] = InputChannelSplitConv2d(model.features[12], num_splits=4, combine=True)
    model.features[13] = ParallelReLU()
    model.features[14] = OutputChannelSplitConv2d(model.features[14], num_splits=4, combine=False)
    model.features[15] = ParallelReLU()
    model.features[16] = InputChannelSplitConv2d(model.features[16], num_splits=4, combine=True)
    model.features[17] = ParallelReLU()
    model.features[18] = ParallelMaxPool2d(model.features[18], combine=True)
    model.features[19] = OutputChannelSplitConv2d(model.features[19], num_splits=4, combine=False)
    model.features[20] = ParallelReLU()
    model.features[21] = InputChannelSplitConv2d(model.features[21], num_splits=4, combine=True)
    model.features[22] = ParallelReLU()
    model.features[23] = OutputChannelSplitConv2d(model.features[23], num_splits=4, combine=False)
    model.features[24] = ParallelReLU()
    model.features[25] = InputChannelSplitConv2d(model.features[25], num_splits=4, combine=True)
    model.features[26] = ParallelReLU()
    model.features[27] = ParallelMaxPool2d(model.features[27], combine=True)
    model.features[28] = OutputChannelSplitConv2d(model.features[28], num_splits=4, combine=False)
    model.features[29] = ParallelReLU()
    model.features[30] = InputChannelSplitConv2d(model.features[30], num_splits=4, combine=True)
    model.features[31] = ParallelReLU()
    model.features[32] = OutputChannelSplitConv2d(model.features[32], num_splits=4, combine=False)
    model.features[33] = ParallelReLU()
    model.features[34] = InputChannelSplitConv2d(model.features[34], num_splits=4, combine=True)
    model.features[35] = ParallelReLU()
    model.features[36] = ParallelMaxPool2d(model.features[36], combine=True)
    
    model.classifier[0] = OutputChannelSplitLinear(model.classifier[0], num_splits=4, combine=False)
    model.classifier[1] = ParallelReLU()
    model.classifier[2] = ParallelDropout(model.classifier[2])
    model.classifier[3] = InputChannelSplitLinear(model.classifier[3], num_splits=4, combine=True)
    model.classifier[4] = ParallelReLU()
    model.classifier[5] = ParallelDropout(model.classifier[5])
    model.classifier[6] = OutputChannelSplitLinear(model.classifier[6], num_splits=4, combine=True)
    
    return model


def test_vgg19parallel_on_cifar():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Define data transformations
    transform = transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])
    
    # Load CIFAR-10 test dataset
    test_dataset = datasets.CIFAR10(root='/scratch/pusunuru/data', train=False, download=False, transform=transform)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)
    
    # Load pretrained VGG19 model
    model = models.vgg19(num_classes=10)
    model.load_state_dict(torch.load('./vgg19_cifar10.pth'))
    
    
    model = model.to(device)


    model = modify_vgg19(model)
    model.eval()
    
    # Test the model
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    
    accuracy = 100 * correct / total
    print(f'Accuracy of VGG19 parallel on CIFAR-10 test set: {accuracy:.2f}%')




# Train the model
def train_model(num_epochs = 100):
    # Device configuration
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Hyperparameters
    batch_size = 128
    learning_rate = 0.01

    # CIFAR-10 dataset
    transform = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomCrop(32, padding=4),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    train_dataset = datasets.CIFAR10(root='/scratch/pusunuru/data', train=True, transform=transform, download=False)
    test_dataset = datasets.CIFAR10(root='/scratch/pusunuru/data', train=False, transform=transform, download=False)

    train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)

    # Load pretrained VGG19 model
    model = models.vgg19(num_classes=10)
    # model.load_state_dict(torch.load('./vgg19_cifar10.pth'))
    
    
    model = model.to(device)


    model = modify_vgg19(model)

    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9, weight_decay=5e-4)

    for epoch in range(num_epochs):
        model.train()
        epoch_loss = 0
        correct = 0
        total = 0
        start_time = time.time()

        for inputs, targets in train_loader:
            inputs, targets = inputs.to(device), targets.to(device)

            # Forward pass
            start_forward = time.time()
            outputs = model(inputs)
            end_forward = time.time()

            # Compute loss
            loss = criterion(outputs, targets)

            # Backward pass
            start_backward = time.time()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            end_backward = time.time()

            epoch_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

        end_time = time.time()
        
        epoch_accuracy = 100. * correct / total
        epoch_loss /= len(train_loader)

        

        print(f'Epoch [{epoch+1}/{num_epochs}] - Loss: {epoch_loss:.4f}, Accuracy: {epoch_accuracy:.2f}%, Time: {end_time - start_time:.2f}s')



if __name__ == "__main__":
    # test_vgg19_on_cifar()
    # test_vgg19parallel_on_cifar()
    train_model(num_epochs = 10)
