import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models as models
from torch.utils.data import DataLoader
from layers import OutputChannelSplitConv2d, InputChannelSplitConv2d, OutputChannelSplitLinear, InputChannelSplitLinear, ParallelReLU

def test_vgg19_on_cifar():
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
    
    # Modify classifier for CIFAR-10 (10 classes)
    # model.classifier[6] = nn.Linear(4096, 10)
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

    model.features[0] = OutputChannelSplitConv2d(model.features[0], num_splits=4, combine=False)
    model.features[1] = ParallelReLU()
    model.features[2] = InputChannelSplitConv2d(model.features[2], num_splits=4 )
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


if __name__ == "__main__":
    test_vgg19_on_cifar()
    test_vgg19parallel_on_cifar()
