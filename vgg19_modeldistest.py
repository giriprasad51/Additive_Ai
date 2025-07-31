import torch
import torch.nn as nn
import torch.distributed as dist
import torch.multiprocessing as mp
# from torch.nn.parallel import DistributedDataParallel as DDP

import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models as models
from torch.utils.data import DataLoader
import torch.optim as optim
import time

import os, datetime
from Dis.layers import (DistributedOutputChannelSplitConv2d, DistributedInputChannelSplitConv2d,
DistributedOutputChannelSplitLinear,DistributedInputChannelSplitLinear, 
DistributedParallelActivations, DistributedParallelDropout)

def modify_vgg19(model, world_size, rank, device):
    # Conv + ReLU replacements (alternating output/input channel split + activation)
    model.features[0] = DistributedOutputChannelSplitConv2d(model.features[0], world_size, rank, device, combine=False)
    model.features[1] = DistributedParallelActivations(model.features[1], world_size, rank, device, combine=False)
    model.features[2] = DistributedInputChannelSplitConv2d(model.features[2], world_size, rank, device, combine=True)
    model.features[3] = DistributedParallelActivations(model.features[3], world_size, rank, device, combine=False)
    model.features[4] = DistributedParallelActivations(model.features[4], world_size, rank, device, combine=False)

    model.features[5] = DistributedOutputChannelSplitConv2d(model.features[5], world_size, rank, device, combine=False)
    model.features[6] = DistributedParallelActivations(model.features[6], world_size, rank, device, combine=False)
    model.features[7] = DistributedInputChannelSplitConv2d(model.features[7], world_size, rank, device, combine=True)
    model.features[8] = DistributedParallelActivations(model.features[8], world_size, rank, device, combine=False)
    model.features[9] = DistributedParallelActivations(model.features[9], world_size, rank, device, combine=False)


    model.features[10] = DistributedOutputChannelSplitConv2d(model.features[10], world_size, rank, device, combine=False)
    model.features[11] = DistributedParallelActivations(model.features[11], world_size, rank, device, combine=False)
    model.features[12] = DistributedInputChannelSplitConv2d(model.features[12], world_size, rank, device, combine=True)
    model.features[13] = DistributedParallelActivations(model.features[13], world_size, rank, device, combine=False)
    model.features[14] = DistributedOutputChannelSplitConv2d(model.features[14], world_size, rank, device, combine=False)
    model.features[15] = DistributedParallelActivations(model.features[15], world_size, rank, device, combine=False)
    model.features[16] = DistributedInputChannelSplitConv2d(model.features[16], world_size, rank, device, combine=True)
    model.features[17] = DistributedParallelActivations(model.features[17], world_size, rank, device, combine=False)
    model.features[18] = DistributedParallelActivations(model.features[18], world_size, rank, device, combine=False)

    model.features[19] = DistributedOutputChannelSplitConv2d(model.features[19], world_size, rank, device, combine=False)
    model.features[20] = DistributedParallelActivations(model.features[20], world_size, rank, device, combine=False)
    model.features[21] = DistributedInputChannelSplitConv2d(model.features[21], world_size, rank, device, combine=True)
    model.features[22] = DistributedParallelActivations(model.features[22], world_size, rank, device, combine=False)
    model.features[23] = DistributedOutputChannelSplitConv2d(model.features[23], world_size, rank, device, combine=False)
    model.features[24] = DistributedParallelActivations(model.features[24], world_size, rank, device, combine=False)
    model.features[25] = DistributedInputChannelSplitConv2d(model.features[25], world_size, rank, device, combine=True)
    model.features[26] = DistributedParallelActivations(model.features[26], world_size, rank, device, combine=False)
    model.features[27] = DistributedParallelActivations(model.features[27], world_size, rank, device, combine=False)

    model.features[28] = DistributedOutputChannelSplitConv2d(model.features[28], world_size, rank, device, combine=False)
    model.features[29] = DistributedParallelActivations(model.features[29], world_size, rank, device, combine=False)
    model.features[30] = DistributedInputChannelSplitConv2d(model.features[30], world_size, rank, device, combine=True)
    model.features[31] = DistributedParallelActivations(model.features[31], world_size, rank, device, combine=False)
    model.features[32] = DistributedOutputChannelSplitConv2d(model.features[32], world_size, rank, device, combine=False)
    model.features[33] = DistributedParallelActivations(model.features[33], world_size, rank, device, combine=False)
    model.features[34] = DistributedInputChannelSplitConv2d(model.features[34], world_size, rank, device, combine=True)
    model.features[35] = DistributedParallelActivations(model.features[35], world_size, rank, device, combine=False)
    model.features[36] = DistributedParallelActivations(model.features[36], world_size, rank, device, combine=False)


    # Fully connected layers
    model.classifier[0] = DistributedOutputChannelSplitLinear(model.classifier[0], world_size, rank, device, combine=False)
    model.classifier[1] = DistributedParallelActivations(model.classifier[1], world_size, rank, device, combine=False)
    model.classifier[2] = DistributedParallelDropout(model.classifier[2], world_size, rank, device, combine=False)

    model.classifier[3] = DistributedInputChannelSplitLinear(model.classifier[3], world_size, rank, device, combine=True)
    model.classifier[4] = DistributedParallelActivations(model.classifier[4], world_size, rank, device, combine=False)
    model.classifier[5] = DistributedParallelDropout(model.classifier[5], world_size, rank, device, combine=False)

    model.classifier[6] = DistributedOutputChannelSplitLinear(model.classifier[6], world_size, rank, device, combine=True)

    return model


# def modify_vgg19(model, world_size, rank, device):
#     # Conv + ReLU replacements (alternating output/input channel split + activation)
#     model.features[0] = DistributedOutputChannelSplitConv2d(model.features[0], world_size, rank, device, combine=False)
#     model.features[1] = DistributedParallelActivations(model.features[1], world_size, rank, device, combine=True)
#     model.features[2] = DistributedOutputChannelSplitConv2d(model.features[2], world_size, rank, device, combine=False)
#     model.features[3] = DistributedParallelActivations(model.features[3], world_size, rank, device, combine=False)
#     model.features[4] = DistributedParallelActivations(model.features[4], world_size, rank, device, combine=True)

#     model.features[5] = DistributedOutputChannelSplitConv2d(model.features[5], world_size, rank, device, combine=False)
#     model.features[6] = DistributedParallelActivations(model.features[6], world_size, rank, device, combine=True)
#     model.features[7] = DistributedOutputChannelSplitConv2d(model.features[7], world_size, rank, device, combine=False)
#     model.features[8] = DistributedParallelActivations(model.features[8], world_size, rank, device, combine=False)
#     model.features[9] = DistributedParallelActivations(model.features[9], world_size, rank, device, combine=True)


#     model.features[10] = DistributedOutputChannelSplitConv2d(model.features[10], world_size, rank, device, combine=False)
#     model.features[11] = DistributedParallelActivations(model.features[11], world_size, rank, device, combine=True)
#     model.features[12] = DistributedOutputChannelSplitConv2d(model.features[12], world_size, rank, device, combine=False)
#     model.features[13] = DistributedParallelActivations(model.features[13], world_size, rank, device, combine=True)
#     model.features[14] = DistributedOutputChannelSplitConv2d(model.features[14], world_size, rank, device, combine=False)
#     model.features[15] = DistributedParallelActivations(model.features[15], world_size, rank, device, combine=True)
#     model.features[16] = DistributedOutputChannelSplitConv2d(model.features[16], world_size, rank, device, combine=False)
#     model.features[17] = DistributedParallelActivations(model.features[17], world_size, rank, device, combine=False)
#     model.features[18] = DistributedParallelActivations(model.features[18], world_size, rank, device, combine=True)

#     model.features[19] = DistributedOutputChannelSplitConv2d(model.features[19], world_size, rank, device, combine=False)
#     model.features[20] = DistributedParallelActivations(model.features[20], world_size, rank, device, combine=True)
#     model.features[21] = DistributedOutputChannelSplitConv2d(model.features[21], world_size, rank, device, combine=False)
#     model.features[22] = DistributedParallelActivations(model.features[22], world_size, rank, device, combine=True)
#     model.features[23] = DistributedOutputChannelSplitConv2d(model.features[23], world_size, rank, device, combine=False)
#     model.features[24] = DistributedParallelActivations(model.features[24], world_size, rank, device, combine=True)
#     model.features[25] = DistributedOutputChannelSplitConv2d(model.features[25], world_size, rank, device, combine=False)
#     model.features[26] = DistributedParallelActivations(model.features[26], world_size, rank, device, combine=False)
#     model.features[27] = DistributedParallelActivations(model.features[27], world_size, rank, device, combine=True)

#     model.features[28] = DistributedOutputChannelSplitConv2d(model.features[28], world_size, rank, device, combine=False)
#     model.features[29] = DistributedParallelActivations(model.features[29], world_size, rank, device, combine=True)
#     model.features[30] = DistributedOutputChannelSplitConv2d(model.features[30], world_size, rank, device, combine=False)
#     model.features[31] = DistributedParallelActivations(model.features[31], world_size, rank, device, combine=True)
#     model.features[32] = DistributedOutputChannelSplitConv2d(model.features[32], world_size, rank, device, combine=False)
#     model.features[33] = DistributedParallelActivations(model.features[33], world_size, rank, device, combine=True)
#     model.features[34] = DistributedOutputChannelSplitConv2d(model.features[34], world_size, rank, device, combine=False)
#     model.features[35] = DistributedParallelActivations(model.features[35], world_size, rank, device, combine=False)
#     model.features[36] = DistributedParallelActivations(model.features[36], world_size, rank, device, combine=True)


#     # Fully connected layers
#     model.classifier[0] = DistributedOutputChannelSplitLinear(model.classifier[0], world_size, rank, device, combine=False)
#     model.classifier[1] = DistributedParallelActivations(model.classifier[1], world_size, rank, device, combine=True)
#     model.classifier[2] = DistributedParallelDropout(model.classifier[2], world_size, rank, device, combine=True)

#     model.classifier[3] = DistributedOutputChannelSplitLinear(model.classifier[3], world_size, rank, device, combine=False)
#     model.classifier[4] = DistributedParallelActivations(model.classifier[4], world_size, rank, device, combine=True)
#     model.classifier[5] = DistributedParallelDropout(model.classifier[5], world_size, rank, device, combine=True)

#     model.classifier[6] = DistributedOutputChannelSplitLinear(model.classifier[6], world_size, rank, device, combine=True)

#     return model

# Train the model
def train_model(world_size, rank, device, num_epochs = 100):
    # Device configuration
    # device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    device = 'cuda:'+str(device)

    # Hyperparameters
    batch_size = 128
    learning_rate = 1e-2

    # CIFAR-10 dataset
    

    transform=transforms.Compose([
				transforms.Resize((256,256)),
				transforms.CenterCrop(224),
				transforms.ToTensor(),
				transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
				])

    train_dataset = datasets.CIFAR10(root='/scratch/pusunuru/data', train=True, transform=transform, download=False)
    test_dataset = datasets.CIFAR10(root='/scratch/pusunuru/data', train=False, transform=transform, download=False)

    train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)

    # Load pretrained VGG19 model
    model = models.vgg19(num_classes=10)
    model.to(device)
    for param in model.parameters():
        dist.broadcast(param.data, src=0)
    
    print(model)
    
    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9)

    model = modify_vgg19(model, world_size, rank, device)
    print(model)
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
            if torch.isnan(outputs).any():
                print("Any NaNs in outputs?", torch.isnan(outputs).sum().item(), (~torch.isnan(outputs)).sum().item(), outputs.shape)
                print("Any NaNs in target?", torch.isnan(targets).sum().item(), (~torch.isnan(targets)).sum().item(), targets.shape)
                print("Any NaNs in inpust?", torch.isnan(inputs).sum().item(), (~torch.isnan(inputs)).sum().item(), inputs.shape)



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
            print("--------step--------")

        end_time = time.time()
        
        epoch_accuracy = 100. * correct / total
        epoch_loss /= len(train_loader)

        

        print(f'Epoch [{epoch+1}/{num_epochs}] - Loss: {epoch_loss:.4f}, Accuracy: {epoch_accuracy:.2f}%, Time: {end_time - start_time:.2f}s')

def test_vgg19parallel_on_cifar(model):
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
    
def run(rank, world_size, device):
    
    torch.cuda.set_device(device)
    device = 'cuda:'+str(device)

    model = models.vgg19(num_classes=10)
    model = modify_vgg19(model, world_size, rank, device)
    
    

    x = torch.randn(2, 3, 224, 224).to(device)
    y = torch.randn(2, 10).to(device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

    for i in range(3):
        optimizer.zero_grad()
        output = model(x)
        loss = criterion(output, y)
        loss.backward()
        optimizer.step()
        if rank == 0:
            print(f"[Step {i}] Loss: {loss.item():.4f}")

    


def main():
    rank = int(os.environ['RANK'])
    world_size = int(os.environ['WORLD_SIZE'])
    gpu_id = int(os.environ['LOCAL_RANK'])
    device = gpu_id
    
    print(device)
    # print(torch.cuda.device_count())
    dist.init_process_group(backend='gloo', timeout=datetime.timedelta(seconds=7200))
    # run(rank, world_size, device)
    train_model(world_size, rank, device, num_epochs = 30)
    print(f"Rank {device} Total time taken: ", datetime.datetime.now() - FULL_START)
    dist.destroy_process_group()
    


if __name__ == '__main__':
    total_epochs = 200
    FULL_START = datetime.datetime.now()
    main()
    exit(0)