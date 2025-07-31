import torch
import torch.nn as nn
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

import sys
import os
import datetime
import subprocess
from models_1D import  (
    ParallelAlexNet, 
    ParallelDenseNet,
    ParallelResNet50,
    ParallelVGG19,
    ParallelVGG16

)

# def checkcuda(rank):
#     result = subprocess.check_output(
#         ['nvidia-smi', '--query-gpu=memory.total,memory.used,memory.free', '--format=csv,nounits,noheader'],
#         encoding='utf-8'
#     )

#     gpus = result.strip().split('\n')
    
#     # Prepare data
#     data = []
#     headers = ["Rank", "CUDA", "Total Memory (GB)", "Used Memory (GB)", "Free Memory (GB)"]
    
#     for i, gpu in enumerate(gpus):
#         total, used, free = map(int, gpu.split(', '))
#         data.append([
#             f"CUDA:{i}", 
#             f"{total / 1024:.2f}", 
#             f"{used / 1024:.2f}", 
#             f"{free / 1024:.2f}"
#         ])
    
#     # Calculate column widths
#     col_widths = [max(len(str(item)) for item in col) for col in zip(headers, *data)]
    
#     # Print table header
#     header_row = " | ".join(f"{header:<{width}}" for header, width in zip(headers, col_widths))
#     print(header_row)
#     print("-" * len(header_row))
    
#     # Print table rows using second loop
#     for row in data:
#         print("| "+ str(rank) +" | ".join(f"{item:<{width}}" for item, width in zip(row, col_widths)))

class TrainerTP:
    def __init__(self, model, train_data, test_data, optimizer, device_mesh):
        self.start_time = datetime.datetime.now()
        self.train_data = train_data
        self.global_rank = int(os.environ['RANK'])
        self.local_rank = int(os.environ['LOCAL_RANK'])
        self.device_id = "cuda:" + str(self.local_rank)
        self.model = model.to(self.device_id)
        self.test_data = test_data
        self.optimizer = optimizer
        self.device_mesh = device_mesh
        
        # Initialize timing variables
        self.total_train_time = datetime.timedelta(0)
        self.total_test_time = datetime.timedelta(0)
        self.epoch_train_times = []
        self.epoch_test_times = []

        torch.distributed.barrier()  # Sync all ranks before starting
        print(f"|| Initialized TP Trainer || Rank {self.global_rank} Device {self.device_id}")

    def _run_batch(self, X, y):
        self.optimizer.zero_grad()
        out = self.model(X)
        loss = nn.CrossEntropyLoss()(out, y)
        correct = (out.argmax(1) == y).type(torch.float).sum().item()
        loss.backward()
        self.optimizer.step()
        return correct, loss.item()

    def _run_epoch(self, epoch):
        epoch_start_time = datetime.datetime.now()
        
        total_correct = 0
        total_loss = 0.0
        total_samples = 0

        self.model.train()

        for batch, (X, y) in enumerate(self.train_data):
            X, y = X.to(self.device_id), y.to(self.device_id)
            correct, loss = self._run_batch(X, y)

            total_correct += correct
            total_loss += loss * X.size(0)
            total_samples += X.size(0)

            del X, y

        # Reduce across TP ranks
        total_correct_tensor = torch.tensor([total_correct], dtype=torch.float64, device=self.device_id)
        total_samples_tensor = torch.tensor([total_samples], dtype=torch.float64, device=self.device_id)

        dist.all_reduce(total_correct_tensor, op=dist.ReduceOp.SUM)
        dist.all_reduce(total_samples_tensor, op=dist.ReduceOp.SUM)

        accuracy = total_correct_tensor.item() / total_samples_tensor.item()
        
        epoch_time = datetime.datetime.now() - epoch_start_time
        self.epoch_train_times.append(epoch_time)
        self.total_train_time += epoch_time
        
        return accuracy

    def train(self, max_epochs, early_stop=True):
        TRAIN_TIME = datetime.datetime.now()
        for epoch in range(max_epochs):
            train_acc = self._run_epoch(epoch)

            if self.global_rank == 0:
                epoch_time = self.epoch_train_times[-1]
                print(f"Epoch {epoch} | Train Accuracy: {train_acc*100:.2f}% |")

            # Run test and measure its time
            test_start = datetime.datetime.now()
            acc = self.test()
            test_time = datetime.datetime.now() - test_start
            self.total_test_time += test_time
            self.epoch_test_times.append(test_time)

            # Print final timing summary
            if self.global_rank == 0:
                print("=== Training Complete ===")
                print(f"Total Training Time: {self.total_train_time}")
                print(f"Total Testing Time: {self.total_test_time}")
                print(f"Total Time (Train+Test): {self.total_train_time + self.total_test_time} \n\n")
                

    def test(self):
        self.test_start_time = datetime.datetime.now()
        size = len(self.test_data.dataset)
        num_batches = len(self.test_data)

        self.model.eval()
        test_loss, correct = 0, 0

        with torch.no_grad():
            for X, y in self.test_data:
                X, y = X.to(self.device_id), y.to(self.device_id)
                out = self.model(X)
                test_loss += nn.CrossEntropyLoss()(out, y).item()
                correct += (out.argmax(1) == y).type(torch.float).sum().item()
                del X, y

        # Reduce across TP ranks
        correct_tensor = torch.tensor([correct], dtype=torch.float64, device=self.device_id)
        size_tensor = torch.tensor([size], dtype=torch.float64, device=self.device_id)

        dist.all_reduce(correct_tensor, op=dist.ReduceOp.SUM)
        dist.all_reduce(size_tensor, op=dist.ReduceOp.SUM)

        correct = correct_tensor.item()
        size = size_tensor.item()

        test_loss /= num_batches
        acc = correct / size

        test_time = datetime.datetime.now() - self.test_start_time
        
        if self.global_rank == 0:
            print(f"Rank {self.global_rank}: Test Accuracy: {(100 * acc):>0.1f}%, Avg loss: {test_loss:>8f} | ")

        return acc




def setup(rank, world_size):
    dist.init_process_group("nccl", rank=rank, world_size=world_size)

def cleanup():
    dist.destroy_process_group()

def get_data_loaders(rank, world_size, batch_size, dataset_name):
    # transform = transforms.Compose([
    #     transforms.RandomCrop(32, padding=4),
    #     transforms.RandomHorizontalFlip(),
    #     transforms.ToTensor(),
    #     transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    # ])

    # test_transform = transforms.Compose([
    #     transforms.ToTensor(),
    #     transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    # ])
    if 'cifar' in dataset_name.lower():
        transform = transforms.Compose([
                    transforms.Resize((256,256)),
                    transforms.CenterCrop(224),
                    transforms.ToTensor(),
                    transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
        ])

        test_transform =transforms.Compose([
                    transforms.Resize((256, 256)),
                    transforms.CenterCrop(224),
                    transforms.ToTensor(),
                    transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
        ])

    
        train_dataset = datasets.CIFAR10(
            root='/scratch/cdssona/data',
            train=True,
            download=False,
            transform=transform
        )

        test_dataset = datasets.CIFAR10(
            root='/scratch/cdssona/data',
            train=False,
            download=False,
            transform=test_transform
        )
    elif "mnist" in dataset_name.lower():
        # Data transforms
        transform = transforms.Compose([
            transforms.Resize(224),
            transforms.Grayscale(3),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])

        # Download and load the training dataset
        train_dataset = datasets.MNIST(root='/scratch/cdssona/data', train=True, download=False, transform=transform)

        # Download and load the test dataset
        test_dataset = datasets.MNIST(root='/scratch/cdssona/data', train=False, download=False, transform=transform)


    train_sampler = DistributedSampler(
        train_dataset,
        num_replicas=world_size,
        rank=rank,
        shuffle=True
    )

    test_sampler = DistributedSampler(
        test_dataset,
        num_replicas=world_size,
        rank=rank,
        shuffle=False
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        # num_workers=4,  # Reduce CPU bottleneck
        # pin_memory=True 
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=True,
        # num_workers=4,  # Reduce CPU bottleneck
        # pin_memory=True, 
    )
    
    return train_loader, test_loader, train_sampler

def main():

    model_name, dataset_name = sys.argv[1], sys.argv[2]
    rank = int(os.environ['RANK'])
    world_size = int(os.environ['WORLD_SIZE'])
    local_rank = int(os.environ['LOCAL_RANK'])
    batch_size = 64
    print(batch_size)
    setup(rank, world_size)
    torch.cuda.set_device(local_rank)
    
    # Create 1D device mesh
    device_mesh = DeviceMesh("cuda", list(range(world_size)))

    
    # checkcuda(rank)
    # Model and optimizer
    
    # Model selection
    if model_name.lower() == "alexnet":
        model = ParallelAlexNet(device_mesh)
    elif model_name.lower() == "densenet121":
        model = ParallelDenseNet(device_mesh)
    elif model_name.lower() == "resnet50":
        model = ParallelResNet50(device_mesh, num_classes=10)
    elif model_name.lower() == "vgg19":
        model = ParallelVGG19(device_mesh)
    elif model_name.lower() == "vgg16":
        model = ParallelVGG16(device_mesh)
    else:
        raise ValueError(f"Unknown model: {model_name}")
    print(model)
    # checkcuda(rank)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9, weight_decay=5e-4)
    
    if "mnist" in dataset_name.lower():
        optimizer = torch.optim.Adam(model.parameters(), lr=0.00001)
    
    # Data loaders
    train_loader, test_loader, train_sampler = get_data_loaders(rank, world_size, batch_size, dataset_name)
    
    # Initialize trainer
    trainer = TrainerTP(model, train_loader, test_loader, optimizer, device_mesh)
    
    # Train the model
    trainer.train(max_epochs=30)
    
    cleanup()

if __name__ == "__main__":
    main()