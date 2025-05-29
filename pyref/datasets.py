import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

def get_mnist_loaders(batch_size, num_workers=0):
    transform = transforms.Compose([
        transforms.ToTensor(), 
        transforms.Normalize((0.1307,), (0.3081,)) 
    ])
    train_dataset = datasets.MNIST(
        root='data/',
        train=True,
        download=True,
        transform=transform
    )
    test_dataset = datasets.MNIST(
        root='data/',
        train=False,
        download=True,
        transform=transform
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    return train_loader, test_loader

if __name__ == '__main__':
    train_loader, test_loader = get_mnist_loaders(256)
    for idx, batch in enumerate(train_loader):
        print(batch[0].shape)
        break