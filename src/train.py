import torch
from torchvision import datasets, transforms

# MNIST dataset laden
train_dataset = datasets.MNIST(
    root="data",
    train=True,
    download=True,
    transform=transforms.ToTensor()
)

print("MNIST dataset loaded")
