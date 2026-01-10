from torchvision import datasets
import os

DATA_ROOT = os.path.expanduser("~/datasets/cifar10")

datasets.CIFAR10(root=DATA_ROOT, train=True, download=True)
datasets.CIFAR10(root=DATA_ROOT, train=False, download=True)

print("CIFAR-10 downloaded to:", DATA_ROOT)
