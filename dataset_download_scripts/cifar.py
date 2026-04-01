from torchvision import datasets
import os

# Updated path to match new location
DATA_ROOT = os.path.expanduser(
    "~/Projects/diff2flow-alignment/datasets/cifar10"
)

# Download CIFAR-10 (if not already present)
datasets.CIFAR10(root=DATA_ROOT, train=True, download=True)
datasets.CIFAR10(root=DATA_ROOT, train=False, download=True)

print("CIFAR-10 downloaded to:", DATA_ROOT)