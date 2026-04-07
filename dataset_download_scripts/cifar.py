from torchvision import datasets
import os

# Updated path to match new location
CIFAR10_ROOT = os.path.expanduser(
    "/scratch/ram1g23/datasets/cifar10"
)

if __name__ == "__main__":
    # Download CIFAR-10 (if not already present)
    datasets.CIFAR10(root=CIFAR10_ROOT, train=True, download=True)
    datasets.CIFAR10(root=CIFAR10_ROOT, train=False, download=True)
    print("CIFAR-10 downloaded to:", CIFAR10_ROOT)
