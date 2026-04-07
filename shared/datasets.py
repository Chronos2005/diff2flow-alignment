"""Shared dataset loading and default configurations."""

import os

from torchvision import datasets, transforms

CIFAR10_ROOT = os.path.expanduser("/scratch/ram1g23/datasets/cifar10")
CELEBA_ROOT = "/scratch/ram1g23/datasets/celeba"

DATASET_DEFAULTS = {
    "cifar10": {"image_size": 32, "train_batch_size": 128, "num_epochs": 50},
    "celeba":  {"image_size": 128, "train_batch_size": 128, "num_epochs": 100},
}


def get_dataset(name, data_root, image_size):
    """Load CIFAR-10 or CelebA with standard preprocessing."""
    if name == "cifar10":
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5]),
        ])
        return datasets.CIFAR10(root=data_root, train=True, download=False, transform=transform)
    elif name == "celeba":
        transform = transforms.Compose([
            transforms.CenterCrop(178),
            transforms.Resize(image_size),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
        ])
        return datasets.CelebA(root=data_root, split="train", target_type="attr", download=False, transform=transform)
    else:
        raise ValueError(f"Unknown dataset: {name}")


def get_data_root(dataset_name, data_root=None):
    """Return the data root, falling back to the default paths."""
    if data_root is not None:
        return data_root
    if dataset_name == "cifar10":
        return CIFAR10_ROOT
    elif dataset_name == "celeba":
        return CELEBA_ROOT
    raise ValueError(f"Unknown dataset: {dataset_name}")
