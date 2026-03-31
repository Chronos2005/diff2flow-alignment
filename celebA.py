from torchvision import datasets
import os

DATA_ROOT = "/scratch/ram1g23/datasets/celeba"

os.makedirs(DATA_ROOT, exist_ok=True)

datasets.CelebA(root=DATA_ROOT, split="all", target_type="attr", download=True)

print("CelebA downloaded to:", DATA_ROOT)