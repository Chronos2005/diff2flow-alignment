from torchvision import datasets
import os

CELEBA_ROOT = "/scratch/ram1g23/datasets/celeba"

os.makedirs(CELEBA_ROOT, exist_ok=True)

datasets.CelebA(root=CELEBA_ROOT, split="all", target_type="attr", download=True)

print("CelebA downloaded to:", CELEBA_ROOT)