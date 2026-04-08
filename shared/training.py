"""Shared training utilities for train_* scripts."""

import torch
from diffusers import UNet2DModel
from diffusers.optimization import get_cosine_schedule_with_warmup
from torch.utils.data import DataLoader


def make_dataloader(dataset, batch_size, num_workers):
    """Standard shuffled DataLoader for image training."""
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
    )


def make_optimizer_and_scheduler(params, lr, num_warmup_steps, num_training_steps):
    """AdamW optimizer with cosine LR schedule with warmup."""
    optimizer = torch.optim.AdamW(params, lr=lr)
    lr_scheduler = get_cosine_schedule_with_warmup(
        optimizer=optimizer,
        num_warmup_steps=num_warmup_steps,
        num_training_steps=num_training_steps,
    )
    return optimizer, lr_scheduler


def build_unet(image_size):
    """Standard UNet2DModel for image generation (DDPM / Flow Matching)."""
    return UNet2DModel(
        sample_size=image_size,
        in_channels=3,
        out_channels=3,
        layers_per_block=2,
        block_out_channels=(128, 128, 256, 256, 512, 512),
        down_block_types=(
            "DownBlock2D",
            "DownBlock2D",
            "DownBlock2D",
            "DownBlock2D",
            "AttnDownBlock2D",
            "DownBlock2D",
        ),
        up_block_types=(
            "UpBlock2D",
            "AttnUpBlock2D",
            "UpBlock2D",
            "UpBlock2D",
            "UpBlock2D",
            "UpBlock2D",
        ),
    )
