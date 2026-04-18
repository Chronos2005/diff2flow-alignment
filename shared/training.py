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


# Per-resolution UNet configs. Attention is placed at 8×8 in both cases,
# giving a 64-element sequence — enough to be useful without being wasteful.
_UNET_CONFIGS = {
    32: dict(  # CIFAR-10: 32→16→8→4(bottleneck), attention at 8×8
        block_out_channels=(128, 256, 256, 256),
        down_block_types=(
            "DownBlock2D",       # 32→16
            "DownBlock2D",       # 16→8
            "AttnDownBlock2D",   # 8→4  (attention over 8×8 features)
            "DownBlock2D",       # bottleneck at 4×4
        ),
        up_block_types=(
            "UpBlock2D",
            "AttnUpBlock2D",     # mirrors AttnDownBlock2D
            "UpBlock2D",
            "UpBlock2D",
        ),
    ),
    64: dict(  # CelebA: 64→32→16→8→4(bottleneck), attention at 8×8
        block_out_channels=(128, 128, 256, 256, 512),
        down_block_types=(
            "DownBlock2D",       # 64→32
            "DownBlock2D",       # 32→16
            "DownBlock2D",       # 16→8
            "AttnDownBlock2D",   # 8→4  (attention over 8×8 features)
            "DownBlock2D",       # bottleneck at 4×4
        ),
        up_block_types=(
            "UpBlock2D",
            "AttnUpBlock2D",     # mirrors AttnDownBlock2D
            "UpBlock2D",
            "UpBlock2D",
            "UpBlock2D",
        ),
    ),
}


def build_unet(image_size):
    """Standard UNet2DModel for image generation (DDPM / Flow Matching).

    Architecture is chosen based on image_size so attention always lands at 8×8,
    regardless of dataset resolution.
    """
    if image_size not in _UNET_CONFIGS:
        raise ValueError(f"No UNet config for image_size={image_size}. Add one to _UNET_CONFIGS.")
    cfg = _UNET_CONFIGS[image_size]
    return UNet2DModel(
        sample_size=image_size,
        in_channels=3,
        out_channels=3,
        layers_per_block=2,
        **cfg,
    )
