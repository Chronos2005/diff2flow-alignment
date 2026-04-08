"""Shared utility functions: image grids, tensor conversion, schedule building."""

import numpy as np
import torch
from PIL import Image


def batch_generate(sample_fn, num_images, batch_size, seed):
    """Generate `num_images` samples in batches via `sample_fn(num_samples, seed) -> Tensor`."""
    all_samples = []
    remaining = num_images
    batch_idx = 0
    while remaining > 0:
        bs = min(batch_size, remaining)
        batch_seed = seed + batch_idx if seed is not None else None
        all_samples.append(sample_fn(bs, batch_seed))
        remaining -= bs
        batch_idx += 1
    return torch.cat(all_samples, dim=0)[:num_images]


def make_grid(images, rows, cols):
    """Create a PIL grid from a list of PIL images."""
    w, h = images[0].size
    grid = Image.new("RGB", size=(cols * w, rows * h))
    for i, image in enumerate(images):
        grid.paste(image, box=(i % cols * w, i // cols * h))
    return grid


def tensor_to_pil(images):
    """Convert a batch of [-1,1] tensors to a list of PIL images."""
    images = (images / 2 + 0.5).clamp(0, 1)
    images = images.permute(0, 2, 3, 1).cpu().numpy()
    return [Image.fromarray((img * 255).astype("uint8")) for img in images]


def build_schedule(num_timesteps: int):
    """
    Build the DDPM linear beta schedule and derived quantities.
    Returns a dict of numpy arrays.
    """
    betas = np.linspace(0.0001, 0.02, num_timesteps, dtype=np.float64)
    alphas = 1.0 - betas
    alphas_cumprod = np.cumprod(alphas, axis=0)
    return {
        "betas": betas,
        "alphas": alphas,
        "alphas_cumprod": alphas_cumprod,
        "sqrt_alphas_cumprod": np.sqrt(alphas_cumprod),
        "sqrt_one_minus_alphas_cumprod": np.sqrt(1.0 - alphas_cumprod),
    }
