"""
Diff2Flow Inference Script

Loads a Diff2Flow-finetuned model checkpoint and generates images using
Euler integration on the flow matching ODE. No training — just sampling.

Usage:
    # Basic usage
    python diff2flow_inference.py \
        --checkpoint_path diff2flow_cifar10/final_model/diff2flow_final.pt \
        --num_images 16 \
        --num_steps 50

    # Fewer steps (faster, slightly lower quality)
    python diff2flow_inference.py \
        --checkpoint_path diff2flow_cifar10/final_model/diff2flow_final.pt \
        --num_images 64 \
        --num_steps 10

    # Compare different step counts
    python diff2flow_inference.py \
        --checkpoint_path diff2flow_cifar10/final_model/diff2flow_final.pt \
        --num_images 16 \
        --num_steps 2 4 10 25 50 100 \
        --output_dir step_comparison
"""

import argparse
import math
import os
import time

import torch
from diffusers import DDPMScheduler, UNet2DModel
from PIL import Image


# ---------------------------------------------------------------------------
# Diff2Flow Aligner (same as in training script)
# ---------------------------------------------------------------------------

class Diff2FlowAligner:
    """Analytical alignment between diffusion and flow matching trajectories."""

    def __init__(self, noise_scheduler: DDPMScheduler):
        alphas_cumprod = noise_scheduler.alphas_cumprod
        self.alpha = torch.sqrt(alphas_cumprod)
        self.sigma = torch.sqrt(1.0 - alphas_cumprod)
        self.T = len(alphas_cumprod)
        self.ft_values = self.alpha / (self.alpha + self.sigma)

    def t_fm_to_t_dm(self, t_fm: torch.Tensor) -> torch.Tensor:
        device = t_fm.device
        ft = self.ft_values.to(device)
        ft_ascending = ft.flip(0)
        idx_asc = torch.searchsorted(ft_ascending, t_fm.clamp(ft_ascending[0], ft_ascending[-1]))
        idx_asc = idx_asc.clamp(1, len(ft_ascending) - 1)

        idx_hi_asc = idx_asc
        idx_lo_asc = idx_asc - 1

        ft_lo = ft_ascending[idx_lo_asc]
        ft_hi = ft_ascending[idx_hi_asc]

        t_dm_lo_asc = (self.T - 1 - idx_lo_asc).float()
        t_dm_hi_asc = (self.T - 1 - idx_hi_asc).float()

        denom = (ft_hi - ft_lo).clamp(min=1e-8)
        w = (t_fm - ft_lo) / denom
        t_dm = t_dm_lo_asc + w * (t_dm_hi_asc - t_dm_lo_asc)
        return t_dm

    def get_alpha_sigma(self, t_dm: torch.Tensor) -> tuple:
        device = t_dm.device
        alpha = self.alpha.to(device)
        sigma = self.sigma.to(device)

        t_lo = t_dm.long().clamp(0, self.T - 2)
        t_hi = t_lo + 1
        w = (t_dm - t_lo.float()).clamp(0, 1)

        alpha_t = alpha[t_lo] * (1 - w) + alpha[t_hi] * w
        sigma_t = sigma[t_lo] * (1 - w) + sigma[t_hi] * w
        return alpha_t, sigma_t

    def x_fm_to_x_dm(self, x_fm, alpha_t, sigma_t):
        scale = (alpha_t + sigma_t)
        while scale.dim() < x_fm.dim():
            scale = scale.unsqueeze(-1)
        return scale * x_fm

    def eps_to_velocity(self, eps_pred, x_dm, alpha_t, sigma_t):
        a = alpha_t.clone()
        s = sigma_t.clone()
        while a.dim() < x_dm.dim():
            a = a.unsqueeze(-1)
            s = s.unsqueeze(-1)

        x0_hat = (x_dm - s * eps_pred) / a.clamp(min=1e-8)
        velocity = x0_hat - eps_pred
        return velocity


# ---------------------------------------------------------------------------
# LoRA modules (needed to load LoRA checkpoints)
# ---------------------------------------------------------------------------

class LoRALinear(torch.nn.Module):
    def __init__(self, original, rank):
        super().__init__()
        self.original = original
        self.lora_down = torch.nn.Linear(original.in_features, rank, bias=False)
        self.lora_up = torch.nn.Linear(rank, original.out_features, bias=False)

    def forward(self, x):
        return self.original(x) + self.lora_up(self.lora_down(x))


class LoRAConv2d(torch.nn.Module):
    def __init__(self, original, rank):
        super().__init__()
        self.original = original
        self.lora_down = torch.nn.Conv2d(original.in_channels, rank, 1, bias=False)
        self.lora_up = torch.nn.Conv2d(rank, original.out_channels, 1, bias=False)

    def forward(self, x):
        return self.original(x) + self.lora_up(self.lora_down(x))


def apply_lora(model, rank=64):
    """Apply LoRA to all Linear and Conv2d layers (must match training config)."""
    import torch.nn as nn
    replaced = 0
    for name, module in list(model.named_modules()):
        parts = name.split(".")
        parent = model
        for p in parts[:-1]:
            parent = getattr(parent, p)
        attr_name = parts[-1] if parts else None

        if attr_name and isinstance(module, nn.Linear):
            r = min(rank, module.in_features, module.out_features)
            setattr(parent, attr_name, LoRALinear(module, r))
            replaced += 1
        elif attr_name and isinstance(module, nn.Conv2d):
            r = min(rank, module.in_channels, module.out_channels)
            setattr(parent, attr_name, LoRAConv2d(module, r))
            replaced += 1
    return replaced


# ---------------------------------------------------------------------------
# Sampling
# ---------------------------------------------------------------------------

@torch.no_grad()
def sample_euler(model, aligner, num_samples, image_size, num_steps, device, seed=None):
    """
    Generate images via Euler integration on the FM ODE.
    t=0 (noise) -> t=1 (data)
    """
    if seed is not None:
        torch.manual_seed(seed)

    shape = (num_samples, 3, image_size, image_size)
    x = torch.randn(shape, device=device)

    dt = 1.0 / num_steps
    for i in range(num_steps):
        t_fm = torch.full((num_samples,), i * dt, device=device)

        t_dm = aligner.t_fm_to_t_dm(t_fm)
        alpha_t, sigma_t = aligner.get_alpha_sigma(t_dm)
        x_dm = aligner.x_fm_to_x_dm(x, alpha_t, sigma_t)

        t_dm_input = t_dm.long().clamp(0, aligner.T - 1)
        eps_pred = model(x_dm, t_dm_input, return_dict=False)[0]
        velocity = aligner.eps_to_velocity(eps_pred, x_dm, alpha_t, sigma_t)

        x = x + dt * velocity

    return x.clamp(-1, 1)


def tensor_to_pil(images):
    images = (images / 2 + 0.5).clamp(0, 1)
    images = images.permute(0, 2, 3, 1).cpu().numpy()
    return [Image.fromarray((img * 255).astype("uint8")) for img in images]


def make_grid(images, cols=None):
    if cols is None:
        cols = int(math.ceil(math.sqrt(len(images))))
    rows = int(math.ceil(len(images) / cols))
    w, h = images[0].size
    grid = Image.new("RGB", size=(cols * w, rows * h))
    for i, image in enumerate(images):
        grid.paste(image, box=(i % cols * w, i // cols * h))
    return grid


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(description="Diff2Flow Inference")

    parser.add_argument("--checkpoint_path", type=str, required=True,
                        help="Path to Diff2Flow checkpoint (.pt file)")
    parser.add_argument("--pretrained_model_path", type=str, default=None,
                        help="Path to original DDPM model (for architecture). "
                             "If not provided, will try to infer from checkpoint args.")
    parser.add_argument("--num_images", type=int, default=16,
                        help="Number of images to generate")
    parser.add_argument("--num_steps", type=int, nargs="+", default=[50],
                        help="Number of Euler steps. Pass multiple values to compare, "
                             "e.g. --num_steps 2 4 10 50")
    parser.add_argument("--image_size", type=int, default=32)
    parser.add_argument("--num_train_timesteps", type=int, default=1000)
    parser.add_argument("--batch_size", type=int, default=16,
                        help="Batch size for generation (if num_images > batch_size)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output_dir", type=str, default="diff2flow_samples")
    parser.add_argument("--use_lora", action="store_true",
                        help="Set if the checkpoint was trained with LoRA")
    parser.add_argument("--lora_rank", type=int, default=64)
    parser.add_argument("--device", type=str, default=None,
                        help="Device (default: cuda if available, else cpu)")
    parser.add_argument("--save_individual", action="store_true",
                        help="Also save each image individually")

    return parser.parse_args()


def load_model(args, device):
    """Load the model architecture and Diff2Flow checkpoint weights."""

    checkpoint = torch.load(args.checkpoint_path, map_location="cpu", weights_only=False)

    # Try to get the pretrained model path from checkpoint metadata
    ckpt_args = checkpoint.get("args", {})
    pretrained_path = args.pretrained_model_path or ckpt_args.get("pretrained_model_path")

    if pretrained_path is None:
        raise ValueError(
            "Cannot determine the original DDPM model path. "
            "Please provide --pretrained_model_path."
        )

    print(f"Loading architecture from: {pretrained_path}")
    model = UNet2DModel.from_pretrained(pretrained_path)

    # Apply LoRA if the checkpoint was trained with it
    use_lora = args.use_lora or ckpt_args.get("use_lora", False)
    lora_rank = args.lora_rank or ckpt_args.get("lora_rank", 64)

    if use_lora:
        print(f"Applying LoRA (rank={lora_rank}) to match training config")
        apply_lora(model, rank=lora_rank)

    # Load the finetuned weights
    model.load_state_dict(checkpoint["model_state_dict"])

    model = model.to(device)
    model.eval()

    epoch = checkpoint.get("epoch", "?")
    step = checkpoint.get("global_step", "?")
    print(f"Loaded checkpoint: epoch={epoch}, global_step={step}")

    return model


def generate_samples(model, aligner, num_images, image_size, num_steps, batch_size, device, seed):
    """Generate samples in batches."""
    all_images = []
    remaining = num_images
    batch_idx = 0

    while remaining > 0:
        bs = min(batch_size, remaining)
        batch_seed = seed + batch_idx if seed is not None else None

        samples = sample_euler(
            model, aligner,
            num_samples=bs,
            image_size=image_size,
            num_steps=num_steps,
            device=device,
            seed=batch_seed,
        )
        all_images.append(samples)
        remaining -= bs
        batch_idx += 1

    return torch.cat(all_images, dim=0)[:num_images]


def main():
    args = parse_args()

    # Device
    if args.device:
        device = torch.device(args.device)
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    os.makedirs(args.output_dir, exist_ok=True)

    # Load model
    model = load_model(args, device)

    # Build aligner
    noise_scheduler = DDPMScheduler(num_train_timesteps=args.num_train_timesteps)
    aligner = Diff2FlowAligner(noise_scheduler)

    # Generate for each step count
    for num_steps in args.num_steps:
        print(f"\n--- Generating {args.num_images} images with {num_steps} Euler steps ---")

        start = time.time()
        samples = generate_samples(
            model, aligner,
            num_images=args.num_images,
            image_size=args.image_size,
            num_steps=num_steps,
            batch_size=args.batch_size,
            device=device,
            seed=args.seed,
        )
        elapsed = time.time() - start

        pil_images = tensor_to_pil(samples)

        # Save grid
        grid = make_grid(pil_images)
        grid_path = os.path.join(args.output_dir, f"grid_steps_{num_steps}.png")
        grid.save(grid_path)
        print(f"Saved grid: {grid_path}")
        print(f"Time: {elapsed:.2f}s ({elapsed/args.num_images:.3f}s per image)")

        # Optionally save individual images
        if args.save_individual:
            ind_dir = os.path.join(args.output_dir, f"steps_{num_steps}")
            os.makedirs(ind_dir, exist_ok=True)
            for i, img in enumerate(pil_images):
                img.save(os.path.join(ind_dir, f"{i:04d}.png"))
            print(f"Saved {len(pil_images)} individual images to {ind_dir}/")

    print("\nDone!")


if __name__ == "__main__":
    main()