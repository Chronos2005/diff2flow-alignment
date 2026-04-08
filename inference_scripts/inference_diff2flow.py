"""
Diff2Flow Inference Script

Loads a Diff2Flow-finetuned model checkpoint and generates images using
Euler integration on the flow matching ODE. No training — just sampling.

Usage:
    # Basic usage
    python diff2flow_inference.py \
        --checkpoint_path diff2flow_cifar10/final_model \
        --num_images 16 \
        --num_steps 50

    # Fewer steps (faster, slightly lower quality)
    python diff2flow_inference.py \
        --checkpoint_path diff2flow_cifar10/final_model \
        --num_images 64 \
        --num_steps 10

    # Compare different step counts
    python diff2flow_inference.py \
        --checkpoint_path diff2flow_cifar10/final_model \
        --num_images 16 \
        --num_steps 2 4 10 25 50 100 \
        --output_dir step_comparison
"""

import argparse
import math
import os
import sys
import time

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import torch
from diffusers import DDPMScheduler, UNet2DModel
from PIL import Image

from shared.aligner import Diff2FlowAligner
from shared.args import add_common_args, add_inference_args, add_lora_args, add_diffusion_args
from shared.lora import apply_lora
from shared.utils import tensor_to_pil


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
                        help="Path to Diff2Flow checkpoint directory (saved with save_pretrained)")
    add_inference_args(parser, output_dir_default="diff2flow_samples", num_steps_default=[50])
    add_lora_args(parser)
    add_diffusion_args(parser)
    add_common_args(parser)

    return parser.parse_args()


def load_model(args, device):
    """Load the Diff2Flow checkpoint with from_pretrained."""

    ckpt_dir = args.checkpoint_path

    # Load training metadata (args, epoch, global_step)
    state_path = os.path.join(ckpt_dir, "training_state.pt")
    training_state = {}
    if os.path.exists(state_path):
        training_state = torch.load(state_path, map_location="cpu", weights_only=False)
    ckpt_args = training_state.get("args", {})

    # Apply LoRA before loading weights if the checkpoint was trained with it
    use_lora = args.use_lora or ckpt_args.get("use_lora", False)
    lora_rank = args.lora_rank or ckpt_args.get("lora_rank", 64)

    model = UNet2DModel.from_pretrained(ckpt_dir)

    if use_lora:
        print(f"Applying LoRA (rank={lora_rank}) to match training config")
        apply_lora(model, rank=lora_rank)

    model = model.to(device)
    model.eval()

    epoch = training_state.get("epoch", "?")
    step = training_state.get("global_step", "?")
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
