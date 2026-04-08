import argparse
import math
import os
import sys
import time

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import torch
from diffusers import UNet2DModel

from shared.args import add_common_args, add_inference_args
from shared.utils import make_grid, tensor_to_pil


def parse_args():
    parser = argparse.ArgumentParser(description="Run inference with a trained Flow Matching model")

    parser.add_argument("--model_path", type=str, required=True,
                        help="Path to the saved model directory (e.g. flow_matching_cifar10/final_model)")
    add_inference_args(parser, output_dir_default="fm_samples", num_steps_default=[100])
    add_common_args(parser)

    return parser.parse_args()


@torch.no_grad()
def sample_euler(model, num_samples, image_size, num_steps, device, seed=None):
    """Generate images via Euler integration on the FM ODE: t=0 (noise) -> t=1 (data)."""
    if seed is not None:
        torch.manual_seed(seed)

    shape = (num_samples, 3, image_size, image_size)
    x = torch.randn(shape, device=device)

    dt = 1.0 / num_steps
    for i in range(num_steps):
        t = i / num_steps
        t_scaled = torch.full((num_samples,), t * 999.0, device=device)
        v = model(x, t_scaled, return_dict=False)[0]
        x = x + v * dt

    return x.clamp(-1, 1)


def generate_samples(model, num_images, image_size, num_steps, batch_size, device, seed):
    all_samples = []
    remaining = num_images
    batch_idx = 0

    while remaining > 0:
        bs = min(batch_size, remaining)
        batch_seed = seed + batch_idx if seed is not None else None

        samples = sample_euler(
            model,
            num_samples=bs,
            image_size=image_size,
            num_steps=num_steps,
            device=device,
            seed=batch_seed,
        )
        all_samples.append(samples)
        remaining -= bs
        batch_idx += 1

    return torch.cat(all_samples, dim=0)[:num_images]


def main():
    args = parse_args()

    device = torch.device(args.device) if args.device else torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    print(f"Loading model from: {args.model_path}")

    os.makedirs(args.output_dir, exist_ok=True)

    model = UNet2DModel.from_pretrained(args.model_path).to(device)
    model.eval()

    for num_steps in args.num_steps:
        print(f"\n--- Generating {args.num_images} images with {num_steps} Euler steps ---")

        start = time.time()
        samples = generate_samples(
            model,
            num_images=args.num_images,
            image_size=args.image_size,
            num_steps=num_steps,
            batch_size=args.batch_size,
            device=device,
            seed=args.seed,
        )
        elapsed = time.time() - start

        pil_images = tensor_to_pil(samples)

        cols = math.ceil(math.sqrt(len(pil_images)))
        rows = math.ceil(len(pil_images) / cols)
        grid = make_grid(pil_images, rows=rows, cols=cols)
        grid_path = os.path.join(args.output_dir, f"grid_steps_{num_steps}.png")
        grid.save(grid_path)
        print(f"Saved grid: {grid_path}")
        print(f"Time: {elapsed:.2f}s ({elapsed / args.num_images:.3f}s per image)")

        if args.save_individual:
            ind_dir = os.path.join(args.output_dir, f"steps_{num_steps}")
            os.makedirs(ind_dir, exist_ok=True)
            for i, img in enumerate(pil_images):
                img.save(os.path.join(ind_dir, f"{i:04d}.png"))
            print(f"Saved {len(pil_images)} individual images to {ind_dir}/")

    print("\nDone!")


if __name__ == "__main__":
    main()
