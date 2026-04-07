import argparse
import math
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import torch
from diffusers import DDPMPipeline, DDPMScheduler, UNet2DModel
from PIL import Image

from shared.utils import make_grid


def parse_args():
    parser = argparse.ArgumentParser(description="Run inference with a trained DDPM model")

    parser.add_argument("--model_path", type=str, required=True,
                        help="Path to the saved model directory (e.g. ddpm_cifar10/final_model)")
    parser.add_argument("--output_dir", type=str, default="inference_output",
                        help="Directory to save generated images")
    parser.add_argument("--num_images", type=int, default=16,
                        help="Number of images to generate")
    parser.add_argument("--num_inference_steps", type=int, default=1000,
                        help="Number of denoising steps (more = better quality, slower)")
    parser.add_argument("--batch_size", type=int, default=16,
                        help="Images to generate per batch")
    parser.add_argument("--seed", type=int, default=None,
                        help="Random seed for reproducibility")

    return parser.parse_args()


@torch.inference_mode()
def generate(pipeline, batch_size, num_inference_steps):
    return pipeline(
        batch_size=batch_size,
        num_inference_steps=num_inference_steps,
    ).images


def main():
    args = parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Running inference on: {device}")
    print(f"Loading model from:   {args.model_path}")

    os.makedirs(args.output_dir, exist_ok=True)

    if args.seed is not None:
        torch.manual_seed(args.seed)
        print(f"Seed: {args.seed}")

    model = UNet2DModel.from_pretrained(args.model_path).to(device)
    noise_scheduler = DDPMScheduler.from_pretrained(args.model_path)


    pipeline = DDPMPipeline(unet=model, scheduler=noise_scheduler).to(device)

    all_images = []
    remaining = args.num_images

    while remaining > 0:
        batch_size = min(args.batch_size, remaining)
        print(f"Generating {batch_size} images ({len(all_images)}/{args.num_images} done)...")

        batch_images = generate(pipeline, batch_size, args.num_inference_steps)
        all_images.extend(batch_images)
        remaining -= batch_size

    for i, img in enumerate(all_images):
        img.save(os.path.join(args.output_dir, f"sample_{i:04d}.png"))

    cols = math.ceil(math.sqrt(len(all_images)))
    rows = math.ceil(len(all_images) / cols)
    grid = make_grid(all_images, rows=rows, cols=cols)
    grid_path = os.path.join(args.output_dir, "grid.png")
    grid.save(grid_path)

    print(f"Saved {len(all_images)} images to {args.output_dir}/")
    print(f"Saved grid → {grid_path}")


if __name__ == "__main__":
    main()
