"""Shared evaluation utilities for metrics_* scripts.

Provides:
  - prepare_real_images  — save dataset images for FID
  - compute_fid          — clean-fid wrapper
  - generate_and_save_samples — timed batch generation + PNG saving
  - print_summary        — formatted results table
  - save_results         — JSON + CSV output
"""

import json
import os
import time

import torch
from PIL import Image
from tqdm import tqdm

from shared.utils import tensor_to_pil


def prepare_real_images(args, output_dir):
    """Save real dataset images to a folder for FID computation."""
    real_dir = os.path.join(output_dir, "real_images")

    if os.path.exists(real_dir) and len(os.listdir(real_dir)) >= args.num_samples:
        print(f"Real images already cached at {real_dir} ({len(os.listdir(real_dir))} images)")
        return real_dir

    os.makedirs(real_dir, exist_ok=True)
    print(f"Saving {args.num_samples} real images to {real_dir}...")

    if args.dataset == "cifar10":
        from torchvision import datasets, transforms
        from shared.datasets import get_data_root
        data_root = get_data_root("cifar10", args.data_root)
        transform = transforms.Compose([transforms.ToTensor()])
        dataset = datasets.CIFAR10(root=data_root, train=True, download=False, transform=transform)

    elif args.dataset == "celeba":
        from torchvision import datasets, transforms
        from shared.datasets import get_data_root
        data_root = get_data_root("celeba", args.data_root)
        transform = transforms.Compose([
            transforms.CenterCrop(178),
            transforms.Resize(args.image_size),
            transforms.ToTensor(),
        ])
        dataset = datasets.CelebA(root=data_root, split="train", download=False, transform=transform)

    elif args.dataset == "custom":
        if args.data_root is None:
            raise ValueError("--data_root must be specified for custom dataset")
        return args.data_root

    num_to_save = min(args.num_samples, len(dataset))
    for i in tqdm(range(num_to_save), desc="Saving real images"):
        img_tensor = dataset[i][0]
        img = Image.fromarray((img_tensor.permute(1, 2, 0).numpy() * 255).astype("uint8"))
        img.save(os.path.join(real_dir, f"{i:06d}.png"))

    print(f"Saved {num_to_save} real images")
    return real_dir


def compute_fid(real_dir, fake_dir):
    from cleanfid import fid
    print("Computing FID using clean-fid...")
    return fid.compute_fid(real_dir, fake_dir)


def generate_and_save_samples(sample_fn, num_samples, num_steps, batch_size, device, seed, output_dir, desc=None):
    """Generate samples and save as individual PNGs. Returns (sample_dir, total_time).

    Args:
        sample_fn: callable(num_samples, device, seed) -> Tensor of shape (N,C,H,W) in [-1,1]
        desc: tqdm description (defaults to "Generating (steps=<num_steps>)")
    """
    if desc is None:
        desc = f"Generating (steps={num_steps})"

    sample_dir = os.path.join(output_dir, f"generated_steps_{num_steps}")
    os.makedirs(sample_dir, exist_ok=True)

    total_time = 0.0
    num_generated = 0
    batch_idx = 0
    pbar = tqdm(total=num_samples, desc=desc)

    while num_generated < num_samples:
        bs = min(batch_size, num_samples - num_generated)
        batch_seed = seed + batch_idx if seed is not None else None

        if device.type == "cuda":
            torch.cuda.synchronize()
        t_start = time.time()

        samples = sample_fn(bs, device, batch_seed)

        if device.type == "cuda":
            torch.cuda.synchronize()
        total_time += time.time() - t_start

        for i, img in enumerate(tensor_to_pil(samples)):
            img.save(os.path.join(sample_dir, f"{num_generated + i:06d}.png"))

        num_generated += bs
        batch_idx += 1
        pbar.update(bs)

    pbar.close()
    return sample_dir, total_time


def print_summary(results):
    """Print a formatted summary table of evaluation results."""
    has_sampler = results and "sampler" in results[0]
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    if has_sampler:
        print(f"{'Sampler':>8} {'Steps (NFE)':>12} {'FID':>10} {'Time/img (s)':>14} {'Img/sec':>10}")
        print(f"{'-'*8} {'-'*12} {'-'*10} {'-'*14} {'-'*10}")
        for r in results:
            print(f"{r['sampler'].upper():>8} {r['num_steps']:>12} {r['fid']:>10.4f} "
                  f"{r['time_per_image_s']:>14.5f} {r['images_per_second']:>10.2f}")
    else:
        print(f"{'Steps (NFE)':>12} {'FID':>10} {'Time/img (s)':>14} {'Img/sec':>10}")
        print(f"{'-'*12} {'-'*10} {'-'*14} {'-'*10}")
        for r in results:
            print(f"{r['num_steps']:>12} {r['fid']:>10.4f} "
                  f"{r['time_per_image_s']:>14.5f} {r['images_per_second']:>10.2f}")


def save_results(results, metadata, output_dir):
    """Save evaluation results to JSON and CSV."""
    results_path = os.path.join(output_dir, "evaluation_results.json")
    with open(results_path, "w") as f:
        json.dump({**metadata, "results": results}, f, indent=2)
    print(f"\nResults saved to {results_path}")

    has_sampler = results and "sampler" in results[0]
    csv_path = os.path.join(output_dir, "evaluation_results.csv")
    with open(csv_path, "w") as f:
        if has_sampler:
            f.write("sampler,steps,nfe,fid,time_per_image,images_per_second\n")
            for r in results:
                f.write(f"{r['sampler']},{r['num_steps']},{r['nfe_per_image']},{r['fid']},"
                        f"{r['time_per_image_s']},{r['images_per_second']}\n")
        else:
            f.write("steps,nfe,fid,time_per_image,images_per_second\n")
            for r in results:
                f.write(f"{r['num_steps']},{r['nfe_per_image']},{r['fid']},"
                        f"{r['time_per_image_s']},{r['images_per_second']}\n")
    print(f"CSV saved to {csv_path}")
