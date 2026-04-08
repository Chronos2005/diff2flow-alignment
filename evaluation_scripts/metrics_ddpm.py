"""
DDPM Evaluation Script

Computes FID, NFE (number of function evaluations), and sampling time
for a trained DDPM model across different step counts.

Two samplers are supported:
  - ddim  (default): deterministic DDIM; any step count, NFE = steps
  - ddpm           : ancestral stochastic DDPM; always runs all
                     num_train_timesteps steps (NFE = 1000 by default)

Using DDIM enables fair comparison with FM and Diff2Flow at matched NFE.
Use --sampler ddpm to get the baseline full-chain DDPM number.

Requirements:
    pip install clean-fid

Usage:
    # DDIM at multiple step counts
    python metrics_ddpm.py \
        --model_path ddpm_cifar10/final_model \
        --dataset cifar10 \
        --num_samples 10000 \
        --step_counts 10 25 50 100 250 1000

    # Full ancestral DDPM (single run at 1000 steps)
    python metrics_ddpm.py \
        --model_path ddpm_cifar10/final_model \
        --sampler ddpm \
        --num_samples 10000

    # Quick test
    python metrics_ddpm.py \
        --model_path ddpm_cifar10/final_model \
        --num_samples 1000 \
        --step_counts 50 100
"""

import argparse
import json
import os
import shutil
import sys
import time

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from diffusers import DDIMScheduler, DDPMScheduler, UNet2DModel
from PIL import Image
from tqdm import tqdm

from shared.args import add_common_args, add_dataset_args, add_eval_args, add_diffusion_args, add_metrics_output_args
from shared.utils import tensor_to_pil


def parse_args():
    parser = argparse.ArgumentParser(
        description="DDPM Evaluation: FID, NFE, Timing (DDIM or ancestral DDPM)"
    )

    parser.add_argument("--model_path", type=str, required=True,
                        help="Path to saved DDPM model directory")
    parser.add_argument("--sampler", type=str, default="ddim",
                        choices=["ddim", "ddpm"],
                        help="'ddim' for deterministic variable-step sampling; "
                             "'ddpm' for full ancestral sampling at num_train_timesteps")

    add_dataset_args(parser)
    add_eval_args(parser)
    add_diffusion_args(parser)

    add_metrics_output_args(parser,
                            output_dir_default="ddpm_eval",
                            scratch_dir_default="/scratch/ram1g23/ddpm_eval_tmp",
                            step_counts_default=[10, 25, 50, 100, 250, 1000])

    add_common_args(parser)
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Real data preparation
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# Samplers
# ---------------------------------------------------------------------------

@torch.no_grad()
def sample_ddim(model, scheduler, num_samples, image_size, num_steps, device, seed=None):
    """Deterministic DDIM sampling with `num_steps` denoising steps."""
    if seed is not None:
        torch.manual_seed(seed)

    scheduler.set_timesteps(num_steps)
    x = torch.randn(num_samples, 3, image_size, image_size, device=device)

    for t in scheduler.timesteps:
        t_batch = t.expand(num_samples).to(device)
        eps_pred = model(x, t_batch, return_dict=False)[0]
        x = scheduler.step(eps_pred, t, x).prev_sample

    return x.clamp(-1, 1)


@torch.no_grad()
def sample_ddpm_ancestral(model, scheduler, num_samples, image_size, device, seed=None):
    """Full ancestral DDPM sampling over all num_train_timesteps steps."""
    if seed is not None:
        torch.manual_seed(seed)

    scheduler.set_timesteps(scheduler.config.num_train_timesteps)
    x = torch.randn(num_samples, 3, image_size, image_size, device=device)

    for t in scheduler.timesteps:
        t_batch = t.expand(num_samples).to(device)
        eps_pred = model(x, t_batch, return_dict=False)[0]
        x = scheduler.step(eps_pred, t, x).prev_sample

    return x.clamp(-1, 1)


# ---------------------------------------------------------------------------
# Sample generation
# ---------------------------------------------------------------------------

def generate_and_save_samples(model, scheduler, num_samples, image_size, num_steps,
                               batch_size, device, seed, output_dir, sampler):
    """Generate samples and save as individual PNGs. Returns (sample_dir, total_time)."""
    sample_dir = os.path.join(output_dir, f"generated_steps_{num_steps}")
    os.makedirs(sample_dir, exist_ok=True)

    total_time = 0.0
    num_generated = 0
    pbar = tqdm(total=num_samples, desc=f"Generating ({sampler}, steps={num_steps})")

    batch_idx = 0
    while num_generated < num_samples:
        bs = min(batch_size, num_samples - num_generated)
        batch_seed = seed + batch_idx if seed is not None else None

        if device.type == "cuda":
            torch.cuda.synchronize()
        t_start = time.time()

        if sampler == "ddim":
            samples = sample_ddim(
                model, scheduler,
                num_samples=bs,
                image_size=image_size,
                num_steps=num_steps,
                device=device,
                seed=batch_seed,
            )
        else:  # ddpm
            samples = sample_ddpm_ancestral(
                model, scheduler,
                num_samples=bs,
                image_size=image_size,
                device=device,
                seed=batch_seed,
            )

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


# ---------------------------------------------------------------------------
# FID computation
# ---------------------------------------------------------------------------

def compute_fid(real_dir, fake_dir):
    from cleanfid import fid
    print("Computing FID using clean-fid...")
    return fid.compute_fid(real_dir, fake_dir)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args = parse_args()

    device = torch.device(args.device) if args.device else \
        torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    print(f"Sampler: {args.sampler.upper()}")

    os.makedirs(args.output_dir, exist_ok=True)

    scratch_dir = args.scratch_dir
    os.makedirs(scratch_dir, exist_ok=True)
    print(f"Scratch dir (images): {scratch_dir}")

    print(f"Loading model from {args.model_path}...")
    model = UNet2DModel.from_pretrained(args.model_path).to(device)
    model.eval()

    if args.sampler == "ddim":
        scheduler = DDIMScheduler.from_pretrained(
            args.model_path,
            num_train_timesteps=args.num_train_timesteps,
        )
        step_counts = sorted(args.step_counts)
    else:
        scheduler = DDPMScheduler.from_pretrained(
            args.model_path,
            num_train_timesteps=args.num_train_timesteps,
        )
        # Full DDPM always runs num_train_timesteps steps
        step_counts = [args.num_train_timesteps]

    real_dir = prepare_real_images(args, scratch_dir)

    results = []

    for num_steps in step_counts:
        print(f"\n{'='*60}")
        print(f"Evaluating: {num_steps} steps  (NFE = {num_steps}, sampler = {args.sampler.upper()})")
        print(f"{'='*60}")

        sample_dir, total_time = generate_and_save_samples(
            model, scheduler,
            num_samples=args.num_samples,
            image_size=args.image_size,
            num_steps=num_steps,
            batch_size=args.batch_size,
            device=device,
            seed=args.seed,
            output_dir=scratch_dir,
            sampler=args.sampler,
        )

        fid_score = compute_fid(real_dir, sample_dir)

        time_per_image = total_time / args.num_samples
        images_per_second = args.num_samples / total_time

        result = {
            "sampler": args.sampler,
            "num_steps": num_steps,
            "nfe_per_image": num_steps,
            "fid": round(fid_score, 4),
            "total_time_s": round(total_time, 2),
            "time_per_image_s": round(time_per_image, 5),
            "images_per_second": round(images_per_second, 2),
            "num_samples": args.num_samples,
        }
        results.append(result)

        print(f"\n  Sampler:     {args.sampler.upper()}")
        print(f"  Steps (NFE): {num_steps}")
        print(f"  FID:         {fid_score:.4f}")
        print(f"  Time/image:  {time_per_image:.5f}s")
        print(f"  Images/sec:  {images_per_second:.2f}")

        if not args.keep_samples:
            shutil.rmtree(sample_dir)

    # Summary
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    print(f"{'Sampler':>8} {'Steps (NFE)':>12} {'FID':>10} {'Time/img (s)':>14} {'Img/sec':>10}")
    print(f"{'-'*8} {'-'*12} {'-'*10} {'-'*14} {'-'*10}")
    for r in results:
        print(f"{r['sampler'].upper():>8} {r['num_steps']:>12} {r['fid']:>10.4f} "
              f"{r['time_per_image_s']:>14.5f} {r['images_per_second']:>10.2f}")

    results_path = os.path.join(args.output_dir, "evaluation_results.json")
    with open(results_path, "w") as f:
        json.dump({
            "model_path": args.model_path,
            "sampler": args.sampler,
            "num_samples": args.num_samples,
            "image_size": args.image_size,
            "seed": args.seed,
            "device": str(device),
            "results": results,
        }, f, indent=2)
    print(f"\nResults saved to {results_path}")

    csv_path = os.path.join(args.output_dir, "evaluation_results.csv")
    with open(csv_path, "w") as f:
        f.write("sampler,steps,nfe,fid,time_per_image,images_per_second\n")
        for r in results:
            f.write(f"{r['sampler']},{r['num_steps']},{r['nfe_per_image']},{r['fid']},"
                    f"{r['time_per_image_s']},{r['images_per_second']}\n")
    print(f"CSV saved to {csv_path}")

    if not args.keep_samples:
        print(f"Cleaning up scratch dir: {scratch_dir}")
        shutil.rmtree(scratch_dir, ignore_errors=True)


if __name__ == "__main__":
    main()
