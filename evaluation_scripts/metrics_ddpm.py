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
import os
import shutil
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from diffusers import DDIMScheduler, DDPMScheduler, UNet2DModel

from shared.args import add_common_args, add_dataset_args, add_eval_args, add_diffusion_args, add_metrics_output_args
from shared.evaluation import (
    prepare_real_images, compute_fid, generate_and_save_samples, print_summary, save_results
)


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

        if args.sampler == "ddim":
            sample_fn = lambda n, dev, s: sample_ddim(model, scheduler, n, args.image_size, num_steps, dev, s)
        else:
            sample_fn = lambda n, dev, s: sample_ddpm_ancestral(model, scheduler, n, args.image_size, dev, s)
        sample_dir, total_time = generate_and_save_samples(
            sample_fn,
            num_samples=args.num_samples,
            num_steps=num_steps,
            batch_size=args.batch_size,
            device=device,
            seed=args.seed,
            output_dir=scratch_dir,
            desc=f"Generating ({args.sampler}, steps={num_steps})",
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

    print_summary(results)
    save_results(results, {
        "model_path": args.model_path,
        "sampler": args.sampler,
        "num_samples": args.num_samples,
        "image_size": args.image_size,
        "seed": args.seed,
        "device": str(device),
    }, args.output_dir)

    if not args.keep_samples:
        print(f"Cleaning up scratch dir: {scratch_dir}")
        shutil.rmtree(scratch_dir, ignore_errors=True)


if __name__ == "__main__":
    main()
