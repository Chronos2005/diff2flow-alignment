"""
Flow Matching Evaluation Script

Computes FID, NFE (number of function evaluations), and sampling time
for a trained Flow Matching model across different Euler step counts.

Generates a batch of samples, saves them, then computes FID against
real data (CIFAR-10 train set or a folder of real images).

Requirements:
    pip install clean-fid

Usage:
    python metrics_fm.py \
        --model_path flow_matching_cifar10/final_model \
        --dataset cifar10 \
        --num_samples 10000 \
        --step_counts 2 4 10 25 50 100

    # Quick test
    python metrics_fm.py \
        --model_path flow_matching_cifar10/final_model \
        --num_samples 1000 \
        --step_counts 10 50
"""

import argparse
import os
import shutil
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from diffusers import UNet2DModel

from shared.args import add_common_args, add_dataset_args, add_eval_args, add_metrics_output_args
from shared.evaluation import (
    prepare_real_images, compute_fid, generate_and_save_samples, print_summary, save_results
)
from inference_scripts.inference_fm import sample_euler


def parse_args():
    parser = argparse.ArgumentParser(description="Flow Matching Evaluation: FID, NFE, Timing")

    parser.add_argument("--model_path", type=str, required=True,
                        help="Path to saved FM model directory")

    add_dataset_args(parser)
    add_eval_args(parser)

    add_metrics_output_args(parser,
                            output_dir_default="fm_eval",
                            scratch_dir_default="/scratch/ram1g23/fm_eval_tmp",
                            step_counts_default=[2, 4, 10, 25, 50, 100])

    add_common_args(parser)
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args = parse_args()

    device = torch.device(args.device) if args.device else \
        torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    os.makedirs(args.output_dir, exist_ok=True)

    scratch_dir = args.scratch_dir
    os.makedirs(scratch_dir, exist_ok=True)
    print(f"Scratch dir (images): {scratch_dir}")

    print(f"Loading model from {args.model_path}...")
    model = UNet2DModel.from_pretrained(args.model_path).to(device)
    model.eval()

    real_dir = prepare_real_images(args, scratch_dir)

    results = []

    for num_steps in sorted(args.step_counts):
        print(f"\n{'='*60}")
        print(f"Evaluating: {num_steps} Euler steps  (NFE = {num_steps})")
        print(f"{'='*60}")

        sample_fn = lambda n, dev, s: sample_euler(model, n, args.image_size, num_steps, dev, s)
        sample_dir, total_time = generate_and_save_samples(
            sample_fn,
            num_samples=args.num_samples,
            num_steps=num_steps,
            batch_size=args.batch_size,
            device=device,
            seed=args.seed,
            output_dir=scratch_dir,
        )

        fid_score = compute_fid(real_dir, sample_dir)

        time_per_image = total_time / args.num_samples
        images_per_second = args.num_samples / total_time

        result = {
            "num_steps": num_steps,
            "nfe_per_image": num_steps,
            "fid": round(fid_score, 4),
            "total_time_s": round(total_time, 2),
            "time_per_image_s": round(time_per_image, 5),
            "images_per_second": round(images_per_second, 2),
            "num_samples": args.num_samples,
        }
        results.append(result)

        print(f"\n  Steps (NFE): {num_steps}")
        print(f"  FID:         {fid_score:.4f}")
        print(f"  Time/image:  {time_per_image:.5f}s")
        print(f"  Images/sec:  {images_per_second:.2f}")

        if not args.keep_samples:
            shutil.rmtree(sample_dir)

    print_summary(results)
    save_results(results, {
        "model_path": args.model_path,
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
