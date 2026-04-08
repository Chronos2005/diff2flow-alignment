"""
Diff2Flow Evaluation Script

Computes FID, NFE (number of function evaluations), and sampling time
for a Diff2Flow-finetuned model across different step counts.

Generates a batch of samples, saves them, then computes FID against
real data (CIFAR-10 train set or a folder of real images).

Requirements:
    pip install clean-fid

Usage:
    # Evaluate at multiple step counts
    python metrics_diff2flow.py \
        --checkpoint_path diff2flow_cifar10/final_model/diff2flow_final.safetensors \
        --pretrained_model_path ddpm_cifar10/final_model \
        --dataset cifar10 \
        --num_samples 10000 \
        --step_counts 2 4 10 25 50 100

    # Quick test with fewer samples
    python metrics_diff2flow.py \
        --checkpoint_path diff2flow_cifar10/final_model/diff2flow_final.safetensors \
        --pretrained_model_path ddpm_cifar10/final_model \
        --dataset cifar10 \
        --num_samples 1000 \
        --step_counts 10 50
"""

import argparse
import os
import shutil
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from diffusers import DDPMScheduler, UNet2DModel

from shared.aligner import Diff2FlowAligner
from shared.args import add_common_args, add_dataset_args, add_eval_args, add_lora_args, add_diffusion_args, add_metrics_output_args, add_alignment_args
from shared.evaluation import (
    prepare_real_images, compute_fid, generate_and_save_samples, print_summary, save_results
)
from shared.lora import apply_lora
from inference_scripts.inference_diff2flow import sample_euler, sample_heun


def parse_args():
    parser = argparse.ArgumentParser(description="Diff2Flow Evaluation: FID, NFE, Timing")

    # Model
    parser.add_argument("--checkpoint_path", type=str, required=True,
                        help="Path to Diff2Flow checkpoint (.safetensors file)")
    parser.add_argument("--pretrained_model_path", type=str, required=True,
                        help="Path to original DDPM model (for architecture)")
    add_lora_args(parser)

    # Evaluation settings
    add_dataset_args(parser)
    add_eval_args(parser)
    add_diffusion_args(parser)
    add_alignment_args(parser)

    add_metrics_output_args(parser,
                            output_dir_default="diff2flow_eval",
                            scratch_dir_default="/scratch/ram1g23/diff2flow_eval_tmp",
                            step_counts_default=[2, 4, 10, 25, 50, 100])

    parser.add_argument("--solver", type=str, default="euler",
                        choices=["euler", "heun"],
                        help="ODE solver for sampling. Heun uses 2x NFE per step.")

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

    # Load model
    print("Loading model...")
    from safetensors.torch import load_file as load_safetensors

    model = UNet2DModel.from_pretrained(args.pretrained_model_path)

    if args.use_lora:
        print(f"Applying LoRA (rank={args.lora_rank}, placement={args.lora_placement})")
        apply_lora(model, rank=args.lora_rank, placement=args.lora_placement)

    state_dict = load_safetensors(args.checkpoint_path, device="cpu")
    model.load_state_dict(state_dict)
    model = model.to(device)
    model.eval()
    print(f"Loaded checkpoint: {args.checkpoint_path}")

    # Build aligner (must match the flags used during training)
    noise_scheduler = DDPMScheduler(num_train_timesteps=args.num_train_timesteps)
    aligner = Diff2FlowAligner(
        noise_scheduler,
        use_timestep_rescaling=not args.no_timestep_rescaling,
        use_interpolant_rescaling=not args.no_interpolant_rescaling,
        use_velocity_translation=not args.no_velocity_translation,
    )

    sampler = sample_heun if args.solver == "heun" else sample_euler
    print(f"Solver: {args.solver}")

    real_dir = prepare_real_images(args, scratch_dir)

    results = []

    for num_steps in sorted(args.step_counts):
        print(f"\n{'='*60}")
        print(f"Evaluating: {num_steps} Euler steps  (NFE = {num_steps})")
        print(f"{'='*60}")

        sample_fn = lambda n, dev, s: sampler(model, aligner, n, args.image_size, num_steps, dev, s)
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

        nfe = num_steps * (2 if args.solver == "heun" else 1)
        result = {
            "num_steps": num_steps,
            "nfe_per_image": nfe,
            "solver": args.solver,
            "fid": round(fid_score, 4),
            "total_time_s": round(total_time, 2),
            "time_per_image_s": round(time_per_image, 5),
            "images_per_second": round(images_per_second, 2),
            "num_samples": args.num_samples,
        }
        results.append(result)

        print(f"\n  Steps: {num_steps}  (NFE: {nfe})")
        print(f"  FID:         {fid_score:.4f}")
        print(f"  Time/image:  {time_per_image:.5f}s")
        print(f"  Images/sec:  {images_per_second:.2f}")

    print_summary(results)
    save_results(results, {
        "model_checkpoint": args.checkpoint_path,
        "pretrained_model_path": args.pretrained_model_path,
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
