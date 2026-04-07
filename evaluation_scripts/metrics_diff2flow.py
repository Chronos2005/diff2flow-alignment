"""
Diff2Flow Evaluation Script

Computes FID, NFE (number of function evaluations), and sampling time
for a Diff2Flow-finetuned model across different step counts.

Generates a batch of samples, saves them, then computes FID against
real data (CIFAR-10 train set or a folder of real images).

Requirements:
    pip install torch-fidelity   (or)   pip install clean-fid

Usage:
    # Evaluate at multiple step counts
    python diff2flow_evaluate.py \
        --checkpoint_path diff2flow_cifar10/final_model/diff2flow_final.pt \
        --pretrained_model_path ddpm_cifar10/final_model \
        --dataset cifar10 \
        --num_samples 10000 \
        --step_counts 2 4 10 25 50 100

    # Quick test with fewer samples
    python diff2flow_evaluate.py \
        --checkpoint_path diff2flow_cifar10/final_model/diff2flow_final.pt \
        --pretrained_model_path ddpm_cifar10/final_model \
        --dataset cifar10 \
        --num_samples 1000 \
        --step_counts 10 50
"""

import argparse
import json
import math
import os
import shutil
import time

import torch
import numpy as np
from diffusers import DDPMScheduler, UNet2DModel
from PIL import Image
from tqdm import tqdm

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import the aligner and LoRA utilities from the inference script
from inference_scripts.inference_diff2flow import (
    Diff2FlowAligner,
    apply_lora,
    sample_euler,
    tensor_to_pil,
    make_grid,
)


def parse_args():
    parser = argparse.ArgumentParser(description="Diff2Flow Evaluation: FID, NFE, Timing")

    # Model
    parser.add_argument("--checkpoint_path", type=str, required=True,
                        help="Path to Diff2Flow checkpoint (.safetensors file)")
    parser.add_argument("--pretrained_model_path", type=str, default=None,
                        help="Path to original DDPM model (for architecture)")
    parser.add_argument("--use_lora", action="store_true")
    parser.add_argument("--lora_rank", type=int, default=64)

    # Evaluation settings
    parser.add_argument("--dataset", type=str, default="cifar10", choices=["cifar10", "celeba", "custom"])
    parser.add_argument("--data_root", type=str, default=None,
                        help="Dataset root (for cifar10/celeba) or folder of real images (for custom)")
    parser.add_argument("--num_samples", type=int, default=10000,
                        help="Number of samples to generate for FID (10k-50k recommended)")
    parser.add_argument("--step_counts", type=int, nargs="+", default=[2, 4, 10, 25, 50, 100],
                        help="List of Euler step counts to evaluate")
    parser.add_argument("--batch_size", type=int, default=128,
                        help="Batch size for generation")
    parser.add_argument("--image_size", type=int, default=32)
    parser.add_argument("--num_train_timesteps", type=int, default=1000)

    # Output
    parser.add_argument("--output_dir", type=str, default="diff2flow_eval")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--keep_samples", action="store_true",
                        help="Keep generated sample folders after FID computation")
    parser.add_argument("--fid_backend", type=str, default="auto",
                        choices=["auto", "torch_fidelity", "clean_fid"],
                        help="Which library to use for FID computation")

    return parser.parse_args()


# ---------------------------------------------------------------------------
# Real data preparation
# ---------------------------------------------------------------------------

def prepare_real_images(args, output_dir):
    """
    Save real dataset images to a folder for FID computation.
    Returns the path to the folder of real images.
    """
    real_dir = os.path.join(output_dir, "real_images")

    if os.path.exists(real_dir) and len(os.listdir(real_dir)) >= args.num_samples:
        print(f"Real images already cached at {real_dir} ({len(os.listdir(real_dir))} images)")
        return real_dir

    os.makedirs(real_dir, exist_ok=True)
    print(f"Saving {args.num_samples} real images to {real_dir}...")

    if args.dataset == "cifar10":
        from torchvision import datasets, transforms
        transform = transforms.Compose([
            transforms.ToTensor(),
        ])
        data_root = args.data_root
        if data_root is None:
            try:
                from dataset_download_scripts.cifar import CIFAR10_ROOT
                data_root = CIFAR10_ROOT
            except ImportError:
                data_root = "./data"
        dataset = datasets.CIFAR10(root=data_root, train=True, download=False, transform=transform)

    elif args.dataset == "celeba":
        from torchvision import datasets, transforms
        transform = transforms.Compose([
            transforms.CenterCrop(178),
            transforms.Resize(args.image_size),
            transforms.ToTensor(),
        ])
        data_root = args.data_root
        if data_root is None:
            try:
                from dataset_download_scripts.celebA import CELEBA_ROOT
                data_root = CELEBA_ROOT
            except ImportError:
                data_root = "./data"
        dataset = datasets.CelebA(root=data_root, split="train", download=False, transform=transform)

    elif args.dataset == "custom":
        if args.data_root is None:
            raise ValueError("--data_root must be specified for custom dataset")
        return args.data_root  # assume it's already a folder of images

    num_to_save = min(args.num_samples, len(dataset))
    for i in tqdm(range(num_to_save), desc="Saving real images"):
        img_tensor = dataset[i][0]  # (C, H, W) in [0, 1]
        img = Image.fromarray((img_tensor.permute(1, 2, 0).numpy() * 255).astype("uint8"))
        img.save(os.path.join(real_dir, f"{i:06d}.png"))

    print(f"Saved {num_to_save} real images")
    return real_dir


# ---------------------------------------------------------------------------
# Sample generation
# ---------------------------------------------------------------------------

def generate_and_save_samples(model, aligner, num_samples, image_size, num_steps,
                               batch_size, device, seed, output_dir):
    """
    Generate samples and save them as individual PNGs.
    Returns (sample_dir, total_time, total_nfe).
    """
    sample_dir = os.path.join(output_dir, f"generated_steps_{num_steps}")
    os.makedirs(sample_dir, exist_ok=True)

    total_time = 0.0
    total_nfe = 0
    num_generated = 0

    pbar = tqdm(total=num_samples, desc=f"Generating (steps={num_steps})")

    batch_idx = 0
    while num_generated < num_samples:
        bs = min(batch_size, num_samples - num_generated)
        batch_seed = seed + batch_idx if seed is not None else None

        # Time the sampling
        torch.cuda.synchronize() if device.type == "cuda" else None
        t_start = time.time()

        samples = sample_euler(
            model, aligner,
            num_samples=bs,
            image_size=image_size,
            num_steps=num_steps,
            device=device,
            seed=batch_seed,
        )

        torch.cuda.synchronize() if device.type == "cuda" else None
        t_end = time.time()

        total_time += (t_end - t_start)
        total_nfe += bs * num_steps  # each sample takes num_steps model evaluations

        # Save images
        pil_images = tensor_to_pil(samples)
        for i, img in enumerate(pil_images):
            img.save(os.path.join(sample_dir, f"{num_generated + i:06d}.png"))

        num_generated += bs
        batch_idx += 1
        pbar.update(bs)

    pbar.close()
    return sample_dir, total_time, total_nfe


# ---------------------------------------------------------------------------
# FID computation
# ---------------------------------------------------------------------------

def compute_fid_torch_fidelity(real_dir, fake_dir):
    """Compute FID using torch-fidelity."""
    import torch_fidelity

    metrics = torch_fidelity.calculate_metrics(
        input1=fake_dir,
        input2=real_dir,
        cuda=torch.cuda.is_available(),
        fid=True,
        verbose=False,
    )
    return metrics["frechet_inception_distance"]


def compute_fid_clean_fid(real_dir, fake_dir):
    """Compute FID using clean-fid."""
    from cleanfid import fid

    score = fid.compute_fid(real_dir, fake_dir)
    return score


def compute_fid(real_dir, fake_dir, backend="auto"):
    """Compute FID using the available library."""

    if backend == "auto":
        # Try torch_fidelity first, then clean_fid
        try:
            import torch_fidelity
            backend = "torch_fidelity"
        except ImportError:
            try:
                import cleanfid
                backend = "clean_fid"
            except ImportError:
                raise ImportError(
                    "No FID library found. Install one:\n"
                    "  pip install torch-fidelity\n"
                    "  pip install clean-fid"
                )

    print(f"Computing FID using {backend}...")

    if backend == "torch_fidelity":
        return compute_fid_torch_fidelity(real_dir, fake_dir)
    elif backend == "clean_fid":
        return compute_fid_clean_fid(real_dir, fake_dir)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args = parse_args()

    # Device
    if args.device:
        device = torch.device(args.device)
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    os.makedirs(args.output_dir, exist_ok=True)

    # -----------------------------------------------------------------------
    # Load model
    # -----------------------------------------------------------------------
    print("Loading model...")
    from safetensors.torch import load_file as load_safetensors

    pretrained_path = args.pretrained_model_path
    if pretrained_path is None:
        raise ValueError("Provide --pretrained_model_path")

    model = UNet2DModel.from_pretrained(pretrained_path)

    if args.use_lora:
        print(f"Applying LoRA (rank={args.lora_rank})")
        apply_lora(model, rank=args.lora_rank)

    state_dict = load_safetensors(args.checkpoint_path, device="cpu")
    model.load_state_dict(state_dict)
    model = model.to(device)
    model.eval()

    print(f"Loaded safetensors checkpoint: {args.checkpoint_path}")

    # -----------------------------------------------------------------------
    # Build aligner
    # -----------------------------------------------------------------------
    noise_scheduler = DDPMScheduler(num_train_timesteps=args.num_train_timesteps)
    aligner = Diff2FlowAligner(noise_scheduler)

    # -----------------------------------------------------------------------
    # Prepare real images for FID
    # -----------------------------------------------------------------------
    real_dir = prepare_real_images(args, args.output_dir)

    # -----------------------------------------------------------------------
    # Evaluate across step counts
    # -----------------------------------------------------------------------
    results = []

    for num_steps in sorted(args.step_counts):
        print(f"\n{'='*60}")
        print(f"Evaluating: {num_steps} Euler steps (NFE per sample = {num_steps})")
        print(f"{'='*60}")

        # Generate samples
        sample_dir, total_time, total_nfe = generate_and_save_samples(
            model, aligner,
            num_samples=args.num_samples,
            image_size=args.image_size,
            num_steps=num_steps,
            batch_size=args.batch_size,
            device=device,
            seed=args.seed,
            output_dir=args.output_dir,
        )

        # Compute FID
        fid_score = compute_fid(real_dir, sample_dir, backend=args.fid_backend)

        # Compute timing stats
        time_per_image = total_time / args.num_samples
        nfe_per_image = num_steps  # Euler: 1 model call per step
        images_per_second = args.num_samples / total_time

        result = {
            "num_steps": num_steps,
            "nfe_per_image": nfe_per_image,
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

        # Clean up generated images unless asked to keep them
        if not args.keep_samples:
            shutil.rmtree(sample_dir)

    # -----------------------------------------------------------------------
    # Summary
    # -----------------------------------------------------------------------
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    print(f"{'Steps (NFE)':>12} {'FID':>10} {'Time/img (s)':>14} {'Img/sec':>10}")
    print(f"{'-'*12:>12} {'-'*10:>10} {'-'*14:>14} {'-'*10:>10}")
    for r in results:
        print(f"{r['num_steps']:>12} {r['fid']:>10.4f} {r['time_per_image_s']:>14.5f} {r['images_per_second']:>10.2f}")

    # Save results as JSON
    results_path = os.path.join(args.output_dir, "evaluation_results.json")
    with open(results_path, "w") as f:
        json.dump({
            "model_checkpoint": args.checkpoint_path,
            "num_samples": args.num_samples,
            "image_size": args.image_size,
            "seed": args.seed,
            "device": str(device),
            "results": results,
        }, f, indent=2)
    print(f"\nResults saved to {results_path}")

    # Save results as CSV for easy plotting
    csv_path = os.path.join(args.output_dir, "evaluation_results.csv")
    with open(csv_path, "w") as f:
        f.write("steps,nfe,fid,time_per_image,images_per_second\n")
        for r in results:
            f.write(f"{r['num_steps']},{r['nfe_per_image']},{r['fid']},{r['time_per_image_s']},{r['images_per_second']}\n")
    print(f"CSV saved to {csv_path}")


if __name__ == "__main__":
    main()