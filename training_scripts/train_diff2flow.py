"""
Diff2Flow: Finetune a pretrained DDPM using Flow Matching via Diffusion Model Alignment.

This script loads a pretrained DDPM (epsilon-parameterized) checkpoint and finetunes it
using the Diff2Flow framework, which aligns diffusion and flow matching trajectories
through timestep remapping, interpolant rescaling, and velocity derivation — all done
analytically in the forward pass with zero extra learnable parameters.

Reference: Schusterbauer et al., "Diff2Flow: Training Flow Matching Models via Diffusion
Model Alignment", arXiv:2506.02221, 2025.
"""

import argparse
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import torch
import torch.nn.functional as F
from diffusers import DDPMScheduler, UNet2DModel
from diffusers.optimization import get_cosine_schedule_with_warmup
from torch.utils.data import DataLoader
from tqdm import tqdm
from accelerate import Accelerator

from shared.aligner import Diff2FlowAligner
from shared.args import add_dataset_args, add_training_args, add_lora_args, add_diffusion_args
from shared.lora import apply_lora
from shared.datasets import DATASET_DEFAULTS, get_dataset, get_data_root
from shared.utils import make_grid, tensor_to_pil

OUTPUT_DIRS = {"cifar10": "diff2flow_cifar10", "celeba": "diff2flow_celeba"}


# ---------------------------------------------------------------------------
# FM Euler sampler
# ---------------------------------------------------------------------------

@torch.no_grad()
def sample_euler(model, aligner, num_samples, image_size, num_steps, device):
    """
    Sample from the Diff2Flow model using forward Euler integration on the FM ODE.

    Starting from x_FM(t=0) = noise, integrate to x_FM(t=1) = data.
    At each step, we remap to diffusion space, run the model, derive velocity,
    and take an Euler step.
    """
    shape = (num_samples, 3, image_size, image_size)
    x = torch.randn(shape, device=device)  # x_FM at t=0 (pure noise)

    dt = 1.0 / num_steps
    for i in range(num_steps):
        t_fm = torch.full((num_samples,), i * dt, device=device)

        # 1) Remap t_FM -> t_DM
        t_dm = aligner.t_fm_to_t_dm(t_fm)

        # 2) Remap x_FM -> x_DM
        alpha_t, sigma_t = aligner.get_alpha_sigma(t_dm)
        x_dm = aligner.x_fm_to_x_dm(x, alpha_t, sigma_t)

        # 3) Run the diffusion model (expects integer-ish timesteps)
        t_dm_input = t_dm.long().clamp(0, aligner.T - 1)
        eps_pred = model(x_dm, t_dm_input, return_dict=False)[0]

        # 4) Derive FM velocity from eps prediction
        velocity = aligner.eps_to_velocity(eps_pred, x_dm, alpha_t, sigma_t)

        # 5) Euler step in FM space
        x = x + dt * velocity

    # Clamp to valid range
    x = x.clamp(-1, 1)
    return x


# ---------------------------------------------------------------------------
# Main training loop
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(description="Diff2Flow: Finetune a DDPM with Flow Matching alignment")

    parser.add_argument("--pretrained_model_path", type=str, required=True,
                        help="Path to pretrained DDPM model directory (saved via save_pretrained)")
    add_dataset_args(parser, include_custom=False, dataset_required=True)
    # learning_rate_default=1e-5 (lower than pretraining since we're finetuning)
    add_training_args(parser, learning_rate_default=1e-5)
    add_lora_args(parser)
    add_diffusion_args(parser)

    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument("--image_size", type=int, default=None)
    parser.add_argument("--train_batch_size", type=int, default=None)
    parser.add_argument("--num_warmup_steps", type=int, default=200)
    parser.add_argument("--num_inference_steps", type=int, default=50,
                        help="Number of Euler steps for FM sampling (can be much fewer than DDPM)")

    return parser.parse_args()


def main():
    args = parse_args()

    # Fill defaults
    defaults = DATASET_DEFAULTS[args.dataset]
    for key, val in defaults.items():
        if getattr(args, key) is None:
            setattr(args, key, val)

    if args.output_dir is None:
        args.output_dir = OUTPUT_DIRS[args.dataset]

    args.data_root = get_data_root(args.dataset, args.data_root)

    accelerator = Accelerator()
    device = accelerator.device

    if accelerator.is_main_process:
        os.makedirs(args.output_dir, exist_ok=True)
        os.makedirs(f"{args.output_dir}/samples", exist_ok=True)

    # -----------------------------------------------------------------------
    # 1) Load pretrained DDPM
    # -----------------------------------------------------------------------
    if accelerator.is_main_process:
        print(f"Loading pretrained DDPM from: {args.pretrained_model_path}")
    model = UNet2DModel.from_pretrained(args.pretrained_model_path)

    # Build the DDPM noise scheduler (needed for alpha/sigma schedule)
    noise_scheduler = DDPMScheduler(num_train_timesteps=args.num_train_timesteps)

    # -----------------------------------------------------------------------
    # 2) Build the Diff2Flow aligner
    # -----------------------------------------------------------------------
    aligner = Diff2FlowAligner(noise_scheduler)

    # -----------------------------------------------------------------------
    # 3) Optionally apply LoRA
    # -----------------------------------------------------------------------
    if args.use_lora:
        num_replaced = apply_lora(model, rank=args.lora_rank)
        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        total = sum(p.numel() for p in model.parameters())
        if accelerator.is_main_process:
            print(f"LoRA applied: {num_replaced} layers replaced (rank={args.lora_rank})")
            print(f"Trainable params: {trainable/1e6:.2f}M / {total/1e6:.2f}M "
                  f"({100*trainable/total:.1f}%)")
    else:
        if accelerator.is_main_process:
            total = sum(p.numel() for p in model.parameters())
            print(f"Full finetuning: {total/1e6:.2f}M parameters")

    # -----------------------------------------------------------------------
    # 4) Dataset & dataloader
    # -----------------------------------------------------------------------
    dataset = get_dataset(args.dataset, args.data_root, args.image_size)
    dataloader = DataLoader(
        dataset,
        batch_size=args.train_batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True,
    )

    if accelerator.is_main_process:
        print(f"Dataset: {args.dataset} ({len(dataset):,} images)")
        print(f"Training on: {device} ({accelerator.num_processes} GPU(s))")
        print(f"Epochs: {args.num_epochs}, Batch size: {args.train_batch_size}")
        print(f"FM inference steps: {args.num_inference_steps}")

    # -----------------------------------------------------------------------
    # 5) Optimizer & scheduler
    # -----------------------------------------------------------------------
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable_params, lr=args.learning_rate)
    lr_scheduler = get_cosine_schedule_with_warmup(
        optimizer=optimizer,
        num_warmup_steps=args.num_warmup_steps,
        num_training_steps=len(dataloader) * args.num_epochs,
    )

    model, optimizer, dataloader, lr_scheduler = accelerator.prepare(
        model, optimizer, dataloader, lr_scheduler
    )

    # -----------------------------------------------------------------------
    # 6) Training loop — Diff2Flow (Algorithm 1 from the paper)
    # -----------------------------------------------------------------------
    global_step = 0

    for epoch in range(args.num_epochs):
        model.train()
        epoch_loss = 0.0
        progress_bar = tqdm(
            dataloader,
            desc=f"Epoch {epoch+1}/{args.num_epochs}",
            disable=not accelerator.is_main_process,
        )

        for batch in progress_bar:
            images = batch[0]  # x1 in FM convention (data)
            bs = images.shape[0]

            # --- Step 1: Sample FM timesteps and build FM interpolant ---
            t_fm = torch.rand(bs, device=images.device)
            noise = torch.randn_like(images)  # x_FM_0

            # FM interpolant: x_FM_t = t * x1 + (1-t) * x0   (Eq. 4)
            t_expanded = t_fm.view(bs, 1, 1, 1)
            x_fm_t = t_expanded * images + (1 - t_expanded) * noise

            # FM ground-truth velocity: v = x1 - x0   (Eq. 5)
            velocity_target = images - noise

            # --- Step 2: Remap t_FM -> t_DM  (Eq. 12) ---
            t_dm = aligner.t_fm_to_t_dm(t_fm)

            # --- Step 3: Remap x_FM -> x_DM  (Eq. 13) ---
            alpha_t, sigma_t = aligner.get_alpha_sigma(t_dm)
            x_dm_t = aligner.x_fm_to_x_dm(x_fm_t, alpha_t, sigma_t)

            # --- Step 4: Run the diffusion model and derive velocity ---
            t_dm_input = t_dm.long().clamp(0, aligner.T - 1)
            eps_pred = model(x_dm_t, t_dm_input, return_dict=False)[0]

            # Convert eps prediction to FM velocity estimate
            velocity_pred = aligner.eps_to_velocity(eps_pred, x_dm_t, alpha_t, sigma_t)

            # --- Step 5: FM loss  (Eq. 5) ---
            loss = F.mse_loss(velocity_pred, velocity_target)

            optimizer.zero_grad()
            accelerator.backward(loss)
            torch.nn.utils.clip_grad_norm_(
                [p for p in model.parameters() if p.requires_grad], 1.0
            )
            optimizer.step()
            lr_scheduler.step()

            epoch_loss += loss.item()
            progress_bar.set_postfix({"loss": f"{loss.item():.4f}"})
            global_step += 1

        if accelerator.is_main_process:
            avg_loss = epoch_loss / len(dataloader)
            print(f"Epoch {epoch+1} avg loss: {avg_loss:.4f}")

        # --- Sample images using FM Euler sampling ---
        if (epoch + 1) % args.save_images_every == 0 and accelerator.is_main_process:
            unwrapped = accelerator.unwrap_model(model)
            unwrapped.eval()
            with torch.no_grad():
                samples = sample_euler(
                    unwrapped, aligner,
                    num_samples=16,
                    image_size=args.image_size,
                    num_steps=args.num_inference_steps,
                    device=device,
                )
                pil_images = tensor_to_pil(samples)
                grid = make_grid(pil_images, rows=4, cols=4)
                save_path = f"{args.output_dir}/samples/epoch_{epoch+1}.png"
                grid.save(save_path)
                print(f"Saved samples -> {save_path}")
            model.train()

        # --- Save checkpoint ---
        if (epoch + 1) % args.save_model_every == 0 and accelerator.is_main_process:
            unwrapped = accelerator.unwrap_model(model)
            ckpt_dir = f"{args.output_dir}/checkpoint_epoch_{epoch+1}"
            os.makedirs(ckpt_dir, exist_ok=True)
            unwrapped.save_pretrained(ckpt_dir)
            torch.save({
                "epoch": epoch + 1,
                "global_step": global_step,
                "args": vars(args),
            }, os.path.join(ckpt_dir, "training_state.pt"))
            print(f"Saved checkpoint -> {ckpt_dir}")

    # --- Save final model ---
    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        unwrapped = accelerator.unwrap_model(model)
        final_dir = f"{args.output_dir}/final_model"
        os.makedirs(final_dir, exist_ok=True)
        unwrapped.save_pretrained(final_dir)
        torch.save({
            "epoch": args.num_epochs,
            "global_step": global_step,
            "args": vars(args),
        }, os.path.join(final_dir, "training_state.pt"))
        print(f"Training complete! Final model saved to {final_dir}")


if __name__ == "__main__":
    main()
