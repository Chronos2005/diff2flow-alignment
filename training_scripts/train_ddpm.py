import argparse
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import torch
import torch.nn.functional as F
from diffusers import DDPMPipeline, DDPMScheduler, UNet2DModel
from diffusers.optimization import get_cosine_schedule_with_warmup
from torch.utils.data import DataLoader
from tqdm import tqdm
from accelerate import Accelerator

from shared.args import add_dataset_args, add_training_args, add_diffusion_args
from shared.datasets import DATASET_DEFAULTS, get_dataset, get_data_root
from shared.utils import make_grid

OUTPUT_DIRS = {"cifar10": "ddpm_cifar10", "celeba": "ddpm_celeba"}


def parse_args():
    parser = argparse.ArgumentParser(description="Train a DDPM on CIFAR-10 or CelebA")

    add_dataset_args(parser, include_custom=False, dataset_required=True)
    add_training_args(parser)
    add_diffusion_args(parser)

    parser.add_argument("--output_dir", type=str, default=None,
                        help="Output directory (default: ddpm_cifar10 or ddpm_celeba)")
    parser.add_argument("--image_size", type=int, default=None,
                        help="Image size (default: 32 for CIFAR-10, 64 for CelebA)")
    parser.add_argument("--train_batch_size", type=int, default=None,
                        help="Training batch size (default: 128 for CIFAR-10, 64 for CelebA)")
    parser.add_argument("--num_warmup_steps", type=int, default=500)
    parser.add_argument("--num_inference_steps", type=int, default=1000)

    return parser.parse_args()


def build_model(image_size):
    return UNet2DModel(
        sample_size=image_size,
        in_channels=3,
        out_channels=3,
        layers_per_block=2,
        block_out_channels=(128, 128, 256, 256, 512, 512),
        down_block_types=(
            "DownBlock2D",
            "DownBlock2D",
            "DownBlock2D",
            "DownBlock2D",
            "AttnDownBlock2D",
            "DownBlock2D",
        ),
        up_block_types=(
            "UpBlock2D",
            "AttnUpBlock2D",
            "UpBlock2D",
            "UpBlock2D",
            "UpBlock2D",
            "UpBlock2D",
        ),
    )


def main():
    args = parse_args()

    # Fill in dataset-specific defaults for any unset args
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
        print(f"Dataset:      {args.dataset}")
        print(f"Training on:  {device} ({accelerator.num_processes} GPU(s))")
        print(f"Output dir:   {args.output_dir}")

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
        print(f"Dataset size: {len(dataset):,} images")

    model = build_model(args.image_size)

    if accelerator.is_main_process:
        num_params = sum(p.numel() for p in model.parameters()) / 1e6
        print(f"Model parameters: {num_params:.1f}M")

    noise_scheduler = DDPMScheduler(num_train_timesteps=args.num_train_timesteps)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)
    lr_scheduler = get_cosine_schedule_with_warmup(
        optimizer=optimizer,
        num_warmup_steps=args.num_warmup_steps,
        num_training_steps=len(dataloader) * args.num_epochs,
    )

    model, optimizer, dataloader, lr_scheduler = accelerator.prepare(
        model, optimizer, dataloader, lr_scheduler
    )

    global_step = 0

    for epoch in range(args.num_epochs):
        model.train()
        epoch_loss = 0.0
        progress_bar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{args.num_epochs}", disable=not accelerator.is_main_process)

        for batch in progress_bar:
            images = batch[0].to(device)
            batch_size = images.shape[0]

            noise = torch.randn_like(images)
            timesteps = torch.randint(
                0, noise_scheduler.config.num_train_timesteps,
                (batch_size,), device=device,
            ).long()

            noisy_images = noise_scheduler.add_noise(images, noise, timesteps)
            noise_pred = model(noisy_images, timesteps, return_dict=False)[0]
            loss = F.mse_loss(noise_pred, noise)

            optimizer.zero_grad()
            accelerator.backward(loss)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            lr_scheduler.step()

            epoch_loss += loss.item()
            progress_bar.set_postfix({"loss": f"{loss.item():.4f}"})
            global_step += 1

        if accelerator.is_main_process:
            avg_loss = epoch_loss / len(dataloader)
            print(f"Epoch {epoch+1} avg loss: {avg_loss:.4f}")

        if (epoch + 1) % args.save_images_every == 0 and accelerator.is_main_process:
            unwrapped = accelerator.unwrap_model(model)
            unwrapped.eval()
            with torch.no_grad():
                pipeline = DDPMPipeline(unet=unwrapped, scheduler=noise_scheduler)
                sample_images = pipeline(
                    batch_size=16,
                    num_inference_steps=args.num_inference_steps,
                ).images
                grid = make_grid(sample_images, rows=4, cols=4)
                save_path = f"{args.output_dir}/samples/epoch_{epoch+1}.png"
                grid.save(save_path)
                print(f"Saved samples → {save_path}")
            model.train()

        if (epoch + 1) % args.save_model_every == 0 and accelerator.is_main_process:
            unwrapped = accelerator.unwrap_model(model)
            checkpoint_dir = f"{args.output_dir}/checkpoint_epoch_{epoch+1}"
            unwrapped.save_pretrained(checkpoint_dir)
            print(f"Saved checkpoint → {checkpoint_dir}")

    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        unwrapped = accelerator.unwrap_model(model)
        unwrapped.save_pretrained(f"{args.output_dir}/final_model")
        print("Training complete!")


if __name__ == "__main__":
    main()
