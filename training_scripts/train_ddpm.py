import argparse
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import torch
import torch.nn.functional as F
from diffusers import DDPMPipeline, DDPMScheduler
from diffusers.optimization import get_cosine_schedule_with_warmup
from diffusers.training_utils import EMAModel
from tqdm import tqdm
from accelerate import Accelerator

from shared.args import add_common_args, add_dataset_args, add_training_args, add_diffusion_args
from shared.datasets import DATASET_DEFAULTS, get_dataset, get_data_root
from shared.training import build_unet, make_dataloader
from shared.utils import make_grid

OUTPUT_DIRS = {"cifar10": "ddpm_cifar10", "celeba": "ddpm_celeba"}


def parse_args():
    parser = argparse.ArgumentParser(description="Train a DDPM on CIFAR-10 or CelebA")

    add_dataset_args(parser, include_custom=False, dataset_required=True)
    add_training_args(parser)
    add_diffusion_args(parser)
    add_common_args(parser)

    parser.add_argument("--output_dir", type=str, default=None,
                        help="Output directory (default: ddpm_cifar10 or ddpm_celeba)")
    parser.add_argument("--image_size", type=int, default=None,
                        help="Image size (default: 32 for CIFAR-10, 64 for CelebA)")
    parser.add_argument("--train_batch_size", type=int, default=None,
                        help="Training batch size (default: 128 for CIFAR-10, 64 for CelebA)")
    parser.add_argument("--num_warmup_steps", type=int, default=500)
    parser.add_argument("--num_inference_steps", type=int, default=1000)
    parser.add_argument("--ema_decay", type=float, default=0.9999)
    parser.add_argument("--ema_inv_gamma", type=float, default=1.0)
    parser.add_argument("--ema_power", type=float, default=0.75)

    return parser.parse_args()


def main():
    args = parse_args()

    defaults = DATASET_DEFAULTS[args.dataset]
    for key, val in defaults.items():
        if getattr(args, key) is None:
            setattr(args, key, val)

    if args.output_dir is None:
        args.output_dir = OUTPUT_DIRS[args.dataset]

    args.data_root = get_data_root(args.dataset, args.data_root)

    accelerator = Accelerator()
    device = accelerator.device

    if args.seed is not None:
        from accelerate.utils import set_seed
        set_seed(args.seed)

    if accelerator.is_main_process:
        os.makedirs(args.output_dir, exist_ok=True)
        os.makedirs(f"{args.output_dir}/samples", exist_ok=True)
        print(f"Dataset:      {args.dataset}")
        print(f"Training on:  {device} ({accelerator.num_processes} GPU(s))")
        print(f"Output dir:   {args.output_dir}")

    dataset = get_dataset(args.dataset, args.data_root, args.image_size)
    dataloader = make_dataloader(dataset, args.train_batch_size, args.num_workers)

    if accelerator.is_main_process:
        print(f"Dataset size: {len(dataset):,} images")

    model = build_unet(args.image_size)

    if accelerator.is_main_process:
        num_params = sum(p.numel() for p in model.parameters()) / 1e6
        print(f"Model parameters: {num_params:.1f}M")

    noise_scheduler = DDPMScheduler(num_train_timesteps=args.num_train_timesteps)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)

    model, optimizer, dataloader = accelerator.prepare(
        model, optimizer, dataloader
    )

    ema_model = EMAModel(
        accelerator.unwrap_model(model).parameters(),
        decay=args.ema_decay,
        use_ema_warmup=True,
        inv_gamma=args.ema_inv_gamma,
        power=args.ema_power,
        model_cls=type(accelerator.unwrap_model(model)),
        model_config=accelerator.unwrap_model(model).config,
    )
    ema_model.to(device)

    steps_per_epoch = len(dataloader)
    total_steps = steps_per_epoch * args.num_epochs

    lr_scheduler = get_cosine_schedule_with_warmup(
        optimizer=optimizer,
        num_warmup_steps=args.num_warmup_steps,
        num_training_steps=total_steps,
    )
    lr_scheduler = accelerator.prepare(lr_scheduler)

    if accelerator.is_main_process:
        print(f"Steps per epoch: {steps_per_epoch}")
        print(f"Total steps:     {total_steps}")

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
            accelerator.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            lr_scheduler.step()
            ema_model.step(accelerator.unwrap_model(model).parameters())

            epoch_loss += loss.item()
            progress_bar.set_postfix({"loss": f"{loss.item():.4f}"})
            global_step += 1

        if accelerator.is_main_process:
            avg_loss = epoch_loss / steps_per_epoch
            print(f"Epoch {epoch+1} avg loss: {avg_loss:.4f}")

        if (epoch + 1) % args.save_images_every == 0:
            accelerator.wait_for_everyone()
            if accelerator.is_main_process:
                unwrapped = accelerator.unwrap_model(model)
                ema_model.store(unwrapped.parameters())
                ema_model.copy_to(unwrapped.parameters())
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
                    print(f"Saved EMA samples → {save_path}")
                ema_model.restore(unwrapped.parameters())
                unwrapped.train()
            accelerator.wait_for_everyone()

        if (epoch + 1) % args.save_model_every == 0:
            accelerator.wait_for_everyone()
            if accelerator.is_main_process:
                unwrapped = accelerator.unwrap_model(model)
                checkpoint_dir = f"{args.output_dir}/checkpoint_epoch_{epoch+1}"
                unwrapped.save_pretrained(checkpoint_dir)
                ema_model.save_pretrained(f"{checkpoint_dir}_ema")
                print(f"Saved checkpoint → {checkpoint_dir} (+ EMA)")
            accelerator.save_state(f"{args.output_dir}/full_training_state")

    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        unwrapped = accelerator.unwrap_model(model)
        unwrapped.save_pretrained(f"{args.output_dir}/final_model")
        ema_model.save_pretrained(f"{args.output_dir}/final_model_ema")
        print("Training complete!")
    accelerator.save_state(f"{args.output_dir}/full_training_state")


if __name__ == "__main__":
    main()
