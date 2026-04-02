import argparse
import os

import torch
import torch.nn.functional as F
from diffusers import DDPMPipeline, DDPMScheduler, UNet2DModel
from diffusers.optimization import get_cosine_schedule_with_warmup
from PIL import Image
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from tqdm import tqdm


def parse_args():
    parser = argparse.ArgumentParser(description="Train a DDPM on CIFAR-10 or CelebA")

    parser.add_argument(
        "--dataset",
        type=str,
        required=True,
        choices=["cifar10", "celeba"],
        help="Dataset to train on",
    )
    parser.add_argument(
        "--data_root",
        type=str,
        default=None,
        help="Path to the dataset root directory (defaults to the path defined in the dataset download script)",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Output directory (default: ddpm_cifar10 or ddpm_celeba)",
    )
    parser.add_argument("--image_size", type=int, default=None, help="Image size (default: 32 for CIFAR-10, 64 for CelebA)")
    parser.add_argument("--train_batch_size", type=int, default=None, help="Training batch size (default: 128 for CIFAR-10, 64 for CelebA)")
    parser.add_argument("--num_epochs", type=int, default=None, help="Number of training epochs (default: 50 for CIFAR-10, 100 for CelebA)")
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--save_images_every", type=int, default=5)
    parser.add_argument("--save_model_every", type=int, default=10)
    parser.add_argument("--num_warmup_steps", type=int, default=500)
    parser.add_argument("--num_train_timesteps", type=int, default=1000)
    parser.add_argument("--num_inference_steps", type=int, default=1000)

    return parser.parse_args()


DATASET_DEFAULTS = {
    "cifar10": {"image_size": 32, "train_batch_size": 128, "num_epochs": 50,  "output_dir": "ddpm_cifar10"},
    "celeba":  {"image_size": 128, "train_batch_size": 128, "num_epochs": 100, "output_dir": "ddpm_celeba"},
}


def get_dataset(args):
    if args.dataset == "cifar10":
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5]),
        ])
        return datasets.CIFAR10(
            root=args.data_root, train=True, download=False, transform=transform
        )
    elif args.dataset == "celeba":
        transform = transforms.Compose([
            transforms.CenterCrop(178),
            transforms.Resize(args.image_size),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
        ])
        return datasets.CelebA(
            root=args.data_root,
            split="train",
            target_type="attr",
            download=False,
            transform=transform,
        )


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


def make_grid(images, rows, cols):
    w, h = images[0].size
    grid = Image.new("RGB", size=(cols * w, rows * h))
    for i, image in enumerate(images):
        grid.paste(image, box=(i % cols * w, i // cols * h))
    return grid


def main():
    args = parse_args()

    # Fill in dataset-specific defaults for any unset args
    defaults = DATASET_DEFAULTS[args.dataset]
    for key, val in defaults.items():
        if getattr(args, key) is None:
            setattr(args, key, val)

    # Fall back to the path defined in the dataset download script
    if args.data_root is None:
        if args.dataset == "cifar10":
            from dataset_download_scripts.cifar import CIFAR10_ROOT
            args.data_root = CIFAR10_ROOT
        elif args.dataset == "celeba":
            from dataset_download_scripts.celebA import CELEBA_ROOT
            args.data_root = CELEBA_ROOT

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(f"{args.output_dir}/samples", exist_ok=True)

    print(f"Dataset:      {args.dataset}")
    print(f"Training on:  {device}")
    print(f"Output dir:   {args.output_dir}")

    dataset = get_dataset(args)
    dataloader = DataLoader(
        dataset,
        batch_size=args.train_batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True,
    )
    print(f"Dataset size: {len(dataset):,} images")

    model = build_model(args.image_size).to(device)
    num_params = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"Model parameters: {num_params:.1f}M")

    noise_scheduler = DDPMScheduler(num_train_timesteps=args.num_train_timesteps)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)
    lr_scheduler = get_cosine_schedule_with_warmup(
        optimizer=optimizer,
        num_warmup_steps=args.num_warmup_steps,
        num_training_steps=len(dataloader) * args.num_epochs,
    )

    global_step = 0

    for epoch in range(args.num_epochs):
        model.train()
        epoch_loss = 0.0
        progress_bar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{args.num_epochs}")

        for batch in progress_bar:
            # CelebA returns (images, attrs); CIFAR-10 returns (images, labels)
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
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            lr_scheduler.step()

            epoch_loss += loss.item()
            progress_bar.set_postfix({"loss": f"{loss.item():.4f}"})
            global_step += 1

        avg_loss = epoch_loss / len(dataloader)
        print(f"Epoch {epoch+1} avg loss: {avg_loss:.4f}")

        if (epoch + 1) % args.save_images_every == 0:
            model.eval()
            with torch.no_grad():
                pipeline = DDPMPipeline(unet=model, scheduler=noise_scheduler)
                sample_images = pipeline(
                    batch_size=16,
                    num_inference_steps=args.num_inference_steps,
                ).images
                grid = make_grid(sample_images, rows=4, cols=4)
                save_path = f"{args.output_dir}/samples/epoch_{epoch+1}.png"
                grid.save(save_path)
                print(f"Saved samples → {save_path}")
            model.train()

        if (epoch + 1) % args.save_model_every == 0:
            checkpoint_dir = f"{args.output_dir}/checkpoint_epoch_{epoch+1}"
            model.save_pretrained(checkpoint_dir)
            print(f"Saved checkpoint → {checkpoint_dir}")

    model.save_pretrained(f"{args.output_dir}/final_model")
    print("Training complete!")


if __name__ == "__main__":
    main()