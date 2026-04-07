import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import argparse
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import transforms, datasets
from diffusers import UNet2DModel
from diffusers.optimization import get_cosine_schedule_with_warmup
from accelerate import Accelerator
from tqdm import tqdm
from PIL import Image
import numpy as np
from dataset_download_scripts.cifar import CIFAR10_ROOT
from dataset_download_scripts.celebA import CELEBA_ROOT


# --- Dataset configs ---
DATASET_CONFIGS = {
    "cifar10": {
        "image_size": 32,
        "train_batch_size": 128,
        "num_epochs": 50,
        "output_dir": "flow_matching_cifar10",
        "block_out_channels": (128, 128, 256, 256, 512, 512),
        "data_root": CIFAR10_ROOT,
    },
    "celeba": {
        "image_size": 128,
        "train_batch_size": 128,
        "num_epochs": 100,
        "output_dir": "flow_matching_celeba",
        "block_out_channels": (128, 128, 256, 256, 512, 512),
        "data_root": CELEBA_ROOT,
    },
}


class FlowMatching:
    """Flow matching with optimal transport conditional flow matching (OT-CFM)"""

    def __init__(self, sigma_min=1e-4):
        self.sigma_min = sigma_min

    def sample_time(self, batch_size, device):
        return torch.rand(batch_size, device=device)

    def compute_conditional_flow(self, x0, x1, t):
        t = t.view(-1, 1, 1, 1)
        x_t = t * x1 + (1 - t) * x0
        x_t = x_t + self.sigma_min * torch.randn_like(x_t)
        v_t = x1 - x0
        return x_t, v_t

    def sample(self, model, shape, num_steps=1000, device="cuda"):
        x = torch.randn(shape, device=device)
        dt = 1.0 / num_steps

        for i in range(num_steps):
            t = i / num_steps
            t_scaled = t * 999.0
            t_tensor = torch.full((shape[0],), t_scaled, device=device)

            with torch.no_grad():
                v = model(x, t_tensor, return_dict=False)[0]
                x = x + v * dt

        return x


def get_dataset(dataset_name, data_root, image_size):
    if dataset_name == "cifar10":
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5]),
        ])
        dataset = datasets.CIFAR10(
            root=data_root,
            train=True,
            download=False,
            transform=transform,
        )
    elif dataset_name == "celeba":
        transform = transforms.Compose([
            transforms.CenterCrop(178),
            transforms.Resize(image_size),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
        ])
        dataset = datasets.CelebA(
            root=data_root,
            split="all",
            target_type="attr",
            download=False,
            transform=transform,
        )
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")

    return dataset


def make_grid(images, rows, cols):
    w, h = images[0].size
    grid = Image.new("RGB", size=(cols * w, rows * h))
    for i, image in enumerate(images):
        grid.paste(image, box=(i % cols * w, i // cols * h))
    return grid


def main():
    parser = argparse.ArgumentParser(description="Train a Flow Matching model on CIFAR-10 or CelebA")

    parser.add_argument("--dataset", type=str, required=True, choices=["cifar10", "celeba"], help="Dataset to train on")
    parser.add_argument("--data_root", type=str, default=None, help="Root directory of the dataset")
    parser.add_argument("--output_dir", type=str, default=None, help="Output directory")
    parser.add_argument("--image_size", type=int, default=None, help="Image size")
    parser.add_argument("--batch_size", type=int, default=None, help="Per-GPU training batch size")
    parser.add_argument("--num_epochs", type=int, default=None, help="Number of training epochs")
    parser.add_argument("--learning_rate", type=float, default=1e-4, help="Learning rate (default: 1e-4)")
    parser.add_argument("--sigma_min", type=float, default=1e-4, help="Minimum sigma for flow matching (default: 1e-4)")
    parser.add_argument("--num_workers", type=int, default=4, help="Number of DataLoader workers (default: 4)")
    parser.add_argument("--save_images_every", type=int, default=5, help="Save sample images every N epochs (default: 5)")
    parser.add_argument("--save_model_every", type=int, default=10, help="Save model checkpoint every N epochs (default: 10)")
    parser.add_argument("--num_inference_steps", type=int, default=1000, help="Number of Euler steps during sampling (default: 1000)")
    parser.add_argument("--resume", type=str, default=None, help="Path to a checkpoint .pt file to resume training from")
    parser.add_argument("--mixed_precision", type=str, default="no", choices=["no", "fp16", "bf16"], help="Mixed precision mode (default: no)")

    args = parser.parse_args()

    # Initialize Accelerator — handles DDP, device placement, mixed precision
    accelerator = Accelerator(
        mixed_precision=args.mixed_precision,
        gradient_accumulation_steps=1,
    )

    # Merge dataset defaults with CLI overrides
    defaults = DATASET_CONFIGS[args.dataset]
    image_size = args.image_size or defaults["image_size"]
    per_gpu_batch_size = args.batch_size or defaults["train_batch_size"]
    num_epochs = args.num_epochs or defaults["num_epochs"]
    output_dir = args.output_dir or defaults["output_dir"]
    data_root = args.data_root or defaults["data_root"]

    if accelerator.is_main_process:
        os.makedirs(output_dir, exist_ok=True)
        os.makedirs(f"{output_dir}/samples", exist_ok=True)

    accelerator.print(f"Dataset     : {args.dataset}")
    accelerator.print(f"Num processes: {accelerator.num_processes}")
    accelerator.print(f"Device      : {accelerator.device}")
    accelerator.print(f"Mixed prec  : {args.mixed_precision}")
    accelerator.print(f"Image sz    : {image_size}  |  Batch/GPU: {per_gpu_batch_size}  |  Effective batch: {per_gpu_batch_size * accelerator.num_processes}  |  Epochs: {num_epochs}")

    # Data — Accelerate handles DistributedSampler automatically via prepare()
    dataset = get_dataset(args.dataset, data_root, image_size)
    dataloader = DataLoader(
        dataset,
        batch_size=per_gpu_batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True,
    )
    accelerator.print(f"Dataset size: {len(dataset):,} images")

    # Model
    model = UNet2DModel(
        sample_size=image_size,
        in_channels=3,
        out_channels=3,
        layers_per_block=2,
        block_out_channels=defaults["block_out_channels"],
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

    num_params = sum(p.numel() for p in model.parameters()) / 1e6
    accelerator.print(f"Model parameters: {num_params:.1f}M")

    flow_matching = FlowMatching(sigma_min=args.sigma_min)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)
    lr_scheduler = get_cosine_schedule_with_warmup(
        optimizer=optimizer,
        num_warmup_steps=500,
        num_training_steps=len(dataloader) * num_epochs,
    )

    # Accelerate prepares everything — wraps model in DDP, moves to device,
    # sets up distributed sampler, and wraps optimizer/scheduler
    model, optimizer, dataloader, lr_scheduler = accelerator.prepare(
        model, optimizer, dataloader, lr_scheduler
    )

    # Optionally resume from checkpoint
    start_epoch = 0
    if args.resume:
        accelerator.print(f"Resuming from checkpoint: {args.resume}")
        checkpoint = torch.load(args.resume, map_location=accelerator.device)
        accelerator.unwrap_model(model).load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        if "lr_scheduler_state_dict" in checkpoint:
            lr_scheduler.load_state_dict(checkpoint["lr_scheduler_state_dict"])
        start_epoch = checkpoint["epoch"]
        accelerator.print(f"Resumed at epoch {start_epoch}")

    # Training loop
    for epoch in range(start_epoch, num_epochs):
        model.train()
        epoch_loss = 0.0
        num_batches = 0

        progress_bar = tqdm(
            dataloader,
            desc=f"Epoch {epoch+1}/{num_epochs}",
            disable=not accelerator.is_main_process,
        )

        for batch in progress_bar:
            # CelebA returns (images, attrs); CIFAR-10 returns (images, labels)
            images = batch[0]  # Accelerate already placed on the right device
            current_batch_size = images.shape[0]

            noise = torch.randn_like(images)
            t = flow_matching.sample_time(current_batch_size, images.device)
            x_t, v_target = flow_matching.compute_conditional_flow(noise, images, t)

            # Use float timesteps for consistency with sampling
            t_scaled = t * 999.0
            v_pred = model(x_t, t_scaled, return_dict=False)[0]

            loss = F.mse_loss(v_pred, v_target)

            accelerator.backward(loss)
            accelerator.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad()

            epoch_loss += loss.item()
            num_batches += 1
            progress_bar.set_postfix({"loss": f"{loss.item():.4f}"})

        avg_loss = epoch_loss / num_batches
        accelerator.print(f"Epoch {epoch+1} avg loss: {avg_loss:.4f}")

        # Sample images (only on main process)
        if (epoch + 1) % args.save_images_every == 0 and accelerator.is_main_process:
            unwrapped_model = accelerator.unwrap_model(model)
            unwrapped_model.eval()
            accelerator.print(f"Generating samples at epoch {epoch+1}...")
            with torch.no_grad():
                samples = flow_matching.sample(
                    model=unwrapped_model,
                    shape=(16, 3, image_size, image_size),
                    num_steps=args.num_inference_steps,
                    device=accelerator.device,
                )
                samples = torch.clamp((samples + 1) / 2, 0, 1)
                pil_images = []
                for i in range(16):
                    img = (samples[i].cpu().numpy().transpose(1, 2, 0) * 255).astype(np.uint8)
                    pil_images.append(Image.fromarray(img))
                grid = make_grid(pil_images, rows=4, cols=4)
                save_path = f"{output_dir}/samples/epoch_{epoch+1}.png"
                grid.save(save_path)
                accelerator.print(f"Saved samples → {save_path}")

        # Save checkpoint (only on main process)
        if (epoch + 1) % args.save_model_every == 0 and accelerator.is_main_process:
            checkpoint_dir = f"{output_dir}/checkpoint_epoch_{epoch+1}"
            os.makedirs(checkpoint_dir, exist_ok=True)
            unwrapped_model = accelerator.unwrap_model(model)
            torch.save({
                "epoch": epoch + 1,
                "model_state_dict": unwrapped_model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "lr_scheduler_state_dict": lr_scheduler.state_dict(),
                "loss": avg_loss,
            }, f"{checkpoint_dir}/model.pt")
            accelerator.print(f"Saved checkpoint → {checkpoint_dir}")

        accelerator.wait_for_everyone()

    # Final model
    if accelerator.is_main_process:
        final_dir = f"{output_dir}/final_model"
        os.makedirs(final_dir, exist_ok=True)
        unwrapped_model = accelerator.unwrap_model(model)
        torch.save({
            "model_state_dict": unwrapped_model.state_dict(),
            "dataset": args.dataset,
            "image_size": image_size,
        }, f"{final_dir}/model.pt")
        accelerator.print("Training complete!")


if __name__ == "__main__":
    main()