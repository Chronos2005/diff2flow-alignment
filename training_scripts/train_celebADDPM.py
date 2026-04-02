import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import transforms, datasets
from diffusers import UNet2DModel, DDPMScheduler, DDPMPipeline
from diffusers.optimization import get_cosine_schedule_with_warmup
import os
from tqdm import tqdm
from PIL import Image
from dataset_download_scripts.celebA import CELEBA_ROOT

# --- Configuration ---
CONFIG = {
    "data_root": CELEBA_ROOT,      # Path to CelebA data root
    "image_size": 64,               # CelebA faces work well at 64x64
    "train_batch_size": 64,         # Reduced from 128 due to larger image size
    "num_epochs": 100,
    "learning_rate": 1e-4,
    "output_dir": "/scratch/ram1g23/ddpm_celeba",
    "save_images_every": 5,
    "save_model_every": 10,
    "num_workers": 4,
}

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(CONFIG["output_dir"], exist_ok=True)
    os.makedirs(f"{CONFIG['output_dir']}/samples", exist_ok=True)

    print(f"Training on: {device}")

    # CelebA preprocessing — center-crop to faces, resize, normalize to [-1, 1]
    transform = transforms.Compose([
        transforms.CenterCrop(178),                          # Standard CelebA face crop
        transforms.Resize(CONFIG["image_size"]),
        transforms.RandomHorizontalFlip(),                   # Light augmentation
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])  # Scale to [-1, 1]
    ])

    dataset = datasets.CelebA(
        root=CONFIG["data_root"],
        split="train",
        target_type="attr",          # Required by torchvision CelebA
        download=False,              # Set to True to auto-download (requires ~1.4GB)
        transform=transform
    )

    dataloader = DataLoader(
        dataset,
        batch_size=CONFIG["train_batch_size"],
        shuffle=True,
        num_workers=CONFIG["num_workers"],
        pin_memory=True,
        drop_last=True,
    )

    print(f"Dataset size: {len(dataset):,} images")

    # UNet — slightly larger for 64x64 faces
    model = UNet2DModel(
        sample_size=CONFIG["image_size"],
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
    model.to(device)

    num_params = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"Model parameters: {num_params:.1f}M")

    noise_scheduler = DDPMScheduler(num_train_timesteps=1000)

    optimizer = torch.optim.AdamW(model.parameters(), lr=CONFIG["learning_rate"])

    lr_scheduler = get_cosine_schedule_with_warmup(
        optimizer=optimizer,
        num_warmup_steps=500,
        num_training_steps=len(dataloader) * CONFIG["num_epochs"],
    )

    global_step = 0

    for epoch in range(CONFIG["num_epochs"]):
        model.train()
        epoch_loss = 0.0
        progress_bar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{CONFIG['num_epochs']}")

        for batch in progress_bar:
            images = batch[0].to(device)   # CelebA returns (images, attrs)
            batch_size = images.shape[0]

            noise = torch.randn_like(images)

            timesteps = torch.randint(
                0, noise_scheduler.config.num_train_timesteps,
                (batch_size,),
                device=device
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

        # Generate sample images
        if (epoch + 1) % CONFIG["save_images_every"] == 0:
            model.eval()
            with torch.no_grad():
                pipeline = DDPMPipeline(unet=model, scheduler=noise_scheduler)
                sample_images = pipeline(
                    batch_size=16,
                    num_inference_steps=1000,
                ).images

                grid = make_grid(sample_images, rows=4, cols=4)
                save_path = f"{CONFIG['output_dir']}/samples/epoch_{epoch+1}.png"
                grid.save(save_path)
                print(f"Saved samples → {save_path}")
            model.train()

        # Save checkpoint
        if (epoch + 1) % CONFIG["save_model_every"] == 0:
            checkpoint_dir = f"{CONFIG['output_dir']}/checkpoint_epoch_{epoch+1}"
            model.save_pretrained(checkpoint_dir)
            print(f"Saved checkpoint → {checkpoint_dir}")

    model.save_pretrained(f"{CONFIG['output_dir']}/final_model")
    print("Training complete!")


def make_grid(images, rows, cols):
    w, h = images[0].size
    grid = Image.new("RGB", size=(cols * w, rows * h))
    for i, image in enumerate(images):
        grid.paste(image, box=(i % cols * w, i // cols * h))
    return grid


if __name__ == "__main__":
    main()