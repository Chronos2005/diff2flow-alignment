import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import transforms, datasets
from diffusers import UNet2DModel
from diffusers.optimization import get_cosine_schedule_with_warmup
import os
from tqdm import tqdm
from PIL import Image
import numpy as np
from dataset_download_scripts.celebA import CELEBA_ROOT

# --- Configuration ---
CONFIG = {
    "data_root": CELEBA_ROOT,
    "image_size": 64,
    "train_batch_size": 64,
    "num_epochs": 100,
    "learning_rate": 1e-4,
    "output_dir": "/scratch/ram1g23/flow_matching_celeba",
    "save_images_every": 5,
    "save_model_every": 10,
    "sigma_min": 1e-4,
    "num_workers": 4,
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
            t_scaled = t * 999
            t_tensor = torch.full((shape[0],), t_scaled, device=device)

            with torch.no_grad():
                v = model(x, t_tensor, return_dict=False)[0]
                x = x + v * dt

        return x


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(CONFIG["output_dir"], exist_ok=True)
    os.makedirs(f"{CONFIG['output_dir']}/samples", exist_ok=True)

    print(f"Training Flow Matching on: {device}")

    transform = transforms.Compose([
        transforms.CenterCrop(178),
        transforms.Resize(CONFIG["image_size"]),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
    ])

    dataset = datasets.CelebA(
        root=CONFIG["data_root"],
        split="all",
        target_type="attr",
        download=False,
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

    flow_matching = FlowMatching(sigma_min=CONFIG["sigma_min"])

    optimizer = torch.optim.AdamW(model.parameters(), lr=CONFIG["learning_rate"])

    lr_scheduler = get_cosine_schedule_with_warmup(
        optimizer=optimizer,
        num_warmup_steps=500,
        num_training_steps=len(dataloader) * CONFIG["num_epochs"],
    )

    global_step = 0

    for epoch in range(CONFIG["num_epochs"]):
        model.train()
        epoch_loss = 0
        num_batches = 0
        progress_bar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{CONFIG['num_epochs']}")

        for batch in progress_bar:
            images = batch[0].to(device)  # CelebA returns (images, attrs)
            batch_size = images.shape[0]

            noise = torch.randn_like(images)

            t = flow_matching.sample_time(batch_size, device)

            x_t, v_target = flow_matching.compute_conditional_flow(noise, images, t)

            t_scaled = (t * 999).long()

            v_pred = model(x_t, t_scaled, return_dict=False)[0]

            loss = F.mse_loss(v_pred, v_target)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            lr_scheduler.step()

            epoch_loss += loss.item()
            num_batches += 1
            progress_bar.set_postfix({"loss": f"{loss.item():.4f}"})
            global_step += 1

        avg_loss = epoch_loss / num_batches
        print(f"Epoch {epoch+1} avg loss: {avg_loss:.4f}")

        # Generate sample images
        if (epoch + 1) % CONFIG["save_images_every"] == 0:
            model.eval()
            print(f"Generating samples at epoch {epoch+1}...")

            with torch.no_grad():
                samples = flow_matching.sample(
                    model=model,
                    shape=(16, 3, CONFIG["image_size"], CONFIG["image_size"]),
                    num_steps=1000,
                    device=device
                )

                samples = (samples + 1) / 2
                samples = torch.clamp(samples, 0, 1)

                pil_images = []
                for i in range(16):
                    img = samples[i].cpu().numpy().transpose(1, 2, 0)
                    img = (img * 255).astype(np.uint8)
                    pil_images.append(Image.fromarray(img))

                grid = make_grid(pil_images, rows=4, cols=4)
                save_path = f"{CONFIG['output_dir']}/samples/epoch_{epoch+1}.png"
                grid.save(save_path)
                print(f"Saved samples → {save_path}")

            model.train()

        # Save checkpoint
        if (epoch + 1) % CONFIG["save_model_every"] == 0:
            checkpoint_dir = f"{CONFIG['output_dir']}/checkpoint_epoch_{epoch+1}"
            os.makedirs(checkpoint_dir, exist_ok=True)
            torch.save({
                "epoch": epoch + 1,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "loss": avg_loss,
            }, f"{checkpoint_dir}/model.pt")
            print(f"Saved checkpoint → {checkpoint_dir}")

    final_dir = f"{CONFIG['output_dir']}/final_model"
    os.makedirs(final_dir, exist_ok=True)
    torch.save({
        "model_state_dict": model.state_dict(),
        "config": CONFIG,
    }, f"{final_dir}/model.pt")
    print("Training complete!")


def make_grid(images, rows, cols):
    w, h = images[0].size
    grid = Image.new("RGB", size=(cols * w, rows * h))
    for i, image in enumerate(images):
        grid.paste(image, box=(i % cols * w, i // cols * h))
    return grid


if __name__ == "__main__":
    main()