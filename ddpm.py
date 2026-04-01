import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import transforms, datasets
from diffusers import UNet2DModel, DDPMScheduler, DDPMPipeline
from diffusers.optimization import get_cosine_schedule_with_warmup
import os
from tqdm import tqdm
from PIL import Image
import numpy as np

# Import DATA_ROOT from your cifar module
from dataset_download_scripts.cifar import DATA_ROOT

# --- Configuration ---
CONFIG = {
    "image_size": 32,
    "train_batch_size": 128,
    "num_epochs": 50,
    "learning_rate": 1e-4,
    "output_dir": "ddpm_cifar10",
    "save_images_every": 5,
    "save_model_every": 10,
}

def main():
    # Setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(CONFIG["output_dir"], exist_ok=True)
    os.makedirs(f"{CONFIG['output_dir']}/samples", exist_ok=True)
    
    print(f"Training on: {device}")
    
    # Load CIFAR-10
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5])  # Scale to [-1, 1]
    ])
    
    dataset = datasets.CIFAR10(
        root=DATA_ROOT,
        train=True,
        download=False,
        transform=transform
    )
    
    dataloader = DataLoader(
        dataset,
        batch_size=CONFIG["train_batch_size"],
        shuffle=True,
        num_workers=4
    )
    
    # Create model
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
    
    # Noise scheduler
    noise_scheduler = DDPMScheduler(num_train_timesteps=1000)
    
    # Optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=CONFIG["learning_rate"])
    
    # Learning rate scheduler
    lr_scheduler = get_cosine_schedule_with_warmup(
        optimizer=optimizer,
        num_warmup_steps=500,
        num_training_steps=len(dataloader) * CONFIG["num_epochs"],
    )
    
    # Training loop
    global_step = 0
    
    for epoch in range(CONFIG["num_epochs"]):
        model.train()
        progress_bar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{CONFIG['num_epochs']}")
        
        for batch in progress_bar:
            images = batch[0].to(device)
            batch_size = images.shape[0]
            
            # Sample random noise
            noise = torch.randn_like(images)
            
            # Sample random timesteps
            timesteps = torch.randint(
                0, noise_scheduler.config.num_train_timesteps,
                (batch_size,),
                device=device
            ).long()
            
            # Add noise to images
            noisy_images = noise_scheduler.add_noise(images, noise, timesteps)
            
            # Predict the noise
            noise_pred = model(noisy_images, timesteps, return_dict=False)[0]
            
            # Calculate loss
            loss = F.mse_loss(noise_pred, noise)
            
            # Backpropagation
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            lr_scheduler.step()
            
            progress_bar.set_postfix({"loss": loss.item()})
            global_step += 1
        
        # Generate and save sample images
        if (epoch + 1) % CONFIG["save_images_every"] == 0:
            model.eval()
            with torch.no_grad():
                # Create pipeline for sampling
                pipeline = DDPMPipeline(
                    unet=model,
                    scheduler=noise_scheduler
                )
                
                # Generate 16 images
                images = pipeline(
                    batch_size=16,
                    num_inference_steps=1000,
                ).images
                
                # Save as grid
                image_grid = make_grid(images, rows=4, cols=4)
                image_grid.save(
                    f"{CONFIG['output_dir']}/samples/epoch_{epoch+1}.png"
                )
            
            print(f"Saved samples at epoch {epoch+1}")
            model.train()
        
        # Save checkpoint
        if (epoch + 1) % CONFIG["save_model_every"] == 0:
            checkpoint_dir = f"{CONFIG['output_dir']}/checkpoint_epoch_{epoch+1}"
            model.save_pretrained(checkpoint_dir)
            print(f"Saved checkpoint at epoch {epoch+1}")
    
    # Save final model
    model.save_pretrained(f"{CONFIG['output_dir']}/final_model")
    print("Training complete!")

def make_grid(images, rows, cols):
    """Create a grid of images"""
    w, h = images[0].size
    grid = Image.new('RGB', size=(cols*w, rows*h))
    
    for i, image in enumerate(images):
        grid.paste(image, box=(i%cols*w, i//cols*h))
    
    return grid

if __name__ == "__main__":
    main()