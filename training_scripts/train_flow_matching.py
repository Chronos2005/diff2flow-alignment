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

# Import DATA_ROOT from your cifar module
from dataset_download_scripts.cifar import CIFAR10_ROOT

# --- Configuration ---
CONFIG = {
    "image_size": 32,
    "train_batch_size": 128,
    "num_epochs": 50,
    "learning_rate": 1e-4,
    "output_dir": "flow_matching_cifar10",
    "save_images_every": 5,
    "save_model_every": 10,
    "sigma_min": 1e-4,  
}

class FlowMatching:
    """Flow matching with optimal transport conditional flow matching (OT-CFM)"""
    
    def __init__(self, sigma_min=1e-4):
        self.sigma_min = sigma_min
    
    def sample_time(self, batch_size, device):
        """Sample random timesteps uniformly from [0, 1]"""
        return torch.rand(batch_size, device=device)
    
    def compute_conditional_flow(self, x0, x1, t):
        """
        Compute the conditional probability path and target velocity.
        Using optimal transport path: x_t = t * x1 + (1 - t) * x0
        Target velocity: v_t = x1 - x0
        
        Args:
            x0: Source samples (noise)
            x1: Target samples (data)
            t: Time steps [0, 1]
        
        Returns:
            x_t: Samples along the flow path
            v_t: Target velocity field
        """
        t = t.view(-1, 1, 1, 1)  # Reshape for broadcasting
        
        # Optimal transport path
        x_t = t * x1 + (1 - t) * x0
        
        # Add small noise for numerical stability
        x_t = x_t + self.sigma_min * torch.randn_like(x_t)
        
        # Target velocity
        v_t = x1 - x0
        
        return x_t, v_t
    
    def sample(self, model, shape, num_steps=1000, device='cuda'):
        """
        Sample from the model using Euler integration.
        
        Args:
            model: The velocity prediction network
            shape: Shape of samples to generate
            num_steps: Number of integration steps (using 1000 to match DDPM)
            device: Device to run on
        
        Returns:
            Generated samples
        """
        # Start from noise
        x = torch.randn(shape, device=device)
        
        dt = 1.0 / num_steps
        
        for i in range(num_steps):
            t = i / num_steps
            # Scale t to [0, 999] range to match DDPM timestep range
            t_scaled = t * 999
            t_tensor = torch.full((shape[0],), t_scaled, device=device)
            
            with torch.no_grad():
                # Predict velocity
                v = model(x, t_tensor, return_dict=False)[0]
                
                # Euler step
                x = x + v * dt
        
        return x


def main():
    # Setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(CONFIG["output_dir"], exist_ok=True)
    os.makedirs(f"{CONFIG['output_dir']}/samples", exist_ok=True)
    
    print(f"Training Flow Matching on: {device}")
    
    # Load CIFAR-10
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5])  # Scale to [-1, 1]
    ])
    
    dataset = datasets.CIFAR10(
        root=CIFAR10_ROOT,
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
    
    # Create model (velocity network)
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
    
    # Flow matching
    flow_matching = FlowMatching(sigma_min=CONFIG["sigma_min"])
    
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
        
        epoch_loss = 0
        num_batches = 0
        
        for batch in progress_bar:
            images = batch[0].to(device)  # x1 (target data)
            batch_size = images.shape[0]
            
            # Sample source noise x0
            noise = torch.randn_like(images)
            
            # Sample random timesteps in [0, 1]
            t = flow_matching.sample_time(batch_size, device)
            
            # Compute conditional flow path and target velocity
            x_t, v_target = flow_matching.compute_conditional_flow(noise, images, t)
            
            # Scale timesteps to [0, 999] to match DDPM's timestep range
            t_scaled = (t * 999).long()
            
            # Predict velocity
            v_pred = model(x_t, t_scaled, return_dict=False)[0]
            
            # Calculate loss (MSE between predicted and target velocity)
            loss = F.mse_loss(v_pred, v_target)
            
            # Backpropagation
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            lr_scheduler.step()
            
            epoch_loss += loss.item()
            num_batches += 1
            progress_bar.set_postfix({"loss": loss.item()})
            global_step += 1
        
        avg_loss = epoch_loss / num_batches
        print(f"Epoch {epoch+1} average loss: {avg_loss:.4f}")
        
        # Generate and save sample images
        if (epoch + 1) % CONFIG["save_images_every"] == 0:
            model.eval()
            print(f"Generating samples at epoch {epoch+1}...")
            
            # Generate 16 images
            with torch.no_grad():
                samples = flow_matching.sample(
                    model=model,
                    shape=(16, 3, CONFIG["image_size"], CONFIG["image_size"]),
                    num_steps=1000,  # Match DDPM's 1000 inference steps
                    device=device
                )
                
                # Denormalize from [-1, 1] to [0, 1]
                samples = (samples + 1) / 2
                samples = torch.clamp(samples, 0, 1)
                
                # Convert to PIL images
                images = []
                for i in range(16):
                    img = samples[i].cpu().numpy().transpose(1, 2, 0)
                    img = (img * 255).astype(np.uint8)
                    images.append(Image.fromarray(img))
                
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
            os.makedirs(checkpoint_dir, exist_ok=True)
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': avg_loss,
            }, f"{checkpoint_dir}/model.pt")
            print(f"Saved checkpoint at epoch {epoch+1}")
    
    # Save final model
    final_dir = f"{CONFIG['output_dir']}/final_model"
    os.makedirs(final_dir, exist_ok=True)
    torch.save({
        'model_state_dict': model.state_dict(),
        'config': CONFIG,
    }, f"{final_dir}/model.pt")
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