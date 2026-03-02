import os
import math
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import transforms, datasets, utils
from diffusers import UNet2DModel, DDPMScheduler

# Diff2Flow pieces from this repo
from flow_obj import FlowModelObj
from cifar import DATA_ROOT

# --- Config ---
PRETRAINED_UNET_DIR = "ddpm_cifar10/final_model"
OUT_DIR = "diff2flow_cifar10"
IMAGE_SIZE = 32
BATCH_SIZE = 128
EPOCHS = 20
LR = 1e-4
NUM_TIMESTEPS = 1000

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
os.makedirs(OUT_DIR, exist_ok=True)

# --- Dataset ---
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize([0.5], [0.5]),
])

dataset = datasets.CIFAR10(root=DATA_ROOT, train=True, download=False, transform=transform)
loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)

# --- Load pretrained diffusion UNet ---
unet = UNet2DModel.from_pretrained(PRETRAINED_UNET_DIR).to(device)

# --- Wrap UNet so Diff2Flow gets a tensor output ---
class DiffusersUNetWrapper(torch.nn.Module):
    def __init__(self, unet):
        super().__init__()
        self.unet = unet
    def forward(self, x, t, **kwargs):
        # diffusers UNet returns UNet2DOutput unless return_dict=False
        return self.unet(x, t, return_dict=False)[0]

wrapped_unet = DiffusersUNetWrapper(unet)

# --- Build Diff2Flow model (alignment happens in FlowModelObj) ---
flow_model = FlowModelObj(
    net_cfg=wrapped_unet,
    schedule="linear",                    # FM schedule (not diffusion schedule)
    diffusion_parameterization="eps",     # DDPM predicts eps/noise
    enforce_zero_snr=False
).to(device)

# --- IMPORTANT: register diffusion schedule from your DDPM scheduler ---
# This replaces the SDV2 schedule hardcoded in FlowModelObj.
def register_schedule_from_betas(flow_model, betas):
    # betas: torch tensor [T]
    betas = betas.detach().cpu().numpy()
    alphas = 1.0 - betas
    alphas_cumprod = (alphas).cumprod(axis=0)
    alphas_cumprod_full = np.append(1.0, alphas_cumprod)

    try:
        model_device = next(flow_model.parameters()).device
    except StopIteration:
        model_device = torch.device("cpu")
    to_torch = lambda x: torch.tensor(x, dtype=torch.float32, device=model_device)

    flow_model.num_timesteps = int(betas.shape[0])
    flow_model.register_buffer("betas", to_torch(betas))
    flow_model.register_buffer("alphas_cumprod", to_torch(alphas_cumprod))
    flow_model.register_buffer("alphas_cumprod_full", to_torch(alphas_cumprod_full))

    flow_model.register_buffer("sqrt_alphas_cumprod", to_torch(np.sqrt(alphas_cumprod)))
    flow_model.register_buffer("sqrt_one_minus_alphas_cumprod", to_torch(np.sqrt(1.0 - alphas_cumprod)))
    flow_model.register_buffer("sqrt_alphas_cumprod_full", to_torch(np.sqrt(alphas_cumprod_full)))
    flow_model.register_buffer("sqrt_one_minus_alphas_cumprod_full", to_torch(np.sqrt(1.0 - alphas_cumprod_full)))

    flow_model.register_buffer("sqrt_recip_alphas_cumprod", to_torch(np.sqrt(1.0 / alphas_cumprod)))
    flow_model.register_buffer("sqrt_recipm1_alphas_cumprod", to_torch(np.sqrt(1.0 / alphas_cumprod - 1.0)))

    flow_model.register_buffer(
        "rectified_alphas_cumprod_full",
        flow_model.sqrt_alphas_cumprod_full /
        (flow_model.sqrt_alphas_cumprod_full + flow_model.sqrt_one_minus_alphas_cumprod_full)
    )
    flow_model.register_buffer(
        "rectified_sqrt_alphas_cumprod_full",
        flow_model.sqrt_one_minus_alphas_cumprod_full /
        (flow_model.sqrt_alphas_cumprod_full + flow_model.sqrt_one_minus_alphas_cumprod_full)
    )

import numpy as np
scheduler = DDPMScheduler(num_train_timesteps=NUM_TIMESTEPS)
register_schedule_from_betas(flow_model, scheduler.betas)
flow_model = flow_model.to(device)

# --- Optimizer ---
optimizer = torch.optim.AdamW(flow_model.parameters(), lr=LR)

# --- Finetune loop ---
for epoch in range(EPOCHS):
    flow_model.train()
    for batch in loader:
        x1 = batch[0].to(device)  # data
        loss = flow_model.training_losses(x1).mean()

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(flow_model.parameters(), 1.0)
        optimizer.step()

    print(f"Epoch {epoch+1}/{EPOCHS} | loss={loss.item():.4f}")

    # Sample every few epochs
    if (epoch + 1) % 5 == 0:
        flow_model.eval()
        with torch.no_grad():
            z = torch.randn(16, 3, IMAGE_SIZE, IMAGE_SIZE, device=device)
            samples = flow_model.generate(z, sample_kwargs={"num_steps": 1000, "method": "euler"})
            samples = (samples.clamp(-1, 1) + 1) / 2
            grid = utils.make_grid(samples, nrow=4)
            utils.save_image(grid, f"{OUT_DIR}/samples_epoch_{epoch+1}.png")

# Save finetuned flow model
torch.save(flow_model.state_dict(), f"{OUT_DIR}/diff2flow_flowmodel.pt")
print("Done.")
