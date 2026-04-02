import os

# Set to prevent any accidental network access by Hugging Face libraries, ensuring we only use local files.
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["HF_DATASETS_OFFLINE"] = "1"

import argparse
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import transforms, datasets, utils
from diffusers import UNet2DModel, DDPMScheduler

from utils.flow_obj import FlowModelObj
from dataset_download_scripts.cifar import CIFAR10_ROOT

# --- Argument parsing ---
parser = argparse.ArgumentParser()
parser.add_argument(
    "--mode",
    type=str,
    default="full",
    choices=["full", "freeze_encoder"],
    help=(
        "full: finetune all parameters (default, recommended). "
        "freeze_encoder: freeze the first N down blocks, train decoder + bottleneck only."
    ),
)
parser.add_argument(
    "--freeze_down_blocks",
    type=int,
    default=4,
    help="Number of down blocks to freeze when --mode=freeze_encoder (default: 4 out of 6).",
)
args = parser.parse_args()

# --- Config ---
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PRETRAINED_UNET_DIR = os.path.join(SCRIPT_DIR, "ddpm_cifar10/final_model")
OUT_DIR = f"diff2flow_cifar10_{args.mode}"
IMAGE_SIZE = 32
BATCH_SIZE = 128
EPOCHS = 20
LR = 1e-4
NUM_TIMESTEPS = 1000

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
os.makedirs(OUT_DIR, exist_ok=True)

print(f"Device: {device}")
print(f"Loading pretrained UNet from: {PRETRAINED_UNET_DIR}")

# --- Dataset ---
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize([0.5], [0.5]),
])
dataset = datasets.CIFAR10(root=CIFAR10_ROOT, train=True, download=False, transform=transform)
loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)

# --- Load pretrained diffusion UNet (local only, no network) ---
unet = UNet2DModel.from_pretrained(PRETRAINED_UNET_DIR, local_files_only=True).to(device)

# --- Wrap UNet so Diff2Flow receives a plain tensor ---
class DiffusersUNetWrapper(torch.nn.Module):
    def __init__(self, unet):
        super().__init__()
        self.unet = unet

    def forward(self, x, t, **kwargs):
        return self.unet(x, t, return_dict=False)[0]

wrapped_unet = DiffusersUNetWrapper(unet)

# --- Build Diff2Flow model ---
flow_model = FlowModelObj(
    net_cfg=wrapped_unet,
    schedule="linear",
    diffusion_parameterization="eps",
    enforce_zero_snr=False,
).to(device)

# --- Register diffusion schedule from DDPM betas ---
def register_schedule_from_betas(flow_model, betas):
    betas_np = betas.detach().cpu().numpy()
    alphas = 1.0 - betas_np
    alphas_cumprod = alphas.cumprod(axis=0)
    alphas_cumprod_full = np.append(1.0, alphas_cumprod)

    try:
        model_device = next(flow_model.parameters()).device
    except StopIteration:
        model_device = torch.device("cpu")

    def to_torch(x):
        return torch.tensor(x, dtype=torch.float32, device=model_device)

    flow_model.num_timesteps = int(betas_np.shape[0])
    flow_model.register_buffer("betas", to_torch(betas_np))
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
        (flow_model.sqrt_alphas_cumprod_full + flow_model.sqrt_one_minus_alphas_cumprod_full),
    )
    flow_model.register_buffer(
        "rectified_sqrt_alphas_cumprod_full",
        flow_model.sqrt_one_minus_alphas_cumprod_full /
        (flow_model.sqrt_alphas_cumprod_full + flow_model.sqrt_one_minus_alphas_cumprod_full),
    )

scheduler = DDPMScheduler(num_train_timesteps=NUM_TIMESTEPS)
register_schedule_from_betas(flow_model, scheduler.betas)
flow_model = flow_model.to(device)

# --- Apply freezing strategy ---
if args.mode == "freeze_encoder":
    n = args.freeze_down_blocks
    frozen, trainable_count = 0, 0

    for name, param in flow_model.named_parameters():
        should_freeze = any(
            f"unet.down_blocks.{i}." in name for i in range(n)
        )
        if should_freeze:
            param.requires_grad_(False)
            frozen += param.numel()
        else:
            trainable_count += param.numel()

    total = frozen + trainable_count
    print(
        f"Mode: freeze_encoder (freezing first {n} down blocks)\n"
        f"  Frozen params:    {frozen:>12,}  ({100*frozen/total:.1f}%)\n"
        f"  Trainable params: {trainable_count:>12,}  ({100*trainable_count/total:.1f}%)\n"
        f"  Total params:     {total:>12,}"
    )
else:
    total = sum(p.numel() for p in flow_model.parameters())
    print(
        f"Mode: full finetuning\n"
        f"  Trainable params: {total:>12,}  (100.0%)\n"
        f"  Total params:     {total:>12,}"
    )

# --- Optimizer: only pass trainable parameters ---
trainable_params = [p for p in flow_model.parameters() if p.requires_grad]
optimizer = torch.optim.AdamW(trainable_params, lr=LR)

# --- Training loop ---
for epoch in range(EPOCHS):
    flow_model.train()
    epoch_loss = 0.0

    for batch in loader:
        x1 = batch[0].to(device)
        loss = flow_model.training_losses(x1).mean()

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(trainable_params, 1.0)
        optimizer.step()

        epoch_loss += loss.item()

    avg_loss = epoch_loss / len(loader)
    print(f"Epoch {epoch+1}/{EPOCHS} | avg_loss={avg_loss:.4f} | last_loss={loss.item():.4f}")

    if (epoch + 1) % 5 == 0:
        flow_model.eval()
        with torch.no_grad():
            z = torch.randn(16, 3, IMAGE_SIZE, IMAGE_SIZE, device=device)
            samples = flow_model.generate(z, sample_kwargs={"num_steps": 1000, "method": "euler"})
            samples = (samples.clamp(-1, 1) + 1) / 2
            grid = utils.make_grid(samples, nrow=4)
            utils.save_image(grid, f"{OUT_DIR}/samples_epoch_{epoch+1}.png")
        print(f"  Saved samples to {OUT_DIR}/samples_epoch_{epoch+1}.png")

# --- Save ---
torch.save(flow_model.state_dict(), f"{OUT_DIR}/diff2flow_flowmodel.pt")
print(f"Model saved to {OUT_DIR}/diff2flow_flowmodel.pt")
print("Done.")