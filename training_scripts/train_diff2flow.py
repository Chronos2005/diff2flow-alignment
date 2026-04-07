"""
Diff2Flow: Finetune a pretrained DDPM using Flow Matching via Diffusion Model Alignment.

This script loads a pretrained DDPM (epsilon-parameterized) checkpoint and finetunes it
using the Diff2Flow framework, which aligns diffusion and flow matching trajectories
through timestep remapping, interpolant rescaling, and velocity derivation — all done
analytically in the forward pass with zero extra learnable parameters.

Reference: Schusterbauer et al., "Diff2Flow: Training Flow Matching Models via Diffusion
Model Alignment", arXiv:2506.02221, 2025.
"""

import argparse
import math
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import torch
import torch.nn.functional as F
from diffusers import DDPMScheduler, UNet2DModel
from diffusers.optimization import get_cosine_schedule_with_warmup
from PIL import Image
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from tqdm import tqdm
from accelerate import Accelerator


# ---------------------------------------------------------------------------
# Diff2Flow core: trajectory alignment helpers
# ---------------------------------------------------------------------------

class Diff2FlowAligner:
    """
    Handles the analytical alignment between diffusion and flow matching:
      - Timestep remapping:  t_FM <-> t_DM
      - Interpolant rescaling: x_FM <-> x_DM
      - Velocity derivation from epsilon prediction

    Assumes a variance-preserving (VP) DDPM schedule where:
      alpha_t = sqrt(alpha_bar_t)
      sigma_t = sqrt(1 - alpha_bar_t)
    """

    def __init__(self, noise_scheduler: DDPMScheduler):
        # Extract the cumulative alpha products from the DDPM scheduler
        alphas_cumprod = noise_scheduler.alphas_cumprod  # shape: (T,)
        T = len(alphas_cumprod)

        # alpha_t and sigma_t for each discrete diffusion timestep 0..T-1
        # In DDPM convention: x_t = alpha_t * x_0 + sigma_t * eps
        self.alpha = torch.sqrt(alphas_cumprod)          # (T,)
        self.sigma = torch.sqrt(1.0 - alphas_cumprod)    # (T,)
        self.T = T  # number of discrete diffusion timesteps (typically 1000)

        # Precompute f_t for each discrete timestep: f_t = alpha / (alpha + sigma)
        # This maps t_DM -> t_FM.  Note: t_DM=0 is data, t_DM=T-1 is noise.
        # In FM convention: t_FM=1 is data, t_FM=0 is noise.
        # So f_t(t_DM=0) should be close to 1, f_t(t_DM=T-1) close to 0.
        self.ft_values = self.alpha / (self.alpha + self.sigma)  # (T,)

    def t_fm_to_t_dm(self, t_fm: torch.Tensor) -> torch.Tensor:
        """
        Inverse timestep mapping: f_t^{-1}(t_FM) -> t_DM (continuous).

        For each t_FM value, find the two nearest discrete neighbors in ft_values
        and linearly interpolate to get a continuous t_DM.

        Args:
            t_fm: tensor of FM timesteps in [0, 1]

        Returns:
            t_dm: tensor of continuous diffusion timesteps in [0, T-1]
        """
        device = t_fm.device
        ft = self.ft_values.to(device)  # (T,) monotonically decreasing

        # ft is decreasing (ft[0] ~ 1, ft[T-1] ~ 0), so flip for searchsorted
        # which expects ascending order
        ft_ascending = ft.flip(0)  # now ascending
        # searchsorted finds insertion point in ascending array
        idx_asc = torch.searchsorted(ft_ascending, t_fm.clamp(ft_ascending[0], ft_ascending[-1]))
        idx_asc = idx_asc.clamp(1, len(ft_ascending) - 1)

        # Convert back to original (descending) indices
        # In ascending array, idx_asc points to the first value >= t_fm
        # In descending array: idx_desc = T - 1 - idx_asc  (the lower neighbor)
        # The upper neighbor in descending: idx_desc + 1 doesn't work simply.
        # Let's just work directly:
        # idx_asc-1 and idx_asc in ascending = (T-1-(idx_asc-1)) and (T-1-idx_asc) in descending
        idx_hi_asc = idx_asc       # first index >= t_fm in ascending
        idx_lo_asc = idx_asc - 1   # last index < t_fm in ascending

        ft_lo = ft_ascending[idx_lo_asc]  # value just below t_fm
        ft_hi = ft_ascending[idx_hi_asc]  # value just above t_fm

        # The corresponding discrete t_DM values (in descending ft order):
        # ascending index i corresponds to descending index T-1-i,
        # which is the diffusion timestep itself
        t_dm_lo_asc = (self.T - 1 - idx_lo_asc).float()  # t_DM for lower ft value
        t_dm_hi_asc = (self.T - 1 - idx_hi_asc).float()  # t_DM for higher ft value

        # But wait: in the descending array, higher ft = lower t_DM (closer to data).
        # ft_lo < t_fm <= ft_hi  (in ascending order)
        # ft_lo corresponds to a HIGHER t_DM (more noise), ft_hi to LOWER t_DM (less noise)
        # So t_dm_lo_asc > t_dm_hi_asc

        # Linear interpolation: find where t_fm sits between ft_lo and ft_hi
        denom = (ft_hi - ft_lo).clamp(min=1e-8)
        w = (t_fm - ft_lo) / denom  # 0 at ft_lo, 1 at ft_hi

        # Interpolate t_DM
        t_dm = t_dm_lo_asc + w * (t_dm_hi_asc - t_dm_lo_asc)
        return t_dm

    def get_alpha_sigma(self, t_dm: torch.Tensor) -> tuple:
        """
        Get interpolated alpha and sigma for continuous t_DM values.

        Args:
            t_dm: continuous diffusion timesteps in [0, T-1]

        Returns:
            (alpha, sigma) interpolated at the given t_dm values
        """
        device = t_dm.device
        alpha = self.alpha.to(device)
        sigma = self.sigma.to(device)

        t_lo = t_dm.long().clamp(0, self.T - 2)
        t_hi = t_lo + 1
        w = (t_dm - t_lo.float()).clamp(0, 1)

        alpha_t = alpha[t_lo] * (1 - w) + alpha[t_hi] * w
        sigma_t = sigma[t_lo] * (1 - w) + sigma[t_hi] * w

        return alpha_t, sigma_t

    def x_fm_to_x_dm(self, x_fm: torch.Tensor, alpha_t: torch.Tensor, sigma_t: torch.Tensor) -> torch.Tensor:
        """
        f_x^{-1}: transform FM interpolant to DM interpolant.
        x_DM = (alpha + sigma) * x_FM       (Eq. 13)

        Args:
            x_fm: FM interpolant
            alpha_t, sigma_t: schedule values at the corresponding t_DM
        """
        scale = (alpha_t + sigma_t)
        # Broadcast: alpha_t, sigma_t are (B,), x_fm is (B, C, H, W)
        while scale.dim() < x_fm.dim():
            scale = scale.unsqueeze(-1)
        return scale * x_fm

    def eps_to_velocity(
        self,
        eps_pred: torch.Tensor,
        x_dm: torch.Tensor,
        alpha_t: torch.Tensor,
        sigma_t: torch.Tensor,
    ) -> torch.Tensor:
        """
        Derive FM velocity from epsilon prediction.

        From epsilon parameterization:
            x_DM_0_hat = (x_DM - sigma_t * eps_pred) / alpha_t
            x_DM_T_hat = (x_DM - alpha_t * x_DM_0_hat) / sigma_t
                       = eps_pred  (by construction)

        Then velocity (Eq. 16 adapted for eps-param):
            v_hat = x_DM_0_hat - x_DM_T_hat

        For epsilon parameterization, using the diffusion interpolant:
            x_DM = alpha_t * x0 + sigma_t * eps
        We can recover:
            x0_hat = (x_DM - sigma_t * eps_pred) / alpha_t
            eps_hat = eps_pred
        And the FM velocity is:
            v = x1 - x0 = x0_hat - eps_hat  (since x_FM_1 = data = x_DM_0, x_FM_0 = noise = x_DM_T)
        """
        # Broadcast schedule values
        a = alpha_t.clone()
        s = sigma_t.clone()
        while a.dim() < x_dm.dim():
            a = a.unsqueeze(-1)
            s = s.unsqueeze(-1)

        # Recover data and noise estimates
        x0_hat = (x_dm - s * eps_pred) / a.clamp(min=1e-8)
        eps_hat = eps_pred  # the noise estimate IS the eps prediction

        # FM velocity: data - noise (in FM convention x1=data, x0=noise)
        velocity = x0_hat - eps_hat
        return velocity


# ---------------------------------------------------------------------------
# Dataset loading (reused from the DDPM trainer)
# ---------------------------------------------------------------------------

DATASET_DEFAULTS = {
    "cifar10": {"image_size": 32, "train_batch_size": 128, "num_epochs": 50, "output_dir": "diff2flow_cifar10"},
    "celeba":  {"image_size": 128, "train_batch_size": 128, "num_epochs": 100, "output_dir": "diff2flow_celeba"},
}


def get_dataset(name, data_root, image_size):
    if name == "cifar10":
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5]),
        ])
        return datasets.CIFAR10(root=data_root, train=True, download=False, transform=transform)
    elif name == "celeba":
        transform = transforms.Compose([
            transforms.CenterCrop(178),
            transforms.Resize(image_size),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
        ])
        return datasets.CelebA(root=data_root, split="train", target_type="attr", download=False, transform=transform)


# ---------------------------------------------------------------------------
# FM Euler sampler
# ---------------------------------------------------------------------------

@torch.no_grad()
def sample_euler(model, aligner, num_samples, image_size, num_steps, device):
    """
    Sample from the Diff2Flow model using forward Euler integration on the FM ODE.

    Starting from x_FM(t=0) = noise, integrate to x_FM(t=1) = data.
    At each step, we remap to diffusion space, run the model, derive velocity,
    and take an Euler step.
    """
    shape = (num_samples, 3, image_size, image_size)
    x = torch.randn(shape, device=device)  # x_FM at t=0 (pure noise)

    dt = 1.0 / num_steps
    for i in range(num_steps):
        t_fm = torch.full((num_samples,), i * dt, device=device)

        # 1) Remap t_FM -> t_DM
        t_dm = aligner.t_fm_to_t_dm(t_fm)

        # 2) Remap x_FM -> x_DM
        alpha_t, sigma_t = aligner.get_alpha_sigma(t_dm)
        x_dm = aligner.x_fm_to_x_dm(x, alpha_t, sigma_t)

        # 3) Run the diffusion model (expects integer-ish timesteps)
        t_dm_input = t_dm.long().clamp(0, aligner.T - 1)
        eps_pred = model(x_dm, t_dm_input, return_dict=False)[0]

        # 4) Derive FM velocity from eps prediction
        velocity = aligner.eps_to_velocity(eps_pred, x_dm, alpha_t, sigma_t)

        # 5) Euler step in FM space
        x = x + dt * velocity

    # Clamp to valid range
    x = x.clamp(-1, 1)
    return x


def tensor_to_pil(images):
    """Convert a batch of [-1,1] tensors to PIL images."""
    images = (images / 2 + 0.5).clamp(0, 1)
    images = images.permute(0, 2, 3, 1).cpu().numpy()
    return [Image.fromarray((img * 255).astype("uint8")) for img in images]


def make_grid(images, rows, cols):
    w, h = images[0].size
    grid = Image.new("RGB", size=(cols * w, rows * h))
    for i, image in enumerate(images):
        grid.paste(image, box=(i % cols * w, i // cols * h))
    return grid


# ---------------------------------------------------------------------------
# Main training loop
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(description="Diff2Flow: Finetune a DDPM with Flow Matching alignment")

    parser.add_argument("--pretrained_model_path", type=str, required=True,
                        help="Path to pretrained DDPM model directory (saved via save_pretrained)")
    parser.add_argument("--dataset", type=str, required=True, choices=["cifar10", "celeba"])
    parser.add_argument("--data_root", type=str, default=None)
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument("--image_size", type=int, default=None)
    parser.add_argument("--train_batch_size", type=int, default=None)
    parser.add_argument("--num_epochs", type=int, default=None)
    parser.add_argument("--learning_rate", type=float, default=1e-5,
                        help="Learning rate (lower than pretraining since we're finetuning)")
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--save_images_every", type=int, default=5)
    parser.add_argument("--save_model_every", type=int, default=10)
    parser.add_argument("--num_warmup_steps", type=int, default=200)
    parser.add_argument("--num_train_timesteps", type=int, default=1000,
                        help="Number of diffusion timesteps in the original DDPM schedule")
    parser.add_argument("--num_inference_steps", type=int, default=50,
                        help="Number of Euler steps for FM sampling (can be much fewer than DDPM)")
    parser.add_argument("--use_lora", action="store_true",
                        help="Use LoRA for parameter-efficient finetuning")
    parser.add_argument("--lora_rank", type=int, default=64,
                        help="LoRA rank (only used if --use_lora)")

    return parser.parse_args()


def apply_lora(model, rank=64):
    """
    Apply a simple LoRA-style low-rank adaptation to all linear and conv layers.
    Freezes the original weights and adds trainable low-rank deltas.

    This is a minimal implementation. For production use, consider using
    the peft library or diffusers' built-in LoRA support.
    """
    import torch.nn as nn

    class LoRALinear(nn.Module):
        def __init__(self, original: nn.Linear, rank: int):
            super().__init__()
            self.original = original
            self.original.weight.requires_grad_(False)
            if self.original.bias is not None:
                self.original.bias.requires_grad_(False)

            self.lora_down = nn.Linear(original.in_features, rank, bias=False)
            self.lora_up = nn.Linear(rank, original.out_features, bias=False)
            nn.init.kaiming_uniform_(self.lora_down.weight, a=math.sqrt(5))
            nn.init.zeros_(self.lora_up.weight)

        def forward(self, x):
            return self.original(x) + self.lora_up(self.lora_down(x))

    class LoRAConv2d(nn.Module):
        def __init__(self, original: nn.Conv2d, rank: int):
            super().__init__()
            self.original = original
            self.original.weight.requires_grad_(False)
            if self.original.bias is not None:
                self.original.bias.requires_grad_(False)

            # Low-rank factorization via 1x1 convolutions
            self.lora_down = nn.Conv2d(original.in_channels, rank, 1, bias=False)
            self.lora_up = nn.Conv2d(rank, original.out_channels, 1, bias=False)
            nn.init.kaiming_uniform_(self.lora_down.weight, a=math.sqrt(5))
            nn.init.zeros_(self.lora_up.weight)

        def forward(self, x):
            return self.original(x) + self.lora_up(self.lora_down(x))

    replaced = 0
    for name, module in list(model.named_modules()):
        # Navigate to parent module
        parts = name.split(".")
        parent = model
        for p in parts[:-1]:
            parent = getattr(parent, p)
        attr_name = parts[-1] if parts else None

        if attr_name and isinstance(module, nn.Linear):
            r = min(rank, module.in_features, module.out_features)
            setattr(parent, attr_name, LoRALinear(module, r))
            replaced += 1
        elif attr_name and isinstance(module, nn.Conv2d):
            r = min(rank, module.in_channels, module.out_channels)
            setattr(parent, attr_name, LoRAConv2d(module, r))
            replaced += 1

    return replaced


def main():
    args = parse_args()

    # Fill defaults
    defaults = DATASET_DEFAULTS[args.dataset]
    for key, val in defaults.items():
        if getattr(args, key) is None:
            setattr(args, key, val)

    if args.data_root is None:
        if args.dataset == "cifar10":
            from dataset_download_scripts.cifar import CIFAR10_ROOT
            args.data_root = CIFAR10_ROOT
        elif args.dataset == "celeba":
            from dataset_download_scripts.celebA import CELEBA_ROOT
            args.data_root = CELEBA_ROOT

    accelerator = Accelerator()
    device = accelerator.device

    if accelerator.is_main_process:
        os.makedirs(args.output_dir, exist_ok=True)
        os.makedirs(f"{args.output_dir}/samples", exist_ok=True)

    # -----------------------------------------------------------------------
    # 1) Load pretrained DDPM
    # -----------------------------------------------------------------------
    if accelerator.is_main_process:
        print(f"Loading pretrained DDPM from: {args.pretrained_model_path}")
    model = UNet2DModel.from_pretrained(args.pretrained_model_path)

    # Build the DDPM noise scheduler (needed for alpha/sigma schedule)
    noise_scheduler = DDPMScheduler(num_train_timesteps=args.num_train_timesteps)

    # -----------------------------------------------------------------------
    # 2) Build the Diff2Flow aligner
    # -----------------------------------------------------------------------
    aligner = Diff2FlowAligner(noise_scheduler)

    # -----------------------------------------------------------------------
    # 3) Optionally apply LoRA
    # -----------------------------------------------------------------------
    if args.use_lora:
        num_replaced = apply_lora(model, rank=args.lora_rank)
        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        total = sum(p.numel() for p in model.parameters())
        if accelerator.is_main_process:
            print(f"LoRA applied: {num_replaced} layers replaced (rank={args.lora_rank})")
            print(f"Trainable params: {trainable/1e6:.2f}M / {total/1e6:.2f}M "
                  f"({100*trainable/total:.1f}%)")
    else:
        if accelerator.is_main_process:
            total = sum(p.numel() for p in model.parameters())
            print(f"Full finetuning: {total/1e6:.2f}M parameters")

    # -----------------------------------------------------------------------
    # 4) Dataset & dataloader
    # -----------------------------------------------------------------------
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
        print(f"Dataset: {args.dataset} ({len(dataset):,} images)")
        print(f"Training on: {device} ({accelerator.num_processes} GPU(s))")
        print(f"Epochs: {args.num_epochs}, Batch size: {args.train_batch_size}")
        print(f"FM inference steps: {args.num_inference_steps}")

    # -----------------------------------------------------------------------
    # 5) Optimizer & scheduler
    # -----------------------------------------------------------------------
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable_params, lr=args.learning_rate)
    lr_scheduler = get_cosine_schedule_with_warmup(
        optimizer=optimizer,
        num_warmup_steps=args.num_warmup_steps,
        num_training_steps=len(dataloader) * args.num_epochs,
    )

    model, optimizer, dataloader, lr_scheduler = accelerator.prepare(
        model, optimizer, dataloader, lr_scheduler
    )

    # -----------------------------------------------------------------------
    # 6) Training loop — Diff2Flow (Algorithm 1 from the paper)
    # -----------------------------------------------------------------------
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
            images = batch[0]  # x1 in FM convention (data)
            bs = images.shape[0]

            # --- Step 1: Sample FM timesteps and build FM interpolant ---
            # t_FM ~ U(0, 1), x0 ~ N(0,I) (noise), x1 = data
            t_fm = torch.rand(bs, device=images.device)
            noise = torch.randn_like(images)  # x_FM_0

            # FM interpolant: x_FM_t = t * x1 + (1-t) * x0   (Eq. 4)
            t_expanded = t_fm.view(bs, 1, 1, 1)
            x_fm_t = t_expanded * images + (1 - t_expanded) * noise

            # FM ground-truth velocity: v = x1 - x0   (Eq. 5)
            velocity_target = images - noise

            # --- Step 2: Remap t_FM -> t_DM  (Eq. 12) ---
            t_dm = aligner.t_fm_to_t_dm(t_fm)

            # --- Step 3: Remap x_FM -> x_DM  (Eq. 13) ---
            alpha_t, sigma_t = aligner.get_alpha_sigma(t_dm)
            x_dm_t = aligner.x_fm_to_x_dm(x_fm_t, alpha_t, sigma_t)

            # --- Step 4: Run the diffusion model and derive velocity ---
            t_dm_input = t_dm.long().clamp(0, aligner.T - 1)
            eps_pred = model(x_dm_t, t_dm_input, return_dict=False)[0]

            # Convert eps prediction to FM velocity estimate
            velocity_pred = aligner.eps_to_velocity(eps_pred, x_dm_t, alpha_t, sigma_t)

            # --- Step 5: FM loss  (Eq. 5) ---
            loss = F.mse_loss(velocity_pred, velocity_target)

            optimizer.zero_grad()
            accelerator.backward(loss)
            torch.nn.utils.clip_grad_norm_(
                [p for p in model.parameters() if p.requires_grad], 1.0
            )
            optimizer.step()
            lr_scheduler.step()

            epoch_loss += loss.item()
            progress_bar.set_postfix({"loss": f"{loss.item():.4f}"})
            global_step += 1

        if accelerator.is_main_process:
            avg_loss = epoch_loss / len(dataloader)
            print(f"Epoch {epoch+1} avg loss: {avg_loss:.4f}")

        # --- Sample images using FM Euler sampling ---
        if (epoch + 1) % args.save_images_every == 0 and accelerator.is_main_process:
            unwrapped = accelerator.unwrap_model(model)
            unwrapped.eval()
            with torch.no_grad():
                samples = sample_euler(
                    unwrapped, aligner,
                    num_samples=16,
                    image_size=args.image_size,
                    num_steps=args.num_inference_steps,
                    device=device,
                )
                pil_images = tensor_to_pil(samples)
                grid = make_grid(pil_images, rows=4, cols=4)
                save_path = f"{args.output_dir}/samples/epoch_{epoch+1}.png"
                grid.save(save_path)
                print(f"Saved samples -> {save_path}")
            model.train()

        # --- Save checkpoint ---
        if (epoch + 1) % args.save_model_every == 0 and accelerator.is_main_process:
            unwrapped = accelerator.unwrap_model(model)
            ckpt_dir = f"{args.output_dir}/checkpoint_epoch_{epoch+1}"
            os.makedirs(ckpt_dir, exist_ok=True)
            torch.save({
                "model_state_dict": unwrapped.state_dict(),
                "epoch": epoch + 1,
                "global_step": global_step,
                "args": vars(args),
            }, os.path.join(ckpt_dir, "diff2flow_checkpoint.pt"))
            print(f"Saved checkpoint -> {ckpt_dir}")

    # --- Save final model ---
    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        unwrapped = accelerator.unwrap_model(model)
        final_dir = f"{args.output_dir}/final_model"
        os.makedirs(final_dir, exist_ok=True)
        torch.save({
            "model_state_dict": unwrapped.state_dict(),
            "epoch": args.num_epochs,
            "global_step": global_step,
            "args": vars(args),
        }, os.path.join(final_dir, "diff2flow_final.pt"))
        print(f"Training complete! Final model saved to {final_dir}")


if __name__ == "__main__":
    main()