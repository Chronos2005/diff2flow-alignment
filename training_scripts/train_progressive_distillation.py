"""
Progressive Distillation for DDPM (Salimans & Ho, 2022)
=======================================================
Halves sampling steps at each stage by training a student model to match
two consecutive deterministic (DDIM) teacher steps in a single step.

Starting from a pre-trained 1000-step DDPM, distils through stages:
  1000 → 500 → 256 → 128 → 64 → 32 → 16 → 8 → 4

Usage:
    python progressive_distillation.py              # full training
    python progressive_distillation.py --dry-run    # smoke-test (2 batches, 1 stage)
"""

import os
import copy
import math
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import transforms, datasets
from diffusers import UNet2DModel, DDPMScheduler
from diffusers.optimization import get_cosine_schedule_with_warmup
from tqdm import tqdm
from PIL import Image
import numpy as np

from dataset_download_scripts.cifar import CIFAR10_ROOT

# ──────────────────────────────────────────────────────────────
# Configuration
# ──────────────────────────────────────────────────────────────
CONFIG = {
    "pretrained_model_dir": "ddpm_cifar10/final_model",
    "output_dir": "progressive_distillation_cifar10",
    "image_size": 32,
    "train_batch_size": 128,
    "epochs_per_stage": 50,
    "learning_rate": 1e-4,
    "initial_steps": 1000,
    "target_steps": 4,
    "save_samples_every": 10,   # Generate samples every N epochs within a stage
    "num_workers": 4,
}


# ──────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────
def make_grid(images, rows, cols):
    """Create a PIL grid from a list of PIL images."""
    w, h = images[0].size
    grid = Image.new("RGB", size=(cols * w, rows * h))
    for i, image in enumerate(images):
        grid.paste(image, box=(i % cols * w, i // cols * h))
    return grid


def build_schedule(num_timesteps: int):
    """
    Build the DDPM linear beta schedule and derived quantities
    for a given number of timesteps.

    Returns a dict of numpy arrays.
    """
    betas = np.linspace(0.0001, 0.02, num_timesteps, dtype=np.float64)
    alphas = 1.0 - betas
    alphas_cumprod = np.cumprod(alphas, axis=0)
    return {
        "betas": betas,
        "alphas": alphas,
        "alphas_cumprod": alphas_cumprod,
        "sqrt_alphas_cumprod": np.sqrt(alphas_cumprod),
        "sqrt_one_minus_alphas_cumprod": np.sqrt(1.0 - alphas_cumprod),
    }


def q_sample(x_0, t, schedule, noise=None):
    """
    Forward diffusion: q(x_t | x_0) = sqrt(ᾱ_t) * x_0  +  sqrt(1 - ᾱ_t) * ε
    x_0 : (B, C, H, W)
    t   : (B,) integer timestep indices
    """
    if noise is None:
        noise = torch.randn_like(x_0)

    sqrt_alpha = torch.tensor(
        schedule["sqrt_alphas_cumprod"], device=x_0.device, dtype=x_0.dtype
    )[t].view(-1, 1, 1, 1)
    sqrt_one_minus_alpha = torch.tensor(
        schedule["sqrt_one_minus_alphas_cumprod"], device=x_0.device, dtype=x_0.dtype
    )[t].view(-1, 1, 1, 1)

    return sqrt_alpha * x_0 + sqrt_one_minus_alpha * noise


@torch.no_grad()
def ddim_step(model, x_t, t_from, t_to, schedule, device):
    """
    One deterministic DDIM step:  x_{t_from} → x_{t_to}

    Given ε_θ(x_t, t), we predict x_0 and then re-noise to t_to.
    If t_to < 0 the step returns the predicted x_0 (fully denoised).

    t_from, t_to : integer tensors (B,)
    """
    # Predict noise
    eps_pred = model(x_t, t_from, return_dict=False)[0]

    # Recover ᾱ at t_from
    alpha_bar_from = torch.tensor(
        schedule["alphas_cumprod"], device=device, dtype=x_t.dtype
    )[t_from].view(-1, 1, 1, 1)
    sqrt_alpha_bar_from = torch.sqrt(alpha_bar_from)
    sqrt_one_minus_alpha_bar_from = torch.sqrt(1.0 - alpha_bar_from)

    # Predict x_0
    x_0_pred = (x_t - sqrt_one_minus_alpha_bar_from * eps_pred) / sqrt_alpha_bar_from

    # If t_to < 0  →  return x_0  (final denoising)
    # For per-sample handling we check min(t_to); in practice all samples
    # in the batch share the same schedule so this is fine.
    if t_to.min().item() < 0:
        return x_0_pred

    # ᾱ at t_to
    alpha_bar_to = torch.tensor(
        schedule["alphas_cumprod"], device=device, dtype=x_t.dtype
    )[t_to].view(-1, 1, 1, 1)
    sqrt_alpha_bar_to = torch.sqrt(alpha_bar_to)
    sqrt_one_minus_alpha_bar_to = torch.sqrt(1.0 - alpha_bar_to)

    # Deterministic DDIM update (η = 0)
    x_t_to = sqrt_alpha_bar_to * x_0_pred + sqrt_one_minus_alpha_bar_to * eps_pred
    return x_t_to


@torch.no_grad()
def generate_samples(model, schedule, num_steps, num_samples, image_size, device):
    """
    Generate images using uniform DDIM sampling with `num_steps` steps.
    """
    # Uniform sub-sequence of timesteps (descending)
    timesteps = np.linspace(0, len(schedule["betas"]) - 1, num_steps + 1, dtype=int)
    timesteps = timesteps[::-1]  # e.g. [999, ..., 0] but only num_steps+1 entries

    x = torch.randn(num_samples, 3, image_size, image_size, device=device)

    for i in range(len(timesteps) - 1):
        t_from = torch.full((num_samples,), timesteps[i], device=device, dtype=torch.long)
        t_to_val = timesteps[i + 1]
        # If this is the last step, go to -1 to signal "return x_0"
        if i == len(timesteps) - 2:
            t_to = torch.full((num_samples,), -1, device=device, dtype=torch.long)
        else:
            t_to = torch.full((num_samples,), t_to_val, device=device, dtype=torch.long)
        x = ddim_step(model, x, t_from, t_to, schedule, device)

    # Clamp and convert to PIL
    x = (x.clamp(-1, 1) + 1) / 2  # [0, 1]
    x = (x * 255).to(torch.uint8).permute(0, 2, 3, 1).cpu().numpy()
    images = [Image.fromarray(img) for img in x]
    return images


# ──────────────────────────────────────────────────────────────
# Progressive Distillation Trainer
# ──────────────────────────────────────────────────────────────
class ProgressiveDistillationTrainer:
    """
    Implements the multi-stage progressive distillation loop.

    At each stage the student is trained to match two deterministic
    teacher steps in a single step, effectively halving the number of
    sampling steps required.
    """

    def __init__(self, config, dry_run=False):
        self.config = config
        self.dry_run = dry_run
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Build the full 1000-step schedule (used as the base noise schedule)
        self.base_schedule = build_schedule(config["initial_steps"])

        # Build dataloader
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5]),
        ])
        dataset = datasets.CIFAR10(
            root=CIFAR10_ROOT, train=True, download=False, transform=transform
        )
        self.dataloader = DataLoader(
            dataset,
            batch_size=config["train_batch_size"],
            shuffle=True,
            num_workers=config["num_workers"],
            pin_memory=True,
            drop_last=True,
        )

        # Compute the list of stage step-counts: 1000 → 500 → 256 → … → target
        self.stage_steps = []
        n = config["initial_steps"]
        while n > config["target_steps"]:
            self.stage_steps.append(n)
            n = n // 2
        self.stage_steps.append(n)
        # stage_steps e.g. [1000, 500, 256, 128, 64, 32, 16, 8, 4]

        print(f"Device: {self.device}")
        print(f"Distillation stages (teacher steps): {self.stage_steps}")
        print(f"Epochs per stage: {config['epochs_per_stage']}")

    # ------------------------------------------------------------------
    def _get_student_timestep_indices(self, num_teacher_steps):
        """
        For a teacher with `num_teacher_steps` uniform timesteps,
        the student uses every-other timestep (halved).

        Returns the array of teacher-schedule indices that the STUDENT
        will use.  Length = num_teacher_steps // 2 + 1 (including 0).
        """
        # Teacher uses a uniform grid of indices into the base 1000-step schedule
        teacher_indices = np.linspace(
            0, self.config["initial_steps"] - 1, num_teacher_steps + 1, dtype=int
        )
        # Student keeps every-other point  → half the steps
        student_indices = teacher_indices[::2]
        return teacher_indices, student_indices

    # ------------------------------------------------------------------
    def _train_one_stage(self, teacher_model, num_teacher_steps, stage_idx):
        """
        Train one distillation stage.

        Parameters
        ----------
        teacher_model : UNet2DModel   (frozen)
        num_teacher_steps : int       e.g. 1000
        stage_idx : int               0-based stage counter

        Returns
        -------
        student_model : UNet2DModel   (trained, will become next teacher)
        """
        num_student_steps = num_teacher_steps // 2
        stage_dir = os.path.join(
            self.config["output_dir"], f"stage_{stage_idx}_steps_{num_student_steps}"
        )
        os.makedirs(stage_dir, exist_ok=True)
        os.makedirs(os.path.join(stage_dir, "samples"), exist_ok=True)

        print(f"\n{'='*60}")
        print(f"Stage {stage_idx}: {num_teacher_steps} teacher steps → "
              f"{num_student_steps} student steps")
        print(f"{'='*60}")

        # Teacher timestep indices on the base 1000-step schedule
        teacher_indices, student_indices = self._get_student_timestep_indices(
            num_teacher_steps
        )
        # teacher_indices: length num_teacher_steps+1 (descending from 999 to 0)
        # student_indices: length num_student_steps+1

        # ---- Initialise student as a copy of teacher ----
        student_model = copy.deepcopy(teacher_model)
        student_model.train()
        teacher_model.eval()
        for p in teacher_model.parameters():
            p.requires_grad = False

        optimizer = torch.optim.AdamW(
            student_model.parameters(), lr=self.config["learning_rate"]
        )

        epochs = self.config["epochs_per_stage"]
        if self.dry_run:
            epochs = 1

        lr_scheduler = get_cosine_schedule_with_warmup(
            optimizer=optimizer,
            num_warmup_steps=500 if not self.dry_run else 0,
            num_training_steps=len(self.dataloader) * epochs,
        )

        # We need the teacher indices in reversed (descending) order for sampling
        # teacher_indices is ascending; reverse it for the denoising direction
        teacher_ts_desc = teacher_indices[::-1].copy()  # 999 → … → 0

        for epoch in range(epochs):
            student_model.train()
            epoch_loss = 0.0
            num_batches = 0

            desc = f"  Stage {stage_idx} | Epoch {epoch+1}/{epochs}"
            progress = tqdm(self.dataloader, desc=desc)

            for batch in progress:
                x_0 = batch[0].to(self.device)
                bs = x_0.shape[0]

                # ── 1) Sample a random student-step index ──
                # student_indices has num_student_steps+1 entries;
                # we pick from indices [0 .. num_student_steps-1] to identify
                # pairs of teacher steps.
                step_idx = torch.randint(
                    0, num_student_steps, (bs,), device=self.device
                )

                # Map to base-schedule timesteps
                # For each sample, the student goes from
                #   t_start = student_indices[-(step_idx+1)]  (= teacher_ts[2*k])
                # to
                #   t_end   = student_indices[-(step_idx)]    (= teacher_ts[2*k+2])
                #
                # And the teacher does two steps:
                #   t_start → t_mid → t_end
                # where t_mid = teacher_ts[2*k+1]

                # Convert student_indices to descending order
                student_ts_desc = student_indices[::-1].copy()  # descending

                t_start_np = np.array(
                    [student_ts_desc[idx] for idx in step_idx.cpu().numpy()]
                )
                t_end_np = np.array(
                    [student_ts_desc[idx + 1] for idx in step_idx.cpu().numpy()]
                )

                # Teacher's midpoint
                # teacher_ts_desc has 2x the resolution; student_ts_desc[k]
                # corresponds to teacher_ts_desc[2*k]
                t_mid_np = np.array(
                    [teacher_ts_desc[2 * idx + 1] for idx in step_idx.cpu().numpy()]
                )

                t_start = torch.tensor(t_start_np, device=self.device, dtype=torch.long)
                t_mid = torch.tensor(t_mid_np, device=self.device, dtype=torch.long)
                t_end = torch.tensor(t_end_np, device=self.device, dtype=torch.long)

                # ── 2) Forward diffusion to t_start ──
                noise = torch.randn_like(x_0)
                x_t = q_sample(x_0, t_start, self.base_schedule, noise=noise)

                # ── 3) Teacher: two DDIM steps  t_start → t_mid → t_end ──
                with torch.no_grad():
                    # Create t_end for teacher steps, handling the case where
                    # t_end could be 0 (the last denoising step)
                    x_mid = ddim_step(
                        teacher_model, x_t, t_start, t_mid,
                        self.base_schedule, self.device
                    )
                    # For the second step, if t_end is 0, we still use 0 as the
                    # target (not -1) since we want x at t=0, not fully denoised
                    x_target = ddim_step(
                        teacher_model, x_mid, t_mid, t_end,
                        self.base_schedule, self.device
                    )

                # ── 4) Student: one step  t_start → t_end ──
                # The student predicts noise at t_start, and we compute its
                # single-step DDIM prediction to t_end.
                eps_student = student_model(x_t, t_start, return_dict=False)[0]

                # Predict x_0 from student's noise prediction
                alpha_bar_start = torch.tensor(
                    self.base_schedule["alphas_cumprod"], device=self.device,
                    dtype=x_t.dtype
                )[t_start].view(-1, 1, 1, 1)
                sqrt_ab_start = torch.sqrt(alpha_bar_start)
                sqrt_1_ab_start = torch.sqrt(1.0 - alpha_bar_start)

                x_0_student = (x_t - sqrt_1_ab_start * eps_student) / sqrt_ab_start

                # Re-noise to t_end
                alpha_bar_end = torch.tensor(
                    self.base_schedule["alphas_cumprod"], device=self.device,
                    dtype=x_t.dtype
                )[t_end].view(-1, 1, 1, 1)
                sqrt_ab_end = torch.sqrt(alpha_bar_end)
                sqrt_1_ab_end = torch.sqrt(1.0 - alpha_bar_end)

                x_pred = sqrt_ab_end * x_0_student + sqrt_1_ab_end * eps_student

                # ── 5) Loss ──
                loss = F.mse_loss(x_pred, x_target)

                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(student_model.parameters(), 1.0)
                optimizer.step()
                lr_scheduler.step()

                epoch_loss += loss.item()
                num_batches += 1
                progress.set_postfix({"loss": f"{loss.item():.5f}"})

                if self.dry_run and num_batches >= 2:
                    break

            avg_loss = epoch_loss / max(num_batches, 1)
            print(f"  Stage {stage_idx} | Epoch {epoch+1} | avg loss: {avg_loss:.5f}")

            # ── Generate samples ──
            if (epoch + 1) % self.config["save_samples_every"] == 0 or \
               epoch == epochs - 1:
                student_model.eval()
                images = generate_samples(
                    student_model, self.base_schedule,
                    num_steps=num_student_steps,
                    num_samples=16,
                    image_size=self.config["image_size"],
                    device=self.device,
                )
                grid = make_grid(images, rows=4, cols=4)
                grid.save(os.path.join(
                    stage_dir, "samples", f"epoch_{epoch+1}.png"
                ))
                print(f"  Saved {num_student_steps}-step samples at epoch {epoch+1}")

        # ── Save student checkpoint ──
        student_model.save_pretrained(os.path.join(stage_dir, "model"))
        print(f"  Saved student model → {stage_dir}/model")

        return student_model

    # ------------------------------------------------------------------
    def run(self):
        """Execute all distillation stages."""
        os.makedirs(self.config["output_dir"], exist_ok=True)

        # Load pre-trained teacher
        print(f"Loading pre-trained DDPM from: {self.config['pretrained_model_dir']}")
        teacher = UNet2DModel.from_pretrained(
            self.config["pretrained_model_dir"]
        ).to(self.device)

        num_params = sum(p.numel() for p in teacher.parameters()) / 1e6
        print(f"Model parameters: {num_params:.1f}M")

        # Run through stages
        for stage_idx in range(len(self.stage_steps) - 1):
            num_teacher_steps = self.stage_steps[stage_idx]
            student = self._train_one_stage(teacher, num_teacher_steps, stage_idx)

            # Student becomes next teacher
            teacher = student

            if self.dry_run:
                print("\n[dry-run] Stopping after 1 stage.")
                break

        print(f"\n{'='*60}")
        print("Progressive distillation complete!")
        final_steps = self.stage_steps[min(
            len(self.stage_steps) - 1,
            1 if self.dry_run else len(self.stage_steps) - 1
        )]
        print(f"Final model uses {final_steps} sampling steps.")
        print(f"All outputs saved to: {self.config['output_dir']}")
        print(f"{'='*60}")


# ──────────────────────────────────────────────────────────────
# Entry point
# ──────────────────────────────────────────────────────────────
def parse_args():
    parser = argparse.ArgumentParser(
        description="Progressive Distillation for DDPM (Salimans & Ho, 2022)"
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Run a quick smoke-test (1 stage, 2 batches) to verify correctness."
    )
    parser.add_argument(
        "--pretrained-model", type=str, default=None,
        help="Path to pre-trained DDPM model directory (overrides CONFIG)."
    )
    parser.add_argument(
        "--output-dir", type=str, default=None,
        help="Output directory (overrides CONFIG)."
    )
    parser.add_argument(
        "--epochs-per-stage", type=int, default=None,
        help="Training epochs per distillation stage (overrides CONFIG)."
    )
    parser.add_argument(
        "--target-steps", type=int, default=None,
        help="Target number of sampling steps (overrides CONFIG)."
    )
    parser.add_argument(
        "--lr", type=float, default=None,
        help="Learning rate (overrides CONFIG)."
    )
    parser.add_argument(
        "--batch-size", type=int, default=None,
        help="Training batch size (overrides CONFIG)."
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    config = CONFIG.copy()
    if args.pretrained_model:
        config["pretrained_model_dir"] = args.pretrained_model
    if args.output_dir:
        config["output_dir"] = args.output_dir
    if args.epochs_per_stage:
        config["epochs_per_stage"] = args.epochs_per_stage
    if args.target_steps:
        config["target_steps"] = args.target_steps
    if args.lr:
        config["learning_rate"] = args.lr
    if args.batch_size:
        config["train_batch_size"] = args.batch_size

    trainer = ProgressiveDistillationTrainer(config, dry_run=args.dry_run)
    trainer.run()
