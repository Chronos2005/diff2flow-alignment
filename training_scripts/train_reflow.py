"""
Reflow (Rectified Flow): Straighten ODE trajectories from a trained Diff2Flow model.

Given a frozen Diff2Flow teacher, this script:
  1. Generates (x0_noise, x1_generated) pairs by running the teacher's ODE.
  2. Trains a student model on straight-line FM loss between those pairs.

The resulting model produces straight ODE trajectories, enabling high-quality
sampling at very low step counts (NFE = 1–4).

The student is a pure flow matching model — no aligner is needed at inference.
Evaluate with metrics_fm.py.

Reference: Liu et al., "Flow Straight and Fast: Learning to Generate and Transfer
Data with Rectified Flow", ICLR 2023.
"""

import argparse
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import torch
import torch.nn.functional as F
from diffusers import DDPMScheduler, UNet2DModel
from tqdm import tqdm
from accelerate import Accelerator

from shared.aligner import Diff2FlowAligner
from shared.args import add_common_args, add_training_args, add_diffusion_args
from shared.training import build_unet, make_optimizer_and_scheduler
from shared.utils import make_grid, tensor_to_pil


# ---------------------------------------------------------------------------
# Teacher sampling
# ---------------------------------------------------------------------------

@torch.no_grad()
def teacher_generate(teacher, batch_size, image_size, num_steps, device,
                     teacher_type="diff2flow", aligner=None):
    """Run the teacher ODE to produce (x0, x1) pairs."""
    shape = (batch_size, 3, image_size, image_size)
    x0 = torch.randn(shape, device=device)
    x = x0.clone()

    dt = 1.0 / num_steps
    for i in range(num_steps):
        t_fm = torch.full((batch_size,), i * dt, device=device)

        if teacher_type == "fm":
            t_scaled = t_fm * 999.0
            velocity = teacher(x, t_scaled, return_dict=False)[0]
        else:
            t_dm = aligner.t_fm_to_t_dm(t_fm)
            alpha_t, sigma_t = aligner.get_alpha_sigma(t_dm)
            x_dm = aligner.x_fm_to_x_dm(x, alpha_t, sigma_t)
            t_dm_input = t_dm.long().clamp(0, aligner.T - 1)
            eps_pred = teacher(x_dm, t_dm_input, return_dict=False)[0]
            velocity = aligner.eps_to_velocity(eps_pred, x_dm, alpha_t, sigma_t)

        x = x + dt * velocity

    x1 = x.clamp(-1, 1)
    return x0, x1


# ---------------------------------------------------------------------------
# Student FM Euler sampler (pure flow matching — no aligner)
# ---------------------------------------------------------------------------

@torch.no_grad()
def sample_student(model, num_samples, image_size, num_steps, device):
    """Euler integration for the reflow student (pure FM, no alignment)."""
    shape = (num_samples, 3, image_size, image_size)
    x = torch.randn(shape, device=device)

    dt = 1.0 / num_steps
    for i in range(num_steps):
        t_scaled = torch.full((num_samples,), i / num_steps * 999.0, device=device)
        v = model(x, t_scaled, return_dict=False)[0]
        x = x + v * dt

    return x.clamp(-1, 1)


# ---------------------------------------------------------------------------
# Args
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(
        description="Reflow: straighten ODE trajectories from a trained Diff2Flow model")

    add_common_args(parser)

    # Teacher
    parser.add_argument("--teacher_checkpoint", type=str, required=True,
                        help="Path to teacher checkpoint directory")
    parser.add_argument("--teacher_type", type=str, default="diff2flow",
                        choices=["diff2flow", "fm"],
                        help="Teacher type: 'diff2flow' (default) or 'fm'")
    parser.add_argument("--pretrained_model_path", type=str, default=None,
                        help="DDPM model path for student init (required when teacher_type=diff2flow)")
    parser.add_argument("--teacher_num_steps", type=int, default=50,
                        help="Euler steps for teacher ODE rollouts (default: 50)")

    add_diffusion_args(parser)

    # Student
    parser.add_argument("--student_init", type=str, default="pretrained",
                        choices=["random", "pretrained"],
                        help="Student initialisation: 'pretrained' from DDPM weights or 'random'")

    # Training
    parser.add_argument("--image_size", type=int, default=32)
    add_training_args(parser, learning_rate_default=1e-5)
    parser.add_argument("--train_batch_size", type=int, default=128)
    parser.add_argument("--steps_per_epoch", type=int, default=1000,
                        help="Gradient steps per epoch (no dataset; pairs generated on the fly)")
    parser.add_argument("--num_warmup_steps", type=int, default=200)
    parser.add_argument("--num_inference_steps", type=int, default=10,
                        help="Student Euler steps for sample grids during training")
    parser.add_argument("--sigma_min", type=float, default=1e-4,
                        help="Noise floor for interpolant (default: 1e-4)")

    # Output
    parser.add_argument("--output_dir", type=str, default="reflow_cifar10")

    return parser.parse_args()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args = parse_args()

    if args.num_epochs is None:
        args.num_epochs = 20

    accelerator = Accelerator()
    device = accelerator.device

    from accelerate.utils import set_seed
    set_seed(args.seed)

    if accelerator.is_main_process:
        os.makedirs(args.output_dir, exist_ok=True)
        os.makedirs(f"{args.output_dir}/samples", exist_ok=True)

    # -------------------------------------------------------------------
    # 1) Load frozen teacher
    # -------------------------------------------------------------------
    if accelerator.is_main_process:
        print(f"Loading teacher from: {args.teacher_checkpoint}")

    teacher = UNet2DModel.from_pretrained(args.teacher_checkpoint)
    teacher = teacher.to(device)
    teacher.eval()
    for p in teacher.parameters():
        p.requires_grad_(False)

    aligner = None
    if args.teacher_type == "diff2flow":
        noise_scheduler = DDPMScheduler(num_train_timesteps=args.num_train_timesteps)
        aligner = Diff2FlowAligner(noise_scheduler)

    # -------------------------------------------------------------------
    # 2) Build student
    # -------------------------------------------------------------------
    if args.student_init == "pretrained":
        init_path = args.pretrained_model_path or args.teacher_checkpoint
        if accelerator.is_main_process:
            print(f"Student init: pretrained from {init_path}")
        student = UNet2DModel.from_pretrained(init_path)
    else:
        if accelerator.is_main_process:
            print("Student init: random")
        student = build_unet(args.image_size)

    total_params = sum(p.numel() for p in student.parameters())
    if accelerator.is_main_process:
        print(f"Student parameters: {total_params/1e6:.2f}M")

    # -------------------------------------------------------------------
    # 3) Optimizer & scheduler
    # -------------------------------------------------------------------
    total_steps = args.steps_per_epoch * args.num_epochs
    optimizer, lr_scheduler = make_optimizer_and_scheduler(
        student.parameters(), args.learning_rate,
        args.num_warmup_steps, total_steps,
    )

    student, optimizer, lr_scheduler = accelerator.prepare(
        student, optimizer, lr_scheduler
    )

    if accelerator.is_main_process:
        print(f"Training: {args.num_epochs} epochs x {args.steps_per_epoch} steps/epoch = {total_steps} steps")
        print(f"Batch size: {args.train_batch_size}  |  Teacher steps: {args.teacher_num_steps}")

    # -------------------------------------------------------------------
    # 4) Training loop
    # -------------------------------------------------------------------
    global_step = 0

    for epoch in range(args.num_epochs):
        student.train()
        epoch_loss = 0.0

        progress_bar = tqdm(
            range(args.steps_per_epoch),
            desc=f"Epoch {epoch+1}/{args.num_epochs}",
            disable=not accelerator.is_main_process,
        )

        for _ in progress_bar:
            # --- Generate (x0, x1) pairs from teacher ---
            with torch.no_grad():
                x0, x1 = teacher_generate(
                    teacher,
                    batch_size=args.train_batch_size,
                    image_size=args.image_size,
                    num_steps=args.teacher_num_steps,
                    device=device,
                    teacher_type=args.teacher_type,
                    aligner=aligner,
                )

            # --- Straight-line FM loss on student ---
            t = torch.rand(args.train_batch_size, device=device)
            t_exp = t.view(-1, 1, 1, 1)

            # Interpolant: x_t = t * x1 + (1-t) * x0
            x_t = t_exp * x1 + (1 - t_exp) * x0
            x_t = x_t + args.sigma_min * torch.randn_like(x_t)

            # Target velocity: straight line from x0 to x1
            v_target = x1 - x0

            # Student prediction (pure FM: timestep scaled to [0, 999])
            t_scaled = t * 999.0
            v_pred = student(x_t, t_scaled, return_dict=False)[0]

            loss = F.mse_loss(v_pred, v_target)

            optimizer.zero_grad()
            accelerator.backward(loss)
            torch.nn.utils.clip_grad_norm_(student.parameters(), 1.0)
            optimizer.step()
            lr_scheduler.step()

            epoch_loss += loss.item()
            progress_bar.set_postfix({"loss": f"{loss.item():.4f}"})
            global_step += 1

        if accelerator.is_main_process:
            avg_loss = epoch_loss / args.steps_per_epoch
            print(f"Epoch {epoch+1} avg loss: {avg_loss:.4f}")

        # --- Sample from student ---
        if (epoch + 1) % args.save_images_every == 0 and accelerator.is_main_process:
            unwrapped = accelerator.unwrap_model(student)
            unwrapped.eval()
            with torch.no_grad():
                samples = sample_student(
                    unwrapped,
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
            student.train()

        # --- Save checkpoint ---
        if (epoch + 1) % args.save_model_every == 0 and accelerator.is_main_process:
            unwrapped = accelerator.unwrap_model(student)
            ckpt_dir = f"{args.output_dir}/checkpoint_epoch_{epoch+1}"
            os.makedirs(ckpt_dir, exist_ok=True)
            unwrapped.save_pretrained(ckpt_dir)
            torch.save({
                "epoch": epoch + 1,
                "global_step": global_step,
                "args": vars(args),
            }, os.path.join(ckpt_dir, "training_state.pt"))
            print(f"Saved checkpoint -> {ckpt_dir}")

    # --- Save final model ---
    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        unwrapped = accelerator.unwrap_model(student)
        final_dir = f"{args.output_dir}/final_model"
        os.makedirs(final_dir, exist_ok=True)
        unwrapped.save_pretrained(final_dir)
        torch.save({
            "epoch": args.num_epochs,
            "global_step": global_step,
            "args": vars(args),
        }, os.path.join(final_dir, "training_state.pt"))
        print(f"Reflow training complete! Final model saved to {final_dir}")
        print("Evaluate with: python evaluation_scripts/metrics_fm.py --model_path "
              f"{final_dir} --step_counts 1 2 4 8")


if __name__ == "__main__":
    main()
