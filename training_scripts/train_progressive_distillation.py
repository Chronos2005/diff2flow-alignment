"""
Progressive Distillation for Flow Matching / Diff2Flow models.

Each round trains a student (pure FM) to match the teacher's two-step
Euler prediction in a single step, halving the required NFE.

Teacher types:
  fm        -- pure FM model (from train_flow_matching.py or train_reflow.py)
  diff2flow -- Diff2Flow-aligned model (from train_diff2flow.py)

After each round the student becomes the teacher for the next round.
All students are pure FM models; evaluate with metrics_fm.py.

Reference: Salimans & Ho, "Progressive Distillation for Fast Sampling of
Diffusion Models", ICLR 2022.
"""

import copy
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import argparse
import torch
import torch.nn.functional as F
from diffusers import DDPMScheduler, UNet2DModel
from tqdm import tqdm
from accelerate import Accelerator

from shared.aligner import Diff2FlowAligner
from shared.args import (add_common_args, add_dataset_args, add_training_args,
                         add_diffusion_args)
from shared.training import build_unet, make_dataloader, make_optimizer_and_scheduler
from shared.datasets import DATASET_DEFAULTS, get_dataset, get_data_root
from shared.utils import make_grid, tensor_to_pil


# ---------------------------------------------------------------------------
# Teacher velocity wrapper
# ---------------------------------------------------------------------------

class TeacherWrapper:
    """Unified FM-space velocity interface for FM and Diff2Flow teachers."""

    def __init__(self, model, teacher_type, aligner=None):
        self.model = model
        self.teacher_type = teacher_type  # "fm" or "diff2flow"
        self.aligner = aligner

    @torch.no_grad()
    def velocity(self, x, t_fm):
        """
        Return FM-space velocity at time t_fm ∈ [0, 1].
          x    : (B, C, H, W) in FM space
          t_fm : (B,)
        """
        if self.teacher_type == "fm":
            return self.model(x, t_fm * 999.0, return_dict=False)[0]

        # diff2flow: remap to DDPM space, predict eps, translate to velocity
        t_dm = self.aligner.t_fm_to_t_dm(t_fm)
        alpha_t, sigma_t = self.aligner.get_alpha_sigma(t_dm)
        x_dm = self.aligner.x_fm_to_x_dm(x, alpha_t, sigma_t)
        eps = self.model(x_dm, t_dm.long().clamp(0, self.aligner.T - 1),
                         return_dict=False)[0]
        return self.aligner.eps_to_velocity(eps, x_dm, alpha_t, sigma_t)

    @torch.no_grad()
    def two_step_euler(self, x_t, t_fm, dt_student):
        """
        Two teacher Euler half-steps (each dt_student/2) from x_t at t_fm.
        Returns x_{t + dt_student} — the distillation target.
        """
        dt = dt_student * 0.5

        v1 = self.velocity(x_t, t_fm)
        x_mid = x_t + dt * v1

        t_mid = (t_fm + dt).clamp(max=1.0 - 1e-7)
        v2 = self.velocity(x_mid, t_mid)
        return x_mid + dt * v2


# ---------------------------------------------------------------------------
# Student sampler (pure FM, Euler)
# ---------------------------------------------------------------------------

@torch.no_grad()
def sample_fm(model, num_samples, image_size, num_steps, device):
    x = torch.randn(num_samples, 3, image_size, image_size, device=device)
    dt = 1.0 / num_steps
    for i in range(num_steps):
        t_scaled = torch.full((num_samples,), i / num_steps * 999.0, device=device)
        x = x + model(x, t_scaled, return_dict=False)[0] * dt
    return x.clamp(-1, 1)


# ---------------------------------------------------------------------------
# Single distillation round
# ---------------------------------------------------------------------------

def distillation_round(teacher_wrapper, student, dataloader, image_size,
                       student_steps, args, accelerator, round_num, round_dir):
    """
    Train student to reproduce the teacher's 2-step Euler result in 1 step.
    Returns the unwrapped, trained student model.
    """
    dt_student = 1.0 / student_steps

    total_train_steps = args.steps_per_epoch * args.num_epochs
    optimizer, lr_scheduler = make_optimizer_and_scheduler(
        student.parameters(), args.learning_rate,
        args.num_warmup_steps, total_train_steps,
    )
    student, optimizer, lr_scheduler = accelerator.prepare(
        student, optimizer, lr_scheduler
    )

    if accelerator.is_main_process:
        os.makedirs(f"{round_dir}/samples", exist_ok=True)
        print(f"\n=== Round {round_num}: distilling to {student_steps} steps "
              f"(dt={dt_student:.4f}) ===")

    global_step = 0

    for epoch in range(args.num_epochs):
        student.train()
        epoch_loss = 0.0
        steps_done = 0

        pbar = tqdm(
            dataloader,
            desc=f"  R{round_num} E{epoch+1}/{args.num_epochs}",
            disable=not accelerator.is_main_process,
        )

        for batch in pbar:
            x1 = batch[0]                                 # real images
            B  = x1.shape[0]
            x0 = torch.randn_like(x1)                    # initial noise

            # Discrete student timestep i ∈ {0, …, student_steps − 1}
            i    = torch.randint(0, student_steps, (B,), device=x1.device)
            t_fm = i.float() / student_steps              # ∈ [0, (N−1)/N]
            t_exp = t_fm.view(-1, 1, 1, 1)

            # FM interpolant as proxy for on-trajectory x_t
            x_t = t_exp * x1 + (1.0 - t_exp) * x0
            x_t = x_t + args.sigma_min * torch.randn_like(x_t)

            # Teacher's 2-step prediction → distillation target velocity
            with torch.no_grad():
                x_next   = teacher_wrapper.two_step_euler(x_t, t_fm, dt_student)
            v_target = (x_next - x_t) / dt_student

            # Student prediction (pure FM; timestep rescaled to [0, 999])
            v_pred = student(x_t, t_fm * 999.0, return_dict=False)[0]
            loss   = F.mse_loss(v_pred, v_target)

            optimizer.zero_grad()
            accelerator.backward(loss)
            torch.nn.utils.clip_grad_norm_(student.parameters(), 1.0)
            optimizer.step()
            lr_scheduler.step()

            epoch_loss += loss.item()
            steps_done += 1
            pbar.set_postfix({"loss": f"{loss.item():.4f}"})
            global_step += 1

            if steps_done >= args.steps_per_epoch:
                break

        avg = epoch_loss / max(steps_done, 1)
        if accelerator.is_main_process:
            print(f"  Epoch {epoch+1} avg loss: {avg:.4f}")

        # Sample grid
        if (epoch + 1) % args.save_images_every == 0 and accelerator.is_main_process:
            uw = accelerator.unwrap_model(student)
            uw.eval()
            with torch.no_grad():
                samples = sample_fm(uw, 16, image_size, student_steps, accelerator.device)
            grid = make_grid(tensor_to_pil(samples), rows=4, cols=4)
            path = f"{round_dir}/samples/epoch_{epoch+1}.png"
            grid.save(path)
            print(f"  Samples → {path}")
            student.train()

        # Checkpoint
        if (epoch + 1) % args.save_model_every == 0 and accelerator.is_main_process:
            uw = accelerator.unwrap_model(student)
            ckpt = f"{round_dir}/checkpoint_epoch_{epoch+1}"
            os.makedirs(ckpt, exist_ok=True)
            uw.save_pretrained(ckpt)
            torch.save({
                "epoch": epoch + 1, "global_step": global_step,
                "round": round_num, "student_steps": student_steps,
                "args": vars(args),
            }, os.path.join(ckpt, "training_state.pt"))
            print(f"  Checkpoint → {ckpt}")

    # Final save
    accelerator.wait_for_everyone()
    unwrapped = accelerator.unwrap_model(student)
    if accelerator.is_main_process:
        final = f"{round_dir}/final_model"
        os.makedirs(final, exist_ok=True)
        unwrapped.save_pretrained(final)
        torch.save({
            "round": round_num, "student_steps": student_steps,
            "args": vars(args),
        }, os.path.join(final, "training_state.pt"))
        print(f"  Round {round_num} done → {final}")
        print(f"  Eval: python evaluation_scripts/metrics_fm.py "
              f"--model_path {final} --step_counts {student_steps}")

    return unwrapped


# ---------------------------------------------------------------------------
# Args
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(
        description="Progressive Distillation: iteratively halve FM sampling steps")

    add_common_args(parser)

    # Teacher
    parser.add_argument("--teacher_checkpoint", type=str, required=True,
                        help="Path to the initial teacher checkpoint directory")
    parser.add_argument("--teacher_type", type=str, default="fm",
                        choices=["fm", "diff2flow"],
                        help="Type of teacher model (default: fm)")
    add_diffusion_args(parser)

    # Dataset
    add_dataset_args(parser, include_custom=False)
    parser.add_argument("--data_root", type=str, default=None)
    parser.add_argument("--image_size", type=int, default=32)

    # Distillation schedule
    parser.add_argument("--initial_teacher_steps", type=int, default=16,
                        help="Solver steps the teacher was trained for (default: 16). "
                             "Each round halves this; stop when < 1 or num_rounds reached.")
    parser.add_argument("--num_rounds", type=int, default=3,
                        help="Distillation rounds to run (default: 3, giving N/8 final steps)")

    # Training
    add_training_args(parser, learning_rate_default=1e-5)
    parser.add_argument("--train_batch_size", type=int, default=128)
    parser.add_argument("--steps_per_epoch", type=int, default=1000,
                        help="Gradient steps per epoch (default: 1000)")
    parser.add_argument("--num_warmup_steps", type=int, default=200)
    parser.add_argument("--sigma_min", type=float, default=1e-4,
                        help="Interpolant noise floor (default: 1e-4)")

    # Output
    parser.add_argument("--output_dir", type=str, default="progressive_distill_cifar10")

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

    # -----------------------------------------------------------------------
    # Dataset
    # -----------------------------------------------------------------------
    defaults   = DATASET_DEFAULTS[args.dataset]
    image_size = args.image_size or defaults["image_size"]
    data_root  = get_data_root(args.dataset, args.data_root)
    dataset    = get_dataset(args.dataset, data_root, image_size)
    dataloader = make_dataloader(dataset, args.train_batch_size, args.num_workers)
    dataloader = accelerator.prepare(dataloader)
    accelerator.print(f"Dataset: {args.dataset} ({len(dataset):,} images)")

    # -----------------------------------------------------------------------
    # Load initial teacher
    # -----------------------------------------------------------------------
    accelerator.print(f"Loading teacher ({args.teacher_type}): {args.teacher_checkpoint}")
    teacher = UNet2DModel.from_pretrained(args.teacher_checkpoint)

    aligner = None
    if args.teacher_type == "diff2flow":
        aligner = Diff2FlowAligner(DDPMScheduler(num_train_timesteps=args.num_train_timesteps))

    teacher = teacher.to(device).eval()
    for p in teacher.parameters():
        p.requires_grad_(False)

    teacher_wrapper = TeacherWrapper(teacher, args.teacher_type, aligner)

    accelerator.print(
        f"Initial teacher steps : {args.initial_teacher_steps}\n"
        f"Rounds                : {args.num_rounds}\n"
        f"Steps per round       : {args.num_epochs} epochs × {args.steps_per_epoch} steps/epoch\n"
        f"Output                : {args.output_dir}"
    )

    # -----------------------------------------------------------------------
    # Progressive distillation rounds
    # -----------------------------------------------------------------------
    current_teacher_steps = args.initial_teacher_steps

    for round_num in range(1, args.num_rounds + 1):
        student_steps = current_teacher_steps // 2
        if student_steps < 1:
            accelerator.print(f"Reached 1-step model after round {round_num - 1}. Stopping.")
            break

        # Student initialised from teacher weights.
        # Round 1 diff2flow: load checkpoint as a clean UNet (no LoRA structure).
        # All other rounds: copy the current teacher directly.
        if round_num == 1:
            student_base = UNet2DModel.from_pretrained(args.teacher_checkpoint)
        else:
            student_base = copy.deepcopy(teacher_wrapper.model)
            for p in student_base.parameters():
                p.requires_grad_(True)

        round_dir = (f"{args.output_dir}/round_{round_num}"
                     f"_steps_{student_steps}")

        trained_student = distillation_round(
            teacher_wrapper=teacher_wrapper,
            student=student_base,
            dataloader=dataloader,
            image_size=image_size,
            student_steps=student_steps,
            args=args,
            accelerator=accelerator,
            round_num=round_num,
            round_dir=round_dir,
        )

        # Student becomes teacher for next round
        trained_student = trained_student.to(device).eval()
        for p in trained_student.parameters():
            p.requires_grad_(False)

        teacher_wrapper = TeacherWrapper(trained_student, "fm")
        current_teacher_steps = student_steps

    accelerator.print(
        f"\nProgressive distillation complete. Results in: {args.output_dir}/"
    )


if __name__ == "__main__":
    main()
