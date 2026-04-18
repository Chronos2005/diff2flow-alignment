import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import argparse
import torch
import torch.nn.functional as F
from diffusers import UNet2DModel
from diffusers.training_utils import EMAModel
from accelerate import Accelerator
from accelerate.utils import set_seed
from tqdm import tqdm
import numpy as np

from shared.args import add_common_args, add_dataset_args, add_training_args
from shared.datasets import DATASET_DEFAULTS, CIFAR10_ROOT, CELEBA_ROOT, get_dataset, get_data_root
from shared.training import build_unet, make_dataloader, make_optimizer_and_scheduler
from shared.utils import make_grid, tensor_to_pil

# --- Dataset configs (extends shared defaults with FM-specific fields) ---
DATASET_CONFIGS = {
    "cifar10": {"output_dir": "flow_matching_cifar10"},
    "celeba":  {"output_dir": "flow_matching_celeba"},
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
            t_scaled = t * 999.0
            t_tensor = torch.full((shape[0],), t_scaled, device=device)

            with torch.no_grad():
                v = model(x, t_tensor, return_dict=False)[0]
                x = x + v * dt

        return x


def parse_args():
    parser = argparse.ArgumentParser(description="Train a Flow Matching model on CIFAR-10 or CelebA")

    add_common_args(parser)
    add_dataset_args(parser, include_custom=False, dataset_required=True)
    add_training_args(parser)

    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument("--image_size", type=int, default=None)
    parser.add_argument("--batch_size", type=int, default=None,
                        help="Per-GPU training batch size")
    parser.add_argument("--sigma_min", type=float, default=1e-4,
                        help="Minimum sigma for flow matching (default: 1e-4)")
    parser.add_argument("--num_inference_steps", type=int, default=1000,
                        help="Number of Euler steps during sampling (default: 1000)")
    parser.add_argument("--resume", type=str, default=None,
                        help="Path to a checkpoint .pt file to resume training from")
    parser.add_argument("--mixed_precision", type=str, default="no",
                        choices=["no", "fp16", "bf16"],
                        help="Mixed precision mode (default: no)")
    parser.add_argument("--ema_decay", type=float, default=0.9999)
    parser.add_argument("--ema_inv_gamma", type=float, default=1.0)
    parser.add_argument("--ema_power", type=float, default=0.75)

    return parser.parse_args()


def main():
    args = parse_args()

    # Initialize Accelerator
    accelerator = Accelerator(
        mixed_precision=args.mixed_precision,
        gradient_accumulation_steps=1,
    )
    set_seed(args.seed)

    # Merge dataset defaults with CLI overrides
    defaults = DATASET_DEFAULTS[args.dataset]
    fm_config = DATASET_CONFIGS[args.dataset]
    image_size = args.image_size or defaults["image_size"]
    per_gpu_batch_size = args.batch_size or defaults["train_batch_size"]
    num_epochs = args.num_epochs or defaults["num_epochs"]
    output_dir = args.output_dir or fm_config["output_dir"]
    data_root = get_data_root(args.dataset, args.data_root)

    if accelerator.is_main_process:
        os.makedirs(output_dir, exist_ok=True)
        os.makedirs(f"{output_dir}/samples", exist_ok=True)

    accelerator.print(f"Dataset     : {args.dataset}")
    accelerator.print(f"Num processes: {accelerator.num_processes}")
    accelerator.print(f"Device      : {accelerator.device}")
    accelerator.print(f"Mixed prec  : {args.mixed_precision}")
    accelerator.print(f"Image sz    : {image_size}  |  Batch/GPU: {per_gpu_batch_size}  |  Effective batch: {per_gpu_batch_size * accelerator.num_processes}  |  Epochs: {num_epochs}")

    # Data
    dataset = get_dataset(args.dataset, data_root, image_size)
    dataloader = make_dataloader(dataset, per_gpu_batch_size, args.num_workers)
    accelerator.print(f"Dataset size: {len(dataset):,} images")

    # Model
    model = build_unet(image_size)

    num_params = sum(p.numel() for p in model.parameters()) / 1e6
    accelerator.print(f"Model parameters: {num_params:.1f}M")

    flow_matching = FlowMatching(sigma_min=args.sigma_min)

    steps_per_epoch = len(dataloader) // accelerator.num_processes
    optimizer, lr_scheduler = make_optimizer_and_scheduler(
        model.parameters(), args.learning_rate, 500, steps_per_epoch * num_epochs,
    )

    model, optimizer, dataloader, lr_scheduler = accelerator.prepare(
        model, optimizer, dataloader, lr_scheduler
    )

    ema_model = EMAModel(
        accelerator.unwrap_model(model).parameters(),
        decay=args.ema_decay,
        use_ema_warmup=True,
        inv_gamma=args.ema_inv_gamma,
        power=args.ema_power,
        model_cls=type(accelerator.unwrap_model(model)),
        model_config=accelerator.unwrap_model(model).config,
    )
    ema_model.to(accelerator.device)

    # Optionally resume from checkpoint
    start_epoch = 0
    if args.resume:
        accelerator.print(f"Resuming from checkpoint: {args.resume}")
        resume_dir = args.resume
        resumed = UNet2DModel.from_pretrained(resume_dir)
        accelerator.unwrap_model(model).load_state_dict(resumed.state_dict())
        state_path = os.path.join(resume_dir, "training_state.pt")
        if os.path.exists(state_path):
            training_state = torch.load(state_path, map_location=accelerator.device)
            optimizer.load_state_dict(training_state["optimizer_state_dict"])
            if "lr_scheduler_state_dict" in training_state:
                lr_scheduler.load_state_dict(training_state["lr_scheduler_state_dict"])
            start_epoch = training_state["epoch"]
        ema_dir = f"{resume_dir}_ema"
        if os.path.isdir(ema_dir):
            loaded_ema = EMAModel.from_pretrained(ema_dir, model_cls=UNet2DModel)
            ema_model.load_state_dict(loaded_ema.state_dict())
            ema_model.to(accelerator.device)
            accelerator.print(f"Loaded EMA weights from {ema_dir}")
        accelerator.print(f"Resumed at epoch {start_epoch}")

    # Training loop
    for epoch in range(start_epoch, num_epochs):
        model.train()
        epoch_loss = 0.0
        num_batches = 0

        progress_bar = tqdm(
            dataloader,
            desc=f"Epoch {epoch+1}/{num_epochs}",
            disable=not accelerator.is_main_process,
        )

        for batch in progress_bar:
            images = batch[0]
            current_batch_size = images.shape[0]

            noise = torch.randn_like(images)
            t = flow_matching.sample_time(current_batch_size, images.device)
            x_t, v_target = flow_matching.compute_conditional_flow(noise, images, t)

            t_scaled = t * 999.0
            v_pred = model(x_t, t_scaled, return_dict=False)[0]

            loss = F.mse_loss(v_pred, v_target)

            accelerator.backward(loss)
            accelerator.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad()
            ema_model.step(accelerator.unwrap_model(model).parameters())

            epoch_loss += loss.item()
            num_batches += 1
            progress_bar.set_postfix({"loss": f"{loss.item():.4f}"})

        avg_loss = epoch_loss / num_batches
        accelerator.print(f"Epoch {epoch+1} avg loss: {avg_loss:.4f}")

        # Sample images (only on main process)
        if (epoch + 1) % args.save_images_every == 0 and accelerator.is_main_process:
            unwrapped_model = accelerator.unwrap_model(model)
            ema_model.store(unwrapped_model.parameters())
            ema_model.copy_to(unwrapped_model.parameters())
            unwrapped_model.eval()
            accelerator.print(f"Generating samples at epoch {epoch+1}...")
            with torch.no_grad():
                samples = flow_matching.sample(
                    model=unwrapped_model,
                    shape=(16, 3, image_size, image_size),
                    num_steps=args.num_inference_steps,
                    device=accelerator.device,
                )
                samples = samples.clamp(-1, 1)
                pil_images = tensor_to_pil(samples)
                grid = make_grid(pil_images, rows=4, cols=4)
                save_path = f"{output_dir}/samples/epoch_{epoch+1}.png"
                grid.save(save_path)
                accelerator.print(f"Saved EMA samples → {save_path}")
            ema_model.restore(unwrapped_model.parameters())

        # Save checkpoint
        if (epoch + 1) % args.save_model_every == 0:
            checkpoint_dir = f"{output_dir}/checkpoint_epoch_{epoch+1}"
            if accelerator.is_main_process:
                os.makedirs(checkpoint_dir, exist_ok=True)
                unwrapped_model = accelerator.unwrap_model(model)
                unwrapped_model.save_pretrained(checkpoint_dir)
                ema_model.save_pretrained(f"{checkpoint_dir}_ema")
                torch.save({
                    "epoch": epoch + 1,
                    "optimizer_state_dict": optimizer.state_dict(),
                    "lr_scheduler_state_dict": lr_scheduler.state_dict(),
                    "loss": avg_loss,
                }, f"{checkpoint_dir}/training_state.pt")
            accelerator.save_state(f"{output_dir}/full_training_state")
            accelerator.print(f"Saved checkpoint → {checkpoint_dir} (+ EMA)")

        accelerator.wait_for_everyone()

    # Final model
    if accelerator.is_main_process:
        final_dir = f"{output_dir}/final_model"
        os.makedirs(final_dir, exist_ok=True)
        unwrapped_model = accelerator.unwrap_model(model)
        unwrapped_model.save_pretrained(final_dir)
        ema_model.save_pretrained(f"{output_dir}/final_model_ema")
    accelerator.save_state(f"{output_dir}/full_training_state")
    if accelerator.is_main_process:
        accelerator.print("Training complete!")


if __name__ == "__main__":
    main()
