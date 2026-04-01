import argparse
import gc
import json
import os
import time
from datetime import datetime

import numpy as np
import torch
from diffusers import UNet2DModel, DDPMScheduler
from torch.amp import autocast
from torch.utils.data import DataLoader, Subset
from torchdiffeq import odeint
from torchmetrics.image.fid import FrechetInceptionDistance
from torchvision import transforms, datasets
from tqdm import tqdm

from dataset_download_scripts.cifar import DATA_ROOT
from flow_obj import FlowModelObj


try:
    from utils.lora_utils import apply_lora, mark_only_lora_as_trainable
except ImportError:  # pragma: no cover - defensive fallback
    apply_lora = None
    mark_only_lora_as_trainable = None
    print(
        "[metrics] Warning: utils.lora_utils not found. "
        "LoRA-based Diff2Flow evaluation will be disabled unless the module is available."
    )


def set_seed(seed: int = 42) -> None:
    """Set random seeds for reproducibility across runs."""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True


class NFECounter:
    """Counter for number of function evaluations during ODE solving."""
    def __init__(self):
        self.nfe = 0

    def reset(self):
        self.nfe = 0


class DiffusersUNetWrapper(torch.nn.Module):
    """Wrap diffusers UNet to return a tensor output."""
    def __init__(self, unet):
        super().__init__()
        self.unet = unet

    def forward(self, x, t, **kwargs):
        return self.unet(x, t, return_dict=False)[0]


class FlowMatchingEvaluator:
    """Evaluator for flow matching models."""

    def __init__(self, model, device='cuda', sigma_min=1e-4, use_amp=False):
        self.model = model
        self.device = device
        self.sigma_min = sigma_min
        self.nfe_counter = NFECounter()
        self.use_amp = use_amp and torch.cuda.is_available()

    def create_ode_func(self):
        """Create ODE function that counts evaluations."""
        class ODEFunc(torch.nn.Module):
            def __init__(self, model, nfe_counter, use_amp):
                super().__init__()
                self.model = model
                self.nfe_counter = nfe_counter
                self.use_amp = use_amp

            def forward(self, t, x):
                # One evaluation per call, not per sample
                self.nfe_counter.nfe += 1
                batch_size = x.shape[0]
                t_scaled = (t * 999).expand(batch_size).to(x.device)
                with torch.no_grad():
                    # autocast(enabled=...) eliminates the duplicated if/else branch
                    with autocast('cuda', enabled=self.use_amp):
                        v = self.model(x, t_scaled, return_dict=False)[0]
                return v

        return ODEFunc(self.model, self.nfe_counter, self.use_amp)

    def generate_samples(self, num_samples, batch_size=128, method='euler',
                         num_steps=100, rtol=1e-5, atol=1e-5):
        """Generate samples and track NFE."""
        self.model.eval()
        all_samples = []
        self.nfe_counter.reset()

        num_batches = (num_samples + batch_size - 1) // batch_size

        for i in tqdm(range(num_batches), desc="Generating samples"):
            current_batch_size = min(batch_size, num_samples - i * batch_size)
            x0 = torch.randn(current_batch_size, 3, 32, 32, device=self.device)

            with torch.no_grad():
                if method == 'euler':
                    dt = 1.0 / num_steps
                    x = x0
                    for step in range(num_steps):
                        t = step * dt
                        t_tensor = torch.tensor(t, device=self.device)
                        batch_t = (t_tensor * 999).expand(current_batch_size).to(self.device)
                        with autocast('cuda', enabled=self.use_amp):
                            v = self.model(x, batch_t, return_dict=False)[0]
                        x = x + v * dt
                        # One forward pass per step, not per sample
                        self.nfe_counter.nfe += 1
                    samples = x
                else:
                    # Adaptive ODE solver (dopri5, etc.)
                    ode_func = self.create_ode_func()
                    t_span = torch.tensor([0.0, 1.0], device=self.device)
                    trajectory = odeint(ode_func, x0, t_span, method=method, rtol=rtol, atol=atol)
                    samples = trajectory[-1]

            all_samples.append(denormalize_and_clamp(samples).cpu())

        all_samples = torch.cat(all_samples, dim=0)[:num_samples]
        # NFE per sample: total forward passes / number of batches
        avg_nfe = self.nfe_counter.nfe / num_batches
        return all_samples, avg_nfe


class DDPMEvaluator:
    """Evaluator for DDPM models."""

    def __init__(self, model, device='cuda', num_inference_steps=1000, use_amp=False):
        self.model = model
        self.device = device
        self.num_inference_steps = num_inference_steps
        self.scheduler = DDPMScheduler(num_train_timesteps=1000)
        self.nfe_counter = NFECounter()
        self.use_amp = use_amp and torch.cuda.is_available()

    def generate_samples(self, num_samples, batch_size=128):
        """Generate samples and track NFE (number of denoising steps)."""
        self.model.eval()
        all_samples = []
        self.nfe_counter.reset()

        num_batches = (num_samples + batch_size - 1) // batch_size

        for i in tqdm(range(num_batches), desc="Generating samples"):
            current_batch_size = min(batch_size, num_samples - i * batch_size)
            image = torch.randn(current_batch_size, 3, 32, 32, device=self.device)
            self.scheduler.set_timesteps(self.num_inference_steps)

            for t in self.scheduler.timesteps:
                with torch.no_grad():
                    with autocast('cuda', enabled=self.use_amp):
                        model_output = self.model(image, t, return_dict=False)[0]
                    # One forward pass per timestep, not per sample
                    self.nfe_counter.nfe += 1
                    image = self.scheduler.step(model_output, t, image, return_dict=False)[0]

            all_samples.append(denormalize_and_clamp(image).cpu())

        all_samples = torch.cat(all_samples, dim=0)[:num_samples]
        # NFE per sample: total forward passes / number of batches
        avg_nfe = self.nfe_counter.nfe / num_batches
        return all_samples, avg_nfe


class Diff2FlowEvaluator:
    """Evaluator for Diff2Flow FlowModelObj."""

    def __init__(self, model, device='cuda', num_steps=1000, method='euler', use_amp=False):
        self.model = model
        self.device = device
        self.num_steps = num_steps
        self.method = method
        self.nfe_counter = NFECounter()
        self.use_amp = use_amp and torch.cuda.is_available()

    def generate_samples(self, num_samples, batch_size=128):
        self.model.eval()
        all_samples = []
        self.nfe_counter.reset()

        num_batches = (num_samples + batch_size - 1) // batch_size

        for i in tqdm(range(num_batches), desc="Generating samples"):
            current_batch_size = min(batch_size, num_samples - i * batch_size)
            z = torch.randn(current_batch_size, 3, 32, 32, device=self.device)

            with torch.no_grad():
                with autocast('cuda', enabled=self.use_amp):
                    samples = self.model.generate(
                        z,
                        sample_kwargs={"num_steps": self.num_steps, "method": self.method},
                    )

            if self.method == "euler":
                # num_steps forward passes per batch, not per sample
                self.nfe_counter.nfe += self.num_steps

            all_samples.append(denormalize_and_clamp(samples).cpu())

        all_samples = torch.cat(all_samples, dim=0)[:num_samples]
        # Return None for adaptive methods where NFE can't be tracked here;
        # None serialises cleanly as JSON null (avoids float('nan') which many parsers reject)
        avg_nfe = self.nfe_counter.nfe / num_batches if self.method == "euler" else None
        return all_samples, avg_nfe


def denormalize_and_clamp(images: torch.Tensor) -> torch.Tensor:
    """Convert images from [-1, 1] to [0, 1] and clamp."""
    return torch.clamp((images + 1) / 2, 0.0, 1.0)


def calculate_fid_in_memory(real_images, generated_images, batch_size=50, device='cuda'):
    """Calculate FID directly from in-memory tensors using torchmetrics.

    Avoids the expensive disk I/O of saving/loading 20k PNG files that the
    pytorch-fid approach required.

    Args:
        real_images: [N, 3, H, W] float tensor in [0, 1]
        generated_images: [N, 3, H, W] float tensor in [0, 1]
        batch_size: how many images to push through Inception at a time
        device: compute device

    Returns:
        FID score as a Python float
    """
    fid = FrechetInceptionDistance(feature=2048, normalize=True).to(device)

    print("Updating real image statistics...")
    for i in tqdm(range(0, len(real_images), batch_size), desc="Real images"):
        fid.update(real_images[i:i + batch_size].to(device), real=True)

    print("Updating generated image statistics...")
    for i in tqdm(range(0, len(generated_images), batch_size), desc="Generated images"):
        fid.update(generated_images[i:i + batch_size].to(device), real=False)

    return fid.compute().item()


def _resolve_pretrained_path(path_str: str) -> str:
    """Resolve a pretrained model path relative to this file, preferring local dirs."""
    if os.path.isabs(path_str) and os.path.isdir(path_str):
        return path_str
    if os.path.isdir(path_str):
        return path_str
    base_dir = os.path.dirname(os.path.abspath(__file__))
    candidate = os.path.join(base_dir, path_str)
    if os.path.isdir(candidate):
        return candidate
    return path_str


def _resolve_checkpoint_path(path_str: str) -> str:
    """Resolve a checkpoint path (file) relative to this file."""
    if os.path.isabs(path_str) and os.path.isfile(path_str):
        return path_str
    if os.path.isfile(path_str):
        return path_str
    base_dir = os.path.dirname(os.path.abspath(__file__))
    candidate = os.path.join(base_dir, path_str)
    if os.path.isfile(candidate):
        return candidate
    return path_str


def register_schedule_from_betas(flow_model, betas):
    betas = betas.detach().cpu().numpy()
    alphas = 1.0 - betas
    alphas_cumprod = alphas.cumprod(axis=0)
    alphas_cumprod_full = np.append(1.0, alphas_cumprod)

    try:
        model_device = next(flow_model.parameters()).device
    except StopIteration:
        model_device = torch.device("cpu")

    def to_torch(x):
        return torch.tensor(x, dtype=torch.float32, device=model_device)

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


@torch.no_grad()
def load_flow_matching_model(checkpoint_path, device='cuda', compile_model=False):
    """Load a Flow Matching model from checkpoint."""
    checkpoint_path = _resolve_checkpoint_path(checkpoint_path)
    model = UNet2DModel(
        sample_size=32,
        in_channels=3,
        out_channels=3,
        layers_per_block=2,
        block_out_channels=(128, 128, 256, 256, 512, 512),
        down_block_types=(
            "DownBlock2D", "DownBlock2D", "DownBlock2D",
            "DownBlock2D", "AttnDownBlock2D", "DownBlock2D",
        ),
        up_block_types=(
            "UpBlock2D", "AttnUpBlock2D", "UpBlock2D",
            "UpBlock2D", "UpBlock2D", "UpBlock2D",
        ),
    )
    # weights_only=True prevents arbitrary code execution from untrusted checkpoints
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
    model.load_state_dict(checkpoint.get('model_state_dict', checkpoint))
    model.to(device).eval()

    if compile_model and hasattr(torch, 'compile'):
        print("Compiling model with torch.compile...")
        model = torch.compile(model, mode='reduce-overhead')
    return model


@torch.no_grad()
def load_ddpm_model(model_dir, device='cuda', compile_model=False):
    """Load a DDPM model from a pretrained diffusers directory."""
    resolved_dir = _resolve_pretrained_path(model_dir)
    model = UNet2DModel.from_pretrained(resolved_dir, local_files_only=True)
    model.to(device).eval()

    if compile_model and hasattr(torch, 'compile'):
        print("Compiling model with torch.compile...")
        model = torch.compile(model, mode='reduce-overhead')
    return model


@torch.no_grad()
def load_diff2flow_model(
    checkpoint_path,
    pretrained_unet_dir,
    device='cuda',
    num_timesteps=1000,
    diffusion_parameterization='eps',
    enforce_zero_snr=False,
    compile_model=False,
    use_lora: bool = False,
    lora_r: int = 4,
    lora_alpha: float = 8.0,
    lora_dropout: float = 0.0,
):
    """Load a Diff2Flow FlowModelObj from checkpoint."""
    resolved_unet_dir = _resolve_pretrained_path(pretrained_unet_dir)
    unet = UNet2DModel.from_pretrained(resolved_unet_dir, local_files_only=True)

    if use_lora:
        if apply_lora is None or mark_only_lora_as_trainable is None:
            raise RuntimeError(
                "LoRA evaluation requested (use_lora=True) but utils.lora_utils "
                "could not be imported. Set use_lora=False or add utils/lora_utils.py."
            )
        print(f"Applying LoRA to UNet (r={lora_r}, alpha={lora_alpha}, dropout={lora_dropout})")
        apply_lora(unet, r=lora_r, lora_alpha=lora_alpha, lora_dropout=lora_dropout)
        mark_only_lora_as_trainable(unet)

    wrapped_unet = DiffusersUNetWrapper(unet)
    flow_model = FlowModelObj(
        net_cfg=wrapped_unet,
        schedule="linear",
        diffusion_parameterization=diffusion_parameterization,
        enforce_zero_snr=enforce_zero_snr,
    )

    scheduler = DDPMScheduler(num_train_timesteps=num_timesteps)
    register_schedule_from_betas(flow_model, scheduler.betas)

    checkpoint_path = _resolve_checkpoint_path(checkpoint_path)
    # weights_only=True prevents arbitrary code execution from untrusted checkpoints
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
    flow_model.load_state_dict(checkpoint.get('model_state_dict', checkpoint))
    flow_model.to(device).eval()

    if compile_model and hasattr(torch, 'compile'):
        print("Compiling Diff2Flow model with torch.compile...")
        flow_model = torch.compile(flow_model, mode='reduce-overhead')
    return flow_model


def load_real_images_efficiently(num_samples, batch_size=256, num_workers=4):
    """Load real CIFAR-10 training images for FID evaluation.

    Uses the training split (50k images), which is standard practice for
    FID real statistics.
    """
    transform = transforms.Compose([transforms.ToTensor()])
    # train=True: use the 50k training set — standard for FID real statistics
    dataset = datasets.CIFAR10(root=DATA_ROOT, train=True, download=False, transform=transform)

    available = len(dataset)
    if num_samples > available:
        raise ValueError(
            f"Requested {num_samples} real images but the dataset only has {available}. "
            f"Reduce --num-samples or use a larger split."
        )

    indices = np.random.choice(available, num_samples, replace=False)
    loader = DataLoader(
        Subset(dataset, indices),
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    real_images = []
    for batch, _ in tqdm(loader, desc="Loading real images"):
        real_images.append(batch)
    return torch.cat(real_images, dim=0)


def evaluate_model(model_path, model_type='flow_matching', num_samples=10000,
                   batch_size=128, device='cuda', use_amp=False, compile_model=False,
                   num_workers=4, **kwargs):
    """Evaluate a generative model, returning FID, NFE, and wall-clock time.

    Args:
        model_path: Path to model checkpoint or directory.
        model_type: 'flow_matching', 'ddpm', or 'diff2flow'.
        num_samples: Number of samples to generate for FID calculation.
        batch_size: Batch size for generation.
        device: Compute device string.
        use_amp: Use automatic mixed precision.
        compile_model: Use torch.compile for faster inference (PyTorch 2.0+).
        num_workers: Number of DataLoader workers.
        **kwargs: Model- and method-specific arguments.

    Returns:
        dict with keys 'fid', 'nfe', 'wall_time', 'samples_per_second'.
    """
    device = torch.device(device if torch.cuda.is_available() else 'cpu')

    print(f"Loading {model_type} model from {model_path}...")
    if model_type == 'flow_matching':
        model = load_flow_matching_model(model_path, device=device, compile_model=compile_model)
    elif model_type == 'ddpm':
        model = load_ddpm_model(model_path, device=device, compile_model=compile_model)
    else:  # diff2flow
        model = load_diff2flow_model(
            checkpoint_path=model_path,
            pretrained_unet_dir=kwargs.get("pretrained_unet_dir", "ddpm_cifar10/final_model"),
            device=device,
            num_timesteps=kwargs.get("num_timesteps", 1000),
            diffusion_parameterization=kwargs.get("diffusion_parameterization", "eps"),
            enforce_zero_snr=kwargs.get("enforce_zero_snr", False),
            compile_model=compile_model,
            use_lora=kwargs.get("use_lora", False),
            lora_r=kwargs.get("lora_r", 4),
            lora_alpha=kwargs.get("lora_alpha", 8.0),
            lora_dropout=kwargs.get("lora_dropout", 0.0),
        )

    print("Loading CIFAR-10 dataset...")
    real_images = load_real_images_efficiently(num_samples, batch_size=256, num_workers=num_workers)

    print(f"Generating {num_samples} samples...")
    start_time = time.time()

    if model_type == 'flow_matching':
        evaluator = FlowMatchingEvaluator(model, device=device, use_amp=use_amp)
        generated_images, nfe = evaluator.generate_samples(
            num_samples, batch_size=batch_size,
            method=kwargs.get('method', 'euler'),
            num_steps=kwargs.get('num_steps', 100),
            rtol=kwargs.get('rtol', 1e-5),
            atol=kwargs.get('atol', 1e-5),
        )
    elif model_type == 'ddpm':
        evaluator = DDPMEvaluator(
            model, device=device,
            num_inference_steps=kwargs.get('num_inference_steps', 1000),
            use_amp=use_amp,
        )
        generated_images, nfe = evaluator.generate_samples(num_samples, batch_size=batch_size)
    else:  # diff2flow
        evaluator = Diff2FlowEvaluator(
            model, device=device,
            num_steps=kwargs.get('num_steps', 1000),
            method=kwargs.get('method', 'euler'),
            use_amp=use_amp,
        )
        generated_images, nfe = evaluator.generate_samples(num_samples, batch_size=batch_size)

    wall_time = time.time() - start_time

    print("Calculating FID score (in-memory)...")
    fid_value = calculate_fid_in_memory(real_images, generated_images, batch_size=50, device=device)

    del real_images, generated_images
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    nfe_display = f"{nfe:.1f}" if nfe is not None else "N/A (adaptive solver)"
    print("\n" + "=" * 50)
    print(f"Evaluation Results for {model_type.upper()}")
    print("=" * 50)
    print(f"FID Score:             {fid_value:.2f}")
    print(f"NFE (avg per sample):  {nfe_display}")
    print(f"Wall Clock Time:       {wall_time:.2f}s")
    print(f"Samples/Second:        {num_samples / wall_time:.2f}")
    print("=" * 50)

    return {
        'fid': fid_value,
        'nfe': nfe,
        'wall_time': wall_time,
        'samples_per_second': num_samples / wall_time,
    }


def save_evaluation_results(results_dict, output_dir="logs", filename_prefix="metrics_results"):
    """Persist evaluation results as a JSON file."""
    os.makedirs(output_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = os.path.join(output_dir, f"{filename_prefix}_{timestamp}.json")

    def _to_serializable(obj):
        if isinstance(obj, torch.Tensor):
            return obj.detach().cpu().tolist()
        if isinstance(obj, (np.floating, np.integer)):
            return obj.item()
        # None → JSON null; avoids float('nan') which many parsers reject
        return obj

    serializable = {
        k: {mk: _to_serializable(mv) for mk, mv in v.items()}
        for k, v in results_dict.items()
    }
    with open(path, "w") as f:
        json.dump(serializable, f, indent=2)
    print(f"Saved evaluation results to {path}")


def main():
    """CLI entry-point to evaluate selected models."""
    parser = argparse.ArgumentParser(description="Evaluate generative models on CIFAR-10")
    parser.add_argument(
        "--model", "-m",
        choices=["flow_matching", "ddpm", "diff2flow", "all"],
        default="all",
        help="Which model to evaluate (default: all)",
    )
    parser.add_argument("--flow-matching-path", default="flow_matching_cifar10/final_model/model.pt",
                        help="Path to flow-matching checkpoint (.pt)")
    parser.add_argument("--ddpm-dir", default="ddpm_cifar10/final_model",
                        help="Directory of the DDPM UNet2DModel (diffusers format)")
    parser.add_argument("--diff2flow-path", default="diff2flow_cifar10/diff2flow_flowmodel.pt",
                        help="Path to Diff2Flow checkpoint (.pt)")
    # Exposed so you don't need to edit the script for quick experiments
    parser.add_argument("--num-samples", type=int, default=10000,
                        help="Number of samples for FID evaluation (default: 10000)")
    parser.add_argument("--batch-size", type=int, default=128,
                        help="Generation batch size (default: 128)")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for reproducibility (default: 42)")
    parser.add_argument("--num-workers", type=int, default=4,
                        help="DataLoader worker count (default: 4)")

    args = parser.parse_args()

    set_seed(args.seed)

    results = {}

    if args.model in ("flow_matching", "all"):
        results["flow_matching"] = evaluate_model(
            model_path=args.flow_matching_path,
            model_type='flow_matching',
            num_samples=args.num_samples,
            batch_size=args.batch_size,
            method='euler',
            num_steps=100,
            use_amp=True,
            compile_model=False,
            num_workers=args.num_workers,
        )

    if args.model in ("ddpm", "all"):
        results["ddpm"] = evaluate_model(
            model_path=args.ddpm_dir,
            model_type='ddpm',
            num_samples=args.num_samples,
            batch_size=args.batch_size,
            num_inference_steps=1000,
            use_amp=True,
            compile_model=True,
            num_workers=args.num_workers,
        )

    if args.model in ("diff2flow", "all"):
        results["diff2flow"] = evaluate_model(
            model_path=args.diff2flow_path,
            model_type='diff2flow',
            num_samples=args.num_samples,
            batch_size=args.batch_size,
            num_steps=1000,
            method='euler',
            pretrained_unet_dir="ddpm_cifar10/final_model",
            use_lora=False,
            lora_r=4,
            lora_alpha=8.0,
            lora_dropout=0.0,
            use_amp=True,
            compile_model=True,
            num_workers=args.num_workers,
        )

    if not results:
        print("No models were evaluated. Check --model argument.")
        return

    save_evaluation_results(results)

    if len(results) > 1:
        print("\n" + "=" * 50)
        print("COMPARISON")
        print("=" * 50)
        fm  = results.get("flow_matching")
        dd  = results.get("ddpm")
        d2f = results.get("diff2flow")
        if fm and dd and d2f:
            def _nfe(r):
                return f"{r['nfe']:.1f}" if r['nfe'] is not None else "N/A"
            print(f"FID   — Flow Matching: {fm['fid']:.2f}  |  DDPM: {dd['fid']:.2f}  |  Diff2Flow: {d2f['fid']:.2f}")
            print(f"NFE   — Flow Matching: {_nfe(fm)}  |  DDPM: {_nfe(dd)}  |  Diff2Flow: {_nfe(d2f)}")
            print(f"Time  — Flow Matching: {fm['wall_time']:.2f}s  |  DDPM: {dd['wall_time']:.2f}s  |  Diff2Flow: {d2f['wall_time']:.2f}s")
            print(f"Speed — Flow Matching: {fm['samples_per_second']:.2f}  |  DDPM: {dd['samples_per_second']:.2f}  |  Diff2Flow: {d2f['samples_per_second']:.2f} samples/s")
        print("=" * 50)


if __name__ == "__main__":
    main()