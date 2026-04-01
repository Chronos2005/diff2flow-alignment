import argparse
import gc
import json
import os
import shutil
import tempfile
import time
from datetime import datetime

import numpy as np
import torch
from PIL import Image
from diffusers import UNet2DModel, DDPMScheduler, DDPMPipeline
from pytorch_fid import fid_score
from torch.amp import autocast
from torch.utils.data import DataLoader, Subset
from torchdiffeq import odeint
from torchvision import transforms, datasets
from tqdm import tqdm

from cifar import DATA_ROOT
from flow_obj import FlowModelObj

# LoRA utilities are optional; make the import robust so evaluation still works
# even if LoRA training helpers are not present in this checkout.
try:
    from utils.lora_utils import apply_lora, mark_only_lora_as_trainable
except ImportError:  # pragma: no cover - defensive fallback
    apply_lora = None
    mark_only_lora_as_trainable = None
    print(
        "[metrics] Warning: utils.lora_utils not found. "
        "LoRA-based Diff2Flow evaluation will be disabled unless the module is available."
    )


class NFECounter:
    """Counter for number of function evaluations during ODE solving"""
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
    """Evaluator for flow matching models"""
    
    def __init__(self, model, device='cuda', sigma_min=1e-4, use_amp=False):
        self.model = model
        self.device = device
        self.sigma_min = sigma_min
        self.nfe_counter = NFECounter()
        self.use_amp = use_amp and torch.cuda.is_available()
    
    def create_ode_func(self):
        """Create ODE function that counts evaluations"""
        class ODEFunc(torch.nn.Module):
            def __init__(self, model, nfe_counter, use_amp):
                super().__init__()
                self.model = model
                self.nfe_counter = nfe_counter
                self.use_amp = use_amp
            
            def forward(self, t, x):
                self.nfe_counter.nfe += 1
                batch_size = x.shape[0]
                t_scaled = (t * 999).expand(batch_size).to(x.device)
                
                with torch.no_grad():
                    if self.use_amp:
                        with autocast('cuda'):
                            v = self.model(x, t_scaled, return_dict=False)[0]
                    else:
                        v = self.model(x, t_scaled, return_dict=False)[0]
                return v
        
        return ODEFunc(self.model, self.nfe_counter, self.use_amp)
    
    def generate_samples(self, num_samples, batch_size=128, method='euler', num_steps=100, rtol=1e-5, atol=1e-5):
        """Generate samples and track NFE"""
        self.model.eval()
        all_samples = []
        self.nfe_counter.reset()
        
        num_batches = (num_samples + batch_size - 1) // batch_size
        
        for i in tqdm(range(num_batches), desc="Generating samples"):
            current_batch_size = min(batch_size, num_samples - i * batch_size)
            
            # Start from noise
            x0 = torch.randn(current_batch_size, 3, 32, 32, device=self.device)
            
            with torch.no_grad():
                if method == 'euler':
                    # Euler integration
                    dt = 1.0 / num_steps
                    x = x0
                    for step in range(num_steps):
                        t = step * dt
                        t_tensor = torch.tensor(t, device=self.device)
                        batch_t = (t_tensor * 999).expand(current_batch_size).to(self.device)
                        if self.use_amp:
                            with autocast('cuda'):
                                v = self.model(x, batch_t, return_dict=False)[0]
                        else:
                            v = self.model(x, batch_t, return_dict=False)[0]
                        x = x + v * dt
                        self.nfe_counter.nfe += 1
                    samples = x
                else:
                    # Adaptive ODE solver (dopri5, etc.)
                    ode_func = self.create_ode_func()
                    t_span = torch.tensor([0.0, 1.0], device=self.device)
                    trajectory = odeint(
                        ode_func,
                        x0,
                        t_span,
                        method=method,
                        rtol=rtol,
                        atol=atol,
                    )
                    samples = trajectory[-1]

            # Denormalize and move to CPU
            samples = denormalize_and_clamp(samples).cpu()
            all_samples.append(samples)
            
            # Clear GPU cache periodically
            if (i + 1) % 10 == 0:
                torch.cuda.empty_cache()
        
        all_samples = torch.cat(all_samples, dim=0)[:num_samples]
        avg_nfe = self.nfe_counter.nfe / num_samples
        
        return all_samples, avg_nfe


class DDPMEvaluator:
    """Evaluator for DDPM models"""
    
    def __init__(self, model, device='cuda', num_inference_steps=1000, use_amp=False):
        self.model = model
        self.device = device
        self.num_inference_steps = num_inference_steps
        self.scheduler = DDPMScheduler(num_train_timesteps=1000)
        self.nfe_counter = NFECounter()
        self.use_amp = use_amp and torch.cuda.is_available()
    
    def generate_samples(self, num_samples, batch_size=128):
        """Generate samples and track NFE (number of denoising steps)"""
        self.model.eval()
        all_samples = []
        self.nfe_counter.reset()
        
        num_batches = (num_samples + batch_size - 1) // batch_size
        
        for i in tqdm(range(num_batches), desc="Generating samples"):
            current_batch_size = min(batch_size, num_samples - i * batch_size)
            
            # Manual generation to count NFE
            image = torch.randn(current_batch_size, 3, 32, 32, device=self.device)
            
            self.scheduler.set_timesteps(self.num_inference_steps)
            
            for t in self.scheduler.timesteps:
                with torch.no_grad():
                    # Predict noise with optional AMP
                    if self.use_amp:
                        with autocast('cuda'):
                            model_output = self.model(image, t, return_dict=False)[0]
                    else:
                        model_output = self.model(image, t, return_dict=False)[0]
                    
                    self.nfe_counter.nfe += current_batch_size
                    
                    # Compute previous image
                    image = self.scheduler.step(model_output, t, image, return_dict=False)[0]

            # Denormalize and move to CPU
            samples = denormalize_and_clamp(image).cpu()
            all_samples.append(samples)
            
            # Clear GPU cache periodically
            if (i + 1) % 10 == 0:
                torch.cuda.empty_cache()
        
        all_samples = torch.cat(all_samples, dim=0)[:num_samples]
        avg_nfe = self.nfe_counter.nfe / num_samples
        
        return all_samples, avg_nfe


class Diff2FlowEvaluator:
    """Evaluator for Diff2Flow FlowModelObj"""
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
                if self.use_amp:
                    with autocast('cuda'):
                        samples = self.model.generate(
                            z,
                            sample_kwargs={"num_steps": self.num_steps, "method": self.method},
                        )
                else:
                    samples = self.model.generate(
                        z,
                        sample_kwargs={"num_steps": self.num_steps, "method": self.method},
                    )

            if self.method == "euler":
                self.nfe_counter.nfe += current_batch_size * self.num_steps

            # Denormalize and move to CPU
            samples = denormalize_and_clamp(samples).cpu()
            all_samples.append(samples)
            
            # Clear GPU cache periodically
            if (i + 1) % 10 == 0:
                torch.cuda.empty_cache()

        all_samples = torch.cat(all_samples, dim=0)[:num_samples]
        avg_nfe = self.nfe_counter.nfe / num_samples if self.method == "euler" else float("nan")

        return all_samples, avg_nfe


def denormalize_and_clamp(images: torch.Tensor) -> torch.Tensor:
    """Convert images from [-1, 1] to [0, 1] and clamp.

    This helper removes duplicated logic across evaluators.
    """
    images = (images + 1) / 2
    images = torch.clamp(images, 0.0, 1.0)
    return images


def save_images_to_dir(images, output_dir, num_workers=4):
    """Save tensor images to directory for FID calculation - optimized with multiprocessing"""
    os.makedirs(output_dir, exist_ok=True)
    
    # Convert all tensors to numpy first (faster in batch)
    images_np = (images.numpy().transpose(0, 2, 3, 1) * 255).astype(np.uint8)
    
    for idx, img_array in enumerate(tqdm(images_np, desc=f"Saving images to {output_dir}")):
        img = Image.fromarray(img_array)
        img.save(os.path.join(output_dir, f'{idx:05d}.png'))


def calculate_fid_from_tensors(real_images, generated_images, batch_size=50, device='cuda', num_workers=4):
    """
    Calculate FID using pytorch-fid library.
    
    Args:
        real_images: Tensor of real images [N, 3, H, W] in range [0, 1]
        generated_images: Tensor of generated images [N, 3, H, W] in range [0, 1]
        batch_size: Batch size for FID calculation
        device: Device to use
        num_workers: Number of workers for data loading
    
    Returns:
        FID score
    """
    # Create temporary directories
    with tempfile.TemporaryDirectory() as temp_dir:
        real_dir = os.path.join(temp_dir, 'real')
        gen_dir = os.path.join(temp_dir, 'generated')
        
        # Save images to directories
        print("Preparing images for FID calculation...")
        save_images_to_dir(real_images, real_dir, num_workers=num_workers)
        save_images_to_dir(generated_images, gen_dir, num_workers=num_workers)
        
        # Calculate FID using pytorch-fid
        print("Calculating FID score...")
        fid_value = fid_score.calculate_fid_given_paths(
            [real_dir, gen_dir],
            batch_size=batch_size,
            device=device,
            dims=2048,  # InceptionV3 feature dimension
            num_workers=num_workers
        )
    
    return fid_value


def _resolve_pretrained_path(path_str: str) -> str:
    """Resolve a pretrained model path relative to this file, preferring local dirs.

    This avoids accidentally treating a local folder name as a Hugging Face repo ID
    (which would trigger network calls on clusters with no internet access).
    """
    # Absolute path pointing to a directory
    if os.path.isabs(path_str) and os.path.isdir(path_str):
        return path_str

    # Relative path from current working directory
    if os.path.isdir(path_str):
        return path_str

    # Relative to this metrics.py file
    base_dir = os.path.dirname(os.path.abspath(__file__))
    candidate = os.path.join(base_dir, path_str)
    if os.path.isdir(candidate):
        return candidate

    # Fall back to original string; diffusers may interpret it as a repo ID
    return path_str


def _resolve_checkpoint_path(path_str: str) -> str:
    """Resolve a checkpoint path (file) relative to this file.

    This makes default relative paths work even when the working directory is
    different (e.g. when called from job_scripts on a cluster).
    """
    # Absolute file path
    if os.path.isabs(path_str) and os.path.isfile(path_str):
        return path_str

    # Relative to current working directory
    if os.path.isfile(path_str):
        return path_str

    # Relative to this metrics.py file
    base_dir = os.path.dirname(os.path.abspath(__file__))
    candidate = os.path.join(base_dir, path_str)
    if os.path.isfile(candidate):
        return candidate

    # Fall back; torch.load will raise a clear FileNotFoundError
    return path_str


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


@torch.no_grad()
def load_flow_matching_model(checkpoint_path, device='cuda', compile_model=False):
    """Load a Flow Matching model from checkpoint"""
    checkpoint_path = _resolve_checkpoint_path(checkpoint_path)
    model = UNet2DModel(
        sample_size=32,
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
    
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    
    # Handle different checkpoint formats
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)
    
    model.to(device)
    model.eval()
    
    # Optional: compile model for faster inference (PyTorch 2.0+)
    if compile_model and hasattr(torch, 'compile'):
        print("Compiling model with torch.compile...")
        model = torch.compile(model, mode='reduce-overhead')
    
    return model


@torch.no_grad()
def load_ddpm_model(model_dir, device='cuda', compile_model=False):
    """Load a DDPM model from pretrained directory"""
    resolved_dir = _resolve_pretrained_path(model_dir)
    # local_files_only=True prevents accidental network calls on clusters
    model = UNet2DModel.from_pretrained(resolved_dir, local_files_only=True)
    model.to(device)
    model.eval()
    
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
    """Load a Diff2Flow FlowModelObj from checkpoint.

    If ``use_lora`` is True, we inject LoRA adapters into the UNet with the
    same hyperparameters used during training so the state dict matches.
    """

    resolved_unet_dir = _resolve_pretrained_path(pretrained_unet_dir)
    # local_files_only=True prevents accidental network calls on clusters
    unet = UNet2DModel.from_pretrained(resolved_unet_dir, local_files_only=True)

    if use_lora:
        if apply_lora is None or mark_only_lora_as_trainable is None:
            raise RuntimeError(
                "LoRA evaluation requested (use_lora=True) but utils.lora_utils "
                "could not be imported. Please ensure utils/lora_utils.py is "
                "available or set use_lora=False."
            )
        print(f"Applying LoRA to UNet for Diff2Flow (r={lora_r}, alpha={lora_alpha}, dropout={lora_dropout})")
        apply_lora(unet, r=lora_r, lora_alpha=lora_alpha, lora_dropout=lora_dropout)
        # Not strictly necessary for evaluation, but keeps the model consistent
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
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    if 'model_state_dict' in checkpoint:
        flow_model.load_state_dict(checkpoint['model_state_dict'])
    else:
        flow_model.load_state_dict(checkpoint)

    flow_model.to(device)
    flow_model.eval()
    
    if compile_model and hasattr(torch, 'compile'):
        print("Compiling Diff2Flow model with torch.compile...")
        flow_model = torch.compile(flow_model, mode='reduce-overhead')

    return flow_model


def load_real_images_efficiently(num_samples, batch_size=256):
    """Load real images more efficiently using DataLoader"""
    transform = transforms.Compose([
        transforms.ToTensor(),
    ])
    
    dataset = datasets.CIFAR10(
        root=DATA_ROOT,
        train=False,
        download=False,
        transform=transform
    )
    
    # Sample random indices
    indices = np.random.choice(len(dataset), num_samples, replace=False)
    subset = Subset(dataset, indices)
    
    # Use DataLoader for efficient batched loading
    loader = DataLoader(
        subset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )
    
    real_images = []
    for batch, _ in tqdm(loader, desc="Loading real images"):
        real_images.append(batch)
    
    return torch.cat(real_images, dim=0)


def evaluate_model(model_path, model_type='flow_matching', num_samples=10000, 
                   batch_size=128, device='cuda', use_amp=False, compile_model=False,
                   num_workers=4, **kwargs):
    """
    Evaluate a generative model with optimizations.
    
    Args:
        model_path: Path to model checkpoint or directory
        model_type: 'flow_matching', 'ddpm', or 'diff2flow'
        num_samples: Number of samples to generate for FID calculation
        batch_size: Batch size for generation
        device: Device to use
        use_amp: Use automatic mixed precision (faster on modern GPUs)
        compile_model: Use torch.compile for faster inference (PyTorch 2.0+)
        num_workers: Number of workers for data loading
        **kwargs: Additional arguments
    
    Returns:
        dict with 'fid', 'nfe', and 'wall_time'
    """
    device = torch.device(device if torch.cuda.is_available() else 'cpu')
    
    # Load model
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
    
    # Load real CIFAR-10 data efficiently
    print("Loading CIFAR-10 dataset...")
    real_images = load_real_images_efficiently(num_samples, batch_size=256)
    
    # Generate samples and measure time
    print(f"Generating {num_samples} samples...")
    start_time = time.time()
    
    if model_type == 'flow_matching':
        method = kwargs.get('method', 'euler')
        num_steps = kwargs.get('num_steps', 100)
        rtol = kwargs.get('rtol', 1e-5)
        atol = kwargs.get('atol', 1e-5)
        
        evaluator = FlowMatchingEvaluator(model, device=device, use_amp=use_amp)
        generated_images, nfe = evaluator.generate_samples(
            num_samples, batch_size=batch_size, 
            method=method, num_steps=num_steps, rtol=rtol, atol=atol
        )
    elif model_type == 'ddpm':
        num_inference_steps = kwargs.get('num_inference_steps', 1000)
        evaluator = DDPMEvaluator(model, device=device, num_inference_steps=num_inference_steps, use_amp=use_amp)
        generated_images, nfe = evaluator.generate_samples(num_samples, batch_size=batch_size)
    else:  # diff2flow
        num_steps = kwargs.get('num_steps', 1000)
        method = kwargs.get('method', 'euler')
        evaluator = Diff2FlowEvaluator(model, device=device, num_steps=num_steps, method=method, use_amp=use_amp)
        generated_images, nfe = evaluator.generate_samples(num_samples, batch_size=batch_size)
    
    wall_time = time.time() - start_time
    
    # Calculate FID using pytorch-fid library
    print("Calculating FID score...")
    fid_value = calculate_fid_from_tensors(
        real_images, 
        generated_images, 
        batch_size=50,
        device=device,
        num_workers=num_workers
    )
    
    # Cleanup
    del real_images, generated_images
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    results = {
        'fid': fid_value,
        'nfe': nfe,
        'wall_time': wall_time,
        'samples_per_second': num_samples / wall_time,
    }
    
    print("\n" + "="*50)
    print(f"Evaluation Results for {model_type.upper()}")
    print("="*50)
    print(f"FID Score: {fid_value:.2f}")
    print(f"NFE (Average per sample): {nfe:.1f}")
    print(f"Wall Clock Time: {wall_time:.2f}s")
    print(f"Samples/Second: {num_samples/wall_time:.2f}")
    print("="*50)
    
    return results


def save_evaluation_results(results_dict, output_dir="logs", filename_prefix="metrics_results"):
    """Persist evaluation results as a JSON file.

    Args:
        results_dict: Mapping of model name -> metrics dict.
        output_dir: Directory where the file will be created.
        filename_prefix: Prefix for the results file name.
    """
    os.makedirs(output_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = os.path.join(output_dir, f"{filename_prefix}_{timestamp}.json")

    # Convert any non-serializable values (e.g. tensors) to plain types
    def _to_serializable(obj):
        if isinstance(obj, torch.Tensor):
            return obj.detach().cpu().tolist()
        if isinstance(obj, (np.floating, np.integer)):
            return obj.item()
        return obj

    serializable = {
        k: {mk: _to_serializable(mv) for mk, mv in v.items()} for k, v in results_dict.items()
    }

    with open(path, "w") as f:
        json.dump(serializable, f, indent=2)

    print(f"Saved evaluation results to {path}")

def main():
    """CLI entry-point to evaluate selected models.

    Use --model to choose which model(s) to run.
    """
    parser = argparse.ArgumentParser(description="Evaluate generative models on CIFAR-10")
    parser.add_argument(
        "--model",
        "-m",
        choices=["flow_matching", "ddpm", "diff2flow", "all"],
        default="all",
        help="Which model to evaluate (default: all)",
    )
    parser.add_argument(
        "--flow-matching-path",
        type=str,
        default="flow_matching_cifar10/final_model/model.pt",
        help="Path to flow-matching checkpoint (.pt)",
    )
    parser.add_argument(
        "--ddpm-dir",
        type=str,
        default="ddpm_cifar10/final_model",
        help="Directory of the DDPM UNet2DModel (diffusers format)",
    )
    parser.add_argument(
        "--diff2flow-path",
        type=str,
        default="diff2flow_cifar10/diff2flow_flowmodel.pt",
        help="Path to Diff2Flow checkpoint (.pt)",
    )

    args = parser.parse_args()

    results = {}

    # Flow matching
    if args.model in ("flow_matching", "all"):
        flow_results = evaluate_model(
            model_path=args.flow_matching_path,
            model_type='flow_matching',
            num_samples=10000,
            batch_size=128,
            method='euler',
            num_steps=100,
            use_amp=True,
            compile_model=False,
            num_workers=4,
        )
        results["flow_matching"] = flow_results

    # DDPM
    if args.model in ("ddpm", "all"):
        ddpm_results = evaluate_model(
            model_path=args.ddpm_dir,
            model_type='ddpm',
            num_samples=10000,
            batch_size=128,
            num_inference_steps=1000,
            use_amp=True,
            compile_model=True,
            num_workers=4,
        )
        results["ddpm"] = ddpm_results

    # Diff2Flow
    if args.model in ("diff2flow", "all"):
        diff2flow_results = evaluate_model(
            model_path=args.diff2flow_path,
            model_type='diff2flow',
            num_samples=10,
            batch_size=128,
            num_steps=1000,
            method='euler',
            pretrained_unet_dir="ddpm_cifar10/final_model",
            # Default to no LoRA to keep this runnable without extra utils
            use_lora=False,
            lora_r=4,
            lora_alpha=8.0,
            lora_dropout=0.0,
            use_amp=True,
            compile_model=True,
            num_workers=4,
        )
        results["diff2flow"] = diff2flow_results

    if not results:
        print("No models were evaluated. Check --model argument.")
        return

    # Save all results to disk for later analysis
    save_evaluation_results(results)

    # Optional comparison printout if more than one model was run
    if len(results) > 1:
        print("\n" + "=" * 50)
        print("COMPARISON")
        print("=" * 50)

        # Safely pull metrics, falling back if some models weren't run
        fm = results.get("flow_matching")
        dd = results.get("ddpm")
        d2f = results.get("diff2flow")

        if fm and dd and d2f:
            print(
                f"FID - Flow Matching: {fm['fid']:.2f} | "
                f"DDPM: {dd['fid']:.2f} | "
                f"Diff2Flow: {d2f['fid']:.2f}"
            )
            print(
                f"NFE - Flow Matching: {fm['nfe']:.1f} | "
                f"DDPM: {dd['nfe']:.1f} | "
                f"Diff2Flow: {d2f['nfe']:.1f}"
            )
            print(
                f"Time - Flow Matching: {fm['wall_time']:.2f}s | "
                f"DDPM: {dd['wall_time']:.2f}s | "
                f"Diff2Flow: {d2f['wall_time']:.2f}s"
            )
            print(
                f"Speed - Flow Matching: {fm['samples_per_second']:.2f} | "
                f"DDPM: {dd['samples_per_second']:.2f} | "
                f"Diff2Flow: {d2f['samples_per_second']:.2f} samples/s"
            )
            print("=" * 50)


if __name__ == "__main__":
    main()