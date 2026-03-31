import torch
import numpy as np
from torchvision import transforms, datasets
from torch.utils.data import DataLoader, Subset
import time
import os
from tqdm import tqdm
from diffusers import UNet2DModel, DDPMScheduler, DDPMPipeline
from torchdiffeq import odeint
from pytorch_fid import fid_score
from PIL import Image
import tempfile
import shutil
from torch.amp import autocast
import gc

from cifar import DATA_ROOT
from flow_obj import FlowModelObj
from utils.lora_utils import apply_lora, mark_only_lora_as_trainable


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
            

            
            # Denormalize from [-1, 1] to [0, 1]
            samples = (samples + 1) / 2
            samples = torch.clamp(samples, 0, 1)
            
            all_samples.append(samples.cpu())
            
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
            
            # Denormalize from [-1, 1] to [0, 1]
            samples = (image + 1) / 2
            samples = torch.clamp(samples, 0, 1)
            
            all_samples.append(samples.cpu())
            
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

            # Denormalize from [-1, 1] to [0, 1]
            samples = (samples + 1) / 2
            samples = torch.clamp(samples, 0, 1)
            all_samples.append(samples.cpu())
            
            # Clear GPU cache periodically
            if (i + 1) % 10 == 0:
                torch.cuda.empty_cache()

        all_samples = torch.cat(all_samples, dim=0)[:num_samples]
        avg_nfe = self.nfe_counter.nfe / num_samples if self.method == "euler" else float("nan")

        return all_samples, avg_nfe


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
    model = UNet2DModel.from_pretrained(model_dir)
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

    unet = UNet2DModel.from_pretrained(pretrained_unet_dir)

    if use_lora:
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


# Example usage
if __name__ == "__main__":
    # Evaluate with optimizations enabled
    flow_results = evaluate_model(
        model_path="flow_matching_cifar10/final_model/model.pt",
        model_type='flow_matching',
        num_samples=10000,
        batch_size=128,
        method='euler',
        num_steps=100,
        use_amp=True,
        compile_model=False,
        num_workers=4
    )
    
    ddpm_results = evaluate_model(
        model_path="ddpm_cifar10/final_model",
        model_type='ddpm',
        num_samples=10000,
        batch_size=128,
        num_inference_steps=1000,
        use_amp=True,
        compile_model=True,
        num_workers=4
    )

    diff2flow_results = evaluate_model(
        model_path="diff2flow_cifar10/diff2flow_flowmodel.pt",
        model_type='diff2flow',
        num_samples=10,
        batch_size=128,
        num_steps=1000,
        method='euler',
        pretrained_unet_dir="ddpm_cifar10/final_model",
        use_lora=True,
        lora_r=4,
        lora_alpha=8.0,
        lora_dropout=0.0,
        use_amp=True,
        compile_model=True,
        num_workers=4
    )
    
    # Compare results
    print("\n" + "="*50)
    print("COMPARISON")
    print("="*50)
    print(
        f"FID - Flow Matching: {flow_results['fid']:.2f} | "
        f"DDPM: {ddpm_results['fid']:.2f} | "
        f"Diff2Flow: {diff2flow_results['fid']:.2f}"
    )
    print(
        f"NFE - Flow Matching: {flow_results['nfe']:.1f} | "
        f"DDPM: {ddpm_results['nfe']:.1f} | "
        f"Diff2Flow: {diff2flow_results['nfe']:.1f}"
    )
    print(
        f"Time - Flow Matching: {flow_results['wall_time']:.2f}s | "
        f"DDPM: {ddpm_results['wall_time']:.2f}s | "
        f"Diff2Flow: {diff2flow_results['wall_time']:.2f}s"
    )
    print(
        f"Speed - Flow Matching: {flow_results['samples_per_second']:.2f} | "
        f"DDPM: {ddpm_results['samples_per_second']:.2f} | "
        f"Diff2Flow: {diff2flow_results['samples_per_second']:.2f} samples/s"
    )
    print("="*50)