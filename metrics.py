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

from cifar import DATA_ROOT
from flow_obj import FlowModelObj


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
    
    def __init__(self, model, device='cuda', sigma_min=1e-4):
        self.model = model
        self.device = device
        self.sigma_min = sigma_min
        self.nfe_counter = NFECounter()
    
    def create_ode_func(self):
        """Create ODE function that counts evaluations"""
        class ODEFunc(torch.nn.Module):
            def __init__(self, model, nfe_counter):
                super().__init__()
                self.model = model
                self.nfe_counter = nfe_counter
            
            def forward(self, t, x):
                self.nfe_counter.nfe += 1
                batch_size = x.shape[0]
                t_scaled = (t * 999).expand(batch_size).to(x.device)
                
                with torch.no_grad():
                    v = self.model(x, t_scaled, return_dict=False)[0]
                return v
        
        return ODEFunc(self.model, self.nfe_counter)
    
    def generate_samples(self, num_samples, batch_size=128, method='dopri5', rtol=1e-5, atol=1e-5):
        """Generate samples and track NFE"""
        self.model.eval()
        all_samples = []
        self.nfe_counter.reset()
        
        num_batches = (num_samples + batch_size - 1) // batch_size
        
        for i in tqdm(range(num_batches), desc="Generating samples"):
            current_batch_size = min(batch_size, num_samples - i * batch_size)
            
            # Start from noise
            x0 = torch.randn(current_batch_size, 3, 32, 32, device=self.device)
            
            # Create ODE function
            ode_func = self.create_ode_func()
            
            # Integration time span
            t_span = torch.tensor([0.0, 1.0], device=self.device)
            
            # Solve ODE
            with torch.no_grad():
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
        
        all_samples = torch.cat(all_samples, dim=0)[:num_samples]
        avg_nfe = self.nfe_counter.nfe / num_samples
        
        return all_samples, avg_nfe


class DDPMEvaluator:
    """Evaluator for DDPM models"""
    
    def __init__(self, model, device='cuda', num_inference_steps=1000):
        self.model = model
        self.device = device
        self.num_inference_steps = num_inference_steps
        self.scheduler = DDPMScheduler(num_train_timesteps=1000)
        self.nfe_counter = NFECounter()
    
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
                    # Predict noise
                    model_output = self.model(image, t, return_dict=False)[0]
                    self.nfe_counter.nfe += current_batch_size
                    
                    # Compute previous image
                    image = self.scheduler.step(model_output, t, image, return_dict=False)[0]
            
            # Denormalize from [-1, 1] to [0, 1]
            samples = (image + 1) / 2
            samples = torch.clamp(samples, 0, 1)
            
            all_samples.append(samples.cpu())
        
        all_samples = torch.cat(all_samples, dim=0)[:num_samples]
        avg_nfe = self.nfe_counter.nfe / num_samples
        
        return all_samples, avg_nfe


class Diff2FlowEvaluator:
    """Evaluator for Diff2Flow FlowModelObj"""
    def __init__(self, model, device='cuda', num_steps=1000, method='euler'):
        self.model = model
        self.device = device
        self.num_steps = num_steps
        self.method = method
        self.nfe_counter = NFECounter()

    def generate_samples(self, num_samples, batch_size=128):
        self.model.eval()
        all_samples = []
        self.nfe_counter.reset()

        num_batches = (num_samples + batch_size - 1) // batch_size

        for i in tqdm(range(num_batches), desc="Generating samples"):
            current_batch_size = min(batch_size, num_samples - i * batch_size)
            z = torch.randn(current_batch_size, 3, 32, 32, device=self.device)

            with torch.no_grad():
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

        all_samples = torch.cat(all_samples, dim=0)[:num_samples]
        avg_nfe = self.nfe_counter.nfe / num_samples if self.method == "euler" else float("nan")

        return all_samples, avg_nfe


def save_images_to_dir(images, output_dir):
    """Save tensor images to directory for FID calculation"""
    os.makedirs(output_dir, exist_ok=True)
    
    for idx, img_tensor in enumerate(tqdm(images, desc=f"Saving images to {output_dir}")):
        # Convert tensor to PIL Image
        img_array = (img_tensor.numpy().transpose(1, 2, 0) * 255).astype(np.uint8)
        img = Image.fromarray(img_array)
        img.save(os.path.join(output_dir, f'{idx:05d}.png'))


def calculate_fid_from_tensors(real_images, generated_images, batch_size=50, device='cuda'):
    """
    Calculate FID using pytorch-fid library.
    
    Args:
        real_images: Tensor of real images [N, 3, H, W] in range [0, 1]
        generated_images: Tensor of generated images [N, 3, H, W] in range [0, 1]
        batch_size: Batch size for FID calculation
        device: Device to use
    
    Returns:
        FID score
    """
    # Create temporary directories
    with tempfile.TemporaryDirectory() as temp_dir:
        real_dir = os.path.join(temp_dir, 'real')
        gen_dir = os.path.join(temp_dir, 'generated')
        
        # Save images to directories
        print("Preparing images for FID calculation...")
        save_images_to_dir(real_images, real_dir)
        save_images_to_dir(generated_images, gen_dir)
        
        # Calculate FID using pytorch-fid
        print("Calculating FID score...")
        fid_value = fid_score.calculate_fid_given_paths(
            [real_dir, gen_dir],
            batch_size=batch_size,
            device=device,
            dims=2048,  # InceptionV3 feature dimension
            num_workers=0
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


def load_flow_matching_model(checkpoint_path, device='cuda'):
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
    
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # Handle different checkpoint formats
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)
    
    model.to(device)
    model.eval()
    
    return model


def load_ddpm_model(model_dir, device='cuda'):
    """Load a DDPM model from pretrained directory"""
    model = UNet2DModel.from_pretrained(model_dir)
    model.to(device)
    model.eval()
    
    return model


def load_diff2flow_model(checkpoint_path, pretrained_unet_dir, device='cuda',
                         num_timesteps=1000, diffusion_parameterization='eps',
                         enforce_zero_snr=False):
    """Load a Diff2Flow FlowModelObj from checkpoint"""
    unet = UNet2DModel.from_pretrained(pretrained_unet_dir)
    wrapped_unet = DiffusersUNetWrapper(unet)

    flow_model = FlowModelObj(
        net_cfg=wrapped_unet,
        schedule="linear",
        diffusion_parameterization=diffusion_parameterization,
        enforce_zero_snr=enforce_zero_snr,
    )

    scheduler = DDPMScheduler(num_train_timesteps=num_timesteps)
    register_schedule_from_betas(flow_model, scheduler.betas)

    checkpoint = torch.load(checkpoint_path, map_location=device)
    if 'model_state_dict' in checkpoint:
        flow_model.load_state_dict(checkpoint['model_state_dict'])
    else:
        flow_model.load_state_dict(checkpoint)

    flow_model.to(device)
    flow_model.eval()

    return flow_model


def evaluate_model(model_path, model_type='flow_matching', num_samples=10000, 
                   batch_size=128, device='cuda', **kwargs):
    """
    Evaluate a generative model.
    
    Args:
        model_path: Path to model checkpoint or directory
        model_type: 'flow_matching', 'ddpm', or 'diff2flow'
        num_samples: Number of samples to generate for FID calculation
        batch_size: Batch size for generation
        device: Device to use
        **kwargs: Additional arguments:
            - method, rtol, atol for flow matching
            - num_inference_steps for ddpm
    
    Returns:
        dict with 'fid', 'nfe', and 'wall_time'
    """
    device = torch.device(device if torch.cuda.is_available() else 'cpu')
    
    # Load model
    print(f"Loading {model_type} model from {model_path}...")
    if model_type == 'flow_matching':
        model = load_flow_matching_model(model_path, device=device)
    elif model_type == 'ddpm':
        model = load_ddpm_model(model_path, device=device)
    else:  # diff2flow
        model = load_diff2flow_model(
            checkpoint_path=model_path,
            pretrained_unet_dir=kwargs.get("pretrained_unet_dir", "ddpm_cifar10/final_model"),
            device=device,
            num_timesteps=kwargs.get("num_timesteps", 1000),
            diffusion_parameterization=kwargs.get("diffusion_parameterization", "eps"),
            enforce_zero_snr=kwargs.get("enforce_zero_snr", False),
        )
    
    # Load real CIFAR-10 data
    print("Loading CIFAR-10 dataset...")
    transform = transforms.Compose([
        transforms.ToTensor(),
    ])
    
    dataset = datasets.CIFAR10(
        root=DATA_ROOT,
        train=False,  # Use test set for evaluation
        download=False,
        transform=transform
    )
    
    # Sample random real images
    print(f"Sampling {num_samples} real images...")
    indices = np.random.choice(len(dataset), num_samples, replace=False)
    real_images = torch.stack([dataset[i][0] for i in tqdm(indices, desc="Loading real images")])
    
    # Generate samples and measure time
    print(f"Generating {num_samples} samples...")
    start_time = time.time()
    
    if model_type == 'flow_matching':
        method = kwargs.get('method', 'dopri5')
        rtol = kwargs.get('rtol', 1e-5)
        atol = kwargs.get('atol', 1e-5)
        
        evaluator = FlowMatchingEvaluator(model, device=device)
        generated_images, nfe = evaluator.generate_samples(
            num_samples, batch_size=batch_size, 
            method=method, rtol=rtol, atol=atol
        )
    elif model_type == 'ddpm':
        num_inference_steps = kwargs.get('num_inference_steps', 1000)
        evaluator = DDPMEvaluator(model, device=device, num_inference_steps=num_inference_steps)
        generated_images, nfe = evaluator.generate_samples(num_samples, batch_size=batch_size)
    else:  # diff2flow
        num_steps = kwargs.get('num_steps', 1000)
        method = kwargs.get('method', 'euler')
        evaluator = Diff2FlowEvaluator(model, device=device, num_steps=num_steps, method=method)
        generated_images, nfe = evaluator.generate_samples(num_samples, batch_size=batch_size)
    
    wall_time = time.time() - start_time
    
    # Calculate FID using pytorch-fid library
    print("Calculating FID score...")
    fid_value = calculate_fid_from_tensors(
        real_images, 
        generated_images, 
        batch_size=50,  # FID batch size
        device=device
    )
    
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
    # Evaluate Flow Matching model
    flow_results = evaluate_model(
        model_path="flow_matching_cifar10/final_model/model.pt",
        model_type='flow_matching',
        num_samples=10000,
        batch_size=128,
        method='dopri5',
        rtol=1e-5,
        atol=1e-5
    )
    
    # Evaluate DDPM model
    ddpm_results = evaluate_model(
        model_path="ddpm_cifar10/final_model",
        model_type='ddpm',
        num_samples=10000,
        batch_size=128,
        num_inference_steps=1000
    )

    # Evaluate Diff2Flow model
    diff2flow_results = evaluate_model(
        model_path="diff2flow_cifar10/diff2flow_flowmodel.pt",
        model_type='diff2flow',
        num_samples=10000,
        batch_size=128,
        num_steps=1000,
        method='euler',
        pretrained_unet_dir="ddpm_cifar10/final_model",
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
