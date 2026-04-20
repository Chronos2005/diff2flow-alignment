"""
Shared argparse helpers.

Each function adds a coherent group of arguments to an existing parser.
Scripts call whichever helpers apply to them, then add any script-specific
arguments on top.

Usage example:
    parser = argparse.ArgumentParser(...)
    add_common_args(parser)
    add_dataset_args(parser)
    add_eval_args(parser)
    parser.add_argument("--model_path", type=str, required=True)
    args = parser.parse_args()
"""


def add_common_args(parser):
    """Seed and device — used by almost every script."""
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default=None,
                        help="Device (default: cuda if available, else cpu)")


def add_dataset_args(parser, include_custom=True, dataset_required=False):
    """Dataset selection and data root."""
    choices = ["cifar10", "celeba", "custom"] if include_custom else ["cifar10", "celeba"]
    parser.add_argument("--dataset", type=str,
                        required=dataset_required,
                        default=None if dataset_required else "cifar10",
                        choices=choices)
    parser.add_argument("--data_root", type=str, default=None,
                        help="Dataset root or folder of real images (custom)")


def add_eval_args(parser):
    """Common evaluation settings shared by all metrics_* scripts."""
    parser.add_argument("--num_samples", type=int, default=10000,
                        help="Number of samples to generate for FID")
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--keep_samples", action="store_true",
                        help="Keep generated sample folders after FID computation")


def add_training_args(parser, learning_rate_default=1e-4):
    """Training loop args shared across train_* scripts."""
    parser.add_argument("--num_epochs", type=int, default=None)
    parser.add_argument("--learning_rate", type=float, default=learning_rate_default)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--save_images_every", type=int, default=5)
    parser.add_argument("--save_model_every", type=int, default=10)


def add_lora_args(parser):
    """LoRA finetuning args used by diff2flow scripts."""
    parser.add_argument("--use_lora", action="store_true",
                        help="Use LoRA for parameter-efficient finetuning")
    parser.add_argument("--lora_rank", type=int, default=64,
                        help="LoRA rank (only used if --use_lora)")
    parser.add_argument("--lora_alpha", type=float, default=None,
                        help="LoRA alpha; effective scale = alpha/rank. "
                             "Defaults to rank, giving scale=1.0")
    parser.add_argument("--lora_placement", type=str, default="all",
                        choices=["all", "attention", "feedforward"],
                        help="Which layer types to apply LoRA to (default: all)")


def add_diffusion_args(parser):
    """Diffusion schedule args used by DDPM and Diff2Flow scripts."""
    parser.add_argument("--num_train_timesteps", type=int, default=1000)


def add_inference_args(parser, output_dir_default="samples", num_steps_default=None):
    """Image generation args shared by inference_fm and inference_diff2flow."""
    if num_steps_default is None:
        num_steps_default = [50]
    parser.add_argument("--output_dir", type=str, default=output_dir_default)
    parser.add_argument("--num_images", type=int, default=16,
                        help="Number of images to generate")
    parser.add_argument("--num_steps", type=int, nargs="+", default=num_steps_default,
                        help="Solver steps. Pass multiple values to compare, e.g. --num_steps 10 50 100")
    parser.add_argument("--image_size", type=int, default=32)
    parser.add_argument("--batch_size", type=int, default=16,
                        help="Images to generate per batch")
    parser.add_argument("--save_individual", action="store_true",
                        help="Also save each image individually")


def add_alignment_args(parser):
    """Alignment component ablation flags for Exp 1."""
    parser.add_argument("--no_timestep_rescaling", action="store_true",
                        help="Disable timestep remapping (T component)")
    parser.add_argument("--no_interpolant_rescaling", action="store_true",
                        help="Disable interpolant rescaling (I component)")
    parser.add_argument("--no_velocity_translation", action="store_true",
                        help="Disable velocity translation (V component)")


def add_metrics_output_args(parser, output_dir_default, scratch_dir_default,
                             step_counts_default=None):
    """Output/scratch dirs and step-count sweep args shared by all metrics_* scripts."""
    if step_counts_default is None:
        step_counts_default = [10, 25, 50, 100]
    parser.add_argument("--image_size", type=int, default=32)
    parser.add_argument("--step_counts", type=int, nargs="+", default=step_counts_default,
                        help="Step counts to evaluate")
    parser.add_argument("--output_dir", type=str, default=output_dir_default)
    parser.add_argument("--scratch_dir", type=str, default=scratch_dir_default,
                        help="Scratch directory for temporary generated/real images")
