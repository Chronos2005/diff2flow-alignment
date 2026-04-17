#!/bin/bash
#SBATCH --partition=ecsstudents_l4
#SBATCH --account=ecsstudents
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=12
#SBATCH --time=24:00:00
#SBATCH --output=logs/progressive_distill_%j.out
#SBATCH --error=logs/progressive_distill_%j.err

module load conda
source activate /home/ram1g23/.conda/envs/diffusion_flow_study

echo "Python: $(which python)"
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

echo "===== ENVIRONMENT CHECK ====="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $(hostname)"
echo "CUDA_VISIBLE_DEVICES: $CUDA_VISIBLE_DEVICES"
python - <<EOF
import torch
print("Torch version:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())
if torch.cuda.is_available():
    for i in range(torch.cuda.device_count()):
        print(f"GPU {i}: {torch.cuda.get_device_name(i)}")
EOF
echo "===== END CHECK ====="

# ---------------------------------------------------------------------------
# Progressive Distillation
#
# Distils a trained FM (or Diff2Flow) model from --initial_teacher_steps
# down to 1/8 of that in --num_rounds rounds.
#
# Example: 16 → 8 → 4 → 2 steps (--initial_teacher_steps 16 --num_rounds 3)
#
# Change --teacher_checkpoint to your saved model directory.
# Change --teacher_type to "diff2flow" if using a Diff2Flow checkpoint.
# ---------------------------------------------------------------------------

TEACHER_CKPT=/scratch/ram1g23/reflow_cifar10/final_model   # or diff2flow checkpoint
OUTPUT_DIR=/scratch/ram1g23/progressive_distill_cifar10

accelerate launch --num_processes=2 --mixed_precision="bf16" \
    ../training_scripts/train_progressive_distillation.py \
    --teacher_checkpoint "$TEACHER_CKPT" \
    --teacher_type fm \
    --dataset cifar10 \
    --image_size 32 \
    --initial_teacher_steps 16 \
    --num_rounds 3 \
    --num_epochs 20 \
    --steps_per_epoch 1000 \
    --train_batch_size 512 \
    --learning_rate 1e-5 \
    --num_warmup_steps 200 \
    --sigma_min 1e-4 \
    --save_images_every 5 \
    --save_model_every 10 \
    --output_dir "$OUTPUT_DIR"

kill $NVIDIA_SMI_PID 2>/dev/null
