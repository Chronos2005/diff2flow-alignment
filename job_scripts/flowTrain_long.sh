#!/bin/bash
#SBATCH --partition=ecsstudents_l4
#SBATCH --account=ecsstudents
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=12
#SBATCH --time=24:00:00
#SBATCH --output=logs/fm_long_%j.out
#SBATCH --error=logs/fm_long_%j.err

module load conda

source activate /home/ram1g23/.conda/envs/diffusion_flow_study

echo "Python: $(which python)"

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

echo "===== ENVIRONMENT CHECK ====="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $(hostname)"
echo "Python: $(which python)"
echo "Conda env: $CONDA_DEFAULT_ENV"
echo "Conda prefix: $CONDA_PREFIX"
echo "PATH: $PATH"
echo "CUDA_VISIBLE_DEVICES: $CUDA_VISIBLE_DEVICES"
echo "----- PYTORCH & CUDA -----"

python - <<EOF
import torch
print("Torch version:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("GPUs detected:", torch.cuda.device_count())
    for i in range(torch.cuda.device_count()):
        print(f"GPU {i}: {torch.cuda.get_device_name(i)}")
EOF

echo "===== END CHECK ====="

accelerate launch --num_processes=2 ../training_scripts/train_flow_matching.py \
    --dataset cifar10 \
    --num_epochs 1000 \
    --output_dir /scratch/ram1g23/fm_cifar10_long
