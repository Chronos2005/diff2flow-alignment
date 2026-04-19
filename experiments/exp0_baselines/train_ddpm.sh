#!/bin/bash
#SBATCH --partition=ecsstudents_l4
#SBATCH --account=ecsstudents
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=12
#SBATCH --time=24:00:00
#SBATCH --output=../logs/exp0_ddpm_train_%j.out
#SBATCH --error=../logs/exp0_ddpm_train_%j.err

module load conda
source activate /home/ram1g23/.conda/envs/diffusion_flow_study
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

echo "Job $SLURM_JOB_ID on $(hostname)"
python - <<EOF
import torch
print("Torch:", torch.__version__, "| CUDA:", torch.cuda.is_available(), "| GPUs:", torch.cuda.device_count())
EOF

nvidia-smi --query-gpu=memory.used,memory.free,utilization.gpu --format=csv -l 60 &
NVPID=$!

accelerate launch --num_processes=2 --mixed_precision="bf16" \
  ../../training_scripts/train_ddpm.py \
  --dataset cifar10 \
  --num_epochs 1000 \
  --train_batch_size 512 \
  --seed 42 \
  --output_dir /scratch/ram1g23/Models/Cifar-10/ddpm_cifar10

kill $NVPID 2>/dev/null
