#!/bin/bash
#SBATCH --partition=ecsstudents_l4
#SBATCH --account=ecsstudents
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=12
#SBATCH --time=24:00:00
#SBATCH --output=../logs/exp8_train_%j.out
#SBATCH --error=../logs/exp8_train_%j.err

module load conda
source activate /home/ram1g23/.conda/envs/diffusion_flow_study

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

echo "===== ENVIRONMENT CHECK ====="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $(hostname)"
python - <<EOF
import torch
print("Torch:", torch.__version__, "| CUDA:", torch.cuda.is_available(), "| GPUs:", torch.cuda.device_count())
EOF
echo "===== END CHECK ====="

OUTPUT_DIR=/scratch/ram1g23/exp8_lora_200epochs

echo "Training LoRA rank=32 for 200 epochs -> ${OUTPUT_DIR}"

accelerate launch --num_processes=2 --mixed_precision="bf16" ../../training_scripts/train_diff2flow.py \
  --pretrained_model_path /scratch/ram1g23/ddpm_cifar10_long2/final_model_ema \
  --dataset cifar10 \
  --num_epochs 200 \
  --save_images_every 20 \
  --save_model_every 20 \
  --num_inference_steps 50 \
  --seed 42 \
  --output_dir ${OUTPUT_DIR} \
  --use_lora \
  --lora_rank 32
