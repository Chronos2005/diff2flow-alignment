#!/bin/bash
#SBATCH --partition=ecsstudents_l4
#SBATCH --account=ecsstudents
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=12
#SBATCH --time=24:00:00
#SBATCH --array=1-3
#SBATCH --output=../logs/exp5_train_%A_%a.out
#SBATCH --error=../logs/exp5_train_%A_%a.err

module load conda
source activate /home/ram1g23/.conda/envs/diffusion_flow_study

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

echo "===== ENVIRONMENT CHECK ====="
echo "Job ID: $SLURM_JOB_ID  Array Task: $SLURM_ARRAY_TASK_ID"
echo "Node: $(hostname)"
python - <<EOF
import torch
print("Torch:", torch.__version__, "| CUDA:", torch.cuda.is_available(), "| GPUs:", torch.cuda.device_count())
EOF
echo "===== END CHECK ====="

# ---------------------------------------------------------------------------
# Exp 5: LoRA Placement Ablation
#
#  Index | Placement   | Description
#  ------+-------------+------------------------------------------
#    1   | all         | LoRA on all Linear and Conv2d layers
#    2   | attention   | LoRA only on attention block layers
#    3   | feedforward | LoRA only on non-attention layers (conv, time emb, etc.)
# ---------------------------------------------------------------------------

declare -a PLACEMENTS=("" "all" "attention" "feedforward")
declare -a NAMES=("" "lora_all" "lora_attention" "lora_feedforward")

PLACEMENT=${PLACEMENTS[$SLURM_ARRAY_TASK_ID]}
NAME=${NAMES[$SLURM_ARRAY_TASK_ID]}
OUTPUT_DIR=/scratch/ram1g23/exp5_placement/${NAME}

echo "Placement: ${PLACEMENT}  |  Output: ${OUTPUT_DIR}"

accelerate launch --num_processes=2 ../../training_scripts/train_diff2flow.py \
  --pretrained_model_path /scratch/ram1g23/Models/Cifar-10/ddpm_cifar10/final_model \
  --dataset cifar10 \
  --num_epochs 20 \
  --save_images_every 5 \
  --save_model_every 5 \
  --num_inference_steps 50 \
  --use_lora \
  --lora_rank 64 \
  --lora_placement ${PLACEMENT} \
  --output_dir ${OUTPUT_DIR}
