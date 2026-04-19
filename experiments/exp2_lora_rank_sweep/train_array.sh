#!/bin/bash
#SBATCH --partition=ecsstudents_l4
#SBATCH --account=ecsstudents
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=12
#SBATCH --time=24:00:00
#SBATCH --array=1-7
#SBATCH --output=../logs/exp2_train_%A_%a.out
#SBATCH --error=../logs/exp2_train_%A_%a.err

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
# Exp 2: LoRA Rank Sweep
#
#  Index | Variant      | Flags
#  ------+--------------+--------------------------------
#    1   | rank 4       | --use_lora --lora_rank 4
#    2   | rank 8       | --use_lora --lora_rank 8
#    3   | rank 16      | --use_lora --lora_rank 16
#    4   | rank 32      | --use_lora --lora_rank 32
#    5   | rank 64      | --use_lora --lora_rank 64
#    6   | rank 128     | --use_lora --lora_rank 128
#    7   | full finetune| (no LoRA)
# ---------------------------------------------------------------------------

declare -a RANKS=(0 4 8 16 32 64 128 0)
declare -a NAMES=("" "rank4" "rank8" "rank16" "rank32" "rank64" "rank128" "full_ft")

IDX=$SLURM_ARRAY_TASK_ID
NAME=${NAMES[$IDX]}
RANK=${RANKS[$IDX]}
OUTPUT_DIR=/scratch/ram1g23/exp2_lora_rank/${NAME}

if [ $IDX -eq 7 ]; then
    LORA_FLAGS=""
    echo "Variant: full finetuning (no LoRA)"
else
    LORA_FLAGS="--use_lora --lora_rank ${RANK}"
    echo "Variant: LoRA rank=${RANK}"
fi

echo "Output: ${OUTPUT_DIR}"

accelerate launch --num_processes=2 --mixed_precision="bf16" ../../training_scripts/train_diff2flow.py \
  --pretrained_model_path /scratch/ram1g23/ddpm_cifar10_long2/final_model_ema \
  --dataset cifar10 \
  --num_epochs 20 \
  --save_images_every 5 \
  --save_model_every 5 \
  --num_inference_steps 50 \
  --seed 42 \
  --output_dir ${OUTPUT_DIR} \
  ${LORA_FLAGS}
