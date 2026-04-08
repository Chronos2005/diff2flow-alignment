#!/bin/bash
#SBATCH --partition=ecsstudents_l4
#SBATCH --account=ecsstudents
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=12
#SBATCH --time=24:00:00
#SBATCH --array=1-7
#SBATCH --output=../logs/exp1_train_%A_%a.out
#SBATCH --error=../logs/exp1_train_%A_%a.err

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
# Exp 1: Alignment Ablation — all 7 non-empty subsets of {T, I, V}
#
#  Index | Variant | Active components
#  ------+---------+------------------------------------------
#    1   |  T      | timestep rescaling only
#    2   |  I      | interpolant rescaling only
#    3   |  V      | velocity translation only
#    4   |  T+I    | timestep + interpolant
#    5   |  T+V    | timestep + velocity
#    6   |  I+V    | interpolant + velocity
#    7   |  T+I+V  | full alignment (baseline)
# ---------------------------------------------------------------------------

declare -a VARIANT_FLAGS=(
    ""                                                                    # unused (1-indexed)
    "--no_interpolant_rescaling --no_velocity_translation"                # 1: T
    "--no_timestep_rescaling --no_velocity_translation"                   # 2: I
    "--no_timestep_rescaling --no_interpolant_rescaling"                  # 3: V
    "--no_velocity_translation"                                           # 4: T+I
    "--no_interpolant_rescaling"                                          # 5: T+V
    "--no_timestep_rescaling"                                             # 6: I+V
    ""                                                                    # 7: T+I+V (full)
)
declare -a VARIANT_NAMES=("" "T" "I" "V" "TI" "TV" "IV" "TIV")

FLAGS=${VARIANT_FLAGS[$SLURM_ARRAY_TASK_ID]}
NAME=${VARIANT_NAMES[$SLURM_ARRAY_TASK_ID]}
OUTPUT_DIR=/scratch/ram1g23/exp1_alignment/${NAME}

echo "Variant: ${NAME}  |  Flags: ${FLAGS:-none}  |  Output: ${OUTPUT_DIR}"

accelerate launch --num_processes=2 ../../training_scripts/train_diff2flow.py \
  --pretrained_model_path /scratch/ram1g23/Models/Cifar-10/ddpm_cifar10/final_model \
  --dataset cifar10 \
  --num_epochs 20 \
  --save_images_every 5 \
  --save_model_every 5 \
  --num_inference_steps 50 \
  --output_dir ${OUTPUT_DIR} \
  ${FLAGS}
