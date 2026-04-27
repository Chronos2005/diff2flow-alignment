#!/bin/bash
#SBATCH --partition=ecsstudents_l4
#SBATCH --account=ecsstudents
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=6
#SBATCH --time=4:00:00
#SBATCH --array=1-7
#SBATCH --output=../logs/exp1_eval_%A_%a.out
#SBATCH --error=../logs/exp1_eval_%A_%a.err

module load conda
source activate /home/ram1g23/.conda/envs/diffusion_flow_study
ln -sf ~/inception_cache/inception-2015-12-05.pt /tmp/inception-2015-12-05.pt

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK



declare -a VARIANT_FLAGS=(
    ""
    "--no_interpolant_rescaling --no_velocity_translation"
    "--no_timestep_rescaling --no_velocity_translation"
    "--no_timestep_rescaling --no_interpolant_rescaling"
    "--no_velocity_translation"
    "--no_interpolant_rescaling"
    "--no_timestep_rescaling"
    ""
)
declare -a VARIANT_NAMES=("" "T" "I" "V" "TI" "TV" "IV" "TIV")

RUN_TAG=${RUN_TAG:-${SLURM_JOB_ID:-$(date +%Y%m%d_%H%M%S)}}
FLAGS=${VARIANT_FLAGS[$SLURM_ARRAY_TASK_ID]}
NAME=${VARIANT_NAMES[$SLURM_ARRAY_TASK_ID]}
CKPT=/scratch/ram1g23/exp1_alignment/${NAME}/final_model_ema/diffusion_pytorch_model.safetensors
OUTPUT_DIR=/scratch/ram1g23/exp1_alignment/${NAME}/eval/${RUN_TAG}

echo "Evaluating variant: ${NAME}  |  Checkpoint: ${CKPT}"

python ../../evaluation_scripts/metrics_diff2flow.py \
  --checkpoint_path ${CKPT} \
  --pretrained_model_path /scratch/ram1g23/ddpm_cifar10_long2/final_model_ema \
  --dataset cifar10 \
  --num_samples 10000 \
  --step_counts 2 4 10 25 50 \
  --output_dir ${OUTPUT_DIR} \
  --scratch_dir /scratch/ram1g23/exp1_tmp_${NAME}_${SLURM_JOB_ID} \
  ${FLAGS}
