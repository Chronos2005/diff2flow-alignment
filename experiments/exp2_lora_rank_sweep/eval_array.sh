#!/bin/bash
#SBATCH --partition=ecsstudents_l4
#SBATCH --account=ecsstudents
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=6
#SBATCH --time=4:00:00
#SBATCH --array=1-7
#SBATCH --output=../logs/exp2_eval_%A_%a.out
#SBATCH --error=../logs/exp2_eval_%A_%a.err

module load conda
source activate /home/ram1g23/.conda/envs/diffusion_flow_study

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

cp /iridisfs/home/ram1g23/Projects/diff2flow-alignment/inception-2015-12-05.pt /tmp/

declare -a RANKS=(0 4 8 16 32 64 128 0)
declare -a NAMES=("" "rank4" "rank8" "rank16" "rank32" "rank64" "rank128" "full_ft")

IDX=$SLURM_ARRAY_TASK_ID
NAME=${NAMES[$IDX]}
RANK=${RANKS[$IDX]}
CKPT=/scratch/ram1g23/exp2_lora_rank/${NAME}/final_model/diffusion_pytorch_model.safetensors
OUTPUT_DIR=/scratch/ram1g23/exp2_lora_rank/${NAME}/eval

if [ $IDX -eq 7 ]; then
    LORA_FLAGS=""
    echo "Evaluating: full finetuning"
else
    LORA_FLAGS="--use_lora --lora_rank ${RANK}"
    echo "Evaluating: LoRA rank=${RANK}"
fi

echo "Checkpoint: ${CKPT}"

python ../../evaluation_scripts/metrics_diff2flow.py \
  --checkpoint_path ${CKPT} \
  --pretrained_model_path /scratch/ram1g23/Models/Cifar-10/ddpm_cifar10/final_model \
  --dataset cifar10 \
  --num_samples 10000 \
  --step_counts 10 25 50 \
  --output_dir ${OUTPUT_DIR} \
  --scratch_dir /scratch/ram1g23/exp2_tmp_${NAME}_${SLURM_JOB_ID} \
  ${LORA_FLAGS}
