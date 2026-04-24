#!/bin/bash
#SBATCH --partition=ecsstudents_l4
#SBATCH --account=ecsstudents
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=6
#SBATCH --time=4:00:00
#SBATCH --output=../logs/exp8_eval_%j.out
#SBATCH --error=../logs/exp8_eval_%j.err

module load conda
source activate /home/ram1g23/.conda/envs/diffusion_flow_study
ln -sf ~/inception_cache/inception-2015-12-05.pt /tmp/inception-2015-12-05.pt
cp /iridisfs/home/ram1g23/Projects/diff2flow-alignment/inception-2015-12-05.pt /tmp/

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

RUN_TAG=${RUN_TAG:-${SLURM_JOB_ID:-$(date +%Y%m%d_%H%M%S)}}
CKPT=/scratch/ram1g23/exp8_lora_200epochs/final_model/diffusion_pytorch_model.safetensors
OUTPUT_DIR=/scratch/ram1g23/exp8_lora_200epochs/eval/${RUN_TAG}

echo "Evaluating LoRA rank=32 200-epoch checkpoint: ${CKPT}"

python ../../evaluation_scripts/metrics_diff2flow.py \
  --checkpoint_path ${CKPT} \
  --pretrained_model_path /scratch/ram1g23/ddpm_cifar10_long2/final_model_ema \
  --dataset cifar10 \
  --num_samples 10000 \
  --step_counts 2 4 10 25 50 \
  --output_dir ${OUTPUT_DIR} \
  --scratch_dir /scratch/ram1g23/exp8_tmp_${SLURM_JOB_ID} \
  --use_lora \
  --lora_rank 32
