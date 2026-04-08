#!/bin/bash
#SBATCH --partition=ecsstudents_l4
#SBATCH --account=ecsstudents
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=6
#SBATCH --time=4:00:00
#SBATCH --output=../logs/exp3_eval_%j.out
#SBATCH --error=../logs/exp3_eval_%j.err

module load conda
source activate /home/ram1g23/.conda/envs/diffusion_flow_study

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

cp /iridisfs/home/ram1g23/Projects/diff2flow-alignment/inception-2015-12-05.pt /tmp/

# ---------------------------------------------------------------------------
# Exp 3: Low-NFE Sampling
#
# Evaluates the best fully-aligned (T+I+V) checkpoint at very low step counts
# to measure the speed--quality trade-off curve.
#
# Uses the TIV checkpoint from Exp 1. Update CKPT if using a different run.
# ---------------------------------------------------------------------------

CKPT=/scratch/ram1g23/exp1_alignment/TIV/final_model/diffusion_pytorch_model.safetensors
OUTPUT_DIR=/scratch/ram1g23/exp3_low_nfe/eval

echo "Job ID: $SLURM_JOB_ID"
echo "Node: $(hostname)"
echo "Checkpoint: ${CKPT}"

python ../../evaluation_scripts/metrics_diff2flow.py \
  --checkpoint_path ${CKPT} \
  --pretrained_model_path /scratch/ram1g23/Models/Cifar-10/ddpm_cifar10/final_model \
  --dataset cifar10 \
  --num_samples 10000 \
  --step_counts 2 4 8 16 50 \
  --output_dir ${OUTPUT_DIR} \
  --scratch_dir /scratch/ram1g23/exp3_tmp_${SLURM_JOB_ID}
