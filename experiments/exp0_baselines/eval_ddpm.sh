#!/bin/bash
#SBATCH --partition=ecsstudents_l4
#SBATCH --account=ecsstudents
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=6
#SBATCH --time=6:00:00
#SBATCH --output=../logs/exp0_ddpm_%j.out
#SBATCH --error=../logs/exp0_ddpm_%j.err

# ---------------------------------------------------------------------------
# Baseline 1: DDPM
#
# Run 1: Full ancestral sampling (1000 steps) — the primary DDPM baseline FID.
# Run 2: DDIM at 10/25/50/100/250 steps — NFE curve for comparison.
#
# Model: /scratch/ram1g23/Models/Cifar-10/ddpm_cifar10/final_model
# ---------------------------------------------------------------------------

module load conda
source activate /home/ram1g23/.conda/envs/diffusion_flow_study
ln -sf ~/inception_cache/inception-2015-12-05.pt /tmp/inception-2015-12-05.pt
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

MODEL=/scratch/ram1g23/Models/Cifar-10/ddpm_cifar10/final_model
OUTPUT=/scratch/ram1g23/baselines/ddpm_eval
SCRATCH_BASE=/scratch/ram1g23/baselines/tmp_ddpm_${SLURM_JOB_ID}

echo "Job ID : $SLURM_JOB_ID"
echo "Node   : $(hostname)"
echo "Model  : ${MODEL}"

mkdir -p ${OUTPUT}

# --- Run 1: ancestral DDPM (standard 1000-step baseline) ---
echo ""
echo "===== DDPM ancestral (1000 steps) ====="
python ../../evaluation_scripts/metrics_ddpm.py \
  --model_path      ${MODEL} \
  --sampler         ddpm \
  --dataset         cifar10 \
  --num_samples     10000 \
  --output_dir      ${OUTPUT} \
  --scratch_dir     ${SCRATCH_BASE}_ancestral

# --- Run 2: DDIM NFE curve ---
echo ""
echo "===== DDIM NFE curve (10/25/50/100/250 steps) ====="
python ../../evaluation_scripts/metrics_ddpm.py \
  --model_path      ${MODEL} \
  --sampler         ddim \
  --dataset         cifar10 \
  --num_samples     10000 \
  --step_counts     10 25 50 100 250 \
  --output_dir      ${OUTPUT} \
  --scratch_dir     ${SCRATCH_BASE}_ddim

echo ""
echo "Results saved to ${OUTPUT}"
