#!/bin/bash
#SBATCH --partition=ecsstudents_l4
#SBATCH --account=ecsstudents
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=6
#SBATCH --time=6:00:00
#SBATCH --output=../logs/exp4_eval_%j.out
#SBATCH --error=../logs/exp4_eval_%j.err

module load conda
source activate /home/ram1g23/.conda/envs/diffusion_flow_study

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

cp /iridisfs/home/ram1g23/Projects/diff2flow-alignment/inception-2015-12-05.pt /tmp/

# ---------------------------------------------------------------------------
# Exp 4: Solver Sensitivity — Euler vs Heun at matched NFE
#
# To compare at equal NFE budget:
#   Euler  --step_counts 2 4 8   -> NFE = 2, 4, 8
#   Heun   --step_counts 1 2 4   -> NFE = 2, 4, 8  (2x per step)
#
# Uses the TIV checkpoint from Exp 1. Update CKPT if using a different run.
# ---------------------------------------------------------------------------

CKPT=/scratch/ram1g23/exp1_alignment/TIV/final_model/diffusion_pytorch_model.safetensors
PRETRAIN=/scratch/ram1g23/Models/Cifar-10/ddpm_cifar10/final_model

echo "Job ID: $SLURM_JOB_ID"
echo "Node: $(hostname)"
echo "Checkpoint: ${CKPT}"

# --- Euler solver ---
echo ""
echo "===== Running Euler solver ====="
python ../../evaluation_scripts/metrics_diff2flow.py \
  --checkpoint_path ${CKPT} \
  --pretrained_model_path ${PRETRAIN} \
  --dataset cifar10 \
  --num_samples 10000 \
  --step_counts 2 4 8 \
  --solver euler \
  --output_dir /scratch/ram1g23/exp4_solver/euler \
  --scratch_dir /scratch/ram1g23/exp4_tmp_euler_${SLURM_JOB_ID}

# --- Heun solver (half the steps for same NFE) ---
echo ""
echo "===== Running Heun solver ====="
python ../../evaluation_scripts/metrics_diff2flow.py \
  --checkpoint_path ${CKPT} \
  --pretrained_model_path ${PRETRAIN} \
  --dataset cifar10 \
  --num_samples 10000 \
  --step_counts 1 2 4 \
  --solver heun \
  --output_dir /scratch/ram1g23/exp4_solver/heun \
  --scratch_dir /scratch/ram1g23/exp4_tmp_heun_${SLURM_JOB_ID}
