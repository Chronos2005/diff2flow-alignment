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
ln -sf ~/inception_cache/inception-2015-12-05.pt /tmp/inception-2015-12-05.pt
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

RUN_TAG=${RUN_TAG:-${SLURM_JOB_ID:-$(date +%Y%m%d_%H%M%S)}}
CKPT=/scratch/ram1g23/exp2_lora_rank/rank64/final_model/diffusion_pytorch_model.safetensors
PRETRAIN=/scratch/ram1g23/ddpm_cifar10_long2/final_model_ema
FM_MODEL=/scratch/ram1g23/fm_cifar10_long2/final_model_ema

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
  --step_counts 2 4 8 16 32 \
  --solver euler \
  --output_dir /scratch/ram1g23/exp4_solver/euler/${RUN_TAG} \
  --scratch_dir /scratch/ram1g23/exp4_tmp_euler_${SLURM_JOB_ID}

# --- Heun solver (half the steps for same NFE) ---
echo ""
echo "===== Running Heun solver ====="
python ../../evaluation_scripts/metrics_diff2flow.py \
  --checkpoint_path ${CKPT} \
  --pretrained_model_path ${PRETRAIN} \
  --dataset cifar10 \
  --num_samples 10000 \
  --step_counts 1 2 4 8 16 \
  --solver heun \
  --output_dir /scratch/ram1g23/exp4_solver/heun/${RUN_TAG} \
  --scratch_dir /scratch/ram1g23/exp4_tmp_heun_${SLURM_JOB_ID}

# ---------------------------------------------------------------------------
# Baseline: Flow Matching (trained from scratch)
# ---------------------------------------------------------------------------

# --- FM Euler baseline ---
echo ""
echo "===== Running FM baseline — Euler solver ====="
python ../../evaluation_scripts/metrics_fm.py \
  --model_path ${FM_MODEL} \
  --dataset cifar10 \
  --num_samples 10000 \
  --step_counts 2 4 8 16 32 \
  --solver euler \
  --output_dir /scratch/ram1g23/exp4_solver/fm_euler/${RUN_TAG} \
  --scratch_dir /scratch/ram1g23/exp4_tmp_fm_euler_${SLURM_JOB_ID}

# --- FM Heun baseline (half steps for same NFE) ---
echo ""
echo "===== Running FM baseline — Heun solver ====="
python ../../evaluation_scripts/metrics_fm.py \
  --model_path ${FM_MODEL} \
  --dataset cifar10 \
  --num_samples 10000 \
  --step_counts 1 2 4 8 16 \
  --solver heun \
  --output_dir /scratch/ram1g23/exp4_solver/fm_heun/${RUN_TAG} \
  --scratch_dir /scratch/ram1g23/exp4_tmp_fm_heun_${SLURM_JOB_ID}
