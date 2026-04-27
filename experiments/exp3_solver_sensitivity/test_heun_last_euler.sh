#!/bin/bash
#SBATCH --partition=ecsstudents_l4
#SBATCH --account=ecsstudents
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=6
#SBATCH --time=2:00:00
#SBATCH --output=../logs/exp4_heun_fix_%j.out
#SBATCH --error=../logs/exp4_heun_fix_%j.err

module load conda
source activate /home/ram1g23/.conda/envs/diffusion_flow_study
ln -sf ~/inception_cache/inception-2015-12-05.pt /tmp/inception-2015-12-05.pt
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
cp /iridisfs/home/ram1g23/Projects/diff2flow-alignment/inception-2015-12-05.pt /tmp/

RUN_TAG=${RUN_TAG:-${SLURM_JOB_ID:-$(date +%Y%m%d_%H%M%S)}}
FM_MODEL=/scratch/ram1g23/fm_cifar10_long2/final_model_ema
NUM_SAMPLES=${NUM_SAMPLES:-10000}   # set NUM_SAMPLES=2000 for a quick smoke test

echo "Job ID: ${SLURM_JOB_ID:-local}"
echo "Node: $(hostname)"
echo "FM model: ${FM_MODEL}"
echo "Samples: ${NUM_SAMPLES}"
echo ""

# ---------------------------------------------------------------------------
# Control: vanilla Heun (original behaviour)
# ---------------------------------------------------------------------------
echo "===== [CONTROL] FM Heun — vanilla ====="
HEUN_LAST_EULER=0 python ../../evaluation_scripts/metrics_fm.py \
  --model_path ${FM_MODEL} \
  --dataset cifar10 \
  --num_samples ${NUM_SAMPLES} \
  --step_counts 1 2 4 8 16 \
  --solver heun \
  --output_dir /scratch/ram1g23/exp4_heun_fix/control/${RUN_TAG} \
  --scratch_dir /scratch/ram1g23/exp4_heun_fix_tmp_ctrl_${SLURM_JOB_ID:-$$}

echo ""
# ---------------------------------------------------------------------------
# Treatment: Heun with last-step Euler fallback
# ---------------------------------------------------------------------------
echo "===== [TREATMENT] FM Heun — last step Euler ====="
HEUN_LAST_EULER=1 python ../../evaluation_scripts/metrics_fm.py \
  --model_path ${FM_MODEL} \
  --dataset cifar10 \
  --num_samples ${NUM_SAMPLES} \
  --step_counts 1 2 4 8 16 \
  --solver heun \
  --output_dir /scratch/ram1g23/exp4_heun_fix/treatment/${RUN_TAG} \
  --scratch_dir /scratch/ram1g23/exp4_heun_fix_tmp_trt_${SLURM_JOB_ID:-$$}

echo ""
# ---------------------------------------------------------------------------
# Euler reference at matched NFE (steps = 2 4 8 16 32 → NFE = 2 4 8 16 32)
# Already have this from the main exp4 run; included here for convenience if
# running standalone (comment out if redundant).
# ---------------------------------------------------------------------------
# echo "===== [REFERENCE] FM Euler ====="
# python ../../evaluation_scripts/metrics_fm.py \
#   --model_path ${FM_MODEL} \
#   --dataset cifar10 \
#   --num_samples ${NUM_SAMPLES} \
#   --step_counts 2 4 8 16 32 \
#   --solver euler \
#   --output_dir /scratch/ram1g23/exp4_heun_fix/euler_ref/${RUN_TAG} \
#   --scratch_dir /scratch/ram1g23/exp4_heun_fix_tmp_euler_${SLURM_JOB_ID:-$$}

# echo ""
# echo "Done. Results:"
# echo "  control:   /scratch/ram1g23/exp4_heun_fix/control/${RUN_TAG}"
# echo "  treatment: /scratch/ram1g23/exp4_heun_fix/treatment/${RUN_TAG}"
# echo "  euler ref: /scratch/ram1g23/exp4_heun_fix/euler_ref/${RUN_TAG}"
