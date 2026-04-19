#!/bin/bash
#SBATCH --partition=ecsstudents_l4
#SBATCH --account=ecsstudents
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=6
#SBATCH --time=4:00:00
#SBATCH --output=../logs/exp0_fm_%j.out
#SBATCH --error=../logs/exp0_fm_%j.err

# ---------------------------------------------------------------------------
# Baseline 2: Flow Matching (trained from scratch)
#
# Euler sampling at 2/4/10/25/50/100 steps.
#
# Model: /scratch/ram1g23/fm_cifar10_long2/final_model_ema
# ---------------------------------------------------------------------------

module load conda
source activate /home/ram1g23/.conda/envs/diffusion_flow_study
ln -sf ~/inception_cache/inception-2015-12-05.pt /tmp/inception-2015-12-05.pt
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

RUN_TAG=${RUN_TAG:-${SLURM_JOB_ID:-$(date +%Y%m%d_%H%M%S)}}
MODEL=/scratch/ram1g23/fm_cifar10_long2/final_model_ema
OUTPUT=/scratch/ram1g23/baselines/fm_eval/${RUN_TAG}
SCRATCH=${SCRATCH:-/scratch/ram1g23/baselines/tmp_fm_${SLURM_JOB_ID}}

echo "Job ID : $SLURM_JOB_ID"
echo "Node   : $(hostname)"
echo "Model  : ${MODEL}"

mkdir -p ${OUTPUT}

python ../../evaluation_scripts/metrics_fm.py \
  --model_path  ${MODEL} \
  --dataset     cifar10 \
  --num_samples 10000 \
  --step_counts 2 4 10 25 50 100 \
  --output_dir  ${OUTPUT} \
  --scratch_dir /scratch/ram1g23/baselines/tmp_fm_${SLURM_JOB_ID}

echo ""
echo "Results saved to ${OUTPUT}"
