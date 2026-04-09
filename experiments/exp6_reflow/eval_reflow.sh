#!/bin/bash
#SBATCH --partition=ecsstudents_l4
#SBATCH --account=ecsstudents
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=6
#SBATCH --time=4:00:00
#SBATCH --output=../logs/exp6_eval_%j.out
#SBATCH --error=../logs/exp6_eval_%j.err

module load conda
source activate /home/ram1g23/.conda/envs/diffusion_flow_study
ln -sf ~/inception_cache/inception-2015-12-05.pt /tmp/inception-2015-12-05.pt
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

# ---------------------------------------------------------------------------
# Exp 6 Eval: FID at low NFE for the reflowed model.
#
# The reflow student is a pure FM model — uses metrics_fm.py (no aligner).
# Compare against Exp 3 (unreflowed model at same step counts).
# ---------------------------------------------------------------------------

MODEL_PATH=/scratch/ram1g23/exp6_reflow/final_model
OUTPUT_DIR=/scratch/ram1g23/exp6_reflow/eval

echo "Job ID: $SLURM_JOB_ID"
echo "Node: $(hostname)"
echo "Model: ${MODEL_PATH}"

python ../../evaluation_scripts/metrics_fm.py \
  --model_path ${MODEL_PATH} \
  --dataset cifar10 \
  --num_samples 10000 \
  --step_counts 1 2 4 8 16 50 \
  --output_dir ${OUTPUT_DIR} \
  --scratch_dir /scratch/ram1g23/exp6_tmp_${SLURM_JOB_ID}
