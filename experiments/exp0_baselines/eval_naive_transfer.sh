#!/bin/bash
#SBATCH --partition=ecsstudents_l4
#SBATCH --account=ecsstudents
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=6
#SBATCH --time=4:00:00
#SBATCH --output=../logs/exp0_naive_%j.out
#SBATCH --error=../logs/exp0_naive_%j.err

# ---------------------------------------------------------------------------
# Baseline 3: Naive transfer
#
# Runs the FM Euler sampler directly on the pre-trained DDPM weights with no
# alignment at all. The DDPM epsilon-prediction network is treated as a
# velocity field and integrated with a plain Euler ODE. No Diff2Flow aligner
# is applied — this is the "plug in and hope" lower bound.
# ---------------------------------------------------------------------------

module load conda
source activate /home/ram1g23/.conda/envs/diffusion_flow_study
ln -sf ~/inception_cache/inception-2015-12-05.pt /tmp/inception-2015-12-05.pt
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

DDPM_MODEL=/scratch/ram1g23/Models/Cifar-10/ddpm_cifar10/final_model
OUTPUT=/scratch/ram1g23/baselines/naive_transfer_eval

echo "Job ID : $SLURM_JOB_ID"
echo "Node   : $(hostname)"
echo "Model  : ${DDPM_MODEL}"

mkdir -p ${OUTPUT}

python ../../evaluation_scripts/metrics_fm.py \
  --model_path  ${DDPM_MODEL} \
  --dataset     cifar10 \
  --num_samples 10000 \
  --step_counts 2 4 10 25 50 100 \
  --output_dir  ${OUTPUT} \
  --scratch_dir /scratch/ram1g23/baselines/tmp_naive_${SLURM_JOB_ID}

echo ""
echo "Results saved to ${OUTPUT}"
