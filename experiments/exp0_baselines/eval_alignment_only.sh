#!/bin/bash
#SBATCH --partition=ecsstudents_l4
#SBATCH --account=ecsstudents
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=6
#SBATCH --time=4:00:00
#SBATCH --output=../logs/exp0_alignment_only_%j.out
#SBATCH --error=../logs/exp0_alignment_only_%j.err

# ---------------------------------------------------------------------------
# Baseline 4: Diff2Flow alignment only (no fine-tuning)
#
# Applies all three analytical alignment steps (T+I+V) to the raw pre-trained
# DDPM weights with NO LoRA fine-tuning. Isolates how much the alignment
# transformation alone gains before any learned adaptation.
#
# Checkpoint : raw DDPM weights (diffusion_pytorch_model.safetensors)
# Alignment  : timestep rescaling + interpolant rescaling + velocity translation
# Fine-tuning: none
# ---------------------------------------------------------------------------

module load conda
source activate /home/ram1g23/.conda/envs/diffusion_flow_study
ln -sf ~/inception_cache/inception-2015-12-05.pt /tmp/inception-2015-12-05.pt
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

DDPM_MODEL=/scratch/ram1g23/Models/Cifar-10/ddpm_cifar10/final_model
DDPM_CKPT=${DDPM_MODEL}/diffusion_pytorch_model.safetensors
OUTPUT=/scratch/ram1g23/baselines/alignment_only_eval

echo "Job ID     : $SLURM_JOB_ID"
echo "Node       : $(hostname)"
echo "Checkpoint : ${DDPM_CKPT}"

mkdir -p ${OUTPUT}

python ../../evaluation_scripts/metrics_diff2flow.py \
  --checkpoint_path       ${DDPM_CKPT} \
  --pretrained_model_path ${DDPM_MODEL} \
  --dataset               cifar10 \
  --num_samples           10000 \
  --step_counts           2 4 10 25 50 100 \
  --output_dir            ${OUTPUT} \
  --scratch_dir           /scratch/ram1g23/baselines/tmp_alignment_only_${SLURM_JOB_ID}

echo ""
echo "Results saved to ${OUTPUT}"
