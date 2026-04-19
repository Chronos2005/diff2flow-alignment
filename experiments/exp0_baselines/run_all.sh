#!/bin/bash
# Submit baseline training then evaluation jobs.
# Run from experiments/exp0_baselines/:  bash run_all.sh
set -e
cd "$(dirname "$0")"

# Train DDPM and FM in parallel, skipping if a final_model checkpoint already exists
DDPM_FINAL=/scratch/ram1g23/ddpm_cifar10_long2/final_model_ema
FM_FINAL=/scratch/ram1g23/fm_cifar10_long2/final_model_ema

echo "Submitted training:"
if [ -d "${DDPM_FINAL}" ]; then
  J_DDPM_TRAIN=""
  echo "  DDPM train         -> skipped (found ${DDPM_FINAL})"
else
  J_DDPM_TRAIN=$(sbatch --parsable train_ddpm.sh)
  echo "  DDPM train         -> job ${J_DDPM_TRAIN}"
fi
if [ -d "${FM_FINAL}" ]; then
  J_FM_TRAIN=""
  echo "  FM train           -> skipped (found ${FM_FINAL})"
else
  J_FM_TRAIN=$(sbatch --parsable train_fm.sh)
  echo "  FM train           -> job ${J_FM_TRAIN}"
fi

# Eval jobs run after their respective training completes (or immediately if training was skipped)
ddpm_dep=""
fm_dep=""
[ -n "${J_DDPM_TRAIN}" ] && ddpm_dep="--dependency=afterok:${J_DDPM_TRAIN}"
[ -n "${J_FM_TRAIN}" ] && fm_dep="--dependency=afterok:${J_FM_TRAIN}"

J_DDPM_EVAL=$(sbatch --parsable ${ddpm_dep} eval_ddpm.sh)
J_FM_EVAL=$(sbatch --parsable ${fm_dep} eval_fm.sh)
J_NAIVE=$(sbatch --parsable ${ddpm_dep} eval_naive_transfer.sh)
J_ALIGN=$(sbatch --parsable ${ddpm_dep} eval_alignment_only.sh)

echo ""
echo "Submitted eval:"
echo "  DDPM eval          -> job ${J_DDPM_EVAL}  (after ${J_DDPM_TRAIN:-none})"
echo "  FM eval            -> job ${J_FM_EVAL}  (after ${J_FM_TRAIN:-none})"
echo "  Naive transfer     -> job ${J_NAIVE}  (after ${J_DDPM_TRAIN:-none})"
echo "  Alignment only     -> job ${J_ALIGN}  (after ${J_DDPM_TRAIN:-none})"
echo ""
echo "Results will appear in /scratch/ram1g23/baselines/{ddpm,fm,naive_transfer,alignment_only}_eval/"
