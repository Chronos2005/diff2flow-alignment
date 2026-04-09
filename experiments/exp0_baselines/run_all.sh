#!/bin/bash
# Submit all three baseline evaluation jobs.
# Run from experiments/exp0_baselines/:  bash run_all.sh

set -e
cd "$(dirname "$0")"

JOB_DDPM=$(sbatch --parsable eval_ddpm.sh)
JOB_FM=$(sbatch --parsable eval_fm.sh)
JOB_NAIVE=$(sbatch --parsable eval_naive_transfer.sh)
JOB_ALIGN=$(sbatch --parsable eval_alignment_only.sh)

echo "Submitted:"
echo "  DDPM baseline      -> job ${JOB_DDPM}  (logs: ../logs/exp0_ddpm_${JOB_DDPM}.out)"
echo "  FM baseline        -> job ${JOB_FM}  (logs: ../logs/exp0_fm_${JOB_FM}.out)"
echo "  Naive transfer     -> job ${JOB_NAIVE}  (logs: ../logs/exp0_naive_${JOB_NAIVE}.out)"
echo "  Alignment only     -> job ${JOB_ALIGN}  (logs: ../logs/exp0_alignment_only_${JOB_ALIGN}.out)"
echo ""
echo "Results will appear in /scratch/ram1g23/baselines/{ddpm,fm,naive_transfer,alignment_only}_eval/"
