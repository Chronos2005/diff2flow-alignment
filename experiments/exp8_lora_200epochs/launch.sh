#!/bin/bash
# Submit train job, then eval as a dependency so it runs automatically after training finishes.
set -e
cd "$(dirname "$0")"

TRAIN_JOB=$(sbatch --parsable train.sh)
echo "Submitted train job: ${TRAIN_JOB}"

EVAL_JOB=$(sbatch --parsable --dependency=afterok:${TRAIN_JOB} eval.sh)
echo "Submitted eval job:  ${EVAL_JOB}  (depends on ${TRAIN_JOB})"

echo ""
echo "Monitor with:"
echo "  squeue -u \$USER"
echo "  tail -f ../logs/exp8_train_${TRAIN_JOB}.out"
echo "  tail -f ../logs/exp8_eval_${EVAL_JOB}.out"
