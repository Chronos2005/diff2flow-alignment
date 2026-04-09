#!/bin/bash
# Submit Exp 6: train reflow student, then evaluate at low NFE.
set -e

TRAIN_JOB=$(sbatch --parsable train_reflow.sh)
echo "Submitted reflow training: ${TRAIN_JOB}"

EVAL_JOB=$(sbatch --parsable --dependency=afterok:${TRAIN_JOB} eval_reflow.sh)
echo "Submitted reflow eval: ${EVAL_JOB}  (depends on ${TRAIN_JOB})"
