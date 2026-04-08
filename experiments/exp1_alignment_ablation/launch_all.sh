#!/bin/bash
# Submit Exp 1: train all 7 alignment variants, then evaluate.
set -e

TRAIN_JOB=$(sbatch --parsable train_array.sh)
echo "Submitted training array: ${TRAIN_JOB}"

EVAL_JOB=$(sbatch --parsable --dependency=afterok:${TRAIN_JOB} eval_array.sh)
echo "Submitted eval array: ${EVAL_JOB}  (depends on ${TRAIN_JOB})"
