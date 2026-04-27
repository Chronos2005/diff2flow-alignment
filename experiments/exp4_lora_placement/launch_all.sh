#!/bin/bash
# Submit Exp 5: train all LoRA placement variants, then evaluate.
set -e

TRAIN_JOB=$(sbatch --parsable train_array.sh)
echo "Submitted training array: ${TRAIN_JOB}"

EVAL_JOB=$(sbatch --parsable --dependency=afterok:${TRAIN_JOB} eval_array.sh)
echo "Submitted eval array: ${EVAL_JOB}  (depends on ${TRAIN_JOB})"
