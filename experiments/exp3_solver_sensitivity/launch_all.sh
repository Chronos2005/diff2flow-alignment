#!/bin/bash
# Submit Exp 4: solver sensitivity eval (no training needed).
set -e

JOB=$(sbatch --parsable eval_solvers.sh)
echo "Submitted solver eval: ${JOB}"
