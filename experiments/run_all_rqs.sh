#!/bin/bash
# =============================================================================
# run_all_rqs.sh — Thin orchestrator for the full RQ1-6 pipeline.
#
# Delegates all job submission to each experiment's own sbatch scripts and
# stitches them together with SLURM --dependency flags. The per-experiment
# scripts are the single source of truth for resources, arrays, and commands.
#
# Usage (from any directory):
#   bash /path/to/experiments/run_all_rqs.sh
#
# Dependency graph:
#   Phase 0   exp0/train_ddpm.sh ──┬──► exp0 eval scripts, exp1/2/5/8 train
#             exp0/train_fm.sh   ──┘──► exp0 fm eval, exp4 fm baseline, exp6
#   exp1 train ──► exp1 eval
#                └──► exp3 eval, exp4 eval, exp6 eval  (need TIV ckpt)
#   exp2 train ──► exp2 eval
#   exp5 train ──► exp5 eval
#   exp7 train ──► exp7 eval   (reflow, optional)
#   exp8 train ──► exp8 eval   (LoRA 200 epochs, optional)
# =============================================================================
set -e
cd "$(dirname "$0")"
mkdir -p logs

export RUN_TAG=${RUN_TAG:-$(date +%Y%m%d_%H%M%S)}
echo "RUN_TAG: ${RUN_TAG}"

echo "=============================================="
echo "  Diff2Flow — Full pipeline: Phase 0 + RQ1–6"
echo "=============================================="

# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------
# submit <dep-list> <path-to-script> [extra sbatch args...]
#   dep-list: space-separated list of job IDs to depend on (empty = no deps)
submit() {
    local deps="$1"; shift
    local script="$1"; shift
    local dep_flag=""
    if [ -n "$deps" ]; then
        # strip leading/trailing whitespace, replace spaces with ':'
        local clean
        clean=$(echo "$deps" | tr -s ' ' ':' | sed 's/^://;s/:$//')
        if [ -n "$clean" ]; then
            dep_flag="--dependency=afterok:${clean}"
        fi
    fi
    sbatch --parsable ${dep_flag} "$@" "$script"
}

# =============================================================================
# Phase 0  Pretrain DDPM and FM baselines (both run in parallel)
# =============================================================================
echo ""
echo "── Phase 0: Pretraining DDPM and FM (parallel) ──"

DDPM_MODEL=/scratch/ram1g23/ddpm_cifar10_long2/final_model_ema
FM_MODEL=/scratch/ram1g23/fm_cifar10_long2/final_model_ema

if [ -d "$DDPM_MODEL" ]; then
    echo "  DDPM model found at $DDPM_MODEL — skipping pretraining"
    J_DDPM_TRAIN=""
else
    J_DDPM_TRAIN=$(submit "" exp0_baselines/train_ddpm.sh)
    echo "  DDPM pretrain          -> job $J_DDPM_TRAIN"
fi

if [ -d "$FM_MODEL" ]; then
    echo "  FM model found at $FM_MODEL — skipping pretraining"
    J_FM_TRAIN=""
else
    J_FM_TRAIN=$(submit "" exp0_baselines/train_fm.sh)
    echo "  FM pretrain            -> job $J_FM_TRAIN"
fi

# =============================================================================
# RQ1a  Speed–quality tradeoff — baseline evals (depend on Phase 0)
# =============================================================================
echo ""
echo "── RQ1a: Baselines (eval, after Phase 0) ──"

J_RQ1_DDPM=$(submit "$J_DDPM_TRAIN"      exp0_baselines/eval_ddpm.sh)
echo "  DDPM (ancestral+DDIM)  -> job $J_RQ1_DDPM"

J_RQ1_FM=$(submit "$J_FM_TRAIN"          exp0_baselines/eval_fm.sh)
echo "  FM from scratch        -> job $J_RQ1_FM"

J_RQ1_NAIVE=$(submit "$J_DDPM_TRAIN"     exp0_baselines/eval_naive_transfer.sh)
echo "  Naive transfer         -> job $J_RQ1_NAIVE"

J_RQ1_ALIGNONLY=$(submit "$J_DDPM_TRAIN" exp0_baselines/eval_alignment_only.sh)
echo "  Alignment-only (no FT) -> job $J_RQ1_ALIGNONLY"

# =============================================================================
# RQ2  Alignment ablation — exp1 (7 T/I/V variants)
#      TIV checkpoint feeds RQ1b, RQ4, and RQ6.
# =============================================================================
echo ""
echo "── RQ2: Alignment ablation — 7 T/I/V variants (exp1) ──"

J_RQ2_TRAIN=$(submit "$J_DDPM_TRAIN" exp1_alignment_ablation/train_array.sh)
echo "  Exp1 train (7-way)     -> job $J_RQ2_TRAIN"

J_RQ2_EVAL=$(submit "$J_RQ2_TRAIN"   exp1_alignment_ablation/eval_array.sh)
echo "  Exp1 eval  (7-way)     -> job $J_RQ2_EVAL  (after $J_RQ2_TRAIN)"

# =============================================================================
# RQ1b  Diff2Flow TIV at low NFE — exp3 (needs TIV ckpt from RQ2)
# =============================================================================
echo ""
echo "── RQ1b: Diff2Flow low-NFE curve (exp3, after RQ2 train) ──"

J_RQ1_LOWNFE=$(submit "$J_RQ2_TRAIN" exp3_low_nfe/eval_low_nfe.sh)
echo "  Diff2Flow TIV low-NFE  -> job $J_RQ1_LOWNFE  (after $J_RQ2_TRAIN)"

# =============================================================================
# RQ3  Parameter efficiency — exp2 (LoRA rank sweep + full FT)
#      Chained after RQ2 train to avoid hitting QOSMaxGRESPerUser.
# =============================================================================
echo ""
echo "── RQ3: LoRA rank sweep — 6 ranks + full FT (exp2, after RQ2 train) ──"

J_RQ3_TRAIN=$(submit "$J_RQ2_TRAIN" exp2_lora_rank_sweep/train_array.sh)
echo "  Exp2 train (7-way)     -> job $J_RQ3_TRAIN  (after $J_RQ2_TRAIN)"

J_RQ3_EVAL=$(submit "$J_RQ3_TRAIN"   exp2_lora_rank_sweep/eval_array.sh)
echo "  Exp2 eval  (7-way)     -> job $J_RQ3_EVAL  (after $J_RQ3_TRAIN)"

# =============================================================================
# RQ4  Solver sensitivity — exp4 (Euler vs Heun on TIV ckpt)
# =============================================================================
echo ""
echo "── RQ4: Solver sensitivity — Euler vs Heun (exp4, after RQ2 train) ──"

# exp4 also reads the FM ckpt for the FM baseline — depend on both.
J_RQ4_EVAL=$(submit "$J_RQ2_TRAIN $J_FM_TRAIN" exp4_solver_sensitivity/eval_solvers.sh)
echo "  Euler vs Heun eval     -> job $J_RQ4_EVAL  (after $J_RQ2_TRAIN $J_FM_TRAIN)"

# =============================================================================
# RQ5  LoRA placement — exp5 (attention / feedforward / all)
#      Chained after RQ3 train to avoid hitting QOSMaxGRESPerUser.
# =============================================================================
echo ""
echo "── RQ5: LoRA placement — 3 variants (exp5, after RQ3 train) ──"

J_RQ5_TRAIN=$(submit "$J_RQ3_TRAIN" exp5_lora_placement/train_array.sh)
echo "  Exp5 train (3-way)     -> job $J_RQ5_TRAIN  (after $J_RQ3_TRAIN)"

J_RQ5_EVAL=$(submit "$J_RQ5_TRAIN"   exp5_lora_placement/eval_array.sh)
echo "  Exp5 eval  (3-way)     -> job $J_RQ5_EVAL  (after $J_RQ5_TRAIN)"

# =============================================================================
# RQ6  Trajectory curvature — exp6 (DDPM vs FM vs TIV)
# =============================================================================
echo ""
echo "── RQ6: Trajectory curvature visualisation (after RQ2 train) ──"

J_RQ6_TRAJ=$(submit "$J_RQ2_TRAIN $J_FM_TRAIN" exp6_visualise_trajectories/visualise_trajectory.sh)
echo "  Trajectory curvature   -> job $J_RQ6_TRAJ  (after $J_RQ2_TRAIN $J_FM_TRAIN)"

# =============================================================================
# Summary
# =============================================================================
echo ""
echo "=============================================="
echo "  All jobs submitted. Summary:"
echo "=============================================="
echo ""
echo "  Phase 0  Pretraining"
echo "       DDPM                     -> ${J_DDPM_TRAIN:-(skipped)}"
echo "       FM                       -> ${J_FM_TRAIN:-(skipped)}"
echo ""
echo "  RQ1  Speed-quality tradeoff"
echo "       DDPM (ancestral+DDIM)    -> $J_RQ1_DDPM"
echo "       FM from scratch          -> $J_RQ1_FM"
echo "       Naive transfer           -> $J_RQ1_NAIVE"
echo "       Alignment-only (no FT)   -> $J_RQ1_ALIGNONLY"
echo "       Diff2Flow TIV low-NFE    -> $J_RQ1_LOWNFE"
echo ""
echo "  RQ2  Alignment impact (T/I/V ablation)"
echo "       Train 7-way array        -> $J_RQ2_TRAIN"
echo "       Eval  7-way array        -> $J_RQ2_EVAL"
echo ""
echo "  RQ3  Parameter efficiency (LoRA rank sweep)"
echo "       Train 7-way array        -> $J_RQ3_TRAIN"
echo "       Eval  7-way array        -> $J_RQ3_EVAL"
echo ""
echo "  RQ4  Solver sensitivity (Euler vs Heun)"
echo "       Eval                     -> $J_RQ4_EVAL"
echo ""
echo "  RQ5  LoRA placement"
echo "       Train 3-way array        -> $J_RQ5_TRAIN"
echo "       Eval  3-way array        -> $J_RQ5_EVAL"
echo ""
echo "  RQ6  Trajectory curvature"
echo "       Visualise DDPM/FM/D2F    -> $J_RQ6_TRAJ"
echo ""
echo "  Logs: experiments/logs/"
echo "=============================================="
