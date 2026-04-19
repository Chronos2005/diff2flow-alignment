#!/bin/bash
# =============================================================================
# run_all_rqs.sh — Full pipeline: Phase 0 pretraining + all RQ1–RQ6 experiments
#
# Usage (from any directory):
#   bash /path/to/experiments/run_all_rqs.sh
#
# Dependency graph:
#   Phase 0   ddpm_train ──┬──► RQ1a baselines, RQ2/3/5 training
#             fm_train   ──┘──► RQ1a fm eval, RQ6 traj vis
#                               │
#   RQ2   exp1_train ──► exp1_eval
#              │
#              ├──► RQ1b  exp3_lownfe eval   (needs TIV ckpt)
#              ├──► RQ4   solver eval         (needs TIV ckpt)
#              └──► RQ6   trajectory vis      (needs TIV + FM ckpt)
#   RQ3   exp2_train ──► exp2_eval
#   RQ5   exp5_train ──► exp5_eval
# =============================================================================
set -e
cd "$(dirname "$0")"
mkdir -p logs

export RUN_TAG=${RUN_TAG:-$(date +%Y%m%d_%H%M%S)}
echo "RUN_TAG: ${RUN_TAG}"

echo "=============================================="
echo "  Diff2Flow — Full pipeline: Phase 0 + RQ1–6"
echo "=============================================="

# ==============================================================================
# Phase 0  Pretrain DDPM and FM baselines (both run in parallel)
# ==============================================================================
echo ""
echo "── Phase 0: Pretraining DDPM and FM (parallel) ──"

DDPM_MODEL=/scratch/ram1g23/ddpm_cifar10_long2/final_model_ema
FM_MODEL=/scratch/ram1g23/fm_cifar10_long2/final_model_ema

if [ -d "$DDPM_MODEL" ]; then
    echo "  DDPM model found at $DDPM_MODEL — skipping pretraining"
    J_DDPM_TRAIN=""
else
    J_DDPM_TRAIN=$(sbatch --parsable exp0_baselines/train_ddpm.sh)
    echo "  DDPM pretrain          -> job $J_DDPM_TRAIN"
fi

if [ -d "$FM_MODEL" ]; then
    echo "  FM model found at $FM_MODEL — skipping pretraining"
    J_FM_TRAIN=""
else
    J_FM_TRAIN=$(sbatch --parsable exp0_baselines/train_fm.sh)
    echo "  FM pretrain            -> job $J_FM_TRAIN"
fi

# ==============================================================================
# RQ1a  Speed–quality tradeoff — baseline evals (depend on Phase 0)
#       DDPM (ancestral + DDIM), FM scratch, naive transfer, alignment-only
# ==============================================================================
echo ""
echo "── RQ1a: Baselines (eval, after Phase 0) ──"

J_RQ1_DDPM=$(sbatch --parsable --job-name=rq1_ddpm \
    ${J_DDPM_TRAIN:+--dependency=afterok:${J_DDPM_TRAIN}} << 'ENDJOB'
#!/bin/bash
#SBATCH --partition=ecsstudents_l4
#SBATCH --account=ecsstudents
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=6
#SBATCH --time=6:00:00
#SBATCH --output=logs/rq1_ddpm_%j.out
#SBATCH --error=logs/rq1_ddpm_%j.err
module load conda
source activate /home/ram1g23/.conda/envs/diffusion_flow_study
ln -sf ~/inception_cache/inception-2015-12-05.pt /tmp/inception-2015-12-05.pt
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

MODEL=/scratch/ram1g23/ddpm_cifar10_long2/final_model_ema
OUT=/iridisfs/home/ram1g23/Projects/diff2flow-alignment/results/exp0_baselines/ddpm_eval/${RUN_TAG}
echo "Job $SLURM_JOB_ID on $(hostname)"
mkdir -p $OUT

python /iridisfs/home/ram1g23/Projects/diff2flow-alignment/evaluation_scripts/metrics_ddpm.py \
  --model_path $MODEL --sampler ddpm --dataset cifar10 \
  --num_samples 10000 --output_dir $OUT \
  --scratch_dir /scratch/ram1g23/baselines/tmp_ddpm_ancestral_${SLURM_JOB_ID}

python /iridisfs/home/ram1g23/Projects/diff2flow-alignment/evaluation_scripts/metrics_ddpm.py \
  --model_path $MODEL --sampler ddim --dataset cifar10 \
  --num_samples 10000 --step_counts 10 25 50 100 250 \
  --output_dir $OUT \
  --scratch_dir /scratch/ram1g23/baselines/tmp_ddpm_ddim_${SLURM_JOB_ID}
ENDJOB
)
echo "  DDPM (ancestral+DDIM)  -> job $J_RQ1_DDPM"

J_RQ1_FM=$(sbatch --parsable --job-name=rq1_fm \
    ${J_FM_TRAIN:+--dependency=afterok:${J_FM_TRAIN}} << 'ENDJOB'
#!/bin/bash
#SBATCH --partition=ecsstudents_l4
#SBATCH --account=ecsstudents
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=6
#SBATCH --time=4:00:00
#SBATCH --output=logs/rq1_fm_%j.out
#SBATCH --error=logs/rq1_fm_%j.err
module load conda
source activate /home/ram1g23/.conda/envs/diffusion_flow_study
ln -sf ~/inception_cache/inception-2015-12-05.pt /tmp/inception-2015-12-05.pt
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

MODEL=/scratch/ram1g23/fm_cifar10_long2/final_model_ema
OUT=/iridisfs/home/ram1g23/Projects/diff2flow-alignment/results/exp0_baselines/fm_eval/${RUN_TAG}
echo "Job $SLURM_JOB_ID on $(hostname)"
mkdir -p $OUT

python /iridisfs/home/ram1g23/Projects/diff2flow-alignment/evaluation_scripts/metrics_fm.py \
  --model_path $MODEL --dataset cifar10 \
  --num_samples 10000 --step_counts 2 4 10 25 50 100 \
  --output_dir $OUT \
  --scratch_dir /scratch/ram1g23/baselines/tmp_fm_${SLURM_JOB_ID}
ENDJOB
)
echo "  FM from scratch        -> job $J_RQ1_FM"

J_RQ1_NAIVE=$(sbatch --parsable --job-name=rq1_naive \
    ${J_DDPM_TRAIN:+--dependency=afterok:${J_DDPM_TRAIN}} << 'ENDJOB'
#!/bin/bash
#SBATCH --partition=ecsstudents_l4
#SBATCH --account=ecsstudents
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=6
#SBATCH --time=4:00:00
#SBATCH --output=logs/rq1_naive_%j.out
#SBATCH --error=logs/rq1_naive_%j.err
module load conda
source activate /home/ram1g23/.conda/envs/diffusion_flow_study
ln -sf ~/inception_cache/inception-2015-12-05.pt /tmp/inception-2015-12-05.pt
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

MODEL=/scratch/ram1g23/ddpm_cifar10_long2/final_model_ema
OUT=/iridisfs/home/ram1g23/Projects/diff2flow-alignment/results/exp0_baselines/naive_transfer_eval/${RUN_TAG}
echo "Job $SLURM_JOB_ID on $(hostname)"
mkdir -p $OUT

python /iridisfs/home/ram1g23/Projects/diff2flow-alignment/evaluation_scripts/metrics_fm.py \
  --model_path $MODEL --dataset cifar10 \
  --num_samples 10000 --step_counts 2 4 10 25 50 100 \
  --output_dir $OUT \
  --scratch_dir /scratch/ram1g23/baselines/tmp_naive_${SLURM_JOB_ID}
ENDJOB
)
echo "  Naive transfer         -> job $J_RQ1_NAIVE"

J_RQ1_ALIGNONLY=$(sbatch --parsable --job-name=rq1_alignonly \
    ${J_DDPM_TRAIN:+--dependency=afterok:${J_DDPM_TRAIN}} << 'ENDJOB'
#!/bin/bash
#SBATCH --partition=ecsstudents_l4
#SBATCH --account=ecsstudents
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=6
#SBATCH --time=4:00:00
#SBATCH --output=logs/rq1_alignonly_%j.out
#SBATCH --error=logs/rq1_alignonly_%j.err
module load conda
source activate /home/ram1g23/.conda/envs/diffusion_flow_study
ln -sf ~/inception_cache/inception-2015-12-05.pt /tmp/inception-2015-12-05.pt
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

DDPM_MODEL=/scratch/ram1g23/ddpm_cifar10_long2/final_model_ema
OUT=/iridisfs/home/ram1g23/Projects/diff2flow-alignment/results/exp0_baselines/alignment_only_eval/${RUN_TAG}
echo "Job $SLURM_JOB_ID on $(hostname)"
mkdir -p $OUT

python /iridisfs/home/ram1g23/Projects/diff2flow-alignment/evaluation_scripts/metrics_diff2flow.py \
  --checkpoint_path $DDPM_MODEL/diffusion_pytorch_model.safetensors \
  --pretrained_model_path $DDPM_MODEL \
  --dataset cifar10 --num_samples 10000 \
  --step_counts 2 4 10 25 50 100 \
  --output_dir $OUT \
  --scratch_dir /scratch/ram1g23/baselines/tmp_alignonly_${SLURM_JOB_ID}
ENDJOB
)
echo "  Alignment-only (no FT) -> job $J_RQ1_ALIGNONLY"

# ==============================================================================
# RQ2  Alignment impact — train all 7 T/I/V subsets, then evaluate
#      The TIV checkpoint this produces feeds RQ1b, RQ4, and RQ6.
# ==============================================================================
echo ""
echo "── RQ2: Alignment ablation — 7 T/I/V variants (exp1) ──"

J_RQ2_TRAIN=$(sbatch --parsable --array=1-7 --job-name=rq2_train \
    ${J_DDPM_TRAIN:+--dependency=afterok:${J_DDPM_TRAIN}} << 'ENDJOB'
#!/bin/bash
#SBATCH --partition=ecsstudents_l4
#SBATCH --account=ecsstudents
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=12
#SBATCH --time=24:00:00
#SBATCH --output=logs/rq2_train_%A_%a.out
#SBATCH --error=logs/rq2_train_%A_%a.err
module load conda
source activate /home/ram1g23/.conda/envs/diffusion_flow_study
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
echo "Job $SLURM_JOB_ID  Array $SLURM_ARRAY_TASK_ID on $(hostname)"
python - <<PYEOF
import torch
print("Torch:", torch.__version__, "| CUDA:", torch.cuda.is_available(), "| GPUs:", torch.cuda.device_count())
PYEOF

# Index | Variant | Active components
#   1   | T       | timestep rescaling only
#   2   | I       | interpolant rescaling only
#   3   | V       | velocity translation only
#   4   | T+I     | timestep + interpolant
#   5   | T+V     | timestep + velocity
#   6   | I+V     | interpolant + velocity
#   7   | T+I+V   | full alignment (canonical Diff2Flow)
declare -a VARIANT_FLAGS=(
    ""
    "--no_interpolant_rescaling --no_velocity_translation"
    "--no_timestep_rescaling --no_velocity_translation"
    "--no_timestep_rescaling --no_interpolant_rescaling"
    "--no_velocity_translation"
    "--no_interpolant_rescaling"
    "--no_timestep_rescaling"
    ""
)
declare -a VARIANT_NAMES=("" "T" "I" "V" "TI" "TV" "IV" "TIV")

FLAGS=${VARIANT_FLAGS[$SLURM_ARRAY_TASK_ID]}
NAME=${VARIANT_NAMES[$SLURM_ARRAY_TASK_ID]}
OUT=/scratch/ram1g23/exp1_alignment/${NAME}
echo "Variant: ${NAME}  Flags: ${FLAGS:-none}  Output: ${OUT}"

accelerate launch --num_processes=2 \
  /iridisfs/home/ram1g23/Projects/diff2flow-alignment/training_scripts/train_diff2flow.py \
  --pretrained_model_path /scratch/ram1g23/ddpm_cifar10_long2/final_model_ema \
  --dataset cifar10 --num_epochs 20 \
  --save_images_every 5 --save_model_every 5 \
  --num_inference_steps 50 \
  --output_dir ${OUT} \
  ${FLAGS}
ENDJOB
)
echo "  Exp1 train (7-way)     -> job $J_RQ2_TRAIN"

J_RQ2_EVAL=$(sbatch --parsable --array=1-7 --job-name=rq2_eval \
    --dependency=afterok:${J_RQ2_TRAIN} << 'ENDJOB'
#!/bin/bash
#SBATCH --partition=ecsstudents_l4
#SBATCH --account=ecsstudents
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=6
#SBATCH --time=4:00:00
#SBATCH --output=logs/rq2_eval_%A_%a.out
#SBATCH --error=logs/rq2_eval_%A_%a.err
module load conda
source activate /home/ram1g23/.conda/envs/diffusion_flow_study
ln -sf ~/inception_cache/inception-2015-12-05.pt /tmp/inception-2015-12-05.pt
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

declare -a VARIANT_FLAGS=(
    ""
    "--no_interpolant_rescaling --no_velocity_translation"
    "--no_timestep_rescaling --no_velocity_translation"
    "--no_timestep_rescaling --no_interpolant_rescaling"
    "--no_velocity_translation"
    "--no_interpolant_rescaling"
    "--no_timestep_rescaling"
    ""
)
declare -a VARIANT_NAMES=("" "T" "I" "V" "TI" "TV" "IV" "TIV")

FLAGS=${VARIANT_FLAGS[$SLURM_ARRAY_TASK_ID]}
NAME=${VARIANT_NAMES[$SLURM_ARRAY_TASK_ID]}
CKPT=/scratch/ram1g23/exp1_alignment/${NAME}/final_model/diffusion_pytorch_model.safetensors
OUT=/iridisfs/home/ram1g23/Projects/diff2flow-alignment/results/exp1_alignment_ablation/${NAME}/${RUN_TAG}
echo "Evaluating: ${NAME}  Checkpoint: ${CKPT}"

python /iridisfs/home/ram1g23/Projects/diff2flow-alignment/evaluation_scripts/metrics_diff2flow.py \
  --checkpoint_path ${CKPT} \
  --pretrained_model_path /scratch/ram1g23/ddpm_cifar10_long2/final_model_ema \
  --dataset cifar10 --num_samples 10000 \
  --step_counts 2 4 10 25 50 \
  --output_dir ${OUT} \
  --scratch_dir /scratch/ram1g23/exp1_tmp_${NAME}_${SLURM_JOB_ID} \
  ${FLAGS}
ENDJOB
)
echo "  Exp1 eval  (7-way)     -> job $J_RQ2_EVAL  (after $J_RQ2_TRAIN)"

# ==============================================================================
# RQ1b  Speed–quality — Diff2Flow TIV at low NFE (exp3)
#       Depends on RQ2 training so the TIV checkpoint exists.
# ==============================================================================
echo ""
echo "── RQ1b: Diff2Flow low-NFE curve (exp3, after RQ2 train) ──"

J_RQ1_LOWNFE=$(sbatch --parsable --job-name=rq1_lownfe \
    --dependency=afterok:${J_RQ2_TRAIN} << 'ENDJOB'
#!/bin/bash
#SBATCH --partition=ecsstudents_l4
#SBATCH --account=ecsstudents
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=6
#SBATCH --time=4:00:00
#SBATCH --output=logs/rq1_lownfe_%j.out
#SBATCH --error=logs/rq1_lownfe_%j.err
module load conda
source activate /home/ram1g23/.conda/envs/diffusion_flow_study
ln -sf ~/inception_cache/inception-2015-12-05.pt /tmp/inception-2015-12-05.pt
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

CKPT=/scratch/ram1g23/exp1_alignment/TIV/final_model/diffusion_pytorch_model.safetensors
OUT=/iridisfs/home/ram1g23/Projects/diff2flow-alignment/results/exp1_lownfe/${RUN_TAG}
echo "Job $SLURM_JOB_ID on $(hostname)"
echo "Checkpoint: ${CKPT}"

python /iridisfs/home/ram1g23/Projects/diff2flow-alignment/evaluation_scripts/metrics_diff2flow.py \
  --checkpoint_path ${CKPT} \
  --pretrained_model_path /scratch/ram1g23/ddpm_cifar10_long2/final_model_ema \
  --dataset cifar10 --num_samples 10000 \
  --step_counts 2 4 8 16 50 \
  --output_dir ${OUT} \
  --scratch_dir /scratch/ram1g23/exp3_tmp_${SLURM_JOB_ID}
ENDJOB
)
echo "  Diff2Flow TIV low-NFE  -> job $J_RQ1_LOWNFE  (after $J_RQ2_TRAIN)"

# ==============================================================================
# RQ3  Parameter efficiency — LoRA ranks 4/8/16/32/64/128 vs full fine-tuning
# ==============================================================================
echo ""
echo "── RQ3: LoRA rank sweep — 6 ranks + full FT (exp2) ──"

J_RQ3_TRAIN=$(sbatch --parsable --array=1-7 --job-name=rq3_train \
    ${J_DDPM_TRAIN:+--dependency=afterok:${J_DDPM_TRAIN}} << 'ENDJOB'
#!/bin/bash
#SBATCH --partition=ecsstudents_l4
#SBATCH --account=ecsstudents
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=12
#SBATCH --time=24:00:00
#SBATCH --output=logs/rq3_train_%A_%a.out
#SBATCH --error=logs/rq3_train_%A_%a.err
module load conda
source activate /home/ram1g23/.conda/envs/diffusion_flow_study
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
echo "Job $SLURM_JOB_ID  Array $SLURM_ARRAY_TASK_ID on $(hostname)"
python - <<PYEOF
import torch
print("Torch:", torch.__version__, "| CUDA:", torch.cuda.is_available(), "| GPUs:", torch.cuda.device_count())
PYEOF

# Index | Variant    | Flags
#   1   | rank 4     | --use_lora --lora_rank 4
#   2   | rank 8     | --use_lora --lora_rank 8
#   3   | rank 16    | --use_lora --lora_rank 16
#   4   | rank 32    | --use_lora --lora_rank 32
#   5   | rank 64    | --use_lora --lora_rank 64
#   6   | rank 128   | --use_lora --lora_rank 128
#   7   | full FT    | (no LoRA flags)
declare -a RANKS=(0 4 8 16 32 64 128 0)
declare -a NAMES=("" "rank4" "rank8" "rank16" "rank32" "rank64" "rank128" "full_ft")

IDX=$SLURM_ARRAY_TASK_ID
NAME=${NAMES[$IDX]}
RANK=${RANKS[$IDX]}
OUT=/scratch/ram1g23/exp2_lora_rank/${NAME}

if [ $IDX -eq 7 ]; then
    LORA_FLAGS=""
    echo "Variant: full finetuning  Output: ${OUT}"
else
    LORA_FLAGS="--use_lora --lora_rank ${RANK}"
    echo "Variant: LoRA rank=${RANK}  Output: ${OUT}"
fi

accelerate launch --num_processes=2 \
  /iridisfs/home/ram1g23/Projects/diff2flow-alignment/training_scripts/train_diff2flow.py \
  --pretrained_model_path /scratch/ram1g23/ddpm_cifar10_long2/final_model_ema \
  --dataset cifar10 --num_epochs 20 \
  --save_images_every 5 --save_model_every 5 \
  --num_inference_steps 50 \
  --output_dir ${OUT} \
  ${LORA_FLAGS}
ENDJOB
)
echo "  Exp2 train (7-way)     -> job $J_RQ3_TRAIN"

J_RQ3_EVAL=$(sbatch --parsable --array=1-7 --job-name=rq3_eval \
    --dependency=afterok:${J_RQ3_TRAIN} << 'ENDJOB'
#!/bin/bash
#SBATCH --partition=ecsstudents_l4
#SBATCH --account=ecsstudents
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=6
#SBATCH --time=4:00:00
#SBATCH --output=logs/rq3_eval_%A_%a.out
#SBATCH --error=logs/rq3_eval_%A_%a.err
module load conda
source activate /home/ram1g23/.conda/envs/diffusion_flow_study
ln -sf ~/inception_cache/inception-2015-12-05.pt /tmp/inception-2015-12-05.pt
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

declare -a RANKS=(0 4 8 16 32 64 128 0)
declare -a NAMES=("" "rank4" "rank8" "rank16" "rank32" "rank64" "rank128" "full_ft")

IDX=$SLURM_ARRAY_TASK_ID
NAME=${NAMES[$IDX]}
RANK=${RANKS[$IDX]}
CKPT=/scratch/ram1g23/exp2_lora_rank/${NAME}/final_model/diffusion_pytorch_model.safetensors
OUT=/iridisfs/home/ram1g23/Projects/diff2flow-alignment/results/exp2_lora_rank_sweep/${NAME}/${RUN_TAG}

if [ $IDX -eq 7 ]; then
    LORA_FLAGS=""
    echo "Evaluating: full finetuning  Checkpoint: ${CKPT}"
else
    LORA_FLAGS="--use_lora --lora_rank ${RANK}"
    echo "Evaluating: LoRA rank=${RANK}  Checkpoint: ${CKPT}"
fi

python /iridisfs/home/ram1g23/Projects/diff2flow-alignment/evaluation_scripts/metrics_diff2flow.py \
  --checkpoint_path ${CKPT} \
  --pretrained_model_path /scratch/ram1g23/ddpm_cifar10_long2/final_model_ema \
  --dataset cifar10 --num_samples 10000 \
  --step_counts 10 25 50 \
  --output_dir ${OUT} \
  --scratch_dir /scratch/ram1g23/exp2_tmp_${NAME}_${SLURM_JOB_ID} \
  ${LORA_FLAGS}
ENDJOB
)
echo "  Exp2 eval  (7-way)     -> job $J_RQ3_EVAL  (after $J_RQ3_TRAIN)"

# ==============================================================================
# RQ4  Solver sensitivity — Euler vs Heun at matched NFE budget
#      Reuses the TIV checkpoint; depends on RQ2 training.
# ==============================================================================
echo ""
echo "── RQ4: Solver sensitivity — Euler vs Heun (exp4, after RQ2 train) ──"

J_RQ4_EVAL=$(sbatch --parsable --job-name=rq4_solvers \
    --dependency=afterok:${J_RQ2_TRAIN} << 'ENDJOB'
#!/bin/bash
#SBATCH --partition=ecsstudents_l4
#SBATCH --account=ecsstudents
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=6
#SBATCH --time=6:00:00
#SBATCH --output=logs/rq4_solvers_%j.out
#SBATCH --error=logs/rq4_solvers_%j.err
module load conda
source activate /home/ram1g23/.conda/envs/diffusion_flow_study
ln -sf ~/inception_cache/inception-2015-12-05.pt /tmp/inception-2015-12-05.pt
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

CKPT=/scratch/ram1g23/exp1_alignment/TIV/final_model/diffusion_pytorch_model.safetensors
PRETRAIN=/scratch/ram1g23/ddpm_cifar10_long2/final_model_ema
echo "Job $SLURM_JOB_ID on $(hostname)"
echo "Checkpoint: ${CKPT}"

# Euler: NFE = step count directly
echo "===== Euler (NFE = 2, 4, 8) ====="
python /iridisfs/home/ram1g23/Projects/diff2flow-alignment/evaluation_scripts/metrics_diff2flow.py \
  --checkpoint_path ${CKPT} --pretrained_model_path ${PRETRAIN} \
  --dataset cifar10 --num_samples 10000 \
  --step_counts 2 4 8 --solver euler \
  --output_dir /iridisfs/home/ram1g23/Projects/diff2flow-alignment/results/exp4_solver_sensitivity/euler/${RUN_TAG} \
  --scratch_dir /scratch/ram1g23/exp4_tmp_euler_${SLURM_JOB_ID}

# Heun: NFE = 2× step count; use half the steps to match the same NFE budget
echo "===== Heun (NFE = 2, 4, 8 at half the step count) ====="
python /iridisfs/home/ram1g23/Projects/diff2flow-alignment/evaluation_scripts/metrics_diff2flow.py \
  --checkpoint_path ${CKPT} --pretrained_model_path ${PRETRAIN} \
  --dataset cifar10 --num_samples 10000 \
  --step_counts 1 2 4 --solver heun \
  --output_dir /iridisfs/home/ram1g23/Projects/diff2flow-alignment/results/exp4_solver_sensitivity/heun/${RUN_TAG} \
  --scratch_dir /scratch/ram1g23/exp4_tmp_heun_${SLURM_JOB_ID}
ENDJOB
)
echo "  Euler vs Heun eval     -> job $J_RQ4_EVAL  (after $J_RQ2_TRAIN)"

# ==============================================================================
# RQ5  LoRA placement — attention-only / feedforward-only / all layers
#      Rank fixed at 64; uses full T+I+V alignment.
# ==============================================================================
echo ""
echo "── RQ5: LoRA placement — 3 variants (exp5) ──"

J_RQ5_TRAIN=$(sbatch --parsable --array=1-3 --job-name=rq5_train \
    ${J_DDPM_TRAIN:+--dependency=afterok:${J_DDPM_TRAIN}} << 'ENDJOB'
#!/bin/bash
#SBATCH --partition=ecsstudents_l4
#SBATCH --account=ecsstudents
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=12
#SBATCH --time=24:00:00
#SBATCH --output=logs/rq5_train_%A_%a.out
#SBATCH --error=logs/rq5_train_%A_%a.err
module load conda
source activate /home/ram1g23/.conda/envs/diffusion_flow_study
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
echo "Job $SLURM_JOB_ID  Array $SLURM_ARRAY_TASK_ID on $(hostname)"
python - <<PYEOF
import torch
print("Torch:", torch.__version__, "| CUDA:", torch.cuda.is_available(), "| GPUs:", torch.cuda.device_count())
PYEOF

# Index | Placement   | Description
#   1   | all         | LoRA on all Linear and Conv2d layers
#   2   | attention   | LoRA on attention blocks only
#   3   | feedforward | LoRA on non-attention layers (conv, time emb, etc.)
declare -a PLACEMENTS=("" "all" "attention" "feedforward")
declare -a NAMES=("" "lora_all" "lora_attention" "lora_feedforward")

PLACEMENT=${PLACEMENTS[$SLURM_ARRAY_TASK_ID]}
NAME=${NAMES[$SLURM_ARRAY_TASK_ID]}
OUT=/scratch/ram1g23/exp5_placement/${NAME}
echo "Placement: ${PLACEMENT}  Output: ${OUT}"

accelerate launch --num_processes=2 \
  /iridisfs/home/ram1g23/Projects/diff2flow-alignment/training_scripts/train_diff2flow.py \
  --pretrained_model_path /scratch/ram1g23/ddpm_cifar10_long2/final_model_ema \
  --dataset cifar10 --num_epochs 20 \
  --save_images_every 5 --save_model_every 5 \
  --num_inference_steps 50 \
  --use_lora --lora_rank 64 \
  --lora_placement ${PLACEMENT} \
  --output_dir ${OUT}
ENDJOB
)
echo "  Exp5 train (3-way)     -> job $J_RQ5_TRAIN"

J_RQ5_EVAL=$(sbatch --parsable --array=1-3 --job-name=rq5_eval \
    --dependency=afterok:${J_RQ5_TRAIN} << 'ENDJOB'
#!/bin/bash
#SBATCH --partition=ecsstudents_l4
#SBATCH --account=ecsstudents
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=6
#SBATCH --time=4:00:00
#SBATCH --output=logs/rq5_eval_%A_%a.out
#SBATCH --error=logs/rq5_eval_%A_%a.err
module load conda
source activate /home/ram1g23/.conda/envs/diffusion_flow_study
ln -sf ~/inception_cache/inception-2015-12-05.pt /tmp/inception-2015-12-05.pt
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

declare -a PLACEMENTS=("" "all" "attention" "feedforward")
declare -a NAMES=("" "lora_all" "lora_attention" "lora_feedforward")

PLACEMENT=${PLACEMENTS[$SLURM_ARRAY_TASK_ID]}
NAME=${NAMES[$SLURM_ARRAY_TASK_ID]}
CKPT=/scratch/ram1g23/exp5_placement/${NAME}/final_model/diffusion_pytorch_model.safetensors
OUT=/iridisfs/home/ram1g23/Projects/diff2flow-alignment/results/exp5_lora_placement/${NAME}/${RUN_TAG}
echo "Evaluating placement: ${PLACEMENT}  Checkpoint: ${CKPT}"

python /iridisfs/home/ram1g23/Projects/diff2flow-alignment/evaluation_scripts/metrics_diff2flow.py \
  --checkpoint_path ${CKPT} \
  --pretrained_model_path /scratch/ram1g23/ddpm_cifar10_long2/final_model_ema \
  --dataset cifar10 --num_samples 10000 \
  --step_counts 10 25 50 \
  --use_lora --lora_rank 64 \
  --lora_placement ${PLACEMENT} \
  --output_dir ${OUT} \
  --scratch_dir /scratch/ram1g23/exp5_tmp_${NAME}_${SLURM_JOB_ID}
ENDJOB
)
echo "  Exp5 eval  (3-way)     -> job $J_RQ5_EVAL  (after $J_RQ5_TRAIN)"

# ==============================================================================
# RQ6  Trajectory curvature — DDPM vs FM vs Diff2Flow (TIV)
#      Depends on RQ2 training for the TIV checkpoint.
# ==============================================================================
echo ""
echo "── RQ6: Trajectory curvature visualisation (after RQ2 train) ──"

J_RQ6_TRAJ=$(sbatch --parsable --job-name=rq6_traj \
    --dependency=afterok:${J_RQ2_TRAIN}${J_FM_TRAIN:+:${J_FM_TRAIN}} << 'ENDJOB'
#!/bin/bash
#SBATCH --partition=ecsstudents_l4
#SBATCH --account=ecsstudents
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --time=2:00:00
#SBATCH --output=logs/rq6_traj_%j.out
#SBATCH --error=logs/rq6_traj_%j.err
module load conda
source activate /home/ram1g23/.conda/envs/diffusion_flow_study
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

echo "Job $SLURM_JOB_ID on $(hostname)"

python /iridisfs/home/ram1g23/Projects/diff2flow-alignment/evaluation_scripts/visualise_trajectory.py \
  --ddpm_model      /scratch/ram1g23/ddpm_cifar10_long2/final_model_ema \
  --flow_model      /scratch/ram1g23/fm_cifar10_long2/final_model_ema \
  --diff2flow_model /scratch/ram1g23/exp1_alignment/TIV/final_model \
  --output_dir      /iridisfs/home/ram1g23/Projects/diff2flow-alignment/results/trajectory_vis/${RUN_TAG} \
  --num_samples     500 \
  --num_steps       50 \
  --num_snapshots   200 \
  --seed            42
ENDJOB
)
echo "  Trajectory curvature   -> job $J_RQ6_TRAJ  (after $J_RQ2_TRAIN)"

# ==============================================================================
# Summary
# ==============================================================================
echo ""
echo "=============================================="
echo "  All jobs submitted. Summary:"
echo "=============================================="
echo ""
echo "  Phase 0  Pretraining"
echo "       DDPM (1000 epochs)        -> $J_DDPM_TRAIN"
echo "       FM   (1000 epochs)        -> $J_FM_TRAIN"
echo ""
echo "  RQ1  Speed-quality tradeoff"
echo "       DDPM (ancestral+DDIM)    -> $J_RQ1_DDPM"
echo "       FM from scratch          -> $J_RQ1_FM"
echo "       Naive transfer           -> $J_RQ1_NAIVE"
echo "       Alignment-only (no FT)   -> $J_RQ1_ALIGNONLY"
echo "       Diff2Flow TIV low-NFE    -> $J_RQ1_LOWNFE  (after $J_RQ2_TRAIN)"
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
echo "       Eval                     -> $J_RQ4_EVAL  (after $J_RQ2_TRAIN)"
echo ""
echo "  RQ5  LoRA placement"
echo "       Train 3-way array        -> $J_RQ5_TRAIN"
echo "       Eval  3-way array        -> $J_RQ5_EVAL"
echo ""
echo "  RQ6  Trajectory curvature"
echo "       Visualise DDPM/FM/D2F    -> $J_RQ6_TRAJ  (after $J_RQ2_TRAIN)"
echo ""
echo "  Logs: experiments/logs/"
echo "=============================================="
