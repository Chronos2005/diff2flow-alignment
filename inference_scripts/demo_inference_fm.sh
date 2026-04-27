#!/bin/bash
#SBATCH --partition=ecsstudents_l4
#SBATCH --account=ecsstudents
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --time=0:15:00
#SBATCH --output=../experiments/logs/demo_inference_fm_%j.out
#SBATCH --error=../experiments/logs/demo_inference_fm_%j.err

module load conda
source activate /home/ram1g23/.conda/envs/diffusion_flow_study

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

MODEL_PATH=${MODEL_PATH:-/scratch/ram1g23/fm_cifar10_long2/final_model_ema}
OUTPUT_DIR=${OUTPUT_DIR:-/home/ram1g23/Projects/diff2flow-alignment/inference_scripts/demo_fm_samples}

echo "Model:      ${MODEL_PATH}"
echo "Output dir: ${OUTPUT_DIR}"

python inference_fm.py \
  --model_path   "${MODEL_PATH}" \
  --output_dir   "${OUTPUT_DIR}" \
  --num_images   6 \
  --num_steps    50 \
  --batch_size   32 \
  --solver       euler \
  --seed         42 \
  --save_individual
