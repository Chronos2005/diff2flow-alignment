#!/bin/bash
#SBATCH --partition=ecsstudents_l4
#SBATCH --account=ecsstudents
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --time=02:00:00
#SBATCH --output=../logs/exp6_traj_%j.out
#SBATCH --error=../logs/exp6_traj_%j.err

module load conda

source activate /home/ram1g23/.conda/envs/diffusion_flow_study

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

echo "===== ENVIRONMENT CHECK ====="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $(hostname)"
echo "Python: $(which python)"
echo "Conda env: $CONDA_DEFAULT_ENV"
echo "CUDA_VISIBLE_DEVICES: $CUDA_VISIBLE_DEVICES"
echo "----- PYTORCH & CUDA -----"

python - <<EOF
import torch
print("Torch version:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("GPUs detected:", torch.cuda.device_count())
    for i in range(torch.cuda.device_count()):
        print(f"GPU {i}: {torch.cuda.get_device_name(i)}")
EOF

echo "===== END CHECK ====="

RUN_TAG=${RUN_TAG:-${SLURM_JOB_ID:-$(date +%Y%m%d_%H%M%S)}}

python ../../evaluation_scripts/visualise_trajectory.py \
    --ddpm_model      /scratch/ram1g23/ddpm_cifar10_long2/final_model_ema \
    --flow_model      /scratch/ram1g23/fm_cifar10_long2/final_model_ema \
    --diff2flow_model /scratch/ram1g23/exp2_lora_rank/rank64/final_model\
    --output_dir      /scratch/ram1g23/trajectory_vis/${RUN_TAG} \
    --num_samples     20 \
    --num_steps       50 \
    --num_snapshots   10 \
    --seed            42
