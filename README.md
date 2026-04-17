# diff2flow-alignment

MEng dissertation project: analytically converting pre-trained DDPM models into Flow Matching models for fast generative sampling, without training from scratch.

Based on Schusterbauer et al., *Diff2Flow: Training Flow Matching Models via Diffusion Model Alignment*, arXiv:2506.02221, 2025.

## Overview

Diffusion models produce high-quality samples but require thousands of function evaluations (NFE). Flow Matching models can generate samples in far fewer steps because their probability trajectories are straighter. **Diff2Flow** bridges these two paradigms by analytically aligning a pre-trained DDPM to the Flow Matching framework via three steps:

1. **Timestep rescaling** — non-linear SNR schedule → linear FM time t∈[0,1]
2. **Interpolant alignment** — diffusion αt/σt parameterisation → linear interpolant
3. **Velocity translation** — ε-prediction → FM velocity field v = x₀ - ε

After alignment, LoRA fine-tuning adapts the frozen DDPM weights to the FM objective with minimal extra parameters.

## Key Results (CIFAR-10)

| Model | FID ↓ | NFE | Samples/s |
|---|---|---|---|
| DDPM (baseline) | 33.25 | 1000 | 3.41 |
| Flow Matching (scratch) | 37.70 | 100 | 14.55 |
| Diff2Flow | — | — | — |

Diff2Flow trajectory straightness: **0.9795 ± 0.0074** (vs FM: 0.9785, DDPM: 0.9332)

## Repository Structure

```
diff2flow-alignment/
├── training_scripts/
│   ├── train_ddpm.py           # Phase 1: train DDPM baseline
│   ├── train_diff2flow.py      # Phase 2: Diff2Flow alignment + LoRA fine-tuning
│   ├── train_flow_matching.py  # Scratch FM training (baseline)
│   └── train_reflow.py         # Reflow for trajectory straightening
├── inference_scripts/
│   ├── inference_ddpm.py
│   ├── inference_diff2flow.py
│   └── inference_fm.py
├── evaluation_scripts/
│   ├── metrics_ddpm.py         # FID / NFE sweep for DDPM
│   ├── metrics_diff2flow.py    # FID / NFE sweep for Diff2Flow
│   ├── metrics_fm.py
│   ├── visualise_trajectory.py
│   ├── plot_fid_vs_nfe.py
│   └── plot_ablation_bars.py
├── shared/
│   ├── aligner.py              # Diff2FlowAligner: the three alignment transforms
│   ├── lora.py                 # LoRA injection and merging
│   ├── args.py                 # Shared argparse helpers
│   ├── datasets.py
│   ├── training.py
│   └── utils.py
├── job_scripts/                # SLURM batch scripts for IRIDIS X
├── trajectory_vis/             # Trajectory visualisation outputs
├── figures/                    # Paper-ready plots
├── results/                    # Evaluation CSVs and metrics
└── environment.yml
```

## Setup

```bash
conda env create -f environment.yml
conda activate diffusion_flow_study
```

Requires Python 3.10, PyTorch 2.5.1, CUDA 12.1, and `diffusers==0.35.2`.

## Usage

### 1. Train a DDPM baseline

```bash
accelerate launch training_scripts/train_ddpm.py --dataset cifar10
```

### 2. Align to Flow Matching (Diff2Flow)

```bash
accelerate launch training_scripts/train_diff2flow.py \
  --pretrained_model_path /path/to/ddpm_checkpoint \
  --dataset cifar10 \
  --num_epochs 20 \
  --use_lora --lora_rank 64
```

Alignment components can be ablated individually:

```bash
--no_timestep_rescaling     # disable step T
--no_interpolant_rescaling  # disable step I
--no_velocity_translation   # disable step V
```

### 3. Train Flow Matching from scratch (baseline)

```bash
accelerate launch training_scripts/train_flow_matching.py --dataset cifar10
```

### 4. Evaluate (FID vs NFE sweep)

```bash
python evaluation_scripts/metrics_diff2flow.py \
  --model_path /path/to/model \
  --dataset cifar10 \
  --step_counts 2 4 8 25 50 100
```

### 5. Run on IRIDIS X (SLURM)

```bash
cd job_scripts
sbatch ddpmTrain.sh
sbatch train_diff2flow_Alignment.sh
```

## Experiments (Research Questions)

| RQ | Description |
|---|---|
| RQ1 | Alignment ablation — which of T/I/V components matter? |
| RQ2 | LoRA parameter efficiency — rank vs quality trade-off |
| RQ3 | Low-NFE sampling (2–8 steps) via trajectory straightening |
| RQ4 | Solver sensitivity — Euler vs Heun |
| RQ5 | LoRA placement — attention vs feed-forward vs both |

## Compute

Trained on **IRIDIS X** (University of Southampton HPC), using 2× NVIDIA L4 GPUs via SLURM. Multi-GPU training uses Hugging Face `accelerate`.

## Citation

```bibtex
@article{schusterbauer2025diff2flow,
  title={Diff2Flow: Training Flow Matching Models via Diffusion Model Alignment},
  author={Schusterbauer et al.},
  journal={arXiv:2506.02221},
  year={2025}
}
```
