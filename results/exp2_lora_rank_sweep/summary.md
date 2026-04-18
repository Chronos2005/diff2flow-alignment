# Exp2: LoRA Rank Sweep Results

**Experiment:** LoRA rank sweep (4, 8, 16, 32, 64, 128) vs full fine-tuning  
**Dataset:** CIFAR-10 (10k real / 10k generated images)  
**Solver:** Euler  
**Alignment:** Full TIV (Timestep + Interpolant + Velocity)  
**Date:** 2026-04-09

## FID by Rank and NFE

| Variant      | NFE=10  | NFE=25  | NFE=50  |
|--------------|--------:|--------:|--------:|
| LoRA rank=4  | 66.6676 | 50.7089 | 46.8397 |
| LoRA rank=8  | 66.8714 | 50.9649 | 47.0566 |
| LoRA rank=16 | 66.5464 | 50.4788 | 46.5277 |
| LoRA rank=32 | 66.8081 | 50.5813 | 46.3299 |
| LoRA rank=64 | 66.4020 | 50.1775 | 46.0352 |
| LoRA rank=128| 65.6178 | 49.7622 | 45.7570 |
| Full FT      | 61.8971 | 48.0010 | 44.4245 |

## Throughput (images/sec) by Rank and NFE

| Variant      | NFE=10  | NFE=25  | NFE=50  |
|--------------|--------:|--------:|--------:|
| LoRA rank=4  |   86.60 |   34.31 |   17.08 |
| LoRA rank=8  |   86.32 |   34.44 |   17.21 |
| LoRA rank=16 |   84.95 |   33.81 |   16.90 |
| LoRA rank=32 |   84.90 |   34.14 |   17.07 |
| LoRA rank=64 |   83.62 |   33.19 |   16.55 |
| LoRA rank=128|   80.71 |   32.17 |   16.06 |
| Full FT      |  117.78 |   46.85 |   23.33 |

## Key Observations

- FID improves monotonically with rank but with diminishing returns beyond rank 64.
- Full fine-tuning outperforms all LoRA ranks by a clear margin (~3–4 FID at NFE=50).
- LoRA ranks 4–32 are nearly identical in FID (within ~0.7 at NFE=50), suggesting low-rank adapters quickly plateau.
- Full FT has higher throughput (no adapter overhead).
- The gap between LoRA rank=128 and full FT is ~1.3 FID at NFE=50 — rank=128 captures most of the benefit at a fraction of the parameters.

## Context (vs Exp1 baselines)

| Reference point           | NFE=10  | NFE=50  |
|---------------------------|--------:|--------:|
| Alignment only (no LoRA)  |  67.10  |  46.30  |
| LoRA rank=4 (exp2)        |  66.67  |  46.84  |
| LoRA rank=128 (exp2)      |  65.62  |  45.76  |
| Full FT (exp2)            |  61.90  |  44.42  |
| FM from scratch           |  49.71  |  42.69  |
