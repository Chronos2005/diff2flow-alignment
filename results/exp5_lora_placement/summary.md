# Exp5: LoRA Placement Ablation Results

**Experiment:** LoRA placement ablation — all layers vs attention-only vs feedforward-only  
**Dataset:** CIFAR-10 (10k real / 10k generated images)  
**Solver:** Euler  
**LoRA Rank:** 64  
**Alignment:** Full TIV (Timestep + Interpolant + Velocity)  
**Date:** 2026-04-11

## FID by Placement and NFE

| Placement      | NFE=10  | NFE=25  | NFE=50  |
|----------------|--------:|--------:|--------:|
| All layers     | 67.0499 | 50.8497 | **46.4314** |
| Attention only | 68.3517 | 50.8087 | 46.7347 |
| Feedforward only | 67.6601 | 51.0180 | 46.7258 |

## Throughput (images/sec) by Placement and NFE

| Placement      | NFE=10  | NFE=25  | NFE=50  |
|----------------|--------:|--------:|--------:|
| All layers     |   84.89 |   34.19 |   17.07 |
| Attention only | **108.31** | **43.87** | **21.92** |
| Feedforward only |   84.63 |   34.07 |   17.02 |

## Key Observations

- LoRA placement has minimal impact on final FID — all three variants converge within ~0.3 FID at NFE=50.
- `lora_all` achieves the lowest FID at 50 steps (46.43), but only marginally over the others.
- `lora_attention` is substantially faster at inference: ~28% higher throughput than `lora_all` at NFE=50 (21.92 vs 17.07 img/s), since fewer LoRA weights are applied during the forward pass.
- At NFE=25, `lora_attention` matches `lora_all` in FID (50.81 vs 50.85) while being 28% faster — the clear efficiency winner.
- `lora_feedforward` offers no throughput benefit over `lora_all` and slightly worse FID, suggesting feedforward layers carry limited useful adaptation signal.

## Context (vs Exp1 / Exp2 references)

| Reference point              | NFE=10  | NFE=50  |
|------------------------------|--------:|--------:|
| Alignment only, no LoRA (exp1) | 67.10  |  46.30  |
| LoRA rank=64, all layers (exp2) | 66.40 |  46.04  |
| LoRA all, rank=64 (exp5)     |  67.05  |  46.43  |
| LoRA attention, rank=64 (exp5) | 68.35 |  46.73  |
| LoRA feedforward, rank=64 (exp5) | 67.66 | 46.73 |
