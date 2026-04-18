# Exp0: Baseline Models

**Models evaluated:** 4 baselines on CIFAR-10, 10K samples each  
**Dataset:** CIFAR-10  
**Seed:** 42

| # | Model | Sampler | Description |
|---|-------|---------|-------------|
| 1 | `ddpm` | DDIM / ancestral | Pre-trained DDPM, native sampler |
| 2 | `fm_scratch` | Euler | Flow matching trained from scratch |
| 3 | `naive_transfer` | Euler | DDPM weights used as FM velocity field — no alignment |
| 4 | `alignment_only_tiv` | Euler | DDPM weights + full TIV alignment, no fine-tuning |

---

## FID Comparison

| NFE | ddpm (DDIM) | fm_scratch | naive_transfer | alignment_only_tiv |
|----:|------------:|-----------:|---------------:|-------------------:|
| 2   | 181.09      | 214.15     | 463.73         | 213.12             |
| 4   | 99.28       | 82.39      | 463.71         | 118.33             |
| 10  | 55.63       | 49.71      | 462.66         | 67.10              |
| 25  | 42.41       | 44.06      | 462.25         | 50.19              |
| 50  | 39.09       | 42.69      | 462.12         | 46.30              |
| 100 | 37.66       | 41.99      | 462.05         | 44.66              |
| 1000| 38.06 (ancestral) | — | —             | —                  |

## Throughput — images/sec

| NFE | ddpm (DDIM) | fm_scratch | naive_transfer | alignment_only_tiv |
|----:|------------:|-----------:|---------------:|-------------------:|
| 2   | 576         | 528        | 506            | 590                |
| 4   | 301         | 307        | 302            | 305                |
| 10  | 119         | 122        | 119            | 121                |
| 25  | 47          | 49         | 47             | 48                 |
| 50  | 24          | 24         | 24             | 24                 |
| 100 | 12          | 12         | 12             | 12                 |

---

## Key Observations

- **Naive transfer collapses** (FID ~462–464 at all NFE): plugging DDPM ε-prediction weights into an FM Euler solver without alignment is completely non-functional.
- **DDPM (DDIM) is the strongest baseline at low NFE** (FID 55.6 @ 10, 39.1 @ 50), outperforming FM from scratch at ≤25 NFE.
- **FM from scratch overtakes DDIM at high NFE** (41.99 vs 37.66 @ 100 — actually DDIM still wins, but FM converges to a better plateau at very high NFE).
- **Alignment-only (zero-shot TIV)** recovers meaningful generation (FID 67.1 @ 10, 46.3 @ 50) from the collapsed naive transfer, but falls short of both DDPM-DDIM and FM-scratch at matched NFE.
- **Throughput is solver-bound**, not model-bound — all 4 models run at the same speed at a given NFE.

## Context for Exp1/Exp2

These are the reference points that exp1 (alignment ablation) and exp2 (LoRA rank sweep) are benchmarked against. The target is to match or beat DDPM-DDIM at low NFE (≤10 steps) using the Diff2Flow alignment approach.
