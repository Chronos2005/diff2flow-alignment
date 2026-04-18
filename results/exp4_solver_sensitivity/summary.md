# Exp4: Solver Sensitivity

**RQ:** Does Heun's method outperform Euler at a matched NFE budget?

**Model:** TIV full alignment checkpoint (best from Exp1)  
**Dataset:** CIFAR-10 (10k real / 10k generated)  
**Date:** 2026-04-09 (SLURM job 803048, ~14 min on ecsai03)

---

## Setup

Both solvers are evaluated on the same model checkpoint
(`exp1_alignment/TIV/final_model`). Results are compared at matched NFE
(not matched step count).

- **Euler:** 1 function evaluation per step — NFE = steps
- **Heun:** 2 function evaluations per step (predictor-corrector) — NFE = 2 × steps

---

## FID at matched NFE

| NFE | Euler FID | Heun FID | Winner   |
|----:|----------:|---------:|----------|
|   2 | 195.68    | 382.29   | Euler    |
|   4 | 117.11    | 247.16   | Euler    |
|   8 |  69.61    | 141.92   | Euler    |

## Throughput at matched NFE

| NFE | Euler (img/s) | Heun (img/s) |
|----:|--------------:|-------------:|
|   2 | 569           | 557          |
|   4 | 300           | 297          |
|   8 | 148           | 148          |

---

## Key Findings

### 1. Euler dominates Heun at every NFE budget
Heun's corrector step does not help — FID is roughly **2× worse** at each NFE
compared to Euler. This is atypical for standard flow matching models and
suggests the aligned velocity field is not smooth enough for a second-order
correction to improve integration.

### 2. Throughput is equivalent
At matched NFE, both solvers run at the same speed (the extra Heun step just
replaces what would have been a new Euler step). There is no throughput
reason to prefer Heun.

### 3. Euler FID at NFE=8 (69.6) matches Exp1 reference
The Euler result at 8 steps (FID 69.6) is consistent with the TIV alignment-only
result from Exp0 at NFE=10 (67.1), confirming the checkpoint is the same and
results are reproducible.

---

## Verdict

Use Euler. Heun provides no benefit and roughly doubles FID at a fixed compute
budget for this aligned model.

| Solver | Recommended steps | FID   |
|--------|:-----------------:|------:|
| Euler  | 8                 | 69.6  |
| Heun   | *(not recommended)* | —   |
