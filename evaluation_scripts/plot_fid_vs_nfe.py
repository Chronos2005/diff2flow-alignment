"""
FID vs NFE line plot — paper Figure (RQ3).

Three curves:
  - DDPM (DDIM sampler)          ddpm_853268
  - Flow Matching (from scratch)  fm_853101
  - Diff2Flow (alignment only)    alignment_only_853103

Usage:
    python evaluation_scripts/plot_fid_vs_nfe.py [--out figures/fid_vs_nfe.pdf]
"""

import argparse
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from pathlib import Path

# ---------------------------------------------------------------------------
# Data — hard-coded from CSVs so the plot can run without pandas
# ---------------------------------------------------------------------------

DDPM_DDIM = {
    "nfe":  [2,        4,       10,      25,      50,      100,     1000],
    "fid":  [237.0774, 74.6178, 29.9521, 22.3153, 21.3021, 21.6954, 18.9246],
}
# NFE=1000 is the ancestral sampler — plotted as a distinct marker
# NFE 2/4 from job 853268 re-run; 10–250 from results/exp0_baselines/results.csv

FM_SCRATCH = {
    "nfe":  [2,        4,       10,      25,      50,      100],
    "fid":  [187.2398, 75.3885, 30.5979, 21.8021, 20.6279, 20.7843],
}
# fm_853101

ALIGNMENT_ONLY = {
    "nfe":  [2,        4,       10,      25,      50,      100],
    "fid":  [209.2463, 90.0449, 38.659,  28.7515, 27.5729, 27.8026],
}
# alignment_only_853103


def log_interp_crossover(nfe_a, fid_a, nfe_b, fid_b):
    """Return NFE where curve A and curve B cross, interpolating in log-NFE space."""
    log_nfe_a = np.log(nfe_a)
    log_nfe_b = np.log(nfe_b)
    gaps = np.array(fid_a) - np.array(fid_b)

    # Find consecutive pairs where sign changes
    crossovers = []
    for i in range(len(gaps) - 1):
        if gaps[i] * gaps[i + 1] <= 0:          # sign change (or exact zero)
            t = gaps[i] / (gaps[i] - gaps[i + 1])
            log_x = log_nfe_a[i] + t * (log_nfe_a[i + 1] - log_nfe_a[i])
            crossovers.append(np.exp(log_x))
    return crossovers


def main(out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # -----------------------------------------------------------------------
    # Style
    # -----------------------------------------------------------------------
    matplotlib.rcParams.update({
        "font.family":      "serif",
        "font.size":        11,
        "axes.titlesize":   12,
        "axes.labelsize":   12,
        "legend.fontsize":  10,
        "xtick.labelsize":  10,
        "ytick.labelsize":  10,
        "lines.linewidth":  1.8,
        "lines.markersize": 6,
        "figure.dpi":       150,
        "savefig.dpi":      300,
        "savefig.bbox":     "tight",
    })

    DDPM_COLOR  = "#2166ac"   # blue
    FM_COLOR    = "#d6604d"   # red-orange
    D2F_COLOR   = "#1a9641"   # green

    fig, ax = plt.subplots(figsize=(6.5, 4.2))

    # -----------------------------------------------------------------------
    # DDPM: main DDIM curve (NFE 2–100) + ancestral marker at 1000
    # -----------------------------------------------------------------------
    ddpm_x = DDPM_DDIM["nfe"][:-1]          # 2–100
    ddpm_y = DDPM_DDIM["fid"][:-1]
    ax.plot(ddpm_x, ddpm_y,
            color=DDPM_COLOR, marker="s", label="DDPM (DDIM sampler)")
    # Ancestral point at NFE=1000
    ax.plot(DDPM_DDIM["nfe"][-1], DDPM_DDIM["fid"][-1],
            color=DDPM_COLOR, marker="*", markersize=10, linestyle="none",
            label="DDPM (ancestral, 1000 steps)")

    # -----------------------------------------------------------------------
    # FM scratch
    # -----------------------------------------------------------------------
    ax.plot(FM_SCRATCH["nfe"], FM_SCRATCH["fid"],
            color=FM_COLOR, marker="^", label="FM (trained from scratch)")

    # -----------------------------------------------------------------------
    # Diff2Flow (alignment only)
    # -----------------------------------------------------------------------
    ax.plot(ALIGNMENT_ONLY["nfe"], ALIGNMENT_ONLY["fid"],
            color=D2F_COLOR, marker="o", label="Diff2Flow (no-fine-tuning)")

    # -----------------------------------------------------------------------
    # Crossover annotations
    # -----------------------------------------------------------------------
    shared_nfe  = ALIGNMENT_ONLY["nfe"]
    ddpm_shared = DDPM_DDIM["fid"][:-1]   # 2–100, drop ancestral
    fm_shared   = FM_SCRATCH["fid"]
    d2f_shared  = ALIGNMENT_ONLY["fid"]

    # Diff2Flow vs FM scratch crossover
    d2f_vs_fm = log_interp_crossover(
        shared_nfe, d2f_shared,
        shared_nfe, fm_shared,
    )
    if d2f_vs_fm:
        cx = d2f_vs_fm[0]
        cy = np.interp(np.log(cx), np.log(shared_nfe), d2f_shared)
        ax.axvline(cx, color="grey", linestyle=":", linewidth=1.0, alpha=0.7)
        ax.annotate(
            f"D2F ≈ FM\n(NFE≈{cx:.0f})",
            xy=(cx, cy),
            xytext=(cx * 1.4, cy + 18),
            fontsize=8.5,
            color="grey",
            arrowprops=dict(arrowstyle="->", color="grey", lw=0.8),
        )

    # Diff2Flow vs DDPM gap at NFE=25
    gap_nfe = 25
    ddpm_y_at_25 = np.interp(np.log(gap_nfe), np.log(shared_nfe), ddpm_shared)
    d2f_y_at_25  = np.interp(np.log(gap_nfe), np.log(shared_nfe), d2f_shared)
    ax.annotate(
        "",
        xy=(gap_nfe, d2f_y_at_25),
        xytext=(gap_nfe, ddpm_y_at_25),
        arrowprops=dict(arrowstyle="<->", color="grey", lw=1.0),
    )
    ax.text(gap_nfe * 1.12, (ddpm_y_at_25 + d2f_y_at_25) / 2,
            f"Δ={ddpm_y_at_25 - d2f_y_at_25:.1f}",
            fontsize=8, color="grey", va="center")

    # -----------------------------------------------------------------------
    # Axes
    # -----------------------------------------------------------------------
    ax.set_xscale("log")
    ax.set_xlim(1.5, 1500)
    ax.set_ylim(15, 260)

    ax.xaxis.set_major_formatter(ticker.FuncFormatter(
        lambda x, _: f"{int(x)}" if x >= 1 else f"{x:.1f}"
    ))
    ax.xaxis.set_minor_formatter(ticker.NullFormatter())
    ax.set_xticks([2, 4, 10, 25, 50, 100, 1000])

    ax.set_xlabel("Number of Function Evaluations (NFE)")
    ax.set_ylabel("FID (↓ better)")
    ax.set_title("FID vs NFE on CIFAR-10")

    ax.grid(True, which="major", linestyle="--", linewidth=0.5, alpha=0.5)
    ax.grid(True, which="minor", linestyle=":",  linewidth=0.3, alpha=0.3)

    ax.legend(loc="upper right", framealpha=0.9)

    # -----------------------------------------------------------------------
    # Save
    # -----------------------------------------------------------------------
    fig.tight_layout()
    fig.savefig(out_path)
    print(f"Saved → {out_path}")

    # Also save PNG alongside for quick viewing
    png_path = out_path.with_suffix(".png")
    fig.savefig(png_path)
    print(f"Saved → {png_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default="figures/fid_vs_nfe.pdf",
                        help="Output path (PDF). A .png is saved alongside.")
    args = parser.parse_args()
    main(Path(args.out))
