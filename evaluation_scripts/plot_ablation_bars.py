"""
Alignment ablation grouped bar chart — paper Figure (Table 4 companion).

Seven variants are shown (T, I, T+I, V, T+V, I+V, T+I+V) with three bars
per group: FID@NFE=2, FID@NFE=10, FID@NFE=50.

Key story: V (velocity translation) is the decisive component — any variant
without V sits near FID=464 regardless of what else is included.

Y-axis is capped at 500.  Bars that would exceed 500 are drawn at 500 and
annotated with the true value so no information is lost.

Usage:
    python evaluation_scripts/plot_ablation_bars.py [--out figures/ablation_bars.pdf]
"""

import argparse
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pathlib import Path

# ---------------------------------------------------------------------------
# Data — extracted from results/exp1_alignment_ablation/results.csv
# Only LoRA-fine-tuned ablation variants; FM_scratch used as a reference line.
# ---------------------------------------------------------------------------

# (variant_label, components, fid@2, fid@10, fid@50)
ABLATION = [
    ("T",       "T",     449.62, 415.18, 403.58),
    ("I",       "I",     464.31, 464.31, 464.31),
    ("T+I",     "T+I",   464.35, 464.35, 464.34),
    ("V",       "V",     283.76, 271.67, 268.01),
    ("T+V",     "T+V",   175.25,  53.76,  42.03),
    ("I+V",     "I+V",   288.64, 264.41, 259.74),
    ("T+I+V",   "T+I+V", 195.68,  62.82,  44.65),
]

FM_SCRATCH = {"nfe2": 214.15, "nfe10": 49.71, "nfe50": 42.69}

Y_CAP = 500          # bars clipped here; annotated with true value above
BAR_WIDTH = 0.22     # width of each individual bar
CLIP_LABEL_SIZE = 7  # font size for "↑xxx" overflow annotations


def main(out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # -----------------------------------------------------------------------
    # Style — match the rest of the paper
    # -----------------------------------------------------------------------
    matplotlib.rcParams.update({
        "font.family":      "serif",
        "font.size":        11,
        "axes.titlesize":   12,
        "axes.labelsize":   12,
        "legend.fontsize":  10,
        "xtick.labelsize":  10,
        "ytick.labelsize":  10,
        "figure.dpi":       150,
        "savefig.dpi":      300,
        "savefig.bbox":     "tight",
    })

    # NFE colours: dark→light as NFE decreases (harder→easier)
    C2   = "#c2523c"   # NFE=2  — dark red (hardest)
    C10  = "#f4a261"   # NFE=10 — amber
    C50  = "#2a9d8f"   # NFE=50 — teal (easiest / best FID)

    variants = [row[0] for row in ABLATION]
    fid2  = np.array([row[2] for row in ABLATION])
    fid10 = np.array([row[3] for row in ABLATION])
    fid50 = np.array([row[4] for row in ABLATION])

    n = len(variants)
    x = np.arange(n)

    offsets = np.array([-1, 0, 1]) * BAR_WIDTH   # centres for the 3 bars

    fig, ax = plt.subplots(figsize=(8.5, 4.6))

    # -----------------------------------------------------------------------
    # Draw bars, clipping at Y_CAP and annotating overflows
    # -----------------------------------------------------------------------
    def draw_bars(ax, positions, values, color, label):
        clipped = np.minimum(values, Y_CAP)
        bars = ax.bar(positions, clipped, width=BAR_WIDTH,
                      color=color, label=label,
                      edgecolor="white", linewidth=0.4, zorder=3)

        for bar, raw in zip(bars, values):
            if raw > Y_CAP:
                bx = bar.get_x() + bar.get_width() / 2
                ax.text(bx, Y_CAP + 6, f"↑{raw:.0f}",
                        ha="center", va="bottom",
                        fontsize=CLIP_LABEL_SIZE, color=color,
                        rotation=90, clip_on=False)
        return bars

    draw_bars(ax, x + offsets[0], fid2,  C2,  "NFE = 2")
    draw_bars(ax, x + offsets[1], fid10, C10, "NFE = 10")
    draw_bars(ax, x + offsets[2], fid50, C50, "NFE = 50")

    # -----------------------------------------------------------------------
    # FM-from-scratch reference lines (dashed, per NFE colour)
    # -----------------------------------------------------------------------
    ref_alpha = 0.55
    ax.axhline(FM_SCRATCH["nfe2"],  color=C2,  linestyle="--",
               linewidth=1.1, alpha=ref_alpha, zorder=2)
    ax.axhline(FM_SCRATCH["nfe10"], color=C10, linestyle="--",
               linewidth=1.1, alpha=ref_alpha, zorder=2)
    ax.axhline(FM_SCRATCH["nfe50"], color=C50, linestyle="--",
               linewidth=1.1, alpha=ref_alpha, zorder=2)

    # Small "FM scratch" labels at the right margin for each reference line
    right_x = n - 0.35
    for fid_val, color, nfe in [
        (FM_SCRATCH["nfe2"],  C2,  2),
        (FM_SCRATCH["nfe10"], C10, 10),
        (FM_SCRATCH["nfe50"], C50, 50),
    ]:
        ax.text(right_x, fid_val + 4, f"FM scratch\n(NFE={nfe})",
                ha="right", va="bottom", fontsize=7,
                color=color, alpha=0.8)

    # -----------------------------------------------------------------------
    # Vertical divider: "no V" vs "contains V"
    # -----------------------------------------------------------------------
    divider_x = 2.5   # between T+I (index 2) and V (index 3)
    ax.axvline(divider_x, color="gray", linestyle=":", linewidth=1.2,
               alpha=0.6, zorder=1)

    # Bracket labels above the divider
    ax.text(1.0,  Y_CAP * 1.04, "No V component",
            ha="center", va="bottom", fontsize=9,
            color="gray", style="italic")
    ax.text(4.5,  Y_CAP * 1.04, "Contains V",
            ha="center", va="bottom", fontsize=9,
            color="gray", style="italic")

    # -----------------------------------------------------------------------
    # Axes & labels
    # -----------------------------------------------------------------------
    ax.set_xticks(x)
    ax.set_xticklabels(variants)
    ax.set_xlim(-0.6, n - 0.4)
    ax.set_ylim(0, Y_CAP)

    ax.set_xlabel("Alignment components active (all variants LoRA-fine-tuned)")
    ax.set_ylabel("FID  (↓ better)")
    ax.set_title("Alignment Component Ablation on CIFAR-10")

    ax.grid(True, axis="y", linestyle="--", linewidth=0.5, alpha=0.4, zorder=0)
    ax.set_axisbelow(True)

    # -----------------------------------------------------------------------
    # Legend
    # -----------------------------------------------------------------------
    handles, labels = ax.get_legend_handles_labels()
    dash_patch = mpatches.Patch(
        facecolor="none", edgecolor="gray",
        linestyle="--", linewidth=1.5,
        label="FM scratch reference (dashed)"
    )
    ax.legend(handles + [dash_patch], labels + [dash_patch.get_label()],
              loc="upper right", framealpha=0.9)

    # -----------------------------------------------------------------------
    # Save
    # -----------------------------------------------------------------------
    fig.tight_layout()
    fig.savefig(out_path)
    print(f"Saved → {out_path}")

    png_path = out_path.with_suffix(".png")
    fig.savefig(png_path)
    print(f"Saved → {png_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default="figures/ablation_bars.pdf",
                        help="Output path (PDF). A .png is saved alongside.")
    args = parser.parse_args()
    main(Path(args.out))
