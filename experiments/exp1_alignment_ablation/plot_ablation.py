import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

data = {
    "DDIM": {
        "nfe": [10, 25, 50, 100, 250],
        "fid": [29.9521, 22.3153, 21.3021, 21.6954, 22.3839],
        "ips": [96.76, 38.62, 19.24, 9.61, 3.86],
        "color": "#4C72B0",
        "marker": "s",
        "linestyle": "--",
    },
    "Naive Transfer": {
        "nfe": [2, 4, 10, 25, 50, 100],
        "fid": [462.7037, 462.6602, 461.8495, 461.5029, 461.3824, 461.3169],
        "ips": [475.76, 245.13, 96.73, 38.51, 19.21, 9.59],
        "color": "#DD8452",
        "marker": "^",
        "linestyle": ":",
    },
    "Flow Matching (scratch)": {
        "nfe": [2, 4, 10, 25, 50, 100],
        "fid": [187.2398, 75.3885, 30.5979, 21.8021, 20.6279, 20.7843],
        "ips": [468.37, 240.82, 94.28, 37.42, 18.69, 9.34],
        "color": "#55A868",
        "marker": "D",
        "linestyle": "-.",
    },
    "Alignment Only": {
        "nfe": [2, 4, 10, 25, 50, 100],
        "fid": [209.2463, 90.0449, 38.659, 28.7515, 27.5729, 27.8026],
        "ips": [460.67, 242.0, 96.0, 38.31, 19.15, 9.57],
        "color": "#C44E52",
        "marker": "o",
        "linestyle": "-",
    },
}

DDPM_FID = 18.9246
DDPM_IPS = 3.41  # from memory: 1000-step DDPM baseline

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 4.5))

# --- Left panel: FID vs NFE ---
for label, d in data.items():
    ax1.plot(
        d["nfe"], d["fid"],
        label=label,
        color=d["color"],
        marker=d["marker"],
        linestyle=d["linestyle"],
        linewidth=1.8,
        markersize=6,
        markeredgewidth=0.8,
        markeredgecolor="white",
    )

ax1.scatter([1000], [DDPM_FID], color="#4C72B0", marker="*", s=120, zorder=5,
            label="DDPM (1000 steps)")
ax1.axhline(DDPM_FID, color="#4C72B0", linestyle=":", linewidth=0.9, alpha=0.5)

ax1.set_xscale("log")
ax1.set_xlabel("Number of Function Evaluations (NFE)", fontsize=11)
ax1.set_ylabel("FID ↓", fontsize=11)
ax1.set_title("(a) FID vs NFE", fontsize=12, fontweight="bold")
ax1.xaxis.set_major_formatter(ticker.ScalarFormatter())
ax1.xaxis.set_minor_formatter(ticker.NullFormatter())
ax1.set_xticks([2, 4, 10, 25, 50, 100, 250, 1000])
ax1.set_ylim(15, 220)
ax1.annotate("Naive Transfer ~462\n(off-scale)", xy=(10, 100), fontsize=8,
             color="#DD8452", ha="center")
ax1.legend(fontsize=9, loc="upper right")
ax1.grid(True, which="major", linestyle="--", linewidth=0.5, alpha=0.6)
ax1.grid(True, which="minor", linestyle=":", linewidth=0.3, alpha=0.4)

# --- Right panel: FID vs throughput ---
for label, d in data.items():
    ax2.plot(
        d["ips"], d["fid"],
        label=label,
        color=d["color"],
        marker=d["marker"],
        linestyle=d["linestyle"],
        linewidth=1.8,
        markersize=6,
        markeredgewidth=0.8,
        markeredgecolor="white",
    )

ax2.scatter([DDPM_IPS], [DDPM_FID], color="#4C72B0", marker="*", s=120, zorder=5,
            label="DDPM (1000 steps)")

ax2.set_xscale("log")
ax2.set_xlabel("Throughput (images/second) ↑", fontsize=11)
ax2.set_ylabel("FID ↓", fontsize=11)
ax2.set_title("(b) FID vs Throughput", fontsize=12, fontweight="bold")
ax2.xaxis.set_major_formatter(ticker.ScalarFormatter())
ax2.xaxis.set_minor_formatter(ticker.NullFormatter())
ax2.set_ylim(15, 220)
ax2.annotate("Naive Transfer ~462\n(off-scale)", xy=(100, 100), fontsize=8,
             color="#DD8452", ha="center")
ax2.legend(fontsize=9, loc="upper right")
ax2.grid(True, which="major", linestyle="--", linewidth=0.5, alpha=0.6)
ax2.grid(True, which="minor", linestyle=":", linewidth=0.3, alpha=0.4)

fig.suptitle("RQ1: Speed-Quality Tradeoff", fontsize=13, fontweight="bold", y=1.01)
fig.tight_layout()

out = "ablation_fid_vs_nfe.pdf"
fig.savefig(out, dpi=300, bbox_inches="tight")
print(f"Saved {out}")
out_png = out.replace(".pdf", ".png")
fig.savefig(out_png, dpi=150, bbox_inches="tight")
print(f"Saved {out_png}")
