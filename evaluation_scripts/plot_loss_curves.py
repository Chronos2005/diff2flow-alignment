"""Parse SLURM training logs and plot per-epoch loss curves for all experiments."""

import re
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from pathlib import Path

LOG_DIR = Path(__file__).parent.parent / "experiments" / "logs"
OUT_DIR = Path(__file__).parent.parent / "results"
OUT_DIR.mkdir(exist_ok=True)

LOSS_RE = re.compile(r"Epoch (\d+) avg loss: ([0-9.]+)")


def parse_losses(path):
    losses = {}
    with open(path) as f:
        for line in f:
            m = LOSS_RE.search(line)
            if m:
                losses[int(m.group(1))] = float(m.group(2))
    return [losses[k] for k in sorted(losses)]


def parse_exp1():
    files = {
        1: ("T",   "T only\n(no I, V)"),
        2: ("I",   "I only\n(no T, V)"),
        3: ("V",   "V only\n(no T, I)"),
        4: ("TI",  "T+I\n(no V)"),
        5: ("TV",  "T+V\n(no I)"),
        6: ("IV",  "I+V\n(no T)"),
        7: ("TIV", "T+I+V\n(full)"),
    }
    data = {}
    for task_id, (key, label) in files.items():
        p = LOG_DIR / f"exp1_train_853379_{task_id}.out"
        if p.exists():
            data[label] = parse_losses(p)
    return data


def parse_exp2():
    labels = {
        1: "rank=4",
        2: "rank=8",
        3: "rank=16",
        4: "rank=32",
        5: "rank=64",
        6: "rank=128",
        7: "full FT",
    }
    data = {}
    for task_id, label in labels.items():
        p = LOG_DIR / f"exp2_train_853381_{task_id}.out"
        if p.exists():
            data[label] = parse_losses(p)
    return data


def parse_exp5():
    # Multiple placements may be in one file; parse per-placement sections
    placements_order = ["all", "attention", "feedforward"]
    result = {pl: [] for pl in placements_order}
    current = None
    for task_id in range(1, 4):
        p = LOG_DIR / f"exp5_train_853384_{task_id}.out"
        if not p.exists():
            continue
        with open(p) as f:
            for line in f:
                pm = re.search(r"Placement:\s*(\w+)", line)
                if pm:
                    current = pm.group(1)
                lm = LOSS_RE.search(line)
                if lm and current in result:
                    result[current].append(float(lm.group(2)))
    return {k: v for k, v in result.items() if v}


def parse_exp8():
    p = LOG_DIR / "exp8_train_877524.out"
    return parse_losses(p) if p.exists() else []


def plot_experiment(ax, data_dict, title, xlabel="Epoch", ylabel="Avg MSE Loss",
                    highlight=None):
    for label, losses in data_dict.items():
        epochs = list(range(1, len(losses) + 1))
        lw = 2.2 if label == highlight else 1.4
        ls = "-" if label == highlight else "--"
        ax.plot(epochs, losses, label=label, linewidth=lw, linestyle=ls)
    ax.set_title(title, fontsize=11, fontweight="bold")
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.legend(fontsize=7, ncol=2)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(left=1)


def main():
    exp1 = parse_exp1()
    exp2 = parse_exp2()
    exp5 = parse_exp5()
    exp8 = parse_exp8()

    fig = plt.figure(figsize=(16, 10))
    gs = gridspec.GridSpec(2, 3, figure=fig, hspace=0.45, wspace=0.35)

    # Exp1 — alignment ablation (all variants together)
    ax1 = fig.add_subplot(gs[0, 0])
    plot_experiment(ax1, exp1, "Exp 1 – Alignment Component Ablation",
                    highlight="T+I+V\n(full)")

    # Exp1 — highlight full vs ablated (split into two groups for clarity)
    ax1b = fig.add_subplot(gs[0, 1])
    good = {k: v for k, v in exp1.items() if "V" in k}
    bad  = {k: v for k, v in exp1.items() if "V" not in k}
    plot_experiment(ax1b, {**good, **bad}, "Exp 1 – With vs Without Velocity (V)",
                    highlight="T+I+V\n(full)")

    # Exp2 — LoRA rank sweep
    ax2 = fig.add_subplot(gs[0, 2])
    plot_experiment(ax2, exp2, "Exp 2 – LoRA Rank Sweep", highlight="full FT")

    # Exp5 — LoRA placement
    ax5 = fig.add_subplot(gs[1, 0])
    plot_experiment(ax5, exp5, "Exp 5 – LoRA Placement", highlight="all")

    # Exp8 — 200-epoch run
    ax8 = fig.add_subplot(gs[1, 1:])
    if exp8:
        epochs = list(range(1, len(exp8) + 1))
        ax8.plot(epochs, exp8, linewidth=1.6, color="steelblue")
        ax8.set_title("Exp 8 – LoRA rank=32, 200 Epochs", fontsize=11, fontweight="bold")
        ax8.set_xlabel("Epoch")
        ax8.set_ylabel("Avg MSE Loss")
        ax8.grid(True, alpha=0.3)
        ax8.set_xlim(left=1)

    fig.suptitle("Training Loss Curves – Diff2Flow Experiments", fontsize=13, fontweight="bold")

    out_path = OUT_DIR / "loss_curves.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"Saved → {out_path}")

    # Also save per-experiment PNGs
    for name, data_dict, title, hl in [
        ("loss_exp1_alignment", exp1, "Exp 1 – Alignment Component Ablation", "T+I+V\n(full)"),
        ("loss_exp2_lora_rank", exp2, "Exp 2 – LoRA Rank Sweep", "full FT"),
        ("loss_exp5_placement", exp5, "Exp 5 – LoRA Placement", "all"),
    ]:
        if not data_dict:
            continue
        fig2, ax = plt.subplots(figsize=(7, 4))
        plot_experiment(ax, data_dict, title, highlight=hl)
        fig2.tight_layout()
        p = OUT_DIR / f"{name}.png"
        fig2.savefig(p, dpi=150, bbox_inches="tight")
        plt.close(fig2)
        print(f"Saved → {p}")

    if exp8:
        fig3, ax = plt.subplots(figsize=(9, 4))
        ax.plot(range(1, len(exp8) + 1), exp8, linewidth=1.6, color="steelblue")
        ax.set_title("Exp 8 – LoRA rank=32, 200 Epochs")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Avg MSE Loss")
        ax.grid(True, alpha=0.3)
        fig3.tight_layout()
        p = OUT_DIR / "loss_exp8_200epochs.png"
        fig3.savefig(p, dpi=150, bbox_inches="tight")
        plt.close(fig3)
        print(f"Saved → {p}")

    plt.show()


if __name__ == "__main__":
    main()
