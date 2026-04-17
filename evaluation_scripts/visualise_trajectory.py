"""
Trajectory Visualisation: DDPM vs Flow Matching vs Diff2Flow
=============================================================
Runs N samples from the **same initial noises** with all three models and
produces:
  1. trajectory_strips.png      – noisy x_t snapshots for 1 example sample
  2. x0_prediction_strips.png   – x₀ estimate snapshots (how fast content resolves)
  3. trajectory_pca_bundle.png  – bundled PCA trajectories (N paths per model)
  4. straightness_boxplot.png   – straightness distribution per model
  5. step_magnitude.png         – per-step ||Δx|| over time (mean ± std)
  6. curvature.png              – per-step cosine similarity (mean ± std)
  7. metrics.txt                – numerical summary

Usage:
    python visualise_trajectory.py \\
        --ddpm_model ddpm_cifar10/final_model \\
        --flow_model flow_matching_cifar10/final_model \\
        --diff2flow_model diff2flow_cifar10/final_model
    python visualise_trajectory.py --num_samples 50 --num_steps 50
"""

import argparse
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from diffusers import DDIMScheduler, DDPMScheduler, UNet2DModel
from PIL import Image
from sklearn.decomposition import PCA
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from shared.aligner import Diff2FlowAligner

# ──────────────────────────────────────────────────────────────
# Config defaults
# ──────────────────────────────────────────────────────────────
DEFAULTS = dict(
    ddpm_model="ddpm_cifar10/final_model",
    flow_model="flow_matching_cifar10/final_model",
    diff2flow_model="diff2flow_cifar10/final_model",
    output_dir="trajectory_vis",
    image_size=32,
    num_steps=50,
    num_snapshots=10,
    num_samples=50,
    num_train_timesteps=1000,
    seed=42,
)


# ──────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────
def _to_pil(x: torch.Tensor) -> Image.Image:
    """(C,H,W) tensor in [-1,1] → PIL Image."""
    x = x.clamp(-1, 1).float()
    x = ((x + 1) / 2 * 255).to(torch.uint8).permute(1, 2, 0).cpu().numpy()
    return Image.fromarray(x)


def compute_straightness(traj: list) -> float:
    """straight-line distance / total arc length  (1.0 = perfectly straight)."""
    vecs = [t.cpu().numpy().flatten() for t in traj]
    straight = np.linalg.norm(vecs[-1] - vecs[0])
    arc = sum(np.linalg.norm(vecs[i + 1] - vecs[i]) for i in range(len(vecs) - 1))
    return float(straight / arc) if arc > 1e-10 else 1.0


def compute_step_magnitudes(traj: list) -> np.ndarray:
    """||x_{t+1} - x_t||₂ at each step  →  shape (num_steps,)."""
    mags = []
    for i in range(len(traj) - 1):
        delta = traj[i + 1].numpy().flatten() - traj[i].numpy().flatten()
        mags.append(np.linalg.norm(delta))
    return np.array(mags)


def compute_curvature(traj: list) -> np.ndarray:
    """
    Cosine similarity between consecutive step vectors  →  shape (num_steps-1,).
    1.0 = perfectly straight, -1.0 = U-turn.
    """
    cosines = []
    for i in range(len(traj) - 2):
        d1 = traj[i + 1].numpy().flatten() - traj[i].numpy().flatten()
        d2 = traj[i + 2].numpy().flatten() - traj[i + 1].numpy().flatten()
        n1, n2 = np.linalg.norm(d1), np.linalg.norm(d2)
        if n1 < 1e-10 or n2 < 1e-10:
            cosines.append(1.0)
        else:
            cosines.append(float(np.dot(d1, d2) / (n1 * n2)))
    return np.array(cosines)


# ──────────────────────────────────────────────────────────────
# Trajectory functions — return (traj, x0_traj)
# traj     : list of (C,H,W) CPU tensors, length num_steps+1
# x0_traj  : list of (C,H,W) CPU tensors, length num_steps
#             (x₀ estimate after each model call)
# ──────────────────────────────────────────────────────────────
@torch.no_grad()
def ddpm_trajectory(model, scheduler: DDIMScheduler, z: torch.Tensor,
                    num_steps: int, device):
    scheduler.set_timesteps(num_steps, device=device)
    x = z.clone()
    traj = [x.squeeze(0).cpu()]
    x0_traj = []
    for t in scheduler.timesteps:
        noise_pred = model(x, t, return_dict=False)[0]
        out = scheduler.step(noise_pred, t, x, return_dict=True)
        x = out.prev_sample
        traj.append(x.squeeze(0).cpu())
        x0_traj.append(out.pred_original_sample.squeeze(0).cpu())
    return traj, x0_traj


@torch.no_grad()
def flow_matching_trajectory(model, z: torch.Tensor, num_steps: int, device):
    x = z.clone()
    dt = 1.0 / num_steps
    traj = [x.squeeze(0).cpu()]
    x0_traj = []
    for i in range(num_steps):
        t_fm = i / num_steps
        t_scaled = torch.full((1,), t_fm * 999.0, device=device)
        v = model(x, t_scaled, return_dict=False)[0]
        # x₀ estimate: extrapolate from t to t=1 along velocity
        x0_hat = x + (1.0 - t_fm) * v
        x0_traj.append(x0_hat.squeeze(0).cpu())
        x = x + v * dt
        traj.append(x.squeeze(0).cpu())
    return traj, x0_traj


@torch.no_grad()
def diff2flow_trajectory(model, aligner: Diff2FlowAligner, z: torch.Tensor,
                         num_steps: int, device):
    x = z.clone()
    dt = 1.0 / num_steps
    traj = [x.squeeze(0).cpu()]
    x0_traj = []
    for i in range(num_steps):
        t_fm = torch.full((1,), i * dt, device=device)
        t_dm = aligner.t_fm_to_t_dm(t_fm)
        alpha_t, sigma_t = aligner.get_alpha_sigma(t_dm)
        x_dm = aligner.x_fm_to_x_dm(x, alpha_t, sigma_t)
        t_dm_input = t_dm.long().clamp(0, aligner.T - 1)
        eps_pred = model(x_dm, t_dm_input, return_dict=False)[0]
        # x₀ estimate: denoise from DM space
        a = alpha_t.view(-1, 1, 1, 1)
        s = sigma_t.view(-1, 1, 1, 1)
        x0_hat = (x_dm - s * eps_pred) / a.clamp(min=1e-8)
        x0_traj.append(x0_hat.squeeze(0).cpu())
        velocity = aligner.eps_to_velocity(eps_pred, x_dm, alpha_t, sigma_t)
        x = x + dt * velocity
        traj.append(x.squeeze(0).cpu())
    return traj, x0_traj


# ──────────────────────────────────────────────────────────────
# Model loaders
# ──────────────────────────────────────────────────────────────
def load_ddpm(model_path: str, device, num_train_timesteps: int = 1000):
    model = UNet2DModel.from_pretrained(model_path).to(device).eval()
    try:
        ddpm_scheduler = DDPMScheduler.from_pretrained(model_path)
    except OSError:
        ddpm_scheduler = DDPMScheduler(num_train_timesteps=num_train_timesteps)
    scheduler = DDIMScheduler.from_config(ddpm_scheduler.config)
    return model, scheduler


def load_flow_matching(model_path: str, device):
    return UNet2DModel.from_pretrained(model_path).to(device).eval()


def load_diff2flow(model_path: str, num_train_timesteps: int, device):
    model = UNet2DModel.from_pretrained(model_path).to(device).eval()
    noise_scheduler = DDPMScheduler(num_train_timesteps=num_train_timesteps)
    aligner = Diff2FlowAligner(noise_scheduler)
    return model, aligner


# ──────────────────────────────────────────────────────────────
# Plotting helpers
# ──────────────────────────────────────────────────────────────
def _select_snapshots(traj: list, n: int):
    indices = np.linspace(0, len(traj) - 1, n, dtype=int)
    return [traj[i] for i in indices], indices


def _plot_mean_std(ax, arrays, label, colour, x=None):
    """Plot mean ± 1 std across a list of 1-D arrays of the same length."""
    arr = np.stack(arrays, axis=0)
    mean, std = arr.mean(0), arr.std(0)
    if x is None:
        x = np.arange(len(mean))
    ax.plot(x, mean, color=colour, linewidth=2, label=label)
    ax.fill_between(x, mean - std, mean + std, color=colour, alpha=0.2)


# ──────────────────────────────────────────────────────────────
# Plot 1: noisy trajectory strip
# ──────────────────────────────────────────────────────────────
def plot_strips(single_trajs: dict, labels: list, num_snapshots: int, save_path: str):
    n_rows = len(labels)
    fig, axes = plt.subplots(n_rows, num_snapshots,
                             figsize=(num_snapshots * 1.5, n_rows * 1.8), dpi=150)
    if n_rows == 1:
        axes = [axes]
    for row, label in enumerate(labels):
        snaps, snap_idx = _select_snapshots(single_trajs[label], num_snapshots)
        total = len(single_trajs[label]) - 1
        for col, (img_t, idx) in enumerate(zip(snaps, snap_idx)):
            ax = axes[row][col]
            ax.imshow(_to_pil(img_t))
            ax.set_xticks([]); ax.set_yticks([])
            if col == 0:
                ax.set_ylabel(label, fontsize=10, fontweight="bold",
                              rotation=0, labelpad=60, va="center")
            if row == 0:
                frac = idx / total if total > 0 else 0
                ax.set_title(f"t={frac:.2f}", fontsize=8)
    plt.suptitle("Sampling Trajectories  (noise → image)", fontsize=13,
                 fontweight="bold", y=1.02)
    plt.tight_layout()
    plt.savefig(save_path, bbox_inches="tight", pad_inches=0.1)
    plt.close()
    print(f"  Saved strip            → {save_path}")


# ──────────────────────────────────────────────────────────────
# Plot 2: x₀ prediction strip
# ──────────────────────────────────────────────────────────────
def plot_x0_strips(single_x0_trajs: dict, labels: list,
                   num_snapshots: int, save_path: str):
    """Show how quickly each model's x₀ estimate resolves into a clean image."""
    n_rows = len(labels)
    # x0_traj has num_steps entries; snapshots are selected from these
    fig, axes = plt.subplots(n_rows, num_snapshots,
                             figsize=(num_snapshots * 1.5, n_rows * 1.8), dpi=150)
    if n_rows == 1:
        axes = [axes]
    for row, label in enumerate(labels):
        snaps, snap_idx = _select_snapshots(single_x0_trajs[label], num_snapshots)
        total = len(single_x0_trajs[label]) - 1
        for col, (img_t, idx) in enumerate(zip(snaps, snap_idx)):
            ax = axes[row][col]
            ax.imshow(_to_pil(img_t))
            ax.set_xticks([]); ax.set_yticks([])
            if col == 0:
                ax.set_ylabel(label, fontsize=10, fontweight="bold",
                              rotation=0, labelpad=60, va="center")
            if row == 0:
                frac = idx / total if total > 0 else 0
                ax.set_title(f"t={frac:.2f}", fontsize=8)
    plt.suptitle("x₀ Prediction Evolution  (how fast content resolves)", fontsize=13,
                 fontweight="bold", y=1.02)
    plt.tight_layout()
    plt.savefig(save_path, bbox_inches="tight", pad_inches=0.1)
    plt.close()
    print(f"  Saved x₀ strip         → {save_path}")


# ──────────────────────────────────────────────────────────────
# Plot 3: PCA bundle
# ──────────────────────────────────────────────────────────────
def plot_pca_bundle(all_trajs: dict, labels: list, save_path: str):
    flat_vecs = []
    for label in labels:
        for traj in all_trajs[label]:
            for wp in traj:
                flat_vecs.append(wp.numpy().flatten())

    X2 = PCA(n_components=2).fit_transform(np.stack(flat_vecs, axis=0))
    pca_obj = PCA(n_components=2).fit(np.stack(flat_vecs, axis=0))
    X2 = pca_obj.transform(np.stack(flat_vecs, axis=0))

    proj = {label: [] for label in labels}
    ptr = 0
    for label in labels:
        for traj in all_trajs[label]:
            n = len(traj)
            proj[label].append(X2[ptr: ptr + n])
            ptr += n

    cmap = matplotlib.colormaps["tab10"]
    colours = {label: cmap(i) for i, label in enumerate(labels)}

    fig, ax = plt.subplots(figsize=(8, 7), dpi=150)
    for label in labels:
        c = colours[label]
        for i, pts in enumerate(proj[label]):
            ax.plot(pts[:, 0], pts[:, 1], "-", color=c,
                    alpha=0.15, linewidth=0.8, label=label if i == 0 else None)
            ax.scatter(pts[0, 0], pts[0, 1], marker="o", s=20, color=c,
                       edgecolors="black", linewidths=0.3, alpha=0.35, zorder=5)
            ax.scatter(pts[-1, 0], pts[-1, 1], marker="*", s=45, color=c,
                       edgecolors="black", linewidths=0.3, alpha=0.35, zorder=5)
        min_len = min(len(t) for t in proj[label])
        mean_traj = np.stack([t[:min_len] for t in proj[label]], axis=0).mean(0)
        ax.plot(mean_traj[:, 0], mean_traj[:, 1], "-", color=c,
                alpha=0.9, linewidth=2.5, zorder=10)

    ax.legend(fontsize=10)
    ax.set_xlabel(f"PC 1 ({pca_obj.explained_variance_ratio_[0]*100:.1f}%)", fontsize=11)
    ax.set_ylabel(f"PC 2 ({pca_obj.explained_variance_ratio_[1]*100:.1f}%)", fontsize=11)
    ax.set_title(
        f"Trajectory Bundles in PCA Space ({len(all_trajs[labels[0]])} samples/model)\n"
        "thin = individual, thick = mean   |   ○ = noise, ★ = image",
        fontsize=11, fontweight="bold",
    )
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, bbox_inches="tight")
    plt.close()
    print(f"  Saved PCA bundle       → {save_path}")


# ──────────────────────────────────────────────────────────────
# Plot 4: straightness boxplot
# ──────────────────────────────────────────────────────────────
def plot_straightness(straightness: dict, labels: list, save_path: str):
    cmap = matplotlib.colormaps["tab10"]
    colours = [cmap(i) for i in range(len(labels))]
    fig, ax = plt.subplots(figsize=(6, 5), dpi=150)
    data = [straightness[label] for label in labels]
    bp = ax.boxplot(data, tick_labels=labels, patch_artist=True, widths=0.5,
                    showmeans=True, meanline=True,
                    meanprops=dict(color="black", linewidth=1.5),
                    medianprops=dict(color="black", linewidth=1.5))
    for patch, colour in zip(bp["boxes"], colours):
        patch.set_facecolor((*colour[:3], 0.4))
        patch.set_edgecolor(colour)
    rng = np.random.default_rng(42)
    for i, (label, vals) in enumerate(zip(labels, data)):
        jitter = rng.normal(0, 0.04, len(vals)) + (i + 1)
        ax.scatter(jitter, vals, color=colours[i], alpha=0.4, s=15, zorder=5)
    ax.set_ylabel("Straightness  (higher = straighter)", fontsize=11)
    ax.set_title("Trajectory Straightness Distribution", fontsize=12, fontweight="bold")
    ax.set_ylim(0, 1.05)
    ax.grid(True, axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, bbox_inches="tight")
    plt.close()
    print(f"  Saved straightness     → {save_path}")


# ──────────────────────────────────────────────────────────────
# Plot 5: per-step velocity magnitude
# ──────────────────────────────────────────────────────────────
def plot_step_magnitudes(all_trajs: dict, labels: list, save_path: str):
    """
    ||x_{t+1} - x_t||₂ at each step.
    Shows WHERE along the trajectory each model does its work.
    DDPM concentrates effort early; FM spreads it uniformly;
    Diff2Flow should be intermediate-to-uniform.
    """
    cmap = matplotlib.colormaps["tab10"]
    colours = {label: cmap(i) for i, label in enumerate(labels)}

    fig, ax = plt.subplots(figsize=(8, 5), dpi=150)
    for label in labels:
        mags_list = [compute_step_magnitudes(traj) for traj in all_trajs[label]]
        # normalise x-axis to [0, 1] regardless of num_steps
        n_steps = len(mags_list[0])
        x = np.linspace(0, 1, n_steps)
        _plot_mean_std(ax, mags_list, label, colours[label], x=x)

    ax.set_xlabel("Normalised step  (0 = noise, 1 = image)", fontsize=11)
    ax.set_ylabel("Step magnitude  ||Δx||₂", fontsize=11)
    ax.set_title("Per-Step Velocity Magnitude  (mean ± 1 std)",
                 fontsize=12, fontweight="bold")
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, bbox_inches="tight")
    plt.close()
    print(f"  Saved step magnitude   → {save_path}")


# ──────────────────────────────────────────────────────────────
# Plot 6: per-step curvature
# ──────────────────────────────────────────────────────────────
def plot_curvature(all_trajs: dict, labels: list, save_path: str):
    """
    Cosine similarity between consecutive step vectors.
    1.0 = perfectly straight, lower = more curved at that step.
    Reveals WHERE trajectories bend, not just overall straightness.
    """
    cmap = matplotlib.colormaps["tab10"]
    colours = {label: cmap(i) for i, label in enumerate(labels)}

    fig, ax = plt.subplots(figsize=(8, 5), dpi=150)
    for label in labels:
        curv_list = [compute_curvature(traj) for traj in all_trajs[label]]
        n_steps = len(curv_list[0])
        # curvature has num_steps-1 values; centre on step midpoints
        x = np.linspace(0, 1, n_steps)
        _plot_mean_std(ax, curv_list, label, colours[label], x=x)

    ax.axhline(1.0, color="gray", linewidth=0.8, linestyle="--", alpha=0.6)
    ax.set_xlabel("Normalised step  (0 = noise, 1 = image)", fontsize=11)
    ax.set_ylabel("cos(Δx_t, Δx_{t+1})  →  1 = straight", fontsize=11)
    ax.set_title("Trajectory Curvature Profile  (mean ± 1 std)",
                 fontsize=12, fontweight="bold")
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, bbox_inches="tight")
    plt.close()
    print(f"  Saved curvature        → {save_path}")


# ──────────────────────────────────────────────────────────────
# Metrics
# ──────────────────────────────────────────────────────────────
def save_metrics(straightness: dict, labels: list, save_path: str):
    lines = ["Trajectory Straightness (straight-line / arc-length)", "=" * 55]
    for label in labels:
        vals = straightness[label]
        lines.append(f"  {label:15s}:  {np.mean(vals):.4f} ± {np.std(vals):.4f}  (n={len(vals)})")
    text = "\n".join(lines) + "\n"
    with open(save_path, "w") as f:
        f.write(text)
    print(f"  Saved metrics          → {save_path}")
    print(text)


# ──────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────
def parse_args():
    p = argparse.ArgumentParser(description="Trajectory visualisation")
    p.add_argument("--ddpm_model",           type=str, default=DEFAULTS["ddpm_model"])
    p.add_argument("--flow_model",           type=str, default=DEFAULTS["flow_model"])
    p.add_argument("--diff2flow_model",      type=str, default=DEFAULTS["diff2flow_model"])
    p.add_argument("--output_dir",           type=str, default=DEFAULTS["output_dir"])
    p.add_argument("--num_steps",            type=int, default=DEFAULTS["num_steps"])
    p.add_argument("--num_snapshots",        type=int, default=DEFAULTS["num_snapshots"])
    p.add_argument("--num_samples",          type=int, default=DEFAULTS["num_samples"])
    p.add_argument("--num_train_timesteps",  type=int, default=DEFAULTS["num_train_timesteps"])
    p.add_argument("--image_size",           type=int, default=DEFAULTS["image_size"])
    p.add_argument("--seed",                 type=int, default=DEFAULTS["seed"])
    p.add_argument("--skip_ddpm",            action="store_true")
    p.add_argument("--skip_flow",            action="store_true")
    p.add_argument("--skip_diff2flow",       action="store_true")
    return p.parse_args()


def _run_model(name, fn, all_z, desc):
    """Collect (traj, x0_traj) pairs for all starting noises."""
    trajs, x0_trajs = [], []
    for z in tqdm(all_z, desc=f"  {desc}"):
        traj, x0_traj = fn(z)
        trajs.append(traj)
        x0_trajs.append(x0_traj)
    return trajs, x0_trajs


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    print(f"Samples per model: {args.num_samples}  |  Steps: {args.num_steps}")

    torch.manual_seed(args.seed)
    all_z = [
        torch.randn(1, 3, args.image_size, args.image_size, device=device)
        for _ in range(args.num_samples)
    ]

    all_trajs: dict = {}
    single_trajs: dict = {}
    single_x0_trajs: dict = {}
    straightness: dict = {}
    labels: list = []

    # ── DDPM ──
    if not args.skip_ddpm:
        if os.path.isdir(args.ddpm_model):
            print(f"\n[1/3] DDPM (DDIM, {args.num_steps} steps) …")
            model, scheduler = load_ddpm(args.ddpm_model, device, args.num_train_timesteps)
            trajs, x0_trajs = _run_model(
                "DDPM",
                lambda z: ddpm_trajectory(model, scheduler, z, args.num_steps, device),
                all_z, "DDPM",
            )
            all_trajs["DDPM"] = trajs
            single_trajs["DDPM"] = trajs[0]
            single_x0_trajs["DDPM"] = x0_trajs[0]
            straightness["DDPM"] = [compute_straightness(t) for t in trajs]
            labels.append("DDPM")
            del model
            if device.type == "cuda":
                torch.cuda.empty_cache()
        else:
            print(f"[!] DDPM model not found at '{args.ddpm_model}', skipping.")

    # ── Flow Matching ──
    if not args.skip_flow:
        if os.path.isdir(args.flow_model):
            print(f"\n[2/3] Flow Matching (Euler, {args.num_steps} steps) …")
            model = load_flow_matching(args.flow_model, device)
            trajs, x0_trajs = _run_model(
                "Flow",
                lambda z: flow_matching_trajectory(model, z, args.num_steps, device),
                all_z, "Flow",
            )
            all_trajs["Flow Matching"] = trajs
            single_trajs["Flow Matching"] = trajs[0]
            single_x0_trajs["Flow Matching"] = x0_trajs[0]
            straightness["Flow Matching"] = [compute_straightness(t) for t in trajs]
            labels.append("Flow Matching")
            del model
            if device.type == "cuda":
                torch.cuda.empty_cache()
        else:
            print(f"[!] Flow model not found at '{args.flow_model}', skipping.")

    # ── Diff2Flow ──
    if not args.skip_diff2flow:
        if os.path.isdir(args.diff2flow_model):
            print(f"\n[3/3] Diff2Flow (Euler via aligner, {args.num_steps} steps) …")
            model, aligner = load_diff2flow(
                args.diff2flow_model, args.num_train_timesteps, device
            )
            trajs, x0_trajs = _run_model(
                "Diff2Flow",
                lambda z: diff2flow_trajectory(model, aligner, z, args.num_steps, device),
                all_z, "Diff2Flow",
            )
            all_trajs["Diff2Flow"] = trajs
            single_trajs["Diff2Flow"] = trajs[0]
            single_x0_trajs["Diff2Flow"] = x0_trajs[0]
            straightness["Diff2Flow"] = [compute_straightness(t) for t in trajs]
            labels.append("Diff2Flow")
            del model
            if device.type == "cuda":
                torch.cuda.empty_cache()
        else:
            print(f"[!] Diff2Flow model not found at '{args.diff2flow_model}', skipping.")

    if not labels:
        print("\nNo models found — nothing to visualise.")
        return

    print("\nGenerating plots …")
    out = args.output_dir

    plot_strips(single_trajs, labels, args.num_snapshots,
                os.path.join(out, "trajectory_strips.png"))

    plot_x0_strips(single_x0_trajs, labels, args.num_snapshots,
                   os.path.join(out, "x0_prediction_strips.png"))

    plot_pca_bundle(all_trajs, labels,
                    os.path.join(out, "trajectory_pca_bundle.png"))

    plot_straightness(straightness, labels,
                      os.path.join(out, "straightness_boxplot.png"))

    plot_step_magnitudes(all_trajs, labels,
                         os.path.join(out, "step_magnitude.png"))

    plot_curvature(all_trajs, labels,
                   os.path.join(out, "curvature.png"))

    save_metrics(straightness, labels,
                 os.path.join(out, "metrics.txt"))

    print("\nDone.")


if __name__ == "__main__":
    main()
