"""
Trajectory Visualisation: DDPM vs Flow Matching vs Diff2Flow
=============================================================
Runs N samples from the **same initial noises** with all three models and
produces:
  1. trajectory_strips.png    – image snapshots for 1 example sample
  2. trajectory_pca_bundle.png – bundled PCA trajectories (N paths per model)
  3. straightness_boxplot.png  – straightness distribution per model
  4. metrics.txt               – numerical summary (mean ± std)

Usage:
    python visualise_trajectory.py
    python visualise_trajectory.py --num-samples 50 --num-steps 50
    python visualise_trajectory.py --output-dir my_vis
"""

import os
import sys
import argparse
import numpy as np
import torch
import torch.nn as nn
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image
from tqdm import tqdm
from diffusers import UNet2DModel, DDPMScheduler
from sklearn.decomposition import PCA

# ---------- project imports ----------
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from flow_obj import FlowModelObj

# ──────────────────────────────────────────────────────────────
# Config defaults
# ──────────────────────────────────────────────────────────────
DEFAULTS = dict(
    ddpm_model="ddpm_cifar10/final_model",
    flow_model="flow_matching_cifar10/final_model",
    diff2flow_model="diff2flow_cifar10/diff2flow_flowmodel.pt",
    ddpm_unet_for_diff2flow="ddpm_cifar10/final_model",
    output_dir="trajectory_vis",
    image_size=32,
    num_steps=50,        # sampling steps per model
    num_snapshots=10,    # number of intermediate frames in strip
    num_samples=50,      # number of trajectories for PCA bundle
    seed=42,
)


# ──────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────
def build_schedule(num_timesteps: int):
    """DDPM linear beta schedule."""
    betas = np.linspace(0.0001, 0.02, num_timesteps, dtype=np.float64)
    alphas = 1.0 - betas
    alphas_cumprod = np.cumprod(alphas, axis=0)
    return {
        "betas": betas,
        "alphas_cumprod": alphas_cumprod,
    }


def tensor_to_pil(x):
    """(C,H,W) tensor in [-1,1] → PIL Image."""
    x = x.clamp(-1, 1).float()
    x = (x + 1) / 2
    x = (x * 255).to(torch.uint8).permute(1, 2, 0).cpu().numpy()
    return Image.fromarray(x)


def compute_straightness(trajectory):
    """
    Straightness = straight-line distance / total arc length.
    trajectory: list of (C,H,W) tensors.
    Returns a float in (0, 1]. 1.0 = perfectly straight.
    """
    vecs = [t.numpy().flatten() for t in trajectory]
    # Straight-line distance: first → last
    straight = np.linalg.norm(vecs[-1] - vecs[0])
    # Arc length: sum of consecutive distances
    arc = sum(np.linalg.norm(vecs[i+1] - vecs[i]) for i in range(len(vecs) - 1))
    if arc < 1e-10:
        return 1.0
    return float(straight / arc)


class DiffusersUNetWrapper(nn.Module):
    """Thin wrapper so FlowModelObj can call the UNet directly."""
    def __init__(self, unet):
        super().__init__()
        self.unet = unet
    def forward(self, x, t, **kwargs):
        return self.unet(x, t, return_dict=False)[0]


# ──────────────────────────────────────────────────────────────
# 1) DDPM trajectory  (DDIM deterministic sampling)
# ──────────────────────────────────────────────────────────────
@torch.no_grad()
def ddpm_trajectory(model, z, num_steps, schedule, device):
    """Return list of (C,H,W) tensors at each DDIM step."""
    T = len(schedule["betas"])
    timesteps = np.linspace(0, T - 1, num_steps + 1, dtype=int)[::-1]

    x = z.clone()
    trajectory = [x.squeeze(0).cpu()]

    for i in range(len(timesteps) - 1):
        t_from = torch.full((1,), timesteps[i], device=device, dtype=torch.long)
        t_to_val = timesteps[i + 1]
        if i == len(timesteps) - 2:
            t_to = torch.full((1,), -1, device=device, dtype=torch.long)
        else:
            t_to = torch.full((1,), t_to_val, device=device, dtype=torch.long)

        eps_pred = model(x, t_from, return_dict=False)[0]
        alpha_bar_from = torch.tensor(
            schedule["alphas_cumprod"], device=device, dtype=x.dtype
        )[t_from].view(-1, 1, 1, 1)
        sqrt_ab_from = torch.sqrt(alpha_bar_from)
        sqrt_1_ab_from = torch.sqrt(1.0 - alpha_bar_from)
        x_0_pred = (x - sqrt_1_ab_from * eps_pred) / sqrt_ab_from

        if t_to.min().item() < 0:
            x = x_0_pred
        else:
            alpha_bar_to = torch.tensor(
                schedule["alphas_cumprod"], device=device, dtype=x.dtype
            )[t_to].view(-1, 1, 1, 1)
            sqrt_ab_to = torch.sqrt(alpha_bar_to)
            sqrt_1_ab_to = torch.sqrt(1.0 - alpha_bar_to)
            x = sqrt_ab_to * x_0_pred + sqrt_1_ab_to * eps_pred

        trajectory.append(x.squeeze(0).cpu())

    return trajectory


# ──────────────────────────────────────────────────────────────
# 2) Flow Matching trajectory  (Euler on velocity field)
# ──────────────────────────────────────────────────────────────
@torch.no_grad()
def flow_matching_trajectory(model, z, num_steps, device):
    """Return list of (C,H,W) tensors at each Euler step."""
    x = z.clone()
    dt = 1.0 / num_steps
    trajectory = [x.squeeze(0).cpu()]

    for i in range(num_steps):
        t_scaled = (i / num_steps) * 999
        t_tensor = torch.full((1,), t_scaled, device=device)
        v = model(x, t_tensor, return_dict=False)[0]
        x = x + v * dt
        trajectory.append(x.squeeze(0).cpu())

    return trajectory


# ──────────────────────────────────────────────────────────────
# 3) Diff2Flow trajectory  (ODE via FlowModelObj)
# ──────────────────────────────────────────────────────────────
@torch.no_grad()
def diff2flow_trajectory(flow_model, z, num_steps, device):
    """Return list of (C,H,W) tensors at each Euler step."""
    x = z.clone()
    dt = 1.0 / num_steps
    trajectory = [x.squeeze(0).cpu()]

    for i in range(num_steps):
        t = torch.full((1,), i * dt, device=device)
        v = flow_model.ode_fn(t, x)
        x = x + dt * v
        trajectory.append(x.squeeze(0).cpu())

    return trajectory


# ──────────────────────────────────────────────────────────────
# Plotting
# ──────────────────────────────────────────────────────────────
def select_snapshots(trajectory, num_snapshots):
    """Pick evenly-spaced frames including first and last."""
    n = len(trajectory)
    indices = np.linspace(0, n - 1, num_snapshots, dtype=int)
    return [trajectory[i] for i in indices], indices


def plot_strips(trajs, labels, num_snapshots, save_path):
    """
    Image strip for a single sample per model.
    trajs  : dict  label → list of (C,H,W) tensors (one trajectory)
    """
    n_rows = len(labels)
    fig, axes = plt.subplots(
        n_rows, num_snapshots,
        figsize=(num_snapshots * 1.5, n_rows * 1.8),
        dpi=150,
    )
    if n_rows == 1:
        axes = [axes]

    for row, label in enumerate(labels):
        snaps, snap_idx = select_snapshots(trajs[label], num_snapshots)
        for col, (img_t, idx) in enumerate(zip(snaps, snap_idx)):
            ax = axes[row][col]
            pil = tensor_to_pil(img_t)
            ax.imshow(pil)
            ax.set_xticks([])
            ax.set_yticks([])
            if col == 0:
                ax.set_ylabel(label, fontsize=10, fontweight="bold", rotation=0,
                              labelpad=60, va="center")
            if row == 0:
                total = len(trajs[label]) - 1
                frac = idx / total if total > 0 else 0
                ax.set_title(f"t={frac:.2f}", fontsize=8)

    plt.suptitle("Sampling Trajectories (noise → image)", fontsize=13, fontweight="bold", y=1.02)
    plt.tight_layout()
    plt.savefig(save_path, bbox_inches="tight", pad_inches=0.1)
    plt.close()
    print(f"  Saved strip       → {save_path}")


def plot_pca_bundle(all_trajs, labels, save_path):
    """
    Bundled PCA: N trajectories per model, all projected into the same 2D space.
    all_trajs: dict  label → list of trajectories, each trajectory = list of (C,H,W) tensors
    """
    # Flatten every waypoint from every trajectory from every model
    all_flat = []
    # Keep track of structure: (label, sample_idx, waypoint_idx)
    structure = []
    for label in labels:
        for s_idx, traj in enumerate(all_trajs[label]):
            for w_idx, wp in enumerate(traj):
                all_flat.append(wp.numpy().flatten())
                structure.append((label, s_idx, w_idx))

    X = np.stack(all_flat, axis=0)
    pca = PCA(n_components=2)
    X2 = pca.fit_transform(X)

    # Rebuild projected trajectories
    # label → list of (N_waypoints, 2) arrays
    proj_trajs = {label: [] for label in labels}
    idx = 0
    for label in labels:
        for traj in all_trajs[label]:
            n_wp = len(traj)
            proj_trajs[label].append(X2[idx: idx + n_wp])
            idx += n_wp

    # Colours
    cmap = plt.cm.get_cmap("tab10")
    colours = {label: cmap(i) for i, label in enumerate(labels)}

    fig, ax = plt.subplots(figsize=(8, 7), dpi=150)

    for label in labels:
        colour = colours[label]
        trajectories_2d = proj_trajs[label]
        n = len(trajectories_2d)

        # Draw each trajectory as a thin, semi-transparent line
        for i, pts in enumerate(trajectories_2d):
            lbl = label if i == 0 else None  # legend only once
            ax.plot(pts[:, 0], pts[:, 1], "-", color=colour,
                    alpha=0.15, linewidth=0.8, label=lbl)

        # Draw start/end markers on all trajectories
        for pts in trajectories_2d:
            ax.scatter(pts[0, 0], pts[0, 1], marker="o", s=20, color=colour,
                       edgecolors="black", linewidths=0.3, alpha=0.35, zorder=5)
            ax.scatter(pts[-1, 0], pts[-1, 1], marker="*", s=45, color=colour,
                       edgecolors="black", linewidths=0.3, alpha=0.35, zorder=5)

        # Draw the MEAN trajectory as a thick line
        min_len = min(len(t) for t in trajectories_2d)
        stacked = np.stack([t[:min_len] for t in trajectories_2d], axis=0)
        mean_traj = stacked.mean(axis=0)
        ax.plot(mean_traj[:, 0], mean_traj[:, 1], "-", color=colour,
                alpha=0.9, linewidth=2.5, zorder=10)

    ax.legend(fontsize=10, loc="best")
    ax.set_xlabel(f"PC 1 ({pca.explained_variance_ratio_[0]*100:.1f}%)", fontsize=11)
    ax.set_ylabel(f"PC 2 ({pca.explained_variance_ratio_[1]*100:.1f}%)", fontsize=11)
    ax.set_title(
        f"Trajectory Bundles in PCA Space ({len(all_trajs[labels[0]])} samples/model)\n"
        f"thin = individual, thick = mean   |   ○ = noise, ★ = image",
        fontsize=11, fontweight="bold",
    )
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, bbox_inches="tight")
    plt.close()
    print(f"  Saved PCA bundle  → {save_path}")


def plot_straightness(straightness, labels, save_path):
    """
    Box + strip plot of straightness scores per model.
    straightness: dict  label → list of floats
    """
    cmap = plt.cm.get_cmap("tab10")
    colours = [cmap(i) for i in range(len(labels))]

    fig, ax = plt.subplots(figsize=(6, 5), dpi=150)

    data = [straightness[label] for label in labels]
    bp = ax.boxplot(data, labels=labels, patch_artist=True, widths=0.5,
                    showmeans=True, meanline=True,
                    meanprops=dict(color="black", linewidth=1.5),
                    medianprops=dict(color="black", linewidth=1.5))

    for patch, colour in zip(bp["boxes"], colours):
        patch.set_facecolor((*colour[:3], 0.4))
        patch.set_edgecolor(colour)

    # Overlay individual points (jittered)
    for i, (label, vals) in enumerate(zip(labels, data)):
        x_jitter = np.random.default_rng(42).normal(0, 0.04, len(vals)) + (i + 1)
        ax.scatter(x_jitter, vals, color=colours[i], alpha=0.4, s=15, zorder=5)

    ax.set_ylabel("Straightness  (higher = straighter)", fontsize=11)
    ax.set_title("Trajectory Straightness Distribution", fontsize=12, fontweight="bold")
    ax.set_ylim(0, 1.05)
    ax.grid(True, axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, bbox_inches="tight")
    plt.close()
    print(f"  Saved straightness → {save_path}")


def save_metrics(straightness, labels, save_path):
    """Save numerical summary to a text file."""
    lines = ["Trajectory Straightness (straight-line / arc-length)", "=" * 55]
    for label in labels:
        vals = straightness[label]
        mean = np.mean(vals)
        std = np.std(vals)
        lines.append(f"  {label:15s}:  {mean:.4f} ± {std:.4f}  (n={len(vals)})")
    lines.append("")
    text = "\n".join(lines)
    with open(save_path, "w") as f:
        f.write(text)
    print(f"  Saved metrics     → {save_path}")
    print(text)


# ──────────────────────────────────────────────────────────────
# Model loaders (loaded once, used for all samples)
# ──────────────────────────────────────────────────────────────
def load_ddpm(args, device):
    return UNet2DModel.from_pretrained(args.ddpm_model).to(device).eval()


def load_flow_matching(args, device):
    unet = UNet2DModel(
        sample_size=args.image_size, in_channels=3, out_channels=3,
        layers_per_block=2,
        block_out_channels=(128, 128, 256, 256, 512, 512),
        down_block_types=("DownBlock2D", "DownBlock2D", "DownBlock2D",
                          "DownBlock2D", "AttnDownBlock2D", "DownBlock2D"),
        up_block_types=("UpBlock2D", "AttnUpBlock2D", "UpBlock2D",
                        "UpBlock2D", "UpBlock2D", "UpBlock2D"),
    )
    ckpt_path = args.flow_model
    if os.path.isdir(ckpt_path):
        ckpt_path = os.path.join(ckpt_path, "model.pt")
    ckpt = torch.load(ckpt_path, map_location=device)
    unet.load_state_dict(ckpt["model_state_dict"])
    return unet.to(device).eval()


def load_diff2flow(args, device):
    base_unet = UNet2DModel.from_pretrained(args.ddpm_unet_for_diff2flow).to(device)
    wrapped = DiffusersUNetWrapper(base_unet)
    flow_model = FlowModelObj(
        net_cfg=wrapped, schedule="linear",
        diffusion_parameterization="eps", enforce_zero_snr=False,
    ).to(device)
    scheduler = DDPMScheduler(num_train_timesteps=1000)
    _register_schedule_from_betas(flow_model, scheduler.betas)
    flow_model = flow_model.to(device)
    state = torch.load(args.diff2flow_model, map_location=device)
    flow_model.load_state_dict(state)
    return flow_model.eval()


# ──────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────
def parse_args():
    p = argparse.ArgumentParser(description="Trajectory visualisation")
    p.add_argument("--ddpm-model",      type=str, default=DEFAULTS["ddpm_model"])
    p.add_argument("--flow-model",      type=str, default=DEFAULTS["flow_model"])
    p.add_argument("--diff2flow-model", type=str, default=DEFAULTS["diff2flow_model"])
    p.add_argument("--ddpm-unet-for-diff2flow", type=str,
                   default=DEFAULTS["ddpm_unet_for_diff2flow"],
                   help="Pre-trained DDPM UNet used inside Diff2Flow wrapper")
    p.add_argument("--output-dir",      type=str, default=DEFAULTS["output_dir"])
    p.add_argument("--num-steps",       type=int, default=DEFAULTS["num_steps"])
    p.add_argument("--num-snapshots",   type=int, default=DEFAULTS["num_snapshots"])
    p.add_argument("--num-samples",     type=int, default=DEFAULTS["num_samples"],
                   help="Number of trajectories per model for PCA bundle & straightness")
    p.add_argument("--seed",            type=int, default=DEFAULTS["seed"])
    p.add_argument("--image-size",      type=int, default=DEFAULTS["image_size"])
    p.add_argument("--skip-ddpm",       action="store_true", help="Skip DDPM")
    p.add_argument("--skip-flow",       action="store_true", help="Skip Flow Matching")
    p.add_argument("--skip-diff2flow",  action="store_true", help="Skip Diff2Flow")
    return p.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    print(f"Samples per model: {args.num_samples}")
    print(f"Steps per trajectory: {args.num_steps}")

    # Generate all starting noises (shared across models)
    torch.manual_seed(args.seed)
    all_z = [torch.randn(1, 3, args.image_size, args.image_size, device=device)
             for _ in range(args.num_samples)]

    # Containers: label → list of trajectories
    all_trajs = {}       # for PCA bundle (all N samples)
    single_trajs = {}    # for image strip (just sample 0)
    straightness = {}    # label → list of floats
    labels = []
    schedule = build_schedule(1000)

    # ---- DDPM ----
    if not args.skip_ddpm and os.path.isdir(args.ddpm_model):
        print("\n[1/3] DDPM (DDIM sampling) …")
        model = load_ddpm(args, device)
        trajs = []
        for i in tqdm(range(args.num_samples), desc="  DDPM samples"):
            traj = ddpm_trajectory(model, all_z[i], args.num_steps, schedule, device)
            trajs.append(traj)
        all_trajs["DDPM"] = trajs
        single_trajs["DDPM"] = trajs[0]
        straightness["DDPM"] = [compute_straightness(t) for t in trajs]
        labels.append("DDPM")
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()
        print(f"  {args.num_samples} trajectories captured")
    elif not args.skip_ddpm:
        print(f"[!] DDPM model not found at {args.ddpm_model}, skipping.")

    # ---- Flow Matching ----
    if not args.skip_flow and (os.path.isdir(args.flow_model) or os.path.isfile(args.flow_model)):
        print("\n[2/3] Flow Matching (Euler) …")
        model = load_flow_matching(args, device)
        trajs = []
        for i in tqdm(range(args.num_samples), desc="  Flow samples"):
            traj = flow_matching_trajectory(model, all_z[i], args.num_steps, device)
            trajs.append(traj)
        all_trajs["Flow Matching"] = trajs
        single_trajs["Flow Matching"] = trajs[0]
        straightness["Flow Matching"] = [compute_straightness(t) for t in trajs]
        labels.append("Flow Matching")
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()
        print(f"  {args.num_samples} trajectories captured")
    elif not args.skip_flow:
        print(f"[!] Flow Matching model not found at {args.flow_model}, skipping.")

    # ---- Diff2Flow ----
    if not args.skip_diff2flow and os.path.isfile(args.diff2flow_model):
        print("\n[3/3] Diff2Flow (ODE via FlowModelObj) …")
        model = load_diff2flow(args, device)
        trajs = []
        for i in tqdm(range(args.num_samples), desc="  Diff2Flow samples"):
            traj = diff2flow_trajectory(model, all_z[i], args.num_steps, device)
            trajs.append(traj)
        all_trajs["Diff2Flow"] = trajs
        single_trajs["Diff2Flow"] = trajs[0]
        straightness["Diff2Flow"] = [compute_straightness(t) for t in trajs]
        labels.append("Diff2Flow")
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()
        print(f"  {args.num_samples} trajectories captured")
    elif not args.skip_diff2flow:
        print(f"[!] Diff2Flow model not found at {args.diff2flow_model}, skipping.")

    if not labels:
        print("\nNo models found — nothing to visualise.")
        return

    # ---- Plot ----
    print("\nGenerating plots …")

    # 1) Image strip (first sample only)
    plot_strips(
        single_trajs, labels, args.num_snapshots,
        os.path.join(args.output_dir, "trajectory_strips.png"),
    )

    # 2) PCA bundle (all N samples)
    plot_pca_bundle(
        all_trajs, labels,
        os.path.join(args.output_dir, "trajectory_pca_bundle.png"),
    )

    # 3) Straightness boxplot
    plot_straightness(
        straightness, labels,
        os.path.join(args.output_dir, "straightness_boxplot.png"),
    )

    # 4) Numerical metrics
    save_metrics(
        straightness, labels,
        os.path.join(args.output_dir, "metrics.txt"),
    )

    print("\nDone ✓")


# ──────────────────────────────────────────────────────────────
# Schedule helper for Diff2Flow  (copied from diff2flow.py)
# ──────────────────────────────────────────────────────────────
def _register_schedule_from_betas(flow_model, betas):
    """Register DDPM schedule buffers on a FlowModelObj."""
    betas = betas.detach().cpu().numpy()
    alphas = 1.0 - betas
    alphas_cumprod = alphas.cumprod(axis=0)
    alphas_cumprod_full = np.append(1.0, alphas_cumprod)

    try:
        dev = next(flow_model.parameters()).device
    except StopIteration:
        dev = torch.device("cpu")
    to_t = lambda x: torch.tensor(x, dtype=torch.float32, device=dev)

    flow_model.num_timesteps = int(betas.shape[0])
    flow_model.register_buffer("betas", to_t(betas))
    flow_model.register_buffer("alphas_cumprod", to_t(alphas_cumprod))
    flow_model.register_buffer("alphas_cumprod_full", to_t(alphas_cumprod_full))
    flow_model.register_buffer("sqrt_alphas_cumprod", to_t(np.sqrt(alphas_cumprod)))
    flow_model.register_buffer("sqrt_one_minus_alphas_cumprod", to_t(np.sqrt(1.0 - alphas_cumprod)))
    flow_model.register_buffer("sqrt_alphas_cumprod_full", to_t(np.sqrt(alphas_cumprod_full)))
    flow_model.register_buffer("sqrt_one_minus_alphas_cumprod_full", to_t(np.sqrt(1.0 - alphas_cumprod_full)))
    flow_model.register_buffer("sqrt_recip_alphas_cumprod", to_t(np.sqrt(1.0 / alphas_cumprod)))
    flow_model.register_buffer("sqrt_recipm1_alphas_cumprod", to_t(np.sqrt(1.0 / alphas_cumprod - 1.0)))
    flow_model.register_buffer(
        "rectified_alphas_cumprod_full",
        flow_model.sqrt_alphas_cumprod_full /
        (flow_model.sqrt_alphas_cumprod_full + flow_model.sqrt_one_minus_alphas_cumprod_full),
    )
    flow_model.register_buffer(
        "rectified_sqrt_alphas_cumprod_full",
        flow_model.sqrt_one_minus_alphas_cumprod_full /
        (flow_model.sqrt_alphas_cumprod_full + flow_model.sqrt_one_minus_alphas_cumprod_full),
    )


if __name__ == "__main__":
    main()
