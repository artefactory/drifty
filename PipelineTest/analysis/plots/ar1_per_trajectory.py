"""
plots.ar1_per_trajectory
========================
Mode: ar1_per_trajectory

Individual token entropy trajectories per sample:
- 10 tokens evenly spaced in unmasking order
- Data (solid) vs AR1 reconstruction (dashed)
- Side-by-side: correct model recon | halluc model recon
Supports with/without padding.

Usage:
    python -m PipelineTest.scripts.plots.ar1_per_trajectory --config llada --n-samples 10
"""

import os
import argparse
import numpy as np
import matplotlib.pyplot as plt

from .utils import PlotContext


N_TOKENS_TRAJ = 10


def _get_unmasking_order(mask_sample, padding_d=None):
    """Get token unmasking order (earliest to latest)."""
    T, D = mask_sample.shape
    unmask_step = np.full(D, T)
    for d in range(D):
        for t in range(1, T):
            if mask_sample[t - 1, d] == 1 and mask_sample[t, d] == 0:
                unmask_step[d] = t
                break

    if padding_d is not None:
        valid_tokens = np.where((unmask_step < T) & (~padding_d))[0]
    else:
        valid_tokens = np.where(unmask_step < T)[0]

    valid_unmask = unmask_step[valid_tokens]
    token_order = valid_tokens[np.argsort(valid_unmask)]
    return token_order, unmask_step


def plot_ar1_per_trajectory(ctx: PlotContext, n_samples=10, no_padding=True):
    """Plot per-sample 10-token trajectory: data vs AR1 (correct | halluc)."""
    save_dir = ctx.get_save_dir("ar1_per_trajectory")
    ar1_models = ctx.get_ar1_models(no_padding=no_padding)
    phi_c, int_c, _ = ar1_models["correct"]
    phi_h, int_h, _ = ar1_models["halluc"]

    selected, sel_positions, sel_labels = ctx.select_samples(n_samples)
    suffix = "_nopad" if no_padding and ctx.padding_2d is not None else ""

    colors_tokens = plt.cm.tab10(np.linspace(0, 1, N_TOKENS_TRAJ))

    fig, axes = plt.subplots(len(selected), 2, figsize=(18, 4 * len(selected)))
    if len(selected) == 1:
        axes = axes[np.newaxis, :]

    for plot_idx, global_idx in enumerate(selected):
        pos = ctx.positions[global_idx]
        lbl = ctx.labels[global_idx]
        label_str = "Correct" if lbl == 0 else "Hallucination"

        entropy_sample = np.array(ctx.entropies[pos])
        mask_sample = np.array(ctx.masks[pos], dtype=float)
        T, D = entropy_sample.shape

        pad_d = ctx.padding_2d[global_idx] if (no_padding and ctx.padding_2d is not None) else None

        token_order, unmask_step = _get_unmasking_order(mask_sample, pad_d)

        # Pick N_TOKENS_TRAJ tokens evenly spaced in unmasking order
        if len(token_order) >= N_TOKENS_TRAJ:
            pick_indices = np.linspace(0, len(token_order) - 1, N_TOKENS_TRAJ, dtype=int)
            selected_tokens = token_order[pick_indices]
        else:
            selected_tokens = token_order

        steps = np.arange(T)

        # Left: correct model reconstruction
        ax_left = axes[plot_idx, 0]
        for k, tok_d in enumerate(selected_tokens):
            entropy_data = entropy_sample[:, tok_d]
            recon = np.zeros(T)
            recon[0] = entropy_data[0]
            for t in range(1, T):
                recon[t] = phi_c[t, tok_d] * entropy_sample[t - 1, tok_d] + int_c[t, tok_d]

            ax_left.plot(steps, entropy_data, "-", color=colors_tokens[k], alpha=0.4, linewidth=1)
            ax_left.plot(steps, recon, "--", color=colors_tokens[k], linewidth=1.5,
                         label=f"d={tok_d} (unmask={unmask_step[tok_d]})")

        ax_left.set_title(f"Sample {plot_idx + 1} [{label_str}] — Correct model recon", fontweight="bold")
        ax_left.set_xlabel("Diffusion Step")
        ax_left.set_ylabel("Entropy")
        ax_left.legend(fontsize=6, ncol=2)
        ax_left.grid(True, alpha=0.3)

        # Right: hallucination model reconstruction
        ax_right = axes[plot_idx, 1]
        for k, tok_d in enumerate(selected_tokens):
            entropy_data = entropy_sample[:, tok_d]
            recon = np.zeros(T)
            recon[0] = entropy_data[0]
            for t in range(1, T):
                recon[t] = phi_h[t, tok_d] * entropy_sample[t - 1, tok_d] + int_h[t, tok_d]

            ax_right.plot(steps, entropy_data, "-", color=colors_tokens[k], alpha=0.4, linewidth=1)
            ax_right.plot(steps, recon, "--", color=colors_tokens[k], linewidth=1.5,
                          label=f"d={tok_d} (unmask={unmask_step[tok_d]})")

        ax_right.set_title(f"Sample {plot_idx + 1} [{label_str}] — Halluc model recon", fontweight="bold")
        ax_right.set_xlabel("Diffusion Step")
        ax_right.set_ylabel("Entropy")
        ax_right.legend(fontsize=6, ncol=2)
        ax_right.grid(True, alpha=0.3)

    plt.suptitle(
        f"AR(1) Per-Trajectory Reconstruction ({N_TOKENS_TRAJ} tokens) — {ctx.cfg['name']}{suffix}",
        fontsize=14, fontweight="bold", y=1.001,
    )
    plt.tight_layout()
    path = os.path.join(save_dir, f"ar1_per_trajectory{suffix}.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"    Saved: {path}")


def plot(config_name="llada", n_samples=10, no_padding=True):
    """Main entry point for ar1_per_trajectory mode."""
    print("\n[ar1_per_trajectory] Loading data...")
    ctx = PlotContext(config_name, use_padding=no_padding)

    print(f"[ar1_per_trajectory] Plotting {n_samples} samples (no_padding={no_padding})...")
    plot_ar1_per_trajectory(ctx, n_samples=n_samples, no_padding=no_padding)

    if no_padding and ctx.padding_2d is not None:
        print("[ar1_per_trajectory] Also plotting WITH padding for comparison...")
        plot_ar1_per_trajectory(ctx, n_samples=n_samples, no_padding=False)

    print("[ar1_per_trajectory] Done.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="llada", choices=["llada", "dream"])
    parser.add_argument("--n-samples", type=int, default=10)
    parser.add_argument("--no-padding", action="store_true", default=True)
    args = parser.parse_args()
    plot(args.config, args.n_samples, args.no_padding)
