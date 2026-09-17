"""
plots.ar1_per_sample
====================
Mode: ar1_per_sample

Per-sample mean/var masked entropy: data vs AR1 reconstruction (aggregated over tokens).
Generates one plot per sample with 2 subplots (mean + variance).
Supports with/without padding comparison.

Usage:
    python -m PipelineTest.scripts.plots.ar1_per_sample --config llada --n-samples 20
"""

import os
import argparse
import numpy as np
import matplotlib.pyplot as plt

from .utils import PlotContext


def _compute_per_step_stats(entropy_sample, mask_sample, phi, intercept, padding_d=None):
    """Compute per-step mean/var for data + AR1 reconstruction."""
    T, D = entropy_sample.shape
    mean_data = np.zeros(T)
    var_data = np.zeros(T)
    mean_recon = np.zeros(T)
    var_recon = np.zeros(T)

    for t in range(T):
        if padding_d is not None:
            m = (mask_sample[t, :] > 0) & (~padding_d)
        else:
            m = mask_sample[t, :] > 0
        n_masked = int(np.sum(m))
        if n_masked > 0:
            vals = entropy_sample[t, m]
            mean_data[t] = np.mean(vals)
            var_data[t] = np.var(vals) if n_masked > 1 else 0.0

            if t == 0:
                mean_recon[t] = mean_data[t]
                var_recon[t] = var_data[t]
            else:
                x_prev = entropy_sample[t - 1, :]
                pred = phi[t, :] * x_prev + intercept[t, :]
                mean_recon[t] = np.mean(pred[m])
                var_recon[t] = np.var(pred[m]) if n_masked > 1 else 0.0

    return mean_data, var_data, mean_recon, var_recon


def plot_ar1_per_sample(ctx: PlotContext, n_samples=20, no_padding=True):
    """Plot per-sample mean/var masked entropy: data vs AR1 correct + halluc."""
    save_dir = ctx.get_save_dir("ar1_per_sample")
    ar1_models = ctx.get_ar1_models(no_padding=no_padding)
    phi_c, int_c, _ = ar1_models["correct"]
    phi_h, int_h, _ = ar1_models["halluc"]

    selected, sel_positions, sel_labels = ctx.select_samples(n_samples)

    suffix = "_nopad" if no_padding and ctx.padding_2d is not None else ""

    fig, axes = plt.subplots(len(selected), 2, figsize=(16, 4 * len(selected)))
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

        # Data vs correct model
        mean_data, var_data, mean_recon_c, var_recon_c = _compute_per_step_stats(
            entropy_sample, mask_sample, phi_c, int_c, pad_d
        )
        # Data vs halluc model
        _, _, mean_recon_h, var_recon_h = _compute_per_step_stats(
            entropy_sample, mask_sample, phi_h, int_h, pad_d
        )

        steps = np.arange(T)

        # Mean
        ax = axes[plot_idx, 0]
        ax.plot(steps, mean_data, "o-", color="blue", markersize=3, label="Data")
        ax.plot(steps, mean_recon_c, "s--", color="green", markersize=3, label="AR1 (correct)")
        ax.plot(steps, mean_recon_h, "^--", color="red", markersize=3, label="AR1 (halluc)")
        ax.set_title(f"[{label_str}] Mean Masked Entropy{suffix}", fontweight="bold")
        ax.set_xlabel("Diffusion Step")
        ax.set_ylabel("Mean Entropy")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

        # Variance
        ax = axes[plot_idx, 1]
        ax.plot(steps, var_data, "o-", color="blue", markersize=3, label="Data")
        ax.plot(steps, var_recon_c, "s--", color="green", markersize=3, label="AR1 (correct)")
        ax.plot(steps, var_recon_h, "^--", color="red", markersize=3, label="AR1 (halluc)")
        ax.set_title(f"[{label_str}] Var Masked Entropy{suffix}", fontweight="bold")
        ax.set_xlabel("Diffusion Step")
        ax.set_ylabel("Variance")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    plt.suptitle(
        f"AR(1) Per-Sample Reconstruction — {ctx.cfg['name']}{suffix}",
        fontsize=14, fontweight="bold", y=1.001,
    )
    plt.tight_layout()
    path = os.path.join(save_dir, f"ar1_per_sample_mean_var{suffix}.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"    Saved: {path}")


def plot(config_name="llada", n_samples=20, no_padding=True):
    """Main entry point for ar1_per_sample mode."""
    print("\n[ar1_per_sample] Loading data...")
    ctx = PlotContext(config_name, use_padding=no_padding)

    print(f"[ar1_per_sample] Plotting {n_samples} samples (no_padding={no_padding})...")
    plot_ar1_per_sample(ctx, n_samples=n_samples, no_padding=no_padding)

    if no_padding and ctx.padding_2d is not None:
        print("[ar1_per_sample] Also plotting WITH padding for comparison...")
        plot_ar1_per_sample(ctx, n_samples=n_samples, no_padding=False)

    print("[ar1_per_sample] Done.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="llada", choices=["llada", "dream"])
    parser.add_argument("--n-samples", type=int, default=20)
    parser.add_argument("--no-padding", action="store_true", default=True)
    args = parser.parse_args()
    plot(args.config, args.n_samples, args.no_padding)
