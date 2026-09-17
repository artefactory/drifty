"""
plots.benchmark_per_sample
==========================
Mode: benchmark_per_sample

Per-sample feature distributions and histograms:
- Feature value histograms (correct vs halluc)
- Per-step feature evolution for individual samples

Usage:
    python -m PipelineTest.scripts.plots.benchmark_per_sample --config llada
"""

import os
import argparse
import numpy as np
import matplotlib.pyplot as plt

from .utils import PlotContext


def plot_feature_distributions(ctx: PlotContext):
    """Histogram of each feature split by label."""
    save_dir = ctx.get_save_dir("benchmark_per_sample")

    from PipelineTest.features.baseline import get_baseline_features
    from PipelineTest.features.markovian import get_markovian_features

    pad_id = ctx.pad_token_id if ctx.use_padding else None
    feat_b, names_b = get_baseline_features(ctx.outputs, pad_token_id=pad_id)
    feat_b = feat_b[ctx.positions]

    feat_m, names_m = get_markovian_features(ctx.outputs, k_tokens=20)
    feat_m = feat_m[ctx.positions]

    # Baseline distributions
    _plot_histograms(feat_b, names_b, ctx.labels, save_dir, prefix="baseline_")

    # Markovian distributions (remove NaN columns)
    valid = ~np.any(np.isnan(feat_m), axis=0)
    _plot_histograms(
        feat_m[:, valid],
        [n for n, v in zip(names_m, valid) if v],
        ctx.labels, save_dir, prefix="markovian_",
    )


def _plot_histograms(features, feature_names, labels, save_dir, prefix=""):
    """Plot histogram grid for a set of features."""
    n_feats = len(feature_names)
    if n_feats == 0:
        return

    n_cols = min(3, n_feats)
    n_rows = (n_feats + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(6 * n_cols, 4 * n_rows))
    axes = np.atleast_1d(axes).flatten()

    for i, name in enumerate(feature_names):
        ax = axes[i]
        ax.hist(features[labels == 0, i], bins=30, alpha=0.6, color="green",
                label="Correct", density=True)
        ax.hist(features[labels == 1, i], bins=30, alpha=0.6, color="red",
                label="Halluc", density=True)
        ax.set_title(name, fontsize=9, fontweight="bold")
        ax.legend(fontsize=7)
        ax.grid(True, alpha=0.3)

    for i in range(n_feats, len(axes)):
        axes[i].set_visible(False)

    plt.suptitle(f"{prefix.replace('_', ' ').title()}Feature Distributions", fontsize=12, fontweight="bold")
    plt.tight_layout()
    path = os.path.join(save_dir, f"{prefix}feature_distributions.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"    Saved: {path}")


def plot_per_step_features(ctx: PlotContext, n_samples=10):
    """Plot per-step feature evolution for individual samples (mean/var entropy curves)."""
    save_dir = ctx.get_save_dir("benchmark_per_sample")
    selected, sel_positions, sel_labels = ctx.select_samples(n_samples)

    fig, axes = plt.subplots(len(selected), 2, figsize=(16, 4 * len(selected)))
    if len(selected) == 1:
        axes = axes[np.newaxis, :]

    for plot_idx, global_idx in enumerate(selected):
        pos = ctx.positions[global_idx]
        lbl = ctx.labels[global_idx]
        label_str = "Correct" if lbl == 0 else "Hallucination"
        color = "green" if lbl == 0 else "red"

        entropy_sample = np.array(ctx.entropies[pos])
        mask_sample = np.array(ctx.masks[pos], dtype=float)
        T, D = entropy_sample.shape

        pad_d = ctx.padding_2d[global_idx] if ctx.padding_2d is not None else None

        mean_curve = np.zeros(T)
        var_curve = np.zeros(T)
        for t in range(T):
            if pad_d is not None:
                m = (mask_sample[t, :] > 0) & (~pad_d)
            else:
                m = mask_sample[t, :] > 0
            n_m = int(np.sum(m))
            if n_m > 0:
                vals = entropy_sample[t, m]
                mean_curve[t] = np.mean(vals)
                var_curve[t] = np.var(vals) if n_m > 1 else 0.0

        steps = np.arange(T)

        ax = axes[plot_idx, 0]
        ax.plot(steps, mean_curve, "o-", color=color, markersize=3)
        ax.set_title(f"[{label_str}] Mean Masked Entropy", fontweight="bold")
        ax.set_xlabel("Step")
        ax.set_ylabel("Mean Entropy")
        ax.grid(True, alpha=0.3)

        ax = axes[plot_idx, 1]
        ax.plot(steps, var_curve, "o-", color=color, markersize=3)
        ax.set_title(f"[{label_str}] Var Masked Entropy", fontweight="bold")
        ax.set_xlabel("Step")
        ax.set_ylabel("Variance")
        ax.grid(True, alpha=0.3)

    plt.suptitle(f"Per-Sample Feature Evolution — {ctx.cfg['name']}", fontsize=14, fontweight="bold", y=1.001)
    plt.tight_layout()
    path = os.path.join(save_dir, "per_step_features.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"    Saved: {path}")


def plot(config_name="llada", n_samples=10):
    """Main entry point for benchmark_per_sample mode."""
    print("\n[benchmark_per_sample] Loading data...")
    ctx = PlotContext(config_name, use_padding=True)

    print("[benchmark_per_sample] Feature distributions...")
    plot_feature_distributions(ctx)

    print(f"[benchmark_per_sample] Per-step features ({n_samples} samples)...")
    plot_per_step_features(ctx, n_samples=n_samples)

    print("[benchmark_per_sample] Done.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="llada", choices=["llada", "dream"])
    parser.add_argument("--n-samples", type=int, default=10)
    args = parser.parse_args()
    plot(args.config, args.n_samples)
