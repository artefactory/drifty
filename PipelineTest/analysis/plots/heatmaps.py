"""
plots.heatmaps
==============
Mode: heatmaps

Generates:
- AR(1) parameter heatmaps (phi, intercept, sigma) for correct and halluc models
- Correlation heatmaps between features

Usage:
    python -m PipelineTest.scripts.plots.heatmaps --config llada
"""

import os
import sys
import argparse
import numpy as np
import matplotlib.pyplot as plt

# Support both `python -m PipelineTest.scripts.plots.heatmaps` and direct execution
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))

try:
    from .utils import CONFIGS, PlotContext
except ImportError:
    from PipelineTest.analysis.plots.utils import CONFIGS, PlotContext


plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Computer Modern Roman", "DejaVu Serif"],
    "mathtext.fontset": "cm",
    "font.size": 16,
    "axes.titlesize": 20,
    "axes.labelsize": 16,
    "xtick.labelsize": 14,
    "ytick.labelsize": 14,
    "legend.fontsize": 16,
})


def plot_ar1_parameter_heatmaps(ctx: PlotContext, no_padding=True):
    """Heatmaps of phi, intercept, sigma for both correct and halluc AR1 models."""
    save_dir = ctx.get_save_dir("heatmaps")
    ar1_models = ctx.get_ar1_models(no_padding=no_padding)

    suffix = "_nopad" if no_padding and ctx.padding_2d is not None else ""

    for model_name, (phi, intercept, sigma) in ar1_models.items():
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))

        im0 = axes[0].imshow(phi[1:, :], aspect="auto", cmap="RdBu_r", vmin=-1, vmax=1)
        axes[0].set_title(f"$\\phi$ ({model_name})", fontsize=20)
        axes[0].set_xlabel("Token position $d$")
        axes[0].set_ylabel("Diffusion step $t$")
        plt.colorbar(im0, ax=axes[0])

        im1 = axes[1].imshow(intercept[1:, :], aspect="auto", cmap="RdBu_r")
        axes[1].set_title(f"intercept ({model_name})", fontsize=20)
        axes[1].set_xlabel("Token position $d$")
        axes[1].set_ylabel("Diffusion step $t$")
        plt.colorbar(im1, ax=axes[1])

        im2 = axes[2].imshow(sigma[1:, :], aspect="auto", cmap="viridis", vmin=0)
        axes[2].set_title(f"$\\sigma$ ({model_name})", fontsize=20)
        axes[2].set_xlabel("Token position $d$")
        axes[2].set_ylabel("Diffusion step $t$")
        plt.colorbar(im2, ax=axes[2])

        plt.suptitle(f"AR(1) Parameters — {model_name} model", fontweight="bold", fontsize=24)
        plt.tight_layout(rect=(0, 0, 1, 0.94))
        path = os.path.join(save_dir, f"ar1_heatmap_{model_name}{suffix}.png")
        plt.savefig(path, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"    Saved: {path}")


def plot_correlation_heatmap(ctx: PlotContext):
    """Correlation heatmap of all features."""
    save_dir = ctx.get_save_dir("heatmaps")

    from PipelineTest.features.baseline import get_baseline_features
    from PipelineTest.features.markovian import get_markovian_features

    pad_id = ctx.pad_token_id if ctx.use_padding else None
    feat_b, names_b = get_baseline_features(ctx.outputs, pad_token_id=pad_id)
    feat_b = feat_b[ctx.positions]

    feat_m, names_m = get_markovian_features(ctx.outputs, k_tokens=20)
    feat_m = feat_m[ctx.positions]

    # Combine and remove NaN columns
    features = np.hstack([feat_b, feat_m])
    names = names_b + names_m
    valid = ~np.any(np.isnan(features), axis=0)
    features = features[:, valid]
    names = [n for n, v in zip(names, valid) if v]

    corr = np.corrcoef(features.T)

    fig, ax = plt.subplots(figsize=(14, 12))
    im = ax.imshow(corr, cmap="RdBu_r", vmin=-1, vmax=1)
    ax.set_xticks(range(len(names)))
    ax.set_yticks(range(len(names)))
    ax.set_xticklabels(names, rotation=45, ha="right", fontsize=14)
    ax.set_yticklabels(names, fontsize=14)
    # Annotate each cell with the correlation value
    for i in range(len(names)):
        for j in range(len(names)):
            val = corr[i, j]
            color = "white" if abs(val) > 0.6 else "black"
            ax.text(j, i, f"{val:.2f}", ha="center", va="center",
                    fontsize=8, color=color)
    colorbar = plt.colorbar(im, ax=ax)
    colorbar.ax.tick_params(labelsize=14)
    ax.set_title("Feature Correlation Matrix", fontweight="bold", fontsize=24, pad=20)
    plt.tight_layout()
    path = os.path.join(save_dir, "correlation_heatmap.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"    Saved: {path}")


def plot_label_correlation_bars(ctx: PlotContext):
    """Bar plots of Pearson correlation between each feature and the hallucination label,
    split by feature group: Baseline, Markovian, AR(1), and All combined."""
    save_dir = ctx.get_save_dir("heatmaps")

    from PipelineTest.features.baseline import get_baseline_features
    from PipelineTest.features.markovian import get_markovian_features
    from PipelineTest.features.ar1 import get_ar1_features

    pad_id = ctx.pad_token_id if ctx.use_padding else None

    # --- Baseline ---
    feat_b, names_b = get_baseline_features(ctx.outputs, pad_token_id=pad_id)
    feat_b = feat_b[ctx.positions]

    # --- Markovian ---
    feat_m, names_m = get_markovian_features(ctx.outputs, k_tokens=20)
    feat_m = feat_m[ctx.positions]

    # --- AR(1) ---
    # ar1_models = ctx.get_ar1_models(no_padding=True)
    # masks_arr = np.array([ctx.masks[ctx.positions[i]] for i in range(len(ctx.positions))], dtype=float)
    # if ctx.padding_2d is not None:
    #     pad_3d = ctx.padding_2d[:, np.newaxis, :].repeat(masks_arr.shape[1], axis=1)
    #     effective_mask = masks_arr * (~pad_3d).astype(float)
    # else:
    #     effective_mask = masks_arr
    # feat_a, names_a = get_ar1_features(ctx.outputs, ctx.positions, effective_mask, ar1_models)

    # --- All ---
    feat_all = np.hstack([feat_b, feat_m])
    names_all = names_b + names_m

    labels = ctx.labels.astype(float)

    groups = [
        ("Baseline", feat_b, names_b),
        ("Markovian", feat_m, names_m),
        ("All", feat_all, names_all),
    ]

    for group_name, features, names in groups:
        # Remove NaN columns
        valid = ~np.any(np.isnan(features), axis=0)
        features_clean = features[:, valid]
        names_clean = [n for n, v in zip(names, valid) if v]

        correlations = []
        for i in range(features_clean.shape[1]):
            r = np.corrcoef(features_clean[:, i], labels)[0, 1]
            correlations.append(r)
        correlations = np.array(correlations)

        # Sort by absolute correlation
        order = np.argsort(np.abs(correlations))[::-1]
        sorted_names = [names_clean[i] for i in order]
        sorted_corr = correlations[order]

        positive_color = "#D55E00"
        negative_color = "#0072B2"
        colors = [positive_color if correlation > 0 else negative_color
              for correlation in sorted_corr]
        y_pos = np.arange(len(sorted_names))

        fig_h = max(5, 0.32 * len(sorted_names) + 1.5)
        fig, ax = plt.subplots(figsize=(18, fig_h))

        ax.barh(y_pos, sorted_corr, color=colors, alpha=0.7)
        ax.set_yticks(y_pos)
        ax.set_yticklabels(sorted_names, fontsize=12)
        ax.set_xlabel("Pearson correlation with label", fontsize=20)
        ax.set_title(f"Correlation of {group_name} features with label", fontweight="bold", fontsize=24)
        ax.axvline(0, color="black", linewidth=0.5)
        ax.set_xlim(-0.5, 0.5)
        ax.grid(True, alpha=0.3, axis="x")
        ax.invert_yaxis()

        # Annotate values
        for i, val in enumerate(sorted_corr):
            ax.text(val + (0.01 if val >= 0 else -0.01), i, f"{val:.3f}",
                    va="center", ha="left" if val >= 0 else "right", fontsize=12)

        plt.tight_layout()
        fname = f"label_correlation_{group_name.lower().replace('(', '').replace(')', '')}.png"
        path = os.path.join(save_dir, fname)
        plt.savefig(path, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"    Saved: {path}")


def plot(config_name=None, no_padding=True, mode="all"):
    """Main entry point for heatmaps mode."""
    print("\n[heatmaps] Loading data...")
    ctx = PlotContext(config_name, use_padding=no_padding)

    if mode == "all":
        print("[heatmaps] AR(1) parameter heatmaps...")
        plot_ar1_parameter_heatmaps(ctx, no_padding=no_padding)

        if no_padding and ctx.padding_2d is not None:
            print("[heatmaps] AR(1) parameter heatmaps (with padding for comparison)...")
            plot_ar1_parameter_heatmaps(ctx, no_padding=False)

    print("[heatmaps] Correlation heatmap...")
    plot_correlation_heatmap(ctx)

    print("[heatmaps] Label correlation bars (Baseline / Markovian / AR1)...")
    plot_label_correlation_bars(ctx)

    print("[heatmaps] Done.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=next(iter(CONFIGS)), choices=list(CONFIGS))
    parser.add_argument("--mode", choices=["all", "correlation"], default="all")
    parser.add_argument("--no-padding", action="store_true", default=True)
    args = parser.parse_args()
    plot(args.config, args.no_padding, args.mode)
