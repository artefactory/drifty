"""
plots.benchmark_global
======================
Mode: benchmark_global

Population-level visualizations:
- Mean/Var masked entropy trajectories (correct vs hallucination, averaged over samples)
- Scatter plots (MeanMaskedEntropy vs VarMaskedEntropyAcrossTokens)

Usage:
    python -m PipelineTest.scripts.plots.benchmark_global --config llada
"""

import os
import argparse
import numpy as np
import matplotlib.pyplot as plt

from .utils import PlotContext


plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Computer Modern Roman", "DejaVu Serif"],
    "mathtext.fontset": "cm",
    "font.size": 24,
    "axes.titlesize": 20,
    "axes.labelsize": 20,
    "xtick.labelsize": 16,
    "ytick.labelsize": 16,
    "legend.fontsize": 20,
})


def _plot_colors():
    """Return the shared colorblind-friendly colors used by the fit plots."""
    cividis = plt.get_cmap("cividis")
    return cividis(0.15), cividis(0.85)


def plot_mean_var_trajectories(ctx: PlotContext):
    """Population-level mean and variance of masked entropy per step (correct vs halluc)."""
    save_dir = ctx.get_save_dir("benchmark_global")

    correct_pos = ctx.positions[ctx.labels == 0]
    halluc_pos = ctx.positions[ctx.labels == 1]

    def per_step_stats(pos_arr):
        ent_arr = np.array([ctx.entropies[i] for i in pos_arr])
        mask_arr = np.array([ctx.masks[i] for i in pos_arr], dtype=float)
        N, T, D = ent_arr.shape
        mean_per_step = np.zeros(T)
        var_per_step = np.zeros(T)
        for t in range(T):
            step_means = []
            step_vars = []
            for n in range(N):
                m = mask_arr[n, t, :] > 0
                if np.sum(m) > 0:
                    vals = ent_arr[n, t, m]
                    step_means.append(np.mean(vals))
                    if np.sum(m) > 1:
                        step_vars.append(np.var(vals))
            mean_per_step[t] = np.mean(step_means) if step_means else 0
            var_per_step[t] = np.mean(step_vars) if step_vars else 0
        return mean_per_step, var_per_step

    mean_c, var_c = per_step_stats(correct_pos)
    mean_h, var_h = per_step_stats(halluc_pos)
    T = len(mean_c)
    steps = np.arange(T)
    correct_color, halluc_color = _plot_colors()

    fig, axes = plt.subplots(2, 1, figsize=(14, 10))

    ax = axes[0]
    ax.plot(steps, mean_c, "o-", color=correct_color, markersize=6, label="Correct")
    ax.plot(steps, mean_h, "s-", color=halluc_color, markersize=6, label="Hallucination")
    ax.set_title("Mean Masked Entropy (averaged over samples)", fontweight="bold", fontsize=24)
    ax.set_xlabel("Diffusion Step", fontsize=20)
    ax.set_ylabel("Mean Masked Entropy", fontsize=20)
    ax.legend(fontsize=20)
    ax.grid(True, alpha=0.3)

    ax = axes[1]
    ax.plot(steps, var_c, "o-", color=correct_color, markersize=6, label="Correct")
    ax.plot(steps, var_h, "s-", color=halluc_color, markersize=6, label="Hallucination")
    ax.set_title("Variance of Masked Entropy Across Tokens (averaged over samples)", fontweight="bold", fontsize=24)
    ax.set_xlabel("Diffusion Step", fontsize=20)
    ax.set_ylabel("Variance", fontsize=20)
    ax.legend(fontsize=20)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    path = os.path.join(save_dir, "trajectories_mean_var.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"    Saved: {path}")


def plot_mean_var_logprob_trajectories(ctx: PlotContext):
    """Population-level mean and variance of masked log-probabilities per step."""
    save_dir = ctx.get_save_dir("benchmark_global")

    correct_pos = ctx.positions[ctx.labels == 0]
    halluc_pos = ctx.positions[ctx.labels == 1]

    def per_step_stats(pos_arr):
        lp_arr = np.array([ctx.logprobs[i] for i in pos_arr])
        mask_arr = np.array([ctx.masks[i] for i in pos_arr], dtype=float)
        N, T, D = lp_arr.shape
        mean_per_step = np.zeros(T)
        var_per_step = np.zeros(T)
        for t in range(T):
            step_means = []
            step_vars = []
            for n in range(N):
                m = mask_arr[n, t, :] > 0
                if np.sum(m) > 0:
                    vals = lp_arr[n, t, m]
                    step_means.append(np.mean(vals))
                    if np.sum(m) > 1:
                        step_vars.append(np.var(vals))
            mean_per_step[t] = np.mean(step_means) if step_means else 0
            var_per_step[t] = np.mean(step_vars) if step_vars else 0
        return mean_per_step, var_per_step

    mean_c, var_c = per_step_stats(correct_pos)
    mean_h, var_h = per_step_stats(halluc_pos)
    T = len(mean_c)
    steps = np.arange(T)
    correct_color, halluc_color = _plot_colors()

    fig, axes = plt.subplots(2, 1, figsize=(14, 10))

    ax = axes[0]
    ax.plot(steps, mean_c, "o-", color=correct_color, markersize=6, label="Correct")
    ax.plot(steps, mean_h, "s-", color=halluc_color, markersize=6, label="Hallucination")
    ax.set_title("Mean Masked Log-Probability (averaged over samples)", fontweight="bold", fontsize=24)
    ax.set_xlabel("Diffusion Step", fontsize=20)
    ax.set_ylabel("Mean Masked LogProb", fontsize=20)
    ax.legend(fontsize=12)
    ax.grid(True, alpha=0.3)

    ax = axes[1]
    ax.plot(steps, var_c, "o-", color=correct_color, markersize=6, label="Correct")
    ax.plot(steps, var_h, "s-", color=halluc_color, markersize=6, label="Hallucination")
    ax.set_title("Variance of Masked Log-Probability Across Tokens (averaged over samples)", fontweight="bold", fontsize=24)
    ax.set_xlabel("Diffusion Step", fontsize=20)
    ax.set_ylabel("Variance", fontsize=20)
    ax.legend(fontsize=20)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    path = os.path.join(save_dir, "trajectories_mean_var_logprob.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"    Saved: {path}")


def plot_scatter_features(ctx: PlotContext):
    """Scatter plot: MeanMaskedEntropy vs VarMaskedEntropyAcrossTokens."""
    save_dir = ctx.get_save_dir("benchmark_global")

    from PipelineTest.features.baseline import (
        mean_masked_entropy, var_masked_entropy_across_tokens,
    )

    mean_ent = mean_masked_entropy(ctx.outputs)[ctx.positions]
    var_ent = var_masked_entropy_across_tokens(ctx.outputs)[ctx.positions]

    fig, ax = plt.subplots(figsize=(10, 8))
    correct_color, halluc_color = _plot_colors()
    ax.scatter(mean_ent[ctx.labels == 0], var_ent[ctx.labels == 0],
               alpha=0.5, color=correct_color, s=20, label="Correct")
    ax.scatter(mean_ent[ctx.labels == 1], var_ent[ctx.labels == 1],
               alpha=0.5, color=halluc_color, s=20, label="Hallucination")
    ax.set_xlabel("Mean Masked Entropy", fontsize=12)
    ax.set_ylabel("Var Masked Entropy Across Tokens", fontsize=12)
    ax.set_title("Scatter: MeanMaskedEntropy vs VarMaskedEntropyAcrossTokens", fontweight="bold", fontsize=12)
    ax.legend(fontsize=12)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    path = os.path.join(save_dir, "scatter_mean_vs_var.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"    Saved: {path}")


def plot_scatter_all_features(ctx: PlotContext):
    """Scatter matrix of top discriminative features."""
    save_dir = ctx.get_save_dir("benchmark_global")

    from PipelineTest.features.baseline import get_baseline_features
    from PipelineTest.features.markovian import get_markovian_features

    pad_id = ctx.pad_token_id if ctx.use_padding else None
    feat_b, names_b = get_baseline_features(ctx.outputs, pad_token_id=pad_id)
    feat_b = feat_b[ctx.positions]
    feat_m, names_m = get_markovian_features(ctx.outputs, k_tokens=20)
    feat_m = feat_m[ctx.positions]

    features = np.hstack([feat_b, feat_m])
    names = names_b + names_m
    valid = ~np.any(np.isnan(features), axis=0)
    features = features[:, valid]
    names = [n for n, v in zip(names, valid) if v]

    # Pick top 4 features by AUC
    from sklearn.metrics import roc_auc_score
    aucs = []
    for i in range(features.shape[1]):
        try:
            auc = roc_auc_score(ctx.labels, features[:, i])
            aucs.append(max(auc, 1 - auc))
        except ValueError:
            aucs.append(0.5)
    top_k = np.argsort(aucs)[-4:]

    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    correct_color, halluc_color = _plot_colors()
    axes = axes.flatten()
    pair_idx = 0
    for i in range(len(top_k)):
        for j in range(i + 1, len(top_k)):
            if pair_idx >= 6:
                break
            ax = axes[pair_idx]
            fi, fj = top_k[i], top_k[j]
            ax.scatter(features[ctx.labels == 0, fi], features[ctx.labels == 0, fj],
                       alpha=0.4, color=correct_color, s=15, label="Correct")
            ax.scatter(features[ctx.labels == 1, fi], features[ctx.labels == 1, fj],
                       alpha=0.4, color=halluc_color, s=15, label="Halluc")
            ax.set_xlabel(names[fi], fontsize=12)
            ax.set_ylabel(names[fj], fontsize=12)
            ax.legend(fontsize=12)
            ax.grid(True, alpha=0.3)
            pair_idx += 1

    for i in range(pair_idx, 6):
        axes[i].set_visible(False)

    plt.suptitle("Top Feature Pairs (by AUC)", fontweight="bold", fontsize=12)
    plt.tight_layout()
    path = os.path.join(save_dir, "scatter_top_features.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"    Saved: {path}")


def plot(config_name="llada"):
    """Main entry point for benchmark_global mode."""
    print("\n[benchmark_global] Loading data...")
    ctx = PlotContext(config_name, use_padding=True)

    print("[benchmark_global] Mean/Var entropy trajectories...")
    plot_mean_var_trajectories(ctx)

    print("[benchmark_global] Mean/Var logprob trajectories...")
    plot_mean_var_logprob_trajectories(ctx)

    print("[benchmark_global] Scatter (mean vs var)...")
    plot_scatter_features(ctx)

    print("[benchmark_global] Scatter (top features)...")
    plot_scatter_all_features(ctx)

    print("[benchmark_global] Done.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="llada", choices=["llada", "dream"])
    args = parser.parse_args()
    plot(args.config)
