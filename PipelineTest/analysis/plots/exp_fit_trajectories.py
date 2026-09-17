"""
plots.exp_fit_trajectories
===========================
Fit exponential decay c * exp(-t/tau) on mean unmasked entropy
trajectories (correct vs hallucination), via log-linear regression:
    ln(y) = alpha * t + beta  =>  tau = -1/alpha, c = exp(beta)

Produces:
- Plot with data + fitted curves
- Prints R^2 for each fit

Usage:
    python -m PipelineTest.scripts.plots.exp_fit_trajectories --config llada
    python -m PipelineTest.scripts.plots.exp_fit_trajectories --config dream
"""

import os
import sys
import argparse
from typing import cast
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import linregress

# Support both module execution and direct execution of this file.
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))

try:
    from .utils import CONFIGS, PlotContext
except ImportError:
    from PipelineTest.analysis.plots.utils import CONFIGS, PlotContext


plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Computer Modern Roman", "DejaVu Serif"],
    "mathtext.fontset": "cm",
    "font.size": 24,
    "axes.titlesize": 20,
    "axes.labelsize": 20,
    "xtick.labelsize": 16,
    "ytick.labelsize": 16,
    "legend.fontsize": 14,
})


def compute_per_step_stats(ctx: PlotContext, pos_arr):
    """Compute mean entropy per diffusion step without mask filtering."""
    ent_arr = np.array([ctx.entropies[i] for i in pos_arr])
    N, T, D = ent_arr.shape
    mean_per_step = np.zeros(T)
    for t in range(T):
        step_means = []
        for n in range(N):
            vals = ent_arr[n, t, :]
            if ctx.padding_2d is not None:
                vals = vals[~ctx.padding_2d[n]]
            vals = vals[np.isfinite(vals)]
            if len(vals) > 0:
                step_means.append(np.mean(vals))
        mean_per_step[t] = np.mean(step_means) if step_means else 0
    return mean_per_step


def fit_exponential(t, y, label=""):
    """
    Fit y = c * exp(-t/tau) via ln(y) = alpha*t + beta.
    Only uses points where y > 0.
    Returns (alpha, beta, tau, c, r_squared, t_valid, y_fitted).
    """
    valid = y > 0
    t_valid = t[valid]
    y_valid = y[valid]
    
    if len(t_valid) < 3:
        print(f"  [{label}] Not enough positive points for fit.")
        return None
    
    ln_y = np.log(y_valid)
    result = cast(tuple[float, float, float, float, float], linregress(t_valid, ln_y))
    slope, intercept, r_value = result[:3]
    
    r_squared = r_value ** 2
    tau = -1.0 / slope if slope != 0 else np.inf
    c = np.exp(intercept)
    
    # Fitted curve over all valid t
    y_fitted = c * np.exp(slope * t_valid)
    
    return {
        "alpha": slope,
        "beta": intercept,
        "tau": tau,
        "c": c,
        "r_squared": r_squared,
        "t_valid": t_valid,
        "y_fitted": y_fitted,
    }


def plot_exp_fit(ctx: PlotContext):
    """Plot mean entropy trajectories with exponential fits."""
    save_dir = ctx.get_save_dir("exp_fit")

    correct_pos = ctx.positions[ctx.labels == 0]
    halluc_pos = ctx.positions[ctx.labels == 1]

    mean_c = compute_per_step_stats(ctx, correct_pos)
    mean_h = compute_per_step_stats(ctx, halluc_pos)

    T = len(mean_c)
    t = np.arange(T, dtype=float)

    # --- Fits ---
    fits = {}
    fits["mean_correct"] = fit_exponential(t, mean_c, "Mean Correct")
    fits["mean_halluc"] = fit_exponential(t, mean_h, "Mean Halluc")

    # --- Print results ---
    print("\n" + "=" * 60)
    print(f"  Exponential fit results: {ctx.cfg['name']}")
    print("=" * 60)
    for name, fit in fits.items():
        if fit is not None:
            print(f"  {name:20s} | R² = {fit['r_squared']:.4f} | "
                  f"tau = {fit['tau']:.2f} | c = {fit['c']:.4f} | "
                  f"alpha = {fit['alpha']:.6f}")
        else:
            print(f"  {name:20s} | FIT FAILED")
    print("=" * 60)

    # --- Plot ---
    fig, ax = plt.subplots(1, 1, figsize=(14, 5))
    cividis = plt.get_cmap("cividis")
    correct_color, halluc_color = cividis(0.15), cividis(0.85)
    ax.tick_params(axis="both", labelsize=16)

    # Mean entropy
    ax.plot(t, mean_c, "o-", color=correct_color, linewidth=2.2, markersize=6,
            label="Correct (data)")
    ax.plot(t, mean_h, "s-", color=halluc_color, linewidth=2.2, markersize=6,
            label="Hallucination (data)")
    if fits["mean_correct"] is not None:
        f = fits["mean_correct"]
        ax.plot(f["t_valid"], f["y_fitted"], "--", color=correct_color, linewidth=2.5,
                label=f"Correct fit (R²={f['r_squared']:.3f}, τ={f['tau']:.1f})")
    if fits["mean_halluc"] is not None:
        f = fits["mean_halluc"]
        ax.plot(f["t_valid"], f["y_fitted"], "--", color=halluc_color, linewidth=2.5,
                label=f"Halluc fit (R²={f['r_squared']:.3f}, τ={f['tau']:.1f})")
    ax.set_title("Mean Entropy + Exp Fit: $c \\cdot e^{-t/\\tau}$",
                 fontweight="bold", fontsize=24, pad=25)
    ax.set_xlabel("Diffusion Step", fontsize=20)
    ax.set_ylabel("Mean Entropy", fontsize=20)
    ax.legend(fontsize=15)
    ax.grid(True, alpha=0.3)

    plt.tight_layout(rect=(0, 0, 1, 0.94))
    path = os.path.join(save_dir, "exp_fit_mean.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"\n    Saved: {path}")

    return fits


def plot(config_name="llada"):
    """Main entry point."""
    print(f"\n[exp_fit] Loading data ({config_name})...")
    ctx = PlotContext(config_name, use_padding=True)
    print("[exp_fit] Computing fits...")
    results = plot_exp_fit(ctx)
    print("[exp_fit] Done.")
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=next(iter(CONFIGS)), choices=list(CONFIGS.keys()))
    args = parser.parse_args()
    plot(args.config)
