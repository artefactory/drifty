"""
Standalone exponential fit on mean/var masked entropy trajectories.
Avoids importing dllm to sidestep heavy dependencies.

Usage:
    python PipelineTest/scripts/exp_fit_standalone.py --config llada
    python PipelineTest/scripts/exp_fit_standalone.py --config dream
"""

import os
import sys
import argparse
import json
import numpy as np
import torch
import matplotlib.pyplot as plt
from scipy.stats import linregress

# Support execution both as a module and as a direct script.
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
from PipelineTest.eval_configs import CONFIGS


def load_data(config_name):
    """Load outputs and match with eval labels."""
    cfg = CONFIGS[config_name]
    
    # Load outputs
    outputs = torch.load(cfg["outputs_path"], map_location="cpu", weights_only=False)
    
    # Load eval json
    with open(cfg["eval_json"], "r") as f:
        eval_data = json.load(f)
    
    # Extract entropy and masks from outputs
    # histories_entropy: list of T tensors of shape [B, seq_len]
    # histories_mask: list of T tensors of shape [B, seq_len] (bool)
    ent_list = outputs.histories_entropy  # list of [B, D] tensors
    mask_list = outputs.histories_mask    # list of [B, D] bool tensors
    
    B = ent_list[0].shape[0]
    T = len(ent_list)
    D = ent_list[0].shape[1]
    
    # Stack into arrays [B, T, D]
    entropies = torch.stack(ent_list, dim=1).float().numpy()  # [B, T, D]
    masks = torch.stack(mask_list, dim=1).float().numpy()      # [B, T, D]
    
    # Get start indices to crop to generation region only
    if hasattr(outputs, 'start_idx_history') and outputs.start_idx_history is not None:
        start_indices = outputs.start_idx_history
    else:
        start_indices = [0] * B
    
    max_new = outputs.max_new_tokens if hasattr(outputs, 'max_new_tokens') and outputs.max_new_tokens else D
    
    # Crop to generation region
    gen_entropies = np.zeros((B, T, max_new))
    gen_masks = np.zeros((B, T, max_new))
    for i in range(B):
        s = start_indices[i] if isinstance(start_indices, list) else start_indices
        e = s + max_new
        if e <= D:
            gen_entropies[i] = entropies[i, :, s:e]
            gen_masks[i] = masks[i, :, s:e]
    
    # Match with eval json to get labels
    labels = np.array([1 if item.get("is_hallucination", "no") == "yes" else 0 for item in eval_data[:B]])
    
    return gen_entropies, gen_masks, labels, cfg


def per_step_stats(entropies, masks, indices):
    """Compute mean-of-means and mean-of-variances per step for given sample indices."""
    ent = entropies[indices]  # [N, T, D]
    msk = masks[indices]      # [N, T, D]
    N, T, D = ent.shape
    
    mean_per_step = np.zeros(T)
    var_per_step = np.zeros(T)
    
    for t in range(T):
        step_means = []
        step_vars = []
        for n in range(N):
            m = msk[n, t, :] > 0
            if np.sum(m) > 0:
                vals = ent[n, t, m]
                step_means.append(np.mean(vals))
                if np.sum(m) > 1:
                    step_vars.append(np.var(vals))
        mean_per_step[t] = np.mean(step_means) if step_means else 0
        var_per_step[t] = np.mean(step_vars) if step_vars else 0
    
    return mean_per_step, var_per_step


def fit_exponential(t, y, label=""):
    """Fit y = c * exp(-t/tau) via ln(y) = alpha*t + beta."""
    valid = y > 1e-10  # avoid log(0)
    t_valid = t[valid]
    y_valid = y[valid]
    
    if len(t_valid) < 3:
        print(f"  [{label}] Not enough positive points for fit.")
        return None
    
    ln_y = np.log(y_valid)
    result = linregress(t_valid, ln_y)
    
    r_squared = result.rvalue ** 2
    tau = -1.0 / result.slope if result.slope != 0 else np.inf
    c = np.exp(result.intercept)
    
    # Smooth curve for plotting
    t_smooth = np.linspace(t_valid[0], t_valid[-1], 300)
    y_fitted_smooth = c * np.exp(result.slope * t_smooth)
    
    return {
        "alpha": result.slope,
        "beta": result.intercept,
        "tau": tau,
        "c": c,
        "r_squared": r_squared,
        "t_smooth": t_smooth,
        "y_fitted": y_fitted_smooth,
    }


def fit_t_exponential(t, y, label=""):
    """Fit y = C * t * exp(-t/tau) via ln(y/t) = alpha*t + beta (for t > 0)."""
    valid = (y > 1e-10) & (t > 0)
    t_valid = t[valid]
    y_valid = y[valid]
    
    if len(t_valid) < 3:
        print(f"  [{label}] Not enough valid points for t*exp fit.")
        return None
    
    # ln(y/t) = ln(C) + alpha*t  where alpha = -1/tau
    ln_y_over_t = np.log(y_valid / t_valid)
    result = linregress(t_valid, ln_y_over_t)
    
    r_squared = result.rvalue ** 2
    tau = -1.0 / result.slope if result.slope != 0 else np.inf
    C = np.exp(result.intercept)
    
    # Smooth curve for plotting
    t_smooth = np.linspace(t_valid[0], t_valid[-1], 300)
    y_fitted_smooth = C * t_smooth * np.exp(result.slope * t_smooth)
    
    return {
        "alpha": result.slope,
        "beta": result.intercept,
        "tau": tau,
        "c": C,
        "r_squared": r_squared,
        "t_smooth": t_smooth,
        "y_fitted": y_fitted_smooth,
    }

def main(config_name):
    print(f"\n[exp_fit] Loading data ({config_name})...")
    entropies, masks, labels, cfg = load_data(config_name)
    
    correct_idx = np.where(labels == 0)[0]
    halluc_idx = np.where(labels == 1)[0]
    print(f"  Correct: {len(correct_idx)}, Hallucination: {len(halluc_idx)}")
    
    mean_c, var_c = per_step_stats(entropies, masks, correct_idx)
    mean_h, var_h = per_step_stats(entropies, masks, halluc_idx)
    
    T = len(mean_c)
    t = np.arange(T, dtype=float)
    
    # --- Fits ---
    fits = {}
    fits["mean_correct"] = fit_exponential(t, mean_c, "Mean Correct")
    fits["mean_halluc"] = fit_exponential(t, mean_h, "Mean Halluc")
    fits["mean_correct_texp"] = fit_t_exponential(t, mean_c, "Mean Correct (t*exp)")
    fits["mean_halluc_texp"] = fit_t_exponential(t, mean_h, "Mean Halluc (t*exp)")
   
    
    # --- Print results ---
    print("\n" + "=" * 70)
    print(f"  Exponential fit: ln(y) = alpha*t + beta  =>  y = c*exp(-t/tau)")
    print(f"  Config: {cfg['name']}")
    print("=" * 70)
    print(f"  {'Signal':<20s} | {'R²':>8s} | {'tau':>8s} | {'c':>8s} | {'alpha':>10s}")
    print("-" * 70)
    for name, fit in fits.items():
        if fit is not None:
            print(f"  {name:<20s} | {fit['r_squared']:>8.4f} | {fit['tau']:>8.2f} | "
                  f"{fit['c']:>8.4f} | {fit['alpha']:>10.6f}")
        else:
            print(f"  {name:<20s} | {'FAILED':>8s}")
    print("=" * 70)
    
    # --- Plot ---
    fig, axes = plt.subplots(1, 2, figsize=(18, 7))
    
    # Left: c * exp(-t/tau)
    ax = axes[0]
    ax.plot(t, mean_c, "o-", color="green", markersize=3, label="Correct (data)")
    ax.plot(t, mean_h, "s-", color="red", markersize=3, label="Hallucination (data)")
    if fits["mean_correct"] is not None:
        f = fits["mean_correct"]
        ax.plot(f["t_smooth"], f["y_fitted"], "--", color="darkgreen", linewidth=2,
                label=f"Correct: R²={f['r_squared']:.3f}, τ={f['tau']:.1f}")
    if fits["mean_halluc"] is not None:
        f = fits["mean_halluc"]
        ax.plot(f["t_smooth"], f["y_fitted"], "--", color="darkred", linewidth=2,
                label=f"Halluc: R²={f['r_squared']:.3f}, τ={f['tau']:.1f}")
    ax.set_title(r"Fit: $y = c \cdot e^{-t/\tau}$", fontweight="bold")
    ax.set_xlabel("Diffusion Step")
    ax.set_ylabel("Mean Masked Entropy")
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Right: C * t * exp(-t/tau)
    ax = axes[1]
    ax.plot(t, mean_c, "o-", color="green", markersize=3, label="Correct (data)")
    ax.plot(t, mean_h, "s-", color="red", markersize=3, label="Hallucination (data)")
    if fits["mean_correct_texp"] is not None:
        f = fits["mean_correct_texp"]
        ax.plot(f["t_smooth"], f["y_fitted"], "--", color="darkgreen", linewidth=2,
                label=f"Correct: R²={f['r_squared']:.3f}, τ={f['tau']:.1f}")
    if fits["mean_halluc_texp"] is not None:
        f = fits["mean_halluc_texp"]
        ax.plot(f["t_smooth"], f["y_fitted"], "--", color="darkred", linewidth=2,
                label=f"Halluc: R²={f['r_squared']:.3f}, τ={f['tau']:.1f}")
    ax.set_title(r"Fit: $y = C \cdot t \cdot e^{-t/\tau}$", fontweight="bold")
    ax.set_xlabel("Diffusion Step")
    ax.set_ylabel("Mean Masked Entropy")
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.suptitle(f"Mean Masked Entropy — {cfg['name']}", fontweight="bold", fontsize=13)
    plt.tight_layout()
    save_dir = os.path.join(os.path.dirname(cfg["outputs_path"]), "plots")
    os.makedirs(save_dir, exist_ok=True)
    path = os.path.join(save_dir, "exp_fit_mean_var.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"\n    Saved: {path}")


def plot(config_name="llada"):
    """Entry point for run_analysis dispatch."""
    main(config_name)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=next(iter(CONFIGS)), choices=list(CONFIGS.keys()))
    args = parser.parse_args()
    main(args.config)
