"""
AR(1) population-level: mean and variance of masked entropy
data vs AR1 reconstruction, averaged over correct/halluc samples.

Standalone script (avoids importing dllm).

Usage:
    python PipelineTest/scripts/ar1_global_trajectory.py --config llada
    python PipelineTest/scripts/ar1_global_trajectory.py --config dream
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


LEGACY_CONFIGS = {
    "llada": {
        "outputs_path": "PipelineTest/res/results_LLaDa_64steps_64tokens_low_confidence/outputs_LLaDa_64steps_64tokens_low_confidence.pt",
        "eval_json": "PipelineTest/res/eval/results_triviaqa_LLaDa_64steps_64tokens_low_confidence.json",
        "tokenizer_path": "PipelineTest/res/results_LLaDa_64steps_64tokens_low_confidence/tokenizer_LLaDa_64steps_64tokens_low_confidence.pt",
        "name": "LLaDA_64steps_64tokens",
    },
    "dream": {
        "outputs_path": "PipelineTest/res/DREAM_64steps_64tokens_maskgit/outputs_DREAM_64steps_64tokens_maskgit.pt",
        "eval_json": "PipelineTest/res/eval/results_triviaqa_DREAM_64steps_64tokens_maskgit.json",
        "tokenizer_path": "PipelineTest/res/DREAM_64steps_64tokens_maskgit/tokenizer_DREAM_64steps_64tokens_maskgit.pt",
        "name": "DREAM_64steps_64tokens",
    },
    "dream16": {
        "outputs_path": "PipelineTest/res/results_DREAM_16steps_32tokens_maskgit/outputs_DREAM_16steps_32tokens_maskgit.pt",
        "eval_json": "PipelineTest/res/eval/results_triviaqa_DREAM_16steps_32tokens_maskgit.json",
        "tokenizer_path": "PipelineTest/res/results_DREAM_16steps_32tokens_maskgit/tokenizer_DREAM_16steps_32tokens_maskgit.pt",
        "name": "DREAM_16steps_32tokens",
    },
    "dream128": {
        "outputs_path": "PipelineTest/res/results_DREAM_128steps_128tokens_maskgit/outputs_DREAM_128steps_128tokens_maskgit.pt",
        "eval_json": "PipelineTest/res/eval/results_triviaqa_DREAM_128steps_128tokens_maskgit.json",
        "tokenizer_path": "PipelineTest/res/results_DREAM_128steps_128tokens_maskgit/tokenizer_DREAM_128steps_128tokens_maskgit.pt",
        "name": "DREAM_128steps_128tokens",
    },
    "dream64_800samples": {
        "outputs_path": "PipelineTest/res/results_DREAM_64steps_64tokens_maskgit_800samples/outputs_DREAM_64steps_64tokens_maskgit_800samples.pt",
        "eval_json": "PipelineTest/res/eval/results_triviaqa_DREAM_64steps_64tokens_maskgit_800samples.json",
        "tokenizer_path": "PipelineTest/res/results_DREAM_64steps_64tokens_maskgit_800samples/tokenizer_DREAM_64steps_64tokens_maskgit_800samples.pt",
        "name": "DREAM_64steps_64tokens_800samples",
    },
    "llada64_800samples": {
        "outputs_path": "PipelineTest/res/results_LLaDa_64steps_64tokens_low_confidence_800samples/outputs_LLaDa_64steps_64tokens_low_confidence_800samples.pt",
        "eval_json": "PipelineTest/res/eval/results_triviaqa_LLaDa_64steps_64tokens_low_confidence_800samples.json",
        "tokenizer_path": "PipelineTest/res/results_LLaDa_64steps_64tokens_low_confidence_800samples/tokenizer_LLaDa_64steps_64tokens_low_confidence_800samples.pt",
        "name": "LLaDA_64steps_64tokens_800samples",
    },
    "dream64_400samples": {
        "outputs_path": "PipelineTest/res/results_DREAM_64steps_64tokens_maskgit_400samples/outputs_DREAM_64steps_64tokens_maskgit_400samples.pt",
        "eval_json": "PipelineTest/res/eval/results_triviaqa_DREAM_64steps_64tokens_maskgit_400samples.json",
        "tokenizer_path": "PipelineTest/res/results_DREAM_64steps_64tokens_maskgit_400samples/tokenizer_DREAM_64steps_64tokens_maskgit_400samples.pt",
        "name": "DREAM_64steps_64tokens_400samples",
    },
    "llada64_400samples": {
        "outputs_path": "PipelineTest/res/results_LLaDa_64steps_64tokens_low_confidence_400samples/outputs_LLaDa_64steps_64tokens_low_confidence_400samples.pt",
        "eval_json": "PipelineTest/res/eval/results_triviaqa_LLaDa_64steps_64tokens_low_confidence_400samples.json",
        "tokenizer_path": "PipelineTest/res/results_LLaDa_64steps_64tokens_low_confidence_400samples/tokenizer_LLaDa_64steps_64tokens_low_confidence_400samples.pt",
        "name": "LLaDA_64steps_64tokens_400samples",
    },
    "llada128_400samples": {
        "outputs_path": "PipelineTest/res/results_LLaDa_128steps_128tokens_low_confidence_400samples/outputs_LLaDa_128steps_128tokens_low_confidence_400samples.pt",
        "eval_json": "PipelineTest/res/eval/results_triviaqa_LLaDa_128steps_128tokens_low_confidence_400samples.json",
        "tokenizer_path": "PipelineTest/res/results_LLaDa_128steps_128tokens_low_confidence_400samples/tokenizer_LLaDa_128steps_128tokens_low_confidence_400samples.pt",
        "name": "LLaDA_128steps_128tokens_400samples",
    },
}


def load_data(config_name):
    """Load outputs and match with eval labels."""
    cfg = CONFIGS[config_name]
    outputs = torch.load(cfg["outputs_path"], map_location="cpu", weights_only=False)

    with open(cfg["eval_json"], "r") as f:
        eval_data = json.load(f)

    ent_list = outputs.histories_entropy
    mask_list = outputs.histories_mask

    B = ent_list[0].shape[0]
    T = len(ent_list)
    D = ent_list[0].shape[1]

    entropies = torch.stack(ent_list, dim=1).float().numpy()
    masks = torch.stack(mask_list, dim=1).float().numpy()

    if hasattr(outputs, "start_idx_history") and outputs.start_idx_history is not None:
        start_indices = outputs.start_idx_history
    else:
        start_indices = [0] * B

    max_new = outputs.max_new_tokens if hasattr(outputs, "max_new_tokens") and outputs.max_new_tokens else D

    gen_entropies = np.zeros((B, T, max_new))
    gen_masks = np.zeros((B, T, max_new))
    for i in range(B):
        s = start_indices[i] if isinstance(start_indices, list) else start_indices
        e = s + max_new
        if e <= D:
            gen_entropies[i] = entropies[i, :, s:e]
            gen_masks[i] = masks[i, :, s:e]

    labels = np.array([1 if item.get("is_hallucination", "no") == "yes" else 0 for item in eval_data[:B]])
    return gen_entropies, gen_masks, labels, cfg


# =========================================================================
# AR(1) fitting
# =========================================================================

def fit_ar1_population(entropies, masks, indices):
    """
    Fit AR(1) per (step, token): X(t,d) = phi(t,d)*X(t-1,d) + intercept(t,d).
    Returns phi, intercept of shape (T, D).
    """
    ent = entropies[indices]
    N, T, D = ent.shape
    phi = np.zeros((T, D))
    intercept = np.zeros((T, D))
    for t in range(1, T):
        for d in range(D):
            x_prev = ent[:, t - 1, d]
            x_curr = ent[:, t, d]
            if np.std(x_prev) < 1e-12:
                continue
            slope, intcpt, _, _, _ = linregress(x_prev, x_curr)
            phi[t, d] = slope
            intercept[t, d] = intcpt
    return phi, intercept


def ar1_population_stats(entropies, masks, indices, phi, intercept):
    """
    Population-level mean and var per step for data and AR1 one-step-ahead
    reconstruction, averaged over all samples in `indices`.
    """
    ent = entropies[indices]
    msk = masks[indices]
    N, T, D = ent.shape

    mean_data = np.zeros(T)
    var_data = np.zeros(T)
    mean_recon = np.zeros(T)
    var_recon = np.zeros(T)

    for t in range(T):
        data_means, data_vars = [], []
        recon_means, recon_vars = [], []
        for n in range(N):
            m = msk[n, t, :] > 0
            nm = int(np.sum(m))
            if nm == 0:
                continue
            vals = ent[n, t, m]
            data_means.append(np.mean(vals))
            if nm > 1:
                data_vars.append(np.var(vals))

            if t == 0:
                recon_means.append(np.mean(vals))
                if nm > 1:
                    recon_vars.append(np.var(vals))
            else:
                pred = phi[t, :] * ent[n, t - 1, :] + intercept[t, :]
                pred_masked = pred[m]
                recon_means.append(np.mean(pred_masked))
                if nm > 1:
                    recon_vars.append(np.var(pred_masked))

        mean_data[t] = np.mean(data_means) if data_means else 0
        var_data[t] = np.mean(data_vars) if data_vars else 0
        mean_recon[t] = np.mean(recon_means) if recon_means else 0
        var_recon[t] = np.mean(recon_vars) if recon_vars else 0

    return mean_data, var_data, mean_recon, var_recon


# =========================================================================
# Main
# =========================================================================

def main(config_name):
    print(f"\n[AR1 global] Loading data ({config_name})...")
    entropies, masks, labels, cfg = load_data(config_name)

    correct_idx = np.where(labels == 0)[0]
    halluc_idx = np.where(labels == 1)[0]
    print(f"  Correct: {len(correct_idx)}, Hallucination: {len(halluc_idx)}")

    T = entropies.shape[1]
    steps = np.arange(T)
    cividis = plt.get_cmap("cividis")
    data_color = cividis(0.15)
    correct_model_color = cividis(0.5)
    halluc_model_color = cividis(0.85)

    print("[AR1 global] Fitting AR(1) models...")
    phi_c, int_c = fit_ar1_population(entropies, masks, correct_idx)
    phi_h, int_h = fit_ar1_population(entropies, masks, halluc_idx)

    # Same-model reconstruction
    mean_data_c, var_data_c, mean_recon_cc, var_recon_cc = ar1_population_stats(
        entropies, masks, correct_idx, phi_c, int_c
    )
    mean_data_h, var_data_h, mean_recon_hh, var_recon_hh = ar1_population_stats(
        entropies, masks, halluc_idx, phi_h, int_h
    )
    # Cross-model reconstruction
    _, _, mean_recon_hc, var_recon_hc = ar1_population_stats(
        entropies, masks, halluc_idx, phi_c, int_c
    )
    _, _, mean_recon_ch, var_recon_ch = ar1_population_stats(
        entropies, masks, correct_idx, phi_h, int_h
    )

    # --- Plot ---
    fig, axes = plt.subplots(2, 2, figsize=(18, 12))

    # Top-left: Mean — Correct samples
    ax = axes[0, 0]
    ax.plot(steps, mean_data_c, "o-", color=data_color, markersize=6, label="Data (correct)")
    ax.plot(steps, mean_recon_cc, "s--", color=correct_model_color, markersize=6, label="AR1 correct model")
    ax.plot(steps, mean_recon_ch, "^--", color=halluc_model_color, markersize=6, label="AR1 halluc model")
    ax.set_title("Mean Masked Entropy — Correct samples", fontweight="bold", fontsize=24)
    ax.set_xlabel("Diffusion Step", fontsize=20)
    ax.set_ylabel("Mean Entropy", fontsize=20)
    ax.legend(fontsize=20)
    ax.grid(True, alpha=0.3)

    # Top-right: Mean — Halluc samples
    ax = axes[0, 1]
    ax.plot(steps, mean_data_h, "o-", color=data_color, markersize=6, label="Data (halluc)")
    ax.plot(steps, mean_recon_hh, "^--", color=halluc_model_color, markersize=6, label="AR1 halluc model")
    ax.plot(steps, mean_recon_hc, "s--", color=correct_model_color, markersize=6, label="AR1 correct model")
    ax.set_title("Mean Masked Entropy — Halluc samples", fontweight="bold", fontsize=24)
    ax.set_xlabel("Diffusion Step", fontsize=20)
    ax.set_ylabel("Mean Entropy", fontsize=20)
    ax.legend(fontsize=20)
    ax.grid(True, alpha=0.3)

    # Bottom-left: Var — Correct samples
    ax = axes[1, 0]
    ax.plot(steps, var_data_c, "o-", color=data_color, markersize=6, label="Data (correct)")
    ax.plot(steps, var_recon_cc, "s--", color=correct_model_color, markersize=6, label="AR1 correct model")
    ax.plot(steps, var_recon_ch, "^--", color=halluc_model_color, markersize=6, label="AR1 halluc model")
    ax.set_title("Var Masked Entropy — Correct samples", fontweight="bold", fontsize=24)
    ax.set_xlabel("Diffusion Step", fontsize=20)
    ax.set_ylabel("Variance", fontsize=20)
    ax.legend(fontsize=20)
    ax.grid(True, alpha=0.3)

    # Bottom-right: Var — Halluc samples
    ax = axes[1, 1]
    ax.plot(steps, var_data_h, "o-", color=data_color, markersize=6, label="Data (halluc)")
    ax.plot(steps, var_recon_hh, "^--", color=halluc_model_color, markersize=6, label="AR1 halluc model")
    ax.plot(steps, var_recon_hc, "s--", color=correct_model_color, markersize=6, label="AR1 correct model")
    ax.set_title("Var Masked Entropy — Halluc samples", fontweight="bold", fontsize=24)
    ax.set_xlabel("Diffusion Step", fontsize=20)
    ax.set_ylabel("Variance", fontsize=20)
    ax.legend(fontsize=20)
    ax.grid(True, alpha=0.3)

    # plt.suptitle(
    #     f"AR(1) Population-Level: Data vs Reconstruction — {cfg['name']}",
    #     fontweight="bold", fontsize=24,
    # )
    plt.tight_layout()
    save_dir = os.path.join(os.path.dirname(cfg["outputs_path"]), "plots")
    os.makedirs(save_dir, exist_ok=True)
    path = os.path.join(save_dir, "ar1_population_mean_var.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"    Saved: {path}")


def plot(config_name="llada"):
    """Entry point for run_analysis dispatch."""
    main(config_name)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=next(iter(CONFIGS)), choices=list(CONFIGS.keys()))
    args = parser.parse_args()
    main(args.config)
