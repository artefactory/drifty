"""
features.ar1
============
AR(1) model fitting and feature extraction.

The AR(1) model discretises the time-inhomogeneous Ornstein-Uhlenbeck process:
    H_{t,d} = phi_{t,d} * H_{t-1,d} + c_{t,d} + sigma_{t,d} * eps

Two separate models are fitted on the training half: one for correct samples,
one for hallucinated samples. Features are then derived from the predictions
of each model on the test half.
"""

import numpy as np
from scipy.stats import linregress
from sklearn.model_selection import train_test_split

from GenerateBaseSamplerOutputsAndExtractInfo.GetInfoFromBaseSamplerOutput import (
    getEntropy, getEachStepMask,
)
from PipelineTest.features.utils import compute_padding_mask


# =========================================================================
# AR(1) MODEL FITTING
# =========================================================================

def fit_ar1_model(entropy_tensor):
    """
    Fit AR(1) per (step, token) on a population of samples.
    
    Args:
        entropy_tensor: (N, T, D) array of entropy trajectories
        
    Returns:
        phi: (T, D) autoregressive coefficient
        intercept: (T, D) intercept
        sigma: (T, D) residual std
    """
    N, T, D = entropy_tensor.shape
    phi = np.zeros((T, D))
    intercept = np.zeros((T, D))
    sigma = np.zeros((T, D))
    for t in range(1, T):
        X_prev = entropy_tensor[:, t-1, :]
        X_curr = entropy_tensor[:, t, :]
        
        for d in range(D):
            slope, intcpt, _, _, _ = linregress(X_prev[:, d], X_curr[:, d])
            phi[t, d] = slope
            intercept[t, d] = intcpt
            sigma[t, d] = np.std(X_curr[:, d] - (slope * X_prev[:, d] + intcpt))
    return phi, intercept, sigma


def fit_ar1_model_no_padding(entropy_tensor, padding_2d):
    """
    Fit AR(1) per (step, token), excluding padding tokens.
    
    Args:
        entropy_tensor: (N, T, D) array
        padding_2d: (N, D) bool array where True = padding token
        
    Returns:
        phi, intercept, sigma: (T, D) arrays
    """
    N, T, D = entropy_tensor.shape
    phi = np.zeros((T, D))
    intercept = np.zeros((T, D))
    sigma = np.zeros((T, D))
    for t in range(1, T):
        X_prev = entropy_tensor[:, t-1, :]
        X_curr = entropy_tensor[:, t, :]
        for d in range(D):

            valid = ~padding_2d[:, d]
            n_valid = np.sum(valid)
            print(t, d, n_valid)
            if n_valid <= 3:
                continue
            x = X_prev[valid, d]
            y = X_curr[valid, d]
            slope, intcpt, _, _, _ = linregress(x, y)
            phi[t, d] = slope
            intercept[t, d] = intcpt
            sigma[t, d] = np.std(y - (slope * x + intcpt))
    return phi, intercept, sigma


# =========================================================================
# AR(1) FEATURE COMPUTATION
# =========================================================================

def compute_mean_prediction(entropy_test, effective_mask, phi, intercept):
    """
    Per-sample mean of AR(1) one-step-ahead predictions on masked tokens.
    
    Args:
        entropy_test: (N, T, D)
        effective_mask: (N, T, D) float, 1 = masked & non-padding
        phi, intercept: (T, D)
        
    Returns:
        (N,) array of mean predicted entropy
    """
    N, T, D = entropy_test.shape
    sample_means = np.zeros(N)
    for n in range(N):
        total_pred = 0.0
        total_count = 0
        for t in range(1, T):
            x_prev = entropy_test[n, t-1, :]
            mask_t = effective_mask[n, t, :]
            x_pred = phi[t, :] * x_prev + intercept[t, :]
            total_pred += np.sum(x_pred * mask_t)
            total_count += np.sum(mask_t)
        sample_means[n] = total_pred / max(total_count, 1)
    return sample_means


def compute_var_prediction(entropy_test, effective_mask, phi, intercept):
    """
    Per-sample mean inter-token variance of AR(1) predictions at each step.
    
    Returns:
        (N,) array
    """
    N, T, D = entropy_test.shape
    sample_vars = np.zeros(N)
    for n in range(N):
        total_var = 0.0
        count_steps = 0
        for t in range(1, T):
            x_prev = entropy_test[n, t-1, :]
            mask_t = effective_mask[n, t, :]
            n_masked = int(np.sum(mask_t))
            if n_masked < 2:
                continue
            x_pred = phi[t, :] * x_prev + intercept[t, :]
            pred_vals = x_pred[mask_t > 0]
            total_var += np.var(pred_vals)
            count_steps += 1
        sample_vars[n] = total_var / max(count_steps, 1)
    return sample_vars


# =========================================================================
# HIGH-LEVEL API
# =========================================================================

def fit_ar1_models(outputs, positions, labels, padding_2d=None):
    """
    Fit two AR(1) models (correct, hallucination) on the given data split.
    
    Args:
        outputs: merged sampler outputs
        positions: array of positions in outputs
        labels: binary labels (0=correct, 1=halluc)
        padding_2d: optional (N, D) bool, True=padding
        
    Returns:
        dict with keys 'correct' and 'halluc', each containing (phi, intercept, sigma)
    """
    entropies = getEntropy(outputs)

    correct_pos = positions[labels == 0]
    halluc_pos = positions[labels == 1]

    entropy_correct = np.array([entropies[i] for i in correct_pos])
    entropy_halluc = np.array([entropies[i] for i in halluc_pos])
    if padding_2d is not None:
        pad_correct = padding_2d[labels == 0]
        pad_halluc = padding_2d[labels == 1]
        phi_c, int_c, sig_c = fit_ar1_model_no_padding(entropy_correct, pad_correct)
        phi_h, int_h, sig_h = fit_ar1_model_no_padding(entropy_halluc, pad_halluc)
    else:
        phi_c, int_c, sig_c = fit_ar1_model(entropy_correct)
        phi_h, int_h, sig_h = fit_ar1_model(entropy_halluc)

    return {
        "correct": (phi_c, int_c, sig_c),
        "halluc": (phi_h, int_h, sig_h),
    }


def get_ar1_features(outputs, positions, effective_mask, ar1_models):
    """
    Compute AR(1)-derived features for a set of samples.
    
    Args:
        outputs: merged sampler outputs
        positions: array of positions
        effective_mask: (N, T, D) float mask (masked & non-padding = 1)
        ar1_models: dict from fit_ar1_models()
        
    Returns:
        features: (N, 4) array
        feature_names: list of 4 strings
    """
    entropies = getEntropy(outputs)
    entropy_arr = np.array([entropies[i] for i in positions])

    phi_c, int_c, _ = ar1_models["correct"]
    phi_h, int_h, _ = ar1_models["halluc"]

    mean_pred_correct = compute_mean_prediction(entropy_arr, effective_mask, phi_c, int_c)
    mean_pred_halluc = compute_mean_prediction(entropy_arr, effective_mask, phi_h, int_h)
    var_pred_correct = compute_var_prediction(entropy_arr, effective_mask, phi_c, int_c)
    var_pred_halluc = compute_var_prediction(entropy_arr, effective_mask, phi_h, int_h)

    features = np.column_stack([
        mean_pred_correct, mean_pred_halluc,
        var_pred_correct, var_pred_halluc,
    ])
    feature_names = [
        "MeanPred_correct", "MeanPred_halluc",
        "VarPred_correct", "VarPred_halluc",
    ]
    return features, feature_names
