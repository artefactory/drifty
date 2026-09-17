"""
features.baseline
=================
Baseline (non-dynamic) features: mean and variance of entropy/logprob.
These are simple aggregations over the diffusion trajectory.
"""

import numpy as np

from GenerateBaseSamplerOutputsAndExtractInfo.GetInfoFromBaseSamplerOutput import (
    getEntropy, getEachStepMask, getLogProbs, getEntropyJustUnmasked, getLogProbsJustUnmasked,
)
from PipelineTest.features.utils import compute_padding_mask, get_pad_token_id
from scipy.stats import skew, kurtosis
from scipy import integrate

# =========================================================================
# MEAN FEATURES
# =========================================================================

def mean_entropy(outputs):
    """Mean entropy across all steps and tokens."""
    entropies = getEntropy(outputs)
    return np.array([
        float(np.mean(np.asarray(entropies[i], dtype=float)))
        for i in range(len(entropies))
    ])


def mean_masked_entropy(outputs):
    """Mean entropy restricted to masked positions."""
    entropies = getEntropy(outputs)
    masks = getEachStepMask(outputs)
    result = []
    for i in range(len(entropies)):
        ent = np.asarray(entropies[i], dtype=float)
        mask = np.asarray(masks[i], dtype=float)
        masked_entropy = ent * mask
        result.append(float(np.sum(masked_entropy) / max(np.sum(mask), 1)))
    return np.array(result)


def mean_masked_entropy_no_padding(outputs, pad_token_id):
    """Mean entropy over masked non-padding tokens only."""
    positions = list(range(len(outputs.sample_indices)))
    padding_mask_all = compute_padding_mask(outputs, positions, pad_token_id)

    entropies = getEntropy(outputs)
    masks = getEachStepMask(outputs)
    result = []
    for i in range(len(entropies)):
        ent = np.asarray(entropies[i], dtype=float)
        mask = np.asarray(masks[i], dtype=bool)
        pad = padding_mask_all[i, 0, :]
        valid = mask & (~pad[np.newaxis, :])
        vals = ent[valid]
        result.append(float(np.mean(vals)) if vals.size > 0 else 0.0)
    return np.array(result)

def mean_entropy_just_unmasked(outputs):
    """Mean entropy at the step each token is unmasked (Flat global average)."""
    entropies = getEntropyJustUnmasked(outputs) # Contient maintenant une liste de tableaux 1D propres
    result = []
    for i in range(len(entropies)):
        vals = entropies[i]
        
        # Plus besoin de slice [1:] car le tableau ne contient aucun placeholder d'étape vide.
        # np.mean(vals) fait maintenant la vraie moyenne à plat : Somme(entropies) / 32
        result.append(float(np.mean(vals)) if vals.size > 0 else 0.0)
    return np.array(result)


def mean_logprob(outputs):
    """Mean log-probability across all steps and tokens."""
    logprobs = getLogProbs(outputs)
    return np.array([
        float(np.mean(np.asarray(logprobs[i], dtype=float)))
        for i in range(len(logprobs))
    ])

def mean_logprob_just_unmasked(outputs):
    """Mean log-probability at the step each token is unmasked (Flat global average)."""
    logprobs = getLogProbsJustUnmasked(outputs) # Contient maintenant une liste de tableaux 1D propres
    result = []
    for i in range(len(logprobs)):
        vals = logprobs[i]
        
        # Plus besoin de slice [1:] car le tableau ne contient aucun placeholder d'étape vide.
        # np.mean(vals) fait maintenant la vraie moyenne à plat : Somme(logprobs) / 32
        result.append(float(np.mean(vals)) if vals.size > 0 else 0.0)
    return np.array(result)

def mean_logprob_masked(outputs):
    """Mean log-probability restricted to masked positions."""
    logprobs = getLogProbs(outputs)
    masks = getEachStepMask(outputs)
    result = []
    for i in range(len(logprobs)):
        logp = np.asarray(logprobs[i], dtype=float)
        mask = np.asarray(masks[i], dtype=float)
        masked_logp = logp * mask
        result.append(float(np.sum(masked_logp) / max(np.sum(mask), 1)))
    return np.array(result)

def mean_logprob_masked_no_padding(outputs, pad_token_id):
    """Mean log-probability over masked non-padding tokens only."""
    positions = list(range(len(outputs.sample_indices)))
    padding_mask_all = compute_padding_mask(outputs, positions, pad_token_id)

    logprobs = getLogProbs(outputs)
    masks = getEachStepMask(outputs)
    result = []
    for i in range(len(logprobs)):
        logp = np.asarray(logprobs[i], dtype=float)
        mask = np.asarray(masks[i], dtype=bool)
        pad = padding_mask_all[i, 0, :]
        valid = mask & (~pad[np.newaxis, :])
        vals = logp[valid]
        result.append(float(np.mean(vals)) if vals.size > 0 else 0.0)
    return np.array(result)


# =========================================================================
# VARIANCE FEATURES
# =========================================================================

def var_entropy(outputs):
    """Variance of entropy across steps (per token), averaged over tokens."""
    entropies = getEntropy(outputs)
    return np.array([
        float(np.mean(np.var(np.asarray(entropies[i], dtype=float), axis=0)))
        for i in range(len(entropies))
    ])


def var_entropy_across_tokens(outputs):
    """Variance of entropy across tokens (per step), averaged over steps."""
    entropies = getEntropy(outputs)
    return np.array([
        float(np.mean(np.var(np.asarray(entropies[i], dtype=float), axis=1)))
        for i in range(len(entropies))
    ])


def var_masked_entropy(outputs):
    """Variance of entropy across masked steps only (per token), averaged."""
    entropies = getEntropy(outputs)
    masks = getEachStepMask(outputs)
    result = []
    for i in range(len(entropies)):
        ent = np.asarray(entropies[i], dtype=float)
        mask = np.asarray(masks[i], dtype=bool)
        token_vars = []
        for j in range(ent.shape[1]):
            masked_vals = ent[mask[:, j], j]
            if len(masked_vals) > 1:
                token_vars.append(np.var(masked_vals))
        result.append(float(np.mean(token_vars)) if token_vars else 0.0)
    return np.array(result)


def var_masked_entropy_across_tokens(outputs):
    """Variance of entropy among masked tokens at each step, averaged over steps."""
    entropies = getEntropy(outputs)
    masks = getEachStepMask(outputs)
    result = []
    for i in range(len(entropies)):
        ent = np.asarray(entropies[i], dtype=float)
        mask = np.asarray(masks[i], dtype=bool)
        step_vars = []
        for s in range(ent.shape[0]):
            masked_vals = ent[s, mask[s]]
            if len(masked_vals) > 1:
                step_vars.append(np.var(masked_vals))
        result.append(float(np.mean(step_vars)) if step_vars else 0.0)
    return np.array(result)


def var_masked_entropy_across_tokens_no_padding(outputs, pad_token_id):
    """Variance of entropy across masked non-padding tokens, averaged over steps."""
    positions = list(range(len(outputs.sample_indices)))
    padding_mask_all = compute_padding_mask(outputs, positions, pad_token_id)

    entropies = getEntropy(outputs)
    masks = getEachStepMask(outputs)
    result = []
    for i in range(len(entropies)):
        ent = np.asarray(entropies[i], dtype=float)
        mask = np.asarray(masks[i], dtype=bool)
        pad = padding_mask_all[i, 0, :]
        step_vars = []
        for s in range(ent.shape[0]):
            valid = mask[s] & (~pad)
            masked_vals = ent[s, valid]
            if len(masked_vals) > 1:
                step_vars.append(np.var(masked_vals))
        result.append(float(np.mean(step_vars)) if step_vars else 0.0)
    return np.array(result)


def var_entropy_just_unmasked(outputs):
    """Variance of entropy at unmasking across steps."""
    entropies = getEntropyJustUnmasked(outputs)
    result = []
    for i in range(len(entropies)):
        values = np.asarray(entropies[i], dtype=float)[1:]
        result.append(float(np.var(values)))
    return np.array(result)


def var_logprob(outputs):
    """Variance of log-probabilities across steps (per token), averaged."""
    logprobs = getLogProbs(outputs)
    return np.array([
        float(np.mean(np.var(np.asarray(logprobs[i], dtype=float), axis=0)))
        for i in range(len(logprobs))
    ])

def var_logprob_across_tokens(outputs):
    """Variance of log-probabilities across tokens (per step), averaged."""
    logprobs = getLogProbs(outputs)
    return np.array([
        float(np.mean(np.var(np.asarray(logprobs[i], dtype=float), axis=1)))
        for i in range(len(logprobs))
    ])


def var_logprob_masked(outputs):
    """Variance of log-probabilities across masked steps (per token), averaged."""
    logprobs = getLogProbs(outputs)
    masks = getEachStepMask(outputs)
    result = []
    for i in range(len(logprobs)):
        logp = np.asarray(logprobs[i], dtype=float)
        mask = np.asarray(masks[i], dtype=bool)
        token_vars = []
        for j in range(logp.shape[1]):
            masked_vals = logp[mask[:, j], j]
            if len(masked_vals) > 1:
                token_vars.append(np.var(masked_vals))
        result.append(float(np.mean(token_vars)) if token_vars else 0.0)
    return np.array(result)

def var_logprob_masked_across_tokens(outputs):
    """Variance of log-probabilities among masked tokens (per step), averaged."""
    logprobs = getLogProbs(outputs)
    masks = getEachStepMask(outputs)
    result = []
    for i in range(len(logprobs)):
        logp = np.asarray(logprobs[i], dtype=float)
        mask = np.asarray(masks[i], dtype=bool)
        step_vars = []
        for s in range(logp.shape[0]):
            masked_vals = logp[s, mask[s]]
            if len(masked_vals) > 1:
                step_vars.append(np.var(masked_vals))
        result.append(float(np.mean(step_vars)) if step_vars else 0.0)
    return np.array(result)

def var_logprob_masked_across_tokens_no_padding(outputs, pad_token_id):
    """Variance of log-probabilities across masked non-padding tokens, averaged."""
    positions = list(range(len(outputs.sample_indices)))
    padding_mask_all = compute_padding_mask(outputs, positions, pad_token_id)

    logprobs = getLogProbs(outputs)
    masks = getEachStepMask(outputs)
    result = []
    for i in range(len(logprobs)):
        logp = np.asarray(logprobs[i], dtype=float)
        mask = np.asarray(masks[i], dtype=bool)
        pad = padding_mask_all[i, 0, :]
        step_vars = []
        for s in range(logp.shape[0]):
            valid = mask[s] & (~pad)
            masked_vals = logp[s, valid]
            if len(masked_vals) > 1:
                step_vars.append(np.var(masked_vals))
        result.append(float(np.mean(step_vars)) if step_vars else 0.0)
    return np.array(result)

def var_logprob_just_unmasked(outputs):
    """Variance of log-probabilities at unmasking across steps."""
    logprobs = getLogProbsJustUnmasked(outputs)
    result = []
    for i in range(len(logprobs)):
        values = np.asarray(logprobs[i], dtype=float)[1:]
        result.append(float(np.var(values)))
    return np.array(result)



# =========================================================================
# FINITE-DIFFERENCE RATE OF CHANGE (Tau)
# =========================================================================

def mean_var_tau_entropy(outputs, window=5):
    """Discrete rate of change of step-averaged entropy."""
    entropies = getEntropy(outputs)
    list_mean_tau, list_var_tau = [], []
    for i in range(len(entropies)):
        ent = np.asarray(entropies[i], dtype=float)
        ent_avg = np.mean(ent, axis=1)
        n_steps = len(ent_avg)
        t_indices = np.arange(0, n_steps - window, window)
        tau = (ent_avg[t_indices + window] - ent_avg[t_indices]) / window
        list_mean_tau.append(np.mean(tau))
        list_var_tau.append(np.var(tau))
    return np.array(list_mean_tau), np.array(list_var_tau)


# =========================================================================
# SHAPE FEATURES ON MASKED VARIANCE CURVE V(t)
# =========================================================================

def _masked_entropy_across_tokens_curve(outputs):
    """Compute V(t) = Var_d(H(t,d) | masked) for each sample."""
    entropies = getEntropy(outputs)
    masks = getEachStepMask(outputs)
    all_curves = []
    for i in range(len(entropies)):
        ent = np.asarray(entropies[i], dtype=float)
        mask = np.asarray(masks[i], dtype=bool)
        step_vars = []
        for s in range(ent.shape[0]):
            masked_vals = ent[s, mask[s]]
            if len(masked_vals) > 1:
                step_vars.append(np.var(masked_vals))
        all_curves.append(step_vars)
    return all_curves


def auc_masked_entropy_curve(outputs):
    """AUC (integral) of the V(t) curve."""
    curves = _masked_entropy_across_tokens_curve(outputs)
    return np.array([integrate.trapezoid(c) for c in curves])


def max_masked_entropy_curve(outputs):
    """Maximum of V(t) curve."""
    curves = _masked_entropy_across_tokens_curve(outputs)
    return np.array([np.max(c) for c in curves])


def argmax_masked_entropy_curve(outputs):
    """Step of maximum V(t)."""
    curves = _masked_entropy_across_tokens_curve(outputs)
    return np.array([np.argmax(c) for c in curves])


def skewness_masked_entropy_curve(outputs):
    """Skewness of V(t) distribution."""
    curves = _masked_entropy_across_tokens_curve(outputs)
    return np.array([skew(c) for c in curves])


def kurtosis_masked_entropy_curve(outputs):
    """Kurtosis of V(t) distribution."""
    curves = _masked_entropy_across_tokens_curve(outputs)
    return np.array([kurtosis(c) for c in curves])


def mean_curvature_masked_entropy_curve(outputs):
    """Mean curvature of V(t) curve."""
    curves = _masked_entropy_across_tokens_curve(outputs)
    result = []
    for c in curves:
        y = np.asarray(c)
        x = np.arange(len(y))
        dy = np.gradient(y, x)
        d2y = np.gradient(dy, x)
        curvature = np.abs(d2y) / (1 + dy**2)**(3/2)
        result.append(np.mean(curvature))
    return np.array(result)



# =========================================================================
# COMBINED EXTRACTION
# =========================================================================

def get_baseline_features(outputs, pad_token_id=None):
    """Extract all baseline features.
    
    Args:
        outputs: merged sampler outputs
        pad_token_id: if provided, also computes no-padding variants
        
    Returns:
        features: np.ndarray (N, F)
        feature_names: list of str
    """
    feats = {
        "MeanEntropy": mean_entropy(outputs),
        "MeanMaskedEntropy": mean_masked_entropy(outputs),
        "MeanEntropyJustUnmasked": mean_entropy_just_unmasked(outputs),
        "MeanLogProb": mean_logprob(outputs),
        "MeanLogProbMasked": mean_logprob_masked(outputs),
        "MeanLogProbJustUnmasked": mean_logprob_just_unmasked(outputs),
        "VarEntropy": var_entropy(outputs),
        "VarEntropyAcrossTokens": var_entropy_across_tokens(outputs),
        "VarMaskedEntropy": var_masked_entropy(outputs),
        "VarMaskedEntropyAcrossTokens": var_masked_entropy_across_tokens(outputs),
        "VarEntropyJustUnmasked": var_entropy_just_unmasked(outputs),
        "VarLogProb": var_logprob(outputs),
        "VarLogProbAcrossTokens": var_logprob_across_tokens(outputs),
        "VarLogProbMasked": var_logprob_masked(outputs),
        "VarLogProbMaskedAcrossTokens": var_logprob_masked_across_tokens(outputs),
        "VarLogProbJustUnmasked": var_logprob_just_unmasked(outputs),
        "MeanVarTauEntropy": mean_var_tau_entropy(outputs)[0],
        "VarTauEntropy": mean_var_tau_entropy(outputs)[1],
        "AUCVarMaskedEntropyCurve": auc_masked_entropy_curve(outputs),
        "MaxVarMaskedEntropyCurve": max_masked_entropy_curve(outputs),
        "ArgmaxVarMaskedEntropyCurve": argmax_masked_entropy_curve(outputs),
        "SkewnessVarMaskedEntropyCurve": skewness_masked_entropy_curve(outputs),
        "KurtosisVarMaskedEntropyCurve": kurtosis_masked_entropy_curve(outputs),
        "MeanCurvatureMaskedEntropyCurve": mean_curvature_masked_entropy_curve(outputs),
    }

    if pad_token_id is not None:
        feats["MeanMaskedEntropyNoPad"] = mean_masked_entropy_no_padding(outputs, pad_token_id)
        feats["VarMaskedEntropyAcrossTokensNoPad"] = var_masked_entropy_across_tokens_no_padding(outputs, pad_token_id)
        feats["MeanLogProbMaskedNoPad"] = mean_logprob_masked_no_padding(outputs, pad_token_id)
        feats["VarLogProbMaskedAcrossTokensNoPad"] = var_logprob_masked_across_tokens_no_padding(outputs, pad_token_id)

    feature_names = list(feats.keys())
    features = np.column_stack([feats[k] for k in feature_names])
    return features, feature_names
