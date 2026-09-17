"""
features.markovian
==================
Dynamic features motivated by the Markovian/OU modelisation of entropy decay.

Features:
- alpha/beta from exponential decay fit: ln H(t) ~ alpha*t + beta
- Top-k averaged decay
- Parametric trajectory fit (C, tau, m)
- Finite-difference rate of change (MeanTau, VarTau)
- Shape features on the masked variance curve (AUC, max, argmax, skewness, kurtosis, curvature)
"""

import numpy as np
from scipy.optimize import curve_fit

from GenerateBaseSamplerOutputsAndExtractInfo.GetInfoFromBaseSamplerOutput import (
    getEntropy, getLogProbs,
)


# =========================================================================
# EXPONENTIAL DECAY FIT (alpha, beta)
# =========================================================================

def alpha_beta_entropy(outputs):
    """Fit ln H(t,d) = alpha_d * t + beta_d per token, return mean alpha, mean beta."""
    entropies = getEntropy(outputs)
    alpha_mean_list, beta_mean_list = [], []
    alpha_var_list, beta_var_list = [], []
    for i in range(len(entropies)):
        ent = np.asarray(entropies[i], dtype=float)
        ln_ent = np.log(ent + 1e-10)
        n_steps, n_tokens = ent.shape
        t = np.arange(n_steps)
        A = np.vstack([t, np.ones(n_steps)]).T
        alpha_i, beta_i = [], []
        for j in range(n_tokens):
            a, b = np.linalg.lstsq(A, ln_ent[:, j], rcond=None)[0]
            alpha_i.append(a)
            beta_i.append(b)
        alpha_mean_list.append(np.mean(alpha_i))
        beta_mean_list.append(np.mean(beta_i))
        alpha_var_list.append(np.var(alpha_i))
        beta_var_list.append(np.var(beta_i))

    return np.array(alpha_mean_list), np.array(beta_mean_list), np.array(alpha_var_list), np.array(beta_var_list)


def alpha_beta_entropy_avg(outputs, k_tokens=1):
    """Fit on top-k tokens (by mean entropy) averaged trajectory."""
    entropies = getEntropy(outputs)
    alpha_list, beta_list = [], []
    for i in range(len(entropies)):
        ent = np.asarray(entropies[i], dtype=float)
        mean_ent = np.mean(ent, axis=0)
        if k_tokens:
            indices = np.argsort(mean_ent)[-k_tokens:]
            mean_ent_traj = np.mean(ent[:, indices], axis=1)
        else:
            mean_ent_traj = np.mean(ent, axis=1)
        ln_mean_ent = np.log(mean_ent_traj + 1e-10)
        n_steps = len(mean_ent_traj)
        t = np.arange(n_steps)
        A = np.vstack([t, np.ones(n_steps)]).T
        a, b = np.linalg.lstsq(A, ln_mean_ent, rcond=None)[0]
        alpha_list.append(a)
        beta_list.append(b)
    return np.array(alpha_list), np.array(beta_list)


def alpha_beta_logprob(outputs):
    """Fit ln logprob(t,d) = alpha_d * t + beta_d per token."""
    logprobs = getLogProbs(outputs)
    alpha_mean_list, beta_mean_list = [], []
    alpha_var_list, beta_var_list = [], []

    for i in range(len(logprobs)):
        logp = np.asarray(logprobs[i], dtype=float)
        ln_logp = np.log(logp + 1e-10)
        n_steps, n_tokens = logp.shape
        t = np.arange(n_steps)
        A = np.vstack([t, np.ones(n_steps)]).T
        alpha_i, beta_i = [], []
        for j in range(n_tokens):
            a, b = np.linalg.lstsq(A, ln_logp[:, j], rcond=None)[0]
            alpha_i.append(a)
            beta_i.append(b)
        alpha_mean_list.append(np.mean(alpha_i))
        beta_mean_list.append(np.mean(beta_i))
        alpha_var_list.append(np.var(alpha_i))
        beta_var_list.append(np.var(beta_i))
    return np.array(alpha_mean_list), np.array(beta_mean_list), np.array(alpha_var_list), np.array(beta_var_list)


# =========================================================================
# PARAMETRIC TRAJECTORY FIT: H(t) = C * t * exp(-(t-m)/tau)
# =========================================================================

def parametric_fit_entropy(outputs, k_tokens=20):
    """Fit H(t) = C * t * exp(-(t-m)/tau) on averaged top-k entropy trajectories.
    
    Returns: C, tau, m arrays of shape (N,)
    """
    def model(t, c, tau, m):
        return c * t * np.exp(-(t - m) / tau)

    entropies = getEntropy(outputs)
    list_c, list_tau, list_m = [], [], []

    for i in range(len(entropies)):
        ent = np.asarray(entropies[i], dtype=float)
        mean_ent_per_token = np.mean(ent, axis=0)

        if k_tokens:
            indices = np.argsort(mean_ent_per_token)[-k_tokens:]
            mean_ent = np.mean(ent[:, indices], axis=1)
        else:
            mean_ent = np.mean(ent, axis=1)

        t = np.arange(len(mean_ent), dtype=float)
        valid = np.isfinite(mean_ent) & (mean_ent > 0)
        t_fit, y_fit = t[valid], mean_ent[valid]

        if len(y_fit) < 3:
            list_c.append(np.nan); list_tau.append(np.nan); list_m.append(np.nan)
            continue

        c0 = float(max(y_fit.max(), 1e-3))
        tau0 = float(max(len(y_fit) / 4.0, 1.0))
        m0 = float(len(y_fit) / 2.0)

        try:
            params, _ = curve_fit(
                model, t_fit, y_fit, p0=[c0, tau0, m0],
                bounds=([0.0, 1e-6, -len(y_fit)], [np.inf, np.inf, 2*len(y_fit)]),
                maxfev=20000,
            )
            c_i, tau_i, m_i = params
        except Exception:
            c_i, tau_i, m_i = np.nan, np.nan, np.nan

        list_c.append(float(c_i))
        list_tau.append(float(tau_i))
        list_m.append(float(m_i))

    return np.array(list_c), np.array(list_tau), np.array(list_m)




# =========================================================================
# COMBINED EXTRACTION
# =========================================================================

def get_markovian_features(outputs, k_tokens=None):
    """Extract all Markovian dynamic features.
    
    Returns:
        features: np.ndarray (N, F)
        feature_names: list of str
    """
    alpha_ent, beta_ent, alpha_ent_var, beta_ent_var = alpha_beta_entropy(outputs)
    alpha_ent_avg, beta_ent_avg = alpha_beta_entropy_avg(outputs, k_tokens=k_tokens)
    alpha_lp, beta_lp, alpha_lp_var, beta_lp_var = alpha_beta_logprob(outputs)
    C, tau, m = parametric_fit_entropy(outputs, k_tokens=k_tokens)
    

    feats = {
        "AlphaEntropy": alpha_ent,
        "BetaEntropy": beta_ent,
        # "AlphaEntropyVar": alpha_ent_var,
        # "BetaEntropyVar": beta_ent_var,
        "AlphaEntropyAvg": alpha_ent_avg,
        "BetaEntropyAvg": beta_ent_avg,
        "AlphaLogProb": alpha_lp,
        "BetaLogProb": beta_lp,
        # "AlphaLogProbVar": alpha_lp_var,
        # "BetaLogProbVar": beta_lp_var,
        "C_fit": C,
        "Tau_fit": tau,
        "M_fit": m,
       
    }

    feature_names = list(feats.keys())
    features = np.column_stack([feats[k] for k in feature_names])
    return features, feature_names
