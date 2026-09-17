"""Stability Selection (Meinshausen & Bühlmann, 2010).

Repeatedly sub-samples the data and fits a randomized L1-penalized model.
Features consistently selected across sub-samples are retained.

Theoretical bound on expected false positives:
    E[V] <= q^2 / ((2*pi_thr - 1) * p)
where q = avg number of selected features per sub-sample,
      pi_thr = selection probability threshold,
      p = total number of features.
"""

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler


def stability_selection(X, y, feature_names, n_bootstrap=500, sample_fraction=0.5,
                        lambda_grid=None, pi_threshold=0.6, random_state=42):
    """Run stability selection with randomized Lasso (logistic).

    Args:
        X: (N, p) feature matrix (raw, will be standardized internally).
        y: (N,) binary labels.
        feature_names: list[str] of length p.
        n_bootstrap: number of sub-sampling rounds.
        sample_fraction: fraction of samples drawn each round.
        lambda_grid: list of C values for LogisticRegression (inverse regularization).
                     If None, uses a log-spaced grid.
        pi_threshold: selection probability cutoff (0.6–0.9 recommended).
        random_state: seed for reproducibility.

    Returns:
        selected_names: list[str] — features with Pi_j > pi_threshold.
        selection_probs: dict[str, float] — Pi_j for every feature.
        expected_fp: float — E[V] upper bound on false positives.
    """
    rng = np.random.RandomState(random_state)
    n, p = X.shape

    if lambda_grid is None:
        lambda_grid = np.logspace(-2, 0, 10)

    scaler = StandardScaler()
    X_std = scaler.fit_transform(X)

    selection_counts = np.zeros(p)
    n_sub = max(int(n * sample_fraction), 2)

    for b in range(n_bootstrap):
        idx = rng.choice(n, size=n_sub, replace=False)
        X_b, y_b = X_std[idx], y[idx]

        # Skip degenerate folds
        if len(np.unique(y_b)) < 2:
            continue

        # Randomized penalty weights (Meinshausen & Bühlmann prescription)
        weights = rng.uniform(0.5, 1.0, size=p)
        X_bw = X_b / weights[np.newaxis, :]

        # Pick one random regularization strength per bootstrap
        C = rng.choice(lambda_grid)
        clf = LogisticRegression(
            penalty="l1", C=C, solver="liblinear",
            max_iter=1000, random_state=random_state,
        )
        clf.fit(X_bw, y_b)
        selected = (clf.coef_.ravel() != 0)
        selection_counts += selected.astype(float)

    selection_probs_arr = selection_counts / n_bootstrap

    # Build result dict
    selection_probs = {name: float(selection_probs_arr[j]) for j, name in enumerate(feature_names)}
    selected_names = [name for name in feature_names if selection_probs[name] > pi_threshold]

    # E[V] bound (Meinshausen & Bühlmann, Thm 1)
    q = np.mean(np.sum(selection_counts > 0))  # avg selected per round (approx)
    if pi_threshold > 0.5:
        expected_fp = q ** 2 / ((2 * pi_threshold - 1) * p)
    else:
        expected_fp = float("inf")

    return selected_names, selection_probs, expected_fp
