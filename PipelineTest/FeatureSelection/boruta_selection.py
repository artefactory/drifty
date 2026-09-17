"""Boruta feature selection (Kursa & Rudnicki, 2010).

Wrapper around Random Forest that creates shadow (permuted) copies of
all features and iteratively tests whether each real feature performs
significantly better than the best shadow feature.
"""

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.utils import check_random_state


def boruta_selection(X, y, feature_names, max_iter=100, alpha=0.05, random_state=42):
    """Run Boruta all-relevant feature selection.

    Args:
        X: (N, p) feature matrix.
        y: (N,) binary labels.
        feature_names: list[str] of length p.
        max_iter: maximum Boruta iterations.
        alpha: significance level (Bonferroni-corrected).
        random_state: seed.

    Returns:
        confirmed: list[str] — confirmed important features.
        tentative: list[str] — tentative (not rejected, not confirmed).
        rejected: list[str] — rejected features.
        hit_counts: dict[str, int] — number of times feature beat max shadow.
    """
    rng = check_random_state(random_state)
    n, p = X.shape

    # Track hits: number of iterations where feature importance > max shadow importance
    hits = np.zeros(p, dtype=int)
    status = np.full(p, 0)  # 0=undecided, 1=confirmed, -1=rejected

    for it in range(max_iter):
        # Create shadow features (permuted copies)
        X_shadow = X.copy()
        for j in range(p):
            X_shadow[:, j] = rng.permutation(X_shadow[:, j])

        X_aug = np.concatenate([X, X_shadow], axis=1)

        rf = RandomForestClassifier(
            n_estimators=100, max_depth=5, random_state=rng.randint(0, 2**31),
            n_jobs=-1,
        )
        rf.fit(X_aug, y)
        importances = rf.feature_importances_

        real_imp = importances[:p]
        shadow_imp = importances[p:]
        shadow_max = shadow_imp.max()

        # Count hits for undecided features
        for j in range(p):
            if status[j] == 0 and real_imp[j] > shadow_max:
                hits[j] += 1

        # Binomial test: after 'it+1' rounds, under H0 each hit has prob 0.5
        from scipy.stats import binom
        n_trials = it + 1
        alpha_corrected = alpha / p  # Bonferroni

        for j in range(p):
            if status[j] != 0:
                continue
            # Confirm if significantly more hits than chance
            p_val_confirm = 1.0 - binom.cdf(hits[j] - 1, n_trials, 0.5)
            if p_val_confirm < alpha_corrected:
                status[j] = 1
            # Reject if significantly fewer hits than chance
            p_val_reject = binom.cdf(hits[j], n_trials, 0.5)
            if p_val_reject < alpha_corrected:
                status[j] = -1

        # Stop early if all decided
        if np.all(status != 0):
            break

    confirmed = [feature_names[j] for j in range(p) if status[j] == 1]
    tentative = [feature_names[j] for j in range(p) if status[j] == 0]
    rejected = [feature_names[j] for j in range(p) if status[j] == -1]
    hit_counts = {feature_names[j]: int(hits[j]) for j in range(p)}

    return confirmed, tentative, rejected, hit_counts
