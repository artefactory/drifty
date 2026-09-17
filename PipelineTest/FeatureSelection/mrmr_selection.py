"""Minimum Redundancy Maximum Relevance (mRMR) feature selection.

Greedy forward selection maximising:
    score(f) = I(f; y) - (1/|S|) * sum_{s in S} I(f; s)

where I(·;·) is estimated via mutual information (KNN estimator).
Reference: Peng, Long & Ding (2005), IEEE TPAMI.
"""

import numpy as np
from sklearn.feature_selection import mutual_info_classif, mutual_info_regression


def mrmr_selection(X, y, feature_names, n_select=10, random_state=42):
    """Select top-n features by mRMR criterion.

    Args:
        X: (N, p) feature matrix.
        y: (N,) binary labels.
        feature_names: list[str] of length p.
        n_select: number of features to select.
        random_state: seed for MI estimator.

    Returns:
        selected_names: list[str] in selection order.
        scores: dict[str, float] — mRMR score at time of selection.
    """
    p = X.shape[1]
    n_select = min(n_select, p)

    # Relevance: I(f_j; y)
    relevance = mutual_info_classif(X, y, random_state=random_state)

    # Pre-compute pairwise MI between features (continuous targets → use regression)
    mi_matrix = np.zeros((p, p))
    for j in range(p):
        mi_j = mutual_info_regression(X, X[:, j], random_state=random_state)
        mi_matrix[j, :] = mi_j

    # Symmetrize
    mi_matrix = (mi_matrix + mi_matrix.T) / 2.0

    selected_idx = []
    selected_names_out = []
    scores = {}
    remaining = set(range(p))

    for k in range(n_select):
        best_score = -np.inf
        best_j = None

        for j in remaining:
            rel = relevance[j]
            if len(selected_idx) == 0:
                red = 0.0
            else:
                red = np.mean([mi_matrix[j, s] for s in selected_idx])
            score = rel - red
            if score > best_score:
                best_score = score
                best_j = j

        selected_idx.append(best_j)
        remaining.discard(best_j)
        name = feature_names[best_j]
        selected_names_out.append(name)
        scores[name] = float(best_score)

    return selected_names_out, scores
