"""Pre-filtering: remove near-duplicate features by correlation threshold."""

import numpy as np
import pandas as pd


def correlation_prefilter(X, feature_names, labels, threshold=0.95):
    """Drop one feature from each pair with |r| > threshold.

    Among correlated pairs, keep the feature with higher absolute
    correlation to the label (point-biserial).

    Returns:
        X_filtered: np.ndarray (N, F')
        kept_names: list[str]
        dropped: list[tuple[str, str, float]]  — (dropped, kept, correlation)
    """
    df = pd.DataFrame(X, columns=feature_names)
    corr_matrix = df.corr().abs()

    # Point-biserial = Pearson with binary label
    label_corr = np.array([abs(np.corrcoef(X[:, j], labels)[0, 1]) for j in range(X.shape[1])])

    to_drop = set()
    dropped_log = []
    upper = np.triu(np.ones(corr_matrix.shape, dtype=bool), k=1)

    for i in range(corr_matrix.shape[0]):
        for j in range(i + 1, corr_matrix.shape[1]):
            if not upper[i, j]:
                continue
            r = corr_matrix.iloc[i, j]
            if r > threshold:
                # Drop the feature less correlated with the label
                if label_corr[i] >= label_corr[j]:
                    drop_idx, keep_idx = j, i
                else:
                    drop_idx, keep_idx = i, j
                drop_name = feature_names[drop_idx]
                keep_name = feature_names[keep_idx]
                if drop_name not in to_drop:
                    to_drop.add(drop_name)
                    dropped_log.append((drop_name, keep_name, float(r)))

    kept_names = [n for n in feature_names if n not in to_drop]
    kept_idx = [feature_names.index(n) for n in kept_names]
    X_filtered = X[:, kept_idx]
    return X_filtered, kept_names, dropped_log
