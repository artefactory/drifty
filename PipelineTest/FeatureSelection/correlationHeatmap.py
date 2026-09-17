"""
FeatureSelection.correlationHeatmap
====================================
Compute baseline + markovian features and plot a correlation heatmap
including the binary hallucination label for each configuration.
"""

import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler

# Ensure project root is importable
_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from PipelineTest.FeatureSelection.configs import CONFIGS, SAVE_ROOT, extract_features_and_labels


SAVE_DIR = os.path.join(SAVE_ROOT, "figures", "Correlations")


# =========================================================================
# MAIN
# =========================================================================

def run_heatmap(config_key, cfg):
    """Generate correlation heatmap for a single configuration."""
    name = cfg["name"]

    print(f"\n{'='*60}")
    print(f"Processing: {name}")
    print(f"{'='*60}")

    features, feature_names, labels = extract_features_and_labels(cfg)

    # Standardize
    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(features)

    # Append label column
    X = np.concatenate([features_scaled, labels.reshape(-1, 1)], axis=1)
    all_names = feature_names + ["Label"]

    # Correlation matrix
    corr = np.corrcoef(X, rowvar=False)

    # Plot
    plt.figure(figsize=(30, 24))
    sns.heatmap(
        corr,
        cmap="viridis",
        cbar=True,
        annot=True,
        fmt=".2f",
        xticklabels=all_names,
        yticklabels=all_names,
    )
    plt.xlabel("Features")
    plt.ylabel("Features")
    plt.title(f"Correlation Heatmap (Baseline + Markovian + Label)\n{name}")
    plt.tight_layout()

    os.makedirs(SAVE_DIR, exist_ok=True)
    save_path = os.path.join(SAVE_DIR, f"correlation_heatmap_{name}.png")
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"Heatmap saved to {save_path}")


def main():
    for config_key, cfg in CONFIGS.items():
        try:
            run_heatmap(config_key, cfg)
        except Exception as e:
            print(f"[ERROR] Failed for {config_key}: {e}")
            continue


if __name__ == "__main__":
    main()
