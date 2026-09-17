"""Full feature selection pipeline: prefilter → select → evaluate.

Runs all three selection methods (Stability Selection, mRMR, Boruta) on
every experiment configuration and evaluates each subset with nested CV.
Saves results as JSON and produces summary plots.
"""

import os
import sys
import json
import numpy as np
import matplotlib.pyplot as plt

_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from PipelineTest.FeatureSelection.configs import CONFIGS, SAVE_ROOT, extract_features_and_labels
from PipelineTest.FeatureSelection.prefilter import correlation_prefilter
from PipelineTest.FeatureSelection.stability_selection import stability_selection
from PipelineTest.FeatureSelection.mrmr_selection import mrmr_selection
from PipelineTest.FeatureSelection.boruta_selection import boruta_selection
from PipelineTest.FeatureSelection.evaluate import nested_cv_evaluate, nested_cv_evaluate_with_selection, benchmark_evaluate, benchmark_evaluate_with_selection


def plot_selection_probabilities(selection_probs, config_name, save_dir):
    """Bar chart of stability selection probabilities."""
    names = list(selection_probs.keys())
    probs = [selection_probs[n] for n in names]
    order = np.argsort(probs)[::-1]

    plt.figure(figsize=(12, 6))
    plt.bar(range(len(names)), [probs[i] for i in order], color="steelblue")
    plt.xticks(range(len(names)), [names[i] for i in order], rotation=45, ha="right")
    plt.axhline(0.6, color="red", linestyle="--", label="π_thr = 0.6")
    plt.axhline(0.9, color="orange", linestyle="--", label="π_thr = 0.9")
    plt.ylabel("Selection probability Π̂_j")
    plt.title(f"Stability Selection — {config_name}")
    plt.legend()
    plt.tight_layout()

    os.makedirs(save_dir, exist_ok=True)
    plt.savefig(os.path.join(save_dir, f"stability_probs_{config_name}.png"), dpi=150)
    plt.close()


def plot_mrmr_scores(scores, config_name, save_dir):
    """Bar chart of mRMR scores in selection order."""
    names = list(scores.keys())
    vals = [scores[n] for n in names]

    plt.figure(figsize=(10, 5))
    plt.bar(range(len(names)), vals, color="darkorange")
    plt.xticks(range(len(names)), names, rotation=45, ha="right")
    plt.ylabel("mRMR score (relevance − redundancy)")
    plt.title(f"mRMR selection order — {config_name}")
    plt.tight_layout()

    os.makedirs(save_dir, exist_ok=True)
    plt.savefig(os.path.join(save_dir, f"mrmr_scores_{config_name}.png"), dpi=150)
    plt.close()


def run_pipeline(config_key, cfg):
    """Full pipeline for one configuration."""
    name = cfg["name"]
    print(f"\n{'='*70}")
    print(f"  {name}")
    print(f"{'='*70}")

    features, feature_names, labels = extract_features_and_labels(cfg)
    save_dir = os.path.join(SAVE_ROOT, "figures", name)

    # --- 1. Pre-filter ---
    X_filt, filt_names, dropped = correlation_prefilter(features, feature_names, labels, threshold=0.95)
    print(f"[prefilter] {len(feature_names)} → {len(filt_names)} features (dropped {len(dropped)})")
    for d_name, k_name, r in dropped:
        print(f"  dropped {d_name} (r={r:.3f} with {k_name})")

    # --- 2a. Stability Selection ---
    stab_selected, stab_probs, stab_efp = stability_selection(
        X_filt, labels, filt_names, n_bootstrap=500, pi_threshold=0.6,
    )
    print(f"[stability] selected {len(stab_selected)} features, E[FP] ≤ {stab_efp:.2f}")
    print(f"  → {stab_selected}")
    plot_selection_probabilities(stab_probs, name, save_dir)

    # --- 2b. mRMR (select top-10, then keep only positive scores) ---
    mrmr_all, mrmr_scores = mrmr_selection(
        X_filt, labels, filt_names, n_select=min(10, len(filt_names)),
    )
    mrmr_selected = [f for f in mrmr_all if mrmr_scores[f] > 0]
    if not mrmr_selected:
        mrmr_selected = mrmr_all[:3]
    print(f"[mrmr] selected {len(mrmr_selected)} features (score > 0)")
    print(f"  → {mrmr_selected}")
    plot_mrmr_scores(mrmr_scores, name, save_dir)

    # --- 2c. Boruta ---
    boruta_confirmed, boruta_tentative, boruta_rejected, boruta_hits = boruta_selection(
        X_filt, labels, filt_names,
    )
    print(f"[boruta] confirmed={len(boruta_confirmed)}, tentative={len(boruta_tentative)}, rejected={len(boruta_rejected)}")
    print(f"  confirmed → {boruta_confirmed}")

    # --- 3. Nested CV evaluation (leakage-free: selection inside each fold) ---
    eval_results = {}

    for method in ["stability", "mrmr"]:
        print(f"[eval] nested CV with {method} selection inside folds ...")
        res, sel = benchmark_evaluate_with_selection(
            X_filt, labels, filt_names, selection_method=method,
        )
        eval_results[method] = {
            "selected_features_on_train": sel,
            "n_features_on_train": len(sel),
            "classifiers": res,
        }
        for clf_name, clf_res in res.items():
            print(f"  {clf_name}: AUROC={clf_res['auroc']:.4f}  AUPRC={clf_res['auprc']:.4f}")

    # Also evaluate all features as baseline
    print(f"[eval] all_features ({len(filt_names)} features) ...")
    res_all = benchmark_evaluate(X_filt, labels, filt_names, filt_names)
    eval_results["all_features"] = {
        "selected_features": filt_names,
        "n_features": len(filt_names),
        "classifiers": res_all,
    }
    for clf_name, clf_res in res_all.items():
        print(f"  {clf_name}: AUROC={clf_res['auroc']:.4f}  AUPRC={clf_res['auprc']:.4f}")

    # --- 4. Save JSON ---
    out = {
        "config": name,
        "prefilter": {"dropped": dropped, "remaining": filt_names},
        "stability_selection": {
            "selected": stab_selected,
            "probabilities": stab_probs,
            "expected_fp": stab_efp,
        },
        "mrmr": {"selected": mrmr_selected, "scores": mrmr_scores},
        "boruta": {
            "confirmed": boruta_confirmed,
            "tentative": boruta_tentative,
            "rejected": boruta_rejected,
            "hit_counts": boruta_hits,
        },
        "evaluation": {
            method: {
                "n_features_on_train": info.get("n_features_on_train", info.get("n_features")),
                "classifiers": {
                    clf: {k: v for k, v in cres.items() if k not in ("fold_auroc", "fold_auprc")}
                    for clf, cres in info["classifiers"].items()
                },
            }
            for method, info in eval_results.items()
        },
    }

    json_dir = os.path.join(SAVE_ROOT, "results")
    os.makedirs(json_dir, exist_ok=True)
    json_path = os.path.join(json_dir, f"feature_selection_{name}.json")
    with open(json_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"[saved] {json_path}")


def main():
    for config_key, cfg in CONFIGS.items():
        if "gemma" in config_key:
            try:
                run_pipeline(config_key, cfg)
            except Exception as e:
                print(f"[ERROR] {config_key}: {e}")
                import traceback
                traceback.print_exc()
                continue


if __name__ == "__main__":
    main()
