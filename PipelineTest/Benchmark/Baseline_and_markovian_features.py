"""
scripts/run_evaluation.py
=========================
Centralized evaluation script.

Computes all features (baseline + markovian + AR1 + semantic), trains classifiers
(Logistic Regression with GridSearchCV), and reports metrics (ROC-AUC, PR-AUC, accuracy).

Usage:
    python -m PipelineTest.scripts.run_evaluation [--config CONFIG]
"""

import sys
import os
import json
import argparse
import numpy as np


sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from sklearn.model_selection import StratifiedKFold, GridSearchCV
from sklearn.metrics import roc_auc_score, average_precision_score, accuracy_score, confusion_matrix
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

from PipelineTest.features.utils import load_outputs
from PipelineTest.features.baseline import get_baseline_features
from PipelineTest.features.markovian import get_markovian_features
from PipelineTest.Benchmark.data_split import (
    load_eval_data,
    get_train_test_split,
    get_train_test_split_for_config,
)
from PipelineTest.eval_configs import CONFIGS
from PipelineTest.Benchmark.values_csv import DEFAULT_VALUES_DIR, update_values_csv




# Colonnes que cette baseline ecrit dans values/<model>/<config_name>.csv
# (memes noms que PipelineTest/Benchmark/main.py::SAMPLE_VALUE_COLUMNS).
BASELINE_MARKOV_CSV_COLUMNS = ["baseline", "markov", "baseline_markov"]

# =========================================================================
# LOGISTIC REGRESSION PIPELINE
# =========================================================================

def run_logreg(X, y, feature_names, name, train_idx=None, test_idx=None, full_X=None):
    """Run GridSearchCV logistic regression, return results dict.

    If `full_X` is given, the fitted model also scores every one of its rows
    (train+test) and the per-sample probabilities are returned as
    "sample_scores", aligned positionally with `full_X`.
    """
    pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("logreg", LogisticRegression(max_iter=10000, random_state=42)),
    ])
    param_grid = {
        "logreg__C": [0.001, 0.01, 0.1, 1, 10, 100],
        "logreg__penalty": ["l1", "l2"],
        "logreg__solver": ["saga"],
    }

    if train_idx is not None and test_idx is not None:
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]
    else:
        split_point = int(len(X) * 0.8)
        X_train, X_test, y_train, y_test = X[:split_point], X[split_point:], y[:split_point], y[split_point:]
    print("TRAINING SAMPLES:", len(X_train), "TESTING SAMPLES:", len(X_test))

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    grid = GridSearchCV(pipe, param_grid, cv=cv, scoring="roc_auc", n_jobs=-1, verbose=0)
    grid.fit(X_train, y_train)

    best = grid.best_estimator_
    y_scores = best.predict_proba(X_test)[:, 1]
    y_pred = (y_scores >= 0.5).astype(int)

    acc = accuracy_score(y_test, y_pred)
    roc = roc_auc_score(y_test, y_scores)
    pr = average_precision_score(y_test, y_scores)

    thresholds = np.linspace(0, 1, 201)
    accs = [accuracy_score(y_test, (y_scores >= t).astype(int)) for t in thresholds]
    best_thresh = thresholds[np.argmax(accs)]
    best_acc = max(accs)

    cm = confusion_matrix(y_test, y_pred)
    tn, fp, fn, tp = cm.ravel()

    coefs = best.named_steps["logreg"].coef_[0]

    result = {
        "name": name,
        "features": feature_names,
        "n_train": len(X_train),
        "n_test": len(X_test),
        "best_params": {k: v for k, v in grid.best_params_.items()},
        "best_cv_roc_auc": float(grid.best_score_),
        "test_accuracy": float(acc),
        "test_best_accuracy": float(best_acc),
        "test_best_threshold": float(best_thresh),
        "test_roc_auc": float(roc),
        "test_pr_auc": float(pr),
        "test_pos_rate": float(np.mean(y_test)),
        "confusion_matrix": {"TN": int(tn), "FP": int(fp), "FN": int(fn), "TP": int(tp)},
        "coefficients": {fname: float(c) for fname, c in zip(feature_names, coefs)},
    }

    if full_X is not None:
        full_scores = best.predict_proba(full_X)[:, 1]
        result["sample_scores"] = [float(s) for s in full_scores]

    print(f"  [{name}] ROC-AUC={roc:.4f}  PR-AUC={pr:.4f}  Acc={acc:.4f}  BestAcc={best_acc:.4f} (thresh={best_thresh:.2f})")
    print(f"           Best params: {grid.best_params_}")
    return result




CLEANED_FEATURES = [
    "MeanMaskedEntropy",
    "MeanLogProb",
    "VarMaskedEntropy",
    "VarMaskedEntropyAcrossTokens",
    "VarLogProb",
    "AlphaEntropy",
    "BetaEntropy",
    "AlphaEntropyAvg",
    "BetaEntropyAvg",
    "AlphaLogProb",
    "BetaLogProb",
    "C_fit",
    "Tau_fit",
    "M_fit",
    "MeanTau",
    "VarTau",
    "AUC_V",
    "Max_V",
    "Argmax_V",
    "Skewness_V",
    "Kurtosis_V",
    "MeanCurvature_V"
]


def select_features(feat, names, selected_names):
    """Select a subset of features by name."""
    idx = [names.index(n) for n in selected_names if n in names]
    selected_names = [n for n in selected_names if n in names]
    return feat[:, idx], selected_names


def getBaselineFeatures(outputs, indices):
    feat_baseline, names_baseline = get_baseline_features(outputs, pad_token_id=None)
    feat_baseline = feat_baseline[indices]  
    return feat_baseline, names_baseline

def getMarkovianFeatures(outputs, indices):
    feat_markov, names_markov = get_markovian_features(outputs, k_tokens=64)
    feat_markov = feat_markov[indices]
    return feat_markov, names_markov


# =========================================================================
# MAIN
# =========================================================================

def main(config_name="llada", update_csv=False, values_dir=DEFAULT_VALUES_DIR):
    cfg = CONFIGS[config_name]
    output_path = cfg["outputs_path"]
    eval_json = cfg["eval_json"]
    outputs = load_outputs(output_path)

    print(f"\n{'='*70}")
    print(f"  EVALUATION: {cfg['name']}")
    print(f"  [split: 80/20 sequential]")
    print(f"{'='*70}")

    # Load data
    print("\n[1] Loading outputs...")
    labels_raw, indices_raw, n_missing_label, idx_to_prompt = load_eval_data(eval_json)
    print(
            f"  Raw samples: {len(labels_raw)} "
            f"(correct={np.sum(labels_raw==0)}, halluc={np.sum(labels_raw==1)}) | "
            f"Dropped: {n_missing_label} missing/unusable label"
        )

    labels = labels_raw
    indices = indices_raw
    balance_stats = {
        "n_pos_before": int(np.sum(labels == 1)),
        "n_neg_before": int(np.sum(labels == 0)),
        "n_kept_per_class": "N/A (Sequential Split)",
    }

    train_idx, test_idx = get_train_test_split_for_config(
        config_name, indices, values_dir=values_dir
    )

    sample_indices = outputs.sample_indices.numpy()
    idx_to_pos = {int(idx): pos for pos, idx in enumerate(sample_indices)}
    tensor_positions = [np.array(idx_to_pos[int(idx)]) for idx in indices]
    print('uae  ujn,azfnjqmc')
    feat_baseline, names_baseline = getBaselineFeatures(outputs, tensor_positions)
    feat_markov, names_markov = getMarkovianFeatures(outputs, tensor_positions)
    print('hereeee')
    # Combiner toutes les features pour pouvoir piocher dedans via select_features
    X_base_markov = np.column_stack([feat_baseline, feat_markov])
    names_bm = names_baseline + names_markov

    # Baseline only (on eval half)
    print("\n  === Baseline Features ===")
    res_baseline = run_logreg(feat_baseline, labels, names_baseline, "Baseline", train_idx=train_idx, test_idx=test_idx, full_X=feat_baseline)

    # Markovian only (on eval half)
    print("\n  === Markovian Features ===")
    res_markov = run_logreg(feat_markov, labels, names_markov, "Markovian", train_idx=train_idx, test_idx=test_idx, full_X=feat_markov)

    # Baseline + Markovian
    print("\n  === Baseline + Markovian ===")
    res_bm = run_logreg(X_base_markov, labels, names_bm, "Baseline+Markov", train_idx=train_idx, test_idx=test_idx, full_X=X_base_markov)


    # Cleaned features (22 features selection)
    X_clean, names_clean = select_features(X_base_markov, names_bm, CLEANED_FEATURES)
    print("\n  === Cleaned Features (22 selection) ===")
    print(f"  Features: {names_clean}")
    res_clean = run_logreg(X_clean, labels, names_clean, "CleanedFeatures", train_idx=train_idx, test_idx=test_idx)

    # Per-sample scores (train+test), keyed by original sample index, for
    # downstream per-sample CSV export (Benchmark/main.py).
    split_of = {int(idx): "train" for idx in indices[train_idx]}
    split_of.update({int(idx): "test" for idx in indices[test_idx]})
    sample_scores = {}
    for pos, idx in enumerate(indices):
        sample_scores[int(idx)] = {
            "label_hallucination": int(labels[pos]),
            "split": split_of[int(idx)],
            "baseline": res_baseline["sample_scores"][pos],
            "markov": res_markov["sample_scores"][pos],
            "baseline_markov": res_bm["sample_scores"][pos],
        }

    # --- SAVE ---
    save_dir = "PipelineTest/Benchmark/eval"
    os.makedirs(save_dir, exist_ok=True)

    results = {
        "config": cfg,
        "n_samples": len(labels),
        "n_correct": int(np.sum(labels == 0)),
        "n_halluc": int(np.sum(labels == 1)),
        "results": {
            "Baseline": res_baseline,
            "Markovian": res_markov,
            "Baseline+Markov": res_bm,
            "CleanedFeatures": res_clean,
        },
        "samples": sample_scores,
    }

    out_path = os.path.join(save_dir, f"Baseline_Markov_{cfg['name']}.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"\n    Results saved to: {out_path}")

    # --- SUMMARY TABLE ---
    print(f"\n{'='*70}")
    print(f"  SUMMARY: {cfg['name']}")
    print(f"{'='*70}")
    print(f"  {'Method':<22} {'ROC-AUC':>10} {'PR-AUC':>10} {'Accuracy':>10}")
    print(f"  {'-'*52}")
    for r in [res_baseline, res_markov, res_bm, res_clean]:
        print(f"  {r['name']:<22} {r['test_roc_auc']:>10.4f} {r['test_pr_auc']:>10.4f} {r['test_accuracy']:>10.4f}")

    # Mise a jour du CSV par echantillon (run standalone uniquement : lance via
    # Benchmark/main.py, c'est main.py qui ecrit le CSV consolide a la fin).
    if update_csv:
        update_values_csv(config_name, sample_scores, BASELINE_MARKOV_CSV_COLUMNS,
                          values_dir=values_dir)

    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run hallucination detection evaluation")
    parser.add_argument("--config", type=str, default="llada", choices=list(CONFIGS.keys()) + ["all"],
                        help="Which model config to evaluate, or 'all' for every config")
    parser.add_argument("--values_dir", type=str, default=DEFAULT_VALUES_DIR,
                        help="Racine des CSV par echantillon a mettre a jour "
                             "(values/<model>/<config_name>.csv).")
    parser.add_argument("--no_update_csv", action="store_true",
                        help="Ne pas toucher aux CSV de values/ (par defaut ils sont mis a jour).")
    args = parser.parse_args()
    configs_to_run = list(CONFIGS) if args.config == "all" else [args.config]
    for config_name in configs_to_run:
        main(config_name, update_csv=not args.no_update_csv, values_dir=args.values_dir)