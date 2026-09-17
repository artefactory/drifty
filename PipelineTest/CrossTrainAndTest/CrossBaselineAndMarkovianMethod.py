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



from sklearn.model_selection import StratifiedKFold, GridSearchCV
from sklearn.metrics import roc_auc_score, average_precision_score, accuracy_score, confusion_matrix
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from PipelineTest.features.utils import load_outputs
from PipelineTest.features.baseline import get_baseline_features
from PipelineTest.features.markovian import get_markovian_features

from PipelineTest.CrossTrainAndTest.utils import load_eval_data_for_crossing


# =========================================================================
# CONFIG
# =========================================================================
CONFIGS = {
    # =========================================================================
    # CONFIGURATIONS : DREAM 16 STEPS / 32 TOKENS
    # =========================================================================
    # --- Train: TriviaQA ---
    "dream16_32tokens_2500samples_triviaqa_eval_triviaqa_seed42": {
        "tokenizer_path": "PipelineTest/res/results_dream_16steps_32tokens_triviaqa_2500samples_seed42/tokenizer_dream_16steps_32tokens_triviaqa_2500samples_seed42.pt",
        "name": "dream_16steps_32tokens_triviaqa_2500samples_eval_triviaqa_seed42",
        "train_output_path": "PipelineTest/res/results_dream_16steps_32tokens_triviaqa_2500samples_seed42/outputs_dream_16steps_32tokens_triviaqa_2500samples_seed42.pt",
        "eval_output_path": "PipelineTest/res/results_dream_16steps_32tokens_triviaqa_2500samples_seed42/outputs_dream_16steps_32tokens_triviaqa_2500samples_seed42.pt",
        "train_json": "PipelineTest/res/eval_V2/results_dream_16steps_32tokens_triviaqa_2500samples_seed42.json",
        "eval_json": "PipelineTest/res/eval_V2/results_dream_16steps_32tokens_triviaqa_2500samples_seed42.json",
    },
    "dream16_32tokens_2500samples_triviaqa_eval_naturalquestion_seed42": {
        "tokenizer_path": "PipelineTest/res/results_dream_16steps_32tokens_triviaqa_2500samples_seed42/tokenizer_dream_16steps_32tokens_triviaqa_2500samples_seed42.pt",
        "name": "dream_16steps_32tokens_triviaqa_2500samples_eval_naturalquestion_seed42",
        "train_output_path": "PipelineTest/res/results_dream_16steps_32tokens_triviaqa_2500samples_seed42/outputs_dream_16steps_32tokens_triviaqa_2500samples_seed42.pt",
        "eval_output_path": "PipelineTest/res/results_dream_16steps_32tokens_naturalquestion_2500samples_seed42/outputs_dream_16steps_32tokens_naturalquestion_2500samples_seed42.pt",
        "train_json": "PipelineTest/res/eval_V2/results_dream_16steps_32tokens_triviaqa_2500samples_seed42.json",
        "eval_json": "PipelineTest/res/eval_V2/results_dream_16steps_32tokens_naturalquestion_2500samples_seed42.json",
    },
    "dream16_32tokens_2500samples_triviaqa_eval_hotpotqa_seed42": {
        "tokenizer_path": "PipelineTest/res/results_dream_16steps_32tokens_triviaqa_2500samples_seed42/tokenizer_dream_16steps_32tokens_triviaqa_2500samples_seed42.pt",
        "name": "dream_16steps_32tokens_triviaqa_2500samples_eval_hotpotqa_seed42",
        "train_output_path": "PipelineTest/res/results_dream_16steps_32tokens_triviaqa_2500samples_seed42/outputs_dream_16steps_32tokens_triviaqa_2500samples_seed42.pt",
        "eval_output_path": "PipelineTest/res/results_dream_16steps_32tokens_hotpotqa_2500samples_seed42/outputs_dream_16steps_32tokens_hotpotqa_2500samples_seed42.pt",
        "train_json": "PipelineTest/res/eval_V2/results_dream_16steps_32tokens_triviaqa_2500samples_seed42.json",
        "eval_json": "PipelineTest/res/eval_V2/results_dream_16steps_32tokens_hotpotqa_2500samples_seed42.json",
    },

    # --- Train: NaturalQuestion ---
    "dream16_32tokens_2500samples_naturalquestion_eval_triviaqa_seed42": {
        "tokenizer_path": "PipelineTest/res/results_dream_16steps_32tokens_naturalquestion_2500samples_seed42/tokenizer_dream_16steps_32tokens_naturalquestion_2500samples_seed42.pt",
        "name": "dream_16steps_32tokens_naturalquestion_2500samples_eval_triviaqa_seed42",
        "train_output_path": "PipelineTest/res/results_dream_16steps_32tokens_naturalquestion_2500samples_seed42/outputs_dream_16steps_32tokens_naturalquestion_2500samples_seed42.pt",
        "eval_output_path": "PipelineTest/res/results_dream_16steps_32tokens_triviaqa_2500samples_seed42/outputs_dream_16steps_32tokens_triviaqa_2500samples_seed42.pt",
        "train_json": "PipelineTest/res/eval_V2/results_dream_16steps_32tokens_naturalquestion_2500samples_seed42.json",
        "eval_json": "PipelineTest/res/eval_V2/results_dream_16steps_32tokens_triviaqa_2500samples_seed42.json",
    },
    "dream16_32tokens_2500samples_naturalquestion_eval_naturalquestion_seed42": {
        "tokenizer_path": "PipelineTest/res/results_dream_16steps_32tokens_naturalquestion_2500samples_seed42/tokenizer_dream_16steps_32tokens_naturalquestion_2500samples_seed42.pt",
        "name": "dream_16steps_32tokens_naturalquestion_2500samples_eval_naturalquestion_seed42",
        "train_output_path": "PipelineTest/res/results_dream_16steps_32tokens_naturalquestion_2500samples_seed42/outputs_dream_16steps_32tokens_naturalquestion_2500samples_seed42.pt",
        "eval_output_path": "PipelineTest/res/results_dream_16steps_32tokens_naturalquestion_2500samples_seed42/outputs_dream_16steps_32tokens_naturalquestion_2500samples_seed42.pt",
        "train_json": "PipelineTest/res/eval_V2/results_dream_16steps_32tokens_naturalquestion_2500samples_seed42.json",
        "eval_json": "PipelineTest/res/eval_V2/results_dream_16steps_32tokens_naturalquestion_2500samples_seed42.json",
    },
    "dream16_32tokens_2500samples_naturalquestion_eval_hotpotqa_seed42": {
        "tokenizer_path": "PipelineTest/res/results_dream_16steps_32tokens_naturalquestion_2500samples_seed42/tokenizer_dream_16steps_32tokens_naturalquestion_2500samples_seed42.pt",
        "name": "dream_16steps_32tokens_naturalquestion_2500samples_eval_hotpotqa_seed42",
        "train_output_path": "PipelineTest/res/results_dream_16steps_32tokens_naturalquestion_2500samples_seed42/outputs_dream_16steps_32tokens_naturalquestion_2500samples_seed42.pt",
        "eval_output_path": "PipelineTest/res/results_dream_16steps_32tokens_hotpotqa_2500samples_seed42/outputs_dream_16steps_32tokens_hotpotqa_2500samples_seed42.pt",
        "train_json": "PipelineTest/res/eval_V2/results_dream_16steps_32tokens_naturalquestion_2500samples_seed42.json",
        "eval_json": "PipelineTest/res/eval_V2/results_dream_16steps_32tokens_hotpotqa_2500samples_seed42.json",
    },

    # --- Train: HotpotQA ---
    "dream16_32tokens_2500samples_hotpotqa_eval_triviaqa_seed42": {
        "tokenizer_path": "PipelineTest/res/results_dream_16steps_32tokens_hotpotqa_2500samples_seed42/tokenizer_dream_16steps_32tokens_hotpotqa_2500samples_seed42.pt",
        "name": "dream_16steps_32tokens_hotpotqa_2500samples_eval_triviaqa_seed42",
        "train_output_path": "PipelineTest/res/results_dream_16steps_32tokens_hotpotqa_2500samples_seed42/outputs_dream_16steps_32tokens_hotpotqa_2500samples_seed42.pt",
        "eval_output_path": "PipelineTest/res/results_dream_16steps_32tokens_triviaqa_2500samples_seed42/outputs_dream_16steps_32tokens_triviaqa_2500samples_seed42.pt",
        "train_json": "PipelineTest/res/eval_V2/results_dream_16steps_32tokens_hotpotqa_2500samples_seed42.json",
        "eval_json": "PipelineTest/res/eval_V2/results_dream_16steps_32tokens_triviaqa_2500samples_seed42.json",
    },
    "dream16_32tokens_2500samples_hotpotqa_eval_naturalquestion_seed42": {
        "tokenizer_path": "PipelineTest/res/results_dream_16steps_32tokens_hotpotqa_2500samples_seed42/tokenizer_dream_16steps_32tokens_hotpotqa_2500samples_seed42.pt",
        "name": "dream_16steps_32tokens_hotpotqa_2500samples_eval_naturalquestion_seed42",
        "train_output_path": "PipelineTest/res/results_dream_16steps_32tokens_hotpotqa_2500samples_seed42/outputs_dream_16steps_32tokens_hotpotqa_2500samples_seed42.pt",
        "eval_output_path": "PipelineTest/res/results_dream_16steps_32tokens_naturalquestion_2500samples_seed42/outputs_dream_16steps_32tokens_naturalquestion_2500samples_seed42.pt",
        "train_json": "PipelineTest/res/eval_V2/results_dream_16steps_32tokens_hotpotqa_2500samples_seed42.json",
        "eval_json": "PipelineTest/res/eval_V2/results_dream_16steps_32tokens_naturalquestion_2500samples_seed42.json",
    },
    "dream16_32tokens_2500samples_hotpotqa_eval_hotpotqa_seed42": {
        "tokenizer_path": "PipelineTest/res/results_dream_16steps_32tokens_hotpotqa_2500samples_seed42/tokenizer_dream_16steps_32tokens_hotpotqa_2500samples_seed42.pt",
        "name": "dream_16steps_32tokens_hotpotqa_2500samples_eval_hotpotqa_seed42",
        "train_output_path": "PipelineTest/res/results_dream_16steps_32tokens_hotpotqa_2500samples_seed42/outputs_dream_16steps_32tokens_hotpotqa_2500samples_seed42.pt",
        "eval_output_path": "PipelineTest/res/results_dream_16steps_32tokens_hotpotqa_2500samples_seed42/outputs_dream_16steps_32tokens_hotpotqa_2500samples_seed42.pt",
        "train_json": "PipelineTest/res/eval_V2/results_dream_16steps_32tokens_hotpotqa_2500samples_seed42.json",
        "eval_json": "PipelineTest/res/eval_V2/results_dream_16steps_32tokens_hotpotqa_2500samples_seed42.json",
    },


    # =========================================================================
    # CONFIGURATIONS : LLADA 16 STEPS / 32 TOKENS
    # =========================================================================
    # --- Train: TriviaQA ---
    "llada16_32tokens_2500samples_triviaqa_eval_triviaqa_seed42": {
        "tokenizer_path": "PipelineTest/res/results_llada_16steps_32tokens_triviaqa_2500samples_seed42/tokenizer_llada_16steps_32tokens_triviaqa_2500samples_seed42.pt",
        "name": "llada_16steps_32tokens_triviaqa_2500samples_eval_triviaqa_seed42",
        "train_output_path": "PipelineTest/res/results_llada_16steps_32tokens_triviaqa_2500samples_seed42/outputs_llada_16steps_32tokens_triviaqa_2500samples_seed42.pt",
        "eval_output_path": "PipelineTest/res/results_llada_16steps_32tokens_triviaqa_2500samples_seed42/outputs_llada_16steps_32tokens_triviaqa_2500samples_seed42.pt",
        "train_json": "PipelineTest/res/eval_V2/results_llada_16steps_32tokens_triviaqa_2500samples_seed42.json",
        "eval_json": "PipelineTest/res/eval_V2/results_llada_16steps_32tokens_triviaqa_2500samples_seed42.json",
    },
    "llada16_32tokens_2500samples_triviaqa_eval_naturalquestion_seed42": {
        "tokenizer_path": "PipelineTest/res/results_llada_16steps_32tokens_triviaqa_2500samples_seed42/tokenizer_llada_16steps_32tokens_triviaqa_2500samples_seed42.pt",
        "name": "llada_16steps_32tokens_triviaqa_2500samples_eval_naturalquestion_seed42",
        "train_output_path": "PipelineTest/res/results_llada_16steps_32tokens_triviaqa_2500samples_seed42/outputs_llada_16steps_32tokens_triviaqa_2500samples_seed42.pt",
        "eval_output_path": "PipelineTest/res/results_llada_16steps_32tokens_naturalquestion_2500samples_seed42/outputs_llada_16steps_32tokens_naturalquestion_2500samples_seed42.pt",
        "train_json": "PipelineTest/res/eval_V2/results_llada_16steps_32tokens_triviaqa_2500samples_seed42.json",
        "eval_json": "PipelineTest/res/eval_V2/results_llada_16steps_32tokens_naturalquestion_2500samples_seed42.json",
    },
    "llada16_32tokens_2500samples_triviaqa_eval_hotpotqa_seed42": {
        "tokenizer_path": "PipelineTest/res/results_llada_16steps_32tokens_triviaqa_2500samples_seed42/tokenizer_llada_16steps_32tokens_triviaqa_2500samples_seed42.pt",
        "name": "llada_16steps_32tokens_triviaqa_2500samples_eval_hotpotqa_seed42",
        "train_output_path": "PipelineTest/res/results_llada_16steps_32tokens_triviaqa_2500samples_seed42/outputs_llada_16steps_32tokens_triviaqa_2500samples_seed42.pt",
        "eval_output_path": "PipelineTest/res/results_llada_16steps_32tokens_hotpotqa_2500samples_seed42/outputs_llada_16steps_32tokens_hotpotqa_2500samples_seed42.pt",
        "train_json": "PipelineTest/res/eval_V2/results_llada_16steps_32tokens_triviaqa_2500samples_seed42.json",
        "eval_json": "PipelineTest/res/eval_V2/results_llada_16steps_32tokens_hotpotqa_2500samples_seed42.json",
    },

    # --- Train: NaturalQuestion ---
    "llada16_32tokens_2500samples_naturalquestion_eval_triviaqa_seed42": {
        "tokenizer_path": "PipelineTest/res/results_llada_16steps_32tokens_naturalquestion_2500samples_seed42/tokenizer_llada_16steps_32tokens_naturalquestion_2500samples_seed42.pt",
        "name": "llada_16steps_32tokens_naturalquestion_2500samples_eval_triviaqa_seed42",
        "train_output_path": "PipelineTest/res/results_llada_16steps_32tokens_naturalquestion_2500samples_seed42/outputs_llada_16steps_32tokens_naturalquestion_2500samples_seed42.pt",
        "eval_output_path": "PipelineTest/res/results_llada_16steps_32tokens_triviaqa_2500samples_seed42/outputs_llada_16steps_32tokens_triviaqa_2500samples_seed42.pt",
        "train_json": "PipelineTest/res/eval_V2/results_llada_16steps_32tokens_naturalquestion_2500samples_seed42.json",
        "eval_json": "PipelineTest/res/eval_V2/results_llada_16steps_32tokens_triviaqa_2500samples_seed42.json",
    },
    "llada16_32tokens_2500samples_naturalquestion_eval_naturalquestion_seed42": {
        "tokenizer_path": "PipelineTest/res/results_llada_16steps_32tokens_naturalquestion_2500samples_seed42/tokenizer_llada_16steps_32tokens_naturalquestion_2500samples_seed42.pt",
        "name": "llada_16steps_32tokens_naturalquestion_2500samples_eval_naturalquestion_seed42",
        "train_output_path": "PipelineTest/res/results_llada_16steps_32tokens_naturalquestion_2500samples_seed42/outputs_llada_16steps_32tokens_naturalquestion_2500samples_seed42.pt",
        "eval_output_path": "PipelineTest/res/results_llada_16steps_32tokens_naturalquestion_2500samples_seed42/outputs_llada_16steps_32tokens_naturalquestion_2500samples_seed42.pt",
        "train_json": "PipelineTest/res/eval_V2/results_llada_16steps_32tokens_naturalquestion_2500samples_seed42.json",
        "eval_json": "PipelineTest/res/eval_V2/results_llada_16steps_32tokens_naturalquestion_2500samples_seed42.json",
    },
    "llada16_32tokens_2500samples_naturalquestion_eval_hotpotqa_seed42": {
        "tokenizer_path": "PipelineTest/res/results_llada_16steps_32tokens_naturalquestion_2500samples_seed42/tokenizer_llada_16steps_32tokens_naturalquestion_2500samples_seed42.pt",
        "name": "llada_16steps_32tokens_naturalquestion_2500samples_eval_hotpotqa_seed42",
        "train_output_path": "PipelineTest/res/results_llada_16steps_32tokens_naturalquestion_2500samples_seed42/outputs_llada_16steps_32tokens_naturalquestion_2500samples_seed42.pt",
        "eval_output_path": "PipelineTest/res/results_llada_16steps_32tokens_hotpotqa_2500samples_seed42/outputs_llada_16steps_32tokens_hotpotqa_2500samples_seed42.pt",
        "train_json": "PipelineTest/res/eval_V2/results_llada_16steps_32tokens_naturalquestion_2500samples_seed42.json",
        "eval_json": "PipelineTest/res/eval_V2/results_llada_16steps_32tokens_hotpotqa_2500samples_seed42.json",
    },

    # --- Train: HotpotQA ---
    "llada16_32tokens_2500samples_hotpotqa_eval_triviaqa_seed42": {
        "tokenizer_path": "PipelineTest/res/results_llada_16steps_32tokens_hotpotqa_2500samples_seed42/tokenizer_llada_16steps_32tokens_hotpotqa_2500samples_seed42.pt",
        "name": "llada_16steps_32tokens_hotpotqa_2500samples_eval_triviaqa_seed42",
        "train_output_path": "PipelineTest/res/results_llada_16steps_32tokens_hotpotqa_2500samples_seed42/outputs_llada_16steps_32tokens_hotpotqa_2500samples_seed42.pt",
        "eval_output_path": "PipelineTest/res/results_llada_16steps_32tokens_triviaqa_2500samples_seed42/outputs_llada_16steps_32tokens_triviaqa_2500samples_seed42.pt",
        "train_json": "PipelineTest/res/eval_V2/results_llada_16steps_32tokens_hotpotqa_2500samples_seed42.json",
        "eval_json": "PipelineTest/res/eval_V2/results_llada_16steps_32tokens_triviaqa_2500samples_seed42.json",
    },
    "llada16_32tokens_2500samples_hotpotqa_eval_naturalquestion_seed42": {
        "tokenizer_path": "PipelineTest/res/results_llada_16steps_32tokens_hotpotqa_2500samples_seed42/tokenizer_llada_16steps_32tokens_hotpotqa_2500samples_seed42.pt",
        "name": "llada_16steps_32tokens_hotpotqa_2500samples_eval_naturalquestion_seed42",
        "train_output_path": "PipelineTest/res/results_llada_16steps_32tokens_hotpotqa_2500samples_seed42/outputs_llada_16steps_32tokens_hotpotqa_2500samples_seed42.pt",
        "eval_output_path": "PipelineTest/res/results_llada_16steps_32tokens_naturalquestion_2500samples_seed42/outputs_llada_16steps_32tokens_naturalquestion_2500samples_seed42.pt",
        "train_json": "PipelineTest/res/eval_V2/results_llada_16steps_32tokens_hotpotqa_2500samples_seed42.json",
        "eval_json": "PipelineTest/res/eval_V2/results_llada_16steps_32tokens_naturalquestion_2500samples_seed42.json",
    },
    "llada16_32tokens_2500samples_hotpotqa_eval_hotpotqa_seed42": {
        "tokenizer_path": "PipelineTest/res/results_llada_16steps_32tokens_hotpotqa_2500samples_seed42/tokenizer_llada_16steps_32tokens_hotpotqa_2500samples_seed42.pt",
        "name": "llada_16steps_32tokens_hotpotqa_2500samples_eval_hotpotqa_seed42",
        "train_output_path": "PipelineTest/res/results_llada_16steps_32tokens_hotpotqa_2500samples_seed42/outputs_llada_16steps_32tokens_hotpotqa_2500samples_seed42.pt",
        "eval_output_path": "PipelineTest/res/results_llada_16steps_32tokens_hotpotqa_2500samples_seed42/outputs_llada_16steps_32tokens_hotpotqa_2500samples_seed42.pt",
        "train_json": "PipelineTest/res/eval_V2/results_llada_16steps_32tokens_hotpotqa_2500samples_seed42.json",
        "eval_json": "PipelineTest/res/eval_V2/results_llada_16steps_32tokens_hotpotqa_2500samples_seed42.json",
    },
}


# =========================================================================
# LOGISTIC REGRESSION PIPELINE
# =========================================================================


def run_logreg(X_train, y_train, X_test, y_test, feature_names, name,):
    """Run GridSearchCV logistic regression, return results dict."""
    pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("logreg", LogisticRegression(max_iter=10000, random_state=42)),
    ])
    param_grid = {
        "logreg__C": [0.001, 0.01, 0.1, 1, 10, 100],
        "logreg__penalty": ["l1", "l2"],
        "logreg__solver": ["saga"],
    }


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

def run_config(cfg):
    train_output_path = cfg["train_output_path"]
    eval_output_path = cfg["eval_output_path"]
    train_json = cfg["train_json"]
    eval_json = cfg["eval_json"]

    train_outputs = load_outputs(train_output_path)
    eval_outputs = load_outputs(eval_output_path)  # Assuming same outputs for eval; adjust if needed


    # Load data
    print("\n[1] Loading outputs...")
    train_labels, train_indices, train_n_missing_label, eval_labels, eval_indices, eval_n_missing_label = load_eval_data_for_crossing(train_json, eval_json)
    

    train_sample_indices = train_outputs.sample_indices.numpy()
    train_idx_to_pos = {int(idx): pos for pos, idx in enumerate(train_sample_indices)}
    train_tensor_positions = [np.array(train_idx_to_pos[int(idx)]) for idx in train_indices]



    train_feat_baseline, train_names_baseline = getBaselineFeatures(train_outputs, train_tensor_positions)
    train_feat_markov, train_names_markov = getMarkovianFeatures(train_outputs, train_tensor_positions)
    # Combiner toutes les features pour pouvoir piocher dedans via select_features
    train_X_base_markov = np.column_stack([train_feat_baseline, train_feat_markov])
    train_names_bm = train_names_baseline + train_names_markov

    eval_sample_indices = eval_outputs.sample_indices.numpy()
    eval_idx_to_pos = {int(idx): pos for pos, idx in enumerate(eval_sample_indices)}
    eval_tensor_positions = [np.array(eval_idx_to_pos[int(idx)]) for idx in eval_indices]



    eval_feat_baseline, eval_names_baseline = getBaselineFeatures(eval_outputs, eval_tensor_positions)
    eval_feat_markov, eval_names_markov = getMarkovianFeatures(eval_outputs, eval_tensor_positions)
    # Combiner toutes les features pour pouvoir piocher dedans via select_features
    eval_X_base_markov = np.column_stack([eval_feat_baseline, eval_feat_markov])
    eval_names_bm = eval_names_baseline + eval_names_markov

    





    # Baseline only (on eval half)
    print("\n  === Baseline Features ===")
    res_baseline = run_logreg(train_feat_baseline, train_labels,eval_feat_baseline, eval_labels, train_names_baseline, "Baseline", )

    # Markovian only (on eval half)
    print("\n  === Markovian Features ===")
    res_markov = run_logreg(train_feat_markov, train_labels, eval_feat_markov, eval_labels, train_names_markov, "Markovian", )

    # Baseline + Markovian
    print("\n  === Baseline + Markovian ===")
    res_bm = run_logreg(train_X_base_markov, train_labels, eval_X_base_markov, eval_labels, train_names_bm, "Baseline+Markov", )


    # Cleaned features (22 features selection)
    train_X_clean, train_names_clean = select_features(train_X_base_markov, train_names_bm, CLEANED_FEATURES)
    print("\n  === Cleaned Features (22 selection) ===")
    print(f"  Features: {train_names_clean}")
    eval_X_clean, eval_names_clean = select_features(eval_X_base_markov, eval_names_bm, CLEANED_FEATURES)


    res_clean = run_logreg(train_X_clean, train_labels, eval_X_clean, eval_labels, train_names_clean, "CleanedFeatures", )

    # --- SAVE ---
    save_dir = "PipelineTest/CrossTrainAndTest/eval"
    os.makedirs(save_dir, exist_ok=True)

    results = {
        "config": cfg['name'],
        "n_samples_train": len(train_labels),
        "n_correct_train": int(np.sum(train_labels == 0)),
        "n_halluc_train": int(np.sum(train_labels == 1)),
        "n_samples_eval": len(eval_labels),
        "n_correct_eval": int(np.sum(eval_labels == 0)),
        "n_halluc_eval": int(np.sum(eval_labels == 1)),
        "results": {
            "Baseline": res_baseline,
            "Markovian": res_markov,
            "Baseline+Markov": res_bm,
            "CleanedFeatures": res_clean,
        },
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

    return results


def main():
    res = {}
    for conf_key in CONFIGS.keys():
        conf = CONFIGS[conf_key]
        print(f"\n\n{'='*70}\n  Running config: {conf['name']}\n{'='*70}")
        results = run_config(conf)
        res[conf['name']] = results

    os.makedirs("PipelineTest/CrossTrainAndTest/eval", exist_ok=True)
    with open("PipelineTest/CrossTrainAndTest/eval/all_results.json", "w", encoding="utf-8") as f:
        json.dump(res, f, indent=2, ensure_ascii=False)





if __name__ == "__main__":
    main()