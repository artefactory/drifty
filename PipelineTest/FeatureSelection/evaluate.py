"""Evaluation of selected feature subsets.

Two modes:
- nested_cv_evaluate_with_selection: leakage-free nested CV (feature selection inside fold)
- benchmark_evaluate: uses the same 80/20 sequential split as Baseline_and_markovian_features.py
"""

import numpy as np
from sklearn.model_selection import StratifiedKFold, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import roc_auc_score, average_precision_score, accuracy_score
from sklearn.pipeline import Pipeline
from PipelineTest.FeatureSelection.stability_selection import stability_selection
from PipelineTest.FeatureSelection.mrmr_selection import mrmr_selection
from PipelineTest.Benchmark.data_split import get_train_test_split


CLASSIFIERS = {
    "LR": {
        "model": LogisticRegression(solver="liblinear", max_iter=1000),
        "param_grid": {"clf__C": [0.001, 0.01, 0.1, 1, 10, 100], "clf__penalty": ["l1", "l2"]},
    },
    "RF": {
        "model": RandomForestClassifier(n_jobs=-1, random_state=42),
        "param_grid": {"clf__n_estimators": [100, 300], "clf__max_depth": [3, 5, 10, None]},
    },
    "LGBM": {
        "model": GradientBoostingClassifier(random_state=42),
        "param_grid": {
            "clf__n_estimators": [100, 300],
            "clf__max_depth": [3, 5],
            "clf__learning_rate": [0.01, 0.1],
        },
    },
}


def _select_features_on_train(X_train, y_train, feature_names, method="stability"):
    """Run feature selection on training data only."""
    if method == "stability":
        selected, _, _ = stability_selection(
            X_train, y_train, feature_names, n_bootstrap=200, pi_threshold=0.6,
        )
    elif method == "mrmr":
        all_sel, scores = mrmr_selection(
            X_train, y_train, feature_names, n_select=min(10, len(feature_names)),
        )
        selected = [f for f in all_sel if scores[f] > 0]
        if not selected:
            selected = all_sel[:3]
    else:
        selected = feature_names

    if not selected:
        selected = feature_names
    return selected


def nested_cv_evaluate(X, y, feature_names, selected_names,
                       outer_folds=5, inner_folds=5, random_state=42):
    """Evaluate a feature subset with nested CV (selection done outside).

    Use this for quick evaluation of a pre-determined subset.
    For leakage-free evaluation, use nested_cv_evaluate_with_selection.
    """
    selected_idx = [feature_names.index(n) for n in selected_names]
    X_sel = X[:, selected_idx]

    outer_cv = StratifiedKFold(n_splits=outer_folds, shuffle=True, random_state=random_state)
    inner_cv = StratifiedKFold(n_splits=inner_folds, shuffle=True, random_state=random_state)

    results = {}

    for clf_name, clf_spec in CLASSIFIERS.items():
        pipe = Pipeline([
            ("scaler", StandardScaler()),
            ("clf", clf_spec["model"]),
        ])

        fold_auroc = []
        fold_auprc = []

        for train_idx, test_idx in outer_cv.split(X_sel, y):
            X_train, X_test = X_sel[train_idx], X_sel[test_idx]
            y_train, y_test = y[train_idx], y[test_idx]

            gs = GridSearchCV(
                pipe, clf_spec["param_grid"], cv=inner_cv,
                scoring="roc_auc", n_jobs=-1, refit=True,
            )
            gs.fit(X_train, y_train)

            if hasattr(gs.best_estimator_.named_steps["clf"], "predict_proba"):
                y_prob = gs.predict_proba(X_test)[:, 1]
            else:
                y_prob = gs.decision_function(X_test)

            fold_auroc.append(roc_auc_score(y_test, y_prob))
            fold_auprc.append(average_precision_score(y_test, y_prob))

        results[clf_name] = {
            "auroc_mean": float(np.mean(fold_auroc)),
            "auroc_std": float(np.std(fold_auroc)),
            "auprc_mean": float(np.mean(fold_auprc)),
            "auprc_std": float(np.std(fold_auprc)),
            "fold_auroc": fold_auroc,
            "fold_auprc": fold_auprc,
        }

    return results


def nested_cv_evaluate_with_selection(X, y, feature_names, selection_method="stability",
                                      outer_folds=5, inner_folds=5, random_state=42):
    """Leakage-free nested CV: feature selection is redone inside each outer fold.

    Args:
        X: (N, p) feature matrix.
        y: (N,) binary labels.
        feature_names: list[str] of length p.
        selection_method: "stability" or "mrmr".
        outer_folds: number of outer CV folds.
        inner_folds: number of inner CV folds.
        random_state: seed.

    Returns:
        results: dict[str, dict] with AUROC/AUPRC per classifier.
        fold_selections: list of selected feature names per outer fold.
    """
    outer_cv = StratifiedKFold(n_splits=outer_folds, shuffle=True, random_state=random_state)
    inner_cv = StratifiedKFold(n_splits=inner_folds, shuffle=True, random_state=random_state)

    results = {}
    fold_selections = []

    for clf_name, clf_spec in CLASSIFIERS.items():
        fold_auroc = []
        fold_auprc = []

        for train_idx, test_idx in outer_cv.split(X, y):
            X_train, X_test = X[train_idx], X[test_idx]
            y_train, y_test = y[train_idx], y[test_idx]

            # Feature selection on train only
            selected = _select_features_on_train(X_train, y_train, feature_names, selection_method)
            if clf_name == list(CLASSIFIERS.keys())[0]:
                fold_selections.append(selected)

            sel_idx = [feature_names.index(n) for n in selected]
            X_train_sel = X_train[:, sel_idx]
            X_test_sel = X_test[:, sel_idx]

            pipe = Pipeline([
                ("scaler", StandardScaler()),
                ("clf", clf_spec["model"]),
            ])
            gs = GridSearchCV(
                pipe, clf_spec["param_grid"], cv=inner_cv,
                scoring="roc_auc", n_jobs=-1, refit=True,
            )
            gs.fit(X_train_sel, y_train)

            if hasattr(gs.best_estimator_.named_steps["clf"], "predict_proba"):
                y_prob = gs.predict_proba(X_test_sel)[:, 1]
            else:
                y_prob = gs.decision_function(X_test_sel)

            fold_auroc.append(roc_auc_score(y_test, y_prob))
            fold_auprc.append(average_precision_score(y_test, y_prob))

        results[clf_name] = {
            "auroc_mean": float(np.mean(fold_auroc)),
            "auroc_std": float(np.std(fold_auroc)),
            "auprc_mean": float(np.mean(fold_auprc)),
            "auprc_std": float(np.std(fold_auprc)),
            "fold_auroc": fold_auroc,
            "fold_auprc": fold_auprc,
        }

    return results, fold_selections


def benchmark_evaluate(X, y, feature_names, selected_names):
    """Evaluate using the same 80/20 sequential split as Baseline_and_markovian_features.py.

    Feature selection is done on the TRAIN split only, then evaluated on TEST.
    This matches the benchmark protocol exactly.

    Returns:
        results: dict[str, dict] with metrics per classifier.
    """
    train_idx, test_idx = get_train_test_split(len(y))

    X_train, X_test = X[train_idx], X[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]

    # Select features
    sel_idx = [feature_names.index(n) for n in selected_names]
    X_train_sel = X_train[:, sel_idx]
    X_test_sel = X_test[:, sel_idx]

    results = {}

    for clf_name, clf_spec in CLASSIFIERS.items():
        pipe = Pipeline([
            ("scaler", StandardScaler()),
            ("clf", clf_spec["model"]),
        ])
        inner_cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        gs = GridSearchCV(
            pipe, clf_spec["param_grid"], cv=inner_cv,
            scoring="roc_auc", n_jobs=-1, refit=True,
        )
        gs.fit(X_train_sel, y_train)

        if hasattr(gs.best_estimator_.named_steps["clf"], "predict_proba"):
            y_prob = gs.predict_proba(X_test_sel)[:, 1]
        else:
            y_prob = gs.decision_function(X_test_sel)

        roc = roc_auc_score(y_test, y_prob)
        pr = average_precision_score(y_test, y_prob)
        acc = accuracy_score(y_test, (y_prob >= 0.5).astype(int))

        results[clf_name] = {
            "auroc": float(roc),
            "auprc": float(pr),
            "accuracy": float(acc),
            "n_train": int(len(X_train_sel)),
            "n_test": int(len(X_test_sel)),
            "best_params": {k: str(v) for k, v in gs.best_params_.items()},
        }

    return results


def benchmark_evaluate_with_selection(X, y, feature_names, selection_method="stability"):
    """80/20 split with feature selection on train only (no leakage).

    Matches the exact same train/test partition as Baseline_and_markovian_features.py.
    """
    train_idx, test_idx = get_train_test_split(len(y))

    X_train, X_test = X[train_idx], X[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]

    # Feature selection on train only
    selected = _select_features_on_train(X_train, y_train, feature_names, selection_method)

    sel_idx = [feature_names.index(n) for n in selected]
    X_train_sel = X_train[:, sel_idx]
    X_test_sel = X_test[:, sel_idx]

    results = {}

    for clf_name, clf_spec in CLASSIFIERS.items():
        pipe = Pipeline([
            ("scaler", StandardScaler()),
            ("clf", clf_spec["model"]),
        ])
        inner_cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        gs = GridSearchCV(
            pipe, clf_spec["param_grid"], cv=inner_cv,
            scoring="roc_auc", n_jobs=-1, refit=True,
        )
        gs.fit(X_train_sel, y_train)

        if hasattr(gs.best_estimator_.named_steps["clf"], "predict_proba"):
            y_prob = gs.predict_proba(X_test_sel)[:, 1]
        else:
            y_prob = gs.decision_function(X_test_sel)

        roc = roc_auc_score(y_test, y_prob)
        pr = average_precision_score(y_test, y_prob)
        acc = accuracy_score(y_test, (y_prob >= 0.5).astype(int))

        results[clf_name] = {
            "auroc": float(roc),
            "auprc": float(pr),
            "accuracy": float(acc),
            "n_train": int(len(X_train_sel)),
            "n_test": int(len(X_test_sel)),
            "n_features": len(selected),
            "selected_features": selected,
            "best_params": {k: str(v) for k, v in gs.best_params_.items()},
        }

    return results, selected
