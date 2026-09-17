# PipelineTest/FeatureSelection — Feature selection

This folder selects, among the baseline + Markovian features of [`PipelineTest/features`](../features/README.md), those that actually detect hallucinations, then measures the performance of the selected subsets. Run after step 2, from the repository root.

## Quick start

```bash
# Full pipeline (prefilter → selection → evaluation) on the configs of configs.py
uv run --extra lladadream python PipelineTest/FeatureSelection/run_all.py

# Selection frequency of each feature across all configs
uv run --extra lladadream python PipelineTest/FeatureSelection/FeatureFrequency.py

# Correlation heatmap of features + label, per config
uv run --extra lladadream python PipelineTest/FeatureSelection/correlationHeatmap.py
```

## Pipeline

```
extraction (baseline + Markovian)
   → prefilter |r| > 0.95
   → selection: Stability Selection / mRMR / Boruta
   → 80/20 evaluation (selection redone on the train split only)
```

| File | Purpose |
|------|---------|
| `configs.py` | `CONFIGS` dictionary (same format as `eval_configs.py`), `SAVE_ROOT` and `extract_features_and_labels(cfg)`, which returns `(X, names, labels)` |
| `prefilter.py` | `correlation_prefilter`: for each pair with \|r\| > 0.95, keeps the feature most correlated with the label (point-biserial) |
| `stability_selection.py` | **Primary method.** Subsampling (50 %) + L1 LogReg with random `C`, keeps features whose selection probability is ≥ `pi_threshold`. Theoretical bound on E[false positives] (Meinshausen & Bühlmann, 2010) |
| `mrmr_selection.py` | Ablation. Greedy max-relevance min-redundancy, KNN mutual information (Peng et al., 2005) |
| `boruta_selection.py` | Ablation. Test against permuted "shadow features" with a Random Forest (Kursa & Rudnicki, 2010) |
| `evaluate.py` | `LR`, `RF`, `LGBM` (GradientBoosting) classifiers, each with 5-fold `GridSearchCV`; evaluation protocols (see below) |
| `run_all.py` | Per-config orchestration + figures + JSON |
| `FeatureFrequency.py` | Aggregates the features kept by Stability Selection (LR classifier) across all result JSONs and ranks them by frequency |
| `correlationHeatmap.py` | Correlation matrix (standardized features + label) |

### Evaluation protocols (`evaluate.py`)

| Function | Protocol |
|----------|----------|
| `benchmark_evaluate(X, y, names, selected)` | Sequential 80/20 split, identical to `Benchmark/`, with a fixed feature subset |
| `benchmark_evaluate_with_selection(X, y, names, method)` | Same split; selection (`stability`: 200 bootstraps, π = 0.6; `mrmr`: top-10 with score > 0) is redone **on the train split only** |
| `nested_cv_evaluate` / `nested_cv_evaluate_with_selection` | 5 × 5 nested CV, selection redone in each outer fold |

## What `run_all.py` does for each config

1. Extracts features and labels, then applies the correlation prefilter.
2. Runs Stability Selection on the whole pool (500 bootstraps, π = 0.6), then mRMR (top-10, score > 0) and Boruta.
3. Evaluates `stability` and `mrmr` with `benchmark_evaluate_with_selection`, and `all_features` (all prefiltered features) with `benchmark_evaluate`.
4. Saves the JSON and the figures.

The `main()` loop filters `CONFIGS` keys on a substring (currently `"gemma"`). Adjust this filter and the contents of `configs.py` to the configurations you want to process.

## Outputs

Everything is written under `PipelineTest/res/FeatureSelection/`:

```
PipelineTest/res/FeatureSelection/
├── results/feature_selection_<name>.json          # prefilter, selections, probabilities, scores, AUROC/AUPRC per classifier
├── results_top<n>/evaluation_top<n>_<name>.json   # FeatureFrequency.py (block to uncomment)
└── figures/
    ├── <name>/stability_probs_<name>.png          # Stability Selection probabilities
    ├── <name>/mrmr_scores_<name>.png              # mRMR scores
    └── Correlations/correlation_heatmap_<name>.png
```

`FeatureFrequency.py` reads `results/` and prints the selection frequency of each feature. The commented-out blocks at the end of the script re-evaluate the `top_n` most frequent features on each config (`evaluate_topn_features`) and compare the result with per-config Stability Selection (`compute_diff_stability_selection_and_topn_features`).
