# PipelineTest/CrossTrainAndTest — Cross-dataset generalization

This folder measures whether a detector trained on one dataset (e.g. TriviaQA) **generalizes** to another (e.g. HotpotQA). It uses the same features and classifier as `Benchmark/Baseline_and_markovian_features.py`, but train and test come from two different configurations.

Run after step 2 (the `outputs_*.pt` files and evaluated JSONs of both datasets are required), from the repository root.

```
CrossBaselineAndMarkovianMethod.py  → train on A, test on B, for every (A, B) pair
utils.py                            → train / eval label loading
make_cross_dataset_tables.py        → ROC-AUC / PR-AUC LaTeX tables
```

---

## 1. `CrossBaselineAndMarkovianMethod.py`

```bash
uv run --extra lladadream python PipelineTest/CrossTrainAndTest/CrossBaselineAndMarkovianMethod.py
```

The script iterates over every entry of the `CONFIGS` dictionary defined **inside the script**. Each entry describes a train → test pair:

```python
"llada16_32tokens_2500samples_triviaqa_eval_hotpotqa_seed42": {
    "name":              "llada_16steps_32tokens_triviaqa_2500samples_eval_hotpotqa_seed42",
    "train_output_path": "PipelineTest/res/results_<triviaqa config>/outputs_<...>.pt",
    "train_json":        "PipelineTest/res/eval_V2/results_<triviaqa config>.json",
    "eval_output_path":  "PipelineTest/res/results_<hotpotqa config>/outputs_<...>.pt",
    "eval_json":         "PipelineTest/res/eval_V2/results_<hotpotqa config>.json",
    "tokenizer_path":    "...",
},
```

The provided configurations cover LLaDA and Dream at 16 steps / 32 tokens, for all 9 pairs (3 × 3) between TriviaQA, NaturalQuestions and HotpotQA, diagonal included. `name` must follow `<model>_<steps>steps_<tokens>tokens_<train>_<N>samples_eval_<test>_seed<seed>`, the format expected by `make_cross_dataset_tables.py`.

For each pair:
1. **Labels:** `utils.load_eval_data_for_crossing` loads the labels of both JSONs (only `yes`/`no` are kept, deduplicated by `index`).
2. **Features:** extracted from each history (`get_baseline_features`, `get_markovian_features(k_tokens=64)`).
3. **Models:** four logistic regressions are trained on **all** of dataset A (5-fold `GridSearchCV`, `C` × `l1/l2`) and tested on **all** of dataset B: `Baseline`, `Markovian`, `Baseline+Markov`, `CleanedFeatures`.

### Outputs

| File | Contents |
|------|----------|
| `PipelineTest/CrossTrainAndTest/eval/Baseline_Markov_<name>.json` | Metrics for the pair (ROC-AUC, PR-AUC, accuracy, best threshold, confusion matrix, coefficients) |
| `PipelineTest/CrossTrainAndTest/eval/all_results.json` | All results, keyed by `name` |

---

## 2. `make_cross_dataset_tables.py`

Generates the LaTeX tables (rows = training dataset, columns = test dataset, diagonal in bold). Several `all_results*.json` files can be merged, e.g. LLaDA/Dream on one side and DiffusionGemma on the other.

```bash
uv run --extra lladadream python PipelineTest/CrossTrainAndTest/make_cross_dataset_tables.py \
    PipelineTest/CrossTrainAndTest/eval/all_results.json \
    PipelineTest/CrossTrainAndTest/eval/all_results_diffugemma.json \
    --out PipelineTest/res/tables/tables_cross_dataset_all_results.tex
```

| Argument | Default | Description |
|----------|---------|-------------|
| `results` (positional) | — | One or more aggregated JSON files |
| `--methods` | all | `Baseline`, `Markovian`, `Baseline+Markov`, `CleanedFeatures` |
| `--steps` | all | Filter on the number of steps |
| `--per-model-tables` | off | Adds tables comparing the configurations of a single model |
| `--out` | `PipelineTest/res/tables/tables_cross_dataset.tex` | Output `.tex` file |

By default, the script produces **one table per (steps, tokens) × method**, with one model per row. Only models with a complete 3 × 3 grid are included, and at least two are required. The reference table of the report is `CleanedFeatures`, displayed as "PipelineFinale".

The generated tables (captions, column headers) are in French.

Required LaTeX preamble: `\usepackage{booktabs, multirow, graphicx, float}`.
