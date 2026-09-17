# PipelineTest/Benchmark — Step 3

This folder compares **hallucination detectors** on the configurations that have been sampled (step 1) and evaluated (step 2), registered in [`PipelineTest/eval_configs.py`](../README.md#eval_configspy). Each method produces one score per sample, evaluated with ROC-AUC and PR-AUC, with `is_hallucination = yes` as the positive class.

Commands are run from the repository root.

```
main.py                              → orchestrator: all baselines × all CONFIGS
data_split.py                        → label loading + shared train/test split
values_csv.py                        → in-place update of the per-sample CSVs

perplexity.py                        → Perplexity (tokens at unmasking time)            CPU
LNEntropy.py                         → Length-normalized entropy                        CPU
Baseline_and_markovian_features.py   → LogReg on baseline / Markovian features          CPU
semantic_entropy.py                  → Semantic entropy (N variants + NLI)              GPU, LLaDA / Dream
lexical_similarity.py                → ROUGE-L lexical similarity (N variants + LogReg) GPU, LLaDA / Dream
semantic_entropy_diffugemma.py       → Semantic entropy for DiffusionGemma              GPU, PEP 723 script
lexical_similarity_diffgemma.py      → Lexical similarity for DiffusionGemma            GPU, PEP 723 script

plot_roc_pr_curves.py                → ROC / PR curves from the per-sample CSVs
```

---

## Shared protocol (`data_split.py`)

- **`load_eval_data(eval_json)`:** keeps samples whose `is_hallucination` is `yes` (label 1) or `no` (label 0), deduplicated by `index`. `unclear` and missing labels are excluded.
- **`get_train_test_split(n, 0.8)`:** **sequential** split: the first 80 % of the pool is train, the remaining 20 % is test. All methods are evaluated on the same test set.
- **`get_train_test_split_for_config(...)`:** same split. It also rewrites the `split` column of the CSV `values/<model>/<config>.csv` to match the current partition (rows outside the pool become `excluded`).

Unsupervised methods (perplexity, LN-entropy, semantic entropy) are scored directly on the test set. Supervised methods (Baseline/Markovian, lexical similarity) train a logistic regression (`StandardScaler` + `LogisticRegression`, 5-fold `GridSearchCV` over `C` ∈ {1e-3…100} × `l1/l2`) on the train set.

---

## Running the benchmark (`main.py`)

```bash
# All CONFIGS, CPU baselines only
uv run --extra lladadream python PipelineTest/Benchmark/main.py

# Selected configs and baselines
uv run --extra lladadream python PipelineTest/Benchmark/main.py \
    --configs llada16_32tokens_2500samples_triviaqa_evalqwen_seed42 dream16_32tokens_2500samples_triviaqa_evalqwen_seed42 \
    --benchmarks perplexity ln_entropy baseline_markov

# + GPU baselines (via torchrun, nproc capped at the number of visible GPUs)
uv run --extra lladadream python PipelineTest/Benchmark/main.py --include-gpu --nproc 2 --seed 42
```

| Argument | Default | Description |
|----------|---------|-------------|
| `--benchmarks` | CPU (or all with `--include-gpu`) | `perplexity`, `ln_entropy`, `baseline_markov`, `semantic_entropy`, `lexical_similarity` |
| `--configs` | all `CONFIGS` | Keys from `eval_configs.py` |
| `--include-gpu` | off | Enables `semantic_entropy` and `lexical_similarity` |
| `--nproc` | `2` | GPUs for GPU baselines |
| `--seed` | `42` | Seed for stochastic variant generation |
| `--output_dir` | `Benchmark/eval` | Consolidated JSON |
| `--values_dir` | `Benchmark/values` | Per-sample CSVs |

- **CPU baselines:** run in the current process.
- **GPU baselines:** each one is launched in a `torch.distributed.run` subprocess whose logs are streamed live.

### Outputs

| File | Contents |
|------|----------|
| `Benchmark/eval/benchmarks_<all\|config>.json` | `{config: {baseline: {roc_auc, pr_auc, roc_auc_se, elapsed_seconds}}}`. `roc_auc_se` is the Hanley & McNeil standard error. |
| `Benchmark/values/<llada\|dream\|gemma>/<config>.csv` | One row per sample: `sample_index`, `label_hallucination`, `split`, then one column per score |

CSV score columns: `semantic_entropy`, `perplexity`, `ln_entropy`, `lexical_similarity`, `rouge_l_max_f1`, `rouge_l_mean_f1`, `rouge_l_max_prec`, `rouge_l_max_rec`, `rouge_l_var`, `baseline`, `markov`, `baseline_markov`.

A `ROC-AUC / PR-AUC` summary table per config × baseline is printed at the end of the run.

---

## Baselines

### Perplexity — `perplexity.py` (CPU)

Each token is taken **at the step where it is unmasked**. Score: `exp(-mean(log p))` over these probabilities.

### LN-Entropy — `LNEntropy.py` (CPU)

Mean Shannon entropy of tokens at the step where they are unmasked (`features.baseline.mean_entropy_just_unmasked`).

### Baseline + Markovian — `Baseline_and_markovian_features.py` (CPU)

Extracts the features of [`PipelineTest/features`](../features/README.md) from the history, then trains four logistic regressions:

| Model | Features |
|-------|----------|
| `Baseline` | `get_baseline_features` (means / variances of entropy and log-prob, curve shape) |
| `Markovian` | `get_markovian_features(k_tokens=64)` (α, β, C, τ, m) |
| `Baseline+Markov` | both |
| `CleanedFeatures` | `CLEANED_FEATURES` subset |

Writes `Benchmark/eval/Baseline_Markov_<name>.json` (metrics, coefficients, confusion matrix) and the `baseline`, `markov`, `baseline_markov` CSV columns.

### Semantic entropy — `semantic_entropy.py` (GPU, LLaDA / Dream)

1. Generates `N = 5` variants per question (`temperature = 0.5`, same number of steps and tokens as the config).
2. Clusters the variants by bidirectional entailment with `microsoft/deberta-v2-xlarge-mnli`.
3. Score: Shannon entropy of the cluster size distribution.

The generation model and the NLI pipeline are loaded only once for the whole run.

### Lexical similarity — `lexical_similarity.py` (GPU, LLaDA / Dream)

1. Generates `N = 5` variants and computes ROUGE-L between the first one and the others: `max_f1`, `mean_f1`, `max_prec`, `max_rec`, `var`.
2. Trains a logistic regression on these 5 features. Its score goes into the `lexical_similarity` column.

### Running a single baseline

Each baseline can also run on its own and writes its JSON to `Benchmark/eval/`. In standalone mode, only `Baseline_and_markovian_features.py` and `lexical_similarity.py` update their CSV columns, without touching the others; for the other baselines, `main.py` writes the CSV.

```bash
uv run --extra lladadream python PipelineTest/Benchmark/perplexity.py --config all
uv run --extra lladadream python PipelineTest/Benchmark/LNEntropy.py  --config <key>
uv run --extra lladadream python PipelineTest/Benchmark/Baseline_and_markovian_features.py --config <key|all> [--no_update_csv]
uv run --extra lladadream torchrun --nproc_per_node=2 PipelineTest/Benchmark/semantic_entropy.py   --config <key> --n_variants 5
uv run --extra lladadream torchrun --nproc_per_node=2 PipelineTest/Benchmark/lexical_similarity.py --config <key> --n_variants 5
```

---

## DiffusionGemma

**CPU** baselines (perplexity, LN-entropy, Baseline+Markovian) go through `main.py` as for the other models, with `--configs diffgemma...`. Run them **first**: they create `values/gemma/<config>.csv`, which the GPU scripts below then fill in.

**GPU** baselines use two standalone scripts. Their dependencies are declared in a PEP 723 header (torch 2.11 cu128, transformers 5.16.1, diffusers 0.40.0): `uv run` creates a dedicated environment, independent of the project `.venv`.

```bash
uv run PipelineTest/Benchmark/semantic_entropy_diffugemma.py \
    --config diffgemma16_32tokens_2500samples_triviaqa_evalqwen_seed42 --gen_batch_size 4 --nli_batch_size 32

uv run PipelineTest/Benchmark/lexical_similarity_diffgemma.py \
    --config diffgemma16_32tokens_2500samples_triviaqa_evalqwen_seed42 --gen_batch_size 4
```

- **Config name:** parameters are parsed from the name, which must follow `diffgemma<steps>_<tokens>tokens_2500samples_<dataset>_evalqwen_seed42` (`num_inference_steps = steps`, `canvas_length = tokens`).
- **GPU:** 1 GPU. `CUDA_VISIBLE_DEVICES=0` is forced and the weight overflow is offloaded to CPU. `--multi_gpu` disables this.
- **Batch:** `--gen_batch_size 4` fits on a 48 GB card.
- **Generation:** `N = 5` variants at `temperature = 0.5` with `DiffusionGemmaPipeline`. Each sequence is truncated at the first EOS and the `thought\n` prefix is removed.

| Script | Outputs |
|--------|---------|
| `semantic_entropy_diffugemma.py` | `eval/semantic_entropy_<config>.json` (+ `_variants.json`, checkpoint after generation), `semantic_entropy_diffugemma` column of `values/gemma/<config>.csv` |
| `lexical_similarity_diffgemma.py` | `eval/lexical_similarity_<config>.json`, `rouge_l_*` + `lexical_similarity` columns of `values/gemma/<config>.csv` |

---

## ROC / PR curves (`plot_roc_pr_curves.py`)

Walks through `values/<model>/*.csv` and draws, for each config, one ROC figure and one PR figure overlaying the `semantic_entropy`, `perplexity`, `ln_entropy`, `lexical_similarity` and `baseline_markov` columns.

```bash
uv run --extra lladadream python PipelineTest/Benchmark/plot_roc_pr_curves.py \
    --values-dir PipelineTest/Benchmark/values --output-dir PipelineTest/Benchmark/plots --split test
```

| Argument | Default |
|----------|---------|
| `--values-dir` | `Benchmark/values_test` |
| `--output-dir` | `Benchmark/plots` |
| `--split` | `test` (`train`, `test`, `all`) |

Outputs: `plots/<model>/<config>_roc.png` and `plots/<model>/<config>_pr.png`.
