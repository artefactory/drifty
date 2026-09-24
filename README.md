# drifty — Hallucination detection in diffusion LLMs

This project studies **hallucination detection** in discrete diffusion language models (LLaDA, Dream, DiffusionGemma). It relies on the **denoising trajectory**: the full state of the canvas is recorded at every step (proposed tokens, confidence, entropy, mask), features are extracted from it, and they are compared against standard baselines (perplexity, semantic entropy, lexical similarity…).

The code is built on the dLLM library (`dllm/`, `examples/`, `scripts/`), extended with samplers that record the full history (`*SamplerWithCompleteHistory`).

---

## The 3-step pipeline

The order is fixed: each step consumes the outputs of the previous one.

```
 1. SAMPLE                       2. EVALUATE                      3. BENCHMARK
 ─────────                       ──────────                       ────────────
 Generate answers                Judge each answer                Score the hallucination
 + diffusion history             with an LLM judge (vLLM)         detectors and compare
                                                                  the baselines

 res/results_<cfg>/outputs.pt ──────────────────────────────┐
 res/to_eval_V2/results.json ──► res/eval_V2/results.json ──┴──► Benchmark/eval/*.json
                                 (+ is_hallucination)            Benchmark/values/<model>/<cfg>.csv
                                                                 ROC / PR curves
```

| Step | Script | Environment | Output |
|------|--------|-------------|--------|
| **1. Sampling** | `PipelineTest/sampling/sample_answers_dream_llada.py` <br> `PipelineTest/sampling/sample_answers_diffugemma.py` | `lladadream` <br> `diffugemma` | `PipelineTest/res/results_<cfg>/outputs_<cfg>.pt` <br> `PipelineTest/res/to_eval_V2/results_<cfg>.json` |
| **2. Evaluation** | `PipelineTest/sampling/eval.py` | either + separate vLLM server | `PipelineTest/res/eval_V2/results_<cfg>.json` |
| *(2 → 3)* | `PipelineTest/eval_configs.py` | — | `CONFIGS[<cfg>]` entry |
| **3. Benchmarks** | `PipelineTest/Benchmark/main.py` <br> `PipelineTest/Benchmark/*_diffugemma.py` | `lladadream` <br> PEP 723 script (`uv run`) | `PipelineTest/Benchmark/eval/`, `PipelineTest/Benchmark/values/` |

Further analyses (feature selection, cross-dataset generalization, trajectory plots) come **after step 2** and reuse the same `CONFIGS`: see [Further analyses](#further-analyses).

---

## 0. Installation

The project has **two incompatible environments** (different `transformers` versions), selected with a uv extra:

```bash
uv sync --extra lladadream   # LLaDA / Dream  — transformers 4.57, peft 0.17
uv sync --extra diffugemma   # DiffusionGemma — transformers 5.x, diffusers, tiktoken
```

The two extras are declared as conflicting (`[tool.uv] conflicts` in `pyproject.toml`): only one can live in `.venv` at a time. Running `uv sync` with the other extra switches the environment. To run a script, pass the extra:

```bash
uv run --extra lladadream python <script>.py
```

All commands below are run **from the repository root**: the paths in `CONFIGS` are relative (`PipelineTest/res/...`).

---

## 1. Sample

Generates answers to the questions (TriviaQA, NaturalQuestions, HotpotQA) and records the full diffusion history.

```bash
# LLaDA / Dream — multi-GPU via torchrun (one dataset shard per rank)
uv sync --extra lladadream
uv run --extra lladadream torchrun --nproc_per_node=2 \
    PipelineTest/sampling/sample_answers_dream_llada.py --num_sample 2500 --batch_size 8

# DiffusionGemma — single process, model split across GPUs 0 and 1
uv sync --extra diffugemma
uv run --extra diffugemma python \
    PipelineTest/sampling/sample_answers_diffugemma.py --num_sample 2500 --batch_size 16
```

Model × dataset × (steps, tokens) combinations are chosen by editing the lists at the top of the script (`SAMPLERS`, `DATASETS`, `STEP_TOKEN_COMBOS`, `GEMMA_CONFIGS`).

Each configuration has a suffix `<model>_<steps>steps_<tokens>tokens[_<canvas>canvas]_<dataset>_<N>samples[_brief]_seed<seed>` and produces:
- `PipelineTest/res/results_<suffix>/outputs_<suffix>.pt`: diffusion history (`BaseSamplerOutputCompleteHistory`);
- `PipelineTest/res/results_<suffix>/tokenizer_<suffix>.pt`: tokenizer;
- `PipelineTest/res/to_eval_V2/results_<suffix>.json`: `question`, `label` (gold answer first, then aliases), `answer`, `index`.

Details: [`PipelineTest/sampling/README.md`](PipelineTest/sampling/README.md).

## 2. Evaluate

An LLM judge (`Qwen/Qwen3.5-9B` by default) compares each generated answer with the reference answers and adds `is_hallucination` ∈ {`yes`, `no`, `unclear`} + `explanation`. Empty answers are labeled `yes` without calling the judge; `unclear` answers (refusals, "I don't know") are **excluded** from the benchmarks.

```bash
uv run --extra lladadream python PipelineTest/sampling/eval.py
```

- **Judge server:** vLLM runs in a **separate** uv project (`../eval_server`, next to the repository). `eval.py` starts and stops it through `PipelineTest/vllm_utils/vllm_server_manager.py`. Set `USE_SERVER_MANAGER = False` if a `vllm serve` is already running.
- **Settings:** everything is set in the `CONFIGURATION` block at the top of `eval.py`. `FILENAME_FILTER` selects which files of `to_eval_V2/` are evaluated (`None` = all).

### Register the configuration

Add an entry to `PipelineTest/eval_configs.py` so the next steps can find the files:

```python
"llada16_32tokens_2500samples_triviaqa_evalqwen_seed42": {
    "outputs_path":   "PipelineTest/res/results_<suffix>/outputs_<suffix>.pt",
    "eval_json":      "PipelineTest/res/eval_V2/results_<suffix>.json",
    "tokenizer_path": "PipelineTest/res/results_<suffix>/tokenizer_<suffix>.pt",
    "name":           "<suffix>_evalqwen",
},
```

The key must contain the model name (`llada`, `dream` or `diffgemma`), which is used to route scripts. DiffusionGemma scripts require the format `diffgemma<steps>_<tokens>tokens_2500samples_<dataset>_evalqwen_seed42`.

## 3. Benchmark

```bash
# CPU baselines (perplexity, LN-entropy, Baseline+Markovian) on all CONFIGS
uv run --extra lladadream python PipelineTest/Benchmark/main.py

# Subset of configs + GPU baselines (semantic entropy, lexical similarity — LLaDA/Dream)
uv run --extra lladadream python PipelineTest/Benchmark/main.py \
    --configs llada16_32tokens_2500samples_triviaqa_evalqwen_seed42 --include-gpu --nproc 2

# DiffusionGemma GPU baselines: standalone scripts (inline PEP 723 dependencies, 1 GPU)
uv run PipelineTest/Benchmark/semantic_entropy_diffugemma.py  --config diffgemma16_32tokens_2500samples_triviaqa_evalqwen_seed42
uv run PipelineTest/Benchmark/lexical_similarity_diffgemma.py --config diffgemma16_32tokens_2500samples_triviaqa_evalqwen_seed42

# ROC / PR curves from the per-sample CSVs
uv run --extra lladadream python PipelineTest/Benchmark/plot_roc_pr_curves.py --values-dir PipelineTest/Benchmark/values
```

All methods use the **same sequential 80/20 split** (`Benchmark/data_split.py`), on samples whose label is `yes` or `no`.

Details: [`PipelineTest/Benchmark/README.md`](PipelineTest/Benchmark/README.md).

---

## Further analyses

Run after step 2 (they read `outputs_path` + `eval_json`):

| Folder | Purpose |
|--------|---------|
| [`PipelineTest/features/`](PipelineTest/features/README.md) | Feature extraction (baseline + Markovian) from the history |
| [`PipelineTest/FeatureSelection/`](PipelineTest/FeatureSelection/README.md) | Prefilter, Stability Selection / mRMR / Boruta, evaluation |
| [`PipelineTest/CrossTrainAndTest/`](PipelineTest/CrossTrainAndTest/README.md) | Train on one dataset, test on another + LaTeX tables |
| [`PipelineTest/analysis/`](PipelineTest/analysis/README.md) | Trajectory plots, exponential / AR(1) fits, heatmaps |
| [`GenerateBaseSamplerOutputsAndExtractInfo/`](GenerateBaseSamplerOutputsAndExtractInfo/README.md) | Qualitative exploration on a few prompts + `get*` functions used by the whole pipeline |

---

## Layout

```
drifty/
├── pyproject.toml                            # lladadream / diffugemma extras
├── dllm/                                     # dLLM library (+ *WithCompleteHistory samplers)
├── examples/, scripts/                       # dLLM examples and training scripts
├── GenerateBaseSamplerOutputsAndExtractInfo/ # generation + extraction + visualization (exploration)
└── PipelineTest/
    ├── eval_configs.py                       # central configuration registry
    ├── sampling/                             # step 1 (sampling) + step 2 (eval.py)
    ├── vllm_utils/                           # judge vLLM server launcher
    ├── Benchmark/                            # step 3
    ├── features/                             # feature extraction
    ├── FeatureSelection/                     # feature selection
    ├── CrossTrainAndTest/                    # cross-dataset generalization
    ├── analysis/                             # analysis plots
    └── res/                                  # results (not versioned)
        ├── results_<cfg>/                    # outputs_<cfg>.pt, tokenizer_<cfg>.pt
        ├── to_eval_V2/                       # answers to judge  (step 1 output)
        └── eval_V2/                          # judged answers    (step 2 output)
```

Result files (`*.pt`, `*.json`, `*.csv`, `*.png`, `*.html`…) are ignored by git.

## Licenses and related repos

Our code and code modifications are released under MIT license.

The `dllm` library we use for inference was released under the Apache-2.0 license.
You may find the original source code and a copy of its license at: [https://github.com/ZHZisZZ/dllm](https://github.com/ZHZisZZ/dllm).

The TraceDet implementation, released under the MIT license, is taken from its original repo: [https://github.com/chang-sx/TraceDet/](https://github.com/chang-sx/TraceDet/).

