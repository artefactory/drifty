# PipelineTest

End-to-end hallucination detection pipeline based on the diffusion history: **sample → evaluate → benchmark**, plus the analyses that build on the same configurations. Overview and commands: [root README](../README.md).

All scripts are run **from the repository root** (paths in `eval_configs.py` are relative to it).

## Contents

| Item | Step | Purpose |
|------|------|---------|
| [`sampling/`](sampling/README.md) | 1 + 2 | Answer generation with full history (`sample_answers_*.py`) and LLM judging (`eval.py`) |
| [`vllm_utils/`](vllm_utils/README.md) | 2 | Start / stop the judge vLLM server |
| `eval_configs.py` | 2 → 3 | Central `CONFIGS` registry: output paths of each configuration |
| [`Benchmark/`](Benchmark/README.md) | 3 | Detection baselines and comparison (ROC-AUC / PR-AUC) |
| [`features/`](features/README.md) | 3 | Features extracted from the trajectory (baseline + Markovian) |
| [`FeatureSelection/`](FeatureSelection/README.md) | analysis | Feature selection (Stability Selection, mRMR, Boruta) |
| [`CrossTrainAndTest/`](CrossTrainAndTest/README.md) | analysis | Cross-dataset generalization + LaTeX tables |
| [`analysis/`](analysis/README.md) | analysis | Trajectory plots, fits, heatmaps |
| `res/` | — | Results (not versioned) |

## `eval_configs.py`

Pure data module (no heavy dependencies), imported by `Benchmark/`, `analysis/` and the DiffusionGemma scripts. Each entry describes a configuration that has been sampled **and** evaluated:

```python
CONFIGS = {
    "llada16_32tokens_2500samples_triviaqa_evalqwen_seed42": {
        "outputs_path":   "PipelineTest/res/results_<suffix>/outputs_<suffix>.pt",   # step 1
        "eval_json":      "PipelineTest/res/eval_V2/results_<suffix>.json",          # step 2
        "tokenizer_path": "PipelineTest/res/results_<suffix>/tokenizer_<suffix>.pt", # step 1
        "name":           "llada_16steps_32tokens_triviaqa_2500samples_evalqwen_seed42",
    },
}
```

- **Key:** `<model><steps>_<tokens>tokens_2500samples_<dataset>_evalqwen_seed42`. The model name (`llada`, `dream`, `diffgemma`) routes the scripts (sampler choice, `values/<llada|dream|gemma>/` subfolder).
- **`name`:** used in result file names.

`FeatureSelection/configs.py` and `CrossTrainAndTest/CrossBaselineAndMarkovianMethod.py` have their own `CONFIGS` dictionary, in the same format.

## `res/` layout

```
PipelineTest/res/
├── results_<suffix>/
│   ├── outputs_<suffix>.pt       # merged BaseSamplerOutputCompleteHistory
│   ├── tokenizer_<suffix>.pt
│   └── plots/<mode>/             # analysis/ outputs
├── to_eval_V2/results_<suffix>.json    # generated answers (step 1)
├── eval_V2/results_<suffix>.json       # + is_hallucination / explanation (step 2)
├── FeatureSelection/                   # FeatureSelection/ outputs
└── tables/                             # CrossTrainAndTest/ LaTeX tables
```
