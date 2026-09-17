# PipelineTest/analysis/plots — Plotting library

This folder contains the visualization modules for analyzing hallucinations in discrete diffusion models. Each module exposes a `plot(config_name=..., ...)` function, called by [`../run_analysis.py`](../README.md) or standalone.

## Usage

```bash
# Through the dispatcher
uv run --extra lladadream python PipelineTest/analysis/run_analysis.py --config <key> --mode heatmaps,benchmark_global

# Standalone
uv run --extra lladadream python PipelineTest/analysis/plots/exp_fit_global.py --config <key>
uv run --extra lladadream python PipelineTest/analysis/plots/ar1_global_trajectory.py --config <key>
uv run --extra lladadream python -m PipelineTest.analysis.plots.heatmaps --config <key>
```

`<key>` is a key of `PipelineTest/eval_configs.py::CONFIGS`.

---

## Modes

| Mode | File | Data | Description |
|------|------|------|-------------|
| `heatmaps` | `heatmaps.py` | `PlotContext` | Heatmaps of the AR(1) parameters (φ, intercept, σ) of the correct / halluc models, feature correlation and label-correlation bars |
| `ar1_per_sample` | `ar1_per_sample.py` | `PlotContext` | Mean / variance of masked entropy per sample: data vs AR(1) reconstruction |
| `ar1_per_trajectory` | `ar1_per_trajectory.py` | `PlotContext` | Trajectories of 10 tokens evenly spaced along the unmasking order, data vs AR(1) |
| `benchmark_global` | `benchmark_global.py` | `PlotContext` | Population trajectories (mean / variance of entropy and log-prob), mean vs variance scatter, top features by AUC |
| `benchmark_per_sample` | `benchmark_per_sample.py` | `PlotContext` | Histograms of baseline / Markovian features (correct vs halluc), per-step features |
| `plotly_samples` | `plotly_samples.py` | `PlotContext` | Plotly HTML (entropy / log-prob with proposed tokens on hover), via `GenerateBaseSamplerOutputsAndExtractInfo/PlotResults.py` |
| `exp_fit_trajectories` | `exp_fit_trajectories.py` | `PlotContext` | `c·exp(−t/τ)` fit (log-linear regression) on mean trajectories, correct vs halluc |
| `exp_fit_global` | `exp_fit_global.py` | standalone | `c·exp(−t/τ)` and `C·t·exp(−t/τ)` fits on the mean / variance of masked entropy |
| `ar1_global_trajectory` | `ar1_global_trajectory.py` | standalone | Population AR(1): mean / variance, data vs reconstruction |

"Standalone" modules only import `eval_configs.py` and load the `.pt` and JSON themselves.

---

## Output layout

Plots are saved **next to the history** (`dirname(outputs_path)`), in `plots/<mode>/`:

```
PipelineTest/res/results_<suffix>/
├── outputs_<suffix>.pt
└── plots/
    ├── heatmaps/
    │   ├── ar1_heatmap_correct[_nopad].png
    │   ├── ar1_heatmap_halluc[_nopad].png
    │   └── correlation_heatmap.png
    ├── ar1_per_sample/ar1_per_sample_mean_var[_nopad].png
    ├── ar1_per_trajectory/ar1_per_trajectory[_nopad].png
    ├── benchmark_global/
    │   ├── trajectories_mean_var.png
    │   ├── trajectories_mean_var_logprob.png
    │   ├── scatter_mean_vs_var.png
    │   └── scatter_top_features.png
    ├── benchmark_per_sample/
    │   ├── *feature_distributions.png
    │   └── per_step_features.png
    ├── plotly_samples/*.html
    ├── exp_fit/exp_fit_mean.png          # exp_fit_trajectories
    ├── exp_fit_mean_var.png              # exp_fit_global
    └── ar1_population_mean_var.png       # ar1_global_trajectory
```

- **`_nopad` suffix:** version that excludes padding tokens (detected through the `pad_token_id` of the saved tokenizer).
- **No suffix:** all tokens are included. The AR(1) modes produce both variants when `no_padding=True`.

---

## Architecture

```
plots/
├── __init__.py              # MODES list
├── utils.py                 # PlotContext
├── heatmaps.py
├── ar1_per_sample.py
├── ar1_per_trajectory.py
├── ar1_global_trajectory.py
├── benchmark_global.py
├── benchmark_per_sample.py
├── plotly_samples.py
├── exp_fit_global.py
└── exp_fit_trajectories.py
```

### `PlotContext` (`utils.py`)

Loads a config's data **once** and shares it across modes:

| Attribute / method | Contents |
|--------------------|----------|
| `outputs` | Merged history (`features.utils.load_outputs`) |
| `positions`, `labels`, `data` | Samples matched to the evaluated JSON (`match_samples`) |
| `entropies`, `logprobs`, `masks` | `(steps, tokens)` matrices per sample |
| `pad_token_id`, `padding_2d` | `(N, D)` padding if the tokenizer is available |
| `ar1_idx`, `eval_idx` | Stratified 50/50 split (seed 42): one half to fit the AR(1), the other to plot |
| `get_ar1_models(no_padding)` | Cached fit of the correct and halluc AR(1) models |
| `get_save_dir(mode)` | `dirname(outputs_path)/plots/<mode>/` (created if needed) |
| `select_samples(n)` | Balanced correct / halluc selection from the `eval` half |

### Adding a mode

1. Create `plots/<mode>.py` with a `plot(config_name, ...)` function that uses `PlotContext(config_name)` and `ctx.get_save_dir("<mode>")`.
2. Add `"<mode>"` to `MODES` in `__init__.py`.
3. Import the module and register it in `MODE_DISPATCH` in `../run_analysis.py`.
