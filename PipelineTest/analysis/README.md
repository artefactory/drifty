# PipelineTest/analysis — Analysis plots

This folder visualizes diffusion trajectories, separating **correct answers** from **hallucinations**: population curves, feature distributions, exponential and AR(1) fits, heatmaps. Run after step 2, on the configurations of `PipelineTest/eval_configs.py`, from the repository root.

```
analysis/
├── run_analysis.py   # entry point: dispatches to the plot modes
└── plots/            # one module per mode (see plots/README.md)
```

## `run_analysis.py`

```bash
# All modes, all configs
uv run --extra lladadream python PipelineTest/analysis/run_analysis.py

# One config, one or more modes
uv run --extra lladadream python PipelineTest/analysis/run_analysis.py \
    --config llada16_32tokens_2500samples_triviaqa_evalqwen_seed42 --mode heatmaps
uv run --extra lladadream python PipelineTest/analysis/run_analysis.py \
    --config dream16_32tokens_2500samples_triviaqa_evalqwen_seed42 --mode ar1_per_sample,ar1_per_trajectory --n-samples 20
```

| Argument | Default | Description |
|----------|---------|-------------|
| `--config` | `all` | `CONFIGS` key, or `all` (a failing config is skipped) |
| `--mode` | `all` | Comma-separated modes, or `all` |
| `--n-samples` | `20` | Number of samples for the "per sample" modes |
| `--no-padding` | `True` | Excludes padding tokens when the mode supports it |

Each mode receives `config_name`, plus `n_samples` and `no_padding` if it accepts them.

## Modes

| Mode | Contents |
|------|----------|
| `heatmaps` | AR(1) parameters (φ, intercept, σ) per (step, token); correlation between features |
| `ar1_per_sample` | Mean / variance of masked entropy per sample: data vs correct / halluc AR(1) reconstruction |
| `ar1_per_trajectory` | Trajectories of 10 tokens (spaced along the unmasking order), data vs AR(1) |
| `ar1_global_trajectory` | Population-level AR(1): mean / variance, data vs reconstruction |
| `benchmark_global` | Mean / variance trajectories (entropy, log-prob), scatter plots, top features by AUC |
| `benchmark_per_sample` | Feature histograms (correct vs halluc), per-step evolution |
| `plotly_samples` | Interactive HTML: entropy / log-prob heatmap with proposed tokens on hover |
| `exp_fit_global` | `c·exp(−t/τ)` and `C·t·exp(−t/τ)` fits on the mean masked entropy |
| `exp_fit_trajectories` | Exponential fit on entropy-at-unmasking trajectories (R² printed) |

Each module can also run standalone, e.g. `python PipelineTest/analysis/plots/exp_fit_global.py --config <key>`. Details, outputs and architecture: [`plots/README.md`](plots/README.md).

Plots are written **next to the config's history**, under `PipelineTest/res/results_<suffix>/plots/`.
