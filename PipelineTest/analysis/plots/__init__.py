"""
scripts.plots
=============
Modular plotting library for hallucination detection analysis.

Modes:
- heatmaps: AR(1) parameter heatmaps (phi, intercept, sigma), correlation heatmaps
- ar1_per_sample: Mean/Var masked entropy per sample (data vs AR1 recon, aggregated over tokens)
- ar1_per_trajectory: Individual token entropy trajectories per sample (data vs AR1)
- benchmark_global: Population-level plots (mean/var trajectories, scatter, distributions)
- benchmark_per_sample: Per-sample feature histograms and distributions
- plotly_samples: Interactive Plotly visualizations with proposed text and entropy maps
- exp_fit_global: Exponential fit (c*exp and C*t*exp) on mean masked entropy trajectories
- exp_fit_trajectories: Exponential fit on mean unmasked entropy trajectories (PlotContext)
- ar1_global_trajectory: AR(1) population-level data vs reconstruction (mean/var)

Each module exposes a `plot(...)` function and can be called standalone or from run_analysis.py.
"""

MODES = [
    "heatmaps",
    "ar1_per_sample",
    "ar1_per_trajectory",
    "benchmark_global",
    "benchmark_per_sample",
    "plotly_samples",
    "exp_fit_global",
    "exp_fit_trajectories",
    "ar1_global_trajectory",
]
