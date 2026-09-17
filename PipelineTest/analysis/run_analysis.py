"""
scripts/run_analysis.py
=======================
Centralized analysis and plotting script.

Dispatches to modular plot modes in scripts/plots/:
- heatmaps: AR(1) parameter heatmaps, correlation heatmaps
- ar1_per_sample: Per-sample mean/var masked entropy (data vs AR1 recon)
- ar1_per_trajectory: Per-token entropy trajectories (10 tokens, data vs AR1)
- benchmark_global: Population-level mean/var trajectories, scatter plots
- benchmark_per_sample: Feature distribution histograms, per-step features
- plotly_samples: Interactive Plotly visualizations with proposed tokens
- exp_fit_global: Exponential fit (c*exp and C*t*exp) on mean masked entropy
- exp_fit_trajectories: Exponential fit on mean unmasked entropy trajectories
- ar1_global_trajectory: AR(1) population-level data vs reconstruction (mean/var)

Usage:
    python -m PipelineTest.scripts.run_analysis [--config CONFIG] [--mode MODE] [--n-samples 20]
    python -m PipelineTest.scripts.run_analysis --config llada --mode all
    python -m PipelineTest.scripts.run_analysis --config llada --mode heatmaps
    python -m PipelineTest.scripts.run_analysis --config llada --mode ar1_per_sample,ar1_per_trajectory
"""

import sys
import os
import argparse

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from PipelineTest.analysis.plots import MODES
from PipelineTest.analysis.plots import heatmaps
from PipelineTest.analysis.plots import ar1_per_sample
from PipelineTest.analysis.plots import ar1_per_trajectory
from PipelineTest.analysis.plots import benchmark_global
from PipelineTest.analysis.plots import benchmark_per_sample
from PipelineTest.analysis.plots import plotly_samples
from PipelineTest.analysis.plots import exp_fit_global
from PipelineTest.analysis.plots import exp_fit_trajectories
from PipelineTest.analysis.plots import ar1_global_trajectory


# =========================================================================
# MODE REGISTRY
# =========================================================================

MODE_DISPATCH = {
    "heatmaps": heatmaps.plot,
    "ar1_per_sample": ar1_per_sample.plot,
    "ar1_per_trajectory": ar1_per_trajectory.plot,
    "benchmark_global": benchmark_global.plot,
    "benchmark_per_sample": benchmark_per_sample.plot,
    "plotly_samples": plotly_samples.plot,
    "exp_fit_global": exp_fit_global.plot,
    "exp_fit_trajectories": exp_fit_trajectories.plot,
    "ar1_global_trajectory": ar1_global_trajectory.plot,
}


# =========================================================================
# MAIN
# =========================================================================

def main(config_name="llada", modes=None, n_samples=20, no_padding=True):
    """Run selected plot modes.
    
    Args:
        config_name: "llada" or "dream"
        modes: list of mode names or None for all
        n_samples: number of samples for per-sample modes
        no_padding: whether to exclude padding tokens
    """
    if modes is None:
        modes = MODES

    print(f"\n{'='*70}")
    print(f"  ANALYSIS & PLOTS: {config_name}")
    print(f"  Modes: {', '.join(modes)}")
    print(f"{'='*70}")

    for mode in modes:
        if mode not in MODE_DISPATCH:
            print(f"\n  [WARNING] Unknown mode: {mode}. Skipping.")
            continue

        fn = MODE_DISPATCH[mode]

        # Inspect signature to pass relevant kwargs
        import inspect
        sig = inspect.signature(fn)
        kwargs = {"config_name": config_name}
        if "n_samples" in sig.parameters:
            kwargs["n_samples"] = n_samples
        if "no_padding" in sig.parameters:
            kwargs["no_padding"] = no_padding

        fn(**kwargs)

    print(f"\n{'='*70}")
    print("  All requested plots generated.")
    print(f"{'='*70}\n")


if __name__ == "__main__":
    from PipelineTest.eval_configs import CONFIGS

    parser = argparse.ArgumentParser(description="Analysis and plotting (dispatches to plot modes)")
    parser.add_argument("--config", type=str, default="all",
                        help="Config key or 'all' to run on every config.")
    parser.add_argument("--mode", type=str, default="all",
                        help=f"Comma-separated modes or 'all'. Available: {', '.join(MODES)}")
    parser.add_argument("--n-samples", type=int, default=20, help="Number of samples for per-sample modes")
    parser.add_argument("--no-padding", action="store_true", default=True,
                        help="Exclude padding tokens (default: True)")
    args = parser.parse_args()

    if args.mode == "all":
        modes = MODES
    else:
        modes = [m.strip() for m in args.mode.split(",")]

    if args.config == "all":
        for config_key in CONFIGS:
            try:
                main(config_key, modes, args.n_samples, args.no_padding)
            except Exception as e:
                print(f"[ERROR] {config_key}: {e}")
                continue
    else:
        if args.config not in CONFIGS:
            print(f"[ERROR] Unknown config: {args.config}. Available: {list(CONFIGS.keys())}")
            sys.exit(1)
        main(args.config, modes, args.n_samples, args.no_padding)

