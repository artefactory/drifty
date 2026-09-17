"""
plots.plotly_samples
====================
Mode: plotly_samples

Interactive Plotly visualizations for specific samples:
- Entropy heatmaps with proposed tokens on hover
- Log-probability heatmaps
- Mask evolution
Uses the existing PlotPipeline / PlotInfo infrastructure from AnalyseResults.

Usage:
    python -m PipelineTest.scripts.plots.plotly_samples --config llada --n-samples 10
"""

import os
import sys
import json
import argparse
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))

from .utils import PlotContext, CONFIGS


def _load_tokenizer(config_name):
    """Load tokenizer for the given config."""
    import torch
    cfg = CONFIGS[config_name]
    tokenizer_path = cfg.get("tokenizer_path")
    if tokenizer_path and os.path.exists(tokenizer_path):
        return torch.load(tokenizer_path, map_location="cpu", weights_only=False)
    # Fallback: try to get it from outputs
    from transformers import AutoTokenizer
    if "llada" in config_name:
        return AutoTokenizer.from_pretrained("GSAI-ML/LLaDA-8B-Instruct")
    else:
        return AutoTokenizer.from_pretrained("Dream-org/Dream-v0-Instruct-7B")


def _get_sample_indices_from_positions(ctx: PlotContext, positions):
    """Map internal positions back to original sample indices."""
    outputs = ctx.outputs
    sample_id = outputs.sample_indices.detach().cpu().numpy() if outputs.sample_indices is not None else np.arange(len(ctx.entropies))
    # Reverse map: pos -> original index
    rev_map = {int(sample_id[i]): i for i in range(len(sample_id))}
    # pos_to_orig: position -> original index
    pos_to_orig = {}
    for orig_idx, pos in rev_map.items():
        pos_to_orig[pos] = orig_idx
    # Actually, sample_id[i] is the orig index and i is the position
    # So we need: position -> orig_idx = sample_id[position]
    return [int(sample_id[pos]) for pos in positions]


def plot_plotly_samples(ctx: PlotContext, n_samples=10):
    """Generate interactive Plotly visualizations for selected samples."""
    save_dir = ctx.get_save_dir("plotly_samples")

    from PipelineTest.features.io import PlotInfo

    selected, sel_positions, sel_labels = ctx.select_samples(n_samples)

    # Get original indices for PlotPipeline
    list_orig_indices = _get_sample_indices_from_positions(ctx, sel_positions)

    # Load tokenizer
    tokenizer = _load_tokenizer(ctx.config_name)

    # Build title list
    title_list = []
    with open(ctx.cfg["eval_json"], encoding="utf-8") as f:
        eval_data = json.load(f)
    idx_to_data = {d["index"]: d for d in eval_data}

    for orig_idx, lbl in zip(list_orig_indices, sel_labels):
        label_str = "Correct" if lbl == 0 else "Halluc"
        question = idx_to_data.get(orig_idx, {}).get("question", "")
        question = question[:40] + "..." if len(question) > 40 else question
        title_list.append(f"[{label_str}] {question}")

    # Use PlotInfo which generates Plotly HTML
    print(f"    Generating Plotly plots for {len(list_orig_indices)} samples...")
    PlotInfo(
        list_orig_indices,
        tokenizer,
        title_list,
        save_path=save_dir,
        ouputpath=ctx.cfg["outputs_path"],
    )
    print(f"    Saved Plotly HTMLs to: {save_dir}")


def plot(config_name="llada", n_samples=10):
    """Main entry point for plotly_samples mode."""
    print("\n[plotly_samples] Loading data...")
    ctx = PlotContext(config_name, use_padding=True)

    print(f"[plotly_samples] Generating interactive plots for {n_samples} samples...")
    plot_plotly_samples(ctx, n_samples=n_samples)

    print("[plotly_samples] Done.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="llada", choices=["llada", "dream"])
    parser.add_argument("--n-samples", type=int, default=10)
    args = parser.parse_args()
    plot(args.config, args.n_samples)
