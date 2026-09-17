"""Trace les courbes ROC et Precision-Recall pour chaque config CSV du dossier `values`.

Chaque fichier CSV sous `values/<model>/*.csv` correspond a une config (modele,
nb de steps, nb de tokens, dataset). Chaque colonne de score (semantic_entropy,
perplexity, ln_entropy, lexical_similarity, baseline, markov, baseline_markov)
est evaluee comme detecteur de `label_hallucination` sur le split de test.

Usage:
    uv run python PipelineTest/Benchmark/plot_roc_pr_curves.py
    uv run python PipelineTest/Benchmark/plot_roc_pr_curves.py --split test
"""

import argparse
import textwrap
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
)

plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Computer Modern Roman", "DejaVu Serif"],
    "mathtext.fontset": "cm",
    "font.size": 20,
    "axes.titlesize": 24,
    "axes.labelsize": 20,
    "xtick.labelsize": 20,
    "ytick.labelsize": 20,
    "legend.fontsize": 15,
})

BENCHMARK_DIR = Path(__file__).parent
DEFAULT_VALUES_DIR = BENCHMARK_DIR / "values_test"
DEFAULT_OUTPUT_DIR = BENCHMARK_DIR / "plots"

SCORE_COLUMNS = [
    "semantic_entropy",
    "perplexity",
    "ln_entropy",
    "lexical_similarity",
    # "baseline",
    # "markov",
    "baseline_markov",
]

LINESTYLES = ["-", "--", "-.", ":", "-", "--", "-."]
MARKERS = ["o", "s", "^", "D", "v", "P", "X"]


def iter_config_csvs(values_dir: Path):
    for model_dir in sorted(p for p in values_dir.iterdir() if p.is_dir()):
        for csv_path in sorted(model_dir.glob("*.csv")):
            yield model_dir.name, csv_path


def plot_config(model_name: str, csv_path: Path, output_dir: Path, split: str):
    df = pd.read_csv(csv_path)
    # Idem relabel_and_recompute_roc_pr : les lignes hors pool (split
    # "excluded") ne sont ni train ni test.
    df = df.loc[df["label_hallucination"]!=-1]
    if split == "all":
        df = df[df["split"].isin(["train", "test"])]
    else:
        df = df[df["split"] == split]

    if df.empty:
        print(f"  [SKIP] {csv_path.name}: aucune ligne sur le split '{split}'")
        return

    config_name = csv_path.stem
    fig_roc, ax_roc = plt.subplots(1, 1, figsize=(10, 8))
    fig_pr, ax_pr = plt.subplots(1, 1, figsize=(10, 8))
    plotted = 0
    cividis = plt.get_cmap("cividis")

    for i, col in enumerate(SCORE_COLUMNS):
        metric_df = df[["label_hallucination", col]].dropna()
        n_nan = len(df) - len(metric_df)
        print(f"  [INFO] {config_name} / {col}: {n_nan} NaN retire(s), "
              f"{len(metric_df)} ligne(s) conservee(s)")
        if metric_df.empty:
            print(f"  [SKIP] {config_name} / {col}: aucun score valide")
            continue

        y_true = metric_df["label_hallucination"].to_numpy()
        if y_true.sum() == 0 or y_true.sum() == len(y_true):
            print(f"  [SKIP] {config_name} / {col}: une seule classe apres retrait des NaN")
            continue

        y_score = metric_df[col].to_numpy()
        color = "#B8860B" if col == "baseline_markov" else cividis(
            0.12 + 0.76 * i / max(len(SCORE_COLUMNS) - 1, 1)
        )
        ls = LINESTYLES[i % len(LINESTYLES)]
        marker = MARKERS[i % len(MARKERS)]
        display_name = "Pipeline Finale" if col == "baseline_markov" else col

        fpr, tpr, _ = roc_curve(y_true, y_score)
        roc_auc = roc_auc_score(y_true, y_score)
        ax_roc.plot(
            fpr,
            tpr,
            color=color,
            linestyle=ls,
            linewidth=2.2,
            marker=marker,
            markersize=5,
            markevery=max(len(fpr) // 10, 1),
            label=f"{display_name} (AUC={roc_auc:.3f})",
        )

        precision, recall, _ = precision_recall_curve(y_true, y_score)
        pr_auc = average_precision_score(y_true, y_score)
        ax_pr.plot(
            recall,
            precision,
            color=color,
            linestyle=ls,
            linewidth=2.2,
            marker=marker,
            markersize=5,
            markevery=max(len(recall) // 10, 1),
            label=f"{display_name} (AP={pr_auc:.3f})",
        )
        plotted += 1

    if not plotted:
        plt.close(fig_roc)
        plt.close(fig_pr)
        print(f"  [SKIP] {csv_path.name}: aucune courbe tracable")
        return

    n_pos, n_neg = int(y_true.sum()), int(len(y_true) - y_true.sum())
    prevalence = y_true.mean()

    ax_roc.plot(
        [0, 1], [0, 1], color="#777777", linestyle="--", linewidth=1.2,
        label="Hasard",
    )
    ax_roc.set_xlim(0, 1)
    ax_roc.set_ylim(0, 1.02)
    ax_roc.set_xlabel("Taux de faux positifs (FPR)")
    ax_roc.set_ylabel("Taux de vrais positifs (TPR)")
    ax_roc.set_title("Courbe ROC", fontweight="bold", fontsize=24, pad=12)
    ax_roc.minorticks_on()
    ax_roc.grid(True, which="major", linestyle="-", linewidth=0.8, alpha=1)
    ax_roc.grid(True, which="minor", linestyle=":", linewidth=0.6, alpha=1)
    ax_roc.legend(loc="lower right", fontsize=16, frameon=True)

    ax_pr.axhline(prevalence, color="#777777", linestyle="--", linewidth=1.2,
                  label=f"Hasard (prevalence={prevalence:.3f})")
    ax_pr.set_xlim(0, 1)
    ax_pr.set_ylim(0, 1.02)
    ax_pr.set_xlabel("Rappel (Recall)")
    ax_pr.set_ylabel("Precision")
    ax_pr.set_title("Courbe Precision-Recall", fontweight="bold", fontsize=24, pad=12)
    ax_pr.minorticks_on()
    ax_pr.grid(True, which="major", linestyle="-", linewidth=0.8, alpha=1)
    ax_pr.grid(True, which="minor", linestyle=":", linewidth=0.6, alpha=1)
    ax_pr.legend(loc="lower left", fontsize=16, frameon=True)

    title = f"{config_name}  (split={split}, n={len(y_true)}, pos={n_pos}, neg={n_neg})"

    fig_roc.tight_layout(rect=(0, 0, 1, 0.93))
    fig_pr.tight_layout(rect=(0, 0, 1, 0.93))

    out_dir = output_dir / model_name
    out_dir.mkdir(parents=True, exist_ok=True)
    roc_path = out_dir / f"{config_name}_roc.png"
    pr_path = out_dir / f"{config_name}_pr.png"
    fig_roc.savefig(roc_path, dpi=200, bbox_inches="tight")
    fig_pr.savefig(pr_path, dpi=200, bbox_inches="tight")
    plt.close(fig_roc)
    plt.close(fig_pr)
    print(f"  [OK] {roc_path}")
    print(f"  [OK] {pr_path}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--values-dir", type=Path, default=DEFAULT_VALUES_DIR,
                         help="Dossier contenant les sous-dossiers de configs (defaut: values/)")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR,
                         help="Dossier de sortie des figures (defaut: plots/)")
    parser.add_argument("--split", choices=["train", "test", "all"], default="test",
                         help="Split a evaluer (defaut: test)")
    args = parser.parse_args()

    for model_name, csv_path in iter_config_csvs(args.values_dir):
        print(f"[{model_name}] {csv_path.name}")
        plot_config(model_name, csv_path, args.output_dir, args.split)


if __name__ == "__main__":
    main()
