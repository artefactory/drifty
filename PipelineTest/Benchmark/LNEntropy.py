import sys
import os
import json
import argparse
import numpy as np
import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
from PipelineTest.features.baseline import mean_entropy_just_unmasked

from PipelineTest.eval_configs import CONFIGS
from PipelineTest.Benchmark.perplexity import evaluate
from PipelineTest.Benchmark.data_split import (
    load_eval_data,
    get_train_test_split,
    get_train_test_split_for_config,
)


# =========================================================================
# LENGTH-NORMALIZED ENTROPY COMPUTATION
# =========================================================================
def compute_ln_entropy_diffusion(outputs, eval_indices):
    """
    Compute per-sample Length-Normalized Entropy from the unmasked tokens history.
    Calculates a flat global average of Shannon entropy over all validated tokens 
    to properly evaluate trajectory uncertainty.
    """
    ln_entropies = mean_entropy_just_unmasked(outputs)
    return np.array(ln_entropies)[eval_indices]


# =========================================================================
# RUN ONE CONFIG
# =========================================================================

def run_config(config_name, balance_seed=42, test_size=None):
    cfg = CONFIGS[config_name]
    outputs_path = cfg["outputs_path"]
    eval_json = cfg["eval_json"]

    print(f"\n{'='*70}")
    print(f"  LN-ENTROPY: {cfg['name']} ({config_name})")
    print(f"  [split: 80/20 sequential]")
    print(f"{'='*70}")

    if not os.path.exists(outputs_path):
        print(f"  [SKIP] outputs not found: {outputs_path}")
        return None
    if not os.path.exists(eval_json):
        print(f"  [SKIP] eval_json not found: {eval_json}")
        return None

    outputs = torch.load(outputs_path, map_location="cpu", weights_only=False)

    labels_raw, indices_raw, n_missing_label, idx_to_prompt = load_eval_data(eval_json)
    print(
            f"  Raw samples: {len(labels_raw)} "
            f"(correct={np.sum(labels_raw==0)}, halluc={np.sum(labels_raw==1)}) | "
            f"Dropped: {n_missing_label} unusable"
        )
    if len(labels_raw) == 0:
        print("  [SKIP] no usable samples.")
        return None

    labels = labels_raw
    original_question_ids = indices_raw
    balance_stats = {
        "n_pos_before": int(np.sum(labels == 1)),
        "n_neg_before": int(np.sum(labels == 0)),
        "n_kept_per_class": "N/A (Sequential Split)",
    }

    train_idx, test_idx = get_train_test_split_for_config(
        config_name, original_question_ids
    )

    sample_indices = outputs.sample_indices.numpy()
    idx_to_pos = {int(idx): pos for pos, idx in enumerate(sample_indices)}
    tensor_positions = [idx_to_pos[int(idx)] for idx in original_question_ids]

    # Calcul de la métrique LN-Entropy sur les positions du tenseur
    ln_entropies = compute_ln_entropy_diffusion(outputs, tensor_positions)

    X_test = ln_entropies[test_idx]
    y_test = labels[test_idx]

    # --- EXTRACTION ET AFFICHAGE DES IDENTIFIANTS ORIGINAUX ---
    question_ids_train = original_question_ids[train_idx].tolist()
    question_ids_test = original_question_ids[test_idx].tolist()

    print("\n" + "="*50)
    print(f"  INDEX TRIVIAQA SÉLECTIONNÉS POUR LE TEST SPLIT (N={len(question_ids_test)})")
    print("="*50)
    print(question_ids_test)

    print("\n" + "="*50)
    print(f"  INDEX TRIVIAQA SÉLECTIONNÉS POUR LE TRAIN SPLIT (N={len(question_ids_train)})")
    print("="*50)
    print(question_ids_train)
    print("="*50 + "\n")

    print('correlation between ln_entropy and label:', np.corrcoef(X_test, y_test)[0, 1])
    valid = ~np.isnan(X_test)
    print(f"  Test samples: {len(X_test)} (valid={valid.sum()})")
    print(
        f"  Mean LN-Entropy: {X_test[valid].mean():.4f} | "
        f"halluc: {X_test[valid & (y_test==1)].mean():.4f} | "
        f"correct: {X_test[valid & (y_test==0)].mean():.4f}"
    )

    # Étape 4 : Évaluation des performances sur l'ensemble de test
    results = evaluate(X_test, y_test, name=f"LN-Entropy_{config_name}")
    results["config"] = config_name
    results["outputs_path"] = outputs_path
    results["eval_json"] = eval_json
    results["n_test"] = int(len(test_idx))
    results["n_collapse_dropped"] = 0
    results["n_balanced_pool"] = int(len(labels))

    # Per-sample values (train+test), keyed by original sample index, for
    # downstream per-sample CSV export (Benchmark/main.py).
    split_of_pos = np.array(["train"] * len(labels))
    split_of_pos[test_idx] = "test"
    samples = {}
    for pos, idx in enumerate(original_question_ids):
        val = ln_entropies[pos]
        samples[int(idx)] = {
            "label_hallucination": int(labels[pos]),
            "split": str(split_of_pos[pos]),
            "ln_entropy": float(val) if not np.isnan(val) else None,
        }
    results["samples"] = samples

    print(f"  ROC-AUC: {results['test_roc_auc']:.4f}  |  PR-AUC: {results['test_pr_auc']:.4f}  |  BestAcc: {results['test_best_accuracy']:.4f}")

    return results


# =========================================================================
# MAIN ARGUMENT PARSING & SYSTEM ENTRYPOINT
# =========================================================================

def parse_args():
    parser = argparse.ArgumentParser(description="Length-Normalized Entropy hallucination detection baseline")
    parser.add_argument("--config", type=str, default="all",
                        choices=list(CONFIGS.keys()) + ["all"],
                        help="Config to evaluate, or 'all'")
    parser.add_argument("--output_dir", type=str, default=None,
                        help="Directory to save results")
    parser.add_argument("--balance_seed", type=int, default=42,
                        help="Seed for 50/50 class balancing")
    parser.add_argument("--test_size", type=int, default=None,
                        help="Fixed number of test samples (e.g. 300 for TDGNet). "
                             "If None, uses 15%% split.")
    return parser.parse_args()


def main():
    args = parse_args()

    output_dir = args.output_dir or os.path.abspath(os.path.join(os.path.dirname(__file__), "eval"))
    os.makedirs(output_dir, exist_ok=True)

    if args.config == "all":
        configs_to_run = list(CONFIGS.keys())
    else:
        configs_to_run = [args.config]

    all_results = {}
    for config_name in configs_to_run:
        result = run_config(config_name, balance_seed=args.balance_seed, test_size=args.test_size)
        if result is not None:
            all_results[config_name] = result

    # Summary table output
    print(f"\n\n{'='*70}")
    print(f"  SUMMARY TABLE (LN-ENTROPY)")
    print(f"{'='*70}")
    print(f"  {'Config':<45} {'ROC-AUC':>8} {'PR-AUC':>8} {'BestAcc':>8} {'nPool':>8}")
    print(f"  {'-'*79}")
    for name, r in all_results.items():
        print(f"  {name:<45} {r['test_roc_auc']:>8.4f} {r['test_pr_auc']:>8.4f} {r['test_best_accuracy']:>8.4f} {r['n_balanced_pool']:>8d}")

    # File output saving
    suffix = CONFIGS[configs_to_run[0]]["name"] if len(configs_to_run) == 1 else "all"
    out_path = os.path.join(output_dir, f"ln_entropy_results_{suffix}.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2)
    print(f"\n  Results saved to: {out_path}")

    return all_results


if __name__ == "__main__":
    main()