import sys
import os
import json
import argparse
import numpy as np
import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
from GenerateBaseSamplerOutputsAndExtractInfo.GetInfoFromBaseSamplerOutput import getEachStepMask, getLogProbs

from sklearn.metrics import roc_auc_score, average_precision_score, accuracy_score
from PipelineTest.eval_configs import CONFIGS
from PipelineTest.Benchmark.data_split import (
    load_eval_data,
    get_train_test_split,
    get_train_test_split_for_config,
)


# =========================================================================
# DIFFUSION NLL
# =========================================================================

def compute_nll_diffusion(outputs, eval_indices):
    logprobs = getLogProbs(outputs)
    masks = getEachStepMask(outputs)

    nlls = []
    for i in range(len(logprobs)):
        probs = np.asarray(logprobs[i], dtype=float)
        mask = np.asarray(masks[i], dtype=bool)
        just_unmasked = np.zeros_like(mask, dtype=bool)
        just_unmasked[1:, :] = mask[:-1, :] & (~mask[1:, :])

        just_unmasked[-1, :] = just_unmasked[-1, :] | mask[-1, :]
        nll = -(1/ np.sum(just_unmasked)) * np.sum(np.log(probs) * just_unmasked)

        nlls.append(np.exp(nll))  
    return np.array(nlls)[eval_indices]


# =========================================================================
# EVALUATION METRICS
# =========================================================================

def evaluate(nlls, labels, name="Perplexity"):
    valid = ~np.isnan(nlls)
    nlls = nlls[valid]
    labels = labels[valid]

    roc_auc = roc_auc_score(labels, nlls)
    pr_auc = average_precision_score(labels, nlls)

    thresholds = np.linspace(nlls.min(), nlls.max(), 201)
    accs = [accuracy_score(labels, (nlls >= t).astype(int)) for t in thresholds]
    best_thresh = thresholds[np.argmax(accs)]
    best_acc = max(accs)

    y_pred = (nlls >= np.median(nlls)).astype(int)
    acc = accuracy_score(labels, y_pred)

    return {
        "name": name,
        "test_roc_auc": float(roc_auc),
        "test_pr_auc": float(pr_auc),
        "test_accuracy": float(acc),
        "test_best_accuracy": float(best_acc),
        "test_best_threshold": float(best_thresh),
        "test_pos_rate": float(np.mean(labels)),
        "n_samples": int(len(labels)),
        "n_halluc": int(np.sum(labels == 1)),
        "n_correct": int(np.sum(labels == 0)),
    }


# =========================================================================
# RUN ONE CONFIG
# =========================================================================
def run_config(config_name, balance_seed=42):
    cfg = CONFIGS[config_name]
    outputs_path = cfg["outputs_path"]
    eval_json = cfg["eval_json"]

    print(f"\n{'='*70}")
    print(f"  PERPLEXITY: {cfg['name']} ({config_name})")
    print(f"{'='*70}")

    if not os.path.exists(outputs_path):
        print(f"  [SKIP] outputs not found: {outputs_path}")
        return None
    if not os.path.exists(eval_json):
        print(f"  [SKIP] eval_json not found: {eval_json}")
        return None

    labels_raw, indices_raw, n_missing_label, idx_to_prompt = load_eval_data(eval_json)

    if len(labels_raw) == 0:
        print("  [SKIP] no usable samples.")
        return None

    outputs = torch.load(outputs_path, map_location="cpu", weights_only=False)

    labels = labels_raw
    original_question_ids = indices_raw
    train_idx, test_idx = get_train_test_split_for_config(
        config_name, original_question_ids
    )

    balance_stats = {
        "n_pos_before": int(np.sum(labels == 1)),
        "n_neg_before": int(np.sum(labels == 0)),
        "n_kept_per_class": "N/A (Sequential Split)",
    }

    # =========================================================================
    # ALIGNEMENT DES TENSOURS & CALCUL PERPEXITY
    # =========================================================================
    # Remappage des IDs de questions vers les vraies positions dans le fichier d'outputs .pt
    sample_indices = outputs.sample_indices.numpy()
    idx_to_pos = {int(idx): pos for pos, idx in enumerate(sample_indices)}
    tensor_positions = [idx_to_pos[int(idx)] for idx in original_question_ids]

    print(
        f"  Pool size for evaluation: {len(labels)} samples "
        f"(correct={np.sum(labels==0)}, halluc={np.sum(labels==1)})"
    )

    # Calcul des NLL (Perplexité) alignées
    nlls = compute_nll_diffusion(outputs, tensor_positions)

    nlls_test = nlls[test_idx]
    labels_test = labels[test_idx]

    # --- AFFICHAGE DES IDENTIFIANTS POUR VÉRIFICATION ---
    question_ids_train = original_question_ids[train_idx].tolist()
    question_ids_test = original_question_ids[test_idx].tolist()

    print("\n" + "="*50)
    print(f"  INDEX TRIVIAQA SÉLECTIONNÉS POUR LE TEST SPLIT (N={len(question_ids_test)})")
    print(f"  Ratio Hallucination dans le Test Split : {np.mean(labels_test):.1%}")
    print("="*50)
    print(question_ids_test[:20], "... (truncated)")

    valid = ~np.isnan(nlls_test)
    print(f"\n  Test samples: {len(test_idx)} (valid={valid.sum()})")
    print(f"  Mean NLL: {nlls_test[valid].mean():.4f} | halluc: {nlls_test[valid & (labels_test==1)].mean():.4f} | correct: {nlls_test[valid & (labels_test==0)].mean():.4f}")

    # Évaluation des performances métriques
    results = evaluate(nlls_test, labels_test, name=f"Perplexity_{config_name}")
    results["config"] = config_name
    results["outputs_path"] = outputs_path
    results["eval_json"] = eval_json
    results["n_test"] = int(len(test_idx))
    results["n_collapse_dropped"] = 0
    results["n_missing_label_dropped"] = int(n_missing_label)
    results["n_balanced_pool"] = int(len(labels))
    results["n_per_class_before_balance"] = {
        "halluc": balance_stats["n_pos_before"],
        "correct": balance_stats["n_neg_before"],
    }

    # Per-sample values (train+test), keyed by original sample index, for
    # downstream per-sample CSV export (Benchmark/main.py).
    split_of_pos = np.array(["train"] * len(labels))
    split_of_pos[test_idx] = "test"
    samples = {}
    for pos, idx in enumerate(original_question_ids):
        nll = nlls[pos]
        samples[int(idx)] = {
            "label_hallucination": int(labels[pos]),
            "split": str(split_of_pos[pos]),
            "perplexity": float(nll) if not np.isnan(nll) else None,
        }
    results["samples"] = samples

    print(f"  ROC-AUC: {results['test_roc_auc']:.4f}  |  PR-AUC: {results['test_pr_auc']:.4f}  |  BestAcc: {results['test_best_accuracy']:.4f}")

    return results

# =========================================================================
# MAIN
# =========================================================================

def parse_args():
    parser = argparse.ArgumentParser(description="Perplexity-based hallucination detection (diffusion NLL)")
    parser.add_argument("--config", type=str, default="all",
                        choices=list(CONFIGS.keys()) + ["all"],
                        help="Config to evaluate, or 'all' to run all configs")
    parser.add_argument("--output_dir", type=str, default=None,
                        help="Directory to save results (default: PipelineTest/res/eval/)")
    parser.add_argument("--balance_seed", type=int, default=42,
                        help="Random seed used when subsampling the majority class")
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
        result = run_config(config_name, balance_seed=args.balance_seed)
        if result is not None:
            all_results[config_name] = result

    print(f"\n\n{'='*70}")
    print(f"  SUMMARY TABLE (Perplexity)")
    print(f"{'='*70}")
    print(f"  {'Config':<45} {'ROC-AUC':>8} {'PR-AUC':>8} {'BestAcc':>8} {'nPool':>8}")
    print(f"  {'-'*79}")
    for name, r in all_results.items():
        print(f"  {name:<45} {r['test_roc_auc']:>8.4f} {r['test_pr_auc']:>8.4f} {r['test_best_accuracy']:>8.4f} {r['n_balanced_pool']:>8d}")

    suffix = CONFIGS[configs_to_run[0]]["name"] if len(configs_to_run) == 1 else "all"
    out_path = os.path.join(output_dir, f"perplexity_results_{suffix}.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2)
    print(f"\n  Results saved to: {out_path}")

    return all_results


if __name__ == "__main__":
    main()