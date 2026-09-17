"""
Centralized data loading and train/test splitting for all benchmark scripts.

Ensures every script uses the exact same samples and the exact same train/test partition.
"""

import json
import numpy as np


def load_eval_data(eval_json_path):
  
    with open(eval_json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    labels = []
    indices = []
    idx_to_prompt = {}
    n_missing_label = 0
    seen_indices = set()

    for item in data:
        idx = item.get("index")

        if idx in seen_indices:
            continue
        seen_indices.add(idx)

        if "is_hallucination" not in item:
            n_missing_label += 1
            continue

        raw_label = str(item["is_hallucination"]).lower()
        if raw_label not in ("yes", "no"):
            n_missing_label += 1
            continue

        labels.append(1 if raw_label == "yes" else 0)
        indices.append(idx)
        idx_to_prompt[int(idx)] = item.get("question", "")

    return (
        np.array(labels),
        np.array(indices),
        n_missing_label,
        idx_to_prompt,
    )


def get_train_test_split(n_samples, train_ratio=0.8):
   
    all_idx = np.arange(n_samples)
    split_point = int(n_samples * train_ratio)
    return all_idx[:split_point], all_idx[split_point:]


def get_train_test_split_for_config(config_name, indices, train_ratio=0.8,
                                    sync_csv=True, values_dir=None, verbose=True):
   
    train_idx, test_idx = get_train_test_split(len(indices), train_ratio)

    if not sync_csv:
        return train_idx, test_idx

    from PipelineTest.Benchmark.values_csv import DEFAULT_VALUES_DIR, sync_split_column

    indices = np.asarray(indices)
    report = sync_split_column(
        config_name,
        train_indices=[int(i) for i in indices[train_idx]],
        test_indices=[int(i) for i in indices[test_idx]],
        values_dir=DEFAULT_VALUES_DIR if values_dir is None else values_dir,
    )

    if verbose:
        if not report["csv_exists"]:
            print(f"  [split] pool={report['n_pool']} "
                  f"(train={len(train_idx)}, test={len(test_idx)}) | "
                  f"pas de CSV a resynchroniser")
        else:
            print(f"  [split] pool={report['n_pool']} "
                  f"(train={len(train_idx)}, test={len(test_idx)}) | "
                  f"CSV: {report['n_fixed']} split(s) corrige(s) "
                  f"({report['n_flipped']} bascule(s) train<->test, "
                  f"{report['n_out_of_pool']} ligne(s) hors pool), "
                  f"{report['n_missing_from_csv']} sample(s) du pool absent(s) du CSV")
            if report["stale_trained_columns"]:
                print(f"  [split] ATTENTION fuite train/test : les colonnes "
                      f"{report['stale_trained_columns']} portent, sur les lignes qui "
                      f"viennent de basculer, des scores issus d'un fit sur l'ancienne "
                      f"partition. Rejouer les baselines concernees pour les rafraichir.")

    return train_idx, test_idx
