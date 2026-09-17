
import json
import numpy as np
from transformers import data


def load_eval_data_for_crossing(train_json_path, eval_json_path):
    """
    Load eval JSON of one dataset for train and load another dataset for eval and return aligned (labels, indices, idx_to_prompt).

    Filtering rules (applied uniformly to ALL scripts):
      - Skip items that have no ``is_hallucination`` field.
      - Skip items whose ``is_hallucination`` is neither "yes" nor "no".
      - Skip duplicate indices (keep first occurrence).

    Collapse samples are **never** dropped – they stay in the pool.
    """
    with open(train_json_path, "r", encoding="utf-8") as f:
        train_data = json.load(f)
        
    with open(eval_json_path, "r", encoding="utf-8") as f:
        eval_data = json.load(f)

    train_labels = []
    train_indices = []
    train_n_missing_label = 0
    train_seen_indices = set()

    for item in train_data:

        idx = item.get("index")

        if idx in train_seen_indices:
            continue
        train_seen_indices.add(idx)

        if "is_hallucination" not in item:
            train_n_missing_label += 1
            continue

        raw_label = str(item["is_hallucination"]).lower()
        if raw_label not in ("yes", "no"):
            train_n_missing_label += 1
            continue

        train_labels.append(1 if raw_label == "yes" else 0)
        train_indices.append(idx)

    eval_labels = []
    eval_indices = []
    eval_n_missing_label = 0
    eval_seen_indices = set()

    for item in eval_data:

        idx = item.get("index")

        if idx in eval_seen_indices:
            continue
        eval_seen_indices.add(idx)

        if "is_hallucination" not in item:
            eval_n_missing_label += 1
            continue

        raw_label = str(item["is_hallucination"]).lower()
        if raw_label not in ("yes", "no"):
            eval_n_missing_label += 1
            continue

        eval_labels.append(1 if raw_label == "yes" else 0)
        eval_indices.append(idx)

    return (
        np.array(train_labels),
        np.array(train_indices),
        train_n_missing_label,
        np.array(eval_labels),
        np.array(eval_indices),
        eval_n_missing_label,
    )
