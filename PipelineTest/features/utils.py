"""
features.utils
==============
Shared utilities: output loading, label extraction, padding detection.
"""

import os
import sys
import json
import numpy as np
import torch

# Ensure the project root (dllm/) is on sys.path so `PipelineTest` is importable.
_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from PipelineTest.features.io import mergeOutputsList



# =========================================================================
# OUTPUT LOADING
# =========================================================================

def load_outputs(output_path):
    """Load and merge sampler outputs from a .pt file."""
    return mergeOutputsList(output_path)


# =========================================================================
# LABELS
# =========================================================================

def get_labels(eval_json_path):
    """Load binary hallucination labels from an evaluation JSON file.
    
    Returns:
        labels: np.ndarray of shape (N,) with 1=hallucination, 0=correct
    """
    with open(eval_json_path, "r") as f:
        data = json.load(f)
    data.sort(key=lambda x: x["index"])
    labels = np.array([1 if d["is_hallucination"] == "yes" else 0 for d in data], dtype=np.int64)
    return labels


def get_labels_for_outputs(eval_json_path, outputs):
    """Get labels aligned with outputs.sample_indices."""
    labels = get_labels(eval_json_path)
    if outputs.sample_indices is not None:
        labels = labels[outputs.sample_indices.numpy()]
    return labels


# =========================================================================
# PADDING DETECTION
# =========================================================================

def compute_padding_mask(outputs, positions, pad_token_id):
    """
    Identify padding tokens by checking proposed_sequence (histories_x0).
    A token d is padding if the model proposes pad_token_id at ALL masked steps.
    
    Args:
        outputs: merged sampler outputs
        positions: list/array of sample positions in outputs
        pad_token_id: tokenizer pad token id
        
    Returns:
        padding_mask: (N, T, D) bool array where True = padding (to exclude)
    """
    N = len(positions)
    T = len(outputs.histories_x0)
    D = outputs.max_new_tokens

    padding = np.zeros((N, D), dtype=bool)

    for idx_n, pos in enumerate(positions):
        start_idx = outputs.start_idx_history[pos]

        for d in range(D):
            masked_steps = []
            for t in range(T):
                if outputs.histories_mask[t][pos, start_idx + d].item() > 0:
                    masked_steps.append(t)

            if len(masked_steps) == 0:
                continue

            all_padding = True
            for t in masked_steps:
                proposed_token = outputs.histories_x0[t][pos, start_idx + d].item()
                if proposed_token != pad_token_id:
                    all_padding = False
                    break

            if all_padding:
                padding[idx_n, d] = True

    padding_mask = np.broadcast_to(padding[:, np.newaxis, :], (N, T, D)).copy()
    return padding_mask


def get_pad_token_id(output_path):
    """Derive pad_token_id from the tokenizer file stored alongside outputs."""
    dir_path = os.path.dirname(output_path)
    basename = os.path.basename(output_path)
    tokenizer_name = basename.replace("outputs_", "tokenizer_")
    tokenizer_path = os.path.join(dir_path, tokenizer_name)
    tokenizer = torch.load(tokenizer_path, map_location="cpu", weights_only=False)
    return tokenizer.pad_token_id


# =========================================================================
# DATA MATCHING (eval JSON <-> outputs)
# =========================================================================

def match_samples(eval_json_path, outputs):
    """Match evaluation JSON entries to output positions.
    
    Returns:
        positions: np.ndarray of positions in outputs
        labels: np.ndarray of binary labels (1=halluc, 0=correct)
        data: list of dicts from JSON
    """
    with open(eval_json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    hmap = {}
    for i in range(len(outputs.sample_indices)):
        j = outputs.sample_indices[i].item() if outputs.sample_indices is not None else i
        hmap[j] = i

    samples = []
    for d in data:
        idx = d["index"]
        if idx in hmap:
            pos = hmap[idx]
            label = 0 if d["is_hallucination"] == "no" else 1
            samples.append((pos, label))

    positions = np.array([s[0] for s in samples])
    labels = np.array([s[1] for s in samples])
    return positions, labels, data
