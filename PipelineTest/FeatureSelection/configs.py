"""Shared experiment configurations and feature extraction helper."""

import os
import sys
import numpy as np

_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from PipelineTest.features.utils import load_outputs, get_labels_for_outputs, get_pad_token_id
from PipelineTest.features.baseline import get_baseline_features
from PipelineTest.features.markovian import get_markovian_features


# =========================================================================
# CONFIG
# =========================================================================
CONFIGS = {
    # =========================================================================
    # CONFIGURATIONS : 16 STEPS / 32 TOKENS (Existantes)
    # =========================================================================
    "dream16_32tokens_2500samples_triviaqa_evalqwen_seed42": {
        "outputs_path": "PipelineTest/res/results_dream_16steps_32tokens_triviaqa_2500samples_seed42/outputs_dream_16steps_32tokens_triviaqa_2500samples_seed42.pt",
        "eval_json": "PipelineTest/res/eval/results_dream_16steps_32tokens_triviaqa_2500samples_seed42.json",
        "tokenizer_path": "PipelineTest/res/results_dream_16steps_32tokens_triviaqa_2500samples_seed42/tokenizer_dream_16steps_32tokens_triviaqa_2500samples_seed42.pt",
        "name": "dream_16steps_32tokens_triviaqa_2500samples_evalqwen_seed42",
    },
    "llada16_32tokens_2500samples_naturalquestion_evalqwen_seed42": {
        "outputs_path": "PipelineTest/res/results_llada_16steps_32tokens_naturalquestion_2500samples_seed42/outputs_llada_16steps_32tokens_naturalquestion_2500samples_seed42.pt",
        "eval_json": "PipelineTest/res/eval/results_llada_16steps_32tokens_naturalquestion_2500samples_seed42.json",
        "tokenizer_path": "PipelineTest/res/results_llada_16steps_32tokens_naturalquestion_2500samples_seed42/tokenizer_llada_16steps_32tokens_naturalquestion_2500samples_seed42.pt",
        "name": "llada_16steps_32tokens_naturalquestion_2500samples_evalqwen_seed42",
    },
    "llada16_32tokens_2500samples_triviaqa_evalqwen_seed42": {
        "outputs_path": "PipelineTest/res/results_llada_16steps_32tokens_triviaqa_2500samples_seed42/outputs_llada_16steps_32tokens_triviaqa_2500samples_seed42.pt",
        "eval_json": "PipelineTest/res/eval/results_llada_16steps_32tokens_triviaqa_2500samples_seed42.json",
        "tokenizer_path": "PipelineTest/res/results_llada_16steps_32tokens_triviaqa_2500samples_seed42/tokenizer_llada_16steps_32tokens_triviaqa_2500samples_seed42.pt",
        "name": "llada_16steps_32tokens_triviaqa_2500samples_evalqwen_seed42",
    },
    "dream16_32tokens_2500samples_naturalquestion_evalqwen_seed42": {
        "outputs_path": "PipelineTest/res/results_dream_16steps_32tokens_naturalquestion_2500samples_seed42/outputs_dream_16steps_32tokens_naturalquestion_2500samples_seed42.pt",
        "eval_json": "PipelineTest/res/eval/results_dream_16steps_32tokens_naturalquestion_2500samples_seed42.json",
        "tokenizer_path": "PipelineTest/res/results_dream_16steps_32tokens_naturalquestion_2500samples_seed42/tokenizer_dream_16steps_32tokens_naturalquestion_2500samples_seed42.pt",
        "name": "dream_16steps_32tokens_naturalquestion_2500samples_evalqwen_seed42",
    },
    "llada16_32tokens_2500samples_hotpotqa_evalqwen_seed42": {
        "outputs_path": "PipelineTest/res/results_llada_16steps_32tokens_hotpotqa_2500samples_seed42/outputs_llada_16steps_32tokens_hotpotqa_2500samples_seed42.pt",
        "eval_json": "PipelineTest/res/eval/results_llada_16steps_32tokens_hotpotqa_2500samples_seed42.json",
        "tokenizer_path": "PipelineTest/res/results_llada_16steps_32tokens_hotpotqa_2500samples_seed42/tokenizer_llada_16steps_32tokens_hotpotqa_2500samples_seed42.pt",
        "name": "llada_16steps_32tokens_hotpotqa_2500samples_evalqwen_seed42",
    },
    "dream16_32tokens_2500samples_hotpotqa_evalqwen_seed42": {
        "outputs_path": "PipelineTest/res/results_dream_16steps_32tokens_hotpotqa_2500samples_seed42/outputs_dream_16steps_32tokens_hotpotqa_2500samples_seed42.pt",
        "eval_json": "PipelineTest/res/eval/results_dream_16steps_32tokens_hotpotqa_2500samples_seed42.json",
        "tokenizer_path": "PipelineTest/res/results_dream_16steps_32tokens_hotpotqa_2500samples_seed42/tokenizer_dream_16steps_32tokens_hotpotqa_2500samples_seed42.pt",
        "name": "dream_16steps_32tokens_hotpotqa_2500samples_evalqwen_seed42",
    },

    # =========================================================================
    # CONFIGURATIONS : 32 STEPS / 64 TOKENS (Nouvelles)
    # =========================================================================
    "dream32_64tokens_2500samples_triviaqa_evalqwen_seed42": {
        "outputs_path": "PipelineTest/res/results_dream_32steps_64tokens_triviaqa_2500samples_seed42/outputs_dream_32steps_64tokens_triviaqa_2500samples_seed42.pt",
        "eval_json": "PipelineTest/res/eval/results_dream_32steps_64tokens_triviaqa_2500samples_seed42.json",
        "tokenizer_path": "PipelineTest/res/results_dream_32steps_64tokens_triviaqa_2500samples_seed42/tokenizer_dream_32steps_64tokens_triviaqa_2500samples_seed42.pt",
        "name": "dream_32steps_64tokens_triviaqa_2500samples_evalqwen_seed42",
    },
    "llada32_64tokens_2500samples_naturalquestion_evalqwen_seed42": {
        "outputs_path": "PipelineTest/res/results_llada_32steps_64tokens_naturalquestion_2500samples_seed42/outputs_llada_32steps_64tokens_naturalquestion_2500samples_seed42.pt",
        "eval_json": "PipelineTest/res/eval/results_llada_32steps_64tokens_naturalquestion_2500samples_seed42.json",
        "tokenizer_path": "PipelineTest/res/results_llada_32steps_64tokens_naturalquestion_2500samples_seed42/tokenizer_llada_32steps_64tokens_naturalquestion_2500samples_seed42.pt",
        "name": "llada_32steps_64tokens_naturalquestion_2500samples_evalqwen_seed42",
    },
    "llada32_64tokens_2500samples_triviaqa_evalqwen_seed42": {
        "outputs_path": "PipelineTest/res/results_llada_32steps_64tokens_triviaqa_2500samples_seed42/outputs_llada_32steps_64tokens_triviaqa_2500samples_seed42.pt",
        "eval_json": "PipelineTest/res/eval/results_llada_32steps_64tokens_triviaqa_2500samples_seed42.json",
        "tokenizer_path": "PipelineTest/res/results_llada_32steps_64tokens_triviaqa_2500samples_seed42/tokenizer_llada_32steps_64tokens_triviaqa_2500samples_seed42.pt",
        "name": "llada_32steps_64tokens_triviaqa_2500samples_evalqwen_seed42",
    },
    "dream32_64tokens_2500samples_naturalquestion_evalqwen_seed42": {
        "outputs_path": "PipelineTest/res/results_dream_32steps_64tokens_naturalquestion_2500samples_seed42/outputs_dream_32steps_64tokens_naturalquestion_2500samples_seed42.pt",
        "eval_json": "PipelineTest/res/eval/results_dream_32steps_64tokens_naturalquestion_2500samples_seed42.json",
        "tokenizer_path": "PipelineTest/res/results_dream_32steps_64tokens_naturalquestion_2500samples_seed42/tokenizer_dream_32steps_64tokens_naturalquestion_2500samples_seed42.pt",
        "name": "dream_32steps_64tokens_naturalquestion_2500samples_evalqwen_seed42",
    },
    "llada32_64tokens_2500samples_hotpotqa_evalqwen_seed42": {
        "outputs_path": "PipelineTest/res/results_llada_32steps_64tokens_hotpotqa_2500samples_seed42/outputs_llada_32steps_64tokens_hotpotqa_2500samples_seed42.pt",
        "eval_json": "PipelineTest/res/eval/results_llada_32steps_64tokens_hotpotqa_2500samples_seed42.json",
        "tokenizer_path": "PipelineTest/res/results_llada_32steps_64tokens_hotpotqa_2500samples_seed42/tokenizer_llada_32steps_64tokens_hotpotqa_2500samples_seed42.pt",
        "name": "llada_32steps_64tokens_hotpotqa_2500samples_evalqwen_seed42",
    },
    "dream32_64tokens_2500samples_hotpotqa_evalqwen_seed42": {
        "outputs_path": "PipelineTest/res/results_dream_32steps_64tokens_hotpotqa_2500samples_seed42/outputs_dream_32steps_64tokens_hotpotqa_2500samples_seed42.pt",
        "eval_json": "PipelineTest/res/eval/results_dream_32steps_64tokens_hotpotqa_2500samples_seed42.json",
        "tokenizer_path": "PipelineTest/res/results_dream_32steps_64tokens_hotpotqa_2500samples_seed42/tokenizer_dream_32steps_64tokens_hotpotqa_2500samples_seed42.pt",
        "name": "dream_32steps_64tokens_hotpotqa_2500samples_evalqwen_seed42",
    },

    # =========================================================================
    # CONFIGURATIONS : 32 STEPS / 32 TOKENS 
    # =========================================================================
    "dream32_32tokens_2500samples_triviaqa_evalqwen_seed42": {
        "outputs_path": "PipelineTest/res/results_dream_32steps_32tokens_triviaqa_2500samples_seed42/outputs_dream_32steps_32tokens_triviaqa_2500samples_seed42.pt",
        "eval_json": "PipelineTest/res/eval/results_dream_32steps_32tokens_triviaqa_2500samples_seed42.json",
        "tokenizer_path": "PipelineTest/res/results_dream_32steps_32tokens_triviaqa_2500samples_seed42/tokenizer_dream_32steps_32tokens_triviaqa_2500samples_seed42.pt",
        "name": "dream_32steps_32tokens_triviaqa_2500samples_evalqwen_seed42",
    },
    "llada32_32tokens_2500samples_naturalquestion_evalqwen_seed42": {
        "outputs_path": "PipelineTest/res/results_llada_32steps_32tokens_naturalquestion_2500samples_seed42/outputs_llada_32steps_32tokens_naturalquestion_2500samples_seed42.pt",
        "eval_json": "PipelineTest/res/eval/results_llada_32steps_32tokens_naturalquestion_2500samples_seed42.json",
        "tokenizer_path": "PipelineTest/res/results_llada_32steps_32tokens_naturalquestion_2500samples_seed42/tokenizer_llada_32steps_32tokens_naturalquestion_2500samples_seed42.pt",
        "name": "llada_32steps_32tokens_naturalquestion_2500samples_evalqwen_seed42",
    },
    "llada32_32tokens_2500samples_triviaqa_evalqwen_seed42": {
        "outputs_path": "PipelineTest/res/results_llada_32steps_32tokens_triviaqa_2500samples_seed42/outputs_llada_32steps_32tokens_triviaqa_2500samples_seed42.pt",
        "eval_json": "PipelineTest/res/eval/results_llada_32steps_32tokens_triviaqa_2500samples_seed42.json",
        "tokenizer_path": "PipelineTest/res/results_llada_32steps_32tokens_triviaqa_2500samples_seed42/tokenizer_llada_32steps_32tokens_triviaqa_2500samples_seed42.pt",
        "name": "llada_32steps_32tokens_triviaqa_2500samples_evalqwen_seed42",
    },
    "dream32_32tokens_2500samples_naturalquestion_evalqwen_seed42": {
        "outputs_path": "PipelineTest/res/results_dream_32steps_32tokens_naturalquestion_2500samples_seed42/outputs_dream_32steps_32tokens_naturalquestion_2500samples_seed42.pt",
        "eval_json": "PipelineTest/res/eval/results_dream_32steps_32tokens_naturalquestion_2500samples_seed42.json",
        "tokenizer_path": "PipelineTest/res/results_dream_32steps_32tokens_naturalquestion_2500samples_seed42/tokenizer_dream_32steps_32tokens_naturalquestion_2500samples_seed42.pt",
        "name": "dream_32steps_32tokens_naturalquestion_2500samples_evalqwen_seed42",
    },
    "llada32_32tokens_2500samples_hotpotqa_evalqwen_seed42": {
        "outputs_path": "PipelineTest/res/results_llada_32steps_32tokens_hotpotqa_2500samples_seed42/outputs_llada_32steps_32tokens_hotpotqa_2500samples_seed42.pt",
        "eval_json": "PipelineTest/res/eval/results_llada_32steps_32tokens_hotpotqa_2500samples_seed42.json",
        "tokenizer_path": "PipelineTest/res/results_llada_32steps_32tokens_hotpotqa_2500samples_seed42/tokenizer_llada_32steps_32tokens_hotpotqa_2500samples_seed42.pt",
        "name": "llada_32steps_32tokens_hotpotqa_2500samples_evalqwen_seed42",
    },
    "dream32_32tokens_2500samples_hotpotqa_evalqwen_seed42": {
        "outputs_path": "PipelineTest/res/results_dream_32steps_32tokens_hotpotqa_2500samples_seed42/outputs_dream_32steps_32tokens_hotpotqa_2500samples_seed42.pt",
        "eval_json": "PipelineTest/res/eval/results_dream_32steps_32tokens_hotpotqa_2500samples_seed42.json",
        "tokenizer_path": "PipelineTest/res/results_dream_32steps_32tokens_hotpotqa_2500samples_seed42/tokenizer_dream_32steps_32tokens_hotpotqa_2500samples_seed42.pt",
        "name": "dream_32steps_32tokens_hotpotqa_2500samples_evalqwen_seed42",
    },

    # =========================================================================
    # CONFIGURATIONS : 64 STEPS / 64 TOKENS 
    # =========================================================================
    "dream64_64tokens_2500samples_triviaqa_evalqwen_seed42": {
        "outputs_path": "PipelineTest/res/results_dream_64steps_64tokens_triviaqa_2500samples_seed42/outputs_dream_64steps_64tokens_triviaqa_2500samples_seed42.pt",
        "eval_json": "PipelineTest/res/eval/results_dream_64steps_64tokens_triviaqa_2500samples_seed42.json",
        "tokenizer_path": "PipelineTest/res/results_dream_64steps_64tokens_triviaqa_2500samples_seed42/tokenizer_dream_64steps_64tokens_triviaqa_2500samples_seed42.pt",
        "name": "dream_64steps_64tokens_triviaqa_2500samples_evalqwen_seed42",
    },
    "llada64_64tokens_2500samples_naturalquestion_evalqwen_seed42": {
        "outputs_path": "PipelineTest/res/results_llada_64steps_64tokens_naturalquestion_2500samples_seed42/outputs_llada_64steps_64tokens_naturalquestion_2500samples_seed42.pt",
        "eval_json": "PipelineTest/res/eval/results_llada_64steps_64tokens_naturalquestion_2500samples_seed42.json",
        "tokenizer_path": "PipelineTest/res/results_llada_64steps_64tokens_naturalquestion_2500samples_seed42/tokenizer_llada_64steps_64tokens_naturalquestion_2500samples_seed42.pt",
        "name": "llada_64steps_64tokens_naturalquestion_2500samples_evalqwen_seed42",
    },
    "llada64_64tokens_2500samples_triviaqa_evalqwen_seed42": {
        "outputs_path": "PipelineTest/res/results_llada_64steps_64tokens_triviaqa_2500samples_seed42/outputs_llada_64steps_64tokens_triviaqa_2500samples_seed42.pt",
        "eval_json": "PipelineTest/res/eval/results_llada_64steps_64tokens_triviaqa_2500samples_seed42.json",
        "tokenizer_path": "PipelineTest/res/results_llada_64steps_64tokens_triviaqa_2500samples_seed42/tokenizer_llada_64steps_64tokens_triviaqa_2500samples_seed42.pt",
        "name": "llada_64steps_64tokens_triviaqa_2500samples_evalqwen_seed42",
    },
    "dream64_64tokens_2500samples_naturalquestion_evalqwen_seed42": {
        "outputs_path": "PipelineTest/res/results_dream_64steps_64tokens_naturalquestion_2500samples_seed42/outputs_dream_64steps_64tokens_naturalquestion_2500samples_seed42.pt",
        "eval_json": "PipelineTest/res/eval/results_dream_64steps_64tokens_naturalquestion_2500samples_seed42.json",
        "tokenizer_path": "PipelineTest/res/results_dream_64steps_64tokens_naturalquestion_2500samples_seed42/tokenizer_dream_64steps_64tokens_naturalquestion_2500samples_seed42.pt",
        "name": "dream_64steps_64tokens_naturalquestion_2500samples_evalqwen_seed42",
    },
    "llada64_64tokens_2500samples_hotpotqa_evalqwen_seed42": {
        "outputs_path": "PipelineTest/res/results_llada_64steps_64tokens_hotpotqa_2500samples_seed42/outputs_llada_64steps_64tokens_hotpotqa_2500samples_seed42.pt",
        "eval_json": "PipelineTest/res/eval/results_llada_64steps_64tokens_hotpotqa_2500samples_seed42.json",
        "tokenizer_path": "PipelineTest/res/results_llada_64steps_64tokens_hotpotqa_2500samples_seed42/tokenizer_llada_64steps_64tokens_hotpotqa_2500samples_seed42.pt",
        "name": "llada_64steps_64tokens_hotpotqa_2500samples_evalqwen_seed42",
    },
    "dream64_64tokens_2500samples_hotpotqa_evalqwen_seed42": {
        "outputs_path": "PipelineTest/res/results_dream_64steps_64tokens_hotpotqa_2500samples_seed42/outputs_dream_64steps_64tokens_hotpotqa_2500samples_seed42.pt",
        "eval_json": "PipelineTest/res/eval/results_dream_64steps_64tokens_hotpotqa_2500samples_seed42.json",
        "tokenizer_path": "PipelineTest/res/results_dream_64steps_64tokens_hotpotqa_2500samples_seed42/tokenizer_dream_64steps_64tokens_hotpotqa_2500samples_seed42.pt",
        "name": "dream_64steps_64tokens_hotpotqa_2500samples_evalqwen_seed42",
    },
}
SAVE_ROOT = "PipelineTest/res/FeatureSelection"


def extract_features_and_labels(cfg):
    """Load outputs and return (features, feature_names, labels) for one config."""
    outputs = load_outputs(cfg["outputs_path"])
    labels = get_labels_for_outputs(cfg["eval_json"], outputs)

    try:
        pad_token_id = get_pad_token_id(cfg["outputs_path"])
    except Exception:
        pad_token_id = None

    baseline_feats, baseline_names = get_baseline_features(outputs, pad_token_id=pad_token_id)
    markov_feats, markov_names = get_markovian_features(outputs)

    features = np.concatenate([baseline_feats, markov_feats], axis=1)
    feature_names = baseline_names + markov_names
    return features, feature_names, labels
