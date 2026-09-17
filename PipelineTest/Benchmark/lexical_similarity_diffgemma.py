# /// script
# dependencies = [
#     "accelerate>=1.14.0",
#     "datasets>=5.0.1",
#     "diffusers==0.40.0",
#     "pillow>=12.2.0",
#     "protobuf>=7.36.1",
#     "sentencepiece>=0.2.2",
#     "tiktoken>=0.14.0",
#     "torch==2.11.0+cu128",
#     "torchvision==0.26.0+cu128",
#     "transformers==5.16.1",
#     "numpy",
#     "scikit-learn",
# ]
# [[tool.uv.index]]
# name = "pytorch-cu128"
# url = "https://download.pytorch.org/whl/cu128"
# explicit = true
# [tool.uv.sources]
# torch = [
#   { index = "pytorch-cu128", marker = "sys_platform == 'linux' or sys_platform == 'win32'" },
# ]
# torchvision = [
#   { index = "pytorch-cu128", marker = "sys_platform == 'linux' or sys_platform == 'win32'" },
# ]
# ///

"""Lexical-similarity evaluation for DiffusionGemma configurations (single-GPU variant).

Same script as lexical_similarity_diffgemma.py, with the generation backend of
semantic_entropy_diffugemma_V2.py copied verbatim (no import of that module):

  - CUDA_VISIBLE_DEVICES is pinned to the first GPU (accelerate's multi-GPU
    dispatch_model sharding silently corrupts this model's forward on this
    stack; opt out with --multi_gpu at your own risk) and the ~8GB weight
    overflow is handled by accelerate's CPU offloading. Run with `q -G 1`.
  - model.config.canvas_length is overridden with the configuration's token
    budget, so exactly one canvas of <tokens> tokens is denoised per generation.

The PEP 723 header pins the same torch/transformers/diffusers/accelerate stack
as semantic_entropy_diffugemma_V2.py (plus scikit-learn). Do NOT run this script
in the project .venv (torch 2.6 / transformers 5.11 / accelerate 1.11): there,
CPU offloading with device_map="auto" leaves weights on the meta device and the
forward crashes with "Tensor.item() cannot be called on meta tensors".

Everything except the batch size is derived from the configuration name; the
per-question ROUGE-L features and the logistic score land in the JSON in eval/
and in values/gemma/<config>.csv (column `lexical_similarity`).

Usage (from the repository root):
    q -G 1 uv run PipelineTest/Benchmark/lexical_similarity_diffgemma.py \
        --config diffgemma128_256tokens_2500samples_triviaqa_evalqwen_seed42
    q -G 1 uv run PipelineTest/Benchmark/lexical_similarity_diffgemma.py \
        --config diffgemma16_32tokens_2500samples_triviaqa_evalqwen_seed42 \
        --gen_batch_size 8
"""

import argparse
import csv
import json
import os
import re
import sys
import time
from types import SimpleNamespace

if "--multi_gpu" not in sys.argv:
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
sys.path.insert(0, REPO_ROOT)

import numpy as np
import torch
from diffusers import BlockRefinementScheduler, DiffusionGemmaPipeline
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from transformers import AutoProcessor, DiffusionGemmaForBlockDiffusion

from PipelineTest.Benchmark.data_split import get_train_test_split, load_eval_data

MODEL_ID = "google/diffusiongemma-26B-A4B-it"
THOUGHT_HEADER = "thought\n"
TEMPERATURE = 0.5
N_VARIANTS = 5
CONFIG_NAME_RE = re.compile(r"diffgemma(\d+)_(\d+)tokens_2500samples_(.+)_evalqwen_seed42")


# =========================================================================
# CONFIGS
# =========================================================================
def get_diffugemma_configs():
    """diffgemma* entries of CONFIGS in PipelineTest/eval_configs.py."""
    from PipelineTest.eval_configs import CONFIGS

    return {name: config for name, config in CONFIGS.items() if name.startswith("diffgemma")}


def resolve_repo_path(path):
    return path if os.path.isabs(path) else os.path.join(REPO_ROOT, path)


# =========================================================================
# GENERATION (verbatim copy from semantic_entropy_diffugemma_V2.py)
# =========================================================================
def load_generation_backend(canvas_length: int):
    n_gpus = torch.cuda.device_count() if torch.cuda.is_available() else 0
    print(f"Loading generation model: {MODEL_ID} (device_map=auto, {n_gpus} visible GPU(s))...", flush=True)
    model = DiffusionGemmaForBlockDiffusion.from_pretrained(MODEL_ID, dtype=torch.bfloat16, device_map="auto")
    # The pipeline reads model.config.canvas_length at call time, so overriding it post-load is enough.
    model.config.canvas_length = canvas_length
    print(f"model.config.canvas_length = {model.config.canvas_length}", flush=True)
    processor = AutoProcessor.from_pretrained(MODEL_ID)
    pipe = DiffusionGemmaPipeline(model=model, scheduler=BlockRefinementScheduler(), processor=processor)
    pipe.set_progress_bar_config(disable=True)
    return pipe


def trim_sequences_at_eos(sequences: torch.LongTensor, eos_ids) -> list[list[int]]:
    """Trim each row at its first EOS token (inclusive). Diffusion canvases keep
    sampling content tokens past EOS, so this must consider every EOS id."""
    ids = [int(e) for e in (eos_ids if isinstance(eos_ids, (list, tuple)) else [eos_ids])]
    trimmed = []
    for seq in sequences:
        tokens = seq.tolist()
        cut = len(tokens)
        for pos, tok in enumerate(tokens):
            if tok in ids:
                cut = pos + 1
                break
        trimmed.append(tokens[:cut])
    return trimmed


def decode_variants(output, processor, eos_ids) -> list[str]:
    trimmed = trim_sequences_at_eos(output.sequences, eos_ids)
    texts = processor.batch_decode(trimmed, skip_special_tokens=True)
    return [text.removeprefix(THOUGHT_HEADER) for text in texts]


def generate_variants(pipe, prompts: list[str], n_variants: int, args) -> list[list[str]]:
    """N stochastic variants per prompt, expanded question-major into pooled batches
    (same strategy as generate_variants in the reference script: the cost of a
    diffusion sampler is dominated by steps, so amortize them over a larger batch)."""
    expanded_prompts = []
    expanded_keys = []
    for key, prompt in enumerate(prompts):
        for _ in range(n_variants):
            expanded_prompts.append(prompt)
            expanded_keys.append(key)

    n_batches = (len(expanded_prompts) + args.gen_batch_size - 1) // args.gen_batch_size
    results = [[] for _ in prompts]
    eos_ids = pipe.model.config.eos_token_id

    t_start = time.time()
    print(f"generate_variants: {len(expanded_prompts)} items ({len(prompts)} prompts x N={n_variants}) "
          f"in {n_batches} batches of {args.gen_batch_size}", flush=True)

    with torch.inference_mode():
        for batch_idx, start in enumerate(range(0, len(expanded_prompts), args.gen_batch_size)):
            end = min(start + args.gen_batch_size, len(expanded_prompts))
            output = pipe(
                prompt=expanded_prompts[start:end],
                gen_length=args.canvas_length,
                num_inference_steps=args.num_inference_steps,
                temperature=TEMPERATURE,
                confidence_threshold=None if args.no_adaptive_stop else args.confidence_threshold,
                output_type="seq",
            )
            texts = decode_variants(output, pipe.processor, eos_ids)
            for j, text in enumerate(texts):
                results[expanded_keys[start + j]].append(text)

            if batch_idx == 0 or (batch_idx + 1) % args.log_every == 0 or (batch_idx + 1) == n_batches:
                elapsed = time.time() - t_start
                rate = (batch_idx + 1) / elapsed if elapsed > 0 else 0.0
                eta = (n_batches - batch_idx - 1) / rate if rate > 0 else float("inf")
                print(f"  gen batch {batch_idx + 1}/{n_batches} ({elapsed:.1f}s elapsed, "
                      f"{rate:.2f} batch/s, ETA {eta:.0f}s)", flush=True)

    print(f"generate_variants done in {time.time() - t_start:.1f}s", flush=True)
    return results


def save_json(path, payload):
    tmp_path = path + ".tmp"
    with open(tmp_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    os.replace(tmp_path, path)


# =========================================================================
# LEXICAL SIMILARITY
# =========================================================================
FEATURE_COLUMNS = (
    "rouge_l_max_f1",
    "rouge_l_mean_f1",
    "rouge_l_max_prec",
    "rouge_l_max_rec",
    "rouge_l_var",
)


def tokenize(text):
    return [token for token in re.split(r"[^a-z0-9]+", text.lower()) if token]


def lcs_length(first, second):
    previous = [0] * (len(second) + 1)
    for first_token in first:
        current = [0]
        for index, second_token in enumerate(second, 1):
            if first_token == second_token:
                current.append(previous[index - 1] + 1)
            else:
                current.append(max(previous[index], current[-1]))
        previous = current
    return previous[-1]


def rouge_l_f1(candidate, reference):
    candidate_tokens = tokenize(candidate)
    reference_tokens = tokenize(reference)
    lcs = lcs_length(candidate_tokens, reference_tokens)
    precision = lcs / len(candidate_tokens) if candidate_tokens else 0.0
    recall = lcs / len(reference_tokens) if reference_tokens else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return precision, recall, f1


def compute_lexical_features(variants):
    if len(variants) < 2:
        return dict.fromkeys(FEATURE_COLUMNS, 0.0)

    scores = [rouge_l_f1(variants[0], variant) for variant in variants[1:]]
    precisions = [score[0] for score in scores]
    recalls = [score[1] for score in scores]
    f1s = [score[2] for score in scores]
    best_index = max(range(len(f1s)), key=f1s.__getitem__)
    mean_f1 = sum(f1s) / len(f1s)
    variance = sum((score - mean_f1) ** 2 for score in f1s) / len(f1s)
    return {
        "rouge_l_max_f1": max(f1s),
        "rouge_l_mean_f1": mean_f1,
        "rouge_l_max_prec": precisions[best_index],
        "rouge_l_max_rec": recalls[best_index],
        "rouge_l_var": variance,
    }


def fit_logistic_scores(feature_rows, labels):
    features = np.asarray([[row[name] for name in FEATURE_COLUMNS] for row in feature_rows], dtype=float)
    train_idx, test_idx = get_train_test_split(len(labels))
    model = Pipeline([
        ("scaler", StandardScaler()),
        ("logreg", LogisticRegression(max_iter=10000, random_state=42)),
    ])
    grid = GridSearchCV(
        model,
        {
            "logreg__C": [0.001, 0.01, 0.1, 1, 10, 100],
            "logreg__penalty": ["l1", "l2"],
            "logreg__solver": ["saga"],
        },
        cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=42),
        scoring="roc_auc",
        n_jobs=-1,
    )
    grid.fit(features[train_idx], labels[train_idx])
    scores = grid.best_estimator_.predict_proba(features)[:, 1]
    return scores, train_idx, test_idx, grid.best_params_


def target_csv_path(config_name):
    values_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "values")
    return os.path.join(values_dir, "gemma", f"{config_name}.csv")


def update_csv(csv_path, features_by_id):
    if not os.path.isfile(csv_path):
        print(f"[CSV] Target file not found, skipping: {csv_path}", flush=True)
        return

    with open(csv_path, newline="", encoding="utf-8") as csv_file:
        reader = csv.DictReader(csv_file)
        if not reader.fieldnames or "sample_index" not in reader.fieldnames:
            raise ValueError(f"CSV must contain sample_index: {csv_path}")
        rows = list(reader)
        fieldnames = list(reader.fieldnames)

    feature_names = list(next(iter(features_by_id.values())).keys())
    for name in feature_names:
        if name not in fieldnames:
            fieldnames.append(name)

    updated = 0
    for row in rows:
        features = features_by_id.get(str(row["sample_index"]))
        if features is not None:
            row.update(features)
            updated += 1

    if not updated:
        print(f"[CSV] No matching sample_index values in {csv_path}", flush=True)
        return

    temporary_path = csv_path + ".tmp"
    with open(temporary_path, "w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    os.replace(temporary_path, csv_path)
    print(f"[CSV] Updated {updated} rows in {csv_path}", flush=True)


def parse_args():
    configs = get_diffugemma_configs()
    parser = argparse.ArgumentParser(description="Lexical similarity evaluation for DiffusionGemma (single-GPU).")
    parser.add_argument("--config", required=True, choices=sorted(configs))
    parser.add_argument("--gen_batch_size", type=int, default=4,
                        help="Total batch size for generation (prompts x variants pooled). 4 fits a 48GB card "
                             "in single-GPU+CPU-offload mode; raise cautiously and watch for OOM.")
    parser.add_argument("--nli_batch_size", type=int, default=32,
                        help="Accepted for consistency with semantic_entropy_diffugemma; unused here.")
    parser.add_argument("--multi_gpu", action="store_true",
                        help="Do not pin CUDA_VISIBLE_DEVICES to GPU 0 (known to corrupt the forward; at your own risk).")
    return parser.parse_args()


def main():
    args = parse_args()
    config = get_diffugemma_configs()[args.config]
    match = CONFIG_NAME_RE.fullmatch(args.config)
    if match is None:
        raise ValueError(f"Unsupported DiffusionGemma configuration name: {args.config}")

    # The diffgemma configs were sampled with canvas_length == max_new_tokens, so
    # one canvas of exactly <tokens> tokens is denoised per generation.
    canvas_length = int(match.group(2))
    runtime = SimpleNamespace(
        num_inference_steps=int(match.group(1)),
        canvas_length=canvas_length,
        gen_batch_size=args.gen_batch_size,
        confidence_threshold=0.005,
        no_adaptive_stop=False,
        log_every=10,
    )
    labels, indices, _, prompts_by_id = load_eval_data(resolve_repo_path(config["eval_json"]))
    questions = [
        {"question_id": int(index), "question": prompts_by_id[int(index)]}
        for index in indices
    ]
    print(f"Loaded {len(questions)} questions "
          f"(config={args.config}, steps={runtime.num_inference_steps}, "
          f"canvas_length={canvas_length})", flush=True)

    prompts = [question["question"] for question in questions]
    pipe = load_generation_backend(canvas_length)
    variants = generate_variants(pipe, prompts, N_VARIANTS, runtime)

    feature_rows = [
        compute_lexical_features(question_variants)
        for question, question_variants in zip(questions, variants)
    ]
    scores, train_idx, test_idx, best_params = fit_logistic_scores(feature_rows, labels)
    split_by_position = np.full(len(questions), "train", dtype=object)
    split_by_position[test_idx] = "test"
    features_by_id = {
        str(question["question_id"]): {
            **feature_rows[position],
            "lexical_similarity": float(scores[position]),
            "split": split_by_position[position],
            "label_hallucination": int(labels[position]),
        }
        for position, question in enumerate(questions)
    }
    output_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "eval")
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, f"lexical_similarity_{args.config}.json")
    save_json(output_path, {
        "config": args.config,
        "temperature": TEMPERATURE,
        "n_variants": N_VARIANTS,
        "num_inference_steps": runtime.num_inference_steps,
        "canvas_length": canvas_length,
        "gen_length": canvas_length,
        "gen_batch_size": args.gen_batch_size,
        "logistic_train_size": len(train_idx),
        "logistic_test_size": len(test_idx),
        "logistic_best_params": best_params,
        "results": features_by_id,
    })
    update_csv(target_csv_path(args.config), features_by_id)
    print(f"[SUCCESS] Saved to: {output_path}", flush=True)


if __name__ == "__main__":
    main()
