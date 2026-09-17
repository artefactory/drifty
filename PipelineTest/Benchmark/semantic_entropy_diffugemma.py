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

"""Semantic entropy (NLI clustering) evaluation of DiffusionGemma on TriviaQA.

Same script as semantic_entropy_diffugemma.py (single-GPU + CPU-offload variant),
driven by a PipelineTest configuration:

  1. Load the questions of the configuration's eval JSON
     (PipelineTest/eval_configs.py::CONFIGS[--config]["eval_json"]).
  2. For each question, generate N stochastic variants (question-major expansion,
     batched prompts x variants, temperature sampling) with DiffusionGemmaPipeline.
  3. Cluster the N variants per question with bidirectional DeBERTa-MNLI entailment
     (exact-duplicate shortcut, entailment & score > 0.5).
  4. Semantic entropy = Shannon entropy over the cluster size distribution.

Everything except the two batch sizes is derived from the configuration name
(`diffgemma<steps>_<tokens>tokens_2500samples_<dataset>_evalqwen_seed42`):
num_inference_steps = <steps>, canvas_length = gen_length = <tokens>. The
per-question entropies are written to the JSON in eval/ and appended to
values/gemma/<config>.csv under the column `semantic_entropy_diffugemma`.

Each generation is a single `canvas_length`-token canvas (we set both
`model.config.canvas_length` and `gen_length` to this value, so exactly one
canvas is denoised) that we trim at the first EOS token (<eos> or <end_of_turn>)
before decoding, so post-EOS canvas noise never leaks into the text fed to NLI.
The model restates the chat template's empty thinking channel header
("<|channel>thought\n..."); its special tokens are stripped at decode time,
leaving a literal "thought\n" prefix that we remove so only the actual answer
reaches the NLI clustering (same spirit as the template's own strip_thinking
macro).

Correctness labels are intentionally NOT computed here (handled separately by an
LLM-as-a-judge pipeline).

IMPORTANT - single-GPU + CPU offload only: accelerate's multi-GPU dispatch_model
sharding silently corrupts this model's forward on this stack (torch 2.11 /
RTX 6000 Ada): layers whose weights land on the second GPU read unwritten
(zeros/NaN) activations and emit garbage or all-zero outputs, nondeterministically
per process. The script therefore forces CUDA_VISIBLE_DEVICES to the first GPU
(opt out with --multi_gpu at your own risk) and relies on accelerate's CPU
offloading for the ~8GB weight overflow, which was validated to be numerically
correct. Run with `q -G 1`. The fp32 logits cast over the 262k vocab makes
gen_batch_size=16 OOM on a 48GB card; the default is 4 (raise cautiously).

Everything (model, eval JSON, DeBERTa NLI) is served locally / from the local HF
cache. Set HF_HUB_OFFLINE=1 to force pure-cache mode (default env is respected).

Usage (from the repository root):
    q -G 1 uv run PipelineTest/Benchmark/semantic_entropy_diffugemma.py \
        --config diffgemma128_256tokens_2500samples_triviaqa_evalqwen_seed42
    q -G 1 uv run PipelineTest/Benchmark/semantic_entropy_diffugemma.py \
        --config diffgemma16_32tokens_2500samples_triviaqa_evalqwen_seed42 \
        --gen_batch_size 8 --nli_batch_size 64
"""
import argparse
import csv
import json
import os
import re
import sys
import time

if "--multi_gpu" not in sys.argv:
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
sys.path.insert(0, REPO_ROOT)

import numpy as np
import torch
import transformers
from diffusers import BlockRefinementScheduler, DiffusionGemmaPipeline
from transformers import AutoProcessor, DiffusionGemmaForBlockDiffusion

MODEL_ID = "google/diffusiongemma-26B-A4B-it"
NLI_MODEL_ID = "microsoft/deberta-v2-xlarge-mnli"
THOUGHT_HEADER = "thought\n"
TEMPERATURE = 0.5
N_VARIANTS = 5
CONFIG_NAME_RE = re.compile(r"diffgemma(\d+)_(\d+)tokens_2500samples_(.+)_evalqwen_seed42")


def _load_all_configs():
    """CONFIGS from PipelineTest/eval_configs.py."""
    from PipelineTest.eval_configs import CONFIGS

    return CONFIGS


def get_diffugemma_configs():
    return {
        name: config
        for name, config in _load_all_configs().items()
        if name.startswith("diffgemma")
    }


def parse_args():
    configs = get_diffugemma_configs()
    parser = argparse.ArgumentParser(description="Semantic entropy of DiffusionGemma on TriviaQA (NLI clustering).")
    parser.add_argument("--config", required=True, choices=sorted(configs),
                        help="Configuration from PipelineTest/eval_configs.py.")
    parser.add_argument("--gen_batch_size", type=int, default=4,
                        help="Total batch size for generation (prompts x variants pooled). 4 fits a 48GB card "
                             "in single-GPU+CPU-offload mode (the forward casts full-vocab logits to fp32); "
                             "raise cautiously and watch for OOM.")
    parser.add_argument("--nli_batch_size", type=int, default=32,
                        help="Batch size for NLI inference.")
    parser.add_argument("--multi_gpu", action="store_true",
                        help="Do not pin CUDA_VISIBLE_DEVICES to GPU 0 (known to corrupt the forward; at your own risk).")
    return parser.parse_args()


def resolve_repo_path(path):
    return path if os.path.isabs(path) else os.path.join(REPO_ROOT, path)


# =========================================================================
# DATA
# =========================================================================
def load_questions(eval_json_path):
    """Load the eval JSON of a configuration and return its questions in order."""
    with open(eval_json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    questions = []
    for item in data:
        questions.append({
            "question_id": item.get("index"),
            "question": item.get("question"),
        })
    return questions


# =========================================================================
# GENERATION
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


# =========================================================================
# NLI CLUSTERING / SEMANTIC ENTROPY (verbatim port from the reference script)
# =========================================================================
def _batched_entailment_flags(pairs, pipeline_nli, batch_size=64):
    if not pairs:
        return []
    inputs = [{"text": p, "text_pair": h} for p, h in pairs]
    with torch.inference_mode():
        outputs = pipeline_nli(inputs, batch_size=batch_size)
    flags = []
    for out in outputs:
        res = out[0] if isinstance(out, list) else out
        flags.append(res["label"].lower() == "entailment" and res["score"] > 0.5)
    return flags


def _partition_into_nli_clusters_batched(variants, pipeline_nli, batch_size=64):
    clusters = []
    rep_index_of_cluster = []
    seen_text_to_cluster = {}

    for idx, text in enumerate(variants):
        stripped = text.strip()
        if not stripped:
            clusters.append([idx])
            rep_index_of_cluster.append(idx)
            continue

        if stripped in seen_text_to_cluster:
            clusters[seen_text_to_cluster[stripped]].append(idx)
            continue

        if not clusters:
            clusters.append([idx])
            rep_index_of_cluster.append(idx)
            seen_text_to_cluster[stripped] = 0
            continue

        rep_texts = [variants[r] for r in rep_index_of_cluster]
        forward_pairs = [(text, rep) for rep in rep_texts]
        backward_pairs = [(rep, text) for rep in rep_texts]
        all_pairs = forward_pairs + backward_pairs

        flags = _batched_entailment_flags(all_pairs, pipeline_nli, batch_size=batch_size)
        n = len(rep_texts)
        forward_flags = flags[:n]
        backward_flags = flags[n:]

        assigned = False
        for c_i in range(n):
            if forward_flags[c_i] and backward_flags[c_i]:
                clusters[c_i].append(idx)
                assigned = True
                break

        if not assigned:
            clusters.append([idx])
            rep_index_of_cluster.append(idx)
            seen_text_to_cluster[stripped] = len(clusters) - 1

    return clusters


def calculate_semantic_entropy_nli(dict_results, pipeline_nli, nli_batch_size=64):
    detailed_results = {}
    total_questions = len(dict_results)

    for q_idx, (key, variants) in enumerate(dict_results.items(), 1):
        n = len(variants)
        if n == 0:
            detailed_results[key] = {"variants": [], "clusters": [], "semantic_entropy": 0.0}
            continue

        clusters_indices = _partition_into_nli_clusters_batched(variants, pipeline_nli, batch_size=nli_batch_size)
        clusters_text = [[variants[idx] for idx in cluster] for cluster in clusters_indices]

        probabilities = [len(c) / n for c in clusters_indices]
        shannon_entropy = -sum(p * np.log(p + 1e-12) for p in probabilities)

        detailed_results[key] = {
            "variants": variants,
            "clusters": clusters_text,
            "semantic_entropy": float(shannon_entropy),
        }

        if q_idx % 20 == 0 or q_idx == total_questions:
            print(f"  -> NLI progress: {q_idx}/{total_questions} questions processed.", flush=True)

    return detailed_results


# =========================================================================
# OUTPUT
# =========================================================================
def save_json(path, payload):
    tmp_path = path + ".tmp"
    with open(tmp_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    os.replace(tmp_path, path)


def update_benchmark_csv(csv_path, questions, detailed, column_name):
    """Add semantic entropy to the CSV selected by the generation configuration."""
    if not os.path.isfile(csv_path):
        print(f"[CSV] Target file not found, skipping CSV update: {csv_path}", flush=True)
        return

    entropy_by_id = {
        str(question["question_id"]): detailed[pos]["semantic_entropy"]
        for pos, question in enumerate(questions)
        if pos in detailed
    }

    updated_rows = 0
    with open(csv_path, newline="", encoding="utf-8") as csv_file:
        reader = csv.DictReader(csv_file)
        if not reader.fieldnames or "sample_index" not in reader.fieldnames:
            print(f"[CSV] Skipping {csv_path}: missing sample_index", flush=True)
            return
        rows = list(reader)

    fieldnames = list(reader.fieldnames)
    if column_name not in fieldnames:
        fieldnames.append(column_name)

    for row in rows:
        value = entropy_by_id.get(str(row["sample_index"]))
        if value is not None:
            row[column_name] = value
            updated_rows += 1

    if not updated_rows:
        print(f"[CSV] No matching sample_index values in {csv_path}", flush=True)
        return

    tmp_path = csv_path + ".tmp"
    with open(tmp_path, "w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    os.replace(tmp_path, csv_path)

    print(
        f"[CSV] Updated {updated_rows} rows in {csv_path} "
        f"with column '{column_name}'",
        flush=True,
    )


def get_target_csv_path(csv_dir, config_name):
    return os.path.join(csv_dir, "gemma", f"{config_name}.csv")


def build_results(questions, detailed, max_pos=None):
    results = {}
    upper = len(questions) if max_pos is None else max_pos
    for pos in range(upper):
        q = questions[pos]
        info = detailed.get(pos, {"variants": [], "clusters": [], "semantic_entropy": 0.0})
        results[q["question_id"]] = {
            "question": q["question"],
            "variants": info["variants"],
            "clusters": info["clusters"],
            "semantic_entropy": info["semantic_entropy"],
        }
    return results


def main():
    args = parse_args()
    config = get_diffugemma_configs()[args.config]
    match = CONFIG_NAME_RE.fullmatch(args.config)
    if match is None:
        raise ValueError(f"Unsupported DiffusionGemma configuration name: {args.config}")
    args.num_inference_steps = int(match.group(1))
    # The diffgemma configs were sampled with canvas_length == max_new_tokens, so
    # one canvas of exactly <tokens> tokens is denoised per generation.
    args.canvas_length = int(match.group(2))
    args.csv_dataset = match.group(3)
    args.seed = 42
    args.log_every = 10
    args.confidence_threshold = 0.005
    args.no_adaptive_stop = False
    args.csv_column = "semantic_entropy_diffugemma"
    transformers.set_seed(args.seed)

    output_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "eval")
    os.makedirs(output_dir, exist_ok=True)
    final_path = os.path.join(output_dir, f"semantic_entropy_{args.config}.json")
    variants_path = final_path.replace(".json", "_variants.json")

    questions = load_questions(resolve_repo_path(config["eval_json"]))
    args.n_questions = len(questions)
    print(f"Loaded {len(questions)} questions "
          f"(config={args.config}, steps={args.num_inference_steps}, "
          f"canvas_length={args.canvas_length}, seed={args.seed})", flush=True)
    print(f"Example: {questions[0]['question_id']}: {questions[0]['question']}", flush=True)

    t0 = time.time()
    pipe = load_generation_backend(args.canvas_length)
    prompts = [q["question"] for q in questions]
    variants = generate_variants(pipe, prompts, N_VARIANTS, args)
    t_gen = time.time() - t0

    meta = {
        "model": MODEL_ID,
        "nli_model": NLI_MODEL_ID,
        "dataset": args.csv_dataset,
        "config": args.config,
        "n_questions": len(questions),
        "n_variants": N_VARIANTS,
        "temperature": TEMPERATURE,
        "num_inference_steps": args.num_inference_steps,
        "canvas_length": args.canvas_length,
        "gen_length": args.canvas_length,
        "gen_batch_size": args.gen_batch_size,
        "nli_batch_size": args.nli_batch_size,
        "confidence_threshold": None if args.no_adaptive_stop else args.confidence_threshold,
        "seed": args.seed,
    }
    save_json(variants_path, {"meta": meta, "variants": {q["question_id"]: v for q, v in zip(questions, variants)}})
    print(f"[checkpoint] variants saved to: {variants_path}", flush=True)

    del pipe
    torch.cuda.empty_cache()

    nli_pipeline = _build_nli_pipeline()
    t_nli_start = time.time()
    dict_results = {pos: v for pos, v in enumerate(variants)}
    detailed = {}
    items = list(dict_results.items())
    checkpoint_chunk = 20
    for i in range(0, len(items), checkpoint_chunk):
        chunk = dict(items[i:i + checkpoint_chunk])
        detailed.update(calculate_semantic_entropy_nli(chunk, nli_pipeline, nli_batch_size=args.nli_batch_size))
        if i + checkpoint_chunk < len(items):
            save_json(final_path, {"meta": meta, "results": build_results(questions, detailed, max_pos=i + checkpoint_chunk)})
    results = build_results(questions, detailed)
    save_json(final_path, {"meta": meta, "results": results})
    csv_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "values")
    csv_path = get_target_csv_path(csv_dir, args.config)
    update_benchmark_csv(csv_path, questions, detailed, args.csv_column)
    t_nli = time.time() - t_nli_start

    print(f"\n{'=' * 70}\n  SUMMARY (semantic entropy per question)\n{'=' * 70}", flush=True)
    for pos, q in enumerate(questions):
        info = detailed[pos]
        print(f"  {str(q['question_id']):<16} H={info['semantic_entropy']:.4f} "
              f"n_clusters={len(info['clusters'])} | {q['question'][:60]}", flush=True)

    print(f"\nGeneration: {t_gen:.1f}s | NLI: {t_nli:.1f}s | total: {time.time() - t0:.1f}s", flush=True)
    print(f"[SUCCESS] Saved to: {final_path}", flush=True)

    del nli_pipeline
    torch.cuda.empty_cache()


def _build_nli_pipeline():
    device_index = torch.cuda.device_count() - 1
    if torch.cuda.is_available() and device_index >= 0:
        device_str = f"cuda:{device_index}"
        pipeline_device = device_index
        dtype = torch.float16
    else:
        device_str = "cpu"
        pipeline_device = -1
        dtype = torch.float32

    tok = transformers.AutoTokenizer.from_pretrained(NLI_MODEL_ID)
    model = transformers.AutoModelForSequenceClassification.from_pretrained(NLI_MODEL_ID, dtype=dtype)
    if device_str != "cpu":
        model = model.to(device_str)

    pipeline_nli = transformers.pipeline(
        "text-classification",
        model=model,
        tokenizer=tok,
        device=pipeline_device,
        truncation=True,
        max_length=256,
    )
    print(f"NLI pipeline (DeBERTa) resident on {device_str}", flush=True)
    return pipeline_nli


if __name__ == "__main__":
    main()
