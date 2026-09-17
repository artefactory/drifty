# PipelineTest/sampling — Steps 1 and 2

This folder contains the first two steps of the pipeline:

1. **Sampling** (`sample_answers_*.py`): generates one answer per question while recording the full diffusion history.
2. **Evaluation** (`eval.py`): an LLM judge decides whether each answer is a hallucination.

Commands are run from the repository root.

```
load_data.py                   → QA dataset loading (question + reference answers)
sample_answers_dream_llada.py  → step 1 for LLaDA / Dream        (lladadream env)
sample_answers_diffugemma.py   → step 1 for DiffusionGemma       (diffugemma env)
eval.py                        → step 2: LLM judge via vLLM
```

---

## `load_data.py`

Each loader returns `(messages, labels)`:
- **`messages`**: `[[{"role": "user", "content": question}], ...]` (chat format);
- **`labels`**: `[{"question": question, "label": [gold, alias1, alias2, ...]}, ...]`.

The **gold answer is always first** in `label`: `eval.py` uses it as the "Expected Answer" and the rest as aliases.

| Function | Source | Answers |
|----------|--------|---------|
| `load_triviaqa` | `trivia_qa` / `rc` / train | `answer.value` then `answer.aliases` (deduplicated) |
| `load_naturalquestion` | Natural Questions (parquet from the local HF cache) | short answers from the annotations (questions without a short answer are skipped) |
| `load_hotpotqa` | `hotpot_qa` / `fullwiki` / train | `answer` |

The subsample is drawn with `shuffle(seed=data_seed)`, then the first `num_samples` are kept.

---

## Step 1 — Sampling

### LLaDA / Dream

```bash
uv sync --extra lladadream

# 1 GPU
uv run --extra lladadream python PipelineTest/sampling/sample_answers_dream_llada.py

# multi-GPU: the dataset is sharded by rank (rank r → indices r, r+W, r+2W, ...)
uv run --extra lladadream torchrun --nproc_per_node=4 \
    PipelineTest/sampling/sample_answers_dream_llada.py --num_sample 2500 --batch_size 8
```

| Model | Checkpoint | Sampler | Config |
|-------|-----------|---------|--------|
| `llada` | `GSAI-ML/LLaDA-8B-Instruct` | `MDLMSamplerWithCompleteHistory` | `low_confidence`, `block_size = max_new_tokens` |
| `dream` | `Dream-org/Dream-v0-Instruct-7B` | `DreamSamplerWithCompleteHistory` | `maskgit_plus`, `top_p = 1.0` |

### DiffusionGemma

```bash
uv sync --extra diffugemma
uv run --extra diffugemma python PipelineTest/sampling/sample_answers_diffugemma.py --num_sample 2500 --batch_size 16
```

- **Model:** `google/diffusiongemma-26B-A4B-it`, loaded in `bfloat16`. Encoder and decoder layers 0–14 go on GPU 0, layers 15–29 on GPU 1: this requires **2 GPUs** and **a single process** (no `torchrun`).
- **Sampler:** `DiffusionGemmaSamplerWithCompleteHistory`.
- **Prompt:** tokenization is done prompt by prompt (`apply_chat_template(tokenize=False)` then `encode`).
- **`brief` suffix:** with `"brief": True`, ` please answer briefly` is appended to each question.

### CLI arguments (shared)

| Argument | Default | Description |
|----------|---------|-------------|
| `--num_sample` | `2500` | Questions per dataset |
| `--batch_size` | `8` (LLaDA/Dream), `16` (Gemma) | Batch size per rank |
| `--data_seed` | `42` | Seed for question sampling |

Generation is **greedy** (`temperature = 0.0`). The model seed is `42 + rank`.

### Choosing configurations

Configurations are chosen by editing the constants at the top of the script. The script loops over models × datasets × configs and loads each model only once.

```python
DATASETS = ["hotpotqa", "naturalquestion", "triviaqa"]
SAMPLERS = ["dream"]                        # or ["llada", "dream"]; ["diffgemma"] on the Gemma side
STEP_TOKEN_COMBOS = [(16, 32), (32, 64)]    # (steps, max_new_tokens) — LLaDA / Dream
GEMMA_CONFIGS = [                           # DiffusionGemma
    {"max_new_tokens": 32, "steps": 16, "canvas_length": 32, "brief": True},
]
```

### Outputs

Suffix: `<model>_<steps>steps_<tokens>tokens[_<canvas>canvas]_<dataset>_<N>samples[_brief]_seed<data_seed>`

| File | Contents |
|------|----------|
| `PipelineTest/res/results_<suffix>/outputs_<suffix>.pt` | `BaseSamplerOutputCompleteHistory` merged across all ranks and batches (sorted by global index, `sample_indices` filled in) |
| `PipelineTest/res/results_<suffix>/tokenizer_<suffix>.pt` | Serialized tokenizer (used to recover `pad_token_id`) |
| `PipelineTest/res/to_eval_V2/results_<suffix>.json` | List of `{question, label, answer, index, temperature}` sorted by `index` |

During the run, each rank writes `_outputs_rank<r>.pt` and `_shard_rank<r>.pt`. Rank 0 merges and then deletes them.

Main history fields (one `(batch, seq_len)` tensor per step): `histories_x` (accepted canvas), `histories_x0` (proposed tokens), `histories_logprobs`, `histories_entropy`, `histories_mask`, `histories_H`, plus `start_idx_history`, `max_new_tokens` and `block_size`. See [`GenerateBaseSamplerOutputsAndExtractInfo/README.md`](../../GenerateBaseSamplerOutputsAndExtractInfo/README.md).

---

## Step 2 — Evaluation (`eval.py`)

For each JSON in `PipelineTest/res/to_eval_V2/`, an LLM judge compares `answer` with `label` and writes the enriched file, under the same name, to `PipelineTest/res/eval_V2/`.

```bash
uv run --extra lladadream python PipelineTest/sampling/eval.py
```

### Configuration (block at the top of the file)

| Constant | Default | Purpose |
|----------|---------|---------|
| `TO_EVAL_PATH` | `PipelineTest/res/to_eval_V2` | JSON files to evaluate |
| `SAVE_PATH` | `PipelineTest/res/eval_V2` | Evaluated JSON files |
| `FILENAME_FILTER` | a config name | Substring the file name must contain (`None` = all) |
| `MODEL_NAME` | `Qwen/Qwen3.5-9B` | Model served by vLLM |
| `EVAL_SERVER_DIR` | `<root>/../eval_server` | Separate uv project containing `vllm` |
| `VLLM_PORT` | `8000` | OpenAI-compatible API port |
| `USE_SERVER_MANAGER` | `True` | `eval.py` starts and stops vLLM itself; `False` if the server is already running |
| `MAX_CONCURRENT_REQUESTS` | `16` | Parallel requests |
| `MAX_NEW_TOKENS`, `MAX_RETRIES` | `128`, `3` | Tokens per judgment, retries |

The judge runs in a separate venv: `eval.py` only depends on `openai` and `jinja2`, so it works with both project environments.

### Flow

1. **Empty answers:** labeled `is_hallucination = "yes"` locally, without calling the judge.
2. **Batched judging:** the remaining answers are grouped 32 at a time into a single prompt (`BATCH_TRUE_FALSE_PROMPT`). The judge returns a JSON array `[{index, judgment, explanation}]`, with thinking disabled and `temperature = 0`.
3. **Fallback:** if a batch is still malformed after `MAX_RETRIES` attempts, each sample in it is judged individually (`TRUE_FALSE_PROMPT`).
4. **Judgment → label mapping:**

| Judgment | `is_hallucination` |
|----------|--------------------|
| `true` (compatible with the gold answer or an alias) | `no` |
| `false` (contradiction or different topic) | `yes` |
| `unclear` (refusal, "I don't know", unparsable output) | `unclear` |

Benchmarks only keep `yes` (label 1) and `no` (label 0): `unclear` samples are excluded from the pool.

### After evaluation

Register the configuration in [`PipelineTest/eval_configs.py`](../README.md#eval_configspy) (`outputs_path`, `eval_json`, `tokenizer_path`, `name`), then move on to [step 3](../Benchmark/README.md).
