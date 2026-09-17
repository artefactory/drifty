# GenerateBaseSamplerOutputsAndExtractInfo

This module analyzes the diffusion process of masked language models (LLaDA, Dream, DiffusionGemma…). It lets you:
- **generate sequences while recording the full history** of every diffusion step;
- **extract metrics** from that history;
- **visualize** how the process evolves.

It has two roles:
1. **Qualitative exploration:** `main.py` runs a few prompts and produces heatmaps, per-step texts and interactive visualizations.
2. **Shared library:** the `get*` functions of `GetInfoFromBaseSamplerOutput.py` and the plots of `PlotResults.py` are reused by all of `PipelineTest/` (features, benchmarks, analysis). The large-scale detection pipeline (sample → evaluate → benchmark) is described in the [root README](../README.md).

---

## Architecture

```
GenerateWithMDLMSampler.py      → generation with full history (MDLM / Dream)
GetInfoFromBaseSamplerOutput.py → metric extraction from the history
PlotResults.py                  → visualizations (matplotlib + plotly)
run_experiments.py              → one experiment function per model family
main.py                         → entry point (models, prompts, sampler configs)
```

---

## 1. `GenerateWithMDLMSampler.py` — Generation with full history

`CreateBaseSampleWithHistory(sampler_type, messages, config, Script, seed_override=None)`:
1. parses the `Script` (model, seed, visualize) and `config` (sampler parameters) dataclasses;
2. loads the model and tokenizer through `dllm.utils`;
3. applies the chat template and calls `sampler.sample(..., return_dict=True)`;
4. returns `(outputs, tokenizer)`.

The `*WithCompleteHistory` samplers return a `BaseSamplerOutputCompleteHistory` (`dllm/core/samplers/base.py`). Each `histories_*` field is a list with one `(batch, seq_len)` tensor per step:

| Field | Description |
|-------|-------------|
| `sequences` | Final sequences (prompt + generation) |
| `histories_x` | Accepted canvas at each step: the actual text state after token selection |
| `histories_x0` | Tokens proposed by the model at each step (before confidence filtering) |
| `histories_logprobs` | Model confidence (probability of the predicted token) per position |
| `histories_unmask_logprobs` | Probability assigned to tokens **already unmasked** (consistency measure) |
| `histories_mask` | Binary mask of positions still masked |
| `histories_accepted` | Positions accepted at the step |
| `histories_remasking` | Positions **remasked** at the step (remasking only) |
| `histories_H` | Score $H = p_{\max} - p_{\text{current token}}$, which drives remasking |
| `histories_entropy` | Entropy of the predicted distribution per position |
| `histories_semantic_entropy`, `histories_semantic_dispersion`, `histories_semantic_logprobs`, `histories_semantic_bestlogprobs_label` | Semantic variants (token clustering), if the sampler computes them |
| `histories_num_transfer_tokens` | Number of tokens unmasked per step (dynamic budget, adjusted by remasking) |
| `start_idx_history` | Start index of the generation window for each example of the batch |
| `max_new_tokens`, `block_size`, `step_per_block` | Generation geometry |
| `attention_mask` | Batch attention mask |
| `sample_indices` | Sample indices in the dataset (filled in by `PipelineTest/sampling`) |

Available samplers: `dllm.core.samplers.MDLMSamplerWithCompleteHistory`, `dllm.core.samplers.MDLMSamplerRemaskingWithCompleteHistory`, `dllm.pipelines.dream.DreamSamplerWithCompleteHistory` and `dllm.pipelines.diffusiongemma.sampler.DiffusionGemmaSamplerWithCompleteHistory`.

---

## 2. `GetInfoFromBaseSamplerOutput.py` — Information extraction

Each function slices the generation window (`start_idx_history[i] : +max_new_tokens`) and returns a list indexed by batch example. Each element is a `(steps, max_new_tokens)` matrix unless stated otherwise.

| Function | Returns |
|----------|---------|
| `getLogProbs(outputs)` | Probability of the proposed tokens per position and step |
| `getUnmaskLogProbs(outputs)` | Probability of already revealed tokens (stability) |
| `getEntropy(outputs)` | Entropy per position at each step |
| `getH(outputs)` | H score (max confidence − confidence of the current token) |
| `getEachStepMask(outputs, remask=False)` | Diffusion masks (or remasking masks if `remask=True`) |
| `getEachStepChange(outputs)` | Binary matrix of token id changes between consecutive steps |
| `getNumTransferTokens(outputs)` | Budget of transferred tokens per step |
| `getEachStepGeneratedSequence(outputs, tokenizer)` | Decoded canvas tokens at each step |
| `getEachStepProposedSequence(outputs, tokenizer)` | Decoded $x_0$ prediction tokens at each step |
| `getEachStepProposedTokenIdSequence(outputs, tokenizer)` | Proposed token IDs (numpy) |
| `getLevenshtein(Proposed_sequences)` | Token-level Levenshtein distance between steps |
| `getEntropyJustUnmasked(outputs)` | 1D array: entropy of each token **at the step where it is unmasked** |
| `getLogProbsJustUnmasked(outputs)` | 1D array: probability of each token at the step where it is unmasked |
| `getSemanticEntropy`, `getSemanticDispersion`, `getSemanticLogprobs`, `getSemanticBestLogprobsLabel` | Corresponding semantic histories |

The features of `PipelineTest/features` and the perplexity / LN-entropy baselines of `PipelineTest/Benchmark` are built from these functions.

---

## 3. `PlotResults.py` — Visualizations

Generates heatmaps (axes: step × position) and interactive charts:

| Matplotlib (`.png`) | Plotly (`.html`, tokens on hover) | Contents |
|---------------------|-----------------------------------|----------|
| `plotLogProbs` | `PlotlyLogProbs` | Model confidence (optionally with entropy overlaid) |
| `plotUnmaskLogProbs` | `PlotlyUnmaskLogProbs` | Stability of already revealed tokens |
| `plotMasks` | — | Evolution of the diffusion (or remasking) mask |
| `plotChanges` | `PlotlyChanges` | Positions where the model changes its mind between two steps |
| `plotLevenshtein` | — | Edit distance between successive proposals |
| `plotH` | `PlotlyH` | Uncertainty score $H$ |
| `plotEntropy` | `PlotlyEntropy` | Entropy per position |
| `plotNumTransferredTokens` | — | Unmasking budget per step |
| `plotAttentionMask` | — | Attention mask |
| — | `PlotlySemanticBestLabels` | Best semantic label per position |

- **Remasking:** **remasked** positions are overlaid in black on the heatmaps when remasking is active.
- **Per-step texts:** `getTxt(sequences, save_path, proposed_sequence)` writes `proposed_text EX - <j>.txt` / `generated_text EX - <j>.txt`, with the text at each step followed by the final sentence.

---

## 4. `run_experiments.py` — Experiments

| Function | Sampler | Plots produced |
|----------|---------|----------------|
| `run_experiment_MDML` | `MDLMSamplerWithCompleteHistory` | logprobs, masks, changes, unmask logprobs, Levenshtein, attention, H, transfers + Plotly |
| `run_experiment_MDML_remasking` | `MDLMSamplerRemaskingWithCompleteHistory` | same + overlaid remasking masks |
| `run_experiment_Dream` | `DreamSamplerWithCompleteHistory` | logprobs, masks, changes, unmask logprobs, Levenshtein, attention, H + Plotly (with entropy) |
| `run_experiment_DiffusionGemma` | `DiffusionGemmaSamplerWithCompleteHistory` | logprobs, masks, changes, Levenshtein, attention + Plotly (with entropy) |

For DiffusionGemma, encoder and decoder layers are split across 2 GPUs: layers 0–14 on GPU 0, layers 15–29 on GPU 1. Another `device_map` can be passed in `model_cfg`.

### Experiment without remasking (`run_experiment_MDML`)

The standard diffusion process:
1. The canvas starts fully masked (generation window).
2. At each step, the model proposes tokens for all masked positions.
3. The $k$ highest-confidence positions are permanently unmasked.
4. The budget $k$ follows a linear scheduler.
5. An unmasked token can **never** be masked again.

### Experiment with remasking (`run_experiment_MDML_remasking`)

After each unmasking step (except the last one), the sampler can **remask** already revealed tokens if the model finds them inconsistent with the context:

1. **Computing H:** for each unmasked position, $H_i = p_{\max}(i) - p_{\text{current token}}(i)$.
2. **Decision:** positions with $H > 0$ are candidates. Each one is remasked following a Bernoulli draw with parameter $\text{clamp}(H_i, 0, 1)$.
3. **Budget redistribution:** remasked tokens must be unmasked again. `num_transfer_tokens` is redistributed uniformly over the remaining steps, with the surplus given first to the steps with the smallest budget.
4. **Scope:** only tokens in the generation window (not the prompt) can be remasked.

This mechanism lets the model **correct its mistakes** instead of committing permanently to low-quality tokens.

---

## Usage

```bash
uv sync --extra diffugemma
uv run --extra diffugemma python GenerateBaseSamplerOutputsAndExtractInfo/main.py
```

- **Working directory:** run from the repository root (the `dir` output folders are relative to it).
- **Environment:** `diffugemma`, because `run_experiments.py` imports `DiffusionGemmaForBlockDiffusion` from `transformers`.

Configuration is done in `main.py`:

- **`MODELS_TO_TEST`:** list of `{"name", "path", "dir", "type"?, "remasking"?}`. `type` is `"mdlm"` (default), `"dream"` or `"diffusiongemma"`; `remasking: True` selects the remasking experiment (MDLM).
- **`messages`:** prompts in chat format (one example per conversation).
- **Sampler configs:**

```python
@dataclass
class SamplerConfig(dllm.core.samplers.MDLMSamplerConfig):   # MDLM (LLaDA, ...)
    steps: int = 128
    max_new_tokens: int = 128
    block_size: int = 128
    temperature: float = 0.0
    remasking: str = "low_confidence"

@dataclass
class DreamSamplerConfig(dllm.pipelines.dream.DreamSamplerConfig):
    steps: int = 128
    max_new_tokens: int = 128
    alg: str = "maskgit_plus"

@dataclass
class GemmaSamplerConfig(DiffusionGemmaSamplerConfig):
    max_new_tokens: int = 32
    steps: int = 16
    canvas_length: int = 32
    entropy_bound: float = 0.1
    entropy_threshold: float = 0.005
    stability_threshold: int = 1
```

Results are written to `model_cfg["dir"]` (e.g. `GenerateBaseSamplerOutputsAndExtractInfo/Results/DiffusionGemma/`): per-step text files, `.png` heatmaps, `.html` visualizations.
