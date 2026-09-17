# PipelineTest/features — Feature extraction

This folder computes, for each sample, **scalar features** from the diffusion history (`outputs_<cfg>.pt`, step 1 output). These features feed `Benchmark/Baseline_and_markovian_features.py`, `FeatureSelection/`, `CrossTrainAndTest/` and `analysis/`.

All functions take the merged `BaseSamplerOutputCompleteHistory` object and return an `(N,)` array, in the sample order of the `.pt`. They read the history through the `get*` functions of [`GenerateBaseSamplerOutputsAndExtractInfo/GetInfoFromBaseSamplerOutput.py`](../../GenerateBaseSamplerOutputsAndExtractInfo/README.md). The generation window is sliced from `start_idx_history`, giving `(steps, max_new_tokens)` matrices.

```
features/
├── io.py         # merging the batches of a .pt (mergeOutputsList), tensor padding/concat, plot helpers
├── utils.py      # output loading, labels, eval JSON ↔ outputs alignment, padding
├── baseline.py   # static aggregations (means / variances) + shape of the V(t) curve
└── markovian.py  # dynamic features (exponential decay, parametric fit)
```

---

## `utils.py`

| Function | Purpose |
|----------|---------|
| `load_outputs(path)` | Loads a `.pt` and merges the list of batches into a single object if needed (`io.mergeOutputsList`) |
| `get_labels(eval_json)` | Labels sorted by `index` (`yes` → 1, otherwise 0) |
| `get_labels_for_outputs(eval_json, outputs)` | Same labels, reordered by `outputs.sample_indices` |
| `match_samples(eval_json, outputs)` | `(positions, labels, data)`: position in the `.pt` of each JSON entry |
| `get_pad_token_id(outputs_path)` | Reads `tokenizer_<cfg>.pt`, stored next to `outputs_<cfg>.pt` |
| `compute_padding_mask(outputs, positions, pad_token_id)` | Mask of padding positions, for the `NoPad` variants |

Benchmarks (`Benchmark/`) prefer `Benchmark/data_split.load_eval_data`, which excludes `unclear` samples, then realign through `outputs.sample_indices`.

---

## `baseline.py` — `get_baseline_features(outputs, pad_token_id=None)`

Notation: `H(t, d)` is the entropy at step `t` for position `d`; `p(t, d)` is the probability of the proposed token; "masked" means positions still masked at step `t`; "just unmasked" means the step at which each token is unmasked.

| Family | Features |
|--------|----------|
| Means | `MeanEntropy`, `MeanMaskedEntropy`, `MeanEntropyJustUnmasked`, `MeanLogProb`, `MeanLogProbMasked`, `MeanLogProbJustUnmasked` |
| Variances | `VarEntropy`, `VarEntropyAcrossTokens`, `VarMaskedEntropy`, `VarMaskedEntropyAcrossTokens`, `VarEntropyJustUnmasked`, `VarLogProb`, `VarLogProbAcrossTokens`, `VarLogProbMasked`, `VarLogProbMaskedAcrossTokens`, `VarLogProbJustUnmasked` |
| Rate of change | `MeanVarTauEntropy`, `VarTauEntropy`: finite differences (window 5) of the per-step mean entropy |
| Shape of `V(t) = Var_d(H(t,d) \| masked)` | `AUCVarMaskedEntropyCurve`, `MaxVarMaskedEntropyCurve`, `ArgmaxVarMaskedEntropyCurve`, `SkewnessVarMaskedEntropyCurve`, `KurtosisVarMaskedEntropyCurve`, `MeanCurvatureMaskedEntropyCurve` |
| Without padding (if `pad_token_id`) | `MeanMaskedEntropyNoPad`, `VarMaskedEntropyAcrossTokensNoPad`, `MeanLogProbMaskedNoPad`, `VarLogProbMaskedAcrossTokensNoPad` |

---

## `markovian.py` — `get_markovian_features(outputs, k_tokens=None)`

These features come from the Markovian (Ornstein-Uhlenbeck) model of entropy decay across steps.

| Feature | Definition |
|---------|------------|
| `AlphaEntropy`, `BetaEntropy` | Mean over tokens `d` of the linear fit `ln H(t,d) = α_d·t + β_d` (α ≈ −1/τ) |
| `AlphaEntropyAvg`, `BetaEntropyAvg` | Same fit on the mean trajectory of the `k_tokens` highest-entropy tokens |
| `AlphaLogProb`, `BetaLogProb` | Same fit on `ln p(t,d)` |
| `C_fit`, `Tau_fit`, `M_fit` | Nonlinear fit `H(t) = C·t·exp(−(t−m)/τ)` on the mean top-`k_tokens` trajectory (NaN if the fit fails) |

---

## Usage

```python
from PipelineTest.eval_configs import CONFIGS
from PipelineTest.features.utils import load_outputs, get_pad_token_id
from PipelineTest.features.baseline import get_baseline_features
from PipelineTest.features.markovian import get_markovian_features
from PipelineTest.Benchmark.data_split import load_eval_data
import numpy as np

cfg = CONFIGS["llada16_32tokens_2500samples_triviaqa_evalqwen_seed42"]
outputs = load_outputs(cfg["outputs_path"])

# Features for every sample in the .pt
X_base, names_base = get_baseline_features(outputs, pad_token_id=get_pad_token_id(cfg["outputs_path"]))
X_markov, names_markov = get_markovian_features(outputs, k_tokens=64)

# Align with the yes/no labels of the evaluated JSON
labels, indices, _, _ = load_eval_data(cfg["eval_json"])
pos_of = {int(idx): pos for pos, idx in enumerate(outputs.sample_indices.numpy())}
positions = [pos_of[int(i)] for i in indices]
X = np.column_stack([X_base, X_markov])[positions]      # (n_labels, F), aligned with `labels`
```
