"""
Benchmark/tracedet.py
=====================
TraceDet baseline (Chang et al. 2025, "TraceDet: Hallucination Detection from
the Decoding Trace of Diffusion Large Language Models",
https://github.com/chang-sx/TraceDet), "entropy" task.

Input of the detector : the full entropy trace of each sample, i.e. a
[n_steps, gen_length] matrix (entropy of every generated position at every
denoising step, cf. getEntropy). The model is a faithful re-implementation of
TimeHalu (models/TimeHalu.py + models/encoders/transformer_simple.py of the
original repo), without the TimeX++ / reformer dependencies:

  - encoder  : Linear(d_inp) + sinusoidal time encoding (d_pe=8) -> 3-layer
               Transformer encoder (1 head, ff=8, dropout=0.2) over the steps ;
  - extractor: MaskGenerator, 2-layer Transformer decoder that predicts, for
               every step, the probability of keeping it (information
               bottleneck on the sub-trajectory) ;
  - head     : masked mean-pooling of the step embeddings -> MLP -> 2 classes ;
  - loss     : Poly1-CE(masked) + 0.2 * Poly1-CE(full) + gsat * ||mask||_2 ;
  - AdamW(lr=2e-4, wd=0.1), batch 64, 100 epochs, early stopping on the
    validation AUROC (patience 20), ReduceLROnPlateau after epoch 20.

Split : the common 80/20 sequential split (data_split.py). The validation set
used for early stopping is a stratified slice of the train split : the test
split is never seen during training / model selection.

Usage:
    q -G 1 uv run PipelineTest/Benchmark/tracedet.py --config <CONFIG|all>
"""

import sys
import os
import json
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, average_precision_score, accuracy_score

from GenerateBaseSamplerOutputsAndExtractInfo.GetInfoFromBaseSamplerOutput import getEntropy
from PipelineTest.features.utils import load_outputs
from PipelineTest.eval_configs import CONFIGS
from PipelineTest.Benchmark.data_split import load_eval_data, get_train_test_split_for_config
from PipelineTest.Benchmark.values_csv import DEFAULT_VALUES_DIR, update_values_csv


# Colonne que cette baseline ecrit dans values/<model>/<config_name>.csv
# (meme nom que PipelineTest/Benchmark/main.py::SAMPLE_VALUE_COLUMNS).
TRACEDET_CSV_COLUMNS = ["tracedet"]

# Hyperparametres par defaut de TraceDet (train_main.py du repo original).
DEFAULT_HPARAMS = {
    "num_epochs": 100,
    "batch_size": 64,
    "lr": 2e-4,
    "weight_decay": 0.1,
    "patience": 20,
    "nlayers": 3,
    "dropout": 0.2,
    "d_pe": 8,
    "gsat": 0.1,
    "connect": 0.0,
    "val_ratio": 0.1,
}


# =========================================================================
# MODEL (TimeHalu, task="entropy")
# =========================================================================

class PositionalEncodingTF(nn.Module):
    """Sinusoidal encoding of the (real-valued) step index, as in TimeX++."""

    def __init__(self, d_model, max_len):
        super().__init__()
        self.max_len = max_len
        self._num_timescales = d_model // 2

    def forward(self, times):  # times: [T, B]
        timescales = self.max_len ** torch.linspace(0, 1, self._num_timescales, device=times.device)
        scaled_time = times.float().unsqueeze(2) / timescales[None, None, :]
        return torch.cat([torch.sin(scaled_time), torch.cos(scaled_time)], dim=-1)  # [T, B, d_pe]


class TraceEncoder(nn.Module):
    """TransformerMVTS.embed(aggregate=False): per-step embeddings [T, B, d_pe + d_inp]."""

    def __init__(self, d_inp, max_len, nlayers=3, d_pe=8, dim_feedforward=8, dropout=0.2):
        super().__init__()
        self.pos_encoder = PositionalEncodingTF(d_pe, max_len)
        self.MLP_encoder = nn.Linear(d_inp, d_inp)
        nn.init.xavier_uniform_(self.MLP_encoder.weight)
        layer = nn.TransformerEncoderLayer(
            d_model=d_pe + d_inp, nhead=1, dim_feedforward=dim_feedforward,
            dropout=dropout, batch_first=False,
        )
        self.transformer_encoder = nn.TransformerEncoder(layer, nlayers, enable_nested_tensor=False)

    def forward(self, src, times):
        x = torch.cat([self.pos_encoder(times), self.MLP_encoder(src)], dim=2)
        return self.transformer_encoder(x)


class MaskGenerator(nn.Module):
    """Step-selection network (information bottleneck on the sub-trajectory)."""

    def __init__(self, d_z, max_len, d_pe=8, n_dec_layers=2, nhead=1, dim_feedforward=32,
                 tau=1.0, use_ste=True):
        super().__init__()
        self.tau = tau
        self.use_ste = use_ste
        dec_layer = nn.TransformerDecoderLayer(d_model=d_z, nhead=nhead, dim_feedforward=dim_feedforward)
        self.mask_decoder = nn.TransformerDecoder(dec_layer, num_layers=n_dec_layers)
        self.time_prob_net = nn.Linear(d_z, 2)
        nn.init.xavier_uniform_(self.time_prob_net.weight)
        self.time_prob_net.bias.data.fill_(0.01)
        self.ln_in_tgt = nn.LayerNorm(d_z)
        self.pos_encoder = PositionalEncodingTF(d_pe, max_len)

    def forward(self, z_seq, src, times):
        x = self.ln_in_tgt(torch.cat([src, self.pos_encoder(times)], dim=-1))
        p_time = self.time_prob_net(self.mask_decoder(tgt=x, memory=z_seq)).transpose(0, 1)  # [B, T, 2]
        prob = p_time.softmax(dim=-1)
        mask_reparam = F.gumbel_softmax(torch.log(prob + 1e-9), tau=self.tau, hard=self.use_ste)[..., 1]
        return prob[..., 1].unsqueeze(-1), mask_reparam  # [B, T, 1], [B, T]


def poly1_cross_entropy(logits, labels, epsilon=1.0):
    pt = torch.sum(F.one_hot(labels, num_classes=logits.shape[-1]) * F.softmax(logits, dim=-1), dim=-1)
    return (F.cross_entropy(logits, labels, reduction="none") + epsilon * (1 - pt)).mean()


class TimeHalu(nn.Module):
    def __init__(self, d_inp, max_len, nlayers=3, d_pe=8, dropout=0.2, gsat=0.1, connect=0.0):
        super().__init__()
        d_fi = d_inp + d_pe
        self.d_inp = d_inp
        self.gsat = gsat
        self.connect = connect
        self.encoder = TraceEncoder(d_inp, max_len, nlayers=nlayers, d_pe=d_pe, dropout=dropout)
        self.extractor = MaskGenerator(d_z=d_fi, max_len=max_len, d_pe=d_pe)
        self.mlp = nn.Sequential(
            nn.Linear(d_fi, d_fi),
            nn.ReLU(),
            nn.LayerNorm(d_fi),
            nn.Linear(d_fi, 2),
        )

    def forward(self, X, times):
        """X: [T, B, d_inp], times: [T, B]. Returns (logits_masked, logits_full, mask_prob)."""
        emb = self.encoder(X, times)                    # [T, B, d_fi]
        mask_prob, _ = self.extractor(emb, X, times)    # [B, T, 1]
        m = mask_prob.transpose(0, 1)                   # [T, B, 1]
        emb_masked = torch.sum(emb * m, dim=0) / (torch.sum(m, dim=0) + 1e-8)
        emb_full = torch.sum(emb, dim=0) / self.d_inp   # (sic, as in the original code)
        emb_full = F.layer_norm(emb_full, emb_full.shape[-1:])
        emb_masked = F.layer_norm(emb_masked, emb_masked.shape[-1:])
        return self.mlp(emb_masked), self.mlp(emb_full), mask_prob

    def loss(self, logits_masked, logits_full, mask_prob, y):
        mask_loss = self.gsat * torch.sqrt(torch.sum(mask_prob ** 2))
        if self.connect:
            diff = mask_prob[:, 1:, :] - mask_prob[:, :-1, :]
            mask_loss = mask_loss + self.connect * diff.norm(p=2) / diff.numel()
        return 0.2 * poly1_cross_entropy(logits_full, y) + poly1_cross_entropy(logits_masked, y) + mask_loss

    @torch.no_grad()
    def predict_proba(self, X, times, batch_size=512):
        """P(hallucination) = softmax(logits_masked)[:, 1], X: [T, N, d_inp]."""
        self.eval()
        out = []
        for s in range(0, X.shape[1], batch_size):
            logits_masked, _, _ = self(X[:, s:s + batch_size], times[:, s:s + batch_size])
            out.append(logits_masked.softmax(dim=-1)[:, 1])
        return torch.cat(out).cpu().numpy()


# =========================================================================
# DATA
# =========================================================================

def get_entropy_traces(outputs, positions):
    """Entropy trace [N, n_steps, gen_length] (float32) for the given tensor positions."""
    entropies = getEntropy(outputs)
    traces = np.stack([np.asarray(entropies[p], dtype=np.float32) for p in positions])
    return np.nan_to_num(traces, nan=0.0, posinf=0.0, neginf=0.0)


def _to_time_major(traces, device):
    """[N, T, D] numpy -> X [T, N, D], times [T, N] (steps numbered 1..T, as in TraceDet)."""
    X = torch.from_numpy(traces).float().transpose(0, 1).contiguous().to(device)
    T, N = X.shape[0], X.shape[1]
    times = torch.arange(1, T + 1, device=device, dtype=torch.float32).unsqueeze(1).repeat(1, N)
    return X, times


# =========================================================================
# TRAINING
# =========================================================================

def train_tracedet(X_tr, y_tr, X_val, y_val, hp, device, seed=42, verbose=True):
    """Train TimeHalu on [N, T, D] traces, early stopping on val AUROC. Returns best model."""
    torch.manual_seed(seed)
    np.random.seed(seed)

    Xtr, ttr = _to_time_major(X_tr, device)
    Xva, tva = _to_time_major(X_val, device)
    ytr = torch.from_numpy(np.asarray(y_tr)).long().to(device)

    model = TimeHalu(
        d_inp=Xtr.shape[-1], max_len=Xtr.shape[0], nlayers=hp["nlayers"], d_pe=hp["d_pe"],
        dropout=hp["dropout"], gsat=hp["gsat"], connect=hp["connect"],
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=hp["lr"], weight_decay=hp["weight_decay"])
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=0.1, patience=5, threshold=1e-5,
        threshold_mode="rel", cooldown=0, min_lr=1e-8, eps=1e-8,
    )
    wait_for_scheduler = 20

    gen = torch.Generator(device="cpu").manual_seed(seed)
    n_train = Xtr.shape[1]
    best_val, best_epoch, best_state, counter = -np.inf, -1, None, 0

    for epoch in range(hp["num_epochs"]):
        model.train()
        perm = torch.randperm(n_train, generator=gen).to(device)
        total_loss, n_batches = 0.0, 0
        for s in range(0, n_train, hp["batch_size"]):
            b = perm[s:s + hp["batch_size"]]
            optimizer.zero_grad()
            logits_masked, logits_full, mask_prob = model(Xtr[:, b], ttr[:, b])
            loss = model.loss(logits_masked, logits_full, mask_prob, ytr[b])
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            n_batches += 1

        val_auc = roc_auc_score(y_val, model.predict_proba(Xva, tva))
        if verbose:
            print(f"    epoch {epoch + 1:3d}/{hp['num_epochs']}  loss={total_loss / n_batches:.4f}  "
                  f"val AUROC={val_auc:.4f}", flush=True)

        if val_auc > best_val:
            best_val, best_epoch, counter = val_auc, epoch, 0
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        else:
            counter += 1
            if counter >= hp["patience"]:
                if verbose:
                    print(f"    early stopping at epoch {epoch + 1} (best epoch {best_epoch + 1}, "
                          f"val AUROC={best_val:.4f})")
                break

        if epoch > wait_for_scheduler:
            scheduler.step(val_auc)

    model.load_state_dict(best_state)
    return model, {"best_val_roc_auc": float(best_val), "best_epoch": int(best_epoch + 1)}


# =========================================================================
# RUN ONE CONFIG
# =========================================================================

def run_config(config_name, seed=42, device=None, hparams=None, update_csv=False,
               values_dir=DEFAULT_VALUES_DIR, verbose=True):
    hp = dict(DEFAULT_HPARAMS, **(hparams or {}))
    cfg = CONFIGS[config_name]
    outputs_path = cfg["outputs_path"]
    eval_json = cfg["eval_json"]
    device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))

    print(f"\n{'=' * 70}")
    print(f"  TRACEDET: {cfg['name']} ({config_name})")
    print(f"  [split: 80/20 sequential | val={hp['val_ratio']:.0%} of train | device={device}]")
    print(f"{'=' * 70}")

    if not os.path.exists(outputs_path):
        print(f"  [SKIP] outputs not found: {outputs_path}")
        return None
    if not os.path.exists(eval_json):
        print(f"  [SKIP] eval_json not found: {eval_json}")
        return None

    labels, indices, n_missing_label, _ = load_eval_data(eval_json)
    print(f"  Raw samples: {len(labels)} (correct={np.sum(labels == 0)}, halluc={np.sum(labels == 1)}) | "
          f"Dropped: {n_missing_label} unusable")
    if len(labels) == 0:
        print("  [SKIP] no usable samples.")
        return None

    train_idx, test_idx = get_train_test_split_for_config(config_name, indices, values_dir=values_dir)

    outputs = load_outputs(outputs_path)
    idx_to_pos = {int(idx): pos for pos, idx in enumerate(outputs.sample_indices.numpy())}
    traces = get_entropy_traces(outputs, [idx_to_pos[int(idx)] for idx in indices])
    del outputs
    print(f"  Entropy traces: {traces.shape} (samples, steps, gen_length)")

    # Validation (early stopping) taken inside the train split only.
    fit_idx, val_idx = train_test_split(
        train_idx, test_size=hp["val_ratio"], stratify=labels[train_idx], random_state=seed,
    )
    model, train_info = train_tracedet(
        traces[fit_idx], labels[fit_idx], traces[val_idx], labels[val_idx],
        hp, device, seed=seed, verbose=verbose,
    )

    X_all, t_all = _to_time_major(traces, device)
    scores = model.predict_proba(X_all, t_all)
    y_test, s_test = labels[test_idx], scores[test_idx]

    roc = roc_auc_score(y_test, s_test)
    pr = average_precision_score(y_test, s_test)
    acc = accuracy_score(y_test, (s_test >= 0.5).astype(int))
    thresholds = np.linspace(0, 1, 201)
    accs = [accuracy_score(y_test, (s_test >= t).astype(int)) for t in thresholds]

    results = {
        "name": f"TraceDet_{config_name}",
        "config": config_name,
        "outputs_path": outputs_path,
        "eval_json": eval_json,
        "hparams": hp,
        "seed": seed,
        "trace_shape": list(traces.shape[1:]),
        "n_train": int(len(fit_idx)),
        "n_val": int(len(val_idx)),
        "n_test": int(len(test_idx)),
        **train_info,
        "test_roc_auc": float(roc),
        "test_pr_auc": float(pr),
        "test_accuracy": float(acc),
        "test_best_accuracy": float(max(accs)),
        "test_best_threshold": float(thresholds[int(np.argmax(accs))]),
        "test_pos_rate": float(np.mean(y_test)),
        "n_halluc": int(np.sum(y_test == 1)),
        "n_correct": int(np.sum(y_test == 0)),
    }

    # Per-sample scores (train+test), keyed by original sample index, for
    # downstream per-sample CSV export (Benchmark/main.py).
    split_of = np.array(["train"] * len(labels), dtype=object)
    split_of[test_idx] = "test"
    results["samples"] = {
        int(idx): {
            "label_hallucination": int(labels[pos]),
            "split": str(split_of[pos]),
            "tracedet": float(scores[pos]),
        }
        for pos, idx in enumerate(indices)
    }

    print(f"  ROC-AUC: {roc:.4f}  |  PR-AUC: {pr:.4f}  |  BestAcc: {max(accs):.4f}  "
          f"(best val AUROC={train_info['best_val_roc_auc']:.4f} @ epoch {train_info['best_epoch']})")

    if update_csv:
        update_values_csv(config_name, results["samples"], TRACEDET_CSV_COLUMNS, values_dir=values_dir)

    return results


# =========================================================================
# MAIN
# =========================================================================

def parse_args():
    parser = argparse.ArgumentParser(description="TraceDet (TimeHalu, entropy trace) hallucination detection baseline")
    parser.add_argument("--config", type=str, default="all", choices=list(CONFIGS.keys()) + ["all"],
                        help="Config to evaluate, or 'all'")
    parser.add_argument("--output_dir", type=str, default=None, help="Directory to save results")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default=None, help="cuda / cpu (default: cuda if available)")
    parser.add_argument("--values_dir", type=str, default=DEFAULT_VALUES_DIR,
                        help="Racine des CSV par echantillon a mettre a jour (values/<model>/<config_name>.csv).")
    parser.add_argument("--no_update_csv", action="store_true",
                        help="Ne pas toucher aux CSV de values/ (par defaut ils sont mis a jour).")
    for k, v in DEFAULT_HPARAMS.items():
        parser.add_argument(f"--{k}", type=type(v), default=v)
    return parser.parse_args()


def main():
    args = parse_args()
    hparams = {k: getattr(args, k) for k in DEFAULT_HPARAMS}

    output_dir = args.output_dir or os.path.abspath(os.path.join(os.path.dirname(__file__), "eval"))
    os.makedirs(output_dir, exist_ok=True)
    configs_to_run = list(CONFIGS.keys()) if args.config == "all" else [args.config]

    all_results = {}
    for config_name in configs_to_run:
        result = run_config(config_name, seed=args.seed, device=args.device, hparams=hparams,
                            update_csv=not args.no_update_csv, values_dir=args.values_dir)
        if result is not None:
            all_results[config_name] = result

    print(f"\n\n{'=' * 70}")
    print(f"  SUMMARY TABLE (TRACEDET)")
    print(f"{'=' * 70}")
    print(f"  {'Config':<60} {'ROC-AUC':>8} {'PR-AUC':>8} {'BestAcc':>8}")
    print(f"  {'-' * 86}")
    for name, r in all_results.items():
        print(f"  {name:<60} {r['test_roc_auc']:>8.4f} {r['test_pr_auc']:>8.4f} {r['test_best_accuracy']:>8.4f}")

    suffix = CONFIGS[configs_to_run[0]]["name"] if len(configs_to_run) == 1 else "all"
    out_path = os.path.join(output_dir, f"tracedet_results_{suffix}.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2)
    print(f"\n  Results saved to: {out_path}")
    return all_results


if __name__ == "__main__":
    main()
