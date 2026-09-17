import sys
import os
import json
import argparse
import numpy as np
import torch
import transformers
from dataclasses import dataclass
import time


def _gpu_mem_str(local_rank):
    """Résumé mémoire GPU courant (alloué/réservé/total) pour vérifier la charge réelle."""
    if not torch.cuda.is_available() or local_rank < 0:
        return "cpu"
    alloc = torch.cuda.memory_allocated(local_rank) / 1024**3
    reserved = torch.cuda.memory_reserved(local_rank) / 1024**3
    total = torch.cuda.get_device_properties(local_rank).total_memory / 1024**3
    return f"{alloc:.1f}GiB alloc / {reserved:.1f}GiB reserved / {total:.1f}GiB total"
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
import dllm
from sklearn.metrics import roc_auc_score, average_precision_score, accuracy_score
from PipelineTest.eval_configs import CONFIGS
from PipelineTest.Benchmark.data_split import (
    load_eval_data,
    get_train_test_split,
    get_train_test_split_for_config,
)
from PipelineTest.Benchmark.dist_utils import init_distributed, cleanup_distributed, gather_shards

# =========================================================================
# CONFIGURATIONS DES SAMPLERS
# =========================================================================
@dataclass
class SamplerConfig(dllm.core.samplers.MDLMSamplerConfig):
    steps: int = 16
    max_new_tokens: int = 32
    block_size: int = 32
    temperature: float = 0.5
    remasking: str = "low_confidence"

@dataclass
class DreamSamplerConfig(dllm.pipelines.dream.DreamSamplerConfig):
    steps: int = 16
    max_new_tokens: int = 32
    alg: str = "maskgit_plus"
    alg_temp: float = 0.0
    top_p: float = 1.0
    temperature: float = 0.5

@dataclass
class ScriptArguments:
    model_name_or_path: str = "GSAI-ML/LLaDA-8B-Instruct"
    seed: int = 42
    visualize: bool = False


# Registre model_type -> chemin du checkpoint, classe sampler, factory de SamplerConfig.
# Le chargement des poids ne dépend QUE de model_type (llada/dream), jamais de
# steps/max_new_tokens/block_size : ces derniers ne sont que des paramètres passés à
# sampler.sample(...), pas des propriétés du modèle chargé.
MODEL_REGISTRY = {
    "llada": {
        "model_path": "GSAI-ML/LLaDA-8B-Instruct",
        "sampler_cls": dllm.core.samplers.MDLMSampler,
        "make_sampler_config": lambda steps, tokens: SamplerConfig(
            steps=steps, max_new_tokens=tokens, temperature=0.5, block_size=tokens
        ),
    },
    "dream": {
        "model_path": "Dream-org/Dream-v0-Instruct-7B",
        "sampler_cls": dllm.pipelines.dream.sampler.DreamSampler,
        "make_sampler_config": lambda steps, tokens: DreamSamplerConfig(
            steps=steps, max_new_tokens=tokens, temperature=0.5
        ),
    },
}


def get_model_type(config_name: str) -> str:
    return "dream" if "dream" in config_name.lower() else "llada"


# =========================================================================
# DISTRIBUTED HELPERS
# =========================================================================
# La coordination inter-GPU passe par PipelineTest/Benchmark/dist_utils.py
# (rendez-vous sur fichiers, aucun collective NCCL) : le seul echange dont ce
# pipeline a besoin est la fusion des shards sur le rang 0, et la faire en NCCL
# rendait tout le run dependant d'un communicateur cree seulement apres des
# dizaines de minutes de generation (watchdog timeout -> SIGABRT sur tous les
# rangs, resultats deja calcules perdus).
_init_distributed = init_distributed
_cleanup_distributed = cleanup_distributed


# =========================================================================
# CHARGEMENT DES MODELES — fait UNE SEULE FOIS par model_type, et UNE SEULE
# FOIS pour NLI sur tout le run (au lieu d'une fois par config, x24).
# =========================================================================
def load_generation_backend(model_type: str, local_rank: int, rank: int):
    """Charge model + tokenizer + sampler pour un model_type donné.
    A appeler une seule fois par model_type (pas une fois par config)."""
    model_type = model_type.lower()
    if model_type not in MODEL_REGISTRY:
        raise ValueError("model_type doit être 'llada' ou 'dream'")

    entry = MODEL_REGISTRY[model_type]
    resolved_path = dllm.utils.resolve_with_base_env(entry["model_path"], "BASE_MODELS_DIR")
    print(f"[rank {rank}] Loading generation model ({model_type}): {resolved_path}...", flush=True)
    model = dllm.utils.get_model(
        model_name_or_path=resolved_path,
        device_map={"": local_rank} if torch.cuda.is_available() else None,
    ).eval()
    tokenizer = dllm.utils.get_tokenizer(model_name_or_path=resolved_path)
    sampler = entry["sampler_cls"](model=model, tokenizer=tokenizer)
    if torch.cuda.is_available():
        dev_name = torch.cuda.get_device_name(local_rank)
        print(f"[rank {rank}] Generation model ({model_type}) resident on cuda:{local_rank} "
              f"({dev_name}) | {_gpu_mem_str(local_rank)}", flush=True)
    return model, tokenizer, sampler, entry["make_sampler_config"]


def unload_generation_backend(model, sampler):
    del model, sampler
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def _build_nli_pipeline(local_rank):
    """Initialise le pipeline NLI DeBERTa-v2 sur le GPU ciblé via local_rank.
    Ne dépend d'aucun paramètre de config : à charger UNE SEULE FOIS pour tout le run."""
    if torch.cuda.is_available() and local_rank != -1:
        device_str = f"cuda:{local_rank}"
        pipeline_device = local_rank
        dtype = torch.float16
    else:
        device_str = "cpu"
        pipeline_device = -1
        dtype = torch.float32

    model_id = "microsoft/deberta-v2-xlarge-mnli"

    tok = transformers.AutoTokenizer.from_pretrained(model_id)
    model = transformers.AutoModelForSequenceClassification.from_pretrained(
        model_id, torch_dtype=dtype
    )

    if device_str != "cpu":
        print(f"[rank {local_rank}] Moving DeBERTa weights to {device_str}...", flush=True)
        model = model.to(device_str)

    pipeline_nli = transformers.pipeline(
        "text-classification",
        model=model,
        tokenizer=tok,
        device=pipeline_device,
        truncation=True,
        max_length=256,
    )
    print(f"[rank {local_rank}] NLI pipeline (DeBERTa) resident on {device_str} | {_gpu_mem_str(local_rank)}",
          flush=True)
    return pipeline_nli


# =========================================================================
# GENERATION DES VARIANTES — batchée sur (prompts x variantes) dans le MEME
# batch, au lieu de boucler N fois sur des petits batches (cf. discussion :
# le coût d'un sampler de diffusion dépend surtout de `steps`, pas du batch
# tant qu'on n'est pas memory-bound => on amortit `steps` sur un batch plus
# large plutôt que de le repayer N fois).
# =========================================================================
def generate_variants(tokenizer, sampler, sampler_config, messages: list, N: int, gen_batch_size: int = 8,
                       rank: int = 0, log_every: int = 10, keys: list = None):
    """Génère N variantes stochastiques pour chaque message.

    `keys` donne la clé de sortie de chaque message (par défaut sa position dans
    `messages`). On indexe par position globale et non par texte de question :
    deux samples distincts peuvent partager exactement la même question, et un
    dict keyé par texte fusionnait alors leurs variantes.
    """
    if keys is None:
        keys = list(range(len(messages)))
    assert len(keys) == len(messages)
    results = {k: [] for k in keys}

    expanded_messages = []
    expanded_keys = []
    for key, msg in zip(keys, messages):
        for _ in range(N):
            expanded_messages.append(msg)
            expanded_keys.append(key)

    n_batches = (len(expanded_messages) + gen_batch_size - 1) // max(gen_batch_size, 1)
    t_start = time.time()
    print(f"[rank {rank}] generate_variants: {len(expanded_messages)} items "
          f"({len(messages)} prompts x N={N}) in {n_batches} batches of {gen_batch_size}", flush=True)

    with torch.inference_mode():
        for batch_idx, start_idx in enumerate(range(0, len(expanded_messages), gen_batch_size)):
            end_idx = min(start_idx + gen_batch_size, len(expanded_messages))
            batch_messages = expanded_messages[start_idx:end_idx]
            batch_keys = expanded_keys[start_idx:end_idx]

            inputs = tokenizer.apply_chat_template(batch_messages, add_generation_prompt=True, tokenize=True)
            outputs = sampler.sample(inputs, sampler_config, return_dict=True)
            flat_sequences = dllm.utils.sample_trim(tokenizer, outputs.sequences.tolist(), inputs)

            for key, seq in zip(batch_keys, flat_sequences):
                results[key].append(seq)

            # Le sampler MDLM passe les logits en float64 pour le bruit de Gumbel
            # (cf. dllm/core/samplers/utils.py::add_gumbel_noise) : sur un vocab
            # ~126k, chaque batch alloue plusieurs tenseurs temporaires de plusieurs
            # GiB. Libérer le cache de l'allocateur après chaque batch évite que la
            # fragmentation ne s'accumule sur les ~150+ batches d'une config et ne
            # finisse par déclencher un OOM même quand la VRAM "logique" est dispo.
            del outputs, flat_sequences
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            if (batch_idx + 1) % log_every == 0 or (batch_idx + 1) == n_batches:
                elapsed = time.time() - t_start
                rate = (batch_idx + 1) / elapsed if elapsed > 0 else 0.0
                eta = (n_batches - batch_idx - 1) / rate if rate > 0 else float("inf")
                print(f"  [rank {rank}] gen batch {batch_idx + 1}/{n_batches} "
                      f"({elapsed:.1f}s elapsed, {rate:.2f} batch/s, ETA {eta:.0f}s) | "
                      f"{_gpu_mem_str(sampler.model.device.index if torch.cuda.is_available() else -1)}",
                      flush=True)

    print(f"[rank {rank}] generate_variants done in {time.time() - t_start:.1f}s", flush=True)
    return results


# =========================================================================
# NLI CLUSTERING / ENTROPIE SEMANTIQUE (inchangé sur le fond)
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


def calculate_semantic_entropy_nli(dict_results, pipeline_nli, nli_batch_size=64, rank=0):
    detailed_results = {}
    total_questions = len(dict_results)

    for q_idx, (key, variantes) in enumerate(dict_results.items(), 1):
        N = len(variantes)
        if N == 0:
            detailed_results[key] = {"variants": [], "clusters": [], "semantic_entropy": 0.0}
            continue

        clusters_indices = _partition_into_nli_clusters_batched(variantes, pipeline_nli, batch_size=nli_batch_size)
        clusters_text = [[variantes[idx] for idx in cluster] for cluster in clusters_indices]

        probabilities = [len(c) / N for c in clusters_indices]
        shannon_entropy = -sum(p * np.log(p + 1e-12) for p in probabilities)

        detailed_results[key] = {
            "variants": variantes,
            "clusters": clusters_text,
            "semantic_entropy": float(shannon_entropy)
        }

        if q_idx % 20 == 0 or q_idx == total_questions:
            print(f"  [rank {rank}] -> Avancement NLI : {q_idx}/{total_questions} questions filtrées.", flush=True)

    return detailed_results


# =========================================================================
# GENERATION + NLI POUR UNE CONFIG — modèle et pipeline NLI déjà chargés,
# passés en argument. Cette fonction ne fait plus AUCUN chargement/
# déchargement de poids (c'était le principal bottleneck).
# =========================================================================
def generate_and_evaluate_nli_parallel(tokenizer, sampler, sampler_config, nli_pipeline,
                                        messages: list, N: int, rank: int, world_size: int,
                                        local_rank: int, seed: int = 42, gen_batch_size: int = 8,
                                        nli_batch_size: int = 64, gather_tag: str = "default"):
    """Génère N variantes stochastiques ET calcule l'entropie sémantique par NLI."""

    if seed is not None:
        transformers.set_seed(seed + rank)

    shard_indices = list(range(rank, len(messages), world_size))
    shard_messages = [messages[i] for i in shard_indices]
    t_shard_start = time.time()
    print(f"[rank {rank}/{world_size}] Assigned {len(shard_messages)} prompts for Gen + NLI "
          f"(t={t_shard_start:.0f})", flush=True)

    # 1. Génération des variantes (batchée sur prompts x variantes)
    shard_results = generate_variants(tokenizer, sampler, sampler_config, shard_messages, N,
                                       gen_batch_size=gen_batch_size, rank=rank,
                                       keys=shard_indices)
    print(f"[rank {rank}] Gen phase done at t={time.time():.0f} "
          f"(+{time.time() - t_shard_start:.1f}s since shard start)", flush=True)

    # 2. Calcul de l'entropie sémantique en local sur le shard
    shard_detailed = calculate_semantic_entropy_nli(shard_results, nli_pipeline, nli_batch_size=nli_batch_size, rank=rank)
    print(f"[rank {rank}] NLI phase done at t={time.time():.0f} "
          f"(+{time.time() - t_shard_start:.1f}s since shard start)", flush=True)

    # 3. Fusion inter-GPU par rendez-vous sur fichiers (cf. dist_utils) : pas de
    #    collective NCCL, donc pas de watchdog timeout possible sur cette étape.
    grouped_detailed = gather_shards(
        shard_detailed, rank, world_size, tag=f"semantic_entropy_{gather_tag}", key_type=int
    )
    if grouped_detailed is None:
        return None

    missing = [i for i in range(len(messages)) if i not in grouped_detailed]
    if missing:
        raise RuntimeError(
            f"[rank 0] Fusion incomplete : {len(missing)} prompts sans résultat "
            f"(ex. positions {missing[:10]}). Les shards des rangs ne couvrent pas "
            f"tout le pool."
        )
    return grouped_detailed


# =========================================================================
# METRIQUES STANDARDISÉES
# =========================================================================
def evaluate(scores, labels, name="SemanticEntropy"):
    valid = ~np.isnan(scores)
    scores = scores[valid]
    labels = labels[valid]

    roc_auc = roc_auc_score(labels, scores)
    pr_auc = average_precision_score(labels, scores)

    # Balayage de seuils vectorisé (au lieu de 201 appels sklearn en boucle Python)
    thresholds = np.linspace(scores.min(), scores.max(), 201)
    preds = (scores[:, None] >= thresholds[None, :]).astype(int)  # (n_samples, 201)
    accs = (preds == labels[:, None]).mean(axis=0)
    best_acc = float(accs.max())

    y_pred = (scores >= np.median(scores)).astype(int)
    acc = accuracy_score(labels, y_pred)

    return {
        "name": name,
        "test_roc_auc": float(roc_auc),
        "test_pr_auc": float(pr_auc),
        "test_accuracy": float(acc),
        "test_best_accuracy": best_acc,
        "n_samples": len(labels),
        "n_halluc": int(np.sum(labels == 1)),
        "n_correct": int(np.sum(labels == 0)),
    }


# =========================================================================
# EXECUTION PAR CONFIG — modèle + NLI déjà chargés, passés en argument.
# =========================================================================
def run_config(config_name, tokenizer, sampler, make_sampler_config, nli_pipeline,
                rank, world_size, local_rank,
                n_variants=10, balance_seed=42, seed=42, gen_batch_size=8, nli_batch_size=64):
    cfg = CONFIGS[config_name]
    eval_json = cfg["eval_json"]

    if rank == 0:
        print(f"\n{'='*70}")
        print(f"  SEMANTIC ENTROPY NLI PARALLEL: {cfg['name']} ({config_name})")
        print(f"{'='*70}")

    if not os.path.exists(eval_json):
        if rank == 0:
            print(f"  [SKIP] eval_json introuvable: {eval_json}")
        return None

    labels_raw, indices_raw, n_missing_label, idx_to_prompt = load_eval_data(eval_json)
    if len(labels_raw) == 0:
        return None

    labels_balanced = labels_raw
    indices_balanced = indices_raw

    train_idx, test_idx = get_train_test_split_for_config(
        config_name, indices_balanced, sync_csv=(rank == 0)
    )

    balanced_prompts = [[{"role": "user", "content": idx_to_prompt[int(idx)]}] for idx in indices_balanced]

    import re
    m = re.search(r'(\d+)steps_(\d+)tokens', config_name)
    steps = int(m.group(1)) if m else 16
    tokens = int(m.group(2)) if m else 32

    sampler_config = make_sampler_config(steps, tokens)

    dict_detailed = generate_and_evaluate_nli_parallel(
        tokenizer=tokenizer,
        sampler=sampler,
        sampler_config=sampler_config,
        nli_pipeline=nli_pipeline,
        messages=balanced_prompts,
        N=n_variants,
        rank=rank,
        world_size=world_size,
        local_rank=local_rank,
        seed=seed,
        gen_batch_size=gen_batch_size,
        nli_batch_size=nli_batch_size,
        gather_tag=config_name,
    )

    if rank != 0:
        return None

    scores_vector = np.array([dict_detailed[pos]["semantic_entropy"] for pos in range(len(indices_balanced))])

    test_scores = scores_vector[test_idx]
    test_labels = labels_balanced[test_idx]

    results = evaluate(test_scores, test_labels, name=f"SemanticEntropy_{config_name}")

    individual_samples_log = {}
    for i, idx in enumerate(indices_balanced):
        q_text = idx_to_prompt[int(idx)]
        info = dict_detailed[i]

        individual_samples_log[str(int(idx))] = {
            "question": q_text,
            "label_hallucination": int(labels_balanced[i]),
            "variants": info["variants"],
            "clusters": info["clusters"],
            "semantic_entropy": info["semantic_entropy"],
            "is_in_test_split": bool(i in test_idx)
        }

    results.update({
        "config": config_name,
        "n_collapse_dropped": 0,
        "n_balanced_pool": len(labels_balanced),
        "samples": individual_samples_log
    })

    print(f"  ROC-AUC: {results['test_roc_auc']:.4f}  |  PR-AUC: {results['test_pr_auc']:.4f}")
    return results


def parse_args():
    parser = argparse.ArgumentParser(description="Semantic Entropy Multi-GPU Parallelized pipeline (Gen + NLI).")
    parser.add_argument("--config", type=str, default="all", choices=list(CONFIGS.keys()) + ["all"])
    parser.add_argument("--n_variants", type=int, default=5)
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument("--balance_seed", type=int, default=42)
    parser.add_argument("--seed", type=int, default=42,
                         help="Seed used for stochastic generation (transformers.set_seed), "
                              "to make benchmark runs reproducible.")
    parser.add_argument("--gen_batch_size", type=int, default=16,
                         help="Taille totale de batch pour la génération (prompts x variantes confondus). "
                              "Le sampler MDLM passe les logits en float64 pour le bruit de Gumbel : sur un "
                              "vocab ~126k, chaque unité de batch coûte plusieurs centaines de Mo de tenseurs "
                              "temporaires. A augmenter prudemment tant que la VRAM le permet.")
    parser.add_argument("--nli_batch_size", type=int, default=32,
                         help="Taille de batch pour DeBERTa NLI.")
    return parser.parse_args()


def main():
    args = parse_args()
    rank, world_size, local_rank = _init_distributed()

    output_dir = args.output_dir or os.path.abspath(os.path.join(os.path.dirname(__file__), "eval"))
    if rank == 0:
        os.makedirs(output_dir, exist_ok=True)

    configs_to_run = list(CONFIGS.keys()) if args.config == "all" else [args.config]

    # Regroupe les configs par model_type pour ne charger chaque modèle génératif
    # qu'UNE SEULE FOIS (au lieu d'une fois par config, x12). L'ordre relatif au
    # sein de chaque groupe (steps/tokens/dataset) est préservé (tri stable).
    configs_to_run = sorted(configs_to_run, key=get_model_type)

    # Le pipeline NLI ne dépend d'aucun paramètre de config : chargé une seule fois
    # pour tout le run (au lieu d'une fois par config, x24). Reste résident en VRAM
    # à côté du modèle génératif courant — DeBERTa-xlarge est petit (~1.8GB en fp16)
    # comparé aux modèles 7-8B, donc ça devrait passer sur la plupart des GPU. Si la
    # VRAM est vraiment juste, on peut la recharger à chaque changement de
    # model_type au lieu d'une seule fois pour tout le run.
    print(f"[rank {rank}] Loading NLI pipeline (DeBERTa) once for the whole run...", flush=True)
    nli_pipeline = _build_nli_pipeline(local_rank)

    all_results = {}
    current_model_type = None
    model = tokenizer = sampler = make_sampler_config = None

    for config_name in configs_to_run:
        model_type = get_model_type(config_name)

        if model_type != current_model_type:
            if model is not None:
                unload_generation_backend(model, sampler)
            model, tokenizer, sampler, make_sampler_config = load_generation_backend(model_type, local_rank, rank)
            current_model_type = model_type

        t0 = time.time()
        result = run_config(
            config_name, tokenizer, sampler, make_sampler_config, nli_pipeline,
            rank, world_size, local_rank,
            n_variants=args.n_variants, balance_seed=args.balance_seed, seed=args.seed,
            gen_batch_size=args.gen_batch_size, nli_batch_size=args.nli_batch_size,
        )
        t1 = time.time()
        if result is not None:
            result['time'] = t1 - t0
        print(f"  [INFO] Config '{config_name}' completed in {t1 - t0:.2f} seconds.")
        if rank == 0 and result is not None:
            all_results[config_name] = result

    if model is not None:
        unload_generation_backend(model, sampler)
    del nli_pipeline
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    if rank == 0 and all_results:
        print(f"\n\n{'='*70}\n  SUMMARY TABLE : SEMANTIC ENTROPY (NLI INTERN PARALLEL)\n{'='*70}")
        for name, r in all_results.items():
            print(f"  {name:<45} ROC-AUC: {r['test_roc_auc']:.4f} | PR-AUC: {r['test_pr_auc']:.4f}")

        suffix = CONFIGS[configs_to_run[0]]["name"] if len(configs_to_run) == 1 else "all"
        out_path = os.path.join(output_dir, f"semantic_entropy_results_{suffix}_debertav1.json")
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(all_results, f, indent=2)
        print(f"\n[SUCCESS] Saved to: {out_path}")

    _cleanup_distributed(world_size)


if __name__ == "__main__":
    main()