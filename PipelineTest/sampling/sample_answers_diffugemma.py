import sys
import os
import time
import argparse
import random


# Add both the sampling dir (for load_data, metric_qwen, AnalyseResults) and project root (for PipelineTest, dllm)
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

import gc
import torch
import json
import transformers
import torch.distributed as dist
import dllm
from dataclasses import dataclass, fields as dc_fields
from load_data import load_triviaqa, load_naturalquestion, load_hotpotqa
from dllm.pipelines.diffusiongemma.sampler import DiffusionGemmaSamplerWithCompleteHistory, DiffusionGemmaSamplerConfig


DATASET_LOADERS = {
    "triviaqa": load_triviaqa,
    "naturalquestion": load_naturalquestion,
    "hotpotqa": load_hotpotqa,
}


@dataclass
class SamplerConfig(dllm.core.samplers.MDLMSamplerConfig):
    steps: int = 16
    max_new_tokens: int = 32
    block_size: int = 32
    temperature: float = 0.0
    remasking: str = "low_confidence"


@dataclass
class DreamSamplerConfig(dllm.pipelines.dream.DreamSamplerConfig):
    steps: int = 16
    max_new_tokens: int = 32
    alg: str = "maskgit_plus"
    alg_temp: float = 0.0
    top_p: float = 1.0
    temperature: float = 0.0


@dataclass
class GemmaSamplerConfig(DiffusionGemmaSamplerConfig):
    max_new_tokens: int = 64
    steps: int = 64
    max_temperature: float = None
    min_temperature: float = None
    return_dict: bool = True
    canvas_length: int = 64


@dataclass
class ScriptArguments:
    model_name_or_path: str = "GSAI-ML/LLaDA-8B-Instruct"
    seed: int = 42
    visualize: bool = False

    def __post_init__(self):
        self.model_name_or_path = dllm.utils.resolve_with_base_env(
            self.model_name_or_path, "BASE_MODELS_DIR"
        )


MODEL_FOR_SAMPLER = {
  
    "diffgemma": "google/diffusiongemma-26B-A4B-it",
}

SAMPLER_CLS_FOR_NAME = {
    # "llada": lambda: dllm.core.samplers.MDLMSamplerWithCompleteHistory,
    # "dream": lambda: dllm.pipelines.dream.sampler.DreamSamplerWithCompleteHistory,
    "diffgemma": lambda: DiffusionGemmaSamplerWithCompleteHistory,
}

CONFIG_CLS_FOR_NAME = {
    # "llada": SamplerConfig,
    # "dream": DreamSamplerConfig,
    "diffgemma": GemmaSamplerConfig,
}

# --- Les configurations à lancer ---
DATASETS = ["hotpotqa", "naturalquestion", "triviaqa"]
SAMPLERS = ["diffgemma"]

# (steps, max_new_tokens) pour llada/dream (non utilisé tant que SAMPLERS ne
# contient que "diffgemma", conservé pour pouvoir réactiver le sweep llada/dream).
STEP_TOKEN_COMBOS = [
    (32, 32),
    (32, 64),
    (64, 64),
    
]

# Configurations diffgemma : (max_new_tokens, steps, canvas_length, brief).
# "brief"=True ajoute " please answer briefly" à la fin du prompt utilisateur ;
# seule la config 256 tokens / 128 steps garde le prompt tel quel.
GEMMA_CONFIGS = [
    # {"max_new_tokens": 32, "steps": 16, "canvas_length": 32, "brief": True},
    # {"max_new_tokens": 64, "steps": 32, "canvas_length": 64, "brief": True},
    # {"max_new_tokens": 32, "steps": 32, "canvas_length": 32, "brief": True},
    # {"max_new_tokens": 64, "steps": 64, "canvas_length": 64, "brief": True},
    {"max_new_tokens": 256, "steps": 128, "canvas_length": 256, "brief": False},
]

BRIEF_SUFFIX = " please answer briefly"


def _configs_for_sampler(sampler_name):
    """Liste de dicts {steps, max_new_tokens, canvas_length, brief} pour un sampler donné."""
    if sampler_name == "diffgemma":
        return GEMMA_CONFIGS
    return [
        {"steps": steps, "max_new_tokens": max_new_tokens, "canvas_length": None, "brief": False}
        for steps, max_new_tokens in STEP_TOKEN_COMBOS
    ]


def _init_distributed():
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))

    if world_size > 1 and not dist.is_initialized():
        if not torch.cuda.is_available():
            raise RuntimeError("Multi-GPU requires CUDA.")
        torch.cuda.set_device(local_rank)
        dist.init_process_group(backend="nccl")

    return rank, world_size, local_rank


def _cleanup_distributed(world_size: int):
    # Libère la mémoire GPU dans tous les cas, même en single-GPU
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()

    if world_size > 1 and dist.is_initialized():
        dist.destroy_process_group()


def _free_gpu_memory(rank, local_rank, tag=""):
    """
    Nettoyage mémoire GPU agressif : force le garbage collector Python
    AVANT de vider le cache CUDA (sinon des tenseurs encore référencés par
    des objets Python non collectés ne sont jamais rendus au cache CUDA).
    Affiche l'état mémoire avant/après pour diagnostiquer les fuites.
    """
    if not torch.cuda.is_available():
        return

    before = torch.cuda.memory_allocated(local_rank) / 1e9
    before_reserved = torch.cuda.memory_reserved(local_rank) / 1e9

    gc.collect()
    torch.cuda.synchronize(local_rank)
    torch.cuda.empty_cache()
    torch.cuda.ipc_collect()

    after = torch.cuda.memory_allocated(local_rank) / 1e9
    after_reserved = torch.cuda.memory_reserved(local_rank) / 1e9

    print(
        f"[rank {rank}] [mem{' ' + tag if tag else ''}] "
        f"allocated {before:.2f}GB -> {after:.2f}GB | "
        f"reserved {before_reserved:.2f}GB -> {after_reserved:.2f}GB",
        flush=True,
    )


def _build_config_cls(sampler_name, steps, max_new_tokens, canvas_length=None):
    """Construit dynamiquement une config CLI-overridée pour un sampler donné."""
    base_cls = CONFIG_CLS_FOR_NAME[sampler_name]
    overrides = {
        "__annotations__": {"steps": int, "max_new_tokens": int},
        "steps": steps,
        "max_new_tokens": max_new_tokens,
    }
    if sampler_name == "llada":
        overrides["__annotations__"]["block_size"] = int
        overrides["block_size"] = max_new_tokens
    elif sampler_name == "diffgemma":
        overrides["__annotations__"]["canvas_length"] = int
        overrides["canvas_length"] = canvas_length if canvas_length is not None else max_new_tokens
    return dataclass(type(f"CLI_{base_cls.__name__}", (base_cls,), overrides))


def _apply_brief_suffix(messages):
    """Ajoute BRIEF_SUFFIX à la fin de chaque message utilisateur."""
    new_messages = []
    for convo in messages:
        new_convo = [
            {**msg, "content": msg["content"] + BRIEF_SUFFIX} if msg.get("role") == "user" else msg
            for msg in convo
        ]
        new_messages.append(new_convo)
    return new_messages


def _move_output_to_cpu(outputs):
    """Move every tensor held by a BaseSamplerOutputCompleteHistory to CPU.

    Without this, `_run_batches_for_group` keeps every batch's full
    per-step histories (histories_x, histories_x0, histories_logprobs, ...)
    referenced on GPU for the whole shard, since they're only written to
    disk once at the very end of the shard loop -- GPU memory then grows
    linearly with the number of batches processed instead of with a single
    batch's size.
    """
    for f in dc_fields(outputs):
        value = getattr(outputs, f.name)
        if isinstance(value, torch.Tensor):
            setattr(outputs, f.name, value.cpu())
        elif isinstance(value, list) and value and isinstance(value[0], torch.Tensor):
            setattr(outputs, f.name, [t.cpu() for t in value])
    return outputs


def _run_batches_for_group(
    sampler_obj,
    sampler_config,
    tokenizer,
    group_indices,
    messages,
    labels,
    batch_size,
    rank,
    temp_label,
    sampler_name="llada",
):
    """
    Runs sampling over a subset of the (already-sharded) dataset using a
    single sampler_config (i.e. a fixed temperature regime).

    `group_indices` are positions into the shard-local `messages`/`labels`
    lists (NOT global dataset indices) that belong to this temperature group.
    """
    local_results = []
    local_outputs = []

    group_messages = [messages[i] for i in group_indices]
    group_labels = [labels[i] for i in group_indices]
    total_local = len(group_messages)
    num_local_batches = (total_local + batch_size - 1) // batch_size

    print(
        f"[rank {rank}] Assigned {total_local} samples over "
        f"{num_local_batches} batches (temperature={temp_label}).",
        flush=True,
    )

    for start_idx in range(0, len(group_messages), batch_size):
        batch_t0 = time.time()
        end_idx = min(start_idx + batch_size, len(group_messages))
        batch_messages = group_messages[start_idx:end_idx]
        batch_local_indices = group_indices[start_idx:end_idx]
        batch_labels = group_labels[start_idx:end_idx]
        batch_id = start_idx // batch_size + 1

        print(
            f"[rank {rank}] Batch {batch_id}/{num_local_batches} "
            f"| local range [{start_idx}:{end_idx})",
            flush=True,
        )

        if sampler_name == "diffgemma":
            # Le tokenizer diffgemma (issu de l'AutoProcessor) ne renvoie pas des
            # ids avec apply_chat_template(..., tokenize=True) en mode batch --
            # on formate en texte puis on tokenize nous-mêmes par échantillon,
            # comme dans sample_answers_without_collapse.py et dllm/pipelines/diffusiongemma/eval.py.
            formatted_prompts = [
                tokenizer.apply_chat_template(
                    msg, add_generation_prompt=True, tokenize=False,
                )
                for msg in batch_messages
            ]
            inputs = [
                tokenizer.encode(p, return_tensors="pt").squeeze(0)
                for p in formatted_prompts
            ]
        else:
            inputs = tokenizer.apply_chat_template(
                batch_messages,
                add_generation_prompt=True,
                tokenize=True,
            )

        outputs = sampler_obj.sample(inputs, sampler_config, return_dict=True)
        outputs.sample_indices = torch.tensor(batch_local_indices, dtype=torch.long)
        outputs = _move_output_to_cpu(outputs)

        batch_answers = dllm.utils.sample_trim(tokenizer, outputs.sequences.tolist(), inputs)
        for local_idx, label_entry, answer in zip(batch_local_indices, batch_labels, batch_answers):
            local_results.append({
                "question": label_entry["question"],
                "label": label_entry["label"],
                "answer": answer,
                "index": local_idx,  # shard-local for now; remapped below
                "temperature": temp_label,
            })

        local_outputs.append(outputs)

        torch.cuda.empty_cache()
        print(
            f"[rank {rank}] Batch {batch_id}/{num_local_batches} done in "
            f"{time.time() - batch_t0:.1f}s | cumulative={len(local_results)}/{total_local}",
            flush=True,
        )

    return local_results, local_outputs


def RunOneConfig(
    rank,
    world_size,
    local_rank,
    sampler_obj,
    tokenizer,
    sampler_config_cls,
    suffix,
    dataset,
    num_sample,
    batch_size,
    temp,
    data_seed,
    brief=False,
    sampler_name="llada",
):
    """
    Exécute UNE configuration (dataset + steps/tokens) avec un modèle/sampler
    déjà chargé en mémoire. Ne gère PAS l'init/cleanup distribué ni le
    chargement du modèle -- c'est la responsabilité de l'appelant, pour
    pouvoir réutiliser le même modèle sur plusieurs configs.
    """
    run_t0 = time.time()
    print(
        f"[rank {rank}/{world_size}] Starting run | dataset={dataset} | num_sample={num_sample} "
        f"| batch_size={batch_size} | suffix={suffix} | temp={temp}",
        flush=True,
    )

    config = sampler_config_cls()
    config.temperature = temp

    load_fn = DATASET_LOADERS[dataset]
    messages, labels = load_fn(num_samples=num_sample, seed=data_seed)
    if brief:
        messages = _apply_brief_suffix(messages)
    print(f"[rank {rank}] Loaded dataset with {len(messages)} samples (data_seed={data_seed}).", flush=True)


    # Shard work across ranks: rank r processes indices r, r+world_size, ...
    shard_indices = list(range(rank, len(messages), world_size))
    shard_messages = [messages[i] for i in shard_indices]
    shard_labels = [labels[i] for i in shard_indices]
    total_local = len(shard_messages)
    print(f"[rank {rank}] Assigned {total_local} samples (shard-local).", flush=True)

    shard_local_positions = [
        pos for pos, global_idx in enumerate(shard_indices)
    ]
    print("shard local positions", shard_local_positions)
    BatchResults, BatchOutputs = _run_batches_for_group(
        sampler_obj=sampler_obj,
        sampler_config=config,
        tokenizer=tokenizer,
        group_indices=shard_local_positions,
        messages=shard_messages,
        labels=shard_labels,
        batch_size=batch_size,
        rank=rank,
        temp_label=temp,
        sampler_name=sampler_name,
    )

    for r in BatchResults:
        r["index"] = shard_indices[r["index"]]


    for o in BatchOutputs:
        o.sample_indices = torch.tensor([shard_indices[i] for i in o.sample_indices.tolist()], dtype=torch.long)
    
    
    local_results = BatchResults
    all_outputs = BatchOutputs

    artifacts_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "res", f"results_{suffix}"))
    os.makedirs(artifacts_dir, exist_ok=True)

    outputs_shard_path = os.path.join(artifacts_dir, f"_outputs_rank{rank}.pt")
    torch.save(all_outputs, outputs_shard_path)
    del all_outputs
    print(f"[rank {rank}] Saved outputs shard to disk.", flush=True)

    torch.cuda.empty_cache()

    local_results_path = os.path.join(artifacts_dir, f"_results_rank{rank}.json")
    with open(local_results_path, "w", encoding="utf-8") as f:
        json.dump(local_results, f, indent=2, ensure_ascii=False)

    shard_path = os.path.join(artifacts_dir, f"_shard_rank{rank}.pt")
    torch.save({
        "indices": shard_indices,
        "results": local_results,
    }, shard_path)
    print(f"[rank {rank}] Saved shard to disk.", flush=True)

    os.remove(local_results_path)

    if world_size > 1:
        dist.barrier()

    if rank == 0:
        from PipelineTest.features.io import _pad_and_cat_tensors
        from dllm.core.samplers.base import BaseSamplerOutputCompleteHistory

        results = []
        ordered_batches = []

        for r in range(world_size):
            op = os.path.join(artifacts_dir, f"_outputs_rank{r}.pt")
            rank_outputs = torch.load(op, map_location="cpu", weights_only=False)
            p = os.path.join(artifacts_dir, f"_shard_rank{r}.pt")
            shard = torch.load(p, map_location="cpu", weights_only=False)

            for batch_output in rank_outputs:
                first_global_idx = int(batch_output.sample_indices[0].item())
                ordered_batches.append((first_global_idx, batch_output))

            results.extend(shard["results"])

            os.remove(op)
            os.remove(p)

        ordered_batches.sort(key=lambda x: x[0])
        outputs_list = [batch_output for _, batch_output in ordered_batches]

        merged = outputs_list[0]
        if len(outputs_list) > 1:
            merged = BaseSamplerOutputCompleteHistory()
            for f in dc_fields(BaseSamplerOutputCompleteHistory):
                name = f.name
                values = [getattr(o, name) for o in outputs_list]
                non_none_values = [v for v in values if v is not None]

                if len(non_none_values) == 0:
                    setattr(merged, name, None)
                    continue

                if name == "sample_indices":
                    setattr(merged, name, torch.cat(non_none_values, dim=0))
                    continue


                sample = non_none_values[0]

                if isinstance(sample, torch.Tensor):
                    setattr(merged, name, _pad_and_cat_tensors(non_none_values))
                    continue

                if isinstance(sample, list):
                    if len(sample) == 0:
                        setattr(merged, name, [])
                        continue
                    if all(isinstance(v, list) for v in non_none_values):
                        if all(len(v) == len(non_none_values[0]) for v in non_none_values):
                            if all(len(v) > 0 and isinstance(v[0], torch.Tensor) for v in non_none_values):
                                merged_history = []
                                for step_idx in range(len(non_none_values[0])):
                                    merged_history.append(
                                        _pad_and_cat_tensors([v[step_idx] for v in non_none_values])
                                    )
                                setattr(merged, name, merged_history)
                            else:
                                flat = []
                                for v in non_none_values:
                                    flat.extend(v)
                                setattr(merged, name, flat)
                        else:
                            flat = []
                            for v in non_none_values:
                                flat.extend(v)
                            setattr(merged, name, flat)
                        continue

                setattr(merged, name, sample)

        results.sort(key=lambda x: x["index"])

        torch.save(merged, os.path.join(artifacts_dir, f"outputs_{suffix}.pt"))
        print("sample indices", merged.sample_indices)
        torch.save(tokenizer, os.path.join(artifacts_dir, f"tokenizer_{suffix}.pt"))

        res_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "res", "to_eval_V2"))
        os.makedirs(res_dir, exist_ok=True)
        results_json_path = os.path.join(res_dir, f"results_{suffix}.json")
        with open(results_json_path, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2, ensure_ascii=False)

        print(f"[rank 0] Merged {merged.sequences.shape[0]} samples.", flush=True)
        print(f"[rank 0] Saved results_{dataset}_{suffix}.json ({len(results)} entries)", flush=True)
        print(f"[rank 0] All done in {time.time() - run_t0:.1f}s", flush=True)

        # Libère explicitement les gros objets côté rank 0 (merge de tous
        # les shards) avant de rendre la main -- sinon ils restent référencés
        # jusqu'au prochain appel de fonction et gonflent la RAM/VRAM pour
        # la config suivante.
        del merged, outputs_list, ordered_batches, results

    print(f"[rank {rank}] Finished config '{suffix}' in {time.time() - run_t0:.1f}s", flush=True)


def RunAllConfigs(num_sample, batch_size, data_seed):
    """
    Boucle sur les configurations (samplers x datasets x configs par sampler),
    en réutilisant le modèle chargé pour chaque sampler sur toutes ses configs,
    et en n'initialisant/détruisant le process group distribué qu'une seule
    fois pour tout le run.
    """
    rank, world_size, local_rank = _init_distributed()

    try:
        total_configs = sum(
            len(DATASETS) * len(_configs_for_sampler(sampler_name)) for sampler_name in SAMPLERS
        )
        config_num = 0

        for sampler_name in SAMPLERS:
            model = None
            sampler_obj = None
            try:
                script_args = ScriptArguments(model_name_or_path=MODEL_FOR_SAMPLER[sampler_name])
                if script_args.seed is not None:
                    transformers.set_seed(script_args.seed + rank)

                print(f"[rank {rank}] Loading model/tokenizer for sampler '{sampler_name}': "
                      f"{script_args.model_name_or_path}", flush=True)
                device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")

                if sampler_name == "diffgemma":
                    # DiffusionGemma (26B-A4B) ne tient pas sur un seul GPU : on
                    # répartit manuellement l'encodeur/décodeur sur les GPU 0 et 1,
                    # comme dans sample_answers_without_collapse.py.
                    from transformers import AutoProcessor, DiffusionGemmaForBlockDiffusion
                    model_path = MODEL_FOR_SAMPLER["diffgemma"]
                    processor = AutoProcessor.from_pretrained(model_path)
                    gemma_device_map = {
                        "model.encoder.language_model.embed_tokens": 0,
                        "model.decoder.embed_tokens": 0,
                        "model.encoder.vision_tower": 0,
                        "model.encoder.embed_vision": 0,
                        "model.decoder.self_conditioning": 0,
                        "lm_head": 0,
                        "model.encoder.language_model.norm": 1,
                        "model.decoder.norm": 1,
                    }
                    for i in range(30):
                        gpu = 0 if i < 15 else 1
                        gemma_device_map[f"model.encoder.language_model.layers.{i}"] = gpu
                        gemma_device_map[f"model.decoder.layers.{i}"] = gpu
                    model = DiffusionGemmaForBlockDiffusion.from_pretrained(
                        model_path, torch_dtype=torch.bfloat16, device_map=gemma_device_map,
                    ).eval()
                    tokenizer = processor.tokenizer
                else:
                    model = dllm.utils.get_model(
                        model_name_or_path=script_args.model_name_or_path,
                    ).to(device).eval()
                    tokenizer = dllm.utils.get_tokenizer(model_name_or_path=script_args.model_name_or_path)

                sampler_cls = SAMPLER_CLS_FOR_NAME[sampler_name]()
                sampler_obj = sampler_cls(model=model, tokenizer=tokenizer)

                for dataset in DATASETS:
                    for cfg in _configs_for_sampler(sampler_name):
                        steps = cfg["steps"]
                        max_new_tokens = cfg["max_new_tokens"]
                        canvas_length = cfg.get("canvas_length")
                        brief = cfg.get("brief", False)
                        config_num += 1
                        sampler_config_cls = _build_config_cls(
                            sampler_name, steps, max_new_tokens, canvas_length
                        )
                        suffix = (
                            f"{sampler_name}_{steps}steps_{max_new_tokens}tokens"
                            + (f"_{canvas_length}canvas" if sampler_name == "diffgemma" else "")
                            + f"_{dataset}_{num_sample}samples"
                            + ("_brief" if brief else "")
                            + f"_seed{data_seed}"
                        )
                        print(
                            f"\n[rank {rank}] === Config {config_num}/{total_configs}: "
                            f"sampler={sampler_name} dataset={dataset} steps={steps} "
                            f"max_new_tokens={max_new_tokens} canvas_length={canvas_length} "
                            f"brief={brief} ===",
                            flush=True,
                        )
                        RunOneConfig(
                            rank=rank,
                            world_size=world_size,
                            local_rank=local_rank,
                            sampler_obj=sampler_obj,
                            tokenizer=tokenizer,
                            sampler_config_cls=sampler_config_cls,
                            suffix=suffix,
                            dataset=dataset,
                            num_sample=num_sample,
                            batch_size=batch_size,
                            temp=0.0,
                            data_seed=data_seed,
                            brief=brief,
                            sampler_name=sampler_name,
                        )

                        # Nettoyage mémoire complet après CHAQUE config, pas
                        # seulement après le dernier passage sur ce sampler.
                        # Le modèle reste chargé, mais tout le reste (batches,
                        # historique de diffusion, tenseurs mergés côté rank 0,
                        # etc.) doit être relâché avant la config suivante.
                        _free_gpu_memory(rank, local_rank, tag=f"after config {config_num}/{total_configs}")

                        if world_size > 1:
                            # Barrier APRÈS le nettoyage mémoire : garantit que
                            # tous les ranks ont fini de libérer leur mémoire
                            # avant que quiconque ne démarre la config suivante,
                            # ce qui évite qu'un rank encore en train de faire
                            # empty_cache() OOM pendant que d'autres ont déjà
                            # commencé à charger le prochain batch.
                            dist.barrier()

            finally:
                # Libère le modèle du sampler courant avant de charger le suivant
                try:
                    del sampler_obj
                except NameError:
                    pass
                try:
                    del model
                except NameError:
                    pass
                _free_gpu_memory(rank, local_rank, tag=f"after freeing model '{sampler_name}'")
                if world_size > 1:
                    dist.barrier()
                print(f"[rank {rank}] Freed model for sampler '{sampler_name}' from GPU.", flush=True)

        print(f"[rank {rank}] All {total_configs} configs finished.", flush=True)

    finally:
        _cleanup_distributed(world_size)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Sample answers for all (sampler x dataset x steps/tokens) configurations"
    )
    parser.add_argument("--num_sample", type=int, default=2500, help="Number of samples per dataset")
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size")
    parser.add_argument("--data_seed", type=int, default=42,
                        help="Seed for dataset shuffling/selection (default: 42)")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    RunAllConfigs(
        num_sample=args.num_sample,
        batch_size=args.batch_size,
        data_seed=args.data_seed,
    )

    os._exit(0)