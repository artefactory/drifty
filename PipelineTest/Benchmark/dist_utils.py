"""
Coordination inter-GPU des benchmarks (Benchmark/*.py), sans collective NCCL.

Contexte
--------
Les benchmarks GPU (lexical_similarity, semantic_entropy) sont "embarrassingly
parallel" : chaque rang traite son propre shard de prompts, et le seul echange
reellement necessaire est la fusion finale des shards sur le rang 0.

Ces fusions passaient par des collectives NCCL (`dist.barrier`,
`dist.all_gather_object`). Le premier collective d'un process n'etant emis
qu'apres des dizaines de minutes de generation, tout le run dependait de la
sante du communicateur NCCL a ce moment-la — et un blocage a cet endroit tue
l'ensemble des rangs apres 600 s de watchdog :

    [Rank 0] Watchdog caught collective operation timeout:
    WorkNCCL(SeqNum=1, OpType=ALLREDUCE, NumelIn=1, ...) ran for 600001 ms
    ...
    To avoid data inconsistency, we are taking the entire process down.

... alors meme que les resultats etaient deja calcules (le rang 0 a eu le temps
d'imprimer "Merged 2475 generated vectors" avant d'etre abattu par SIGABRT).

Strategie
---------
On supprime completement le process group : plus aucun collective, donc plus
aucun point de blocage NCCL. Le sharding se deduit des variables d'environnement
(`RANK`/`WORLD_SIZE`/`LOCAL_RANK`) posees par torchrun, et la fusion se fait par
rendez-vous sur fichiers :

  * chaque rang ecrit son shard de facon **atomique** (fichier temporaire +
    `os.replace`), donc un lecteur ne voit jamais un JSON tronque ;
  * le repertoire de rendez-vous est **unique par lancement torchrun**
    (`TORCHELASTIC_RUN_ID`, a defaut `MASTER_PORT`), donc un shard d'un run
    precedent (crashe) ne peut pas etre relu par erreur ;
  * le rang 0 attend les `world_size` fichiers avec un timeout explicite et un
    message d'erreur nommant les rangs manquants, puis nettoie le repertoire.

Les autres rangs n'attendent rien : ils ecrivent et continuent. Si un rang
meurt, torchrun abat de toute facon les autres.

Variables d'environnement :
  DLLM_SHARD_DIR      repertoire de rendez-vous (defaut: $TMPDIR/dllm_benchmark_shards)
  DLLM_GATHER_TIMEOUT timeout (s) d'attente des shards sur le rang 0 (defaut: 7200)
"""

import json
import os
import re
import shutil
import tempfile
import time

import torch
import torch.distributed as dist

DEFAULT_GATHER_TIMEOUT_S = float(os.environ.get("DLLM_GATHER_TIMEOUT", "7200"))
_STALE_RUN_MAX_AGE_S = 24 * 3600
# Tous les rangs d'un meme lancement demarrent ensemble : un fichier de shard
# anterieur a cet instant (moins une marge) vient forcement d'un run precedent.
_PROCESS_START_S = time.time()
_STALE_FILE_SLACK_S = 300.0


# =========================================================================
# RANG / DEVICE
# =========================================================================
def init_distributed():
    """Retourne (rank, world_size, local_rank) et fixe le device CUDA du rang.

    N'initialise **aucun** process group : toute la coordination passe par
    `gather_shards` (fichiers). Voir le docstring du module.
    """
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))

    if world_size > 1 and not torch.cuda.is_available():
        raise RuntimeError("Multi-GPU (WORLD_SIZE > 1) requires CUDA.")
    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank)

    return rank, world_size, local_rank


def cleanup_distributed(world_size=1):
    """Detruit un eventuel process group (aucun n'est cree par ce module)."""
    if dist.is_available() and dist.is_initialized():
        dist.destroy_process_group()


# =========================================================================
# REPERTOIRE DE RENDEZ-VOUS
# =========================================================================
def _sanitize(name):
    return re.sub(r"[^A-Za-z0-9_.-]", "_", str(name))[:120] or "unnamed"


def _run_id():
    """Identifiant partage par tous les rangs d'un meme lancement torchrun.

    `TORCHELASTIC_RUN_ID` vaut litteralement "none" quand torchrun est lance sans
    `--rdzv-id` (le cas ici), donc il ne suffit pas a distinguer deux runs. On y
    ajoute le port maitre (tire au hasard par Benchmark/main.py, et de toute
    facon exclusif entre deux runs simultanes) et le PID de l'agent torchrun,
    identique pour tous les workers locaux d'un meme lancement et unique dans le
    temps.
    """
    parts = []
    run_id = os.environ.get("TORCHELASTIC_RUN_ID", "")
    if run_id and run_id.lower() != "none":
        parts.append(run_id)
    port = os.environ.get("MASTER_PORT")
    if port:
        parts.append(port)
    if int(os.environ.get("WORLD_SIZE", "1")) > 1:
        parts.append(str(os.getppid()))  # PID de l'agent torchrun (partage)
    return _sanitize("_".join(parts)) if parts else "single"


def _base_dir():
    return os.environ.get(
        "DLLM_SHARD_DIR", os.path.join(tempfile.gettempdir(), "dllm_benchmark_shards")
    )


def _rendezvous_dir(tag):
    return os.path.join(_base_dir(), f"run_{_run_id()}", _sanitize(tag))


def _prune_stale_runs():
    """Supprime les repertoires de runs anterieurs (crashes) de plus de 24 h."""
    base = _base_dir()
    if not os.path.isdir(base):
        return
    now = time.time()
    for entry in os.listdir(base):
        path = os.path.join(base, entry)
        try:
            if os.path.isdir(path) and now - os.path.getmtime(path) > _STALE_RUN_MAX_AGE_S:
                shutil.rmtree(path, ignore_errors=True)
        except OSError:
            pass


# =========================================================================
# FUSION DES SHARDS
# =========================================================================
def _write_atomic(path, payload):
    tmp = f"{path}.tmp.{os.getpid()}"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(payload, f)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


def _is_fresh(path):
    """Ecarte un fichier laisse par un run precedent (meme repertoire reutilise)."""
    try:
        return os.path.getmtime(path) >= _PROCESS_START_S - _STALE_FILE_SLACK_S
    except OSError:
        return False


def _read_shard(path, expected_rank, expected_run_id, expected_tag):
    with open(path, "r", encoding="utf-8") as f:
        payload = json.load(f)
    if (payload.get("rank") != expected_rank
            or payload.get("run_id") != expected_run_id
            or payload.get("tag") != expected_tag):
        raise RuntimeError(
            f"Shard incoherent dans {path}: attendu "
            f"(rank={expected_rank}, run_id={expected_run_id}, tag={expected_tag}), "
            f"trouve (rank={payload.get('rank')}, run_id={payload.get('run_id')}, "
            f"tag={payload.get('tag')}). Supprimez le repertoire de rendez-vous et relancez."
        )
    return payload["data"]


def gather_shards(shard, rank, world_size, tag, key_type=str,
                  timeout_s=None, poll_s=2.0, log_every_s=60.0):
    """Fusionne le dict-shard de chaque rang en un seul dict sur le rang 0.

    Remplace `dist.all_gather_object` / `dist.barrier` : aucun collective NCCL,
    donc aucun risque de watchdog timeout (cf. docstring du module).

    Args:
        shard: dict {cle: valeur} JSON-serialisable produit par CE rang.
        rank, world_size: identite du rang (cf. `init_distributed`).
        tag: identifiant du point de fusion (nom du benchmark + config), pour
            que deux fusions successives du meme run n'utilisent pas le meme
            repertoire.
        key_type: type des cles a restaurer apres le passage par JSON
            (`int` si le shard est indexe par position, `str` sinon).
        timeout_s: attente max des shards sur le rang 0 (defaut:
            $DLLM_GATHER_TIMEOUT, 7200 s).

    Returns:
        Le dict fusionne sur le rang 0, `None` sur les autres rangs
        (comportement identique a l'ancien code).
    """
    if world_size <= 1:
        return dict(shard)

    timeout_s = DEFAULT_GATHER_TIMEOUT_S if timeout_s is None else timeout_s
    run_id = _run_id()
    directory = _rendezvous_dir(tag)
    os.makedirs(directory, exist_ok=True)

    shard_path = os.path.join(directory, f"shard_{rank}.json")
    _write_atomic(shard_path, {
        "run_id": run_id,
        "tag": str(tag),
        "rank": rank,
        "world_size": world_size,
        "n": len(shard),
        "data": {str(k): v for k, v in shard.items()},
    })
    print(f"[rank {rank}] Shard '{tag}' ecrit ({len(shard)} entrees) -> {shard_path}", flush=True)

    # Les rangs != 0 n'attendent personne : ils ont fini leur part du travail.
    if rank != 0:
        return None

    _prune_stale_runs()

    paths = {r: os.path.join(directory, f"shard_{r}.json") for r in range(world_size)}
    deadline = time.time() + timeout_s
    last_log = time.time()
    while True:
        missing = [r for r, p in paths.items() if not (os.path.exists(p) and _is_fresh(p))]
        if not missing:
            break
        now = time.time()
        if now > deadline:
            raise RuntimeError(
                f"[rank 0] Timeout ({timeout_s:.0f}s) en attendant les shards '{tag}' "
                f"des rangs {missing} dans {directory}. "
                f"Ces rangs ont probablement crashe (voir leurs logs plus haut)."
            )
        if now - last_log >= log_every_s:
            print(f"[rank 0] En attente des shards '{tag}' des rangs {missing} "
                  f"({now - (deadline - timeout_s):.0f}s ecoulees)...", flush=True)
            last_log = now
        time.sleep(poll_s)

    merged = {}
    n_duplicates = 0
    for r in range(world_size):
        data = _read_shard(paths[r], r, run_id, str(tag))
        for k, v in data.items():
            key = key_type(k) if key_type is not str else k
            if key in merged:
                n_duplicates += 1
            merged[key] = v

    if n_duplicates:
        print(f"[rank 0] [WARN] {n_duplicates} cles dupliquees entre shards '{tag}' "
              f"(des entrees ont ete ecrasees).", flush=True)

    shutil.rmtree(directory, ignore_errors=True)
    try:  # repertoire du run : vide une fois la derniere fusion faite
        os.rmdir(os.path.dirname(directory))
    except OSError:
        pass
    print(f"[rank 0] Merged {len(merged)} entrees depuis {world_size} GPUs (tag '{tag}')", flush=True)
    return merged
