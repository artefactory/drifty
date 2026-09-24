import sys
import os
import csv
import json
import argparse
import subprocess
import tempfile
import shutil
import traceback
import time
import numpy as np
from sympy import python
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from PipelineTest.eval_configs import CONFIGS

# Resolve the Python executable with project dependencies

def _resolve_python():
    """Return the Python executable that has the project dependencies installed."""
    # Try current interpreter first
    try:
        import transformers  # noqa: F401
        return sys.executable
    except ImportError:
        pass
    # Walk up from Benchmark/ to find .venv/bin/python
    here = os.path.dirname(os.path.abspath(__file__))
    for ancestor in [here] + [
        os.path.abspath(os.path.join(here, *[".."] * i)) for i in range(1, 6)
    ]:
        venv_python = os.path.join(ancestor, ".venv", "bin", "python")
        if os.path.isfile(venv_python):
            return venv_python
    return sys.executable

python_exec = _resolve_python()

# ---------------------------------------------------------------------------
# Baseline registry
# ---------------------------------------------------------------------------
BASELINE_REGISTRY = {
    "perplexity": {
        "display": "Perplexity",
        "module": "PipelineTest.Benchmark.perplexity",
        "func": "run_config",
        "needs_gpu": False,
    },
    "ln_entropy": {
        "display": "LN-Entropy",
        "module": "PipelineTest.Benchmark.LNEntropy",
        "func": "run_config",
        "needs_gpu": False,
    },
    "baseline_markov": {
        "display": "Baseline+Markovian",
        "module": "PipelineTest.Benchmark.Baseline_and_markovian_features",
        "func": "main",
        "needs_gpu": False,
    },
    "semantic_entropy": {
        "display": "Semantic Entropy",
        "module": "PipelineTest.Benchmark.semantic_entropy",
        "func": "run_config",
        "needs_gpu": True,
    },
    "lexical_similarity": {
        "display": "Lexical Similarity",
        "module": "PipelineTest.Benchmark.lexical_similarity",
        "func": "run_config",
        "needs_gpu": True,
    },
    # Entraine un petit Transformer : tourne in-process, sur GPU si le job en a un
    # (q -G 1), sinon sur CPU.
    "tracedet": {
        "display": "TraceDet",
        "module": "PipelineTest.Benchmark.tracedet",
        "func": "run_config",
        "needs_gpu": False,
    },
}


def _import_func(module_path, func_name):
    """Lazy-import a function from a dotted module path."""
    import importlib
    mod = importlib.import_module(module_path)
    return getattr(mod, func_name)


def _pos_neg_counts(sub_result):
    """Extract (n_pos, n_neg) of the scored (test) set from a metrics dict, trying
    the various field names used across baseline scripts."""
    if "n_halluc" in sub_result and "n_correct" in sub_result:
        return sub_result["n_halluc"], sub_result["n_correct"]
    if "confusion_matrix" in sub_result:
        cm = sub_result["confusion_matrix"]
        return cm.get("TP", 0) + cm.get("FN", 0), cm.get("TN", 0) + cm.get("FP", 0)
    if "test_pos_rate" in sub_result:
        n = sub_result.get("n_test") or sub_result.get("n_samples")
        if n:
            n_pos = int(round(sub_result["test_pos_rate"] * n))
            return n_pos, n - n_pos
    return None, None


def _auroc_se(roc_auc, n_pos, n_neg):
    """Hanley & McNeil (1982) standard error of an AUROC estimate, from the
    positive/negative counts of the scored set. Returns None if not computable."""
    if roc_auc is None or not n_pos or not n_neg:
        return None
    auc = float(roc_auc)
    q1 = auc / (2 - auc)
    q0 = (2 * auc ** 2) / (1 + auc)
    var = (auc * (1 - auc) + (n_pos - 1) * (q1 - auc ** 2) + (n_neg - 1) * (q0 - auc ** 2)) / (n_pos * n_neg)
    return float(np.sqrt(max(var, 0.0)))


def _extract_metrics(result, baseline_key):
    """
    Normalize baseline results into {roc_auc, pr_auc, roc_auc_se}.
    Handles both flat dicts and nested (Baseline_and_markovian_features) results.
    roc_auc_se is the Hanley-McNeil standard error of the AUROC estimate (our
    uncertainty on the AUROC), computed from the pos/neg counts of the scored set.
    """
    if result is None:
        return None

    if "test_roc_auc" in result and "test_pr_auc" in result:
        n_pos, n_neg = _pos_neg_counts(result)
        return {
            "roc_auc": result["test_roc_auc"],
            "pr_auc": result["test_pr_auc"],
            "n_samples": result.get("n_samples") or result.get("n_test"),
            "roc_auc_se": _auroc_se(result["test_roc_auc"], n_pos, n_neg),
        }

    if "results" in result:
        nested = {}
        for sub_name, sub in result["results"].items():
            n_pos, n_neg = _pos_neg_counts(sub)
            nested[sub_name] = {
                "roc_auc": sub.get("test_roc_auc"),
                "pr_auc": sub.get("test_pr_auc"),
                "roc_auc_se": _auroc_se(sub.get("test_roc_auc"), n_pos, n_neg),
            }
        return nested

    return None


def get_model_folder(config_name):
    """Map a config name to its model sub-folder under values/ (gemma/llada/dream)."""
    name = config_name.lower()
    if "gemma" in name:
        return "gemma"
    if "dream" in name:
        return "dream"
    return "llada"


# Which column(s) each baseline contributes to the per-sample CSV, read off
# its raw result's "samples" dict (see each baseline's run_config/main()).
SAMPLE_VALUE_COLUMNS = {
    "semantic_entropy": ["semantic_entropy"],
    "perplexity": ["perplexity"],
    "ln_entropy": ["ln_entropy"],
    "lexical_similarity": ["lexical_similarity", "rouge_l_max_f1", "rouge_l_mean_f1",
                           "rouge_l_max_prec", "rouge_l_max_rec", "rouge_l_var"],
    "baseline_markov": ["baseline", "markov", "baseline_markov"],
    "tracedet": ["tracedet"],
}


def _write_sample_csv(config_name, raw_results, values_dir):
    """
    Write one CSV per config with, for each sample, the per-sample value of
    every baseline that exposes one (semantic entropy, perplexity, ln-entropy,
    lexical similarity, baseline / markov / baseline+markov).
    `raw_results` maps baseline key (as in BASELINE_REGISTRY) -> raw result dict.
    """
    rows = {}

    for bl_key, columns in SAMPLE_VALUE_COLUMNS.items():
        result = raw_results.get(bl_key)
        if result is None or "samples" not in result:
            continue
        for idx_key, info in result["samples"].items():
            idx = int(idx_key)
            row = rows.setdefault(idx, {})
            row.setdefault("label_hallucination", info.get("label_hallucination"))
            split = info.get("split")
            if split is None and "is_in_test_split" in info:
                split = "test" if info["is_in_test_split"] else "train"
            row.setdefault("split", split)
            for col in columns:
                row[col] = info.get(col)

    if not rows:
        return None

    out_dir = os.path.join(values_dir, get_model_folder(config_name))
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"{config_name}.csv")

    columns = ["sample_index", "label_hallucination", "split"]
    for bl_columns in SAMPLE_VALUE_COLUMNS.values():
        columns.extend(bl_columns)

    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=columns)
        writer.writeheader()
        for idx in sorted(rows.keys()):
            row = {"sample_index": idx}
            row.update(rows[idx])
            writer.writerow(row)

    print(f"  [CSV] Per-sample values saved to: {out_path}")
    return out_path

def _run_gpu_baseline(bl_key, config_name, nproc, seed=42, python_exec=python_exec):
    """
    Launch a GPU baseline via torchrun with nproc processes.
    Writes a temp script, runs it via torchrun, and streams the output live.
    `seed` is threaded through to the stochastic generation step (semantic
    entropy / lexical similarity) so repeated runs give the same results.
    """
    bl = BASELINE_REGISTRY[bl_key]
    module_name = bl["module"].split(".")[-1]
    func_name = bl["func"]

    # Absolute project root
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))

    # semantic_entropy.run_config() needs a pre-loaded generation backend + NLI
    # pipeline + rank/world_size/local_rank that only its own CLI (main()) builds --
    # a bare `run_config(config_name)` call is missing 7 required positional args.
    # So for this baseline we drive its actual CLI as the torchrun target and read
    # back the JSON file its main() writes, instead of importing run_config directly.
    is_semantic_entropy = bl_key == "semantic_entropy"
    out_dir = None
    script_cmd_args = []

    if is_semantic_entropy:
        script_path = os.path.join(project_root, "PipelineTest", "Benchmark", "semantic_entropy.py")
        out_dir = tempfile.mkdtemp(prefix=f"bench_{bl_key}_out_")
        script_cmd_args = ["--config", config_name, "--n_variants", "5", "--output_dir", out_dir,
                            "--seed", str(seed)]
    else:
        # Extra args per baseline
        extra_args = f", n_variants=5, seed={seed}" if bl_key == "lexical_similarity" else ""

        # Write a launcher script to a temp file
        script_content = f"""\
import sys, os, json
sys.path.insert(0, "{project_root}")

from PipelineTest.Benchmark.{module_name} import {func_name}

result = {func_name}("{config_name}"{extra_args})

# Only rank 0 writes the result. On lit RANK dans l'environnement pose par
# torchrun : les benchmarks n'initialisent plus de process group (cf.
# PipelineTest/Benchmark/dist_utils.py), donc dist.is_initialized() est faux
# sur tous les rangs.
rank = int(os.environ.get("RANK", "0"))

if rank == 0 and result is not None:
    out_dir = "/tmp/benchmark_gpu_results"
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "{bl_key}_{config_name}.json")
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2)
    print(f"[RESULT_SAVED] {{out_path}}")
"""

        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".py", prefix=f"bench_{bl_key}_",
            dir="/tmp", delete=False
        ) as f:
            f.write(script_content)
            script_path = f.name

    try:
        python_exe = python_exec or _resolve_python()

        # CORRECTION 1 : Génération d'un vrai port aléatoire pour éviter le freeze du Rendezvous
        import random
        random_port = random.randint(30000, 50000)
        
        cmd = [
            python_exe, "-m", "torch.distributed.run",
            f"--nproc_per_node={nproc}",
            f"--master_port={random_port}",
            script_path,
        ] + script_cmd_args

        # Validation et affichage du GPU actif
        import torch
        if torch.cuda.is_available():
            gpu_names = [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]
            print(f"    [GPU HARDWARE ACTIVE] Alloué(s) au job : {', '.join(gpu_names)}", flush=True)
        else:
            print("    [CRITICAL WARN] Aucun GPU détecté par le script parent !", flush=True)

        print(f"    Launching: {python_exe} -m torch.distributed.run --nproc_per_node={nproc} (Port: {random_port})")
        print(f"    --- DÉBUT DES LOGS EN DIRECT DE {bl['display'].upper()} ---", flush=True)

        # CORRECTION 2 : Forcer l'environnement à être NON-TAMPONNÉ (Unbuffered)
        # Cela oblige le processus enfant à cracher ses logs immédiatement dans le pipe
        env_unbuffered = os.environ.copy()
        env_unbuffered["PYTHONUNBUFFERED"] = "1"
        # Reduit la fragmentation de l'allocateur CUDA (cf. torch.OutOfMemoryError
        # observe malgre de la VRAM "libre" mais reservee/fragmentee par les gros
        # tenseurs temporaires du sampler MDLM).
        env_unbuffered.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

        # Utilisation de Popen pour lire le flux en temps réel
        proc = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            env=env_unbuffered  # Injection de l'environnement corrigé
        )

        # Affiche les lignes au fur et à mesure sans aucun décalage de temps
        for line in proc.stdout:
            print(f"      [torchrun-live] {line}", end="", flush=True)

        proc.wait(timeout=7200)
        print(f"    --- FIN DES LOGS EN DIRECT (Exit Code: {proc.returncode}) ---", flush=True)

        if proc.returncode != 0:
            print(f"    [ERROR] torchrun failed (rc={proc.returncode})")
            return None

        if is_semantic_entropy:
            # semantic_entropy.main() names the file after CONFIGS[config_name]["name"]
            # (single-config run) and keys the JSON dict by config_name.
            cfg_display_name = CONFIGS[config_name]["name"]
            result_path = os.path.join(out_dir, f"semantic_entropy_results_{cfg_display_name}_debertav1.json")
            if os.path.exists(result_path):
                with open(result_path, "r") as f:
                    results_by_config = json.load(f)
                return results_by_config.get(config_name)
            print(f"    [WARN] No result file found at {result_path}")
            return None

        # Read result from temp file
        result_path = f"/tmp/benchmark_gpu_results/{bl_key}_{config_name}.json"
        if os.path.exists(result_path):
            with open(result_path, "r") as f:
                result = json.load(f)
            os.remove(result_path)
            return result

        print(f"    [WARN] No result file found at {result_path}")
        return None

    finally:
        if is_semantic_entropy:
            if out_dir is not None:
                shutil.rmtree(out_dir, ignore_errors=True)
        elif os.path.exists(script_path):
            os.remove(script_path)

# ---------------------------------------------------------------------------
# Main orchestration
# ---------------------------------------------------------------------------

def run_all(config_names, baselines_to_run, include_gpu=False, nproc=2, values_dir=None, seed=42):
    """
    Run selected baselines across all configs.
    CPU baselines run in-process. GPU baselines launch via torchrun.
    Returns nested dict: {config_name: {baseline: metrics}}.
    If `values_dir` is given, also writes a per-sample CSV per config under
    values_dir/<gemma|llada|dream>/<config_name>.csv.
    `seed` is used for the stochastic generation step of GPU baselines
    (semantic entropy / lexical similarity) so runs are reproducible.
    """
    all_results = {}

    for config_name in config_names:
        print(f"\n{'#' * 70}")
        print(f"  CONFIG: {config_name}")
        print(f"{'#' * 70}")
        all_results[config_name] = {}
        raw_for_csv = {bl_key: None for bl_key in SAMPLE_VALUE_COLUMNS}

        for bl_key in baselines_to_run:
            if bl_key not in BASELINE_REGISTRY:
                continue
            bl = BASELINE_REGISTRY[bl_key]

            if bl["needs_gpu"] and not include_gpu:
                print(f"  [SKIP] {bl['display']} requires GPU (use --include-gpu to enable)")
                continue

            print(f"\n  --- Running baseline: {bl['display']} ---")

            bl_t0 = time.time()
            try:
                if bl["needs_gpu"]:
                    result = _run_gpu_baseline(bl_key, config_name, nproc, seed=seed, python_exec=python_exec)
                else:
                    run_fn = _import_func(bl["module"], bl["func"])
                    result = run_fn(config_name)
            except Exception:
                print(f"  [ERROR] {bl['display']} failed for {config_name}:")
                traceback.print_exc()
                result = None
            bl_t1 = time.time()
            elapsed = bl_t1 - bl_t0

            if bl_key in raw_for_csv:
                raw_for_csv[bl_key] = result

            metrics = _extract_metrics(result, bl_key)
            if metrics is not None:
                # Ajoute le timing directement dans le dict de métriques,
                # que ce soit un dict plat (roc_auc/pr_auc) ou nested
                # (Baseline_and_markovian_features -> plusieurs sous-résultats)
                if "roc_auc" in metrics:
                    metrics["elapsed_seconds"] = round(elapsed, 2)
                else:
                    # dict nested : on ajoute le timing global du run
                    # (partagé par tous les sous-résultats de cette baseline)
                    metrics["_elapsed_seconds"] = round(elapsed, 2)

                all_results[config_name][bl["display"]] = metrics
                if isinstance(metrics, dict) and "roc_auc" in metrics:
                    print(f"  => ROC-AUC: {metrics['roc_auc']:.4f}  PR-AUC: {metrics['pr_auc']:.4f}  "
                          f"({elapsed:.1f}s)")
                else:
                    print(f"  => completed in {elapsed:.1f}s (see detailed output)")
            else:
                # Même sans résultat exploitable, on garde une trace du temps passé
                all_results[config_name][bl["display"]] = {"elapsed_seconds": round(elapsed, 2), "result": None}
                print(f"  => no result returned ({elapsed:.1f}s)")

        if values_dir is not None:
            _write_sample_csv(config_name, raw_for_csv, values_dir)

    return all_results


def print_summary(all_results):
    """Print a compact summary table."""
    print(f"\n{'=' * 80}")
    print(f"  CONSOLIDATED BENCHMARK SUMMARY")
    print(f"{'=' * 80}")

    all_baselines = set()
    for cfg_results in all_results.values():
        all_baselines.update(cfg_results.keys())
    all_baselines = sorted(all_baselines)

    if not all_baselines:
        print("  No results collected.")
        return

    header = f"  {'Config':<50}"
    for bl in all_baselines:
        header += f" {bl[:12]:>13}"
    print(header)
    print(f"  {'-' * (50 + 13 * len(all_baselines))}")

    for config_name, cfg_results in all_results.items():
        short_name = config_name[:50]
        row = f"  {short_name:<50}"
        for bl in all_baselines:
            if bl in cfg_results:
                m = cfg_results[bl]
                if isinstance(m, dict) and "roc_auc" in m:
                    row += f" {m['roc_auc']:>6.4f}/{m['pr_auc']:<6.4f}"
                else:
                    if "Baseline+Markov" in m:
                        bm = m["Baseline+Markov"]
                        row += f" {bm.get('roc_auc',0):>6.4f}/{bm.get('pr_auc',0):<6.4f}"
                    else:
                        row += f" {'N/A':>13}"
            else:
                row += f" {'--':>13}"
        print(row)

    print(f"\n  Format per cell: ROC-AUC / PR-AUC")


def main():
    parser = argparse.ArgumentParser(
        description="Run all benchmarks across all CONFIGS and save consolidated results."
    )
    parser.add_argument(
        "--benchmarks", nargs="*", default=None,
        help=f"Available: {list(BASELINE_REGISTRY.keys())}"
    )
    parser.add_argument(
        "--configs", nargs="*", default=None,
        help="Config keys to evaluate (default: all CONFIGS)"
    )
    parser.add_argument(
        "--include-gpu", action="store_true",
        help="Also run GPU-required baselines"
    )
    parser.add_argument(
        "--nproc", type=int, default=2,
        help="Number of GPUs for GPU baselines via torchrun"
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Seed for stochastic generation (semantic entropy / lexical similarity), "
             "so re-running the benchmarks gives the same results."
    )
    parser.add_argument(
        "--output_dir", type=str,
        default=os.path.abspath(os.path.join(os.path.dirname(__file__), "eval")),
        help="Output directory for consolidated JSON"
    )
    parser.add_argument(
        "--values_dir", type=str,
        default=os.path.abspath(os.path.join(os.path.dirname(__file__), "values")),
        help="Output directory for per-sample CSVs (gemma/llada/dream subfolders)"
    )
    args = parser.parse_args()

    if args.benchmarks:
        for b in args.benchmarks:
            if b not in BASELINE_REGISTRY:
                parser.error(f"Unknown baseline '{b}'. Available: {list(BASELINE_REGISTRY.keys())}")
        baselines_to_run = args.benchmarks
    else:
        if args.include_gpu:
            baselines_to_run = list(BASELINE_REGISTRY.keys())
        else:
            baselines_to_run = [
                k for k, v in BASELINE_REGISTRY.items() if not v["needs_gpu"]
            ]

    config_names = args.configs if args.configs else list(CONFIGS.keys())

    for c in config_names:
        if c not in CONFIGS:
            parser.error(f"Unknown config '{c}'. Available: {list(CONFIGS.keys())}")

    # ... (début de la fonction main() inchangé) ...

    print(f"Configs to evaluate: {config_names}")
    print(f"Baselines to run:    {baselines_to_run}")
    print(f"Include GPU:         {args.include_gpu}")
    
    # --- AJUSTEMENT AUTOMATIQUE DU NOMBRE DE GPU ---
    nproc_to_use = args.nproc
    if args.include_gpu:
        import torch
        available_gpus = torch.cuda.device_count() if torch.cuda.is_available() else 0
        if available_gpus == 0:
            print("  [WARN] Aucun GPU détecté par PyTorch dans cet environnement ! Repli sur 1.")
            nproc_to_use = 1
        else:
            # Si tu demandes plus de GPU (ex: 4) que ce que ton "-G 1" t'a donné (ex: 1),
            # le script se bride automatiquement pour éviter le crash.
            nproc_to_use = min(args.nproc, available_gpus)
            print(f"  [INFO] GPU(s) détecté(s) dans le job : {available_gpus} | nproc ajusté à : {nproc_to_use}")

    # Prépare l'arborescence values/{gemma,llada,dream}/ pour les CSV par échantillon
    for model_folder in ("gemma", "llada", "dream"):
        os.makedirs(os.path.join(args.values_dir, model_folder), exist_ok=True)

    # Lancement de l'orchestration avec le bon nombre de GPU
    all_results = run_all(
        config_names, baselines_to_run,
        include_gpu=args.include_gpu, nproc=nproc_to_use,
        values_dir=args.values_dir, seed=args.seed,
    )

    os.makedirs(args.output_dir, exist_ok=True)
    suffix = "all" if len(config_names) > 1 else config_names[0]
    out_path = os.path.join(args.output_dir, f"benchmarks_{suffix}.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)
    print(f"\n  Consolidated results saved to: {out_path}")

    print_summary(all_results)


if __name__ == "__main__":
    main()