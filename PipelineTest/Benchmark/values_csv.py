"""
PipelineTest/Benchmark/values_csv.py
====================================
Mise a jour en place des CSV par echantillon values/<model>/<config_name>.csv.

Chaque baseline lancee en standalone (lexical_similarity.py,
Baseline_and_markovian_features.py, ...) peut ainsi reecrire ses propres
colonnes sans toucher a celles des autres baselines ni relancer tout
Benchmark/main.py. Module volontairement stdlib-only : il est importe par des
scripts CPU qui n'ont pas besoin de torch/dllm.
"""

import csv
import os

# Schema ecrit par PipelineTest/Benchmark/main.py::_write_sample_csv ; on le
# reprend tel quel pour pouvoir (re)creer un CSV manquant sans casser les
# scripts de plot qui lisent values/.
BASE_CSV_COLUMNS = [
    "sample_index", "label_hallucination", "split",
    "semantic_entropy", "perplexity", "ln_entropy", "lexical_similarity",
    "baseline", "markov", "baseline_markov",
]

DEFAULT_VALUES_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "values"))


def get_model_folder(config_name):
    """Sous-dossier de values/ pour une config (gemma/llada/dream), comme main.py."""
    name = config_name.lower()
    if "gemma" in name:
        return "gemma"
    if "dream" in name:
        return "dream"
    return "llada"


def update_values_csv(config_name, samples, columns, values_dir=DEFAULT_VALUES_DIR):
    """
    Met a jour en place le CSV par echantillon values/<model>/<config_name>.csv.

    Pour chaque sample_index present dans `samples`, reecrit les colonnes
    listees dans `columns` (celles que produit la baseline appelante) en les
    lisant dans samples[idx]. Les colonnes des autres baselines et les lignes
    non concernees sont conservees telles quelles ; les colonnes de `columns`
    absentes du header sont ajoutees a la fin.

    Ecriture atomique (fichier temporaire + os.replace) pour ne pas laisser un
    CSV tronque si le run est interrompu.
    """
    csv_path = os.path.join(values_dir, get_model_folder(config_name), f"{config_name}.csv")

    header = list(BASE_CSV_COLUMNS)
    rows = {}
    if os.path.exists(csv_path):
        with open(csv_path, newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            if reader.fieldnames:
                header = list(reader.fieldnames)
            for row in reader:
                try:
                    rows[int(row["sample_index"])] = row
                except (KeyError, TypeError, ValueError):
                    continue
    else:
        print(f"  [CSV] Fichier absent, creation : {csv_path}")

    for col in columns:
        if col not in header:
            header.append(col)

    n_updated, n_added = 0, 0
    for idx_key, info in samples.items():
        idx = int(idx_key)
        row = rows.get(idx)
        if row is None:
            row = {c: "" for c in header}
            row["sample_index"] = idx
            rows[idx] = row
            n_added += 1
        else:
            n_updated += 1

        for col in columns:
            row[col] = info.get(col, "")
        # Label et split : on ne les ecrase pas s'ils sont deja renseignes par
        # une autre baseline, on se contente de combler les trous.
        for col in ("label_hallucination", "split"):
            if not row.get(col) and info.get(col) is not None:
                row[col] = info[col]

    os.makedirs(os.path.dirname(csv_path), exist_ok=True)
    tmp_path = csv_path + ".tmp"
    with open(tmp_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=header)
        writer.writeheader()
        for idx in sorted(rows):
            writer.writerow({c: rows[idx].get(c, "") for c in header})
    os.replace(tmp_path, csv_path)

    print(f"  [CSV] Colonnes {columns} ecrites dans {csv_path} "
          f"({n_updated} lignes mises a jour, {n_added} ajoutees)")
    return csv_path


# Marqueur des lignes du CSV qui ne sont plus dans le pool de la config
# (label devenu "unclear", sample disparu de l'eval JSON, ...). Elles sont
# conservees telles quelles mais sortent des splits train/test.
SPLIT_OUT_OF_POOL = "excluded"

# Colonnes issues d'un classifieur entraine sur le split train : si une ligne
# change de split, la valeur deja presente dans ces colonnes vient d'un fit sur
# l'ancienne partition -> fuite train/test tant que la baseline n'a pas rejoue.
TRAINED_SCORE_COLUMNS = ("lexical_similarity", "baseline", "markov", "baseline_markov")


def sync_split_column(config_name, train_indices, test_indices,
                      values_dir=DEFAULT_VALUES_DIR, mark_out_of_pool=True):
    """
    Reecrit la colonne `split` du CSV values/<model>/<config_name>.csv pour la
    faire correspondre exactement a la partition train/test courante.

    `get_train_test_split` est positionnel (80 % du pool courant) : des que le
    pool change (un label devient/cesse d'etre "unclear", un sample apparait),
    la frontiere train/test se decale. `update_values_csv` ne comble que les
    trous et ne corrige jamais un `split` deja ecrit : les CSV accumulaient donc
    des lignes marquees `train` alors qu'elles etaient en test au moment du fit
    (et inversement), ce qui biaise toute metrique recalculee depuis le CSV.

    Cette fonction est la reference : le split du CSV est ecrase par celui du
    pool courant. Les lignes hors pool passent a `SPLIT_OUT_OF_POOL` (sauf si
    `mark_out_of_pool=False`) pour qu'aucun filtre `split == "test"` ne les
    ramasse.

    Ne cree pas le CSV s'il n'existe pas. Ecriture atomique.
    Retourne un rapport dict.
    """
    csv_path = os.path.join(values_dir, get_model_folder(config_name), f"{config_name}.csv")

    report = {
        "csv": csv_path,
        "csv_exists": os.path.exists(csv_path),
        "n_pool": len(train_indices) + len(test_indices),
        "n_rows": 0,
        "n_fixed": 0,
        "n_flipped": 0,
        "flipped_indices": [],
        "n_out_of_pool": 0,
        "n_missing_from_csv": 0,
        "stale_trained_columns": {},
        "rewritten": False,
    }
    if not report["csv_exists"]:
        return report

    wanted = {int(i): "train" for i in train_indices}
    wanted.update({int(i): "test" for i in test_indices})

    with open(csv_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        header = list(reader.fieldnames or BASE_CSV_COLUMNS)
        rows = list(reader)

    if "split" not in header:
        header.append("split")

    seen = set()
    for row in rows:
        try:
            idx = int(row["sample_index"])
        except (KeyError, TypeError, ValueError):
            continue
        seen.add(idx)
        report["n_rows"] += 1

        old = (row.get("split") or "").strip()
        if idx in wanted:
            new = wanted[idx]
        else:
            report["n_out_of_pool"] += 1
            if not mark_out_of_pool:
                continue
            new = SPLIT_OUT_OF_POOL

        if old == new:
            continue
        row["split"] = new
        report["n_fixed"] += 1

        # Bascule train <-> test : les scores entraines deja presents sur cette
        # ligne proviennent d'un modele fit sur l'ancienne partition.
        if {old, new} == {"train", "test"}:
            report["n_flipped"] += 1
            report["flipped_indices"].append(idx)
            for col in TRAINED_SCORE_COLUMNS:
                if (row.get(col) or "").strip():
                    report["stale_trained_columns"][col] = \
                        report["stale_trained_columns"].get(col, 0) + 1

    report["n_missing_from_csv"] = sum(1 for i in wanted if i not in seen)

    if report["n_fixed"]:
        tmp_path = csv_path + ".tmp"
        with open(tmp_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=header)
            writer.writeheader()
            for row in rows:
                writer.writerow({c: row.get(c, "") for c in header})
        os.replace(tmp_path, csv_path)
        report["rewritten"] = True

    return report
