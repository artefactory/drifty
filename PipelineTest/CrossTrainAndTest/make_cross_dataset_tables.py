"""Génère les tableaux LaTeX de généralisation cross-dataset (ROC / PR)
à partir des résultats agrégés de PipelineTest/CrossTrainAndTest/eval.

Plusieurs JSON peuvent être fusionnés (par exemple un fichier LLaDA/Dream et un
fichier DiffusionGemma) pour obtenir un tableau à plusieurs modèles.

Usage :
    uv run PipelineTest/CrossTrainAndTest/make_cross_dataset_tables.py \
        eval/all_results.json eval/all_results_diffugemma.json \
        --out ../res/tables/tables_cross_dataset_all_results.tex
"""

import argparse
import json
import re
from pathlib import Path

HERE = Path(__file__).resolve().parent
TABLES_DIR = HERE.parent / "res" / "tables"

DATASETS = ["triviaqa", "naturalquestion", "hotpotqa"]
DATASET_LABEL = {
    "triviaqa": "TriviaQA",
    "naturalquestion": "Natural Questions",
    "hotpotqa": "HotpotQA",
}
MODEL_ORDER = ["llada", "dream", "diffgemma"]
MODEL_LABEL = {
    "llada": "LLaDA",
    "dream": "Dream",
    "diffgemma": "DiffGemma",
}
METHOD_LABEL = {
    "Baseline": "Baseline",
    "Markovian": "Markovien",
    "Baseline+Markov": "Baseline + Markovien",
    "CleanedFeatures": "PipelineFinale",
}

# Le tableau de référence du rapport : PipelineFinale, 16 pas / 32 tokens.
MAIN_METHOD = "CleanedFeatures"

KEY_RE = re.compile(
    r"^(?P<model>[a-z]+)_(?P<steps>\d+)steps_(?P<tokens>\d+)tokens_"
    r"(?P<train>[a-z]+)_(?P<n>\d+)samples_eval_(?P<test>[a-z]+)_seed(?P<seed>\d+)$"
)


def load(paths):
    """{(model, steps, tokens, method, train, test): (roc, pr)} pour tous les JSON."""
    table = {}
    for path in paths:
        raw = json.loads(Path(path).read_text())
        for key, entry in raw.items():
            m = KEY_RE.match(key)
            if m is None:
                raise ValueError(f"clé non reconnue dans {path} : {key}")
            g = m.groupdict()
            for method, res in entry["results"].items():
                idx = (g["model"], int(g["steps"]), int(g["tokens"]), method,
                       g["train"], g["test"])
                table[idx] = (res["test_roc_auc"], res["test_pr_auc"])
    return table


def model_sort_key(model):
    return (MODEL_ORDER.index(model) if model in MODEL_ORDER else len(MODEL_ORDER), model)


def fmt(value, bold):
    txt = f"{value:.4f}"
    return rf"\textbf{{{txt}}}" if bold else txt


def make_table(scores, groups, first_col, caption, label):
    """groups : liste de (titre du groupe, fonction (train, test) -> clé).

    Chaque groupe occupe un \\multirow de 3 lignes (une par dataset
    d'entraînement) ; les colonnes sont les 3 datasets de test (ROC / PR).
    """
    lines = [
        r"\begin{table}[H]",
        r"\centering",
        r"\resizebox{\linewidth}{!}{%",
        r"\renewcommand{\arraystretch}{1.25}",
        r"\begin{tabular}{@{}l l cc cc cc@{}}",
        r"\toprule",
        r"& & " + " & ".join(
            rf"\multicolumn{{2}}{{c}}{{{DATASET_LABEL[d]}}}" for d in DATASETS
        ) + r" \\",
        r"\cmidrule(lr){3-4} \cmidrule(lr){5-6} \cmidrule(lr){7-8}",
        rf"\textbf{{{first_col}}} & \textbf{{Train sur}} & "
        + " & ".join(["ROC & PR"] * len(DATASETS)) + r" \\",
        r"\midrule",
    ]
    for gi, (title, key_of) in enumerate(groups):
        if gi:
            lines.append(r"\midrule")
        lines.append(rf"\multirow{{3}}{{*}}{{{title}}}")
        for train in DATASETS:
            cells = []
            for test in DATASETS:
                roc, pr = scores[key_of(train, test)]
                diag = train == test
                cells += [fmt(roc, diag), fmt(pr, diag)]
            lines.append(
                f"& {DATASET_LABEL[train]:<18} & " + " & ".join(cells) + r" \\"
            )
    lines += [
        r"\bottomrule",
        r"\end{tabular}%",
        r"}",
        rf"\caption{{{caption}}}",
        rf"\label{{{label}}}",
        r"\end{table}",
    ]
    return "\n".join(lines)


def complete(scores, model, steps, tokens, method):
    """Vrai si les 9 couples (train, test) sont présents pour cette combinaison."""
    return all((model, steps, tokens, method, tr, te) in scores
               for tr in DATASETS for te in DATASETS)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("results", nargs="+", type=Path,
                        help="JSON agrégés produits par CrossBaselineAndMarkovianMethod.py")
    parser.add_argument("--methods", nargs="+", default=None,
                        choices=list(METHOD_LABEL),
                        help="méthodes à inclure (défaut : toutes)")
    parser.add_argument("--steps", nargs="+", type=int, default=None,
                        help="nombres de pas à inclure (défaut : tous)")
    parser.add_argument("--per-model-tables", action="store_true",
                        help="ajouter les tableaux comparant les configurations "
                             "d'échantillonnage d'un même modèle")
    parser.add_argument("--out", type=Path,
                        default=TABLES_DIR / "tables_cross_dataset.tex",
                        help="fichier .tex de sortie")
    args = parser.parse_args()

    scores = load(args.results)
    models = sorted({k[0] for k in scores}, key=model_sort_key)
    configs = sorted({(k[1], k[2]) for k in scores}
                     if args.steps is None
                     else {(k[1], k[2]) for k in scores if k[1] in args.steps})
    methods = [m for m in METHOD_LABEL
               if any(k[3] == m for k in scores)
               and (args.methods is None or m in args.methods)]

    blocks = [
        "% " + "=" * 74,
        "% Généralisation cross-dataset (ROC-AUC / PR-AUC) — tableaux générés",
        f"% automatiquement par {Path(__file__).name} à partir de :",
        *[f"%   - {p}" for p in args.results],
        r"% Prérequis : \usepackage{booktabs, multirow, graphicx, float}",
        "% " + "=" * 74,
        "",
    ]

    # 1) Par configuration d'échantillonnage : les modèles en lignes (une table
    #    par méthode), comme le tableau de référence à trois modèles.
    for steps, tokens in configs:
        for method in methods:
            present = [m for m in models if complete(scores, m, steps, tokens, method)]
            if len(present) < 2:
                continue
            groups = [
                (MODEL_LABEL.get(m, m),
                 (lambda m_: lambda tr, te: (m_, steps, tokens, method, tr, te))(m))
                for m in present
            ]
            blocks.append(make_table(
                scores, groups, first_col="Modèle",
                caption=(
                    (r"Généralisation cross-dataset de notre \texttt{PipelineFinale}"
                     if method == MAIN_METHOD else
                     rf"Généralisation cross-dataset de la méthode "
                     rf"\textit{{{METHOD_LABEL[method]}}}")
                    + rf" ({steps} pas / {tokens} tokens). "
                    + r"Diagonale (entraînement et test sur le même dataset) en gras."
                ),
                label=("tab:cross-dataset" if method == MAIN_METHOD
                       else f"tab:cross-dataset-{steps}pas-{tokens}tok-"
                            f"{method.lower().replace('+', '-')}"),
            ))
            blocks.append("")

    # 2) Par modèle disposant de plusieurs configurations d'échantillonnage :
    #    les configurations en lignes (une table par méthode).
    for model in models if args.per_model_tables else []:
        model_configs = [c for c in configs
                         if any(complete(scores, model, c[0], c[1], m) for m in methods)]
        if len(model_configs) < 2:
            continue
        for method in methods:
            groups = [
                (rf"{s} pas / {t} tokens",
                 (lambda s_, t_: lambda tr, te: (model, s_, t_, method, tr, te))(s, t))
                for s, t in model_configs if complete(scores, model, s, t, method)
            ]
            if len(groups) < 2:
                continue
            blocks.append(make_table(
                scores, groups, first_col="Configuration",
                caption=(
                    rf"Généralisation cross-dataset de la méthode "
                    rf"\textit{{{METHOD_LABEL[method]}}} pour "
                    rf"{MODEL_LABEL.get(model, model)}, selon la configuration "
                    r"d'échantillonnage. Diagonale (entraînement et test sur le "
                    r"même dataset) en gras."
                ),
                label=f"tab:cross-dataset-{model}-{method.lower().replace('+', '-')}",
            ))
            blocks.append("")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text("\n".join(blocks))
    n_tables = sum(b.startswith(r"\begin{table}") for b in blocks)
    print(f"écrit : {args.out} ({n_tables} tableaux)")


if __name__ == "__main__":
    main()
