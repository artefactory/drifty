import json
from collections import Counter
import os
from pathlib import Path
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from PipelineTest.FeatureSelection.evaluate import benchmark_evaluate
from PipelineTest.FeatureSelection.configs import extract_features_and_labels, CONFIGS

top_n = 31

PATH_RESULTS = "PipelineTest/res/FeatureSelection/results"
SAVE_DIR = f"PipelineTest/res/FeatureSelection/results_top{top_n}"

def compute_lr_feature_frequencies(results_path: str) -> tuple[dict[str, float], int]:
    path = Path(results_path)
    feature_counts = Counter()
    total_configs = 0

    for json_file in path.glob("*.json"):
        with open(json_file, "r", encoding="utf-8") as f:
            data = json.load(f)

        selected_features = (
            data.get("evaluation", {})
            .get("stability", {})
            .get("classifiers", {})
            .get("LR", {})
            .get("selected_features", [])
        )

        feature_counts.update(set(selected_features))
        total_configs += 1

    if total_configs == 0:
        return {}, 0

    frequencies = {
        feature: count / total_configs
        for feature, count in feature_counts.most_common()
    }

    return frequencies, total_configs


def evaluate_topn_features(cfg, top_n=top_n):
    """Evaluate the performance of the top-n features using logistic regression."""

    features, feature_names, labels = extract_features_and_labels(cfg)

    ranked_features = compute_lr_feature_frequencies(PATH_RESULTS)[0]

    selected_features = [f for f, _ in list(ranked_features.items())[:top_n]]

    selected_indices = [feature_names.index(f) for f in selected_features if f in feature_names]
    X_selected = features[:, selected_indices]
    results = benchmark_evaluate(X_selected, labels, selected_features, selected_features)
    return results

def compute_diff_stability_selection_and_topn_features():
    """Compare performance: stability selection (per-config) vs top-n (cross-config)."""
    results_path = Path(PATH_RESULTS)
    topn_path = Path(SAVE_DIR)

    classifiers = ["LR", "RF", "LGBM"]
    metrics = ["auroc", "auprc"]

    print(f"\n{'='*80}")
    print(f" DIFF: Stability Selection (per-config) vs Top-{top_n} (cross-config)")
    print(f"{'='*80}")
    print(f"{'Config':<65} | {'Clf':<5} | {'AUROC diff':>10} | {'AUPRC diff':>10}")
    print("-" * 100)

    for json_file in sorted(results_path.glob("*.json")):
        with open(json_file, "r") as f:
            stab_data = json.load(f)

        config_name = stab_data.get("config", json_file.stem)
        topn_file = topn_path / f"evaluation_top{top_n}_{config_name}.json"

        if not topn_file.exists():
            continue

        with open(topn_file, "r") as f:
            topn_data = json.load(f)

        stab_classifiers = (
            stab_data.get("evaluation", {})
            .get("all_features", {})
            .get("classifiers", {})
        )

        for clf in classifiers:
            stab_res = stab_classifiers.get(clf, {})
            topn_res = topn_data.get(clf, {})
            if not stab_res or not topn_res:
                continue

            auroc_diff = topn_res["auroc"] - stab_res["auroc"]
            auprc_diff = topn_res["auprc"] - stab_res["auprc"]

            print(f"{config_name:<65} | {clf:<5} | {auroc_diff:>+10.4f} | {auprc_diff:>+10.4f}")

    print("=" * 100)
    print("(positive = top-n better, negative = stability better)\n")


if __name__ == "__main__":
    freqs, total = compute_lr_feature_frequencies(PATH_RESULTS)

    print("\n" + "=" * 55)
    print(" 📊 FRÉQUENCE DE SÉLECTION DES FEATURES (LR / STABILITY)")
    print("=" * 55)

    if total == 0:
        print(f"⚠️  Aucun fichier JSON trouvé dans : '{PATH_RESULTS}'")
    else:
        print(f"📁 Configurations (fichiers JSON) analysées : {total}\n")
        print(f"{'Feature':<40} | {'Fréquence':<10}")
        print("-" * 55)
        cmpt = 1
        for feature, freq in freqs.items():
            print(cmpt, f"{feature:<40} | {freq * 100:>6.1f}%")
            cmpt += 1


    print("total = ", len(freqs), '/', 37)

    print("=" * 55 + "\n")


    # # # Évaluation des performances des top-n features
    # print(f"Évaluation des performances des {top_n} meilleures features :")
    # for config_key, cfg in CONFIGS.items():
    #     results = evaluate_topn_features(cfg, top_n=top_n)
    #     os.makedirs(SAVE_DIR, exist_ok=True)
    #     with open(f"{SAVE_DIR}/evaluation_top{top_n}_{cfg['name']}.json", "w") as f:
    #         json.dump(results, f, indent=2)
    #     print(f"Résultats enregistrés dans : {SAVE_DIR}/evaluation_top{top_n}_{cfg['name']}.json")

    # compute_diff_stability_selection_and_topn_features()
