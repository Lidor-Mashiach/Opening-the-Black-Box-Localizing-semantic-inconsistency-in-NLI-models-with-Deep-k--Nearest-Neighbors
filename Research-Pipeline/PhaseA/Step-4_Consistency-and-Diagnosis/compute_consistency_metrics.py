"""Step-4a: the four output-consistency metrics, per (model, dataset) pair.

Relaxed Fooling Rate + Strict Fooling Rate (anchor paper), Paraphrastic
Consistency (Srikanth et al. 2024) and Pattern Accuracy (MERGE). All four are
computed from Step-2's paraphrase_predictions.csv via eval_metrics/ - one
implementation, reused everywhere. Results for all combinations aggregate
into a single results/consistency_metrics.csv.

Run:  python compute_consistency_metrics.py            (all combinations)
      python compute_consistency_metrics.py --models BERT-base --datasets SNLI
"""
import argparse
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[2]
sys.path[:0] = [str(REPO_ROOT), str(REPO_ROOT / "Research-Pipeline")]

import pandas as pd

from common.config_loader import (list_dataset_keys, list_model_keys,
                                  load_config, step_results_dir)
from eval_metrics import (paraphrastic_consistency, pattern_accuracy,
                          relaxed_fooling_rate, strict_fooling_rate)

STEP2_DIRNAME = "Step-2_Paraphrase-Inference"


def combo_metrics(model_key, dataset_key):
    cfg = load_config(model_key, dataset_key)
    csv_path = step_results_dir(STEP2_DIRNAME, cfg) / "paraphrase_predictions.csv"
    if not csv_path.exists():
        print(f"[step-4] SKIP {model_key} x {dataset_key}: Step-2 output missing")
        return None
    df = pd.read_csv(csv_path)
    h = df["hypothesis_pred"].to_numpy()
    p = df["paraphrase_pred"].to_numpy()
    ids = df["pair_id"].to_numpy(dtype=str)
    return {
        "model": model_key,
        "dataset": dataset_key,
        "relaxed_fooling_rate": relaxed_fooling_rate(h, p),
        "strict_fooling_rate": strict_fooling_rate(h, p),
        "paraphrastic_consistency": paraphrastic_consistency(h, p),
        "pattern_accuracy": pattern_accuracy(ids, h, p),
        "paraphrase_rows": len(df),
        "hypotheses": df["pair_id"].nunique(),
    }


def main(models, datasets):
    rows = []
    for model in models:
        for dataset in datasets:
            metrics = combo_metrics(model, dataset)
            if metrics is not None:
                rows.append(metrics)
                print(f"[step-4] {model} x {dataset}: "
                      f"RFR={metrics['relaxed_fooling_rate']:.4f} "
                      f"SFR={metrics['strict_fooling_rate']:.4f} "
                      f"PC={metrics['paraphrastic_consistency']:.4f} "
                      f"PA={metrics['pattern_accuracy']:.4f}")
    if not rows:
        print("[step-4] nothing to aggregate yet")
        return
    out_dir = HERE / "results"
    out_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(out_dir / "consistency_metrics.csv", index=False)
    print(f"[step-4] aggregated table -> {out_dir / 'consistency_metrics.csv'}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--models", default=None)
    parser.add_argument("--datasets", default=None)
    args = parser.parse_args()
    main(args.models.split(",") if args.models else list_model_keys(),
         args.datasets.split(",") if args.datasets else list_dataset_keys())
