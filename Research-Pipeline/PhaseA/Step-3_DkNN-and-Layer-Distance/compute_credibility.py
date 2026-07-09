"""Step-3b: compute DkNN Credibility for every hypothesis and paraphrase.

Uses the K chosen by select_k.py (override with --k). Outputs per-row
credibilities, drops, the data-derived anomalous-drop threshold, and the
methodology's scenario diagnosis (generalization / representation /
within_normal_range) - plus the Phase-A key finding: does a low-credibility
hypothesis predict that its paraphrase will flip? (correlation on the TEST
portion, which K selection never saw).

Run:  python compute_credibility.py --model BERT-base --dataset SNLI
"""
import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path[:0] = [str(REPO_ROOT), str(REPO_ROOT / "Research-Pipeline")]

import numpy as np
import pandas as pd

from common.alignment import load_aligned
from common.config_loader import load_config, step_results_dir
from eval_metrics import (DkNN, anomalous_drop_threshold,
                          credibility_from_neighbor_labels,
                          diagnose_credibility_scenario)

STEP_DIRNAME = "Step-3_DkNN-and-Layer-Distance"


def main(model_key, dataset_key, k=None):
    cfg = load_config(model_key, dataset_key)
    out_dir = step_results_dir(STEP_DIRNAME, cfg)
    if k is None:
        with open(out_dir / "k_selection.json") as f:
            k = json.load(f)["chosen_k"]

    data = load_aligned(cfg)
    print(f"[credibility] {model_key} x {dataset_key} with K={k}")
    dknn = DkNN().fit(data["bank_reps"], data["bank_labels"])

    unique_rows = np.unique(data["hypo_rows"])
    pos_of = {row: i for i, row in enumerate(unique_rows)}
    row_pos = np.array([pos_of[r] for r in data["hypo_rows"]])
    nl_hypo = dknn.top_neighbor_labels(data["bank_reps"][unique_rows], k,
                                       exclude_self_indices=unique_rows)
    nl_para = dknn.top_neighbor_labels(data["para_reps"], k)

    cred_h_unique = credibility_from_neighbor_labels(nl_hypo, data["bank_labels"][unique_rows])
    cred_h = cred_h_unique[row_pos]
    cred_p = credibility_from_neighbor_labels(nl_para, data["para_preds"])
    drops = cred_h - cred_p

    threshold = anomalous_drop_threshold(drops)
    anomalous = drops > threshold
    scenario = diagnose_credibility_scenario(cred_h_unique, cred_p)
    inconsistent = data["inconsistent"].astype(bool)
    test = ~data["val_mask"]

    def _corr(x, y):
        return float(np.corrcoef(x, y)[0, 1]) if x.std() > 0 and y.std() > 0 else 0.0

    df = pd.DataFrame({
        "pair_id": data["pair_ids"],
        "para_idx": data["para_idx"],
        "split": np.where(data["val_mask"], "validation", "test"),
        "cred_hypothesis": cred_h,
        "cred_paraphrase": cred_p,
        "drop": drops,
        "anomalous_drop": anomalous.astype(int),
        "inconsistent": inconsistent.astype(int),
    })
    df.to_csv(out_dir / "credibility.csv", index=False)

    summary = {
        "model": model_key,
        "dataset": dataset_key,
        "k": int(k),
        "scenario": scenario,
        "anomalous_drop_threshold": threshold,
        "mean_drop_consistent": float(drops[~inconsistent].mean()) if (~inconsistent).any() else None,
        "mean_drop_inconsistent": float(drops[inconsistent].mean()) if inconsistent.any() else None,
        "share_anomalous_given_inconsistent":
            float(anomalous[inconsistent].mean()) if inconsistent.any() else None,
        "share_anomalous_given_consistent":
            float(anomalous[~inconsistent].mean()) if (~inconsistent).any() else None,
        "phase_a_key_finding_corr_drop_vs_inconsistency_on_test":
            _corr(drops[test], inconsistent[test].astype(float)),
        "n_rows": int(len(df)),
    }
    with open(out_dir / "credibility_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(f"[credibility] scenario: {scenario['scenario']} | "
          f"test corr(drop, inconsistency) = "
          f"{summary['phase_a_key_finding_corr_drop_vs_inconsistency_on_test']:.4f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--k", type=int, default=None,
                        help="override the K chosen by select_k.py")
    args = parser.parse_args()
    main(args.model, args.dataset, args.k)
