"""Step-4d: the FINAL REPORT - every metric of the whole run, one table.

Joins, per (model, dataset) combination:
  from Step-2   paraphrase rows / hypotheses that entered
  from Step-4a  RFR, SFR, PC, PA (the four output-consistency metrics)
  from Step-3   chosen K, the Phase-A key correlation (test portion),
                credibility scenario, distance crossing layer + region
  from Step-4b  joint verdict, signals_agree, low/high credibility-group
                inconsistency shares

Output: results/final_report.csv (+ a readable table printed to the log).

Run:  python final_report.py
      python final_report.py --models BERT-base --datasets SNLI
"""
import argparse
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[2]
sys.path[:0] = [str(REPO_ROOT), str(REPO_ROOT / "Research-Pipeline")]

import pandas as pd

from common.config_loader import (combo_name, list_dataset_keys,
                                  list_model_keys, load_config,
                                  step_results_dir)
from common.logging_utils import log

STEP3_DIRNAME = "Step-3_DkNN-and-Layer-Distance"


def _read_json(path):
    if not path.exists():
        return None
    with open(path) as f:
        return json.load(f)


def combo_row(model_key, dataset_key, metrics_df):
    cfg = load_config(model_key, dataset_key)
    step3 = step_results_dir(STEP3_DIRNAME, cfg)
    cred = _read_json(step3 / "credibility_summary.json")
    dist = _read_json(step3 / "distance_summary.json")
    diag = _read_json(HERE / "results" / combo_name(cfg) / "diagnosis.json")
    if cred is None or dist is None or diag is None:
        return None

    row = {"model": model_key, "dataset": dataset_key}
    metric_rows = metrics_df[(metrics_df["model"] == model_key) &
                             (metrics_df["dataset"] == dataset_key)]
    if len(metric_rows):
        m = metric_rows.iloc[0]
        row.update({
            "relaxed_fooling_rate": m["relaxed_fooling_rate"],
            "strict_fooling_rate": m["strict_fooling_rate"],
            "paraphrastic_consistency": m["paraphrastic_consistency"],
            "pattern_accuracy": m["pattern_accuracy"],
            "paraphrase_rows": m["paraphrase_rows"],
            "hypotheses": m["hypotheses"],
        })
    groups = diag.get("credibility_groups", {})
    row.update({
        "chosen_k": cred["k"],
        "corr_drop_vs_inconsistency_test":
            cred["phase_a_key_finding_corr_drop_vs_inconsistency_on_test"],
        "credibility_scenario": cred["scenario"]["scenario"],
        "credibility_verdict": diag["credibility_verdict"],
        "distance_crossing_layer": dist["crossing_layer"],
        "distance_region": dist["region"],
        "signals_agree": diag["signals_agree"],
        "low_cred_group_inconsistent_share": groups.get("share_inconsistent_low_group"),
        "high_cred_group_inconsistent_share": groups.get("share_inconsistent_high_group"),
    })
    return row


def main(models, datasets):
    metrics_path = HERE / "results" / "consistency_metrics.csv"
    metrics_df = (pd.read_csv(metrics_path) if metrics_path.exists()
                  else pd.DataFrame(columns=["model", "dataset"]))
    rows = []
    for model in models:
        for dataset in datasets:
            row = combo_row(model, dataset, metrics_df)
            if row is None:
                log("SKIP", "final report: results incomplete", model, dataset)
            else:
                rows.append(row)
    if not rows:
        log("REPORT", "nothing to report yet - run steps 2-4 first")
        return
    out_dir = HERE / "results"
    out_dir.mkdir(parents=True, exist_ok=True)
    report = pd.DataFrame(rows)
    report.to_csv(out_dir / "final_report.csv", index=False)
    log("REPORT", f"final report written -> {out_dir / 'final_report.csv'}")
    print(report.to_string(index=False, float_format=lambda v: f"{v:.4f}"))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--models", default=None)
    parser.add_argument("--datasets", default=None)
    args = parser.parse_args()
    main(args.models.split(",") if args.models else list_model_keys(),
         args.datasets.split(",") if args.datasets else list_dataset_keys())
