"""Step-4b: joint diagnosis - Credibility scenario x Layer-Distance region.

Reads Step-3's outputs for one (model, dataset) pair, combines the two
signals into a final diagnosis, and produces the two presentation plots:
  layer_distance_profile.png - consistent vs inconsistent mean distance per
      layer + the data-derived threshold + the crossing layer
      ("Reading the distance profile" slide)
  credibility_groups.png - split hypotheses into low / high credibility by
      the data-derived threshold (mean - std of hypothesis credibilities);
      per group, the share of paraphrase rows that became inconsistent
      ("Phase A - can we predict it in advance?" slide)

Per the methodology: the two lenses measure DIFFERENT things (where the
representations split vs where the paraphrase lands relative to training
data), so a disagreement between them is NOT a failure - it is a valid
finding that the inconsistency has more than one source.

Run:  python diagnose.py --model BERT-base --dataset SNLI
"""
import argparse
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[2]
sys.path[:0] = [str(REPO_ROOT), str(REPO_ROOT / "Research-Pipeline")]

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from common.config_loader import combo_name, load_config, step_results_dir

STEP3_DIRNAME = "Step-3_DkNN-and-Layer-Distance"


def main(model_key, dataset_key):
    cfg = load_config(model_key, dataset_key)
    step3_dir = step_results_dir(STEP3_DIRNAME, cfg)
    with open(step3_dir / "credibility_summary.json") as f:
        cred = json.load(f)
    with open(step3_dir / "distance_summary.json") as f:
        dist = json.load(f)

    cred_scenario = cred["scenario"]["scenario"]          # generalization / representation / within_normal_range
    dist_region = dist["region"]                          # representation / reasoning / classifier / None
    majority_inconsistent = dist["n_inconsistent"] > dist["n_consistent"]

    if cred_scenario == "within_normal_range":
        credibility_verdict = "classifier" if majority_inconsistent else "no_problem"
    else:
        credibility_verdict = cred_scenario

    signals_agree = (credibility_verdict == dist_region)
    diagnosis = {
        "model": model_key,
        "dataset": dataset_key,
        "credibility_verdict": credibility_verdict,
        "layer_distance_region": dist_region,
        "crossing_layer": dist["crossing_layer"],
        "signals_agree": signals_agree,
        "note": (None if signals_agree else
                 "The two signals disagree - a valid finding: Layer Distance asks "
                 "WHERE the representations split, Credibility asks WHERE the "
                 "paraphrase lands relative to the training data. Disagreement "
                 "means the inconsistency is multi-dimensional (representation "
                 "and classifier both contribute)."),
        "phase_b_recommendation": {
            "generalization": "targeted paraphrase augmentation + fine-tuning (Phase B)",
            "representation": "report as a representation problem - a key finding; "
                              "augmentation must break the wrong grouping, not just add data",
            "classifier": "classifier fine-tuning / regularization; improvement not guaranteed",
            "no_problem": "model is consistent - no intervention needed",
        }[credibility_verdict],
    }

    out_dir = HERE / "results" / combo_name(cfg)
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "diagnosis.json", "w") as f:
        json.dump(diagnosis, f, indent=2)

    # distance-profile plot
    profile = pd.read_csv(step3_dir / "layer_profile.csv")
    plt.figure(figsize=(8, 4.5))
    plt.plot(profile["layer"], profile["mean_distance_consistent"],
             label="consistent pairs", color="teal")
    plt.plot(profile["layer"], profile["mean_distance_inconsistent"],
             label="inconsistent pairs", color="firebrick")
    plt.plot(profile["layer"], profile["threshold_consistent_mean_plus_std"],
             "--", label="threshold (consistent mean + std)", color="gray")
    if dist["crossing_layer"] is not None:
        plt.axvline(dist["crossing_layer"], color="black", linewidth=0.8,
                    label=f"crossing layer = {dist['crossing_layer']}")
    plt.xlabel("layer")
    plt.ylabel("mean cosine distance (hypothesis vs paraphrase)")
    plt.title(f"{model_key} x {dataset_key} - layer distance profile")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_dir / "layer_distance_profile.png", dpi=150)
    plt.close()

    # credibility-groups plot: does LOW hypothesis credibility predict flips?
    cred_df = pd.read_csv(step3_dir / "credibility.csv")
    per_hypo = cred_df.groupby("pair_id")["cred_hypothesis"].first()
    low_threshold = per_hypo.mean() - per_hypo.std()      # data-derived, std-based
    low_rows = cred_df["cred_hypothesis"] < low_threshold
    shares, sizes = [], []
    for mask in (low_rows, ~low_rows):
        group = cred_df.loc[mask, "inconsistent"]
        shares.append(100.0 * group.mean() if len(group) else 0.0)
        sizes.append(int(mask.sum()))
    plt.figure(figsize=(5.5, 4.5))
    bars = plt.bar(["low credibility", "high credibility"], shares,
                   color=["firebrick", "teal"], width=0.55)
    for bar, share, size in zip(bars, shares, sizes):
        plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.8,
                 f"{share:.1f}%\n(n={size})", ha="center", va="bottom")
    plt.ylabel("share that became inconsistent (%)")
    plt.ylim(0, max(shares + [1]) * 1.25 + 5)
    plt.title(f"{model_key} x {dataset_key} - inconsistency rate by hypothesis "
              f"credibility group\n(low = below mean - std = {low_threshold:.3f})")
    plt.tight_layout()
    plt.savefig(out_dir / "credibility_groups.png", dpi=150)
    plt.close()

    groups_summary = {
        "low_credibility_threshold": float(low_threshold),
        "share_inconsistent_low_group": shares[0] / 100.0,
        "share_inconsistent_high_group": shares[1] / 100.0,
        "n_rows_low_group": sizes[0],
        "n_rows_high_group": sizes[1],
    }
    diagnosis["credibility_groups"] = groups_summary
    with open(out_dir / "diagnosis.json", "w") as f:
        json.dump(diagnosis, f, indent=2)

    print(f"[diagnose] {model_key} x {dataset_key}: credibility -> {credibility_verdict}, "
          f"layer distance -> {dist_region}, agree = {signals_agree}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--dataset", required=True)
    args = parser.parse_args()
    main(args.model, args.dataset)
