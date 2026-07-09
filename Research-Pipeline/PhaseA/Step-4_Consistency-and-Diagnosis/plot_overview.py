"""Step-4c: unified overview figures across ALL (model, dataset) combinations.

Two grid figures (rows = models, columns = datasets), built from Step-3
outputs of every combination that has results so far:

  results/layer_distance_overview.png - the distance-per-layer profile of
      every combination on one canvas (consistent vs inconsistent + threshold),
      exactly like the presentation's distance-profile reading, side by side.
  results/credibility_overview.png - the low-vs-high credibility bar pair of
      every combination ("can we predict it in advance?"), side by side.

Run:  python plot_overview.py
      python plot_overview.py --models BERT-base,BART-large --datasets SNLI
Combinations without Step-3 results yet are shown as empty panels.
"""
import argparse
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[2]
sys.path[:0] = [str(REPO_ROOT), str(REPO_ROOT / "Research-Pipeline")]

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from common.config_loader import (list_dataset_keys, list_model_keys,
                                  load_config, step_results_dir)

STEP3_DIRNAME = "Step-3_DkNN-and-Layer-Distance"


def _grid(models, datasets, panel_fn, suptitle, out_path):
    n_rows, n_cols = len(models), len(datasets)
    fig, axes = plt.subplots(n_rows, n_cols,
                             figsize=(4.4 * n_cols, 3.2 * n_rows), squeeze=False)
    drawn = 0
    for r, model in enumerate(models):
        for c, dataset in enumerate(datasets):
            ax = axes[r][c]
            ax.set_title(f"{model} x {dataset}", fontsize=10)
            cfg = load_config(model, dataset)
            drawn += panel_fn(ax, step_results_dir(STEP3_DIRNAME, cfg))
    fig.suptitle(suptitle, fontsize=14)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return drawn


def _distance_panel(ax, step3_dir):
    profile_path = step3_dir / "layer_profile.csv"
    if not profile_path.exists():
        ax.text(0.5, 0.5, "no results yet", ha="center", va="center",
                transform=ax.transAxes, color="gray")
        ax.set_xticks([]); ax.set_yticks([])
        return 0
    profile = pd.read_csv(profile_path)
    ax.plot(profile["layer"], profile["mean_distance_consistent"],
            color="teal", label="consistent")
    ax.plot(profile["layer"], profile["mean_distance_inconsistent"],
            color="firebrick", label="inconsistent")
    ax.plot(profile["layer"], profile["threshold_consistent_mean_plus_std"],
            "--", color="gray", linewidth=0.9, label="threshold")
    ax.set_xlabel("layer", fontsize=8)
    ax.set_ylabel("mean cosine distance", fontsize=8)
    ax.tick_params(labelsize=8)
    ax.legend(fontsize=7)
    return 1


def _credibility_panel(ax, step3_dir):
    cred_path = step3_dir / "credibility.csv"
    if not cred_path.exists():
        ax.text(0.5, 0.5, "no results yet", ha="center", va="center",
                transform=ax.transAxes, color="gray")
        ax.set_xticks([]); ax.set_yticks([])
        return 0
    cred_df = pd.read_csv(cred_path)
    per_hypo = cred_df.groupby("pair_id")["cred_hypothesis"].first()
    low_threshold = per_hypo.mean() - per_hypo.std()
    low_rows = cred_df["cred_hypothesis"] < low_threshold
    shares = []
    for mask in (low_rows, ~low_rows):
        group = cred_df.loc[mask, "inconsistent"]
        shares.append(100.0 * group.mean() if len(group) else 0.0)
    bars = ax.bar(["low", "high"], shares, color=["firebrick", "teal"], width=0.5)
    for bar, share in zip(bars, shares):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
                f"{share:.1f}%", ha="center", va="bottom", fontsize=8)
    ax.set_ylabel("% inconsistent", fontsize=8)
    ax.set_ylim(0, max(shares + [1]) * 1.3 + 5)
    ax.tick_params(labelsize=8)
    return 1


def main(models, datasets):
    out_dir = HERE / "results"
    out_dir.mkdir(parents=True, exist_ok=True)
    n1 = _grid(models, datasets, _distance_panel,
               "Layer-distance profiles - all combinations",
               out_dir / "layer_distance_overview.png")
    n2 = _grid(models, datasets, _credibility_panel,
               "Inconsistency rate by hypothesis-credibility group - all combinations",
               out_dir / "credibility_overview.png")
    print(f"[overview] distance panels drawn: {n1}, credibility panels drawn: {n2} "
          f"-> {out_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--models", default=None)
    parser.add_argument("--datasets", default=None)
    args = parser.parse_args()
    main(args.models.split(",") if args.models else list_model_keys(),
         args.datasets.split(",") if args.datasets else list_dataset_keys())
