"""Step-3c: Layer Distance - cosine distance Hypothesis vs Paraphrase, per layer.

No neighbours here: the distance is measured DIRECTLY between the two pooled
vectors at every layer, for consistent and inconsistent pairs alike. The
per-layer threshold (mean + std of the CONSISTENT pairs) is derived from the
data, and the layer where the inconsistent mean first crosses it names the
problem region: lower / middle / upper -> representation / reasoning / classifier.

Run:  python compute_layer_distances.py --model BERT-base --dataset SNLI
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
from eval_metrics import (distance_threshold_per_layer, first_crossing_layer,
                          layer_region, mean_distance_profile,
                          pairwise_layer_distances)

STEP_DIRNAME = "Step-3_DkNN-and-Layer-Distance"


def main(model_key, dataset_key):
    cfg = load_config(model_key, dataset_key)
    data = load_aligned(cfg)
    print(f"[layer-distance] {model_key} x {dataset_key}: {len(data['pair_ids'])} pairs")

    hypo_reps = data["bank_reps"][data["hypo_rows"]]
    distances = pairwise_layer_distances(hypo_reps, data["para_reps"])
    inconsistent = data["inconsistent"].astype(bool)

    mean_consistent = mean_distance_profile(distances, ~inconsistent)
    mean_inconsistent = mean_distance_profile(distances, inconsistent)
    threshold = distance_threshold_per_layer(distances[~inconsistent])
    crossing = first_crossing_layer(mean_inconsistent, threshold)
    n_layers = distances.shape[1]
    region = layer_region(crossing, n_layers)

    out_dir = step_results_dir(STEP_DIRNAME, cfg)
    out_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out_dir / "layer_distances.npz",
                        distances=distances,
                        pair_ids=data["pair_ids"],
                        para_idx=data["para_idx"],
                        inconsistent=data["inconsistent"])
    profile = pd.DataFrame({
        "layer": np.arange(1, n_layers + 1),
        "mean_distance_consistent": mean_consistent,
        "mean_distance_inconsistent": mean_inconsistent,
        "threshold_consistent_mean_plus_std": threshold,
    })
    profile.to_csv(out_dir / "layer_profile.csv", index=False)

    summary = {
        "model": model_key,
        "dataset": dataset_key,
        "n_layers": int(n_layers),
        "crossing_layer": crossing,
        "region": region,
        "n_consistent": int((~inconsistent).sum()),
        "n_inconsistent": int(inconsistent.sum()),
    }
    with open(out_dir / "distance_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(f"[layer-distance] crossing layer: {crossing} -> region: {region}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--dataset", required=True)
    args = parser.parse_args()
    main(args.model, args.dataset)
