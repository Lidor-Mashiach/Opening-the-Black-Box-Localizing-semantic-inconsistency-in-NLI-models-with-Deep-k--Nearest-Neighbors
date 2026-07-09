"""Meta-runner for eval_metrics: tiny synthetic tests for every metric.

Run:  python eval_metrics/sanity_check.py   (from the repository root)
No GPU, no downloads - pure NumPy toy data. All checks must print OK.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np

from eval_metrics import (
    relaxed_fooling_rate, strict_fooling_rate,
    paraphrastic_consistency, pattern_accuracy,
    pairwise_layer_distances, mean_distance_profile,
    distance_threshold_per_layer, first_crossing_layer, layer_region,
    DkNN, credibility_from_neighbor_labels, credibility_drop,
    anomalous_drop_threshold, select_best_k, diagnose_credibility_scenario,
)


def check(name, condition):
    status = "OK " if condition else "FAIL"
    print(f"[{status}] {name}")
    if not condition:
        sys.exit(1)


def main():
    rng = np.random.default_rng(42)

    # --- consistency metrics -------------------------------------------
    hyp = np.array([0, 0, 1, 2, 2, 0])
    par = np.array([0, 2, 1, 2, 0, 1])   # changes at rows 1, 4, 5
    check("relaxed_fooling_rate == 3/6", np.isclose(relaxed_fooling_rate(hyp, par), 0.5))
    check("strict_fooling_rate == 2/6 (only E<->C)", np.isclose(strict_fooling_rate(hyp, par), 2 / 6))
    check("paraphrastic_consistency == 1 - RFR", np.isclose(paraphrastic_consistency(hyp, par), 0.5))
    ids = np.array(["a", "a", "b", "b", "c", "c"])
    check("pattern_accuracy: only group 'b' fully consistent",
          np.isclose(pattern_accuracy(ids, hyp, par), 1 / 3))

    # --- layer distance -------------------------------------------------
    n_pairs, n_layers, dim = 40, 12, 16
    base = rng.normal(size=(n_pairs, n_layers, dim))
    close = base + rng.normal(scale=0.01, size=base.shape)          # consistent
    far = base.copy()
    far[:, 6:, :] = rng.normal(size=(n_pairs, n_layers - 6, dim))   # drift from layer 7
    d_cons = pairwise_layer_distances(base, close)
    d_incons = pairwise_layer_distances(base, far)
    thr = distance_threshold_per_layer(d_cons)
    cross = first_crossing_layer(mean_distance_profile(d_incons), thr)
    check("distances are ~0 for near-identical vectors", d_cons.mean() < 0.01)
    check(f"crossing layer detected at layer {cross} (expected 7)", cross == 7)
    check("layer 7/12 maps to the 'reasoning' region", layer_region(cross, n_layers) == "reasoning")
    check("layer 2/12 -> representation, 11/12 -> classifier",
          layer_region(2, 12) == "representation" and layer_region(11, 12) == "classifier")

    # --- DkNN credibility ------------------------------------------------
    # Bank: two well-separated label clusters at every layer.
    n_per_class, k = 30, 5
    cluster0 = rng.normal(loc=+3.0, size=(n_per_class, n_layers, dim))
    cluster1 = rng.normal(loc=-3.0, size=(n_per_class, n_layers, dim))
    bank = np.concatenate([cluster0, cluster1])
    labels = np.array([0] * n_per_class + [1] * n_per_class)
    dknn = DkNN().fit(bank, labels)

    query_in_0 = rng.normal(loc=+3.0, size=(4, n_layers, dim))
    nl = dknn.top_neighbor_labels(query_in_0, k)
    cred_right = credibility_from_neighbor_labels(nl, np.zeros(4, dtype=int))
    cred_wrong = credibility_from_neighbor_labels(nl, np.ones(4, dtype=int))
    check("credibility ~1 when final prediction matches the region", cred_right.mean() > 0.95)
    check("credibility ~0 when final prediction contradicts the region", cred_wrong.mean() < 0.05)

    # self-exclusion: scoring bank members must not count themselves
    nl_self = dknn.top_neighbor_labels(bank[:4], k, exclude_self_indices=np.arange(4))
    check("self-exclusion returns k neighbours per layer", nl_self.shape == (4, n_layers, k))

    # slicing one big fetch into several K values
    nl_big = dknn.top_neighbor_labels(query_in_0, 10)
    c5 = credibility_from_neighbor_labels(nl_big, np.zeros(4, dtype=int), k=5)
    check("k-slicing from a single neighbour fetch works", c5.shape == (4,))

    # --- drop, threshold, K selection, scenario --------------------------
    cred_h = np.array([0.9, 0.9, 0.9, 0.9])
    cred_p = np.array([0.85, 0.4, 0.88, 0.9])
    drops = credibility_drop(cred_h, cred_p)
    check("credibility_drop is positive when paraphrase credibility falls", drops[1] > 0.4)
    check("anomalous_drop_threshold = mean + std",
          np.isclose(anomalous_drop_threshold(drops), drops.mean() + drops.std()))

    incons = np.array([0, 1, 0, 1] * 25)
    good_k_drops = incons * 0.5 + rng.normal(scale=0.01, size=100)   # correlated
    bad_k_drops = rng.normal(scale=0.5, size=100)                    # noise
    best_k, corrs = select_best_k({10: good_k_drops, 50: bad_k_drops}, incons)
    check(f"select_best_k picked the correlated K (chose {best_k})", best_k == 10)

    scen = diagnose_credibility_scenario(cred_hypothesis=np.full(50, 0.9),
                                         cred_paraphrase=np.full(50, 0.3))
    check("scenario 'generalization' detected on a clear drop",
          scen["scenario"] == "generalization")

    print("\nAll eval_metrics sanity checks passed.")


if __name__ == "__main__":
    main()
