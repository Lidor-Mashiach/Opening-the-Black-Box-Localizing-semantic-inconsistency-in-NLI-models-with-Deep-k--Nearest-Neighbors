"""Layer Distance - per-layer cosine geometry between representations.

For every model layer i, we take the representation of the original
Hypothesis and the representation of its Paraphrase and compute the cosine
distance between the two vectors (directly - no neighbours involved).
Doing this for every layer yields a "distance profile" along the model.

Key methodology rules implemented here:
* The "too far" threshold is NOT arbitrary - it is derived from the data:
  per layer, mean + 1 std of the distances of the CONSISTENT pairs only.
* Distances are computed for consistent and inconsistent pairs alike;
  the comparison between the two groups is the finding.
* The layer where the inconsistent-mean first crosses the threshold points
  to the problem region: lower third -> representation problem,
  middle third -> reasoning problem, upper third -> classifier problem.
"""
import numpy as np


def pairwise_layer_distances(reps_a, reps_b):
    """Cosine distance per layer for aligned representation tensors.

    Parameters
    ----------
    reps_a, reps_b : np.ndarray of shape [n_pairs, n_layers, dim]
        e.g. hypothesis representations and paraphrase representations,
        aligned row-by-row.

    Returns
    -------
    np.ndarray of shape [n_pairs, n_layers] with values in [0, 2].
    """
    a = np.asarray(reps_a, dtype=np.float32)
    b = np.asarray(reps_b, dtype=np.float32)
    if a.shape != b.shape:
        raise ValueError(f"shape mismatch: {a.shape} vs {b.shape}")
    a_norm = a / (np.linalg.norm(a, axis=-1, keepdims=True) + 1e-12)
    b_norm = b / (np.linalg.norm(b, axis=-1, keepdims=True) + 1e-12)
    cosine_sim = np.sum(a_norm * b_norm, axis=-1)
    return 1.0 - cosine_sim


def mean_distance_profile(distances, mask=None):
    """Mean distance per layer, optionally over a boolean row mask.

    distances : [n_pairs, n_layers];  mask : [n_pairs] bool or None.
    Returns [n_layers].
    """
    d = np.asarray(distances)
    if mask is not None:
        d = d[np.asarray(mask, dtype=bool)]
    if d.shape[0] == 0:
        return np.full(d.shape[1] if d.ndim == 2 else 0, np.nan)
    return d.mean(axis=0)


def distance_threshold_per_layer(consistent_distances):
    """Per-layer threshold = mean + 1 std, computed on CONSISTENT pairs only.

    A pair whose distance exceeds this threshold at a given layer is
    considered to have "moved" at that layer. The threshold is derived from
    the data itself, never set manually.
    """
    d = np.asarray(consistent_distances)
    return d.mean(axis=0) + d.std(axis=0)


def first_crossing_layer(inconsistent_mean_profile, threshold_per_layer):
    """First layer (1-indexed) where the inconsistent mean crosses the threshold.

    Returns None when no crossing occurs.
    """
    profile = np.asarray(inconsistent_mean_profile)
    thr = np.asarray(threshold_per_layer)
    above = np.where(profile > thr)[0]
    if above.size == 0:
        return None
    return int(above[0]) + 1  # human-friendly 1-indexed layer number


def layer_region(layer_number, n_layers):
    """Map a 1-indexed layer number to its diagnostic region.

    Lower third  -> 'representation' (encoded far apart from the start)
    Middle third -> 'reasoning'      (faulty inference mid-model)
    Upper third  -> 'classifier'     (understood alike, final decision differs)
    """
    if layer_number is None:
        return None
    third = n_layers / 3.0
    if layer_number <= third:
        return "representation"
    if layer_number <= 2 * third:
        return "reasoning"
    return "classifier"
