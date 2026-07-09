"""Deep k-Nearest Neighbors (DkNN) Credibility.

Source: Papernot & McDaniel (2018), "Deep k-Nearest Neighbors: Towards
Confident, Interpretable and Robust Deep Learning" (arXiv:1803.04765).

How Credibility is computed (exactly as in the methodology document):
1. A "bank" stores the representation of every training Hypothesis at every
   layer, together with its label.
2. For a query example, at EVERY layer we find its K nearest neighbours in
   the bank (cosine distance) and collect their labels.
3. Credibility = (# neighbours, across ALL layers, whose label equals the
   query's FINAL prediction) / (K * n_layers).
   The final prediction is used whether or not it is correct.

Extra rules implemented here:
* Hypotheses are themselves part of the bank, so when scoring a hypothesis
  we exclude its own bank entry (otherwise it is its own nearest neighbour
  at distance 0 and credibility is inflated). Use `exclude_self_indices`.
* K is a hyper-parameter chosen PER (model, dataset) pair on a validation
  set: the K whose credibility-drop correlates most strongly with actual
  inconsistency wins (`select_best_k`).
* "Low credibility" is never an absolute threshold - it is defined on the
  distribution of drops: drop = Credibility(Hypothesis) - Credibility(Paraphrase).
  A positive drop means the paraphrase's credibility went DOWN relative to
  its hypothesis; a drop is anomalous when it exceeds mean + 1 std of all
  drops for that (model, dataset) pair (`anomalous_drop_threshold`).

Implementation note: pure NumPy (normalized matmul + argpartition), no
external ANN library, so behaviour is fully transparent and reproducible.
"""
import numpy as np


class DkNN:
    """Layer-wise nearest-neighbour search over a labeled representation bank."""

    def __init__(self):
        self._banks = None      # list of [n_bank, dim] L2-normalized, per layer
        self.labels = None      # [n_bank]
        self.n_layers = None

    def fit(self, bank_reps, bank_labels):
        """Store the bank.

        bank_reps   : np.ndarray [n_bank, n_layers, dim]
        bank_labels : np.ndarray [n_bank] of int labels
        """
        reps = np.asarray(bank_reps, dtype=np.float32)
        self.labels = np.asarray(bank_labels)
        if reps.shape[0] != self.labels.shape[0]:
            raise ValueError("bank_reps and bank_labels row counts differ")
        self.n_layers = reps.shape[1]
        norms = np.linalg.norm(reps, axis=-1, keepdims=True) + 1e-12
        reps = reps / norms
        # one contiguous [n_bank, dim] matrix per layer for fast matmul
        self._banks = [np.ascontiguousarray(reps[:, i, :]) for i in range(self.n_layers)]
        return self

    def top_neighbor_labels(self, query_reps, k, exclude_self_indices=None,
                            chunk_size=512):
        """Labels of the K nearest bank neighbours, per query, per layer.

        Parameters
        ----------
        query_reps : np.ndarray [n_query, n_layers, dim]
        k : int
        exclude_self_indices : np.ndarray [n_query] of int, optional
            Bank row index of each query itself (when the queries ARE bank
            members, e.g. scoring the training hypotheses). That row is
            excluded from its own neighbour list.
        chunk_size : int
            Queries are processed in chunks to bound memory.

        Returns
        -------
        np.ndarray [n_query, n_layers, k] of neighbour labels
        (ordered nearest-first).
        """
        q = np.asarray(query_reps, dtype=np.float32)
        if q.ndim != 3 or q.shape[1] != self.n_layers:
            raise ValueError(f"expected [n, {self.n_layers}, dim], got {q.shape}")
        q = q / (np.linalg.norm(q, axis=-1, keepdims=True) + 1e-12)
        n_query = q.shape[0]
        n_bank = self.labels.shape[0]
        fetch = k + 1 if exclude_self_indices is not None else k
        if fetch > n_bank:
            raise ValueError(f"k={k} too large for bank of size {n_bank}")

        out = np.empty((n_query, self.n_layers, k), dtype=self.labels.dtype)
        for start in range(0, n_query, chunk_size):
            end = min(start + chunk_size, n_query)
            for layer in range(self.n_layers):
                # cosine similarity = dot product of L2-normalized vectors
                sims = q[start:end, layer, :] @ self._banks[layer].T  # [chunk, n_bank]
                # top-`fetch` by similarity (unordered), then order nearest-first
                idx = np.argpartition(-sims, fetch - 1, axis=1)[:, :fetch]
                row_sims = np.take_along_axis(sims, idx, axis=1)
                order = np.argsort(-row_sims, axis=1)
                idx = np.take_along_axis(idx, order, axis=1)  # [chunk, fetch]
                if exclude_self_indices is not None:
                    self_idx = np.asarray(exclude_self_indices)[start:end, None]
                    keep = idx != self_idx            # drop the query itself
                    cleaned = np.empty((idx.shape[0], k), dtype=idx.dtype)
                    for r in range(idx.shape[0]):
                        cleaned[r] = idx[r][keep[r]][:k]
                    idx = cleaned
                out[start:end, layer, :] = self.labels[idx]
        return out


def credibility_from_neighbor_labels(neighbor_labels, final_preds, k=None):
    """Credibility = supporters of the final prediction / total neighbours.

    neighbor_labels : [n_query, n_layers, k_max] (from `top_neighbor_labels`)
    final_preds     : [n_query] final model prediction per query
    k               : use only the first k neighbours per layer (allows
                      computing several K values from one neighbour fetch).

    Returns np.ndarray [n_query] of credibility values in [0, 1].
    """
    labels = np.asarray(neighbor_labels)
    preds = np.asarray(final_preds).reshape(-1, 1, 1)
    if k is not None:
        labels = labels[:, :, :k]
    supporters = (labels == preds).sum(axis=(1, 2))
    total = labels.shape[1] * labels.shape[2]
    return supporters / float(total)


def credibility_drop(cred_hypothesis, cred_paraphrase):
    """drop = Credibility(Hypothesis) - Credibility(Paraphrase).

    Positive drop  -> the paraphrase's credibility DECREASED relative to its
                      hypothesis (candidate generalization problem).
    Negative drop  -> the paraphrase's credibility INCREASED (candidate
                      representation problem - it landed in a region that
                      strongly supports a different label).
    """
    return np.asarray(cred_hypothesis) - np.asarray(cred_paraphrase)


def anomalous_drop_threshold(drops):
    """Data-derived threshold for an anomalous credibility drop.

    A drop is anomalous when it exceeds mean(drops) + std(drops), computed
    separately for every (model, dataset) pair - never a fixed constant.
    """
    d = np.asarray(drops, dtype=np.float64)
    return float(d.mean() + d.std())


def select_best_k(drops_by_k, inconsistent_flags):
    """Choose K per (model, dataset) pair on the validation set.

    drops_by_k         : dict {k: np.ndarray of credibility drops}
    inconsistent_flags : np.ndarray of {0,1} - did the prediction change?

    The chosen K maximizes the (point-biserial / Pearson) correlation between
    the credibility drop and actual inconsistency: larger drops should mark
    the pairs that actually became inconsistent.

    Returns (best_k, {k: correlation}).
    """
    y = np.asarray(inconsistent_flags, dtype=np.float64)
    correlations = {}
    for k, drops in drops_by_k.items():
        x = np.asarray(drops, dtype=np.float64)
        if x.std() == 0 or y.std() == 0:
            correlations[k] = 0.0
        else:
            correlations[k] = float(np.corrcoef(x, y)[0, 1])
    best_k = max(correlations, key=lambda kk: correlations[kk])
    return best_k, correlations


def diagnose_credibility_scenario(cred_hypothesis, cred_paraphrase):
    """Apply the three methodology scenarios on group means.

    Let m = mean(cred_hypo), s = std(cred_hypo), p = mean(cred_para):
      p < m - s          -> 'generalization'  (paraphrase credibility dropped:
                            its region is poorly covered by the training set)
      p > m + s          -> 'representation'  (credibility rose for the wrong
                            reason: strong support for a different label)
      m - s <= p <= m+s  -> 'within_normal_range' (if answers still flipped,
                            this points at a classifier problem)
    """
    ch = np.asarray(cred_hypothesis, dtype=np.float64)
    cp = np.asarray(cred_paraphrase, dtype=np.float64)
    m, s, p = ch.mean(), ch.std(), cp.mean()
    if p < m - s:
        scenario = "generalization"
    elif p > m + s:
        scenario = "representation"
    else:
        scenario = "within_normal_range"
    return {
        "scenario": scenario,
        "mean_cred_hypothesis": float(m),
        "std_cred_hypothesis": float(s),
        "mean_cred_paraphrase": float(p),
    }
