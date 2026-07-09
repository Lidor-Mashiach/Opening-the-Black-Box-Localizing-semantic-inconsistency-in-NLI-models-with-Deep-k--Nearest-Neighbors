"""Relaxed Fooling Rate (RFR).

Source: anchor paper - Arakelyan et al. (2024), EACL.
Definition: the fraction of (hypothesis, paraphrase) pairs whose prediction
changed in ANY way after paraphrasing (including moves to/from Neutral).
"""
import numpy as np


def relaxed_fooling_rate(hypothesis_preds, paraphrase_preds):
    """Fraction of rows where the paraphrase prediction differs at all.

    Parameters
    ----------
    hypothesis_preds : array-like of int
        Model prediction for the original (premise, hypothesis) pair.
        One entry per paraphrase row (repeat the hypothesis prediction
        for each of its paraphrases).
    paraphrase_preds : array-like of int
        Model prediction for the (premise, paraphrase) pair.

    Returns
    -------
    float in [0, 1]. Higher = more fooling = less consistent.
    """
    h = np.asarray(hypothesis_preds)
    p = np.asarray(paraphrase_preds)
    if h.shape != p.shape:
        raise ValueError(f"shape mismatch: {h.shape} vs {p.shape}")
    if h.size == 0:
        return 0.0
    return float(np.mean(h != p))
