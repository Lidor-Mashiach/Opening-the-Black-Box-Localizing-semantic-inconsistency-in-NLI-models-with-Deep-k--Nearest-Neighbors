"""Strict Fooling Rate (SFR).

Source: anchor paper - Arakelyan et al. (2024), EACL.
Definition: the fraction of (hypothesis, paraphrase) pairs whose prediction
flipped COMPLETELY, i.e. Entailment <-> Contradiction. Moves involving
Neutral are NOT counted here (they are counted by the Relaxed rate).
"""
import numpy as np

ENTAILMENT = 0
CONTRADICTION = 2


def strict_fooling_rate(hypothesis_preds, paraphrase_preds,
                        entailment_label=ENTAILMENT,
                        contradiction_label=CONTRADICTION):
    """Fraction of rows with a full Entailment <-> Contradiction flip.

    Labels follow the HuggingFace convention for SNLI / MNLI / ANLI:
    0 = entailment, 1 = neutral, 2 = contradiction.
    """
    h = np.asarray(hypothesis_preds)
    p = np.asarray(paraphrase_preds)
    if h.shape != p.shape:
        raise ValueError(f"shape mismatch: {h.shape} vs {p.shape}")
    if h.size == 0:
        return 0.0
    e2c = (h == entailment_label) & (p == contradiction_label)
    c2e = (h == contradiction_label) & (p == entailment_label)
    return float(np.mean(e2c | c2e))
