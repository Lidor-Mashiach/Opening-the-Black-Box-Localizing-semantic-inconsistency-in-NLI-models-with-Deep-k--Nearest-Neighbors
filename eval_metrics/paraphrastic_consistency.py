"""Paraphrastic Consistency (PC).

Source: Srikanth, Carpuat & Rudinger (2024),
"How Often Are Errors in Natural Language Reasoning Due to Paraphrastic
Variability?" (TACL 12).

Definition used here: the probability that the model gives the SAME answer
to a (premise, hypothesis) pair and to its (premise, paraphrase) pair.
Note: PC is the complement of the Relaxed Fooling Rate at the pair level
(PC = 1 - RFR); it is kept as a separate file because it is a separate,
named metric in the literature and is reported separately.
"""
import numpy as np


def paraphrastic_consistency(hypothesis_preds, paraphrase_preds):
    """Probability that the prediction is preserved under paraphrasing."""
    h = np.asarray(hypothesis_preds)
    p = np.asarray(paraphrase_preds)
    if h.shape != p.shape:
        raise ValueError(f"shape mismatch: {h.shape} vs {p.shape}")
    if h.size == 0:
        return 1.0
    return float(np.mean(h == p))
