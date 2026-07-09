"""Pattern Accuracy (PA).

Source: Zgreaban, Deoskar & Abzianidze (2025), MERGE (arXiv:2510.24295).

Definition used here: paraphrases are grouped by their source hypothesis
(~5 paraphrases per hypothesis). A hypothesis scores 1 only if EVERY one of
its paraphrases receives the same prediction as the original hypothesis.
Because only hypotheses the model classified correctly enter the experiment,
"same as the original" is also "correct". PA is stricter than PC: one bad
paraphrase fails the whole group.
"""
import numpy as np


def pattern_accuracy(pair_ids, hypothesis_preds, paraphrase_preds):
    """Share of hypothesis groups in which ALL paraphrases stayed consistent.

    Parameters
    ----------
    pair_ids : array-like of str
        Identifier of the source hypothesis, one entry per paraphrase row.
    hypothesis_preds : array-like of int
        Prediction of the original hypothesis, repeated per paraphrase row.
    paraphrase_preds : array-like of int
        Prediction of each paraphrase.

    Returns
    -------
    float in [0, 1] - the fraction of fully-consistent groups.
    """
    ids = np.asarray(pair_ids)
    h = np.asarray(hypothesis_preds)
    p = np.asarray(paraphrase_preds)
    if not (ids.shape == h.shape == p.shape):
        raise ValueError("pair_ids, hypothesis_preds, paraphrase_preds must align")
    if ids.size == 0:
        return 1.0
    consistent = (h == p)
    group_ok = {}
    for gid, ok in zip(ids, consistent):
        group_ok[gid] = group_ok.get(gid, True) and bool(ok)
    return float(np.mean(list(group_ok.values())))
