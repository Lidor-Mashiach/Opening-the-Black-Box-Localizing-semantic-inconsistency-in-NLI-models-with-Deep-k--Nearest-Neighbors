"""Deterministic, seeded sampling - shared by the bank encoder and the
paraphrase generator so both always agree on WHICH hypotheses are in play.

Chain of subsets (each seeded, each preserving original row order):

    filtered train_correct.csv
        -> deterministic_bank_subset()   (sampling.max_bank_examples)
            -> deterministic_eval_sample()  (sampling.eval_sample_size)

Because the eval sample is drawn FROM the bank subset, every hypothesis that
receives paraphrases is guaranteed to exist in the encoded bank - which
Step-3's alignment relies on (its own hypothesis must be a bank member for
self-exclusion and for Credibility(Hypothesis)).
"""


def deterministic_bank_subset(df, cfg):
    """Cap the DkNN bank (sampling.max_bank_examples); seeded, order-stable."""
    cap = cfg["sampling"]["max_bank_examples"]
    if cap is None or len(df) <= cap:
        return df, False
    return df.sample(n=cap, random_state=cfg["seed"]).sort_index(), True


def deterministic_eval_sample(df, cfg):
    """Pick the hypotheses that receive paraphrases (sampling.eval_sample_size).

    MUST be called on the output of deterministic_bank_subset() - see module
    docstring. Seeded with seed + 1 so it is independent of the bank draw.
    """
    size = cfg["sampling"]["eval_sample_size"]
    if size is None or len(df) <= size:
        return df
    return df.sample(n=size, random_state=cfg["seed"] + 1).sort_index()
