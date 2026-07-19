"""Deterministic, seeded sampling - the glue that keeps the paraphrase pool,
the per-model banks and the analysis aligned.

Chain:

    raw train split (per DATASET, model-independent)
        -> deterministic_pool_sample()      paraphrases.pool_size hypotheses (null = all)
            -> generate + verify            up to per_hypothesis each (partial kept)
            -> paraphrase_bank.csv          the static, shareable asset

    filtered train_correct.csv (per MODEL)
        -> deterministic_bank_subset(must_include_ids = pool intersect filtered)
            -> encoded DkNN bank            pool hypotheses the model got right
                                            are ALWAYS bank members

Because the pool hypotheses are forced into every model's bank, Step-3
alignment never drops a paraphrased row.
"""


def deterministic_pool_sample(df, cfg):
    """The dataset-level paraphrase pool (paraphrases.pool_size); seeded,
    original row order preserved."""
    size = cfg["paraphrases"]["pool_size"]
    if size is None or len(df) <= size:
        return df
    return df.sample(n=size, random_state=cfg["seed"]).sort_index()


def deterministic_bank_subset(df, cfg, must_include_ids=None):
    """Cap the DkNN bank (sampling.max_bank_examples); seeded, order-stable.

    must_include_ids : optional set of pair_id strings that are ALWAYS kept
        (the dataset's paraphrase-pool hypotheses). The remaining budget is
        filled with a seeded sample of the other rows.

    Returns (subset_df, capped: bool, n_forced: int).
    """
    cap = cfg["sampling"]["max_bank_examples"]
    ids = df["pair_id"].astype(str)
    if must_include_ids:
        forced_mask = ids.isin(set(must_include_ids))
        n_forced = int(forced_mask.sum())
    else:
        forced_mask = None
        n_forced = 0

    if cap is None or len(df) <= cap:
        return df, False, n_forced

    if n_forced == 0:
        return df.sample(n=cap, random_state=cfg["seed"]).sort_index(), True, 0

    forced_df = df[forced_mask]
    rest = df[~forced_mask]
    remaining = max(cap - n_forced, 0)
    fill = rest.sample(n=min(remaining, len(rest)), random_state=cfg["seed"])
    import pandas as pd
    subset = pd.concat([forced_df, fill]).sort_index()
    return subset, True, n_forced
