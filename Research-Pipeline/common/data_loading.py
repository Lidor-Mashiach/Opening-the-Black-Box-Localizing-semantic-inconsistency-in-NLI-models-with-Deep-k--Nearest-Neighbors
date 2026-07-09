"""Unified loading of SNLI / MNLI / ANLI from the HuggingFace hub.

All datasets come out with the same schema: premise, hypothesis, label
(0 = entailment, 1 = neutral, 2 = contradiction), plus 'round' for ANLI.
Rows without a gold label (label == -1, present in SNLI/MNLI) are dropped.
The three datasets are NEVER merged - each keeps its own population.
"""
from datasets import concatenate_datasets, load_dataset

_KEEP_COLUMNS = ("premise", "hypothesis", "label", "round")


def load_nli(cfg, split):
    """Load one logical split ('train' / 'validation' / 'test') per the config.

    The logical -> HuggingFace split mapping lives in configs/datasets/*.yaml
    (e.g. MNLI maps test -> validation_mismatched). For ANLI the three rounds
    are concatenated and a 'round' column is kept.
    """
    hf_split = cfg["splits"][split]
    if "rounds" in cfg:  # ANLI
        parts = []
        for r in cfg["rounds"]:
            part = load_dataset(cfg["hf_id"], split=f"{hf_split}_{r}")
            part = part.add_column("round", [r] * len(part))
            parts.append(part)
        ds = concatenate_datasets(parts)
    else:
        ds = load_dataset(cfg["hf_id"], split=hf_split)

    drop = [c for c in ds.column_names if c not in _KEEP_COLUMNS]
    if drop:
        ds = ds.remove_columns(drop)
    ds = ds.filter(lambda ex: ex["label"] in (0, 1, 2))
    return ds


def add_pair_ids(ds, cfg, split):
    """Attach a stable, deterministic identifier to every row.

    Loading + the label filter are deterministic, so the enumeration is
    reproducible across runs and machines.
    """
    ids = [f"{cfg['dataset_key']}-{split}-{i:07d}" for i in range(len(ds))]
    return ds.add_column("pair_id", ids)
