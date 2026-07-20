"""Unified loading of SNLI / MNLI / ANLI from the HuggingFace hub.

All datasets come out with the same schema: premise, hypothesis, label
(0 = entailment, 1 = neutral, 2 = contradiction), plus 'round' for ANLI.
Rows without a gold label (label == -1, present in SNLI/MNLI) are dropped.
The three datasets are NEVER merged - each keeps its own population.
"""
from datasets import Dataset, concatenate_datasets, load_dataset

from .config_loader import dataset_dir

_KEEP_COLUMNS = ("premise", "hypothesis", "label", "round")


def _raw_parquet(cfg, split):
    """The frozen local snapshot path written by download_datasets.py."""
    return dataset_dir(cfg) / "raw" / f"{split}.parquet"


def _load_from_hub(cfg, split):
    """Original hub path: pull + curate. Used to CREATE the parquet snapshot
    (download_datasets.py) and as the fallback when the snapshot is absent."""
    hf_split = cfg["splits"][split]
    if "rounds" in cfg:  # ANLI: concatenate the rounds, keep a 'round' column
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
    return ds.filter(lambda ex: ex["label"] in (0, 1, 2))


def load_nli(cfg, split):
    """Load one logical split ('train'/'validation'/'test') per the config.

    Reads the FROZEN LOCAL PARQUET snapshot at Datasets/<dir>/raw/<split>.parquet
    (written once by setup-files/download_datasets.py) when it exists: it is
    fully local and reproducible and - unlike load_dataset() - needs no
    HuggingFace hub round-trip on every call (no per-call metadata check, no
    auth/rate-limit warning, works offline), so every pipeline stage starts
    faster. Falls back to the hub only when the snapshot is missing, so a
    forgotten download still runs; materialise the snapshot once with
    download_datasets.py to get the local-only fast path everywhere.

    The logical -> HuggingFace split mapping lives in configs/datasets/*.yaml
    (e.g. MNLI maps test -> validation_mismatched). For ANLI the three rounds
    are concatenated and a 'round' column is kept.
    """
    parquet = _raw_parquet(cfg, split)
    if parquet.exists():
        ds = Dataset.from_parquet(str(parquet))
        # The snapshot is already curated (label filtered, columns trimmed) and
        # carries pair_id; load_nli's contract returns WITHOUT pair_id (callers
        # add it, re-derived identically from the same deterministic row order).
        drop = [c for c in ds.column_names if c not in _KEEP_COLUMNS]
        if drop:
            ds = ds.remove_columns(drop)
        return ds
    print(f"[data] {cfg['dataset_key']}/{split}: local parquet snapshot missing "
          f"- reading from the HF hub (run setup-files/download_datasets.py once "
          f"to cache it locally and skip the hub round-trip)", flush=True)
    return _load_from_hub(cfg, split)


def add_pair_ids(ds, cfg, split):
    """Attach a stable, deterministic identifier to every row.

    Loading + the label filter are deterministic, so the enumeration is
    reproducible across runs and machines.
    """
    ids = [f"{cfg['dataset_key']}-{split}-{i:07d}" for i in range(len(ds))]
    return ds.add_column("pair_id", ids)