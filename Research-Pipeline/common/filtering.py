"""Step-1b: build the per-model 'correct only' training subset.

Only (premise, hypothesis) rows the fine-tuned model classifies CORRECTLY on
the train split continue to the experiment - so we later measure sensitivity
to rephrasing, not ordinary mistakes. Every model therefore produces its own
filtered dataset for every dataset; it is saved as a NEW COPY under
Runtime-Data/<MODEL>/<DATASET>/train_correct.csv - the original dataset in
Datasets/ is never modified.
"""
import json

import numpy as np
import pandas as pd

from . import data_loading, model_utils
from .config_loader import (filtered_dir, load_config,
                            resolve_checkpoint, step1_results_dir)
from .logging_utils import log


def build_filtered_dataset(model_key, dataset_key):
    """Create the reduced COPY of the original dataset - correct rows only.

    The original dataset is never modified: this writes a NEW file inside
    the combination's Runtime-Data folder (wiped fresh at run start).
    """
    cfg = load_config(model_key, dataset_key)
    ckpt = resolve_checkpoint(cfg)

    log("FILTER", "building the reduced correct-only dataset copy", model_key, dataset_key)
    train = data_loading.load_nli(cfg, "train")
    train = data_loading.add_pair_ids(train, cfg, "train")
    model, tokenizer, device = model_utils.load_model_and_tokenizer(cfg, checkpoint=ckpt)
    preds = model_utils.predict(model, tokenizer, device,
                                train["premise"], train["hypothesis"], cfg, desc="filter")

    labels = np.asarray(train["label"])
    keep = preds == labels

    columns = {
        "pair_id": np.asarray(train["pair_id"])[keep],
        "premise": np.asarray(train["premise"])[keep],
        "hypothesis": np.asarray(train["hypothesis"])[keep],
        "label": labels[keep],
    }
    if "round" in train.column_names:
        columns["round"] = np.asarray(train["round"])[keep]
    df = pd.DataFrame(columns)

    out_dir = filtered_dir(cfg)
    out_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_dir / "train_correct.csv", index=False)

    stats = {
        "model": model_key,
        "dataset": dataset_key,
        "total_train_rows": int(len(train)),
        "kept_correct_rows": int(keep.sum()),
        "train_accuracy": float(keep.mean()),
        "output_csv": str(out_dir / "train_correct.csv"),
    }
    for target in (out_dir / "filter_stats.json",
                   step1_results_dir(cfg) / "filter_stats.json"):
        target.parent.mkdir(parents=True, exist_ok=True)
        with open(target, "w") as f:
            json.dump(stats, f, indent=2)
    log("FILTER", f"kept {stats['kept_correct_rows']}/{stats['total_train_rows']} "
        f"rows (train accuracy {stats['train_accuracy']:.4f}) -> {out_dir / 'train_correct.csv'}",
        model_key, dataset_key)
    return df
