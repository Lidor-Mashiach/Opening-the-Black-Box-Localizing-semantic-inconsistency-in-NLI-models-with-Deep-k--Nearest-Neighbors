"""Step-1c: build the layer-wise DkNN representation bank.

For every hypothesis in the filtered ('correct only') set we store its pooled
representation at EVERY layer, together with its label - this is the labeled
bank Deep k-NN searches at credibility time.

Scale note: SNLI/MNLI have hundreds of thousands of correct train rows;
storing all of them at every layer is unnecessary for k-NN and very heavy on
disk/RAM. sampling.max_bank_examples (configs/base.yaml) caps the bank with a
seeded, deterministic subsample; set it to null to keep everything.
"""
import json

import numpy as np
import pandas as pd

from . import model_utils
from .config_loader import (encoded_dir, filtered_dir, load_config,
                            resolve_checkpoint)
from .sampling import deterministic_bank_subset


def encode_hypothesis_bank(model_key, dataset_key, force=False):
    cfg = load_config(model_key, dataset_key)
    out_npz = encoded_dir(cfg) / "hypotheses.npz"
    if out_npz.exists() and not force:
        print(f"[encode] SKIP {model_key} x {dataset_key}: {out_npz} exists "
              f"(use --force after retraining)")
        return None
    ckpt = resolve_checkpoint(cfg)
    csv_path = filtered_dir(cfg) / "train_correct.csv"
    if not csv_path.exists():
        raise FileNotFoundError(f"missing {csv_path} - run build_filtered_dataset.py first")

    df = pd.read_csv(csv_path)
    df, capped = deterministic_bank_subset(df, cfg)
    if capped:
        print(f"[encode] bank capped to {len(df)} rows (seeded, deterministic)")

    print(f"[encode] {model_key} on {dataset_key}: {len(df)} hypotheses")
    model, tokenizer, device = model_utils.load_model_and_tokenizer(cfg, checkpoint=ckpt)
    reps = model_utils.extract_layer_representations(
        model, tokenizer, device,
        df["premise"].tolist(), df["hypothesis"].tolist(), cfg, desc="bank")

    out_dir = encoded_dir(cfg)
    out_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out_dir / "hypotheses.npz",
                        reps=reps,
                        labels=df["label"].to_numpy(),
                        pair_ids=df["pair_id"].to_numpy(dtype=str))

    meta = {
        "model": model_key,
        "dataset": dataset_key,
        "n_examples": int(reps.shape[0]),
        "n_layers": int(reps.shape[1]),
        "dim": int(reps.shape[2]),
        "dtype": str(reps.dtype),
        "bank_capped": bool(capped),
    }
    with open(out_dir / "hypotheses_meta.json", "w") as f:
        json.dump(meta, f, indent=2)
    print(f"[encode] bank saved: {meta['n_examples']} x {meta['n_layers']} layers x dim {meta['dim']}")
    return meta
