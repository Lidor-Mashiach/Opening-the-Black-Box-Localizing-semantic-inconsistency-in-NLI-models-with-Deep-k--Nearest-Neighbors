"""Step-2: run every (premise, paraphrase) pair through the fine-tuned model.

Input - the DATASET-level paraphrase bank
    Datasets/<dataset>/paraphrases/paraphrase_bank.csv
    (pair_id, premise, hypothesis, paraphrase, label, para_idx - EXACTLY
    per_hypothesis rows per pooled hypothesis, model-independent), which is
    intersected here with THIS model's correct-only set: only rows whose
    pair_id appears in Runtime-Data/<MODEL>/<DATASET>/train_correct.csv
    (the reduced copy built by Step-1) enter, so
    "label == the model's hypothesis prediction" holds by construction and
    every model is tested on the exact same paraphrases for shared hypotheses.

Outputs - Runtime-Data/<MODEL>/<DATASET>/paraphrases_used.csv  (the per-model
          COPY of the shared bank: bank intersect this model's correct set)
        - Runtime-Data/<MODEL>/<DATASET>/paraphrases.npz  (per-layer reps)
        - Step-2 results/<COMBO>/paraphrase_predictions.csv   (predictions +
          consistent / strict-flip flags used by Steps 3-4)

When the bank does not exist yet the step SKIPS gracefully - build it ONCE
with setup-files/Paraphrase-Generator/generate_paraphrases.py (the main runner checks
this up front and exits with the exact command).
"""
import json

import numpy as np
import pandas as pd

from . import model_utils
from .config_loader import (encoded_dir, filtered_dir, load_config,
                            paraphrase_bank_csv, resolve_checkpoint,
                            step_results_dir)
from .logging_utils import log

STEP_DIRNAME = "Step-2_Paraphrase-Inference"
REQUIRED_COLUMNS = ("pair_id", "premise", "paraphrase", "label")


def run_paraphrase_inference(model_key, dataset_key):
    cfg = load_config(model_key, dataset_key)
    bank_path = paraphrase_bank_csv(cfg)
    if not bank_path.exists():
        print(f"[step-2] SKIP {model_key} x {dataset_key}: no paraphrase bank at\n"
              f"         {bank_path}\n"
              f"         Build it once with setup-files/Paraphrase-Generator/generate_paraphrases.py "
              f"(see Datasets/{cfg['dir']}/paraphrases/README.md).")
        return False
    filtered_path = filtered_dir(cfg) / "train_correct.csv"
    if not filtered_path.exists():
        raise FileNotFoundError(f"missing {filtered_path} - run Step-1 "
                                f"build_filtered_dataset.py first")

    df = pd.read_csv(bank_path)
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"{bank_path} is missing columns: {missing}")
    if "para_idx" not in df.columns:
        df["para_idx"] = df.groupby("pair_id").cumcount()

    # intersect the shared bank with THIS model's correct-only hypotheses
    correct_ids = set(pd.read_csv(filtered_path, usecols=["pair_id"])["pair_id"]
                      .astype(str).unique())
    pool_hypotheses = df["pair_id"].nunique()
    df = df[df["pair_id"].astype(str).isin(correct_ids)].reset_index(drop=True)
    log("INFER", f"shared bank: {pool_hypotheses} hypotheses -> "
        f"{df['pair_id'].nunique()} classified correctly by this model "
        f"({len(df)} paraphrase rows enter)", model_key, dataset_key)

    # materialize the per-model COPY of the paraphrase dataset (the shared
    # bank in Datasets/ is never modified) - Runtime-Data/<M>/<D>/
    used_csv = encoded_dir(cfg) / "paraphrases_used.csv"
    used_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(used_csv, index=False)
    log("INFER", f"per-model paraphrase copy written -> {used_csv}",
        model_key, dataset_key)

    ckpt = resolve_checkpoint(cfg)
    model, tokenizer, device = model_utils.load_model_and_tokenizer(cfg, checkpoint=ckpt)

    log("INFER", f"running (premise, paraphrase) inference: {len(df)} rows",
        model_key, dataset_key)
    preds = model_utils.predict(model, tokenizer, device,
                                df["premise"].tolist(), df["paraphrase"].tolist(),
                                cfg, desc="para-predict")
    reps = model_utils.extract_layer_representations(
        model, tokenizer, device,
        df["premise"].tolist(), df["paraphrase"].tolist(), cfg, desc="para-encode")

    labels = df["label"].to_numpy()
    consistent = preds == labels                       # hypothesis pred == gold by design
    strict_flip = ((labels == 0) & (preds == 2)) | ((labels == 2) & (preds == 0))

    out_encoded = encoded_dir(cfg)
    out_encoded.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out_encoded / "paraphrases.npz",
                        reps=reps,
                        preds=preds,
                        pair_ids=df["pair_id"].to_numpy(dtype=str),
                        para_idx=df["para_idx"].to_numpy())

    out_results = step_results_dir(STEP_DIRNAME, cfg)
    out_results.mkdir(parents=True, exist_ok=True)
    out_df = pd.DataFrame({
        "pair_id": df["pair_id"],
        "para_idx": df["para_idx"],
        "hypothesis_pred": labels,       # = gold label (correct-only design)
        "paraphrase_pred": preds,
        "consistent": consistent.astype(int),
        "strict_flip": strict_flip.astype(int),
    })
    out_df.to_csv(out_results / "paraphrase_predictions.csv", index=False)

    stats = {
        "model": model_key,
        "dataset": dataset_key,
        "paraphrase_rows": int(len(df)),
        "hypotheses": int(df["pair_id"].nunique()),
        "consistent_share": float(consistent.mean()),
        "strict_flip_share": float(strict_flip.mean()),
    }
    with open(out_results / "inference_stats.json", "w") as f:
        json.dump(stats, f, indent=2)
    log("INFER", f"consistent share: {stats['consistent_share']:.4f}",
        model_key, dataset_key)
    return True
