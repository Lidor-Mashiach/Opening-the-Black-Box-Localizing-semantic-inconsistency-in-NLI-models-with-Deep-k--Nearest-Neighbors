"""Step-2: run every (premise, paraphrase) pair through the fine-tuned model.

Inputs  - Datasets/<dataset>/paraphrases/<MODEL>__paraphrases.csv with columns:
              pair_id   : id of the source hypothesis (must exist in the bank)
              premise   : the ORIGINAL premise (unchanged)
              paraphrase: rewording of the hypothesis, SAME logical relation
              label     : the gold label (= the hypothesis prediction, because
                          only correctly-classified hypotheses enter)
              para_idx  : optional 0..4 index (created here when absent)
Outputs - Encoded_Datasets/<MODEL>/<DATASET>/paraphrases.npz  (per-layer reps)
        - Step-2 results/<COMBO>/paraphrase_predictions.csv   (predictions +
          consistent / strict-flip flags used by Steps 3-4)

When the paraphrase CSV does not exist yet the step SKIPS gracefully - the
acquisition protocol is documented in Datasets/<dataset>/paraphrases and in
the root README ("Paraphrase Data Status").
"""
import json

import numpy as np
import pandas as pd

from . import model_utils
from .config_loader import (encoded_dir, load_config, resolve_checkpoint,
                            paraphrases_csv, step_results_dir)

STEP_DIRNAME = "Step-2_Paraphrase-Inference"
REQUIRED_COLUMNS = ("pair_id", "premise", "paraphrase", "label")


def run_paraphrase_inference(model_key, dataset_key, force=False):
    cfg = load_config(model_key, dataset_key)
    done_npz = encoded_dir(cfg) / "paraphrases.npz"
    done_csv = step_results_dir(STEP_DIRNAME, cfg) / "paraphrase_predictions.csv"
    if done_npz.exists() and done_csv.exists() and not force:
        print(f"[step-2] SKIP {model_key} x {dataset_key}: outputs exist "
              f"(use --force after retraining or regenerating paraphrases)")
        return True
    csv_path = paraphrases_csv(cfg)
    if not csv_path.exists():
        print(f"[step-2] SKIP {model_key} x {dataset_key}: no paraphrase file at\n"
              f"         {csv_path}\n"
              f"         See Datasets/{cfg['dir']}/paraphrases/README.md for the "
              f"acquisition protocol.")
        return False

    df = pd.read_csv(csv_path)
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"{csv_path} is missing columns: {missing}")
    if "para_idx" not in df.columns:
        df["para_idx"] = df.groupby("pair_id").cumcount()

    ckpt = resolve_checkpoint(cfg)
    model, tokenizer, device = model_utils.load_model_and_tokenizer(cfg, checkpoint=ckpt)

    print(f"[step-2] {model_key} on {dataset_key}: {len(df)} paraphrase rows")
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
    print(f"[step-2] consistent share: {stats['consistent_share']:.4f}")
    return True
