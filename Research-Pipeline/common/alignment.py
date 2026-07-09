"""Shared loading + alignment of encoded banks, paraphrase encodings and
Step-2 predictions - used by every Step-3 script so the (identical) plumbing
is written exactly once.

Returned dict:
    bank_reps    [n_bank, L, dim]   hypothesis bank (float16 on disk)
    bank_labels  [n_bank]
    bank_ids     [n_bank]           pair_id per bank row
    para_reps    [n_rows, L, dim]   paraphrase representations
    para_preds   [n_rows]           model prediction per paraphrase
    pair_ids     [n_rows]           source hypothesis of every paraphrase row
    para_idx     [n_rows]
    hypo_rows    [n_rows]           bank row index of each row's hypothesis
    inconsistent [n_rows] {0,1}     did the prediction change?
    strict_flip  [n_rows] {0,1}
    val_mask     [n_rows] bool      deterministic split for K selection
"""
import zlib

import numpy as np
import pandas as pd

from .config_loader import encoded_dir, step_results_dir

STEP2_DIRNAME = "Step-2_Paraphrase-Inference"


def _val_mask(pair_ids, val_fraction):
    """Deterministic, id-based validation split (stable across runs/machines).

    All paraphrases of the same hypothesis land on the same side by design.
    """
    limit = int(val_fraction * 1000)
    return np.array([zlib.crc32(str(p).encode()) % 1000 < limit for p in pair_ids])


def load_aligned(cfg):
    enc = encoded_dir(cfg)
    hyp_path = enc / "hypotheses.npz"
    par_path = enc / "paraphrases.npz"
    preds_path = step_results_dir(STEP2_DIRNAME, cfg) / "paraphrase_predictions.csv"
    for p in (hyp_path, par_path, preds_path):
        if not p.exists():
            raise FileNotFoundError(f"missing {p} - run the previous steps first")

    hyp = np.load(hyp_path, allow_pickle=False)
    par = np.load(par_path, allow_pickle=False)
    preds_df = pd.read_csv(preds_path)

    pair_ids = par["pair_ids"].astype(str)
    if len(preds_df) != len(pair_ids) or not (preds_df["pair_id"].astype(str).to_numpy() == pair_ids).all():
        raise ValueError("paraphrases.npz and paraphrase_predictions.csv are out of sync - re-run Step-2")

    bank_ids = hyp["pair_ids"].astype(str)
    bank_index = {pid: i for i, pid in enumerate(bank_ids)}
    in_bank = np.array([pid in bank_index for pid in pair_ids])
    if not in_bank.all():
        dropped = int((~in_bank).sum())
        print(f"[align] WARNING: dropping {dropped} paraphrase rows whose hypothesis "
              f"is not in the (possibly capped) bank")

    pair_ids = pair_ids[in_bank]
    hypo_rows = np.array([bank_index[pid] for pid in pair_ids])

    return {
        "bank_reps": hyp["reps"],
        "bank_labels": hyp["labels"],
        "bank_ids": bank_ids,
        "para_reps": par["reps"][in_bank],
        "para_preds": par["preds"][in_bank],
        "pair_ids": pair_ids,
        "para_idx": par["para_idx"][in_bank],
        "hypo_rows": hypo_rows,
        "inconsistent": 1 - preds_df["consistent"].to_numpy()[in_bank],
        "strict_flip": preds_df["strict_flip"].to_numpy()[in_bank],
        "val_mask": _val_mask(pair_ids, cfg["dknn"]["val_fraction"]),
    }
