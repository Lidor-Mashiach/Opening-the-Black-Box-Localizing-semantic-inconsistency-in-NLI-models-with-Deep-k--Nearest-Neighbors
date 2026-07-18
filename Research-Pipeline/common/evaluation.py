"""Step-1d: baseline evaluation of the fine-tuned model.

Plain accuracy on the validation and test splits of the ORIGINAL dataset
(premise : hypothesis). This is the sanity baseline before any paraphrase is
involved - it should roughly reproduce the accuracies reported in the
literature for each model x dataset pair.
"""
import json

import numpy as np

from . import data_loading, model_utils
from .config_loader import load_config, resolve_checkpoint, step1_results_dir
from .logging_utils import log


def evaluate_baseline(model_key, dataset_key):
    cfg = load_config(model_key, dataset_key)
    ckpt = resolve_checkpoint(cfg)

    model, tokenizer, device = model_utils.load_model_and_tokenizer(cfg, checkpoint=ckpt)
    results = {"model": model_key, "dataset": dataset_key}
    for split in ("validation", "test"):
        ds = data_loading.load_nli(cfg, split)
        preds = model_utils.predict(model, tokenizer, device,
                                    ds["premise"], ds["hypothesis"], cfg, desc=split)
        accuracy = float((preds == np.asarray(ds["label"])).mean())
        results[f"{split}_accuracy"] = accuracy
        results[f"{split}_examples"] = len(ds)
        log("EVAL", f"{split} accuracy: {accuracy:.4f} ({len(ds)} examples)",
            model_key, dataset_key)

    out_dir = step1_results_dir(cfg)
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "baseline_eval.json", "w") as f:
        json.dump(results, f, indent=2)
    return results
