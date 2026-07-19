"""Step-1a: fine-tune one model on one dataset (HuggingFace Trainer).

RUN SEMANTICS (every pipeline run is an independent research sample):
* Combinations with an OFFICIAL published checkpoint for this dataset
  (declared under `nli_checkpoints` in configs/models/) never train - the
  official weights are used as-is by resolve_checkpoint().
* Everything else fine-tunes FRESH on every run, from the official
  pretrained backbone, with the run's seed (a new random seed per run,
  injected by run_pipeline via NLI_SEED).
* Backbone lives IN THE PROJECT at Models/raw/<MODEL>/ (downloaded there once
  from HuggingFace, reused forever after; kept out of git by .gitignore, not
  by living outside the repo). The FINE-TUNED model this run produces is saved
  to results/checkpoints/final/ - a separate location. Training reads the
  backbone and WRITES the checkpoint; it never writes the backbone.
* Precondition gate before any training:
    - backbone already in the project  -> reuse it (Optuna ran here).
    - no backbone but a tuned config exists (came via git from a machine that
      ran Optuna) -> download the backbone into the project, train tuned.
    - neither -> Optuna was never run: print a red error naming the sbatch to
      run, and exit. Training never silently proceeds on defaults in this case.
* Optuna never produces weights (it saves only configs/tuned/*.yaml), so
  there is nothing of Optuna's for training to clobber either. Tuning and
  training are independent: tuning finds numbers, training uses them.
* Crash safety without mixing runs: the rolling epoch checkpoint carries a
  `run_seed.txt` marker. A resubmitted job with the SAME pinned NLI_SEED
  resumes mid-training; any other seed wipes the stale checkpoints and
  starts clean - two runs can never contaminate each other.
* Hyper-parameters: the tuned Optuna overlay (configs/tuned/<COMBO>.yaml)
  is used automatically when present - and the log says loudly, in color,
  whether this training is TUNED or running on base defaults.

Kept deliberately minimal and version-proof: no mid-training evaluation
arguments; validation accuracy is computed explicitly after training with the
same predict() used everywhere else in the pipeline.
"""
import json
from pathlib import Path

import shutil

import numpy as np
import torch
from transformers import DataCollatorWithPadding, Trainer, TrainingArguments
from transformers.trainer_utils import get_last_checkpoint

from . import data_loading, model_utils
from .config_loader import load_config, step1_results_dir
from .logging_utils import log


def prepare_checkpoint_dir(ckpt_dir, seed, find_last=None):
    """Seed-guarded resume: continue ONLY a crash of this very run.

    Returns the checkpoint path to resume from, or None for a fresh start.
    The guard: `run_seed.txt` inside the checkpoints folder. Same seed +
    rolling checkpoint present -> resume mid-training (SLURM requeue with a
    pinned NLI_SEED). Anything else (previous completed run, a crash of a
    DIFFERENT run) -> wipe and train fresh, so runs never mix.
    """
    if find_last is None:
        find_last = lambda d: get_last_checkpoint(str(d))          # noqa: E731
    seed_file = ckpt_dir / "run_seed.txt"
    last = find_last(ckpt_dir) if ckpt_dir.exists() else None
    if last and seed_file.exists() and seed_file.read_text().strip() == str(seed):
        return last
    if ckpt_dir.exists():
        shutil.rmtree(ckpt_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    seed_file.write_text(str(seed))
    return None


def fine_tune(model_key, dataset_key):
    cfg = load_config(model_key, dataset_key)
    results_dir = step1_results_dir(cfg)
    ckpt_dir = results_dir / "checkpoints"
    final_dir = ckpt_dir / "final"
    results_dir.mkdir(parents=True, exist_ok=True)

    official = (cfg.get("nli_checkpoints") or {}).get(dataset_key)
    if official:
        log("TRAIN", f"no training for this combination - the OFFICIAL "
            f"published checkpoint `{official}` is used as-is (declared in "
            f"configs/models/{model_key}.yaml; zero seed-variance by "
            f"construction)", model_key, dataset_key)
        return None

    # ---- precondition gate: backbone-in-project vs tuned-config vs neither ----
    from .config_loader import raw_backbone_dir
    backbone_present = (raw_backbone_dir(cfg) / "config.json").exists()
    tuned = cfg.get("_tuned_overlay")

    if backbone_present:
        # Optuna most likely ran on THIS machine: the backbone was pulled into
        # the project then, so nothing is downloaded again now.
        log("TRAIN", f"backbone found in the project ({raw_backbone_dir(cfg)}) "
            f"- reusing it, no download", model_key, dataset_key)
        if tuned:
            log("TRAIN", f"hyper-parameters: TUNED by Optuna ({tuned})",
                model_key, dataset_key)
        else:
            log("WARN", f"no tuned config for {model_key}__{dataset_key} - "
                f"training on base defaults; run the Optuna job to optimize",
                model_key, dataset_key)
    elif tuned:
        # Optuna ran on ANOTHER machine: its tuned config came via git, but the
        # heavy backbone did not. Fetch the backbone into the project now.
        log("TRAIN", f"tuned config present but backbone missing locally - "
            f"Optuna ran elsewhere; downloading the backbone into the project "
            f"and training with the tuned values ({tuned})",
            model_key, dataset_key)
    else:
        # Neither: tuning was never done. Do NOT silently train on defaults.
        log("ERROR", f"cannot train {model_key} x {dataset_key}: no tuned "
            f"config (configs/tuned/{model_key}__{dataset_key}.yaml) and no "
            f"backbone in the project. Run Optuna first:", model_key, dataset_key)
        log("ERROR", f"    sbatch tuning/sbatch/tune_{model_key}__{dataset_key}.sbatch"
            f"   (or: cd tuning && bash run_all_sbatch.sh)", model_key, dataset_key)
        raise SystemExit(1)

    resume_from = prepare_checkpoint_dir(ckpt_dir, cfg["seed"])
    torch.manual_seed(cfg["seed"])
    np.random.seed(cfg["seed"])

    if resume_from:
        log("TRAIN", f"fresh run seed {cfg['seed']} matched an interrupted "
            f"run - resuming mid-training from {resume_from}",
            model_key, dataset_key)
    else:
        log("TRAIN", f"fine-tuning starts FRESH (run seed {cfg['seed']})",
            model_key, dataset_key)
    train_ds = data_loading.load_nli(cfg, "train")
    model, tokenizer, device = model_utils.load_model_and_tokenizer(cfg)

    def tokenize(batch):
        return tokenizer(batch["premise"], batch["hypothesis"], truncation=True,
                         max_length=cfg["training"]["max_seq_len"])

    tokenized = train_ds.map(
        tokenize, batched=True,
        remove_columns=[c for c in train_ds.column_names if c != "label"],
    )

    fp16 = cfg["training"]["fp16"]
    fp16 = torch.cuda.is_available() if fp16 == "auto" else bool(fp16)

    args = TrainingArguments(
        output_dir=str(ckpt_dir),
        num_train_epochs=cfg["training"]["epochs"],
        per_device_train_batch_size=cfg["training"]["batch_size"],
        learning_rate=float(cfg["training"]["learning_rate"]),
        weight_decay=float(cfg["training"].get("weight_decay", 0.0)),
        warmup_ratio=float(cfg["training"].get("warmup_ratio", 0.0)),
        seed=cfg["seed"],
        save_strategy="epoch",       # crash-resilience: keep one rolling checkpoint
        save_total_limit=1,
        logging_steps=100,
        fp16=fp16,
        report_to=[],
    )
    trainer = Trainer(model=model, args=args, train_dataset=tokenized,
                      data_collator=DataCollatorWithPadding(tokenizer))

    trainer.train(resume_from_checkpoint=resume_from)

    # Self-describing label names -> prediction_remap() maps them to identity
    model.config.id2label = {0: "entailment", 1: "neutral", 2: "contradiction"}
    model.config.label2id = {"entailment": 0, "neutral": 1, "contradiction": 2}

    final_dir.mkdir(parents=True, exist_ok=True)
    trainer.save_model(str(final_dir))
    tokenizer.save_pretrained(str(final_dir))

    # Post-training validation accuracy (shared predict, no Trainer eval args)
    val = data_loading.load_nli(cfg, "validation")
    model.eval()
    preds = model_utils.predict(model, tokenizer, device,
                                val["premise"], val["hypothesis"], cfg, desc="val")
    accuracy = float((preds == np.asarray(val["label"])).mean())

    metrics = {
        "model": model_key,
        "dataset": dataset_key,
        "train_examples": len(train_ds),
        "validation_accuracy": accuracy,
        "checkpoint": str(final_dir),
    }
    with open(results_dir / "train_metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    # rolling epoch checkpoints are no longer needed once `final` is saved
    for rolling in ckpt_dir.glob("checkpoint-*"):
        shutil.rmtree(rolling, ignore_errors=True)
    log("TRAIN", f"done - validation accuracy {accuracy:.4f}", model_key, dataset_key)
    return final_dir
