"""Step-1a: fine-tune one model on one dataset (HuggingFace Trainer).

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


def fine_tune(model_key, dataset_key, force=False):
    cfg = load_config(model_key, dataset_key)
    results_dir = step1_results_dir(cfg)
    ckpt_dir = results_dir / "checkpoints"
    final_dir = ckpt_dir / "final"
    results_dir.mkdir(parents=True, exist_ok=True)

    if final_dir.exists() and (results_dir / "train_metrics.json").exists() and not force:
        print(f"[train] SKIP {model_key} x {dataset_key}: finished checkpoint exists "
              f"({final_dir}). Use --force to retrain (e.g. after Optuna tuning).")
        return final_dir

    torch.manual_seed(cfg["seed"])
    np.random.seed(cfg["seed"])

    print(f"[train] {model_key} on {dataset_key}")
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

    # Resume from the last epoch checkpoint after a crash / job pre-emption
    last_checkpoint = get_last_checkpoint(str(ckpt_dir)) if ckpt_dir.exists() else None
    if last_checkpoint is not None:
        print(f"[train] resuming from crash checkpoint: {last_checkpoint}")
    trainer.train(resume_from_checkpoint=last_checkpoint)

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
    print(f"[train] done - validation accuracy {accuracy:.4f}")
    return final_dir
