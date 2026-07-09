"""Optuna hyper-parameter search for one (model, dataset) combination.

OBJECTIVE (maximize)
    Validation accuracy of the fine-tuned NLI classifier - the exact quantity
    Step-1 cares about, measured with the same shared predict() used across
    the pipeline. Train/validation are seeded subsamples
    (tuning.train_subsample / tuning.val_subsample) so a full study fits a
    single week-long GPU job.

SEARCH SPACE (configs/base.yaml -> tuning.search_space, one entry per feature)
    learning_rate  float, log-uniform 5e-6 .. 5e-5
    weight_decay   float, uniform    0 .. 0.1
    warmup_ratio   float, uniform    0 .. 0.2
    epochs         int              2 .. 3
    batch_size     categorical: [16, 32] for base models (default batch 32),
                   [8, 16] for the large models (default batch 16)

RESUME / CRASH SAFETY
    The study lives in sqlite (tuning/results/<COMBO>/optuna.db,
    load_if_exists=True). If the sbatch job crashes or is pre-empted, simply
    resubmit - finished trials are kept and the study continues from the next
    trial, never exceeding tuning.n_trials in total.

OUTPUT
    tuning/results/<COMBO>/optuna.db          the resumable study
    tuning/results/<COMBO>/best_params.yaml   best trial, human-readable
    tuning/results/<COMBO>/trials.csv         full trial history
    configs/tuned/<COMBO>.yaml                training overlay picked up
                                              automatically by load_config();
                                              rerun the pipeline with --force
                                              to retrain with these values.
"""
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(REPO_ROOT), str(REPO_ROOT / "Research-Pipeline")]

import numpy as np
import optuna
import torch
import yaml
from transformers import (AutoModelForSequenceClassification,
                          DataCollatorWithPadding, Trainer, TrainingArguments)

from common import data_loading, model_utils
from common.config_loader import CONFIGS_DIR, combo_name, load_config

RESULTS_ROOT = REPO_ROOT / "tuning" / "results"


def _subsample(ds, n, seed):
    if n is None or len(ds) <= n:
        return ds
    return ds.shuffle(seed=seed).select(range(n))


def _suggest_params(trial, cfg):
    """Draw one configuration from the per-feature search space."""
    space = cfg["tuning"]["search_space"]
    lr = space["learning_rate"]
    default_batch = cfg["training"]["batch_size"]
    batch_options = (space["batch_size"]["base_models"] if default_batch >= 32
                     else space["batch_size"]["large_models"])
    return {
        "learning_rate": trial.suggest_float("learning_rate", float(lr["low"]),
                                             float(lr["high"]), log=bool(lr.get("log"))),
        "weight_decay": trial.suggest_float("weight_decay",
                                            float(space["weight_decay"]["low"]),
                                            float(space["weight_decay"]["high"])),
        "warmup_ratio": trial.suggest_float("warmup_ratio",
                                            float(space["warmup_ratio"]["low"]),
                                            float(space["warmup_ratio"]["high"])),
        "epochs": trial.suggest_int("epochs", int(space["epochs"]["low"]),
                                    int(space["epochs"]["high"])),
        "batch_size": trial.suggest_categorical("batch_size", list(batch_options)),
    }


def run_study(model_key, dataset_key, n_trials=None):
    cfg = load_config(model_key, dataset_key)
    combo = combo_name(cfg)
    results_dir = RESULTS_ROOT / combo
    results_dir.mkdir(parents=True, exist_ok=True)

    device = model_utils.get_device()
    print(f"[tuning] {combo} | device: {device}"
          + (f" ({torch.cuda.get_device_name(0)})" if device.type == "cuda" else
             "  (no CUDA GPU detected - tuning on CPU will be very slow)"))

    # data prepared ONCE, reused by every trial
    train = _subsample(data_loading.load_nli(cfg, "train"),
                       cfg["tuning"]["train_subsample"], cfg["seed"])
    val = _subsample(data_loading.load_nli(cfg, "validation"),
                     cfg["tuning"]["val_subsample"], cfg["seed"] + 1)
    print(f"[tuning] train subsample: {len(train)}, validation subsample: {len(val)}")

    _, tokenizer, _ = model_utils.load_model_and_tokenizer(cfg)  # tokenizer only

    def tokenize(batch):
        return tokenizer(batch["premise"], batch["hypothesis"], truncation=True,
                         max_length=cfg["training"]["max_seq_len"])

    tokenized_train = train.map(
        tokenize, batched=True,
        remove_columns=[c for c in train.column_names if c != "label"])
    collator = DataCollatorWithPadding(tokenizer)
    val_labels = np.asarray(val["label"])

    def objective(trial):
        params = _suggest_params(trial, cfg)
        print(f"[tuning] trial {trial.number}: {params}")
        torch.manual_seed(cfg["seed"])
        np.random.seed(cfg["seed"])
        model = AutoModelForSequenceClassification.from_pretrained(
            cfg["hf_id"], num_labels=model_utils.NUM_LABELS).to(device)
        args = TrainingArguments(
            output_dir=str(results_dir / "trial_tmp"),
            num_train_epochs=params["epochs"],
            per_device_train_batch_size=params["batch_size"],
            learning_rate=params["learning_rate"],
            weight_decay=params["weight_decay"],
            warmup_ratio=params["warmup_ratio"],
            seed=cfg["seed"],
            save_strategy="no",
            logging_steps=200,
            fp16=torch.cuda.is_available(),
            report_to=[],
        )
        Trainer(model=model, args=args, train_dataset=tokenized_train,
                data_collator=collator).train()
        model.eval()
        preds = model_utils.predict(model, tokenizer, device,
                                    val["premise"], val["hypothesis"], cfg,
                                    desc=f"trial-{trial.number}")
        accuracy = float((preds == val_labels).mean())
        print(f"[tuning] trial {trial.number}: validation accuracy = {accuracy:.4f}")
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        return accuracy

    study = optuna.create_study(
        study_name=combo, direction="maximize",
        storage=f"sqlite:///{results_dir / 'optuna.db'}",
        load_if_exists=True,
        sampler=optuna.samplers.TPESampler(seed=cfg["seed"]))
    target = n_trials if n_trials is not None else cfg["tuning"]["n_trials"]
    finished = len([t for t in study.trials if t.state.is_finished()])
    remaining = max(0, target - finished)
    print(f"[tuning] study '{combo}': {finished} finished trials, "
          f"{remaining} remaining (target {target})")
    if remaining:
        study.optimize(objective, n_trials=remaining)

    best = study.best_trial
    print(f"[tuning] BEST: accuracy {best.value:.4f} with {best.params}")

    with open(results_dir / "best_params.yaml", "w") as f:
        yaml.safe_dump({"model": model_key, "dataset": dataset_key,
                        "best_validation_accuracy": float(best.value),
                        "best_params": best.params}, f, sort_keys=False)
    study.trials_dataframe().to_csv(results_dir / "trials.csv", index=False)

    tuned_dir = CONFIGS_DIR / "tuned"
    tuned_dir.mkdir(parents=True, exist_ok=True)
    overlay = {"training": {
        "learning_rate": float(best.params["learning_rate"]),
        "batch_size": int(best.params["batch_size"]),
        "epochs": int(best.params["epochs"]),
        "weight_decay": float(best.params["weight_decay"]),
        "warmup_ratio": float(best.params["warmup_ratio"]),
    }}
    tuned_path = tuned_dir / f"{combo}.yaml"
    with open(tuned_path, "w") as f:
        f.write(f"# Written by tuning/run_tuning.py (Optuna) - best of "
                f"{len(study.trials)} trials,\n# validation accuracy "
                f"{best.value:.4f}. Picked up automatically by load_config();\n"
                f"# rerun the pipeline with --force to retrain with these values.\n")
        yaml.safe_dump(overlay, f, sort_keys=False)
    print(f"[tuning] tuned overlay written -> {tuned_path}")
    return best
