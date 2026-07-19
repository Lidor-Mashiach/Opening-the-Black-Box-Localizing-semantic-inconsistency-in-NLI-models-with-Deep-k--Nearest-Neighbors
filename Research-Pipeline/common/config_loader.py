"""Configuration loading and repository paths.

Configuration lives in <repo>/configs as YAML:
    base.yaml                global defaults
    models/<MODEL>.yaml      per-model settings and overrides
    datasets/<DATASET>.yaml  per-dataset settings and overrides

Merge order (later wins, nested dicts merge key-by-key):
    base -> model -> dataset -> tuned/<MODEL>__<DATASET>.yaml (if present)

The optional `tuned` overlay is written by tuning/run_tuning.py (Optuna) and
carries the best training hyper-parameters found for that combination.
"""
from pathlib import Path

import os

import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
CONFIGS_DIR = REPO_ROOT / "configs"
DATASETS_DIR = REPO_ROOT / "Datasets"        # READ-ONLY sources: raw corpora + paraphrase banks
RUNTIME_DIR = REPO_ROOT / "Runtime-Data"     # everything a run derives lives here
MODELS_DIR = REPO_ROOT / "Models"            # backbones downloaded from HF live IN the project


def _read_yaml(path):
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def _deep_merge(base, override):
    """Recursively merge `override` into `base` (returns a new dict)."""
    merged = dict(base)
    for key, value in override.items():
        if key in merged and isinstance(merged[key], dict) and isinstance(value, dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def load_config(model_key=None, dataset_key=None):
    """Return the merged configuration dict for a (model, dataset) pair."""
    cfg = _read_yaml(CONFIGS_DIR / "base.yaml")
    if model_key is not None:
        cfg = _deep_merge(cfg, _read_yaml(CONFIGS_DIR / "models" / f"{model_key}.yaml"))
    if dataset_key is not None:
        cfg = _deep_merge(cfg, _read_yaml(CONFIGS_DIR / "datasets" / f"{dataset_key}.yaml"))
    if model_key is not None and dataset_key is not None:
        tuned = CONFIGS_DIR / "tuned" / f"{model_key}__{dataset_key}.yaml"
        if tuned.exists():
            cfg = _deep_merge(cfg, _read_yaml(tuned))
            cfg["_tuned_overlay"] = str(tuned)   # provenance marker
    env_seed = os.environ.get("NLI_SEED")
    if env_seed is not None:
        cfg["seed"] = int(env_seed)      # run_pipeline pins/injects the run seed here
    return cfg


def list_model_keys():
    return sorted(p.stem for p in (CONFIGS_DIR / "models").glob("*.yaml"))


def list_dataset_keys():
    return sorted(p.stem for p in (CONFIGS_DIR / "datasets").glob("*.yaml"))


# ---- Standard locations derived from a merged config -------------------

def dataset_dir(cfg):
    """<repo>/Datasets/<dataset dir> - READ-ONLY (raw/ + paraphrases/ bank).

    Nothing in the pipeline ever writes here except the one-time downloader
    (raw/) and the standalone Paraphrase-Generator (paraphrases/).
    """
    return DATASETS_DIR / cfg["dir"]


def raw_backbone_dir(cfg):
    """<repo>/Models/raw/<MODEL>/ - the pretrained backbone downloaded from
    HuggingFace, stored INSIDE the project (never outside it). Weights are
    heavy, so .gitignore keeps them local; the folder itself is tracked."""
    return MODELS_DIR / "raw" / cfg["model_key"]


def combo_runtime_dir(cfg):
    """<repo>/Runtime-Data/<MODEL>/<DATASET> - ALL run-derived data of one
    combination: the reduced dataset copy, the hypothesis/paraphrase
    encodings and the per-model paraphrase copy. Created fresh (and wiped)
    at the start of every run - see common/workspace.py."""
    return RUNTIME_DIR / cfg["model_key"] / cfg["dataset_key"]


def filtered_dir(cfg):
    """Where the reduced ('correct only') dataset COPY lives - inside the
    combination's Runtime-Data folder (the original dataset is never touched)."""
    return combo_runtime_dir(cfg)


def paraphrase_bank_csv(cfg):
    """The DATASET-level, model-independent paraphrase bank.

    One static file per dataset (Datasets/<dataset>/paraphrases/
    paraphrase_bank.csv) with UP TO paraphrases.per_hypothesis verified
    paraphrases per hypothesis (kept partial; only zero-verified dropped) -
    a shareable research asset.
    Each model consumes only the rows whose hypothesis it classified
    correctly (the intersection happens in Step-2).
    """
    return dataset_dir(cfg) / "paraphrases" / "paraphrase_bank.csv"


def encoded_dir(cfg):
    """Where the layer-wise encodings live - the same Runtime-Data combo folder."""
    return combo_runtime_dir(cfg)


# ---- Pipeline result locations (single source of truth for all steps) ---

PHASE_A_DIR = REPO_ROOT / "Research-Pipeline" / "PhaseA"


def combo_name(cfg):
    """Canonical '<MODEL>__<DATASET>' name used for folders and result files."""
    return f"{cfg['model_key']}__{cfg['dataset_key']}"


def step1_results_dir(cfg):
    """results/ folder inside the combo's Step-1 directory (auto-created)."""
    return PHASE_A_DIR / "Step-1_Train-Filter-Encode" / combo_name(cfg) / "results"


def checkpoint_dir(cfg):
    """Final fine-tuned checkpoint produced by Step-1 training."""
    return step1_results_dir(cfg) / "checkpoints" / "final"


def step_results_dir(step_dirname, cfg):
    """results/<COMBO>/ folder for Steps 2-4 (auto-created by the scripts)."""
    return PHASE_A_DIR / step_dirname / "results" / combo_name(cfg)


def resolve_checkpoint(cfg):
    """The classifier to use for every inference stage, in priority order:

    1. The locally fine-tuned Step-1 checkpoint (results/checkpoints/final).
    2. A published, single-dataset NLI checkpoint declared in the model's
       YAML under  nli_checkpoints: {<DATASET>: <hf_id>}  - this lets a
       combination skip train.py entirely (predictions are auto-remapped to
       the dataset label convention via the checkpoint's id2label).

    Raises with clear guidance when neither exists.
    """
    local = checkpoint_dir(cfg)
    if local.exists():
        return local
    published = (cfg.get("nli_checkpoints") or {}).get(cfg["dataset_key"])
    if published:
        print(f"[checkpoint] using published NLI checkpoint: {published}")
        return published
    raise FileNotFoundError(
        f"No checkpoint for {cfg['model_key']} x {cfg['dataset_key']}: "
        f"run Step-1 train.py, or declare nli_checkpoints in "
        f"configs/models/{cfg['model_key']}.yaml")
