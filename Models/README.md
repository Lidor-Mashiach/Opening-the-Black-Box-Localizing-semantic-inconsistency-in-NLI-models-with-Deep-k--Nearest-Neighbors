# Models - Backbones (in the project) + Tuned Configs

Everything about the models the research fine-tunes, kept INSIDE the project.
Nothing lives in a home-directory cache.

## Layout

```
Models/
+-- raw/
|   +-- <MODEL>/              # the pretrained BACKBONE downloaded from HuggingFace
|       +-- config.json       # (config + tokenizer + safetensors weights)
|       +-- tokenizer.*
|       +-- model.safetensors
+-- finetuned/
    +-- <MODEL>__<DATASET>/   # an OFFICIAL published fine-tuned checkpoint
        +-- config.json       # (its own id2label order, remapped at inference)
        +-- tokenizer.*
        +-- model.safetensors
```

`raw/` holds un-fine-tuned backbones (the starting point for local training).
`finetuned/` holds ready-made fine-tuned weights - the official single-dataset
NLI checkpoints (the `*-mnli` models) declared under `nli_checkpoints` in a
model's config. A combination that has such a checkpoint uses it directly and
is never trained locally. Both trees keep each model in its own clearly-named
subfolder, and both are downloaded into the project once and reused offline.

Locally fine-tuned checkpoints (produced by Step-1 training for combinations
WITHOUT an official checkpoint) are a separate, run-specific artifact and live
under the pipeline's Step-1 results folder, not here - they are wiped and
retrained per run, so they are not part of this persistent model store.

The best hyper-parameters produced by Optuna are NOT stored here - they live
in `configs/tuned/<MODEL>__<DATASET>.yaml`, which is exactly where `train.py`
reads them from.

## Who fills this, and when

* `Models/raw/<MODEL>/` is created on first need by `ensure_raw_backbone()`
  (called by Optuna and by training). The HuggingFace weights are downloaded
  ONCE into the project - every later run reuses the local copy, and the HF
  home cache is never used.
* `Models/finetuned/<MODEL>__<DATASET>/` is created on first need by
  `ensure_finetuned_checkpoint()` (called via `resolve_checkpoint()` when a
  combination has an official checkpoint). The published weights are
  downloaded ONCE into the project - every later Phase A run loads them
  locally and never contacts HuggingFace again. The download is atomic (temp
  dir then rename), so an interrupted download never leaves a half-written
  checkpoint.

## What git does with it

* Both **weight trees are heavy**, so `.gitignore` keeps `Models/raw/` and
  `Models/finetuned/` local (this README and a `.gitkeep` in each are tracked,
  so the folders exist on a fresh clone).
* The **tuned configs are light and ARE committed** (`configs/tuned/`), so a
  machine that pulls the repo has the optimized hyper-parameters and only
  needs to download the backbone - it never has to re-run Optuna.

## Models/ vs configs/models/ - two different things

* **`configs/models/<MODEL>.yaml`** - a tiny TEXT config: the HuggingFace id
  (`hf_id`), pooling type, and any per-model training overrides. It describes
  WHICH backbone to fetch and how to treat it. It exists before anything is
  downloaded - it is just settings.
* **`Models/raw/<MODEL>/`** - the actual downloaded BACKBONE (weights +
  tokenizer), fetched from the Hub using that `hf_id` on first need.

So the config names the model. This folder holds the model. And the tuned
hyper-parameters (a third thing) live in `configs/tuned/`, not here.

## The training precondition (see common/training.py)

1. backbone already in the project -> reuse it (Optuna ran here).
2. no backbone, but a tuned config exists (came via git) -> download the
   backbone into the project, train with the tuned values.
3. neither -> Optuna was never run: training prints a red error and exits.