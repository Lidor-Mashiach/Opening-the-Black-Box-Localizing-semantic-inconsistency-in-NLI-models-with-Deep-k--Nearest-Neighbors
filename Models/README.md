# Models - Backbones (in the project) + Tuned Configs

Everything about the models the research fine-tunes, kept INSIDE the project.
Nothing lives in a home-directory cache.

## Layout

```
Models/
+-- raw/
    +-- <MODEL>/          # the pretrained backbone downloaded from HuggingFace
        +-- config.json   # (config + tokenizer + safetensors weights)
        +-- tokenizer.*
        +-- model.safetensors
```

The best hyper-parameters produced by Optuna are NOT stored here - they live
in `configs/tuned/<MODEL>__<DATASET>.yaml`, which is exactly where `train.py`
reads them from.

## Who fills this, and when

* `Models/raw/<MODEL>/` is created on first need by `ensure_raw_backbone()`
  (called by Optuna and by training). The HuggingFace weights are downloaded
  ONCE into the project; every later run reuses the local copy - the HF home
  cache is never used.

## What git does with it

* The backbone **weights are heavy**, so `.gitignore` keeps `Models/raw/`
  local (this README and a `.gitkeep` are tracked, so the folder exists on a
  fresh clone).
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

So the config names the model; this folder holds the model. And the tuned
hyper-parameters (a third thing) live in `configs/tuned/`, not here.

## The training precondition (see common/training.py)

1. backbone already in the project -> reuse it (Optuna ran here).
2. no backbone, but a tuned config exists (came via git) -> download the
   backbone into the project, train with the tuned values.
3. neither -> Optuna was never run: training prints a red error and exits.
