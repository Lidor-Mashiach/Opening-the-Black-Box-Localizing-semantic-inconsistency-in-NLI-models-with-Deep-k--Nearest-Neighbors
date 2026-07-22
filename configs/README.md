# Configuration

All experiment knobs in one place, as YAML. Nothing is hard-coded in the
pipeline scripts.

## Merge Model

These files are **settings, not weights** - they exist before any model or
dataset is downloaded, and they are what TELLS the pipeline which backbone to
fetch (`hf_id`) and where each dataset lives. A `models/<MODEL>.yaml` is not
the model - it is the recipe for fetching and configuring it. The
`tuned/<COMBO>.yaml` overlay (written later by Optuna) overrides only the
training block - learning rate, batch size, epochs - on top of the base.

One **base** file plus small **override** files, merged in this order
(later wins, nested keys merge deep):

```
base.yaml  ->  models/<MODEL>.yaml  ->  datasets/<DATASET>.yaml  ->  tuned/<MODEL>__<DATASET>.yaml
```

The last, optional layer is written by Optuna ([`tuning/`](../tuning/README.md))
and carries the best training hyper-parameters found for that combination -
to retrain with them, delete the combination's
`results/checkpoints/` and rerun the pipeline (derived data is wiped and
rebuilt automatically at every run start).

`common/config_loader.py::load_config(model_key, dataset_key)` performs the
merge - every script in the pipeline goes through it.

## Files

| File | Holds |
|------|-------|
| `base.yaml` | Seed, label convention, DkNN K candidates + validation fraction, sampling caps, training defaults, encoding settings, Optuna search space |
| `tuned/<MODEL>__<DATASET>.yaml` | Optional best-hyper-parameter overlay, written automatically by `tuning/run_tuning.py` |
| `models/BERT-base.yaml` | `hf_id`, pooling (`cls`) |
| `models/RoBERTa-large.yaml` | `hf_id`, pooling, smaller batch + lower LR |
| `models/DeBERTa-large.yaml` | `hf_id` (v1, v3 swap documented inline), pooling, overrides |
| `models/BART-large.yaml` | `hf_id`, pooling `eos` (encoder-decoder), overrides |
| `datasets/SNLI.yaml` | `hf_id`, folder, split mapping |
| `datasets/MNLI.yaml` | `hf_id`, folder, split mapping (matched-dev = validation, mismatched-dev = test) |
| `datasets/ANLI.yaml` | `hf_id`, folder, rounds r1-r3 |

> Paraphrase-bank knobs deliberately live **outside** `configs/` - in
> [`Paraphrase-Generator/paraphrase_config.yaml`](../setup-files/Paraphrase-Generator/paraphrase_config.yaml).
> The bank is a one-time static INPUT to the pipeline, not a product of it.

## Published Checkpoint Provenance (official sources)

The pre-declared checkpoints are the **official releases of the model
authors' own organizations** - the best published fits for MNLI:

| Combination | Checkpoint | Publisher (official HF org) | Reported MNLI accuracy | Proof / source |
|-------------|------------|----------------------------|------------------------|----------------|
| RoBERTa-large x MNLI | `FacebookAI/roberta-large-mnli` | FacebookAI (Meta) | ~90.2 (matched) | https://huggingface.co/FacebookAI/roberta-large-mnli |
| DeBERTa-large x MNLI | `microsoft/deberta-large-mnli` | Microsoft | 91.3 / 91.1 (m/mm) | https://huggingface.co/microsoft/deberta-large-mnli |
| BART-large x MNLI | `facebook/bart-large-mnli` | Facebook (Meta) | ~89.9 / 90.0 (m/mm) | https://huggingface.co/facebook/bart-large-mnli |

Everything else is fine-tuned locally by Step-1, because: (a) no **official**
single-dataset SNLI / ANLI checkpoints exist for these architectures,
(b) available community mixes (e.g. SNLI+MNLI+FEVER+ANLI models) would
violate the project's dataset-separation principle. (c) BERT-base has no
official NLI checkpoint at all - and is the cheapest to fine-tune.

## Published NLI Checkpoints (optional)

A model YAML may declare `nli_checkpoints: {<DATASET>: <hf_id>}` - a
published, **single-dataset** fine-tuned checkpoint that lets the combination
skip `train.py` entirely (`resolve_checkpoint` prefers a local Step-1
checkpoint, then falls back to this). The published weights are downloaded
into the project once (`Models/finetuned/<MODEL>__<DATASET>/`) and loaded
locally on every later run, so Phase A never re-contacts HuggingFace.
Predictions are auto-remapped to the dataset label convention via the
checkpoint's `id2label`.

Skipping `train.py` skips only local training, not the experiment: the
official checkpoint is a fine-tuned model, so Step-1 still runs it over the
train split to build this model's correct-only subset (the same "which rows
the model classifies correctly" filter every combination uses). There is no
Optuna for these three combinations - a ready-made fine-tuned checkpoint has
no training hyper-parameters to search.

The three official MNLI checkpoints are pre-declared (RoBERTa / DeBERTa /
BART - the best published fits for MNLI). SNLI / ANLI / BERT-base have no
official single-dataset checkpoints, and multi-dataset mixes (e.g. models
trained on SNLI+MNLI+FEVER+ANLI together) are **deliberately not used** -
they would violate the project's dataset-separation principle.

## Adding a Model or Dataset

Drop one YAML into `models/` or `datasets/` - the runners discover it
automatically (`list_model_keys()` / `list_dataset_keys()`), and a matching
Step-1 combination folder is all that is still needed.

Example - reading a merged config:

```python
from common.config_loader import load_config
cfg = load_config("BART-large", "MNLI")
cfg["pooling"]                  # 'eos'   (model file)
cfg["training"]["batch_size"]   # 16      (model override)
cfg["training"]["epochs"]       # 3       (base survives the merge)
cfg["splits"]["test"]           # 'validation_mismatched' (dataset file)
```