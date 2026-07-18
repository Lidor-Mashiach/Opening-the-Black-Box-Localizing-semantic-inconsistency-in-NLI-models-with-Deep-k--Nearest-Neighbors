# 1 Step-1 - Train, Filter, Encode

The heavy step: fine-tuning, building each model's "correct only" subset,
and creating the layer-wise representation banks. Organized as **one folder
per (model, dataset) combination** - 4 models x 3 datasets = 12 folders,
each with the same four thin wrapper scripts and its own `run.py`.

## Combination Folders

Every `<MODEL>__<DATASET>/` folder contains:

| Script | Stage | Output |
|--------|-------|--------|
| `train.py` | 1a - fine-tune | `results/checkpoints/final/`, `results/train_metrics.json` |
| `build_filtered_dataset.py` | 1b - reduced correct-only COPY | `Runtime-Data/<MODEL>/<DATASET>/train_correct.csv` |
| `encode_hypotheses.py` | 1c - layer-wise bank | `Runtime-Data/<MODEL>/<DATASET>/hypotheses.npz` |
| `evaluate.py` | 1d - baseline accuracy | `results/baseline_eval.json` |
| `run.py` | meta-runner for this combination | runs 1a -> 1d in order |

All real logic lives in [`common/`](../../common/README.md); the wrappers only
pin `MODEL_KEY` / `DATASET_KEY`. `results/` folders are created automatically
and are git-ignored (checkpoints are heavy and reproducible).

## Why "correct only"?

Only (premise, hypothesis) rows the model classified **correctly** during
training continue to the experiment - so later steps measure sensitivity to
rephrasing, never ordinary mistakes. Each model therefore defines its own
population; the shared paraphrase bank is intersected with it in Step-2, and
the per-model copy lands in `Runtime-Data/<MODEL>/<DATASET>/paraphrases_used.csv`.

## Workspace Wipe, Fresh Training & Crash Safety

Every `run.py` starts by **wiping the combination's workspace** -
`Runtime-Data/<MODEL>/<DATASET>/` plus its Step-2/3/4 results from any
previous run - so runs never mix leftovers, with no flags involved.

**Training is FRESH on every run** (that is what makes each run an
independent research sample): it starts from the official pretrained
backbone with the run's random seed - injected by `run_pipeline.py` via
`NLI_SEED` - and uses the Optuna-tuned hyper-parameters automatically when
`configs/tuned/<COMBO>.yaml` exists (a green log line says so; a yellow
`[WARN]` names the missing file otherwise). The three large-model MNLI
combinations never train: the OFFICIAL published checkpoints are used as-is.

**Crash safety without contamination:** training keeps one rolling epoch
checkpoint tagged with the run's seed (`run_seed.txt`). A resubmitted /
requeued job carrying the SAME pinned `NLI_SEED` resumes mid-training; any
other seed wipes the stale checkpoints and trains clean - two runs can never
blend into one model.

## Skipping Training with a Published Checkpoint

When a model YAML declares `nli_checkpoints` for a dataset (the official MNLI
checkpoints are pre-declared for RoBERTa / DeBERTa / BART), stages 1b-1d work
directly against that published checkpoint - `train.py` can be skipped for
that combination. Everything else is unchanged; predictions are auto-remapped
to the dataset label convention. See [`configs/README.md`](../../../configs/README.md).

## Meta-Runner

```bash
python run_step1.py                                  # all 12 combinations
python run_step1.py --models BERT-base --datasets SNLI,MNLI
python BERT-base__SNLI/run.py                        # a single combination
```

On the BGU SLURM cluster prefer one combination per job (the last form).
