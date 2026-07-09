# 1️⃣ Step-1 - Train, Filter, Encode

The heavy step: fine-tuning, building each model's "correct only" subset,
and creating the layer-wise representation banks. Organized as **one folder
per (model, dataset) combination** - 4 models x 3 datasets = 12 folders,
each with the same four thin wrapper scripts and its own `run.py`.

## 📁 Combination Folders

Every `<MODEL>__<DATASET>/` folder contains:

| Script | Stage | Output |
|--------|-------|--------|
| `train.py` | 1a - fine-tune | `results/checkpoints/final/`, `results/train_metrics.json` |
| `build_filtered_dataset.py` | 1b - keep correct rows only | `Datasets/<dataset>/filtered/<MODEL>/train_correct.csv` |
| `encode_hypotheses.py` | 1c - layer-wise bank | `Datasets/Encoded_Datasets/<MODEL>/<DATASET>/hypotheses.npz` |
| `evaluate.py` | 1d - baseline accuracy | `results/baseline_eval.json` |
| `run.py` | meta-runner for this combination | runs 1a -> 1d in order |

All real logic lives in [`common/`](../../common/README.md); the wrappers only
pin `MODEL_KEY` / `DATASET_KEY`. `results/` folders are created automatically
and are git-ignored (checkpoints are heavy and reproducible).

## 🔑 Why "correct only"?

Only (premise, hypothesis) rows the model classified **correctly** during
training continue to the experiment - so later steps measure sensitivity to
rephrasing, never ordinary mistakes. Each model therefore defines its own
population, which is also why paraphrase files are per-model.

## 🛟 Resumability & Crash Safety

Built for week-long SLURM jobs: training keeps **one rolling epoch
checkpoint** and automatically resumes from it after a crash or pre-emption
(`resume_from_checkpoint`); once `final/` is saved the rolling checkpoint is
deleted. Every stage (train / filter / encode / evaluate) **skips itself**
when its output already exists - rerunning a combination continues exactly
where it stopped. Pass `--force` to redo (e.g. after Optuna wrote tuned
hyper-parameters into `configs/tuned/`).

## 🏁 Skipping Training with a Published Checkpoint

When a model YAML declares `nli_checkpoints` for a dataset (the official MNLI
checkpoints are pre-declared for RoBERTa / DeBERTa / BART), stages 1b-1d work
directly against that published checkpoint - `train.py` can be skipped for
that combination. Everything else is unchanged; predictions are auto-remapped
to the dataset label convention. See [`configs/README.md`](../../../configs/README.md).

## 🚀 Meta-Runner

```bash
python run_step1.py                                  # all 12 combinations
python run_step1.py --models BERT-base --datasets SNLI,MNLI
python BERT-base__SNLI/run.py                        # a single combination
```

On the BGU SLURM cluster prefer one combination per job (the last form).
