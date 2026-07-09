# DeBERTa-large x SNLI - Step-1 Combination Folder

Thin, per-combination wrappers. All real logic lives in
[`Research-Pipeline/common/`](../../../common/README.md); these scripts only pin
`MODEL_KEY = "DeBERTa-large"`, `DATASET_KEY = "SNLI"` and call it. Configuration comes from
[`configs/`](../../../../../configs) (base -> model -> dataset -> tuned merge).

## Run Order

| # | Script | What it does | Output |
|---|--------|--------------|--------|
| 1 | `train.py` | Fine-tune DeBERTa-large on the SNLI train split | `results/checkpoints/final/`, `results/train_metrics.json` |
| 2 | `build_filtered_dataset.py` | Keep only train rows the model got right | `Datasets/.../filtered/DeBERTa-large/train_correct.csv` |
| 3 | `encode_hypotheses.py` | Store every kept hypothesis at every layer | `Datasets/Encoded_Datasets/DeBERTa-large/SNLI/hypotheses.npz` |
| 4 | `evaluate.py` | Baseline accuracy on validation + test | `results/baseline_eval.json` |

Run everything in order with one command:

```bash
python run.py             # resumable - finished stages are skipped
python run.py --force     # redo everything (e.g. after Optuna tuning)
```

Crash-resilience: training keeps one rolling epoch checkpoint and resumes
from it automatically; every stage skips itself once its output exists.
