# DeBERTa-large x MNLI - Step-1 Combination Folder

Thin, per-combination wrappers. All real logic lives in
[`Research-Pipeline/common/`](../../../common/README.md); these scripts only pin
`MODEL_KEY = "DeBERTa-large"`, `DATASET_KEY = "MNLI"`. Configuration comes from
[`configs/`](../../../../configs) (base -> model -> dataset -> tuned merge).

## Run Order (what `run.py` does, in order)

| # | Stage | What it does | Output (all under `Runtime-Data/DeBERTa-large/MNLI/` unless noted) |
|---|-------|--------------|-----------------------------------------------------------|
| 0 | workspace wipe | Deletes this combination's Runtime-Data + Step-2/3/4 results from previous runs | fresh empty folder |
| 1 | `train.py` | Detects the OFFICIAL published checkpoint and skips training entirely | *(no training - the published weights are used directly)* |
| 2 | `build_filtered_dataset.py` | A NEW reduced COPY: only rows the model got right | `train_correct.csv` + `filter_stats.json` |
| 3 | `encode_hypotheses.py` | Every kept hypothesis at every layer (pool hypotheses always included) | `hypotheses.npz` + meta |
| 4 | `evaluate.py` | Baseline accuracy on validation + test | `results/baseline_eval.json` |

```bash
python run.py
```

No flags. The original MNLI dataset and the paraphrase bank are **never
modified**.

This combination uses the OFFICIAL published checkpoint
`microsoft/deberta-large-mnli` automatically (declared in
`configs/models/DeBERTa-large.yaml`) - `train.py` detects it, skips training, and
stages 2-4 run against it. Its results therefore carry **zero seed-variance**
across repeated runs, by construction.
