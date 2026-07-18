# BART-large x SNLI - Step-1 Combination Folder

Thin, per-combination wrappers. All real logic lives in
[`Research-Pipeline/common/`](../../../common/README.md); these scripts only pin
`MODEL_KEY = "BART-large"`, `DATASET_KEY = "SNLI"`. Configuration comes from
[`configs/`](../../../../configs) (base -> model -> dataset -> tuned merge).

## Run Order (what `run.py` does, in order)

| # | Stage | What it does | Output (all under `Runtime-Data/BART-large/SNLI/` unless noted) |
|---|-------|--------------|-----------------------------------------------------------|
| 0 | workspace wipe | Deletes this combination's Runtime-Data + Step-2/3/4 results from previous runs | fresh empty folder |
| 1 | `train.py` | Fine-tunes BART-large FRESH from the official pretrained backbone on the ORIGINAL SNLI train split (read-only), with this run's random seed and the tuned hyper-parameters when available | `results/checkpoints/final/` (rebuilt every run) |
| 2 | `build_filtered_dataset.py` | A NEW reduced COPY: only rows the model got right | `train_correct.csv` + `filter_stats.json` |
| 3 | `encode_hypotheses.py` | Every kept hypothesis at every layer (pool hypotheses always included) | `hypotheses.npz` + meta |
| 4 | `evaluate.py` | Baseline accuracy on validation + test | `results/baseline_eval.json` |

```bash
python run.py
```

No flags. The original SNLI dataset and the paraphrase bank are **never
modified**.

Every pipeline run is an independent research sample: this combination
re-trains from the backbone with a fresh seed, so its results vary run to run
and are averaged in `runs_summary.csv`. A job requeued mid-training with the
same pinned `NLI_SEED` resumes instead of restarting.
