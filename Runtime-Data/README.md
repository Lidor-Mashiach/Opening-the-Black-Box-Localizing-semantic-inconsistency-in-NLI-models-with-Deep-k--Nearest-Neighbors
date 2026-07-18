# Runtime-Data - Everything a Run Derives, Nothing Else

This folder is created and managed **by the pipeline**. It holds every
run-derived artifact, in a strict `<MODEL>/<DATASET>/` hierarchy - and it is
the ONLY place runs write data to. `Datasets/` stays read-only forever.

## What Lives Here (per combination)

```
Runtime-Data/
+-- <MODEL>/
    +-- <DATASET>/
        +-- train_correct.csv      # the reduced COPY of the original dataset
        |                          # (only rows this model classified correctly)
        +-- filter_stats.json      # kept/total + train accuracy
        +-- hypotheses.npz         # layer-wise encodings of the kept hypotheses
        +-- hypotheses_meta.json
        +-- paraphrases_used.csv   # the per-model COPY of the paraphrase bank
        |                          # (bank intersect this model's correct set)
        +-- paraphrases.npz        # layer-wise encodings of those paraphrases
```

## Lifecycle

- **Wiped at run start.** Each combination's Step-1 `run.py` deletes this
  folder (and the combination's Step-2/3/4 results) before doing anything -
  a rerun never mixes leftovers from previous runs or other models.
- **Never committed.** The whole folder is git-ignored (fully reproducible
  from the read-only sources + the checkpoints).
- **Not the place for sources or results.** Sources live in `Datasets/`
  (read-only); metrics, plots and the final report live under each step's
  `results/` folder.
