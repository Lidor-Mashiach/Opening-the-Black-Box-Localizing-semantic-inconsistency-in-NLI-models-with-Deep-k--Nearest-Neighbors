# Research-Pipeline

The executable side of the research: Phase A (diagnosis) is fully
implemented; Phase B (targeted fix) is derived from Phase A's Credibility
results and is specified in its README.

## Structure

| Directory | Role | README |
|-----------|------|--------|
| [`common/`](common/README.md) | All shared logic - config loading, data loading, model utils, training, filtering, encoding, paraphrase inference, alignment | [-> README](common/README.md) |
| [`PhaseA/`](PhaseA/README.md) | Diagnosis: 4 steps from fine-tuning to joint diagnosis | [-> README](PhaseA/README.md) |
| [`PhaseB/`](PhaseB/README.md) | Constructive fix - pending Phase A results | [-> README](PhaseB/README.md) |

## Meta-Runner

```bash
python run_pipeline.py                             # EVERYTHING, end to end
python run_pipeline.py --steps 3,4                 # analysis only
python run_pipeline.py --models BERT-base --datasets SNLI
```

`run_pipeline.py` is the **flagless** orchestrator - `--steps` / `--models` /
`--datasets` are the only (optional) subset selectors. It verifies the raw
data (downloads what is missing), verifies the static paraphrase banks exist
(**exits** with the exact build command otherwise - it never generates them),
then per combination: **wipes the Runtime-Data workspace** (no leftovers
between runs or models), trains (crash-resume; skips when the checkpoint
exists; official published checkpoints used where declared), builds the
reduced copy + encodings + the per-model paraphrase copy, and runs Steps 2-4
including every plot and the consolidated final report. `Datasets/` is never
modified; to retrain a combination, delete its `results/checkpoints/`.

## Repeating the Research - Just Run It Again

Every run draws a **fresh random seed** (no flag) and fine-tunes every
trainable combination from scratch with it, so runs are independent samples.
At the end, `PhaseA/Step-4_.../archive_run.py` freezes that run into
`results/runs/run_<stamp>_seed_<seed>/` (report + all plots + `run_info.json`)
and rewrites `results/runs_summary.csv` with `<metric>_mean` / `_std` / `_var`,
`n_runs`, the seed list and the `signals_agree` agreement rate. Runs are never
overwritten or mixed - run 5-6 times, whenever, and the statistics stay
current. `NLI_SEED` in the environment pins a run (SLURM requeue resumes the
same sample; set it yourself to reproduce a past run).

## Design Rule

Per-combination scripts are **thin wrappers** (set `MODEL_KEY`,
`DATASET_KEY`, call `common/`). Metrics live once in
[`eval_metrics/`](../eval_metrics/README.md). Configuration lives once in
[`configs/`](../configs/README.md). Nothing is duplicated.
