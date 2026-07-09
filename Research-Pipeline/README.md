# 🔬 Research-Pipeline

The executable side of the research: Phase A (diagnosis) is fully
implemented; Phase B (targeted fix) is derived from Phase A's Credibility
results and is specified in its README.

## 📁 Structure

| Directory | Role | README |
|-----------|------|--------|
| [`common/`](common/README.md) | All shared logic - config loading, data loading, model utils, training, filtering, encoding, paraphrase inference, alignment | [→ README](common/README.md) |
| [`PhaseA/`](PhaseA/README.md) | Diagnosis: 4 steps from fine-tuning to joint diagnosis | [→ README](PhaseA/README.md) |
| [`PhaseB/`](PhaseB/README.md) | Constructive fix - pending Phase A results | [→ README](PhaseB/README.md) |

## 🚀 Meta-Runner

```bash
python run_pipeline.py                             # EVERYTHING, end to end
python run_pipeline.py --steps 3,4                 # analysis only
python run_pipeline.py --models BERT-base --datasets SNLI
python run_pipeline.py --force                     # redo after Optuna tuning
```

`run_pipeline.py` is the orchestrator: it checks the raw data and downloads
what is missing (stage 0), runs Step-1 (resumable; official published
checkpoints used where declared), auto-generates + verifies missing
paraphrase files before Step-2 (disable with `--static-paraphrases`), then
runs Steps 2-4 including every plot. Finished stages skip themselves;
`--force` cascades a full redo.

## 🧩 Design Rule

Per-combination scripts are **thin wrappers** (set `MODEL_KEY`,
`DATASET_KEY`, call `common/`). Metrics live once in
[`eval_metrics/`](../eval_metrics/README.md). Configuration lives once in
[`configs/`](../configs/README.md). Nothing is duplicated.
