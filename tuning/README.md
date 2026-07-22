# tuning - Optuna Hyper-Parameter Search (BGU SLURM)

**What this produces: the best hyper-parameters, NOT a trained model.**
Every trial fine-tunes on a seeded SUBSAMPLE only to *score* one parameter
set - the study's product is `configs/tuned/<MODEL>__<DATASET>.yaml`. The real
fine-tune happens later, inside the pipeline, using those values. Tuning is
**optional** - without it the pipeline trains on the literature defaults in
`configs/base.yaml` and says so in a loud yellow log line.

## Submit (9 studies - one per TRAINABLE combination)

```bash
# submit every tuning study (or pick individual ones)
for f in tuning/sbatch/tune_*.sbatch - do sbatch "$f" - done
squeue --me                   # monitor        (cancel: scancel <job_id>)
```

**Nine jobs, not twelve:** the three large-model MNLI combinations use the
labs' official published checkpoints and are never trained, so they are not
tuned. Each job is self-contained - it downloads its backbone from the Hub and
fine-tunes on a subsample to score hyper-parameters - it needs nothing
pre-existing.

The nine static `.sbatch` files live in `tuning/sbatch/`. They carry absolute
paths:

| Path | Set to |
|------|--------|
| job logs (`%x.out`) | `/home/lidorma/sbatches_and_output_files/NLU-Scripts/output-files/` |
| repo (`cd` target) | `/home/lidorma/projects/NLU-Project` |
| conda env | `nlu_env` |

If your layout differs, edit these paths inside the `.sbatch` files (they are
plain SLURM scripts - one per combination).

## Job Resources - and Why

| Setting | BERT-base | Large models | Reason |
|---------|-----------|--------------|--------|
| `--gpus` | `1` (any GPU) | `rtx_3090:1` (24GB) | Fine-tuning is a GPU task. A large-model trial peaks at ~7-9GB VRAM (fp32 weights + AdamW states + bf16 activations at batch 16 / seq 128 - DeBERTa runs fully fp32 - see note - at ~7GB) -> a 24GB-class card - BERT-base needs < 6GB -> any GPU. Requesting the type via `--gpus=<type>:1` is the BGU-documented way (no `--constraint` needed). NOTE: request the SAME GPU type for every job - a job landing on a card the installed PyTorch was not built for fails with `no kernel image is available` |
| `--mem` | 16G | 32G | System RAM (dataset + tokenization + dataloader workers), not VRAM |
| CPUs | (auto) | (auto) | Asking for a GPU already allocates 4-6 CPUs - no `--cpus-per-task` needed |
| `--time 6-23:59:00` |  |  | The effective 7-day maximum, one minute under the cap so submission is never rejected |
| `--mail-type=END,FAIL` |  |  | Email when a study finishes or fails |

Hit the 7-day wall mid-study? Submit that job again - `run_tuning.py` (the
code, not the job) resumes the sqlite study from the next trial. There is no
`--requeue`: on `main` no one preempts you, and the resume logic lives in the
code, so a fresh submission simply continues.

**Reset for a fresh optimum without losing progress:** if you delete
`tuning/results/<COMBO>/optuna.db` and resubmit (e.g. to re-search from
scratch), the study still exploits what was already found - on startup it reads
the existing `configs/tuned/<COMBO>.yaml`, enqueues its parameters as trial 0
(so TPE explores *around* the known optimum, not from zero), and treats its
recorded accuracy as a write floor: the overlay is overwritten only by a strict
improvement, so a reset can never downgrade a better past result. This only
applies when the recorded parameters still fit the current search space - if they
don't (e.g. the batch options changed), the study starts clean and says so.

**Evaluation throughput:** the per-epoch validation pass (which enables
pruning) runs a much larger batch than training - it is forward-only, so it uses
16x the train batch (capped) with a wide VRAM margin. This is pure throughput
and never touches the tuned search space.

## The Search Space (`configs/base.yaml` -> `tuning:`)

Objective: **maximize validation accuracy** of the fine-tuned NLI classifier,
on a seeded validation subsample. Every range is the standard fine-tuning
range from the BERT / RoBERTa / DeBERTa papers - wide enough to matter,
narrow enough not to waste GPU-days on values nobody uses.

| Parameter | Range | Why |
|-----------|-------|-----|
| `learning_rate` | 5e-6 .. 5e-5, **log**-uniform | The one parameter that dominates transformer fine-tuning. Log scale because the useful values are multiplicative (1e-5 vs 2e-5 matters - 3.1e-5 vs 3.2e-5 does not). Large models like the low end, BERT-base the high end - the search finds each |
| `weight_decay` | 0.0 .. 0.1 | BERT used 0.01, RoBERTa 0.1 - the whole published span |
| `warmup_ratio` | 0.0 .. 0.2 | RoBERTa 0.06, BERT 0.1 - stabilizes early large-model steps |
| `epochs` | 2 .. 3 | The published fine-tuning window - more overfits NLI |
| `batch_size` | BERT-base [16, 32] / large [8, 16] | Bounded by VRAM at seq 128 |

| Knob | Value | Meaning |
|------|-------|---------|
| `n_trials` | 30 | ~1.5-2.5h per large-model trial -> fits the 7-day window |
| `train_subsample` | 50000 | Seeded rows per trial - the ranking of parameter sets is what matters, not the last accuracy decimal |
| `val_subsample` | 5000 | Seeded validation rows for the objective |

Not tuned on purpose: `max_seq_len` (128 is the NLI standard - a design
choice, not a knob) and `seed` (tuning a seed would be tuning noise).

## Outputs

| File | Content |
|------|---------|
| `configs/tuned/<COMBO>.yaml` | **The product**: the winning `training:` block. The pipeline's config merge picks it up automatically - nothing to copy by hand |
| `tuning/results/<COMBO>/optuna.db` | The sqlite study (resume lives here) |
| `tuning/results/<COMBO>/best_params.yaml` | The best trial, human-readable, with its score |

## After Tuning

Nothing to do - the next `run_pipeline.py` run reads `configs/tuned/`
automatically and logs `hyper-parameters: TUNED by Optuna (...)` in green.
Every run fine-tunes fresh from the pretrained backbone anyway, so tuned
values take effect immediately.