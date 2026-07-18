# tuning - Optuna Hyper-Parameter Search (BGU SLURM)

**What this produces: the best hyper-parameters, NOT a trained model.**
Every trial fine-tunes on a seeded SUBSAMPLE only to *score* one parameter
set; the study's product is `configs/tuned/<MODEL>__<DATASET>.yaml`. The real
fine-tune happens later, inside the pipeline, using those values. Tuning is
**optional** - without it the pipeline trains on the literature defaults in
`configs/base.yaml` and says so in a loud yellow log line.

## Submit (9 studies - one per TRAINABLE combination)

```bash
cd tuning
bash run_all_sbatch.sh        # comment out unwanted jobs first
squeue --me                   # monitor        (cancel: scancel <job_id>)
```

**Nine jobs, not twelve:** the three large-model MNLI combinations use the
labs' official published checkpoints and are never trained, so they are not
tuned. Each job is self-contained - it downloads its backbone from the Hub and
fine-tunes on a subsample to score hyper-parameters; it needs nothing
pre-existing.

`run_all_sbatch.sh` copies the `.sbatch` files to your cluster scripts folder
and submits them from there; the files carry absolute paths:

| Path | Set to |
|------|--------|
| sbatch files deployed to | `/home/lidorma/sbatches_and_output_files/NLU-Scripts/sbatch-files/` |
| job logs (`%x-%J.out`) | `/home/lidorma/sbatches_and_output_files/NLU-Scripts/output-files/` |
| repo (`cd` target) | `/home/lidorma/projects/NLU-Project` |
| conda env | `nlu_env` |

Both folders are created automatically. If your layout differs, edit the
constants at the top of `tuning/build_sbatch.py` and run
`python tuning/build_sbatch.py` (rewrites all 9 files consistently), and
update the two paths at the top of `run_all_sbatch.sh`.

## Job Resources - and Why

| Setting | BERT-base | Large models | Reason |
|---------|-----------|--------------|--------|
| `--gpus` | `1` (any GPU) | `rtx_3090:1` (24GB) | Fine-tuning is a GPU task. A large-model trial peaks at ~9-12GB VRAM (fp32 weights + AdamW states + fp16 activations at batch 16 / seq 128) -> a 24GB-class card; BERT-base needs < 6GB -> any GPU. Requesting the type via `--gpus=<type>:1` is the BGU-documented way (no `--constraint` needed) |
| `--mem` | 16G | 32G | System RAM (dataset + tokenization + dataloader workers), not VRAM |
| CPUs | (auto) | (auto) | Asking for a GPU already allocates 4-6 CPUs - no `--cpus-per-task` needed |
| `--time 6-23:59:00` |  |  | The effective 7-day maximum, one minute under the cap so submission is never rejected |
| `--mail-type=END,FAIL` |  |  | Email when a study finishes or fails |

Hit the 7-day wall mid-study? Submit that job again - `run_tuning.py` (the
code, not the job) resumes the sqlite study from the next trial. There is no
`--requeue`: on `main` no one preempts you, and the resume logic lives in the
code, so a fresh submission simply continues.

## The Search Space (`configs/base.yaml` -> `tuning:`)

Objective: **maximize validation accuracy** of the fine-tuned NLI classifier,
on a seeded validation subsample. Every range is the standard fine-tuning
range from the BERT / RoBERTa / DeBERTa papers - wide enough to matter,
narrow enough not to waste GPU-days on values nobody uses.

| Parameter | Range | Why |
|-----------|-------|-----|
| `learning_rate` | 5e-6 .. 5e-5, **log**-uniform | The one parameter that dominates transformer fine-tuning. Log scale because the useful values are multiplicative (1e-5 vs 2e-5 matters; 3.1e-5 vs 3.2e-5 does not). Large models like the low end, BERT-base the high end - the search finds each |
| `weight_decay` | 0.0 .. 0.1 | BERT used 0.01, RoBERTa 0.1 - the whole published span |
| `warmup_ratio` | 0.0 .. 0.2 | RoBERTa 0.06, BERT 0.1; stabilizes early large-model steps |
| `epochs` | 2 .. 3 | The published fine-tuning window; more overfits NLI |
| `batch_size` | BERT-base [16, 32] / large [8, 16] | Bounded by VRAM at seq 128 |

| Knob | Value | Meaning |
|------|-------|---------|
| `n_trials` | 40 | ~1.5-2.5h per large-model trial -> fits the 7-day window |
| `train_subsample` | 80000 | Seeded rows per trial - the ranking of parameter sets is what matters, not the last accuracy decimal |
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
