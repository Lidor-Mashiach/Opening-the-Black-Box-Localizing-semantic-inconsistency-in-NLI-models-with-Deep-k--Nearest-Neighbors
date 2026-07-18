# SETUP - Run Order, Sources, Provenance & Run Semantics

Everything operational in one page: **what to run, in what order**, where
every byte comes from, who trained what, what you must change by hand
(spoiler: **nothing**), and exactly what a rerun does. The research itself is
described in [README.md](../README.md).

##  The Order at a Glance

| # | Command | What it does | When | Machine |
|---|---------|--------------|------|---------|
|  | `python main-setup.py` | **One command for steps 0-3 below** - each auto-skipped if already done | recommended entry point | any |
| 0 | `python setup-files/setup_env.py` | The WHOLE environment: torch build + all dependencies + health checks | once per machine | any |
| 1 | `python setup-files/download_datasets.py` | Raw SNLI / MNLI / ANLI -> `Datasets/<ds>/raw/` | once (auto if skipped) | any |
| 2 | *(nothing)* | Models download themselves from the HuggingFace Hub on first use | - | - |
| 3 | `python setup-files/Paraphrase-Generator/generate_paraphrases.py` | The static paraphrase banks -> then **commit them** | once, ever | GPU |
| 4 | `cd tuning && bash run_all_sbatch.sh` | Optuna (9 trainable combos): best hyper-parameters -> `configs/tuned/` | once (recommended, optional) | cluster |
| 5 | `python Research-Pipeline/run_pipeline.py` | The experiment: fresh seed -> fine-tune -> encode -> DkNN -> metrics -> archive | **every time you want another sample** | GPU |

Steps 0-4 build the assets. Step 5 is the research, repeatable at will.

> **`main-setup.py` runs steps 0-3 for you**, in order, each with a
> "is this needed?" check (datasets skipped if already downloaded, banks
> skipped if already built), colored per-step logs, and a final printout of
> the two remaining stages (tuning + experiment). It deliberately stops
> before them - see "Why tuning is separate" under Step 4.

---

## What Lives in setup-files/

Everything `main-setup.py` runs, in one folder:

```
setup-files/
|-- setup_env.py               # Step 1: OS+GPU detect, torch repair, deps, health-check
|-- download_datasets.py       # Step 2: HF hub -> Datasets/<ds>/raw/
|-- requirements.txt           # the dependency list setup_env installs
|-- Paraphrase-Generator/      # Step 3: the static paraphrase banks
|   |-- generate_paraphrases.py
|   |-- paraphrase_config.yaml
|   +-- README.md
+-- README.md                  # this file
```

`main-setup.py` (in the project root) is a pure runner: its `STEPS` list calls
these three scripts in order. Each script resolves the project root on its own
and writes its outputs to the right place (raw data to `Datasets/`, banks to
`Datasets/<ds>/paraphrases/`), so they work identically whether run through
`main-setup.py` or directly.

## Step 0 - `setup_env.py`: the WHOLE environment, one command

**Do NOT run `pip install -r setup-files/requirements.txt` yourself** - it is one internal
stage of `setup_env.py`, executed *after* the torch build is fixed. Running it
alone can leave a broken (CPU-only) torch in place, because pip considers an
already-installed torch "satisfied".

1. Detects the machine: OS, Python, NVIDIA GPU(s), driver CUDA version.
2. **Discovers PyTorch's live CUDA wheel indexes** (nothing hardcoded),
   filters by what the driver supports, probes newest-first with
   `pip --dry-run`, and only then installs/replaces torch. A failed attempt
   never leaves the environment without torch.
3. Installs the rest of `setup-files/requirements.txt`.
4. **Import health check** - every core library must actually LOAD. On
   Windows, Smart-App-Control DLL blocking is detected, an automatic
   `Unblock-File` pass is attempted, and the exact fix is printed. On Linux
   (the cluster) it is a quiet verification.
5. Prints the `[DEVICE]` verdict (GPU + VRAM + CUDA) and runs the metric
   sanity check.

Idempotent - safe to re-run. `--cpu-only` for machines without NVIDIA.

---

## Step 1 - Raw datasets: where from, and why the source is trustworthy

| Dataset | HF Hub source | Published & maintained by | Paper |
|---------|---------------|---------------------------|-------|
| SNLI | `stanfordnlp/snli` | Stanford NLP Group - the dataset's creators | Bowman et al., EMNLP 2015 |
| MNLI | `nyu-mll/multi_nli` | NYU Machine Learning for Language - the creators | Williams et al., NAACL 2018 |
| ANLI | `facebook/anli` | Facebook AI Research (Meta) - the creators | Nie et al., ACL 2020 |

These are the canonical, original-author organizations on the Hub - the exact
distribution the community cites; downloads are checksum-verified. The
downloader snapshots every split to read-only parquet with the SAME loader the
pipeline uses (unlabeled `-1` rows dropped, ANLI rounds r1-r3 concatenated),
so every run forever sees identical bytes. Running it manually is optional -
`run_pipeline.py` does it automatically when raw data is missing.

---

## Step 2 - Models: nothing to run, nothing to store, full provenance

**Where do the backbones live? INSIDE the project**, at
`Models/raw/<MODEL>/`. On first need the HuggingFace weights (config +
tokenizer + safetensors) are downloaded straight into that project folder;
every later run reuses the local copy. The HuggingFace home cache is never
used - everything the code needs is in the project, in a clear hierarchy.
`.gitignore` keeps the heavy weights local (folder + README tracked); deciding
what to push is `.gitignore`'s job, not a reason to put files outside the
repo. Set `HF_TOKEN` to raise Hub rate limits on the first download.

### The backbones - general language pretraining, by the original labs

| Model | HF source (in `configs/models/*.yaml` -> `hf_id`) | Pretrained by |
|-------|--------------------------------------------------|---------------|
| BERT-base | `bert-base-uncased` | Google |
| RoBERTa-large | `roberta-large` | Meta AI (FacebookAI) |
| DeBERTa-large | `microsoft/deberta-large` | Microsoft |
| BART-large | `facebook/bart-large` | Meta AI |

These are **not task models**. Pretraining taught them *language* (masked /
denoising objectives on huge corpora) - they know nothing about NLI and have
no classification head. That is why **fine-tuning per dataset is mandatory**,
and it is exactly what Step-1 of the pipeline does: it downloads the official
backbone, attaches a 3-class head, and trains it on SNLI / MNLI / ANLI
separately. There is no "pretraining" step of our own anywhere - that would
be weeks on dozens of GPUs, and it is neither needed nor wanted.

**Are these the authentic weights?** Yes - `hf_id` is the official Hub ID of
the lab that created the model, resolved live at download time (not a copy
sitting in the repo that could drift). Nothing to run, nothing to verify by
hand.

### Official task checkpoints - used as-is (MNLI only)

| Checkpoint | Fine-tuned on MNLI by | Used for |
|------------|----------------------|----------|
| `FacebookAI/roberta-large-mnli` | Meta AI | RoBERTa-large x MNLI |
| `microsoft/deberta-large-mnli` | Microsoft | DeBERTa-large x MNLI |
| `facebook/bart-large-mnli` | Meta AI | BART-large x MNLI |

Declared under `nli_checkpoints:` in `configs/models/*.yaml` and selected
automatically by `resolve_checkpoint()`. These three combinations **never
train** - using the labs' own published NLI models is stronger and more
citable than re-training them ourselves (and they have zero seed-variance by
construction). No official single-dataset SNLI / ANLI checkpoints exist
anywhere - those nine combinations are fine-tuned locally, every run.

### Bank-builder models (Step 3 only)

| Model | What it is | Role |
|-------|-----------|------|
| `humarin/chatgpt_paraphraser_on_T5_base` | T5-base fine-tuned on 6.3M ChatGPT paraphrase pairs - the leading open paraphraser | candidate generator |
| `MoritzLaurer/DeBERTa-v3-large-mnli-fever-anli-ling-wanli` | DeBERTa-v3-large fine-tuned on 885K NLI pairs (MNLI+FEVER+ANLI+LingNLI+WANLI) - the strongest public NLI model | double quality gate |

---

## Step 3 - Paraphrase banks: one-time, on a GPU, then commit

```bash
python setup-files/Paraphrase-Generator/generate_paraphrases.py
```

A seeded pool of hypotheses per dataset -> the paraphraser proposes
candidates -> three gates (length ratio; hypothesis <-> candidate must entail
EACH OTHER; the verifier's (premise, candidate) label must equal the gold
label) -> fresh sampling rounds continue until EVERY hypothesis holds EXACTLY
the quota -> atomic write to `Datasets/<ds>/paraphrases/paraphrase_bank.csv`
+ an audit stats file. Two GLOBALS at the top of the script are the only
scaling switches: `PARAPHRASES_PER_HYPOTHESIS` and the `DATASETS` registry.
Full protocol: [Paraphrase-Generator/README.md](Paraphrase-Generator/README.md).

**Then commit the banks.** They are the static input of the whole research:
identical in every run, never regenerated. The pipeline only VERIFIES they
exist - and exits with this exact command if one is missing.

---

## Step 4 - Optuna: the best hyper-parameters (recommended, optional)

```bash
cd tuning && bash run_all_sbatch.sh      # 9 GPU jobs (trainable combos), up to 7 days each
squeue --me
```

**Tuning does not produce a trained model.** Each trial fine-tunes on a
subsample only to *score* one hyper-parameter set. The product is a config:

```
configs/tuned/<MODEL>__<DATASET>.yaml     # 12 files - one PER COMBINATION
```

Per **combination**, not per model - a learning rate that suits RoBERTa on
SNLI is not the one it needs on adversarial ANLI. Objective: maximize
validation accuracy. Search space (ranges straight from the BERT / RoBERTa /
DeBERTa papers): `learning_rate` 5e-6..5e-5 log-uniform, `weight_decay`
0..0.1, `warmup_ratio` 0..0.2, `epochs` 2..3, `batch_size` [16,32] base /
[8,16] large; 40 trials per study, sqlite-resumable. Details, job resources
and the reasoning: [tuning/README.md](../tuning/README.md).

**What do you copy by hand? NOTHING - and this is by design.** Optuna writes
`configs/tuned/<COMBO>.yaml` directly into the repo's `configs/` tree, which
is exactly where `load_config()` reads it (`base -> model -> dataset -> tuned`,
later wins). The tuning job and the pipeline share that one location, so a
finished study is picked up by the next run with zero manual steps - no moving
files from a scratch dir, no copying weights, nothing. (This is also why it
does not depend on how the SLURM node cache behaves: the winning numbers live
in a committed config file, not in a cache.)

### Why tuning is separate, and how it gets its model (the key point)

A fair question: if Optuna optimizes a model's hyper-parameters, and the model
only downloads inside the pipeline's `train.py`, what does tuning run on?

**Answer: the tuning job is fully self-contained.** Each Optuna trial
downloads the official backbone itself (`AutoModelForSequenceClassification
.from_pretrained(hf_id)` - the same Hub source, the same cache) and fine-tunes
it on a subsample to score one hyper-parameter set. It needs nothing from the
pipeline and nothing pre-existing. So the order is simply:

```
tuning (optional)  ->  writes configs/tuned/<COMBO>.yaml
                              |
experiment (run_pipeline.py)  ->  reads it automatically, trains the REAL model
```

Tuning does NOT hand a trained model to the pipeline - it hands over a small
YAML of numbers. The pipeline still does the real, full fine-tune (Step-1),
now using those numbers. Nothing is downloaded "too early" and nothing is
missing: both stages fetch the backbone from the Hub on demand.

**Only nine jobs.** The three large-model MNLI combinations use official
published checkpoints (Step 2) - no training, so nothing to tune. `tuning/`
contains 9 sbatch files, not 12.

### Moving to another machine (the cross-machine story)

After tuning on machine A you `git commit`. What travels, and what does not:

* `configs/tuned/*.yaml` (light) - COMMITTED. Machine B pulls the optimized
  hyper-parameters for free.
* `Models/raw/<MODEL>/` (heavy weights) - NOT committed (`.gitignore`).

On machine B, `run_pipeline.py` hits the training gate, sees a tuned config
but no local backbone, downloads the backbone into B's project, and trains
with the committed tuned values. No machine ever re-runs Optuna. With neither
a tuned config nor a backbone, training stops with a red error telling you to
run Optuna first - it never silently trains on defaults.

**Is tuning required before the experiment?** For the nine trainable
combinations, yes - the training gate enforces it (red error + exit when no
tuned config and no backbone exist). The three official-checkpoint MNLI
combinations need none. At every training the pipeline prints, in color:

| Log | Meaning |
|-----|---------|
|  `[TRAIN] ... hyper-parameters: TUNED by Optuna (configs/tuned/<COMBO>.yaml)` | Training with the optimized values |
|  `[WARN] ... no tuned config found (expected configs/tuned/<COMBO>.yaml) - training with the base defaults; run the Optuna sbatch jobs to optimize` | Training with the literature defaults from `configs/base.yaml` |

The defaults are sane published values, so a run without tuning is valid -
just not optimized. Recommended order: tune once -> then run the experiment
(and all its repeats) on the tuned values.

---

## Step 5 - The experiment: every run is an independent sample

```bash
python Research-Pipeline/run_pipeline.py         # no flags
```

**A fresh RANDOM seed is drawn automatically on every run** (announced in the
banner). No flag, nothing to remember, no chance of accidentally repeating a
sample.

| Per run | Redone? |
|---------|---------|
| The HF backbone download |  cached |
| The paraphrase banks |  static input, never regenerated |
| Optuna tuning |  tuned values are reused by every run |
| **Fine-tuning each model on its dataset** |  FRESH, from the backbone, with this run's seed |
| Reduced dataset copy, per-layer encodings, per-model paraphrase copy |  rebuilt (`Runtime-Data/` is wiped at each combination's start) |
| Inference, DkNN credibility, layer distances, all metrics, plots, report |  recomputed |

The three official-checkpoint MNLI combinations never train (see Step 2), so
their numbers are identical across runs by construction.

### Results: archived per run, statistics accumulate

```
Research-Pipeline/PhaseA/Step-4_Consistency-and-Diagnosis/results/
+-- final_report.csv                        # the run that just finished
+-- runs/
|   +-- run_<timestamp>_seed_<seed>/        # a complete, frozen record per run
|   |   +-- final_report.csv
|   |   +-- run_info.json
|   |   +-- *.png                           # that run's overview figures
|   |   +-- <MODEL>__<DATASET>/             # that run's per-combination plots
|   +-- ...
+-- runs_summary.csv                        # ALL runs: mean / std / var per metric
```

Nothing is ever overwritten and no two runs are ever mixed: run N+1 archives
itself alongside run N. `runs_summary.csv` is rewritten after every run with,
per model x dataset x metric: `<metric>_mean`, `<metric>_std`, `<metric>_var`,
`n_runs`, the seed list, and - for text columns (scenario / verdict / region)
- the set of values observed. `signals_agree_mean` is the agreement RATE
across runs. Run 5-6 times, whenever you like, even weeks apart: the
statistics are always up to date and every individual run stays intact.

**SLURM note:** a job requeued mid-run inherits its `NLI_SEED` from the
environment, so it resumes the interrupted training instead of silently
becoming a different sample. Setting `NLI_SEED` yourself reproduces a past run
exactly.

---

## What You Change by Hand

| Thing | Where | When |
|-------|-------|------|
| The paraphrase quota / dataset registry | the two GLOBALS in `setup-files/Paraphrase-Generator/generate_paraphrases.py` | before building the banks |
| Cluster paths / conda env / per-model GPU / memory | the constants in `tuning/build_sbatch.py`, then `python tuning/build_sbatch.py` (regenerates all 9) + the two paths atop `tuning/run_all_sbatch.sh` | before submitting the tuning jobs |
| Tuned hyper-parameters | **nothing** - Optuna writes `configs/tuned/`, the config merge picks it up | never |
| Which run's results to keep | **nothing** - every run archives itself | never |

## The Safety Net - you cannot silently run the wrong thing

* Raw data missing -> downloaded automatically.
* Paraphrase banks missing -> the runner prints the exact build command and
  **exits** (it never generates them itself).
* No tuned config -> loud yellow `[WARN]` naming the file it expected.
* Tuned config present -> green `[TRAIN]` line naming the file it used.
* Stale run leftovers -> `Runtime-Data/` + Step-2/3/4 results are wiped at
  each combination's start.
* No GPU -> a loud `[ERROR]` explaining exactly why (CPU-only wheel vs
  driver/allocation), before anything slow begins.
