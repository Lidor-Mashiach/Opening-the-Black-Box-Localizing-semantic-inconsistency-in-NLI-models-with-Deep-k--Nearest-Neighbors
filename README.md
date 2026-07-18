# Opening the Black Box

> **Localizing semantic inconsistency in NLI models with Deep k-Nearest Neighbors** - first adaptation of DkNN to Natural Language Inference

---

## Table of Contents

- [ Purpose](#-purpose)
- [ Research Questions](#-research-questions)
- [ Authors](#-authors)
- [ Project Structure](#-project-structure)
- [ Models & Datasets](#-models--datasets)
- [ Metrics](#-metrics)
- [ The Pipeline](#-the-pipeline)
- [ Paraphrase Data Status](#-paraphrase-data-status)
- [ How to Run](#-how-to-run)
- [ Outputs](#-outputs)
- [ License](#-license)

---

## Purpose

NLI models score high on standard benchmarks yet **flip their predictions
under meaning-preserving rewordings** - a phenomenon called *semantic
inconsistency*, documented across dozens of models and datasets
(Arakelyan et al., EACL 2024 - the anchor paper - and many others).

Everyone measured the result. **No one localized the cause.** This project
opens the black box: we adapt **Deep k-Nearest Neighbors** (Papernot &
McDaniel, 2018) to NLI for the first time and ask, at every layer of the
model, *where* the inconsistency is born.

**Two internal lenses:**

```
Premise : Hypothesis          Premise : Paraphrase
        |                             |
        v                             v
   [ fine-tuned NLI model, all layers recorded ]
        |                             |
        +-------------+---------------+
                      |
        +-------------+---------------+
        v                             v
  Layer Distance                DkNN Credibility
  WHERE do the two              WHERE does the paraphrase
  representations split?        land relative to training data?
        |                             |
        +-------------+---------------+
                      v
     representation / reasoning / classifier problem
```

Phase A diagnoses; Phase B turns low Credibility into a **targeted** fix and
tests it against a random-augmentation baseline.

---

## Research Questions

**Main:** Does the semantic inconsistency of NLI models arise already in
their internal representations, not only in the final output?

| # | Secondary question |
|---|--------------------|
| Q1 | Is low Credibility associated with semantic inconsistency? |
| Q2 | In which layer region is the problem most pronounced, and what does that imply? |
| Q3 | Can Credibility serve as a signal for targeted model improvement? |

---

## Authors

| Name | ID | Affiliation |
|------|----|-------------|
| Lidor Mashiach | 209280098 | Dept. of Software & Information Systems Engineering, Ben-Gurion University of the Negev |
| Romy Richter | 212876551 | Dept. of Software & Information Systems Engineering, Ben-Gurion University of the Negev |

---

## Project Structure

```
Opening-The-Black-Box/
+-- configs/                   # All experiment knobs (YAML): base -> model -> dataset -> tuned merge
+-- Documents/                 # Methodology, presentation, bibliography, papers, comparison table
+-- main-setup.py              # ONE-COMMAND setup runner: a STEPS list, each step a script in setup-files/
+-- setup-files/               # everything main-setup runs, in one place
|   +-- setup_env.py           # detect OS+GPU, repair torch, install deps, health-check
|   +-- download_datasets.py   # HF hub -> Datasets/<ds>/raw/
|   +-- requirements.txt
|   +-- Paraphrase-Generator/  # one-time builder of the static paraphrase banks
|   +-- README.md              # the full operational guide (run order, provenance, semantics)
+-- Datasets/                  # READ-ONLY sources: the three raw corpora + static paraphrase banks
|   +-- SNLI-Stanford_NLI/  MNLI-MultiGenre_NLI/  ANLI-Adversarial_NLI/
+-- Models/                    # backbones downloaded from HF, stored IN the project (raw/<MODEL>/)
+-- configs/                   # base + per-model + per-dataset + tuned overrides (deep-merged)
+-- Runtime-Data/              # ALL run-derived data: <MODEL>/<DATASET>/, wiped at each run's start
+-- eval_metrics/              # ALL metrics, one file each (RFR, SFR, PC, PA, Layer Distance, Credibility)
+-- Research-Pipeline/
|   +-- common/                # every piece of shared logic - written once
|   +-- PhaseA/                # diagnosis: 4 steps, 12 combination folders, meta-runners
|   +-- PhaseB/                # targeted fix - specified, pending Phase A results
|   +-- run_pipeline.py        # MAIN RESEARCH RUNNER: a PHASES list (Phase A active, Phase B commented)
+-- tuning/                    # Optuna studies per combination + BGU sbatch jobs + bash runner
+-- LICENSE
+-- README.md                  <- you are here
```

Each directory has its own README with detailed documentation:

| Directory | What's inside | README |
|-----------|--------------|--------|
| [`configs/`](configs/README.md) | Base config + per-model + per-dataset overrides, merge rules | [-> README](configs/README.md) |
| [`Documents/`](Documents/README.md) | Methodology, presentation, bibliography, 14-paper index, comparison table | [-> README](Documents/README.md) |
| [`Datasets/`](Datasets/README.md) | READ-ONLY sources: the three raw corpora + the static paraphrase banks | [-> README](Datasets/README.md) |
| [`setup-files/Paraphrase-Generator/`](setup-files/Paraphrase-Generator/README.md) | Standalone one-time bank builder: gates, its own config, audit-file schema | [-> README](setup-files/Paraphrase-Generator/README.md) |
| [`Models/`](Models/README.md) | Backbones downloaded from HF, stored in the project; tuned configs live in `configs/tuned/` | [-> README](Models/README.md) |
| [`Runtime-Data/`](Runtime-Data/README.md) | ALL run-derived data per model x dataset - wiped every run | [-> README](Runtime-Data/README.md) |
| [`eval_metrics/`](eval_metrics/README.md) | Every metric + its source paper + sanity-check runner | [-> README](eval_metrics/README.md) |
| [`Research-Pipeline/`](Research-Pipeline/README.md) | Phase A steps, Phase B spec, shared logic | [-> README](Research-Pipeline/README.md) |
| [`tuning/`](tuning/README.md) | Optuna hyper-parameter search: objective, per-feature search space, sbatch jobs, main bash runner | [-> README](tuning/README.md) |

---

## Models & Datasets

4 models x 3 datasets = **12 experiment combinations**, each with its own
folder, config merge and results. Chosen as the most frequent across the
reviewed literature (see [`Documents/Comparison-Table/`](Documents/Comparison-Table/README.md)).

**Backbones** (general language pretraining by the original labs; fine-tuned
per dataset by Step-1). Every `hf_id` is the creator's official Hub page -
resolved live at download time, so these are the authentic weights, not a
copy that could drift:

| Model | HF id (click = proof) | Pretrained by | Layers | Pooling |
|-------|----------------------|---------------|--------|---------|
| BERT-base | [`bert-base-uncased`](https://huggingface.co/bert-base-uncased) | Google | 12 | `cls` |
| RoBERTa-large | [`roberta-large`](https://huggingface.co/roberta-large) | Meta AI (FacebookAI) | 24 | `cls` |
| DeBERTa-large | [`microsoft/deberta-large`](https://huggingface.co/microsoft/deberta-large) | Microsoft | 24 | `cls` |
| BART-large | [`facebook/bart-large`](https://huggingface.co/facebook/bart-large) | Meta AI | 12 dec. | `eos` |

**Official MNLI checkpoints** - downloaded and used as-is (no local training)
for the three large models on MNLI. These are the labs' own published
fine-tuned NLI models:

| Combination | HF checkpoint (click = proof) | Fine-tuned on MNLI by |
|-------------|-------------------------------|----------------------|
| RoBERTa-large x MNLI | [`FacebookAI/roberta-large-mnli`](https://huggingface.co/FacebookAI/roberta-large-mnli) | Meta AI |
| DeBERTa-large x MNLI | [`microsoft/deberta-large-mnli`](https://huggingface.co/microsoft/deberta-large-mnli) | Microsoft |
| BART-large x MNLI | [`facebook/bart-large-mnli`](https://huggingface.co/facebook/bart-large-mnli) | Meta AI |

The other nine combinations are fine-tuned locally by Step-1 (no official
single-dataset SNLI / ANLI checkpoints exist). Backbones download themselves
from the Hub on first use **into the project** (`Models/raw/<MODEL>/`) and are
reused from there; `.gitignore` keeps the heavy weights local while the light
tuned configs (`configs/tuned/`) are committed, so a second machine downloads
only the backbone and never re-runs Optuna. Training enforces a precondition
gate (backbone present -> reuse; tuned config but no backbone -> download;
neither -> red error + exit); see [setup-files/README.md](setup-files/README.md).

| Dataset | HF id | Difficulty | Note |
|---------|-------|-----------|------|
| SNLI | `stanfordnlp/snli` | Easiest | image-caption premises |
| MNLI | `nyu-mll/multi_nli` | Medium | ten genres; mismatched-dev = test |
| ANLI | `facebook/anli` | Hardest | adversarial, rounds r1-r3 |

Datasets are kept **separate** - merging would blur where any improvement
came from.

Every combination is fine-tuned locally by Step-1, except where an
**official, single-dataset** published checkpoint exists (MNLI on the three
large models) - full provenance table with links in
[`configs/README.md`](configs/README.md#-published-checkpoint-provenance-official-sources).

---

## Metrics

| Metric | Level | Source | File |
|--------|-------|--------|------|
| Relaxed Fooling Rate | output | Anchor paper | `eval_metrics/relaxed_fooling_rate.py` |
| Strict Fooling Rate | output | Anchor paper | `eval_metrics/strict_fooling_rate.py` |
| Paraphrastic Consistency (PC) | output | Srikanth et al. 2024 | `eval_metrics/paraphrastic_consistency.py` |
| Pattern Accuracy (PA) | output | MERGE 2025 | `eval_metrics/pattern_accuracy.py` |
| Layer Distance | internal | this work | `eval_metrics/layer_distance.py` |
| DkNN Credibility | internal | Papernot & McDaniel 2018 | `eval_metrics/dknn_credibility.py` |

---

## The Pipeline

```
Step-1  Train -> Filter -> Encode        (per model x dataset)
        fine-tune | keep only correctly-classified train rows
        store every kept hypothesis at EVERY layer  -> the DkNN bank
                         |
Step-2  Paraphrase Inference
        (premise, paraphrase) -> predictions + layer representations
                         |
Step-3  DkNN & Layer Distance
        choose K on validation | Credibility + drops + scenario
        per-layer distances + crossing layer -> region
                         |
Step-4  Consistency & Diagnosis
        RFR / SFR / PC / PA | joint verdict + profile plot
                         |
Phase B (pending Phase A)
        targeted paraphrases for low-credibility hypotheses only
        fine-tune -> rebuild bank -> re-test vs TWO baselines
```

---

## Paraphrase Data Status

**Verified (July 2026): no public, verified corpus provides ~5
relation-preserving paraphrases per hypothesis for SNLI / MNLI / ANLI.**

| Resource | Offers | Not a drop-in because |
|----------|--------|----------------------|
| [Anchor paper](https://arxiv.org/abs/2401.14440) (EACL 2024) | Semantics-preserving variations verified by symmetric equivalence entailment | Generated per evaluated model, on the fly - no static release |
| [ParaNlu](https://arxiv.org/abs/2404.11717) (TACL 2024) | 7,782 human-validated label-preserving paraphrases | Abductive / defeasible NLI, not 3-class; ~1,000 base problems |
| [MERGE / AllVar](https://arxiv.org/abs/2510.24295) (2025) | MLM-based minimal expression replacements | SNLI-derived only |

**Adopted protocol** (mirrors the anchor paper), implemented in
[`setup-files/Paraphrase-Generator/generate_paraphrases.py`](setup-files/Paraphrase-Generator/generate_paraphrases.py) as a
**static, dataset-level paraphrase bank** - itself a shareable contribution:
sample a `pool_size` (4,000) of hypotheses per dataset from the raw train
split -> 10 candidates each from the leading open paraphraser
(`humarin/chatgpt_paraphraser_on_T5_base`) -> a length gate keeping
complexity at the hypothesis's own level (word-ratio 0.6-1.5) -> a **double
verification** with the strongest public NLI verifier
(`MoritzLaurer/DeBERTa-v3-large-mnli-fever-anli-ling-wanli`): bidirectional
entailment (same meaning) **and** (premise, candidate) label == gold (same
logical relation) -> as many FRESH-candidate rounds as needed (new seed +
temperature ramp per round, bounded by `max_generation_rounds`) -> **EXACTLY 5
verified paraphrases per hypothesis** (uniform; dropping is only a last-resort
safety valve for degenerate stragglers, expected ~0).
The bank is protected like the raw data: never overwritten (explicit
`--rebuild` only, with an automatic `.backup`). Each model then consumes only the rows
it classified correctly - so all models share the exact same paraphrases for
shared hypotheses. Full contract:
[`Datasets/*/paraphrases/README.md`](Datasets/SNLI-Stanford_NLI/paraphrases/README.md).
The main runner VERIFIES the banks exist and exits with build instructions
otherwise - it never generates them; knobs + the audit-file schema live in
[`setup-files/Paraphrase-Generator/README.md`](setup-files/Paraphrase-Generator/README.md).

---

## How to Run

> **Before anything else, run `python main-setup.py`** to install dependencies
> and build the assets. The full operational picture - run order, dataset &
> model provenance, tuning, rerun semantics - lives in
> **[setup-files/README.md](setup-files/README.md)**.

### One command for setup: `main-setup.py`

```bash
python main-setup.py       # runs Steps 0-2 below, each skipped if already done
```

Prepares everything up to the experiment - environment, datasets, paraphrase
banks - with a "is this needed?" check per step, colored logs, and a final
printout of the two remaining stages (tuning + experiment). The individual
steps are below; run them directly if you prefer.

### Step 0: Environment - one command, any machine

```bash
python setup_env.py        # Windows PowerShell / Linux terminal / BGU cluster
```

Detects the OS + GPU + driver, installs or **repairs** the right torch build
(the CUDA wheel index is discovered live - nothing hardcoded), installs every
dependency, verifies each one actually loads, and prints the `[DEVICE]`
verdict. Do not run `pip install -r setup-files/requirements.txt` yourself - it is one
internal stage of this script.

### Step 1: Data

```bash
python setup-files/download_datasets.py       # SNLI + MNLI + ANLI -> raw/
```

The canonical corpora, from their creators' own HuggingFace organizations,
frozen to read-only parquet. Optional - the pipeline does it automatically
when raw data is missing.

### Step 2: One-Time Static Input - the Paraphrase Banks

```bash
python setup-files/Paraphrase-Generator/generate_paraphrases.py     # once, on a GPU machine
```

EXACTLY 5 verified paraphrases per pooled hypothesis, triple-gated. **Commit
the banks afterwards** - they are the static input of every run, never
regenerated. The pipeline verifies they exist and EXITS with this exact
command otherwise.

### Step 3: Hyper-Parameter Tuning (recommended, optional - BGU SLURM)

```bash
cd tuning && bash run_all_sbatch.sh      # 12 GPU jobs, one per combination
```

Optuna searches the best hyper-parameters **per model x dataset** and writes
`configs/tuned/<COMBO>.yaml`. Nine jobs, not twelve - the three large-model
MNLI combinations use official published checkpoints and are never trained.
Each job is self-contained (it downloads its own backbone and fine-tunes on a
subsample to score parameters); it produces **configs, not weights**, written
straight into `configs/tuned/` where the pipeline reads them - nothing copied
by hand. Skip it and the pipeline trains on the published defaults, saying so
in a loud yellow log line.

### Step 4: The Experiment - one command, end to end

```bash
python Research-Pipeline/run_pipeline.py
```

**Every run is an independent research sample**: a fresh random seed is drawn
automatically, every trainable combination fine-tunes from the official
backbone with this run's seed (the three large-model MNLI combinations use
the labs' published checkpoints as-is), the workspace is wiped, and all
metrics, plots and the final report are recomputed. The run then **archives
itself** to `results/runs/run_<stamp>_seed_<seed>/` and refreshes
`runs_summary.csv` - mean / std / variance across every run so far. Run it 5-6
times to average; nothing is ever overwritten or mixed. Colored logs show
model / dataset / step / action at every moment.

Useful variants (the ONLY flags):

```bash
python Research-Pipeline/run_pipeline.py --models BERT-base --datasets SNLI
python Research-Pipeline/run_pipeline.py --steps 3,4            # analysis only (CPU)
```

> Full detail on all of the above - dataset & model provenance, where the
> weights live, what tuning really produces, exact rerun semantics - is in
> **[setup-files/README.md](setup-files/README.md)**.

Or run any level directly - every folder has its own meta-runner:
`run_phase_a.py`, `run_step{1..4}.py`, or a single combination's
`Step-1_Train-Filter-Encode/BERT-base__SNLI/run.py` (recommended per SLURM
job on the BGU cluster).

Configuration lives in [`configs/`](configs/README.md) - sampling caps, K
candidates, training hyper-parameters. Nothing is hard-coded.

---

## Outputs

| Where | What |
|-------|------|
| `Step-1_*/<COMBO>/results/` | checkpoint, train metrics, baseline accuracy |
| `Runtime-Data/<MODEL>/<DATASET>/` | ALL run-derived data: reduced dataset copy, hypothesis + paraphrase encodings, per-model paraphrase copy (wiped at each run's start) |
| `Step-2_*/results/<COMBO>/` | paraphrase predictions + consistent / strict-flip flags |
| `Step-3_*/results/<COMBO>/` | chosen K, credibility table + summary, layer profile + summary |
| `Step-4_*/results/` | aggregated `consistency_metrics.csv`; per-combo `diagnosis.json`, distance-profile plot + credibility-groups plot; unified `layer_distance_overview.png` + `credibility_overview.png` across all combinations; consolidated `final_report.csv`; per-run archives in `results/runs/run_<stamp>_seed_<seed>/`; and `runs_summary.csv` - mean / std / variance across every run |
| `tuning/results/<COMBO>/` | resumable Optuna study, `best_params.yaml`, `trials.csv` (best values auto-land in `configs/tuned/`) |

The headline number of Phase A lives in `credibility_summary.json`:
the correlation, on the **held-out test portion**, between the credibility
drop and actual inconsistency - can DkNN spot the vulnerable examples
*before* any paraphrase is shown?

---

## License

MIT - see [LICENSE](LICENSE) for details.
