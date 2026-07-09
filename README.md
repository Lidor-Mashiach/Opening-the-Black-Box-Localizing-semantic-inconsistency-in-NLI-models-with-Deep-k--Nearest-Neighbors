# 🔓 Opening the Black Box

> **Localizing semantic inconsistency in NLI models with Deep k-Nearest Neighbors** — first adaptation of DkNN to Natural Language Inference

---

## 📋 Table of Contents

- [🎯 Purpose](#-purpose)
- [❓ Research Questions](#-research-questions)
- [👥 Authors](#-authors)
- [📁 Project Structure](#-project-structure)
- [🤖 Models & Datasets](#-models--datasets)
- [📏 Metrics](#-metrics)
- [🔄 The Pipeline](#-the-pipeline)
- [⚠️ Paraphrase Data Status](#️-paraphrase-data-status)
- [🚀 How to Run](#-how-to-run)
- [📊 Outputs](#-outputs)
- [📜 License](#-license)

---

## 🎯 Purpose

NLI models score high on standard benchmarks yet **flip their predictions
under meaning-preserving rewordings** — a phenomenon called *semantic
inconsistency*, documented across dozens of models and datasets
(Arakelyan et al., EACL 2024 — the anchor paper — and many others).

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

## ❓ Research Questions

**Main:** Does the semantic inconsistency of NLI models arise already in
their internal representations, not only in the final output?

| # | Secondary question |
|---|--------------------|
| Q1 | Is low Credibility associated with semantic inconsistency? |
| Q2 | In which layer region is the problem most pronounced, and what does that imply? |
| Q3 | Can Credibility serve as a signal for targeted model improvement? |

---

## 👥 Authors

| Name | ID | Affiliation |
|------|----|-------------|
| Lidor Mashiach | 209280098 | Dept. of Software & Information Systems Engineering, Ben-Gurion University of the Negev |
| Romy Richter | 212876551 | Dept. of Software & Information Systems Engineering, Ben-Gurion University of the Negev |

---

## 📁 Project Structure

```
Opening-The-Black-Box/
├── configs/                   # All experiment knobs (YAML): base -> model -> dataset -> tuned merge
├── Documents/                 # Methodology, presentation, bibliography, papers, comparison table
├── Datasets/                  # SNLI / MNLI / ANLI + filtered subsets + paraphrases + encoded banks
│   ├── SNLI-Stanford_NLI/
│   ├── MNLI-MultiGenre_NLI/
│   ├── ANLI-Adversarial_NLI/
│   ├── Encoded_Datasets/      # layer-wise representation banks per model x dataset
│   └── download_datasets.py   # meta-runner: HF hub -> raw/
├── eval_metrics/              # ALL metrics, one file each (RFR, SFR, PC, PA, Layer Distance, Credibility)
├── Research-Pipeline/
│   ├── common/                # every piece of shared logic - written once
│   ├── PhaseA/                # diagnosis: 4 steps, 12 combination folders, meta-runners
│   ├── PhaseB/                # targeted fix - specified, pending Phase A results
│   └── run_pipeline.py        # MAIN RUNNER: data check -> train -> paraphrases -> analysis
├── tuning/                    # Optuna studies per combination + BGU sbatch jobs + bash runner
├── requirements.txt
├── LICENSE
└── README.md                  <- you are here
```

Each directory has its own README with detailed documentation:

| Directory | What's inside | README |
|-----------|--------------|--------|
| [`configs/`](configs/README.md) | Base config + per-model + per-dataset overrides, merge rules | [→ README](configs/README.md) |
| [`Documents/`](Documents/README.md) | Methodology, presentation, bibliography, 14-paper index, comparison table | [→ README](Documents/README.md) |
| [`Datasets/`](Datasets/README.md) | The three corpora, filtered subsets, paraphrase contract, encoded banks | [→ README](Datasets/README.md) |
| [`eval_metrics/`](eval_metrics/README.md) | Every metric + its source paper + sanity-check runner | [→ README](eval_metrics/README.md) |
| [`Research-Pipeline/`](Research-Pipeline/README.md) | Phase A steps, Phase B spec, shared logic | [→ README](Research-Pipeline/README.md) |
| [`tuning/`](tuning/README.md) | Optuna hyper-parameter search: objective, per-feature search space, sbatch jobs, main bash runner | [→ README](tuning/README.md) |

---

## 🤖 Models & Datasets

4 models x 3 datasets = **12 experiment combinations**, each with its own
folder, config merge and results. Chosen as the most frequent across the
reviewed literature (see [`Documents/Comparison-Table/`](Documents/Comparison-Table/README.md)).

| Model | HF id | Layers | Pooling |
|-------|-------|--------|---------|
| BERT-base | `bert-base-uncased` | 12 | `cls` |
| RoBERTa-large | `roberta-large` | 24 | `cls` |
| DeBERTa-large | `microsoft/deberta-large` | 24 | `cls` |
| BART-large | `facebook/bart-large` | 12 dec. | `eos` |

| Dataset | HF id | Difficulty | Note |
|---------|-------|-----------|------|
| SNLI | `stanfordnlp/snli` | Easiest | image-caption premises |
| MNLI | `nyu-mll/multi_nli` | Medium | ten genres; mismatched-dev = test |
| ANLI | `facebook/anli` | Hardest | adversarial, rounds r1-r3 |

Datasets are kept **separate** — merging would blur where any improvement
came from.

Every combination is fine-tuned locally by Step-1, except where an
**official, single-dataset** published checkpoint exists (MNLI on the three
large models) — full provenance table with links in
[`configs/README.md`](configs/README.md#-published-checkpoint-provenance-official-sources).

---

## 📏 Metrics

| Metric | Level | Source | File |
|--------|-------|--------|------|
| Relaxed Fooling Rate | output | Anchor paper | `eval_metrics/relaxed_fooling_rate.py` |
| Strict Fooling Rate | output | Anchor paper | `eval_metrics/strict_fooling_rate.py` |
| Paraphrastic Consistency (PC) | output | Srikanth et al. 2024 | `eval_metrics/paraphrastic_consistency.py` |
| Pattern Accuracy (PA) | output | MERGE 2025 | `eval_metrics/pattern_accuracy.py` |
| Layer Distance | internal | this work | `eval_metrics/layer_distance.py` |
| DkNN Credibility | internal | Papernot & McDaniel 2018 | `eval_metrics/dknn_credibility.py` |

---

## 🔄 The Pipeline

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

## ⚠️ Paraphrase Data Status

**Verified (July 2026): no public, verified corpus provides ~5
relation-preserving paraphrases per hypothesis for SNLI / MNLI / ANLI.**

| Resource | Offers | Not a drop-in because |
|----------|--------|----------------------|
| [Anchor paper](https://arxiv.org/abs/2401.14440) (EACL 2024) | Semantics-preserving variations verified by symmetric equivalence entailment | Generated per evaluated model, on the fly — no static release |
| [ParaNlu](https://arxiv.org/abs/2404.11717) (TACL 2024) | 7,782 human-validated label-preserving paraphrases | Abductive / defeasible NLI, not 3-class; ~1,000 base problems |
| [MERGE / AllVar](https://arxiv.org/abs/2510.24295) (2025) | MLM-based minimal expression replacements | SNLI-derived only |

**Adopted protocol** (mirrors the anchor paper), implemented in
[`Datasets/generate_paraphrases.py`](Datasets/generate_paraphrases.py):
sample `eval_sample_size` hypotheses per combination → 10 candidates each
from the leading open paraphraser (`humarin/chatgpt_paraphraser_on_T5_base`)
→ keep only candidates passing **bidirectional entailment** with the
strongest public NLI verifier
(`MoritzLaurer/DeBERTa-v3-large-mnli-fever-anli-ling-wanli`) → first 5
verified per hypothesis. Full contract:
[`Datasets/*/paraphrases/README.md`](Datasets/SNLI-Stanford_NLI/paraphrases/README.md).
Step-2 skips gracefully until the files exist.

---

## 🚀 How to Run

### Step 0: Environment

```bash
pip install -r requirements.txt
python eval_metrics/sanity_check.py        # all lines must print OK (no GPU needed)
```

### Step 1: Data

```bash
python Datasets/download_datasets.py       # SNLI + MNLI + ANLI -> raw/
```

### Step 2: The Experiment - one command, end to end

```bash
python Research-Pipeline/run_pipeline.py
```

The main runner orchestrates everything: checks the raw data (downloads
what's missing), runs Step-1 (training resumes after crashes; MNLI
combinations use the official published checkpoints automatically), generates
+ verifies any missing paraphrase files, then runs inference, DkNN, distances
and every plot. **Fully resumable** - finished stages skip themselves.

Useful variants:

```bash
python Research-Pipeline/run_pipeline.py --models BERT-base --datasets SNLI
python Research-Pipeline/run_pipeline.py --steps 3,4            # analysis only (CPU)
python Research-Pipeline/run_pipeline.py --static-paraphrases   # never auto-generate
python Research-Pipeline/run_pipeline.py --force                # redo after tuning
```

### Step 3: Hyper-Parameter Tuning (optional - BGU SLURM)

```bash
cd tuning && bash run_all_sbatch.sh      # comment out unwanted jobs first
# ... studies write configs/tuned/<COMBO>.yaml automatically, then:
python Research-Pipeline/run_pipeline.py --force
```

See [`tuning/README.md`](tuning/README.md) for the objective, the per-feature
search space, resume behaviour and the sbatch files.

Or run any level directly — every folder has its own meta-runner:
`run_phase_a.py`, `run_step{1..4}.py`, or a single combination's
`Step-1_Train-Filter-Encode/BERT-base__SNLI/run.py` (recommended per SLURM
job on the BGU cluster).

Configuration lives in [`configs/`](configs/README.md) — sampling caps, K
candidates, training hyper-parameters. Nothing is hard-coded.

---

## 📊 Outputs

| Where | What |
|-------|------|
| `Step-1_*/<COMBO>/results/` | checkpoint, train metrics, baseline accuracy |
| `Datasets/<ds>/filtered/<MODEL>/` | per-model correct-only subset |
| `Datasets/Encoded_Datasets/<MODEL>/<DS>/` | hypothesis bank + paraphrase representations |
| `Step-2_*/results/<COMBO>/` | paraphrase predictions + consistent / strict-flip flags |
| `Step-3_*/results/<COMBO>/` | chosen K, credibility table + summary, layer profile + summary |
| `Step-4_*/results/` | aggregated `consistency_metrics.csv`; per-combo `diagnosis.json`, distance-profile plot + credibility-groups plot; unified `layer_distance_overview.png` + `credibility_overview.png` across all combinations |
| `tuning/results/<COMBO>/` | resumable Optuna study, `best_params.yaml`, `trials.csv` (best values auto-land in `configs/tuned/`) |

The headline number of Phase A lives in `credibility_summary.json`:
the correlation, on the **held-out test portion**, between the credibility
drop and actual inconsistency — can DkNN spot the vulnerable examples
*before* any paraphrase is shown?

---

## 📜 License

MIT — see [LICENSE](LICENSE) for details.
