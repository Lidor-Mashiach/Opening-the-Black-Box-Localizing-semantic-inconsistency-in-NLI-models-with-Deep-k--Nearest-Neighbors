# Paraphrase-Generator - Builds the Static Paraphrase Banks (One-Time)

A **standalone module, deliberately outside the pipeline**. It is run ONCE
per dataset, on a GPU machine, to create the static paraphrase banks at
`Datasets/<dataset>/paraphrases/paraphrase_bank.csv`. The main runner only
**checks** that the banks exist - if one is missing it prints the command
below and **exits**; it never generates paraphrases itself. Every run of the
research therefore uses the exact same static bank, which is what makes the
results measurable - and the banks themselves a shareable contribution to
the community.

## TRAIN Split Only - and Why

Paraphrases are generated **exclusively from each dataset's `train` split**
(`Datasets/<dataset>/raw/train.parquet`). The `validation` and `test` splits
are **never paraphrased**. This is not an optimization - it is the correct
methodology:

* The entire research signal - the DkNN credibility of a hypothesis, the
  layer-distance between a hypothesis and its paraphrase, the paraphrase
  consistency - is computed on **train-derived hypotheses and their
  paraphrases**. Step-1 trains on `train`, filters to the hypotheses the
  model got right (`train_correct.csv`), and encodes those; Step-2 encodes
  their paraphrases. All of it lives in `train`.
* `validation` and `test` are used **only** for a separate baseline-accuracy
  number (`evaluate.py`) - they never enter the paraphrase pool, the reduced
  dataset, or the embedding bank.
* Generating paraphrases for `validation`/`test` would therefore produce
  artifacts nothing consumes, and mixing them into the train-derived chain
  would break the separation the research depends on.

So: **one bank per dataset, built from `train`, period.**

## The One-Time Act

```bash
python setup-files/Paraphrase-Generator/generate_paraphrases.py                 # all three
python setup-files/Paraphrase-Generator/generate_paraphrases.py --datasets SNLI
python setup-files/Paraphrase-Generator/generate_paraphrases.py --limit 50      # trial run
```

### Sequential vs Parallel

* **No `--datasets`** - builds all three datasets in sequence. This is the
  mode `main-setup.py` uses (and the `run_main_setup.sbatch` job).
* **`--datasets <ONE>`** - builds only that dataset. Because the pre-generation
  reset is then **scoped to that dataset alone** (it wipes only that dataset's
  Runtime-Data and step results, never another's), three single-dataset jobs
  can run **in parallel** without colliding. Three ready-made sbatch files do
  exactly this - `paraphrases_SNLI.sbatch`, `paraphrases_MNLI.sbatch`,
  `paraphrases_ANLI.sbatch` - so all three banks build at once on three nodes,
  cutting wall-clock time roughly threefold.

GPU strongly recommended - the run is hundreds of thousands of transformer
forward passes (generation + a single fused verifier pass per candidate
batch): a few hours on a modern GPU for the full train splits, days on a CPU.
The script prints a loud `[DEVICE]` line with exactly what it found; a
`no CUDA GPU` message on a Windows machine with an NVIDIA card almost always
means the CPU-only torch wheel - run `python setup-files/setup_env.py` once
and it repairs the build automatically. Tip: set `HF_TOKEN` to avoid
HuggingFace download throttling on the first run. Afterwards, commit the banks
to git and never run this again (an existing bank is SKIPPED; regeneration
requires an explicit `--rebuild`, which still preserves the previous bank as
`paraphrase_bank.backup.csv` and writes atomically).

## What Guarantees the Paraphrases Are Valid?

| Gate | Enforces | How |
|------|----------|-----|
| Fluency / syntax | Well-formed English | `humarin/chatgpt_paraphraser_on_T5_base` - the leading open paraphraser (T5 trained on 6.3M ChatGPT paraphrase pairs), diverse beam search + repetition penalties per its model card |
| Length / complexity | Same level as the hypothesis, per dataset | Word-count ratio inside `length_ratio` (0.6-1.5) |
| Same meaning | Paraphrase == hypothesis semantically | hypothesis <-> candidate must entail **each other** (`MoritzLaurer/DeBERTa-v3-large-mnli-fever-anli-ling-wanli` - the strongest public NLI model) |
| Same logical relation | `Premise:Paraphrase == Premise:Hypothesis` | The verifier's label for (premise, candidate) must **equal the gold label** |
| Target quota (partial kept) | Up to 3 verified paraphrases per hypothesis | FRESH-sampling rounds (new seed + mild temperature ramp) add candidates until a hypothesis reaches 3 or the rounds stop. A hypothesis that reaches fewer is **kept with what it has** (1 or 2); only a hypothesis with **zero** verified paraphrases is dropped. An early-stop halts the rounds once they stop producing anything, so the run never grinds for hours on an unfillable remainder. |

This is the anchor paper's protocol (symmetric equivalence entailment)
**plus** two stricter gates. It is the strongest *automatic* guarantee
available; for the report we recommend also citing the per-dataset stats
below and a small human spot-check (~100 rows).

### Is Using an NLI Model to Verify NLI Paraphrases Circular?

No, and it is worth stating explicitly why:

* **The verifier is not one of the 4 models under study.** The research
  studies `bert-base-uncased`, `roberta-large`, `microsoft/deberta-large`
  (v1), and `facebook/bart-large`. The verifier,
  `MoritzLaurer/DeBERTa-v3-large-...`, is a different checkpoint AND a
  different architecture generation (DeBERTa-**v3**, not v1) - it never
  judges itself.
* **The verifier only builds the STATIC bank, once.** It plays no role in
  training or evaluating the 4 study models afterward - there is no data
  leakage in the classical sense (it never sees a study model's predictions
  or influences its weights).
* **The same verifier is applied uniformly to SNLI, MNLI, and ANLI** - one
  line in `paraphrase_config.yaml`, no per-dataset special-casing. Its name
  lists the corpora IT was fine-tuned on (MNLI+FEVER+ANLI+LingNLI+WANLI, to
  make it a strong general-purpose NLI classifier) - that is unrelated to
  which of the three research datasets it verifies; it verifies all three
  equally.

**The real, known limitation** (worth stating in the write-up, not hiding):
no perfect oracle for "meaning-preserving" exists, so using any strong NLI
model as ground truth inevitably imports some of its own biases into what
counts as a valid paraphrase. Using a verifier from a different, stronger
generation than any of the 4 study models limits (but cannot fully
eliminate) the risk of systematically favoring one architecture family. This
is exactly why we recommend the human spot-check above, and why the anchor
paper itself uses this same class of automatic verification rather than
claiming a perfect ground truth.

## The Two GLOBALS - the Only Scaling Switches

At the top of `generate_paraphrases.py`:

| Global | Now | What it controls |
|--------|-----|------------------|
| `PARAPHRASES_PER_HYPOTHESIS` | 3 | Target verified paraphrases per hypothesis; hypotheses that reach fewer are kept partial, only zero-verified are dropped - change the number, rerun, done |
| `DATASETS` | SNLI / MNLI / ANLI | KEY = dataset name, VALUE = its folder (repo-relative); the bank lands at `<VALUE>/paraphrases/paraphrase_bank.csv` |

**Adding a dataset** = one line in `DATASETS` **+** a small
`configs/datasets/<KEY>.yaml` (`hf_id`, `dir`, splits - the pipeline needs it
too). Two guards keep everything coherent: an unknown key aborts with the
exact instruction, and a registry-vs-configs path mismatch aborts loudly
(the pipeline reads the configs path) and tells you which `dir` to fix.

## Configuration - `paraphrase_config.yaml` (generation knobs only)

| Knob | Default | Meaning |
|------|---------|---------|
| `pool_size` | `null` | `null` = NO sampling: paraphrase EVERY hypothesis in the whole train split. Set an integer only for a quick smoke test. |
| `candidates_per_hypothesis` | 12 | Raw candidates per round, before the gates (one beam call yields all 12; more => fewer rounds needed) |
| `max_generation_rounds` | 6 | Hard cap on rounds; the early-stop below usually ends the loop sooner |
| `early_stop_patience` | 2 | Stop retrying once this many consecutive rounds add zero new paraphrases (the generator is stuck on the hard remainder) - prevents hours that end in failure |
| `length_ratio.min/max` | 0.6 / 1.5 | Allowed paraphrase/hypothesis word-count ratio |
| `generation_batch_size` | 64 | Hypotheses per generation batch - large batch keeps the GPU busy (8 left it mostly idle) |
| `verifier_batch_size` | 256 | (sentence, candidate) pairs the verifier scores per forward pass - the classifier packs a much bigger batch than generation |
| `generator_model`, `generator_prompt_prefix` | see file | The paraphraser + its prompt format |
| `verifier_model` | MoritzLaurer/DeBERTa-v3-large-... | The double-gate NLI verifier |

The per-hypothesis target quota (3) and the dataset registry are GLOBALS at
the top of `generate_paraphrases.py` (`PARAPHRASES_PER_HYPOTHESIS`,
`DATASETS`). The pipeline's `configs/` intentionally knows nothing about
these knobs - the bank is an input to the pipeline, not a product of it.

## The Audit File - `paraphrase_bank_stats.json`

Written **next to each bank** (`Datasets/<dataset>/paraphrases/`), so the
audit stays associated with its dataset. Fields:

| Field | Meaning |
|-------|---------|
| `split` | Always `train` - the only split paraphrased |
| `pool_size`, `hypotheses_kept`, `hypotheses_at_full_quota`, `hypotheses_partial`, `hypotheses_dropped_zero_verified` | Pool -> kept accounting: how many reached the full target, how many were kept partial (1-2), how many had zero and were dropped |
| `paraphrase_count_histogram` | How many hypotheses ended with 0/1/2/3 verified paraphrases |
| `fill_rounds_histogram` | How many hypotheses reached the full quota at each round (round 1 = beam, 2+ = fresh sampling) |
| `gate_rejections` | Rejected candidates per gate: `rejected_length` / `rejected_equivalence` / `rejected_relation` |
| `length_stats` | Mean+-std word counts, hypotheses vs paraphrases, + mean ratio - proof that difficulty tracks the dataset |
| `generator_model`, `verifier_model`, `length_ratio_bounds`, `target_paraphrases_per_hypothesis`, `policy` | Full provenance of how the bank was made |

It is an **output/audit artifact - do not edit it**; the knobs live in
`paraphrase_config.yaml`.