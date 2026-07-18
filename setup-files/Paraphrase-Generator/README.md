# Paraphrase-Generator - Builds the Static Paraphrase Banks (One-Time)

A **standalone module, deliberately outside the pipeline**. It is run ONCE
per dataset, on a GPU machine, to create the static paraphrase banks at
`Datasets/<dataset>/paraphrases/paraphrase_bank.csv`. The main runner only
**checks** that the banks exist - if one is missing it prints the command
below and **exits**; it never generates paraphrases itself. Every run of the
research therefore uses the exact same static bank, which is what makes the
results measurable - and the banks themselves a shareable contribution to
the community.

## The One-Time Act

```bash
python setup-files/Paraphrase-Generator/generate_paraphrases.py                 # all three
python setup-files/Paraphrase-Generator/generate_paraphrases.py --datasets SNLI
python setup-files/Paraphrase-Generator/generate_paraphrases.py --limit 50      # trial run
```

GPU strongly recommended - the run is hundreds of thousands of transformer
forward passes (generation + three verifier passes per candidate): hours on a
GPU, days on a CPU. The script prints a loud `[DEVICE]` line with exactly
what it found; a `no CUDA GPU` message on a Windows machine with an NVIDIA
card almost always means the CPU-only torch wheel - run `python setup_env.py`
once and it repairs the build automatically. Tip: set `HF_TOKEN` to avoid HuggingFace download
throttling on the first run. Afterwards, commit the banks to git and never run
this again (an existing bank is SKIPPED; regeneration requires an explicit
`--rebuild`, which still preserves the previous bank as
`paraphrase_bank.backup.csv` and writes atomically).

## What Guarantees the Paraphrases Are Valid?

| Gate | Enforces | How |
|------|----------|-----|
| Fluency / syntax | Well-formed English | `humarin/chatgpt_paraphraser_on_T5_base` - the leading open paraphraser (T5 trained on 6.3M ChatGPT paraphrase pairs), diverse beam search + repetition penalties per its model card |
| Length / complexity | Same level as the hypothesis, per dataset | Word-count ratio inside `length_ratio` (0.6-1.5) |
| Same meaning | Paraphrase == hypothesis semantically | hypothesis <-> candidate must entail **each other** (`MoritzLaurer/DeBERTa-v3-large-mnli-fever-anli-ling-wanli` - the strongest public NLI model) |
| Same logical relation | `Premise:Paraphrase == Premise:Hypothesis` | The verifier's label for (premise, candidate) must **equal the gold label** |
| Uniform quota | Exactly 5 per hypothesis, no exceptions | FRESH-sampling rounds continue (new seed + mild temperature ramp each round, up to `max_generation_rounds`) until the quota is reached; a drop is only a last-resort safety valve for degenerate cases - expected ~0 and loudly reported |

This is the anchor paper's protocol (symmetric equivalence entailment)
**plus** two stricter gates. It is the strongest *automatic* guarantee
available; for the report we recommend also citing the per-dataset stats
below and a small human spot-check (~100 rows).

## The Two GLOBALS - the Only Scaling Switches

At the top of `generate_paraphrases.py`:

| Global | Now | What it controls |
|--------|-----|------------------|
| `PARAPHRASES_PER_HYPOTHESIS` | 5 | The exact, uniform quota per hypothesis - change the number, rerun, done |
| `DATASETS` | SNLI / MNLI / ANLI | KEY = dataset name, VALUE = its folder (repo-relative); the bank lands at `<VALUE>/paraphrases/paraphrase_bank.csv` |

**Adding a dataset** = one line in `DATASETS` **+** a small
`configs/datasets/<KEY>.yaml` (`hf_id`, `dir`, splits - the pipeline needs it
too). Two guards keep everything coherent: an unknown key aborts with the
exact instruction, and a registry-vs-configs path mismatch aborts loudly
(the pipeline reads the configs path) and tells you which `dir` to fix.

## Configuration - `paraphrase_config.yaml` (generation knobs only)

| Knob | Default | Meaning |
|------|---------|---------|
| `pool_size` | 4000 | Hypotheses sampled per dataset (seeded, from the raw train split) |
| `candidates_per_hypothesis` | 10 | Raw candidates per round, before the gates |
| `max_generation_rounds` | 8 | Generation rounds per hypothesis until the quota fills (~80 candidates); raise for zero drops at extra GPU cost |
| `length_ratio.min/max` | 0.6 / 1.5 | Allowed paraphrase/hypothesis word-count ratio |
| `generator_model`, `generator_prompt_prefix`, `generation_batch_size` | see file | The paraphraser + its prompt format |
| `verifier_model` | MoritzLaurer/DeBERTa-v3-large-... | The double-gate NLI verifier |

The pipeline's `configs/` intentionally knows nothing about these knobs -
the bank is an input to the pipeline, not a product of it.

## The Audit File - `paraphrase_bank_stats.json`

Written **next to each bank** (`Datasets/<dataset>/paraphrases/`), so the
audit stays associated with its dataset. Fields:

| Field | Meaning |
|-------|---------|
| `pool_size`, `hypotheses_kept_exact_quota`, `hypotheses_dropped_after_all_rounds` | Pool -> kept -> last-resort-drop accounting (drops expected ~0) |
| `fill_rounds_histogram` | How many hypotheses reached the quota at each round (round 1 = beam, 2+ = fresh sampling) |
| `gate_rejections` | Rejected candidates per gate: `rejected_length` / `rejected_equivalence` / `rejected_relation` |
| `length_stats` | Mean+-std word counts, hypotheses vs paraphrases, + mean ratio - proof that difficulty tracks the dataset |
| `generator_model`, `verifier_model`, `length_ratio_bounds`, `per_hypothesis` | Full provenance of how the bank was made |

It is an **output/audit artifact - do not edit it**; the knobs live in
`paraphrase_config.yaml`.
