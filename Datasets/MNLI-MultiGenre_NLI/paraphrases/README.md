# Paraphrase Bank - A Static, Shareable Research Asset

This folder holds **`paraphrase_bank.csv`** - the dataset-level paraphrase
bank: one static file, model-independent, with **up to 5 verified
paraphrases per hypothesis** (target 5, kept partial - a hypothesis with
1-4 verified paraphrases is kept as-is - only a hypothesis with zero verified
paraphrases is dropped).

**Base condition (non-negotiable):** every paraphrase holds the *exact same
logical relation* to the premise as the original hypothesis - i.e. in the
ground truth, `Premise : Hypothesis == Premise : Paraphrase`. The
**bidirectional-entailment verification** below enforces this - nothing
unverified ever enters.

## How the Bank Is Built ([`setup-files/Paraphrase-Generator/`](../../../setup-files/Paraphrase-Generator/README.md))

| Stage | What happens |
|-------|--------------|
| 1. Pool | `paraphrases.pool_size` (`null` = **no sampling**, every hypothesis in the raw train split) - seeded, deterministic, same `pair_id`s the whole pipeline uses |
| 2. Generate | 12 candidates per hypothesis - `humarin/chatgpt_paraphraser_on_T5_base` (the leading open paraphraser on the HF Hub), nucleus (top-p) sampling |
| 3. Length gate | Word-count ratio candidate/hypothesis must stay in `paraphrases.length_ratio` (0.6-1.5) - **length & complexity remain at the hypothesis's own level**, so each dataset's difficulty is preserved and measurable |
| 4. Double verify | `MoritzLaurer/DeBERTa-v3-large-mnli-fever-anli-ling-wanli` (the strongest public NLI model) applies TWO gates: **(a) equivalence** - hypothesis <-> candidate entail *each other* - **(b) relation** - the (premise, candidate) label **equals the gold label**. Same meaning AND same logical relation to the premise |
| 5. Retry rounds | Hypotheses short of the target get MORE rounds - fresh nucleus sampling each time (new seed + mild temperature ramp, dedup vs everything tried) - up to `max_generation_rounds` (default 6), with an early stop after `early_stop_patience` (2) consecutive rounds that add nothing |
| 6. Quota | **Up to 5 kept per hypothesis** (target 5), kept partial: a hypothesis that yields 1-4 verified paraphrases is kept as-is. Only a hypothesis with **zero** verified paraphrases is dropped (loudly reported in the stats) |

`paraphrase_bank_stats.json` makes it all **measurable**: pool size, a
fill-rounds histogram, last-resort drops (expected ~0), per-gate rejection
counts (length / equivalence /
relation), and `length_stats` - mean+-std word counts of hypotheses vs
paraphrases and the mean length ratio, per dataset.
Full schema + the generator's configuration (`paraphrase_config.yaml`) live
in [`setup-files/Paraphrase-Generator/README.md`](../../../setup-files/Paraphrase-Generator/README.md).

## Protection - Treated Like the Raw Data

Running the generator or the pipeline **never overwrites an existing bank**:

| Layer | Mechanism |
|-------|-----------|
| Skip by default | An existing `paraphrase_bank.csv` -> the generator SKIPs - regeneration requires an explicit `--rebuild` |
| One-generation backup | On `--rebuild`, the previous bank is preserved as `paraphrase_bank.backup.csv` |
| Atomic write | The new bank is written to a temp file and promoted only when complete - a crash mid-generation can never destroy the existing bank |
| Pipeline runs never touch it | The pipeline only READS the bank - retraining or rerunning never modifies it - the bank is model-independent |

## How Each Model Uses the Bank (Step-2)

Every model consumes **only the rows whose hypothesis it classified
correctly** (intersection with `Runtime-Data/<MODEL>/<DATASET>/train_correct.csv`), so:
- `label == the model's hypothesis prediction` holds by construction
- **all models are tested on the exact same paraphrases** for shared
  hypotheses - apples-to-apples comparison
- the bank itself never changes when models are retrained or the pipeline
  reruns - regenerate only with the generator's explicit `--rebuild`.

The encoded DkNN bank of every model **always includes the pool hypotheses**
(`common/sampling.py`), so Step-3 alignment never drops a row.

## File Contract - `paraphrase_bank.csv`

| Column | Type | Meaning |
|--------|------|---------|
| `pair_id` | str | Id of the source hypothesis (matches the pipeline's ids) |
| `premise` | str | The ORIGINAL premise, unchanged |
| `hypothesis` | str | The original hypothesis (for reference / reuse) |
| `paraphrase` | str | Reworded hypothesis, same logical relation |
| `label` | int | Gold label (0/1/2) |
| `para_idx` | int | 0-based index within the hypothesis (0..2 - fewer when partial) |

## Why We Built It Ourselves (verified July 2026)

No public corpus offers relation-preserving paraphrases per hypothesis for
SNLI / MNLI / ANLI: the anchor paper generates variations per evaluated model
without a static release - ParaNlu targets abductive/defeasible NLI - MERGE is
SNLI-only minimal replacements. This bank fills that gap - and is built to be
shared with the community.