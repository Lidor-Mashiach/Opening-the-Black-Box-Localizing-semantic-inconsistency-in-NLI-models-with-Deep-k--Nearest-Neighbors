# 📊 Datasets

All data used by the research: the three NLI corpora, the per-model
"correct only" filtered subsets, the hypothesis paraphrases, and the
layer-wise encoded representation banks.

The three corpora are kept **strictly separate** - each represents a
different sentence population, and merging them would blur where any
improvement came from (see the methodology document).

## 📁 Structure

| Directory | Difficulty | Contents | README |
|-----------|-----------|----------|--------|
| [`SNLI-Stanford_NLI/`](SNLI-Stanford_NLI/README.md) | Easiest | Image-caption premises (Bowman et al., 2015) | [→ README](SNLI-Stanford_NLI/README.md) |
| [`MNLI-MultiGenre_NLI/`](MNLI-MultiGenre_NLI/README.md) | Medium | Ten genres (Williams et al., 2018) | [→ README](MNLI-MultiGenre_NLI/README.md) |
| [`ANLI-Adversarial_NLI/`](ANLI-Adversarial_NLI/README.md) | Hardest | Built adversarially (Nie et al., 2020) | [→ README](ANLI-Adversarial_NLI/README.md) |
| [`Encoded_Datasets/`](Encoded_Datasets/README.md) | - | Layer-wise representation banks per model x dataset | [→ README](Encoded_Datasets/README.md) |

Every dataset directory has the same three subfolders:

| Subfolder | What lives there | Produced by |
|-----------|------------------|------------|
| `raw/` | Original splits as parquet (unified schema) | `download_datasets.py` |
| `filtered/<MODEL>/` | Train rows that model classified **correctly** | Step-1 `build_filtered_dataset.py` |
| `paraphrases/` | `<MODEL>__paraphrases.csv` - ~5 relation-preserving paraphrases per kept hypothesis | `generate_paraphrases.py` (or any LLM respecting the contract) |

## 🚀 Meta-Runner

```bash
python download_datasets.py                 # all three corpora -> raw/
python download_datasets.py --datasets SNLI # subset

# after Step-1 has produced the filtered subsets:
python generate_paraphrases.py              # generate + verify paraphrases
python generate_paraphrases.py --limit 50   # quick trial run
```

Uses the exact same loader as the pipeline (`common/data_loading.py`):
same schema everywhere (`premise`, `hypothesis`, `label`, `pair_id`,
`round` for ANLI), rows with label `-1` dropped.

## 📏 Label Convention

`0 = entailment · 1 = neutral · 2 = contradiction` (HuggingFace standard).
Neutral is a genuine logical relation, **not** model uncertainty; it is kept
in training and in consistency counting (Neutral -> Neutral is consistent).
