# Datasets - READ-ONLY Sources

The three NLI corpora **plus the static paraphrase banks** - and nothing
else. **No run of any model ever writes here.** The only writers are the
one-time downloader (`setup-files/download_datasets.py` -> `raw/`) and the standalone
[`setup-files/Paraphrase-Generator/`](../setup-files/Paraphrase-Generator/README.md)
(-> `paraphrases/paraphrase_bank.csv`, also one-time).

## Structure

| Folder | Difficulty | Source | README |
|--------|-----------|--------|--------|
| [`SNLI-Stanford_NLI/`](SNLI-Stanford_NLI/README.md) | easiest | image-caption pairs (Bowman et al., 2015) | [-> README](SNLI-Stanford_NLI/README.md) |
| [`MNLI-MultiGenre_NLI/`](MNLI-MultiGenre_NLI/README.md) | medium | ten written & spoken genres (Williams et al., 2018) | [-> README](MNLI-MultiGenre_NLI/README.md) |
| [`ANLI-Adversarial_NLI/`](ANLI-Adversarial_NLI/README.md) | hardest | adversarially built, rounds r1-r3 (Nie et al., 2020) | [-> README](ANLI-Adversarial_NLI/README.md) |

Each dataset folder holds exactly two subfolders: `raw/` (the original
corpus, one parquet per split) and `paraphrases/` (the static bank + its
audit stats). The datasets are kept strictly **separate** - never merged.

## Meta-Runner

```bash
python setup-files/download_datasets.py                 # all three corpora -> raw/
python setup-files/download_datasets.py --datasets SNLI # subset
```

The main pipeline runs this automatically when raw data is missing.

## Where Everything Else Lives

Run-derived data (the reduced per-model copies, all encodings, the per-model
paraphrase copies) lives in [`Runtime-Data/`](../Runtime-Data/README.md) and
is **wiped at every run's start**. The paraphrase-bank generator and its
configuration live in
[`setup-files/Paraphrase-Generator/`](../setup-files/Paraphrase-Generator/README.md).
