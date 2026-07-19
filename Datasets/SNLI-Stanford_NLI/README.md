# SNLI - Stanford Natural Language Inference

570k human-written sentence pairs; premises are Flickr30k image captions, hypotheses were written by crowdworkers. The first large-scale NLI corpus.

| | |
|---|---|
| **HuggingFace id** | `stanfordnlp/snli` |
| **Splits used** | `train` / `validation` / `test` |
| **Difficulty in this study** | Easiest |
| **Source paper** | Bowman, Angeli, Potts & Manning (2015), EMNLP |

## Subfolders

| Subfolder | Contents | Produced by |
|-----------|----------|------------|
| `raw/` | `train / validation / test .parquet` - unified schema (`premise`, `hypothesis`, `label`, `pair_id`) | `../download_datasets.py` |
| `paraphrases/` | `paraphrase_bank.csv` - the static, model-independent bank (up to 3 verified paraphrases per hypothesis; kept partial, only zero-verified dropped) - see [paraphrases/README.md](paraphrases/README.md) | [`setup-files/Paraphrase-Generator/`](../../setup-files/Paraphrase-Generator/README.md) |

Rows with label `-1` (no gold consensus) are dropped at load time.

> **Read-only folder.** Run-derived artifacts for this dataset (the reduced
> per-model copies and all encodings) live in
> [`Runtime-Data/`](../../Runtime-Data/README.md) and are wiped at each run's
> start - nothing here is ever modified by the pipeline.
