# ANLI - Adversarial Natural Language Inference

Built adversarially with a human-and-model-in-the-loop procedure over three rounds (r1-r3) - deliberately hard. The three rounds are concatenated - a `round` column is kept for per-round breakdowns.

| | |
|---|---|
| **HuggingFace id** | `facebook/anli` |
| **Splits used** | `train_r{1,2,3}` / `dev_r{1,2,3}` / `test_r{1,2,3}` (concatenated) |
| **Difficulty in this study** | Hardest |
| **Source paper** | Nie, Williams, Dinan, Bansal, Weston & Kiela (2020), ACL |

## Subfolders

| Subfolder | Contents | Produced by |
|-----------|----------|------------|
| `raw/` | `train / validation / test .parquet` - unified schema (`premise`, `hypothesis`, `label`, `pair_id`, `round`) | `../download_datasets.py` |
| `paraphrases/` | `paraphrase_bank.csv` - the static, model-independent bank (up to 5 verified paraphrases per hypothesis - kept partial, only zero-verified dropped) - see [paraphrases/README.md](paraphrases/README.md) | [`setup-files/Paraphrase-Generator/`](../../setup-files/Paraphrase-Generator/README.md) |

ANLI is the smallest of the three - the sampling caps in `configs/base.yaml` will usually not trigger here.

> **Read-only folder.** Run-derived artifacts for this dataset (the reduced
> per-model copies and all encodings) live in
> [`Runtime-Data/`](../../Runtime-Data/README.md) and are wiped at each run's
> start - nothing here is ever modified by the pipeline.