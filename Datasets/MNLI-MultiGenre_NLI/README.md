# MNLI - Multi-Genre Natural Language Inference

433k sentence pairs across ten genres of written and spoken English (fiction, telephone speech, government reports and more). No overlap with SNLI records.

| | |
|---|---|
| **HuggingFace id** | `nyu-mll/multi_nli` |
| **Splits used** | `train` / `validation_matched` (as validation) / `validation_mismatched` (as test) |
| **Difficulty in this study** | Medium |
| **Source paper** | Williams, Nangia & Bowman (2018), NAACL-HLT |

## Subfolders

| Subfolder | Contents | Produced by |
|-----------|----------|------------|
| `raw/` | `train / validation / test .parquet` - unified schema (`premise`, `hypothesis`, `label`, `pair_id`) | `../download_datasets.py` |
| `paraphrases/` | `paraphrase_bank.csv` - the static, model-independent bank (up to 5 verified paraphrases per hypothesis - kept partial, only zero-verified dropped) - see [paraphrases/README.md](paraphrases/README.md) | [`setup-files/Paraphrase-Generator/`](../../setup-files/Paraphrase-Generator/README.md) |

> The official MNLI test labels are private. Following common practice, the matched dev set serves as **validation** and the mismatched dev set as the held-out **test** set (see `configs/datasets/MNLI.yaml`).

> **Read-only folder.** Run-derived artifacts for this dataset (the reduced
> per-model copies and all encodings) live in
> [`Runtime-Data/`](../../Runtime-Data/README.md) and are wiped at each run's
> start - nothing here is ever modified by the pipeline.