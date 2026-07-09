# 📊 MNLI - Multi-Genre Natural Language Inference

433k sentence pairs across ten genres of written and spoken English (fiction, telephone speech, government reports and more). No overlap with SNLI records.

| | |
|---|---|
| **HuggingFace id** | `nyu-mll/multi_nli` |
| **Splits used** | `train` / `validation_matched` (as validation) / `validation_mismatched` (as test) |
| **Difficulty in this study** | Medium |
| **Source paper** | Williams, Nangia & Bowman (2018), NAACL-HLT |

## 📁 Subfolders

| Subfolder | Contents | Produced by |
|-----------|----------|------------|
| `raw/` | `train / validation / test .parquet` - unified schema (`premise`, `hypothesis`, `label`, `pair_id`) | `../download_datasets.py` |
| `filtered/<MODEL>/` | `train_correct.csv` + `filter_stats.json` - only rows the model classified correctly. **Each model produces a different filtered set** | Step-1 `build_filtered_dataset.py` |
| `paraphrases/` | `<MODEL>__paraphrases.csv` - see [paraphrases/README.md](paraphrases/README.md) for the acquisition protocol and file contract | External protocol |

> The official MNLI test labels are private. Following common practice, the matched dev set serves as **validation** and the mismatched dev set as the held-out **test** set (see `configs/datasets/MNLI.yaml`).
