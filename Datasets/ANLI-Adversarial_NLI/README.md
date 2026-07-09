# 📊 ANLI - Adversarial Natural Language Inference

Built adversarially with a human-and-model-in-the-loop procedure over three rounds (r1-r3) - deliberately hard. The three rounds are concatenated; a `round` column is kept for per-round breakdowns.

| | |
|---|---|
| **HuggingFace id** | `facebook/anli` |
| **Splits used** | `train_r{1,2,3}` / `dev_r{1,2,3}` / `test_r{1,2,3}` (concatenated) |
| **Difficulty in this study** | Hardest |
| **Source paper** | Nie, Williams, Dinan, Bansal, Weston & Kiela (2020), ACL |

## 📁 Subfolders

| Subfolder | Contents | Produced by |
|-----------|----------|------------|
| `raw/` | `train / validation / test .parquet` - unified schema (`premise`, `hypothesis`, `label`, `pair_id`, `round`) | `../download_datasets.py` |
| `filtered/<MODEL>/` | `train_correct.csv` + `filter_stats.json` - only rows the model classified correctly. **Each model produces a different filtered set** | Step-1 `build_filtered_dataset.py` |
| `paraphrases/` | `<MODEL>__paraphrases.csv` - see [paraphrases/README.md](paraphrases/README.md) for the acquisition protocol and file contract | External protocol |

ANLI is the smallest of the three - the sampling caps in `configs/base.yaml` will usually not trigger here.
