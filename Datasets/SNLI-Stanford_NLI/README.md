# 📊 SNLI - Stanford Natural Language Inference

570k human-written sentence pairs; premises are Flickr30k image captions, hypotheses were written by crowdworkers. The first large-scale NLI corpus.

| | |
|---|---|
| **HuggingFace id** | `stanfordnlp/snli` |
| **Splits used** | `train` / `validation` / `test` |
| **Difficulty in this study** | Easiest |
| **Source paper** | Bowman, Angeli, Potts & Manning (2015), EMNLP |

## 📁 Subfolders

| Subfolder | Contents | Produced by |
|-----------|----------|------------|
| `raw/` | `train / validation / test .parquet` - unified schema (`premise`, `hypothesis`, `label`, `pair_id`) | `../download_datasets.py` |
| `filtered/<MODEL>/` | `train_correct.csv` + `filter_stats.json` - only rows the model classified correctly. **Each model produces a different filtered set** | Step-1 `build_filtered_dataset.py` |
| `paraphrases/` | `<MODEL>__paraphrases.csv` - see [paraphrases/README.md](paraphrases/README.md) for the acquisition protocol and file contract | External protocol |

Rows with label `-1` (no gold consensus) are dropped at load time.
