# 🧰 common - Shared Pipeline Logic

Every piece of real logic exists exactly once here; the per-combination
scripts under `PhaseA/Step-1_*` are 10-line wrappers around these modules.

## 📄 Modules

| Module | Responsibility | Used by |
|--------|----------------|---------|
| `config_loader.py` | YAML merge (base -> model -> dataset) + every canonical path + `resolve_checkpoint` (local fine-tuned -> published `nli_checkpoints` fallback) | everything |
| `sampling.py` | Seeded, deterministic subset chain: bank cap -> eval sample. Guarantees paraphrased hypotheses ⊆ encoded bank | encoding, `generate_paraphrases.py` |
| `data_loading.py` | Unified SNLI/MNLI/ANLI loading: one schema, `-1` labels dropped, ANLI rounds concatenated, deterministic `pair_id`s | Step-1, `download_datasets.py` |
| `model_utils.py` | Model/tokenizer loading, batched `predict()`, `extract_layer_representations()` (cls/eos pooling, embedding layer dropped, BART decoder handled) | Steps 1-2 |
| `training.py` | `fine_tune()` - HF Trainer, version-proof, saves final checkpoint + post-training validation accuracy | Step-1a |
| `filtering.py` | `build_filtered_dataset()` - keep only correctly-classified train rows, per model | Step-1b |
| `encoding.py` | `encode_hypothesis_bank()` - the labeled layer-wise DkNN bank (seeded cap via `sampling.max_bank_examples`) | Step-1c |
| `evaluation.py` | `evaluate_baseline()` - plain accuracy on validation + test | Step-1d |
| `paraphrase_inference.py` | `run_paraphrase_inference()` - predictions + representations for (premise, paraphrase); graceful skip while the CSV is missing | Step-2 |
| `alignment.py` | `load_aligned()` - joins bank, paraphrase encodings and Step-2 predictions; deterministic validation/test split by `pair_id` hash | Step-3 |

## 🔑 Conventions

- Logical splits are always `train` / `validation` / `test`; the mapping to
  HuggingFace split names lives in `configs/datasets/*.yaml`.
- `pair_id = "<DATASET>-<split>-<index>"` is deterministic and stable - it is
  the join key across the entire pipeline.
- Representations: `[n, n_layers, dim]`, float16, embedding layer dropped
  (index `i` = transformer layer `i+1`).
