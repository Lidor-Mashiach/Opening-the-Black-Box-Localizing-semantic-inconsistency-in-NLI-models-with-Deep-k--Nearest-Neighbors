# common - Shared Pipeline Logic

Every piece of real logic exists exactly once here; the per-combination
scripts under `PhaseA/Step-1_*` are 10-line wrappers around these modules.

## Modules

| Module | Responsibility | Used by |
|--------|----------------|---------|
| `config_loader.py` | YAML merge (base -> model -> dataset) + every canonical path (all derived data under `Runtime-Data/<MODEL>/<DATASET>/`) + `resolve_checkpoint` (local fine-tuned -> published `nli_checkpoints` fallback) | everything |
| `sampling.py` | Seeded, deterministic sampling: the dataset paraphrase pool + the bank cap with must-include (pool hypotheses are ALWAYS bank members) | encoding, `generate_paraphrases.py` |
| `logging_utils.py` | Colored, structured logs: `[TAG] MODEL x DATASET \| message` - one color per subject (Step-1 cyan, Step-2 magenta, Step-3 yellow, Step-4 green, CLEAN/ERROR red) | everything |
| `workspace.py` | `clean_combo_workspace()` - wipes a combination's Runtime-Data + Step-2/3/4 results at run start (no leftovers). Checkpoints follow training's per-run seed guard; Optuna studies + tuned configs are preserved | combo `run.py` |
| `gpu.py` | Central GPU detection: loud `[DEVICE]` banner (name + VRAM + CUDA version), actionable WHY when missing (CPU-only wheel vs driver/SLURM allocation), optional `NLI_REQUIRE_GPU=1` fail-fast (off by default) | every GPU entry point |
| `data_loading.py` | Unified SNLI/MNLI/ANLI loading: one schema, `-1` labels dropped, ANLI rounds concatenated, deterministic `pair_id`s | Step-1, `download_datasets.py` |
| `model_utils.py` | Model/tokenizer loading, batched `predict()`, `extract_layer_representations()` (cls/eos pooling, embedding layer dropped, BART decoder handled) - backbone downloaded into Models/raw/<MODEL> via ensure_raw_backbone (never HF home cache) | Steps 1-2 |
| `training.py` | `fine_tune()` - HF Trainer, version-proof, saves final checkpoint + post-training validation accuracy | Step-1a |
| `filtering.py` | `build_filtered_dataset()` - the reduced correct-only COPY per model, written into Runtime-Data | Step-1b |
| `encoding.py` | `encode_hypothesis_bank()` - the labeled layer-wise DkNN bank (seeded cap; the paraphrase-pool hypotheses are ALWAYS included) | Step-1c |
| `evaluation.py` | `evaluate_baseline()` - plain accuracy on validation + test | Step-1d |
| `paraphrase_inference.py` | `run_paraphrase_inference()` - intersects the dataset paraphrase bank with the model's correct-only set, writes the per-model copy (`paraphrases_used.csv`), then predictions + representations; graceful skip while the bank is missing | Step-2 |
| `alignment.py` | `load_aligned()` - joins bank, paraphrase encodings and Step-2 predictions; deterministic validation/test split by `pair_id` hash | Step-3 |

## Conventions

- Logical splits are always `train` / `validation` / `test`; the mapping to
  HuggingFace split names lives in `configs/datasets/*.yaml`.
- `pair_id = "<DATASET>-<split>-<index>"` is deterministic and stable - it is
  the join key across the entire pipeline.
- Representations: `[n, n_layers, dim]`, float16, embedding layer dropped
  (index `i` = transformer layer `i+1`).
