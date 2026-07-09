# 3️⃣ Step-3 - DkNN Credibility & Layer Distance

Opening the black box: the two internal-representation lenses, computed per
(model, dataset) pair. Runs on CPU - pure NumPy over the encoded banks.

## 🔄 Script Order

| # | Script | What it does | Output |
|---|--------|--------------|--------|
| 1 | `select_k.py` | Tries every K in `dknn.k_values`; picks the K whose credibility-drop correlates most with actual inconsistency - **on the validation portion only**. One neighbour fetch at max(K), sliced per K | `results/<COMBO>/k_selection.json` |
| 2 | `compute_credibility.py` | Credibility for every hypothesis (self-excluded) and every paraphrase, the drop, the data-derived anomalous-drop threshold, the scenario diagnosis, and the Phase-A key correlation on the held-out test portion | `credibility.csv`, `credibility_summary.json` |
| 3 | `compute_layer_distances.py` | Direct cosine distance hypothesis vs paraphrase at every layer; consistent vs inconsistent profiles; threshold = consistent mean + std; first crossing layer -> region | `layer_distances.npz`, `layer_profile.csv`, `distance_summary.json` |

## 🔑 Methodology Rules Enforced Here

- **K per (model, dataset) pair**, chosen on validation - never global.
- **Validation/test split is deterministic** (CRC32 of `pair_id`), so all
  paraphrases of one hypothesis land on the same side, and the split is
  identical across runs and machines.
- **Self-exclusion**: a hypothesis never votes for itself in the bank.
- **Scenario mapping** (on group means): paraphrase credibility below
  hypothesis mean - std -> `generalization`; above mean + std ->
  `representation`; in between -> `within_normal_range` (classifier check
  happens in Step-4).

## 🚀 Meta-Runner

```bash
python run_step3.py
python run_step3.py --models BERT-base --datasets SNLI,MNLI
```

Combinations without Step-2 output are skipped with a clear message.
