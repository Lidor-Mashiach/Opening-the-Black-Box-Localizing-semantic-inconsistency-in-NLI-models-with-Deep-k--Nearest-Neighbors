# 4️⃣ Step-4 - Consistency Metrics & Joint Diagnosis

The final read-out of Phase A: the four literature metrics on the outputs,
and the combined verdict from the two internal lenses.

## 🔄 Scripts

| Script | What it does | Output |
|--------|--------------|--------|
| `compute_consistency_metrics.py` | RFR + SFR (anchor paper), PC (Srikanth 2024), PA (MERGE) for every combination, via `eval_metrics/` | `results/consistency_metrics.csv` (one aggregated table) |
| `diagnose.py` | Joins Step-3's credibility scenario with the layer-distance region into one verdict + a Phase-B recommendation, and draws the two presentation plots | `results/<COMBO>/diagnosis.json`, `layer_distance_profile.png`, `credibility_groups.png` |
| `plot_overview.py` | Unified grid figures across ALL combinations: every distance profile side by side, and every low-vs-high credibility bar pair side by side | `results/layer_distance_overview.png`, `results/credibility_overview.png` |

## 📊 The Two Presentation Plots (per combination)

- **`layer_distance_profile.png`** - consistent vs inconsistent mean distance
  per layer, the data-derived threshold (consistent mean + std) and the
  crossing layer - the "Reading the distance profile" slide.
- **`credibility_groups.png`** - hypotheses split into low / high credibility
  by the data-derived threshold (mean - std of hypothesis credibilities); the
  bar height is the share of each group that became inconsistent - the
  "Phase A: can we predict it in advance?" slide. The same numbers are stored
  in `diagnosis.json -> credibility_groups`.

## 🔑 When the Two Signals Disagree

Layer Distance asks **where the representations split** (which layer);
Credibility asks **where the paraphrase lands** relative to the training
data. They measure different things, so disagreement is **not a failure** -
it is itself a finding: the inconsistency is multi-dimensional, with both the
representation and the classifier contributing. `diagnosis.json` records this
explicitly (`signals_agree` + note).

## 🚀 Meta-Runner

```bash
python run_step4.py
python run_step4.py --models BERT-base --datasets SNLI
```
