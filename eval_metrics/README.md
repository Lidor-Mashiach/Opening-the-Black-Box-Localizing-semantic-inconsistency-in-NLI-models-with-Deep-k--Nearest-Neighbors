# eval_metrics - All Evaluation Metrics

One file per metric, exactly as characterized in the methodology chapter.
Both Phase A and Phase B import from here - the metric code exists **once**,
never copy-pasted into pipeline steps.

> Naming note: the folder is `eval_metrics` (underscore) rather than
> `eval-metrics` because Python packages cannot contain a hyphen. Same
> folder, importable.

## The Metrics

| File | Metric | Source | One-liner |
|------|--------|--------|-----------|
| `relaxed_fooling_rate.py` | **RFR** | Anchor paper (Arakelyan et al., 2024) | Share of pairs whose prediction changed in *any* way |
| `strict_fooling_rate.py` | **SFR** | Anchor paper | Share of pairs with a full Entailment <-> Contradiction flip |
| `paraphrastic_consistency.py` | **PC** | Srikanth et al. (2024) | Probability a pair keeps the same prediction (= 1 - RFR) |
| `pattern_accuracy.py` | **PA** | MERGE (Zgreaban et al., 2025) | A hypothesis counts only if **all** ~5 of its paraphrases stay consistent |
| `layer_distance.py` | **Layer Distance** | This work | Per-layer cosine distance hypothesis vs paraphrase + data-derived threshold + crossing-layer region |
| `dknn_credibility.py` | **Credibility** | Deep k-NN (Papernot & McDaniel, 2018) | Layer-wise k-NN voting: supporters of the final prediction / (K x layers) |

## Key Rules Baked In

- **Neutral counts.** Neutral -> Neutral is consistent; any change (including
  to/from Neutral) is inconsistent for RFR/PC. SFR counts only full flips.
- **Thresholds come from the data.** Layer-distance threshold = consistent
  mean + std per layer; anomalous credibility drop = mean + std of all drops
  per (model, dataset) pair. Never a fixed constant.
- **Self-exclusion.** Hypotheses are bank members; scoring them excludes
  their own entry (otherwise each is its own nearest neighbour at distance 0
  and credibility is inflated).
- **K is chosen per pair** on the validation portion - `select_best_k`
  maximizes the correlation between credibility drop and actual
  inconsistency. One neighbour fetch at max(K) is sliced per K.
- **Drop convention.** `drop = Credibility(Hypothesis) - Credibility(Paraphrase)`;
  a *positive* drop means the paraphrase's credibility went **down**
  (generalization signal), a *negative* drop means it went up for the wrong
  reason (representation signal).

## Meta-Runner

```bash
python eval_metrics/sanity_check.py     # from the repository root
```

Pure-NumPy synthetic tests for every metric (no GPU, no downloads) - all
lines must print `OK`.
