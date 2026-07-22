# 2 Step-2 - Paraphrase Inference

The actual test: every (premise, **paraphrase**) pair goes through the
fine-tuned model. Predictions feed the consistency metrics (Step-4)
representations feed DkNN Credibility and Layer Distance (Step-3).

## Input

The **dataset-level paraphrase bank**
(`Datasets/<dataset>/paraphrases/paraphrase_bank.csv` - up to 5 verified
paraphrases per hypothesis, model-independent), **intersected here
with this model's correct-only set**: only rows whose `pair_id` appears in
`Runtime-Data/<MODEL>/<DATASET>/train_correct.csv` (the reduced copy built by
Step-1) enter. So every model is tested on the
exact same paraphrases for shared hypotheses, and
`label == hypothesis prediction` holds by construction. Contract + build
protocol: [`paraphrases/README.md`](../../../Datasets/SNLI-Stanford_NLI/paraphrases/README.md).
While the bank is missing the step **skips gracefully** - nothing fails.

## Outputs

| File | Where | Content |
|------|-------|---------|
| `paraphrases_used.csv` | `Runtime-Data/<MODEL>/<DATASET>/` | The per-model COPY of the shared bank (bank intersect this model's correct set) |
| `paraphrases.npz` | `Runtime-Data/<MODEL>/<DATASET>/` | Per-layer representations + predictions |
| `paraphrase_predictions.csv` | `results/<MODEL>__<DATASET>/` | `pair_id, para_idx, hypothesis_pred, paraphrase_pred, consistent, strict_flip` |
| `inference_stats.json` | `results/<MODEL>__<DATASET>/` | Row counts + consistent / strict-flip shares |

`hypothesis_pred` equals the gold label by design - only correctly-classified
hypotheses entered the experiment.

## Meta-Runner

```bash
python run_step2.py
python run_step2.py --models RoBERTa-large --datasets ANLI
```