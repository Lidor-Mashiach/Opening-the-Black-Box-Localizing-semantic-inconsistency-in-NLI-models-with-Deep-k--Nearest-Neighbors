# 2️⃣ Step-2 - Paraphrase Inference

The actual test: every (premise, **paraphrase**) pair goes through the
fine-tuned model. Predictions feed the consistency metrics (Step-4);
representations feed DkNN Credibility and Layer Distance (Step-3).

## 📥 Input

`Datasets/<dataset>/paraphrases/<MODEL>__paraphrases.csv` - the acquisition
protocol and column contract live in each dataset's
[`paraphrases/README.md`](../../../Datasets/SNLI-Stanford_NLI/paraphrases/README.md).
While a file is missing the step **skips gracefully** and prints where the
contract is documented - nothing fails.

## 📤 Outputs

| File | Where | Content |
|------|-------|---------|
| `paraphrases.npz` | `Datasets/Encoded_Datasets/<MODEL>/<DATASET>/` | Per-layer representations + predictions |
| `paraphrase_predictions.csv` | `results/<MODEL>__<DATASET>/` | `pair_id, para_idx, hypothesis_pred, paraphrase_pred, consistent, strict_flip` |
| `inference_stats.json` | `results/<MODEL>__<DATASET>/` | Row counts + consistent / strict-flip shares |

`hypothesis_pred` equals the gold label by design - only correctly-classified
hypotheses entered the experiment.

## 🚀 Meta-Runner

```bash
python run_step2.py
python run_step2.py --models RoBERTa-large --datasets ANLI
```
