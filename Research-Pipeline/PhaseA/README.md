# Phase A - Diagnosis

Does semantic inconsistency show up **inside** the model, at the level of its
internal representations, and not only at the final output? Four steps, each
in its own folder with its own meta-runner.

## The Four Steps

| Step | Folder | What happens | Key outputs |
|------|--------|--------------|-------------|
| 1 | [`Step-1_Train-Filter-Encode/`](Step-1_Train-Filter-Encode/README.md) | Fine-tune each model on each dataset, keep only correctly-classified train rows, store every kept hypothesis at every layer, baseline eval | checkpoints, `train_correct.csv`, `hypotheses.npz`, `baseline_eval.json` |
| 2 | [`Step-2_Paraphrase-Inference/`](Step-2_Paraphrase-Inference/README.md) | Run (premise, paraphrase) through the model; save predictions + representations | `paraphrase_predictions.csv`, `paraphrases.npz` |
| 3 | [`Step-3_DkNN-and-Layer-Distance/`](Step-3_DkNN-and-Layer-Distance/README.md) | Choose K per pair, compute Credibility + drops + scenario, compute per-layer distances + crossing layer | `k_selection.json`, `credibility.csv/_summary.json`, `layer_profile.csv`, `distance_summary.json` |
| 4 | [`Step-4_Consistency-and-Diagnosis/`](Step-4_Consistency-and-Diagnosis/README.md) | RFR / SFR / PC / PA per pair + joint Credibility x Layer-Distance diagnosis + profile plot | `consistency_metrics.csv`, `diagnosis.json`, `layer_distance_profile.png` |

## Phase A's Key Finding

After Step-3, `credibility_summary.json` holds the headline number: the
correlation, **on the test portion K selection never saw**, between the
credibility drop and actual inconsistency. If it is positive and significant,
DkNN identifies the vulnerable examples *in advance* - before any paraphrase
is shown.

## Meta-Runner

```bash
python run_phase_a.py                      # steps 1-4, all 12 combinations
python run_phase_a.py --steps 3,4          # rerun the analysis only
python run_phase_a.py --models BERT-base --datasets SNLI,MNLI
```
