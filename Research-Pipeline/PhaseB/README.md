# 🅱️ Phase B - From Diagnosis to a Targeted Fix

**Status: pending Phase A results.** This folder intentionally holds only
this README (+ `.keep`); the implementation is derived from what Phase A's
Credibility signal shows, per the methodology document.

## 🔀 What Happens per Phase-A Scenario

| Phase-A verdict | Meaning | Phase-B action |
|-----------------|---------|----------------|
| `generalization` (paraphrase credibility **dropped**) | The paraphrase landed in a region the training set barely covers | Targeted paraphrase augmentation + fine-tuning (the main experiment below) |
| `representation` (credibility **rose** for the wrong label) | The model grouped certain phrasings with a wrong label | Report as a key finding; augmentation would have to *break* the wrong grouping, not just add data |
| `classifier` (credibility unchanged, answers still flipped) | Same internal representation, different final decision | Classifier fine-tuning / regularization experiments; improvement is not expected |

## 🔬 The Targeted-Augmentation Experiment (generalization case)

1. **Select** only hypotheses whose original paraphrase showed an anomalous
   credibility drop (`anomalous_drop == 1` in Step-3's `credibility.csv`).
   Adding paraphrases to *all* hypotheses would be blind treatment and would
   not test the signal.
2. **Generate new paraphrases** for those hypotheses only, and add them to
   the **training set**. The test set stays untouched, so the re-test is
   trustworthy.
3. **Fine-tune** the existing checkpoint - never retrain from scratch.
4. **Rebuild the representation bank** (Step-1c) - the model changed, so the
   old representations are no longer valid.
5. **Re-run Steps 2-4** and compare all metrics.

## ⚖️ Two Mandatory Baselines

| Baseline | Purpose |
|----------|---------|
| The original model, untouched | Absolute reference |
| A model given the **same number** of paraphrases chosen **at random** | Separates "the signal is informative" from "more data helps" |

Only if the targeted model beats the random-augmentation baseline can we
claim that Credibility is an informative signal - the differentiator of this
work over naive augmentation.
