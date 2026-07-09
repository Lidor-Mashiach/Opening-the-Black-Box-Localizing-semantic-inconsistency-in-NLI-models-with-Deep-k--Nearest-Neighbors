# 📝 Paraphrases - Generation, Verification & File Contract

This folder holds the hypothesis paraphrases that drive the whole experiment.
**Base condition (non-negotiable):** every paraphrase must hold the *exact
same logical relation* to the premise as the original hypothesis - i.e. in
the ground truth, `Premise : Hypothesis == Premise : Paraphrase`. If the
relation changes, an answer change is justified and there is nothing to
research. The **bidirectional-entailment verification** below is what
enforces this; unverified paraphrases must never enter.

## ⚠️ Availability Status (verified July 2026)

There is **no public, verified corpus** of ~5 relation-preserving paraphrases
per hypothesis for SNLI / MNLI / ANLI:

| Resource | What it offers | Why it is not a drop-in |
|----------|----------------|------------------------|
| Anchor paper (Arakelyan et al., EACL 2024) | Semantics-preserving variations, verified by **symmetric equivalence entailment** | Generated *per evaluated model* by conditional text generation - no static dataset released |
| ParaNlu (Srikanth et al., TACL 2024) | 7,782 human-written, manually validated label-preserving paraphrases | Built on **abductive / defeasible** NLI, not 3-class; ~1,000 base problems |
| MERGE / AllVar (Zgreaban et al., 2025) | MLM-based minimal expression replacements with safeguarding filters | SNLI-derived only; variant generation, not free-form paraphrase |

## 🛠️ Option A - The In-Repo Generator (recommended)

`Datasets/generate_paraphrases.py` implements the full protocol:

| Stage | Model (configs/base.yaml) | Why this one |
|-------|---------------------------|--------------|
| Generation | `humarin/chatgpt_paraphraser_on_T5_base` | The leading open paraphraser on the HF Hub (T5-base trained on 6.3M ChatGPT paraphrase pairs); decoding follows its model card (diverse beam search) |
| Verification | `MoritzLaurer/DeBERTa-v3-large-mnli-fever-anli-ling-wanli` | The strongest public NLI model on the Hub (DeBERTa-v3-large, 885K pairs, 91.2% MNLI-m, SOTA on ANLI) - keeps a candidate only if hypothesis <-> candidate entail **each other** |

```bash
python Datasets/generate_paraphrases.py --models BERT-base --datasets SNLI
python Datasets/generate_paraphrases.py --limit 50     # quick trial
```

The sampled hypotheses are drawn with the exact same seeded chain as the
encoded bank (`common/sampling.py`), so **every paraphrased hypothesis is
guaranteed to exist in the DkNN bank**. A `<MODEL>__paraphrases_stats.json`
records quota coverage and the verification acceptance rate. The verifier
only judges hypothesis <-> paraphrase equivalence - it is independent of the
four models under study.

## ✍️ Option B - Any External Generator (e.g. an instruction LLM)

Any source is acceptable **as long as the same bidirectional-entailment gate
is applied** and the output respects the contract below. The verifier stage
is the research-validity guarantee, not the generator.

## 📄 File Contract - `<MODEL>__paraphrases.csv`

| Column | Type | Meaning |
|--------|------|---------|
| `pair_id` | str | Id of the source hypothesis - **must exist in the encoded bank** |
| `premise` | str | The ORIGINAL premise, unchanged |
| `paraphrase` | str | Reworded hypothesis, same logical relation |
| `label` | int | Gold label (0/1/2) - equals the hypothesis prediction by design |
| `para_idx` | int | 0..4 (optional - created automatically when absent) |

One file **per model** because each model keeps a different "correct only"
hypothesis set. Step-2 skips gracefully while a file is missing.
