# 🧠 Encoded Datasets - Layer-Wise Representation Banks

The internal-representation side of the research. For every
`<MODEL>/<DATASET>/` pair this folder stores the pooled representation of
every example **at every layer** - the raw material for both Deep k-NN
Credibility and Layer Distance.

## 📁 Layout

```
Encoded_Datasets/
└── <MODEL>/                    # BERT-base / RoBERTa-large / DeBERTa-large / BART-large
    └── <DATASET>/              # SNLI / MNLI / ANLI
        ├── hypotheses.npz      # the labeled DkNN bank      (Step-1c)
        ├── hypotheses_meta.json
        └── paraphrases.npz     # paraphrase representations (Step-2)
```

## 📄 File Contract

| File | Arrays | Shapes |
|------|--------|--------|
| `hypotheses.npz` | `reps`, `labels`, `pair_ids` | `[n, L, dim]` float16 · `[n]` int · `[n]` str |
| `paraphrases.npz` | `reps`, `preds`, `pair_ids`, `para_idx` | `[m, L, dim]` float16 · `[m]` int · `[m]` str · `[m]` int |

`L` = number of transformer layers (the embedding layer is dropped, so index
`i` is layer `i+1`). Pooling per model: `cls` (first token) for BERT /
RoBERTa / DeBERTa, `eos` (last non-padding token) for BART - see
`configs/models/*.yaml`.

> Bank size is capped by `sampling.max_bank_examples` (seeded subsample,
> `configs/base.yaml`) - full SNLI/MNLI banks would weigh tens of GB and add
> nothing to k-NN quality. Set to `null` to keep everything.

These `.npz` files are **git-ignored** (heavy, fully reproducible); the
`.keep` files preserve the folder structure.
