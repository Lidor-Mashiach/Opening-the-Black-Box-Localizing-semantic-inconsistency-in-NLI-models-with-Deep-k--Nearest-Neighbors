# Comparison Table - Datasets & Eval Metrics

`DataSets_and_Eval_Metrics.xlsx` - the literature comparison matrix: for
every reviewed paper, which models it tested, which datasets it checked,
which new dataset it introduced (if any), and which evaluation metrics it
used. This table is where the project's model / dataset / metric choices
were made.

## Color Legend

| Highlight | Meaning |
|-----------|---------|
| Yellow rows | The two protocol-defining works: the anchor paper (Arakelyan 2024) and MERGE |
| Orange rows | The dataset papers themselves: SNLI, MNLI, ANLI |
| Red text (models column) | Models **selected** for this study: BERT-base, RoBERTa-large, DeBERTa-large, BART-large |
| Blue text (datasets column) | Datasets **selected**: SNLI, MNLI, ANLI |
| Purple text (metrics column) | Metrics **adopted**: Relaxed/Strict Fooling Rate, PC, PA |

The choices encoded here are exactly what appears in
[`configs/`](../../configs/README.md) and in the pipeline.
