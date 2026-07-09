# 📖 Literature Review - The 14 Papers

Index of every work the research builds on, with its role in the project.
Drop the paper PDFs into this folder (the `.keep` file just preserves it
until then).

## 📄 Foundations & Datasets

| # | Paper | Venue | Role | Link |
|---|-------|-------|------|------|
| 1 | Bowman et al. - *A Large Annotated Corpus for Learning NLI* (SNLI) | EMNLP 2015 | Dataset (easiest) | [arXiv:1508.05326](https://arxiv.org/abs/1508.05326) |
| 2 | Williams et al. - *A Broad-Coverage Challenge Corpus...* (MNLI) | NAACL-HLT 2018 | Dataset (medium) | [arXiv:1704.05426](https://arxiv.org/abs/1704.05426) |
| 3 | Nie et al. - *Adversarial NLI* (ANLI) | ACL 2020 | Dataset (hardest) | [arXiv:1910.14599](https://arxiv.org/abs/1910.14599) |

## 📄 The Problem - Fragility of NLI Models

| # | Paper | Venue | Role | Link |
|---|-------|-------|------|------|
| 4 | **Arakelyan, Liu & Augenstein - *Semantic Sensitivities and Inconsistent Predictions*** | **EACL 2024** | **Anchor paper** - the phenomenon, paraphrase protocol, Fooling Rates | [arXiv:2401.14440](https://arxiv.org/abs/2401.14440) |
| 5 | McCoy, Pavlick & Linzen - *Right for the Wrong Reasons* (HANS) | ACL 2019 | Syntactic-heuristic failures | [arXiv:1902.01007](https://arxiv.org/abs/1902.01007) |
| 6 | Glockner, Shwartz & Goldberg - *Breaking NLI Systems...* | ACL 2018 | Simple lexical-inference failures | [arXiv:1805.02266](https://arxiv.org/abs/1805.02266) |
| 7 | Naik et al. - *Stress Test Evaluation for NLI* | COLING 2018 | Stress-test failures | [arXiv:1806.00692](https://arxiv.org/abs/1806.00692) |
| 8 | Richardson et al. - *Probing NLI Models through Semantic Fragments* | AAAI 2020 | Basic-logic failures | [arXiv:1909.07521](https://arxiv.org/abs/1909.07521) |

## 📄 Output-Level Consistency Metrics

| # | Paper | Venue | Role | Link |
|---|-------|-------|------|------|
| 9 | Srikanth, Carpuat & Rudinger - *How Often Are Errors... Paraphrastic Variability?* | TACL 2024 | PC metric; ParaNlu protocol reference | [arXiv:2404.11717](https://arxiv.org/abs/2404.11717) |
| 10 | Zgreaban, Deoskar & Abzianidze - *MERGE* | arXiv 2025 | PA metric; SNLI variant generation | [arXiv:2510.24295](https://arxiv.org/abs/2510.24295) |
| 11 | Wu & Last - *Transitive Self-Consistency Evaluation of NLI Models* | EMNLP 2025 | Consistency without gold labels (context) | pp. 22626-22642 |

## 📄 The Method - Inside the Model

| # | Paper | Venue | Role | Link |
|---|-------|-------|------|------|
| 12 | Papernot & McDaniel - *Deep k-Nearest Neighbors* | arXiv 2018 | The DkNN method + Credibility | [arXiv:1803.04765](https://arxiv.org/abs/1803.04765) |
| 13 | Ulanovski, Blyachman & Bechler-Speicher - *Improving LLM Final Representations with Inter-Layer Geometry* | arXiv 2026 | Layer-geometry perspective | [arXiv:2603.22665](https://arxiv.org/abs/2603.22665) |
| 14 | Arakelyan - *Reasoning Inconsistencies and How to Mitigate Them* | PhD Thesis, U. Copenhagen 2025 | Extended context for the anchor work | [arXiv:2504.02577](https://arxiv.org/abs/2504.02577) |
