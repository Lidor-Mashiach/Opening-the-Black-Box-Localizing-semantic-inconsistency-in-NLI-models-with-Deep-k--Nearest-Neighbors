"""Step-3a: choose K per (model, dataset) pair - on the validation portion only.

For every candidate K (configs/base.yaml -> dknn.k_values) we compute the
credibility drop  Credibility(Hypothesis) - Credibility(Paraphrase)  and
correlate it with actual inconsistency. The K with the strongest correlation
wins. One neighbour fetch at max(K) is sliced per K, so this costs a single
k-NN pass.

Run:  python select_k.py --model BERT-base --dataset SNLI
"""
import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path[:0] = [str(REPO_ROOT), str(REPO_ROOT / "Research-Pipeline")]

import numpy as np

from common.alignment import load_aligned
from common.config_loader import load_config, step_results_dir
from eval_metrics import DkNN, credibility_from_neighbor_labels, select_best_k

STEP_DIRNAME = "Step-3_DkNN-and-Layer-Distance"


def main(model_key, dataset_key):
    cfg = load_config(model_key, dataset_key)
    data = load_aligned(cfg)
    k_values = cfg["dknn"]["k_values"]
    max_k = max(k_values)

    print(f"[select-k] {model_key} x {dataset_key}: "
          f"bank={len(data['bank_labels'])}, paraphrase rows={len(data['pair_ids'])}")
    dknn = DkNN().fit(data["bank_reps"], data["bank_labels"])

    # hypotheses: score each unique one once (excluding itself from the bank)
    unique_rows = np.unique(data["hypo_rows"])
    pos_of = {row: i for i, row in enumerate(unique_rows)}
    row_pos = np.array([pos_of[r] for r in data["hypo_rows"]])
    nl_hypo = dknn.top_neighbor_labels(data["bank_reps"][unique_rows], max_k,
                                       exclude_self_indices=unique_rows)
    nl_para = dknn.top_neighbor_labels(data["para_reps"], max_k)
    hypo_final = data["bank_labels"][unique_rows]   # hypothesis prediction == gold

    val = data["val_mask"]
    drops_by_k = {}
    for k in k_values:
        cred_h = credibility_from_neighbor_labels(nl_hypo, hypo_final, k=k)[row_pos]
        cred_p = credibility_from_neighbor_labels(nl_para, data["para_preds"], k=k)
        drops_by_k[k] = (cred_h - cred_p)[val]

    best_k, correlations = select_best_k(drops_by_k, data["inconsistent"][val])

    out_dir = step_results_dir(STEP_DIRNAME, cfg)
    out_dir.mkdir(parents=True, exist_ok=True)
    result = {
        "model": model_key,
        "dataset": dataset_key,
        "chosen_k": int(best_k),
        "correlations_on_validation": {str(k): float(v) for k, v in correlations.items()},
        "n_validation_rows": int(val.sum()),
        "n_total_rows": int(len(val)),
    }
    with open(out_dir / "k_selection.json", "w") as f:
        json.dump(result, f, indent=2)
    print(f"[select-k] chosen K = {best_k}  "
          f"(correlations: { {k: round(v, 4) for k, v in correlations.items()} })")
    return best_k


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--dataset", required=True)
    args = parser.parse_args()
    main(args.model, args.dataset)
