"""Meta-runner for Step-3: K selection -> Credibility -> Layer Distance,
for every (model, dataset) combination that finished Step-2.

Run:  python run_step3.py                      (all combinations)
      python run_step3.py --models BERT-base --datasets SNLI,MNLI
"""
import argparse
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[2]
sys.path[:0] = [str(REPO_ROOT), str(REPO_ROOT / "Research-Pipeline")]

from common.config_loader import (encoded_dir, list_dataset_keys,
                                  list_model_keys, load_config)

SCRIPTS = ["select_k.py", "compute_credibility.py", "compute_layer_distances.py"]


def main(models, datasets):
    for model in models:
        for dataset in datasets:
            cfg = load_config(model, dataset)
            if not (encoded_dir(cfg) / "paraphrases.npz").exists():
                print(f"[step-3] SKIP {model} x {dataset}: Step-2 output missing")
                continue
            print(f"\n===== Step-3 | {model} x {dataset} =====")
            for script in SCRIPTS:
                cmd = [sys.executable, str(HERE / script),
                       "--model", model, "--dataset", dataset]
                result = subprocess.run(cmd)
                if result.returncode != 0:
                    sys.exit(f"[step-3] FAILED: {' '.join(cmd)}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--models", default=None, help="comma-separated subset")
    parser.add_argument("--datasets", default=None, help="comma-separated subset")
    parser.add_argument("--force", action="store_true",
                        help="accepted for uniform pass-through; analysis always recomputes")
    args = parser.parse_args()
    models = args.models.split(",") if args.models else list_model_keys()
    datasets = args.datasets.split(",") if args.datasets else list_dataset_keys()
    main(models, datasets)
