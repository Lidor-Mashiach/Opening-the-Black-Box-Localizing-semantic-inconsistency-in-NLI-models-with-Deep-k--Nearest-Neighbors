"""Meta-runner for Step-1: train -> filter -> encode -> evaluate for every
(model, dataset) combination folder in this directory.

Run:  python run_step1.py                          (all 12 combinations)
      python run_step1.py --models BERT-base --datasets SNLI,MNLI
Heads-up: this is the heavy step (12 fine-tunings). On the BGU SLURM cluster
run one combination per job, e.g.:
      python BERT-base__SNLI/run.py
"""
import argparse
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[2]
sys.path[:0] = [str(REPO_ROOT), str(REPO_ROOT / "Research-Pipeline")]

from common.config_loader import list_dataset_keys, list_model_keys


def main(models, datasets, force=False):
    for model in models:
        for dataset in datasets:
            combo = HERE / f"{model}__{dataset}"
            if not combo.is_dir():
                sys.exit(f"[step-1] missing combination folder: {combo}")
            print(f"\n########## Step-1 | {model} x {dataset} ##########")
            cmd = [sys.executable, str(combo / "run.py")] + (["--force"] if force else [])
            if subprocess.run(cmd).returncode != 0:
                sys.exit(f"[step-1] FAILED for {model} x {dataset}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--models", default=None, help="comma-separated subset")
    parser.add_argument("--datasets", default=None, help="comma-separated subset")
    parser.add_argument("--force", action="store_true",
                        help="redo finished stages (e.g. after Optuna tuning)")
    args = parser.parse_args()
    main(args.models.split(",") if args.models else list_model_keys(),
         args.datasets.split(",") if args.datasets else list_dataset_keys(),
         force=args.force)
