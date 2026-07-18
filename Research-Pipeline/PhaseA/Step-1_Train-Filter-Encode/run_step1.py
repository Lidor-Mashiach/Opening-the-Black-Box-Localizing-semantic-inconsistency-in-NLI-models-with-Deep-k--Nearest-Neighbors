"""Meta-runner for Step-1: every (model, dataset) combination, in order.

Each combination's own run.py wipes its workspace first, then runs
train -> filter -> encode -> evaluate. No flags.

Run:  python run_step1.py
      python run_step1.py --models BERT-base --datasets SNLI,MNLI
"""
import argparse
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[2]
sys.path[:0] = [str(REPO_ROOT), str(REPO_ROOT / "Research-Pipeline")]

from common.config_loader import list_dataset_keys, list_model_keys
from common.logging_utils import banner


def main(models, datasets):
    for model in models:
        for dataset in datasets:
            combo = HERE / f"{model}__{dataset}"
            if not combo.is_dir():
                sys.exit(f"[step-1] missing combination folder: {combo}")
            banner("STEP-1", "combination starts", model, dataset)
            if subprocess.run([sys.executable, str(combo / "run.py")]).returncode != 0:
                sys.exit(f"[step-1] FAILED for {model} x {dataset}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--models", default=None, help="comma-separated subset")
    parser.add_argument("--datasets", default=None, help="comma-separated subset")
    args = parser.parse_args()
    main(args.models.split(",") if args.models else list_model_keys(),
         args.datasets.split(",") if args.datasets else list_dataset_keys())
