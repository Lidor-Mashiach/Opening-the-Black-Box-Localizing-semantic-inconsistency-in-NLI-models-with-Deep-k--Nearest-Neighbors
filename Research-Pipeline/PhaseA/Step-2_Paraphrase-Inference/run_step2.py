"""Meta-runner for Step-2: paraphrase inference over every combination.

Run:  python run_step2.py
      python run_step2.py --models RoBERTa-large --datasets ANLI
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
            print(f"\n===== Step-2 | {model} x {dataset} =====")
            cmd = [sys.executable, str(HERE / "paraphrase_inference.py"),
                   "--model", model, "--dataset", dataset] + (["--force"] if force else [])
            if subprocess.run(cmd).returncode != 0:
                sys.exit(f"[step-2] FAILED: {' '.join(cmd)}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--models", default=None)
    parser.add_argument("--datasets", default=None)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    main(args.models.split(",") if args.models else list_model_keys(),
         args.datasets.split(",") if args.datasets else list_dataset_keys(),
         force=args.force)
