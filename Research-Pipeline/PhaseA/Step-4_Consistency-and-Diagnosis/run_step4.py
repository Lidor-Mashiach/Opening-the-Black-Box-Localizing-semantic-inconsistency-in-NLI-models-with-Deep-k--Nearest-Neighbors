"""Meta-runner for Step-4: aggregated consistency metrics + per-pair diagnosis.

Run:  python run_step4.py
      python run_step4.py --models BERT-base --datasets SNLI
"""
import argparse
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[2]
sys.path[:0] = [str(REPO_ROOT), str(REPO_ROOT / "Research-Pipeline")]

from common.config_loader import (list_dataset_keys, list_model_keys,
                                  load_config, step_results_dir)

STEP3_DIRNAME = "Step-3_DkNN-and-Layer-Distance"


def main(models, datasets):
    cmd = [sys.executable, str(HERE / "compute_consistency_metrics.py"),
           "--models", ",".join(models), "--datasets", ",".join(datasets)]
    if subprocess.run(cmd).returncode != 0:
        sys.exit("[step-4] consistency metrics FAILED")

    for model in models:
        for dataset in datasets:
            cfg = load_config(model, dataset)
            if not (step_results_dir(STEP3_DIRNAME, cfg) / "credibility_summary.json").exists():
                print(f"[step-4] SKIP diagnosis {model} x {dataset}: Step-3 output missing")
                continue
            cmd = [sys.executable, str(HERE / "diagnose.py"),
                   "--model", model, "--dataset", dataset]
            if subprocess.run(cmd).returncode != 0:
                sys.exit(f"[step-4] diagnosis FAILED for {model} x {dataset}")

    cmd = [sys.executable, str(HERE / "plot_overview.py"),
           "--models", ",".join(models), "--datasets", ",".join(datasets)]
    if subprocess.run(cmd).returncode != 0:
        sys.exit("[step-4] overview plots FAILED")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--models", default=None)
    parser.add_argument("--datasets", default=None)
    parser.add_argument("--force", action="store_true",
                        help="accepted for uniform pass-through; analysis always recomputes")
    args = parser.parse_args()
    main(args.models.split(",") if args.models else list_model_keys(),
         args.datasets.split(",") if args.datasets else list_dataset_keys())
