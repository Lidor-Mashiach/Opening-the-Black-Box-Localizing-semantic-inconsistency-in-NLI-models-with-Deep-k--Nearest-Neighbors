"""Meta-runner for Optuna tuning.

Run:  python run_tuning.py --model BERT-base --dataset SNLI    (one study -
          this is what every sbatch file calls)
      python run_tuning.py                                     (all 12 studies,
          sequentially - for a local machine)
      python run_tuning.py --n-trials 30                       (override target)

Everything is resumable: rerunning continues the sqlite study from where it
stopped. See optuna_search.py for the objective, search space and outputs.
"""
import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(REPO_ROOT), str(REPO_ROOT / "Research-Pipeline")]

from common.config_loader import list_dataset_keys, list_model_keys
from optuna_search import run_study

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default=None, help="a single model key")
    parser.add_argument("--dataset", default=None, help="a single dataset key")
    parser.add_argument("--models", default=None, help="comma-separated subset")
    parser.add_argument("--datasets", default=None, help="comma-separated subset")
    parser.add_argument("--n-trials", type=int, default=None,
                        help="override tuning.n_trials from the config")
    args = parser.parse_args()

    if args.model and args.dataset:                       # single study (sbatch mode)
        run_study(args.model, args.dataset, n_trials=args.n_trials)
    else:                                                 # sequential meta mode
        models = args.models.split(",") if args.models else list_model_keys()
        datasets = args.datasets.split(",") if args.datasets else list_dataset_keys()
        for dataset in datasets:
            for model in models:
                print(f"\n########## Tuning | {model} x {dataset} ##########")
                run_study(model, dataset, n_trials=args.n_trials)
