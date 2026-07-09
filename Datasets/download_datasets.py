"""Meta-runner for Datasets/: download SNLI / MNLI / ANLI into raw/.

Pulls every dataset from the HuggingFace hub with the exact same loader the
pipeline uses (identical schema, -1 labels dropped, ANLI rounds concatenated)
and saves each logical split as parquet under Datasets/<dataset>/raw/.

Run:  python download_datasets.py
      python download_datasets.py --datasets SNLI,ANLI
"""
import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(REPO_ROOT), str(REPO_ROOT / "Research-Pipeline")]

from common import data_loading
from common.config_loader import dataset_dir, list_dataset_keys, load_config

SPLITS = ("train", "validation", "test")


def main(datasets):
    for dataset_key in datasets:
        cfg = load_config(dataset_key=dataset_key)
        out_dir = dataset_dir(cfg) / "raw"
        out_dir.mkdir(parents=True, exist_ok=True)
        print(f"\n===== {dataset_key} -> {out_dir} =====")
        for split in SPLITS:
            ds = data_loading.load_nli(cfg, split)
            ds = data_loading.add_pair_ids(ds, cfg, split)
            path = out_dir / f"{split}.parquet"
            ds.to_parquet(str(path))
            print(f"  {split:<11} {len(ds):>7} rows -> {path.name}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--datasets", default=None, help="comma-separated subset")
    args = parser.parse_args()
    main(args.datasets.split(",") if args.datasets else list_dataset_keys())
