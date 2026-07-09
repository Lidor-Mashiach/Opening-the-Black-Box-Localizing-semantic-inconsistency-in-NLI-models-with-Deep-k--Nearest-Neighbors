"""Step-1c: layer-wise representation bank of BERT-base's correct ANLI hypotheses.

Thin wrapper around common.encoding - all logic is shared. Skips automatically when its
output already exists; pass --force to redo (e.g. after Optuna tuning).
"""
import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[4]
sys.path[:0] = [str(REPO_ROOT), str(REPO_ROOT / "Research-Pipeline")]

from common.encoding import encode_hypothesis_bank

MODEL_KEY = "BERT-base"
DATASET_KEY = "ANLI"

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--force", action="store_true",
                        help="redo even when the output already exists")
    args = parser.parse_args()
    encode_hypothesis_bank(MODEL_KEY, DATASET_KEY, force=args.force)
