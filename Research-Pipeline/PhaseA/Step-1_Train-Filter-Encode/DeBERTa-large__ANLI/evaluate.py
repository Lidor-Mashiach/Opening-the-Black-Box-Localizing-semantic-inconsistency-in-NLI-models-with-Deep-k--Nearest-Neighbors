"""Step-1d: baseline accuracy of fine-tuned DeBERTa-large on ANLI validation + test.

Thin wrapper around common.evaluation - all logic is shared, nothing is duplicated.
"""
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[4]
sys.path[:0] = [str(REPO_ROOT), str(REPO_ROOT / "Research-Pipeline")]

from common.evaluation import evaluate_baseline

MODEL_KEY = "DeBERTa-large"
DATASET_KEY = "ANLI"

if __name__ == "__main__":
    evaluate_baseline(MODEL_KEY, DATASET_KEY)
