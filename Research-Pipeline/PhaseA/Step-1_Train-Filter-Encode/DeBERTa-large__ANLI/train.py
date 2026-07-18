"""Step-1a: fine-tune DeBERTa-large on ANLI (resumable; skips when finished).

Thin wrapper around common.training - all logic is shared, nothing is duplicated.
"""
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[4]
sys.path[:0] = [str(REPO_ROOT), str(REPO_ROOT / "Research-Pipeline")]

from common.training import fine_tune

MODEL_KEY = "DeBERTa-large"
DATASET_KEY = "ANLI"

if __name__ == "__main__":
    fine_tune(MODEL_KEY, DATASET_KEY)
