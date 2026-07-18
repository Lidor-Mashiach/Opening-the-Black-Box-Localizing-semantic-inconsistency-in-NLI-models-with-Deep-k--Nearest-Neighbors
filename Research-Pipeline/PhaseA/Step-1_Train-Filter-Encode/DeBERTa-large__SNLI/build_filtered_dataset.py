"""Step-1b: reduced COPY of SNLI - only rows DeBERTa-large classifies correctly.

Thin wrapper around common.filtering - all logic is shared, nothing is duplicated.
"""
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[4]
sys.path[:0] = [str(REPO_ROOT), str(REPO_ROOT / "Research-Pipeline")]

from common.filtering import build_filtered_dataset

MODEL_KEY = "DeBERTa-large"
DATASET_KEY = "SNLI"

if __name__ == "__main__":
    build_filtered_dataset(MODEL_KEY, DATASET_KEY)
