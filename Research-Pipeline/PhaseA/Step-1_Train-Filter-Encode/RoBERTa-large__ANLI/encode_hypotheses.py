"""Step-1c: layer-wise representation bank of RoBERTa-large's correct ANLI hypotheses.

Thin wrapper around common.encoding - all logic is shared, nothing is duplicated.
"""
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[4]
sys.path[:0] = [str(REPO_ROOT), str(REPO_ROOT / "Research-Pipeline")]

from common.encoding import encode_hypothesis_bank

MODEL_KEY = "RoBERTa-large"
DATASET_KEY = "ANLI"

if __name__ == "__main__":
    encode_hypothesis_bank(MODEL_KEY, DATASET_KEY)
