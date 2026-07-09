"""Step-2 entry point: paraphrase inference for one (model, dataset) pair.

Thin wrapper around common.paraphrase_inference - see that module for the
full input/output contract. Skips gracefully when the paraphrase CSV is not
in place yet.

Run:  python paraphrase_inference.py --model BERT-base --dataset SNLI
"""
import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path[:0] = [str(REPO_ROOT), str(REPO_ROOT / "Research-Pipeline")]

from common.paraphrase_inference import run_paraphrase_inference

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    run_paraphrase_inference(args.model, args.dataset, force=args.force)
