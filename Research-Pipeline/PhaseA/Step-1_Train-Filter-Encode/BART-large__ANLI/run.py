"""Meta-runner for BART-large x ANLI: train -> filter -> encode -> evaluate, in order.

Resumable: every stage skips itself when its output already exists, and
training resumes from a rolling epoch checkpoint after a crash. Pass --force
to redo everything (e.g. after Optuna wrote tuned hyper-parameters).
results/ is created automatically and is git-ignored.
"""
import argparse
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ORDER = ["train.py", "build_filtered_dataset.py", "encode_hypotheses.py", "evaluate.py"]

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    extra = ["--force"] if args.force else []
    for script in ORDER:
        print(f"\n===== BART-large x ANLI | {script} =====")
        if subprocess.run([sys.executable, str(HERE / script)] + extra).returncode != 0:
            sys.exit(f"FAILED: {script}")
    print("\nStep-1 finished for BART-large x ANLI.")
