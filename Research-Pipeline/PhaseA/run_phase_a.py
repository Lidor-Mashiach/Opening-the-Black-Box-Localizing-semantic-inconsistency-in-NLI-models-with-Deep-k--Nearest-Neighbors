"""Meta-runner for Phase A (diagnosis): Steps 1 -> 2 -> 3 -> 4, in order.

Run:  python run_phase_a.py
      python run_phase_a.py --steps 3,4                (rerun analysis only)
      python run_phase_a.py --models BERT-base --datasets SNLI
"""
import argparse
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent

STEP_RUNNERS = {
    "1": HERE / "Step-1_Train-Filter-Encode" / "run_step1.py",
    "2": HERE / "Step-2_Paraphrase-Inference" / "run_step2.py",
    "3": HERE / "Step-3_DkNN-and-Layer-Distance" / "run_step3.py",
    "4": HERE / "Step-4_Consistency-and-Diagnosis" / "run_step4.py",
}


def main(steps, models, datasets):
    passthrough = []
    if models:
        passthrough += ["--models", models]
    if datasets:
        passthrough += ["--datasets", datasets]
    for step in steps:
        runner = STEP_RUNNERS[step]
        print(f"\n================ Phase A | Step {step} ================")
        if subprocess.run([sys.executable, str(runner)] + passthrough).returncode != 0:
            sys.exit(f"[phase-a] Step {step} FAILED")
    print("\nPhase A finished.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--steps", default="1,2,3,4", help="e.g. 3,4")
    parser.add_argument("--models", default=None)
    parser.add_argument("--datasets", default=None)
    args = parser.parse_args()
    main(args.steps.split(","), args.models, args.datasets)
