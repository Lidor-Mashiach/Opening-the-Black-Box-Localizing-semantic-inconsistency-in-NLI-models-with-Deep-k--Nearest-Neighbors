"""Meta-runner for BART-large x MNLI: clean workspace -> train -> filter -> encode -> evaluate.

Deterministic by design, no flags:
* The combination's Runtime-Data folder and its Step-2/3/4 results from any
  previous run are WIPED first - runs never mix leftovers.
* The original datasets and the paraphrase banks in Datasets/ are never touched.
* This combination never trains: the OFFICIAL published checkpoint is
  used as-is (declared in configs/models/), so stages 2-4 run
  against it and its results carry zero seed-variance.
"""
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[4]
sys.path[:0] = [str(REPO_ROOT), str(REPO_ROOT / "Research-Pipeline")]

from common.config_loader import load_config
from common.logging_utils import banner
from common.workspace import clean_combo_workspace

HERE = Path(__file__).resolve().parent
ORDER = ["train.py", "build_filtered_dataset.py", "encode_hypotheses.py", "evaluate.py"]

if __name__ == "__main__":
    clean_combo_workspace(load_config("BART-large", "MNLI"))
    for script in ORDER:
        banner("STEP-1", f"running {script}", "BART-large", "MNLI")
        if subprocess.run([sys.executable, str(HERE / script)]).returncode != 0:
            sys.exit(f"FAILED: {script}")
    banner("STEP-1", "finished", "BART-large", "MNLI")
