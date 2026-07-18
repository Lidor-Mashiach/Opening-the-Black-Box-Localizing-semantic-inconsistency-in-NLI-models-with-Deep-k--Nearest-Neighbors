"""Meta-runner for BART-large x SNLI: clean workspace -> train -> filter -> encode -> evaluate.

Deterministic by design, no flags:
* The combination's Runtime-Data folder and its Step-2/3/4 results from any
  previous run are WIPED first - runs never mix leftovers.
* The original datasets and the paraphrase banks in Datasets/ are never touched.
* Training fine-tunes FRESH from the pretrained backbone on every
  run, with that run's random seed (injected by run_pipeline via
  NLI_SEED), using the Optuna-tuned hyper-parameters when
  configs/tuned/ has them. A job requeued mid-training with the SAME
  pinned seed resumes its rolling checkpoint; any other seed starts
  clean, so runs can never contaminate each other.
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
    clean_combo_workspace(load_config("BART-large", "SNLI"))
    for script in ORDER:
        banner("STEP-1", f"running {script}", "BART-large", "SNLI")
        if subprocess.run([sys.executable, str(HERE / script)]).returncode != 0:
            sys.exit(f"FAILED: {script}")
    banner("STEP-1", "finished", "BART-large", "SNLI")
