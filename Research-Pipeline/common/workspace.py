"""Runtime-Data lifecycle - the ONLY place run-derived data lives and dies.

Datasets/ is read-only (raw corpora + the static paraphrase banks). Every
run-derived artifact of a combination - the reduced dataset copy, the
hypothesis/paraphrase encodings and the per-model paraphrase copy - lives in
Runtime-Data/<MODEL>/<DATASET>/.

clean_combo_workspace() is called at the START of a combination's Step-1 run:
it deletes the combination's Runtime-Data folder AND its Step-2/3/4 result
folders from any previous run, so runs never mix leftovers. The Optuna studies
and tuned configs are preserved; checkpoints are governed by training's
seed guard - every run (fresh random seed) retrains, while a resubmitted
run with the SAME pinned NLI_SEED resumes its rolling checkpoint.
"""
import shutil

from .config_loader import combo_runtime_dir, step_results_dir
from .logging_utils import log

_DOWNSTREAM_STEP_DIRS = (
    "Step-2_Paraphrase-Inference",
    "Step-3_DkNN-and-Layer-Distance",
    "Step-4_Consistency-and-Diagnosis",
)


def clean_combo_workspace(cfg):
    """Fresh start for one combination: wipe derived data + stale results."""
    targets = [combo_runtime_dir(cfg)]
    targets += [step_results_dir(d, cfg) for d in _DOWNSTREAM_STEP_DIRS]
    removed = 0
    for target in targets:
        if target.exists():
            shutil.rmtree(target)
            removed += 1
    combo_runtime_dir(cfg).mkdir(parents=True, exist_ok=True)
    log("CLEAN", f"workspace reset ({removed} leftover folder(s) removed) -> "
        f"{combo_runtime_dir(cfg)}",
        cfg.get("model_key"), cfg.get("dataset_key"))
