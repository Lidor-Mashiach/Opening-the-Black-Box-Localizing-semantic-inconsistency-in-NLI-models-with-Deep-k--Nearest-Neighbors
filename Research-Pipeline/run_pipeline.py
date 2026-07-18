"""Top-level meta-runner - the whole research flow, end to end. NO FLAGS.

EVERY RUN IS AN INDEPENDENT RESEARCH SAMPLE:
* A fresh RANDOM seed is drawn per run (no flag, nothing to remember) and
  pinned into the environment, so training, sampling and every stage of that
  run share it. Reruns therefore differ - which is exactly what makes
  averaging them meaningful.
* Everything trainable is fine-tuned FRESH from the official pretrained
  backbone on every run. (The three large-model MNLI combinations use the
  OFFICIAL published checkpoints as-is and never train.)
* Never regenerated: the raw corpora and the static paraphrase banks -
  read-only inputs, identical in every run.
* Results are ARCHIVED per run: results/runs/run_<STAMP>_seed_<SEED>/ keeps
  that run's final report + plots forever, and runs_summary.csv accumulates
  mean / std / variance across all runs so far. Nothing is ever overwritten.

Guarantees, by design rather than by flags:
* Datasets/ is READ-ONLY: the original SNLI / MNLI / ANLI corpora and the
  static paraphrase banks are never modified by any run of any model.
* Every run-derived artifact (reduced dataset copy, hypothesis/paraphrase
  encodings, per-model paraphrase copy) lives in Runtime-Data/<MODEL>/<DATASET>/
  and is WIPED at the start of that combination's Step-1.
* The paraphrase banks are REQUIRED STATIC INPUTS: if one is missing the
  runner prints exactly how to build it (a one-time act, in
  setup-files/Paraphrase-Generator/) and EXITS. The pipeline itself NEVER generates
  paraphrases, so every run uses the exact same static bank.

Stages:
  0    DATA CHECK   raw parquet missing? -> setup-files/download_datasets.py
  0.5  BANK CHECK   paraphrase bank missing? -> clear message + EXIT
  1-4  PHASE A      per-combination: wipe workspace -> train/filter/encode/eval
                    -> paraphrase inference -> DkNN + distances -> metrics,
                    diagnosis, all plots and the FINAL REPORT.
  5    ARCHIVE      this run's results -> results/runs/<run id>/ + update
                    runs_summary.csv (mean / std / variance across runs)

Run:  python run_pipeline.py
      python run_pipeline.py --models BERT-base --datasets SNLI
      python run_pipeline.py --steps 3,4        (analysis only, CPU-friendly)

SLURM note: a job that is requeued mid-run inherits the SAME pinned NLI_SEED
from its environment, so it resumes the interrupted training instead of
starting a different sample.
"""
import argparse
import os
import random
import subprocess
import sys
from datetime import datetime
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parent
sys.path[:0] = [str(REPO_ROOT), str(REPO_ROOT / "Research-Pipeline")]

from common.config_loader import (dataset_dir, list_dataset_keys,
                                  list_model_keys, load_config,
                                  paraphrase_bank_csv)
from common.logging_utils import banner, log

RAW_SPLITS = ("train", "validation", "test")
DEFAULT_STEPS = ["1", "2", "3", "4"]
STEP4_RESULTS = (HERE / "PhaseA" / "Step-4_Consistency-and-Diagnosis" / "results")

# ----------------------------------------------------------------------
# THE PHASES  --  the whole research, phase by phase.
# To enable Phase B once it is implemented, uncomment its line below.
# Each entry: (name, runner script). Order == order of execution.
# ----------------------------------------------------------------------
PHASES = [
    ("Phase A - diagnosis (train, encode, DkNN, distances, metrics)",
     HERE / "PhaseA" / "run_phase_a.py"),
    # ("Phase B - Credibility-guided targeted fine-tuning + baselines",
    #  HERE / "PhaseB" / "run_phase_b.py"),
]


def resolve_run_seed():
    """A fresh random seed per run - unless one is already pinned.

    Pinned (NLI_SEED already in the environment) happens in exactly two
    cases: a SLURM requeue of an interrupted run (so it resumes the same
    training instead of becoming a different sample), or a deliberate
    reproduction of a past run. Otherwise: a new random sample.
    """
    pinned = os.environ.get("NLI_SEED")
    if pinned is not None:
        return int(pinned), True
    seed = random.SystemRandom().randint(1, 2 ** 31 - 1)
    os.environ["NLI_SEED"] = str(seed)      # inherited by every step subprocess
    return seed, False


def _print_device_banner():
    try:
        from common.gpu import cuda_status
        kind, info = cuda_status()
        if kind == "cuda":
            log("DEVICE", f"cuda - {info}")
        else:
            log("ERROR", f"no CUDA GPU - {info}")
            log("ERROR", "training would crawl on CPU; analysis (steps 3-4) "
                "is fine on CPU.")
    except ImportError:
        log("ERROR", "torch is not installed - run: python main-setup.py (or python setup-files/setup_env.py)")


def _run(cmd, fail_msg):
    if subprocess.run([sys.executable] + cmd).returncode != 0:
        sys.exit(f"[pipeline] {fail_msg}")


def ensure_raw_data(datasets):
    """Stage 0: download any selected dataset whose raw parquet is missing."""
    missing = []
    for key in datasets:
        cfg = load_config(dataset_key=key)
        raw = dataset_dir(cfg) / "raw"
        if not all((raw / f"{split}.parquet").exists() for split in RAW_SPLITS):
            missing.append(key)
    if missing:
        log("DATA", f"raw data missing for: {', '.join(missing)} - downloading")
        _run([str(REPO_ROOT / "Datasets" / "download_datasets.py"),
              "--datasets", ",".join(missing)], "dataset download FAILED")
    else:
        log("DATA", "raw data present for all selected datasets")


def require_paraphrase_banks(datasets):
    """Stage 0.5: the static banks must already exist - never generated here."""
    missing = [d for d in datasets
               if not paraphrase_bank_csv(load_config(dataset_key=d)).exists()]
    if missing:
        log("ERROR", "paraphrase bank(s) missing for: " + ", ".join(missing))
        log("ERROR", "The bank is a one-time, static input (see "
            "setup-files/Paraphrase-Generator/README.md). Build it once on a GPU machine:")
        log("ERROR", f"    python setup-files/Paraphrase-Generator/generate_paraphrases.py "
            f"--datasets {','.join(missing)}")
        log("ERROR", "then rerun this pipeline. Exiting without touching anything.")
        sys.exit(1)
    log("BANK", "static paraphrase bank present for all selected datasets")


def main(steps, models, datasets):
    seed, pinned = resolve_run_seed()
    banner("PIPELINE", f"run starts | seed {seed} | steps {','.join(steps)} | "
           f"{len(models)} model(s) x {len(datasets)} dataset(s)")
    if pinned:
        log("PIPELINE", f"seed {seed} was PINNED in the environment "
            f"(NLI_SEED) - reproducing / resuming that exact run")
    else:
        log("PIPELINE", f"fresh random seed {seed} - this run is an "
            f"independent research sample; trainable combinations fine-tune "
            f"from scratch and the results are archived on their own")
    _print_device_banner()
    passthrough = ["--models", ",".join(models), "--datasets", ",".join(datasets)]

    # Stage 0 / 0.5 guards apply whenever a phase will train or run inference.
    full_phase_a = steps == DEFAULT_STEPS
    if "1" in steps or "2" in steps:
        ensure_raw_data(datasets)          # stage 0
        require_paraphrase_banks(datasets)  # stage 0.5 - exit if missing

    if full_phase_a:
        # full run: drive the research phase by phase through PHASES.
        for name, runner in PHASES:
            banner("PIPELINE", name)
            _run([str(runner)] + passthrough, f"{name} FAILED")
    else:
        # partial run: drill into Phase A's individual steps (analysis, reruns).
        step_runners = {
            "1": HERE / "PhaseA" / "Step-1_Train-Filter-Encode" / "run_step1.py",
            "2": HERE / "PhaseA" / "Step-2_Paraphrase-Inference" / "run_step2.py",
            "3": HERE / "PhaseA" / "Step-3_DkNN-and-Layer-Distance" / "run_step3.py",
            "4": HERE / "PhaseA" / "Step-4_Consistency-and-Diagnosis" / "run_step4.py",
        }
        for step in steps:
            banner("PIPELINE", f"Phase A - Step {step}")
            _run([str(step_runners[step])] + passthrough, f"Step {step} FAILED")

    banner("PIPELINE", "run finished")
    report = STEP4_RESULTS / "final_report.csv"
    if report.exists():
        log("REPORT", f"final report: {report}")
        stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        _run([str(STEP4_RESULTS.parent / "archive_run.py"),
              "--run-id", f"run_{stamp}_seed_{seed}", "--seed", str(seed)],
             "archiving this run FAILED")
    log("PIPELINE", "Phase B is derived from the Credibility results - "
        "see Research-Pipeline/PhaseB/README.md")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--steps", default=",".join(DEFAULT_STEPS), help="e.g. 3,4")
    parser.add_argument("--models", default=None, help="comma-separated subset")
    parser.add_argument("--datasets", default=None, help="comma-separated subset")
    args = parser.parse_args()
    main(args.steps.split(","),
         args.models.split(",") if args.models else list_model_keys(),
         args.datasets.split(",") if args.datasets else list_dataset_keys())
