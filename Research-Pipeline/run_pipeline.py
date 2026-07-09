"""Top-level meta-runner: the whole research flow, end to end.

Stage 0  DATA CHECK        raw parquet missing for a selected dataset?
                           -> Datasets/download_datasets.py fetches it.
Stage 1  PHASE A / STEP-1  train -> filter -> encode -> evaluate per combo.
                           Resumable: finished stages skip themselves; MNLI
                           combos with a published official checkpoint skip
                           training automatically (configs/models/*.yaml).
Stage P  PARAPHRASE CHECK  <MODEL>__paraphrases.csv missing for a selected
                           combo? -> Datasets/generate_paraphrases.py creates
                           + verifies it. Disable with --static-paraphrases
                           to treat the CSVs as fixed external inputs.
Stage 2  PHASE A / STEP-2  paraphrase inference (predictions + encodings).
Stage 3  PHASE A / STEP-3  K selection, DkNN Credibility, Layer Distance.
Stage 4  PHASE A / STEP-4  RFR/SFR/PC/PA, joint diagnosis, all plots.

Run:  python run_pipeline.py                                  # everything
      python run_pipeline.py --models BERT-base --datasets SNLI
      python run_pipeline.py --steps 3,4                      # analysis only
      python run_pipeline.py --force                          # redo after Optuna
      python run_pipeline.py --static-paraphrases             # never generate

Device: every training / inference stage auto-detects CUDA
(torch.cuda.is_available()) and runs on the GPU when present, CPU otherwise.
"""
import argparse
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parent
sys.path[:0] = [str(REPO_ROOT), str(REPO_ROOT / "Research-Pipeline")]

from common.config_loader import (dataset_dir, list_dataset_keys,
                                  list_model_keys, load_config,
                                  paraphrases_csv)

RAW_SPLITS = ("train", "validation", "test")


def _print_device_banner():
    try:
        import torch
        if torch.cuda.is_available():
            print(f"[pipeline] device: cuda ({torch.cuda.get_device_name(0)})")
        else:
            print("[pipeline] device: cpu - no CUDA GPU detected. "
                  "Training will be very slow; analysis (steps 3-4) is fine on CPU.")
    except ImportError:
        print("[pipeline] torch is not installed - run: pip install -r requirements.txt")


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
        print(f"[pipeline] raw data missing for: {', '.join(missing)} - downloading")
        _run([str(REPO_ROOT / "Datasets" / "download_datasets.py"),
              "--datasets", ",".join(missing)], "dataset download FAILED")
    else:
        print("[pipeline] stage 0: raw data present for all selected datasets")


def ensure_paraphrases(models, datasets, force):
    """Stage P: generate+verify paraphrase CSVs that are missing (or all, on --force)."""
    todo = []
    for d in datasets:
        for m in models:
            cfg = load_config(m, d)
            if force or not paraphrases_csv(cfg).exists():
                todo.append((m, d))
    if not todo:
        print("[pipeline] stage P: paraphrase files present for all selected combinations")
        return
    todo_models = ",".join(sorted({m for m, _ in todo}))
    todo_datasets = ",".join(sorted({d for _, d in todo}))
    print(f"[pipeline] generating paraphrases for: {todo_models} x {todo_datasets}")
    _run([str(REPO_ROOT / "Datasets" / "generate_paraphrases.py"),
          "--models", todo_models, "--datasets", todo_datasets],
         "paraphrase generation FAILED")


def main(steps, models, datasets, force, static_paraphrases):
    _print_device_banner()
    passthrough = ["--models", ",".join(models), "--datasets", ",".join(datasets)]
    if force:
        passthrough.append("--force")
    runners = {
        "1": HERE / "PhaseA" / "Step-1_Train-Filter-Encode" / "run_step1.py",
        "2": HERE / "PhaseA" / "Step-2_Paraphrase-Inference" / "run_step2.py",
        "3": HERE / "PhaseA" / "Step-3_DkNN-and-Layer-Distance" / "run_step3.py",
        "4": HERE / "PhaseA" / "Step-4_Consistency-and-Diagnosis" / "run_step4.py",
    }

    if "1" in steps:
        ensure_raw_data(datasets)                       # stage 0
    for step in steps:
        if step == "2" and not static_paraphrases:
            ensure_paraphrases(models, datasets, force)  # stage P (needs Step-1 output)
        print(f"\n================ Pipeline | Phase A Step {step} ================")
        _run([str(runners[step])] + passthrough, f"Step {step} FAILED")

    print("\n[pipeline] Phase A done. Phase B is derived from the Credibility "
          "results - see Research-Pipeline/PhaseB/README.md")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--steps", default="1,2,3,4", help="e.g. 3,4")
    parser.add_argument("--models", default=None, help="comma-separated subset")
    parser.add_argument("--datasets", default=None, help="comma-separated subset")
    parser.add_argument("--force", action="store_true",
                        help="redo finished stages (retrain -> refilter -> re-encode -> "
                             "regenerate paraphrases -> re-infer). Use after Optuna tuning.")
    parser.add_argument("--static-paraphrases", action="store_true",
                        help="never auto-generate paraphrase CSVs - treat them as "
                             "fixed external inputs")
    args = parser.parse_args()
    main(args.steps.split(","),
         args.models.split(",") if args.models else list_model_keys(),
         args.datasets.split(",") if args.datasets else list_dataset_keys(),
         args.force, args.static_paraphrases)
