"""Step-4e: archive THIS run, then refresh the cross-run statistics.

Called automatically at the end of every run_pipeline.py run - no flags, no
bookkeeping to remember.

1. ARCHIVE: results/final_report.csv + every plot of this run are copied to
   results/runs/<run id>/ (run id = run_<timestamp>_seed_<seed>). Runs are
   never overwritten and never mixed - each is a complete, self-contained
   record you can keep, share or delete.
2. SUMMARY: every archived run is re-read and results/runs_summary.csv is
   rewritten with, per model x dataset x metric:
       <metric>_mean, <metric>_std, <metric>_var
   plus n_runs, the seed list, and - for text columns (scenario / verdict /
   region) - the set of values observed across runs. `signals_agree_mean` is
   the agreement RATE across runs.

Run automatically; standalone use:
    python archive_run.py --run-id run_20260101-120000_seed_1234 --seed 1234
    python archive_run.py --summary-only      (rebuild the summary alone)
"""
import argparse
import json
import re
import shutil
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[2]
sys.path[:0] = [str(REPO_ROOT), str(REPO_ROOT / "Research-Pipeline")]

import pandas as pd

from common.logging_utils import log

RESULTS = HERE / "results"
RUNS_DIR = RESULTS / "runs"
ID_COLS = ("model", "dataset", "seed")


def archive(run_id, seed):
    """Copy this run's report + plots into its own folder."""
    target = RUNS_DIR / run_id
    target.mkdir(parents=True, exist_ok=True)
    report = RESULTS / "final_report.csv"
    shutil.copy2(report, target / "final_report.csv")
    copied = 1
    for png in RESULTS.glob("*.png"):                       # overview figures
        shutil.copy2(png, target / png.name)
        copied += 1
    for combo_dir in RESULTS.iterdir():                     # per-combination
        if combo_dir.is_dir() and "__" in combo_dir.name:
            shutil.copytree(combo_dir, target / combo_dir.name,
                            dirs_exist_ok=True)
            copied += 1
    (target / "run_info.json").write_text(json.dumps(
        {"run_id": run_id, "seed": seed}, indent=2))
    log("REPORT", f"run archived ({copied} artifact group(s)) -> {target}")
    return target


def load_runs():
    frames = []
    for run_dir in sorted(RUNS_DIR.glob("run_*")):
        report = run_dir / "final_report.csv"
        if not report.exists():
            continue
        df = pd.read_csv(report)
        match = re.search(r"_seed_(\d+)$", run_dir.name)
        df["seed"] = int(match.group(1)) if match else -1
        df["run_id"] = run_dir.name
        frames.append(df)
    return pd.concat(frames, ignore_index=True) if frames else None


def summarize(all_runs):
    numeric = [c for c in all_runs.columns
               if c not in ID_COLS + ("run_id",)
               and pd.api.types.is_numeric_dtype(all_runs[c])]
    text = [c for c in all_runs.columns
            if c not in ID_COLS + ("run_id",) and c not in numeric]
    grouped = all_runs.groupby(["model", "dataset"])
    out = grouped[numeric].agg(["mean", "std", "var"])
    out.columns = [f"{col}_{stat}" for col, stat in out.columns]
    for col in text:
        out[f"{col}_values"] = grouped[col].agg(
            lambda s: " | ".join(sorted(set(map(str, s)))))
    out["n_runs"] = grouped["seed"].nunique()
    out["seeds"] = grouped["seed"].agg(lambda s: ",".join(map(str, sorted(set(s)))))
    return out.reset_index()


def refresh_summary():
    all_runs = load_runs()
    if all_runs is None:
        log("REPORT", f"no archived runs under {RUNS_DIR} yet")
        return
    summary = summarize(all_runs)
    out_csv = RESULTS / "runs_summary.csv"
    summary.to_csv(out_csv, index=False)
    n = int(summary["n_runs"].max())
    log("REPORT", f"cross-run statistics over {n} run(s) "
        f"(mean / std / var) -> {out_csv}")
    if n == 1:
        log("PIPELINE", "one run so far - std/var become meaningful from the "
            "second run onwards; every run is archived separately under "
            "results/runs/")
    print(summary.to_string(index=False, float_format=lambda v: f"{v:.4f}"))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--summary-only", action="store_true")
    args = parser.parse_args()
    if not args.summary_only:
        if not args.run_id:
            sys.exit("[archive] --run-id is required (or use --summary-only)")
        archive(args.run_id, args.seed)
    refresh_summary()


if __name__ == "__main__":
    main()
