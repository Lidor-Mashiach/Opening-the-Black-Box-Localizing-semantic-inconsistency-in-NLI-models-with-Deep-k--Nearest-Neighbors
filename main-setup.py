#!/usr/bin/env python
"""
main-setup.py  --  one-shot environment setup for "Opening the Black Box" (NLI-DkNN)
====================================================================================

Runs every one-time setup step in order, with per-step timing and a final
summary. One command replaces the "run setup_env, then download the datasets,
then generate the paraphrase banks by hand" dance.

WHERE THIS LIVES
----------------
This file sits in the project root; every step script lives in setup-files/:

    Opening-The-Black-Box/
    |-- setup-files/
    |   |-- setup_env.py
    |   |-- download_datasets.py
    |   |-- requirements.txt
    |   +-- setup-files/Paraphrase-Generator/
    |       +-- generate_paraphrases.py
    +-- main-setup.py          <-- you are here

Each step script resolves the project root on its own and writes its outputs
to the right place, so they behave exactly as they do when run directly.

THE PIPELINE (in order)
-----------------------
1. environment   - setup_env.py: detect OS+GPU, repair torch, install deps, health-check
2. datasets      - download_datasets.py: raw SNLI/MNLI/ANLI -> Datasets/<ds>/raw/
3. paraphrases   - generate_paraphrases.py: the static paraphrase banks (GPU, one time)

After it finishes, two stages remain (you launch them): the Optuna tuning
sbatch jobs, then the experiment (Research-Pipeline/run_pipeline.py).

USAGE
-----
    conda activate nlu_env
    cd <your-project-dir>
    python main-setup.py

    # skip a step (e.g. banks already built elsewhere):
    python main-setup.py --skip paraphrases
    # run only specific steps:
    python main-setup.py --only environment datasets
    # list the steps:
    python main-setup.py --list

ADDING / MUTING A STEP
----------------------
The pipeline is the STEPS list below. To add a step, append a Step(...) to the
list. To mute one, comment its line out (or use --skip at runtime). Order in
the list == order of execution.
"""

import argparse
import os
import subprocess
import sys
import time
from pathlib import Path

# ----------------------------------------------------------------------
# Paths
# ----------------------------------------------------------------------
_HERE = Path(__file__).resolve().parent          # project root
_SETUP_DIR = _HERE / "setup-files"               # where the step scripts live
PY = sys.executable                              # the active env's interpreter


# ----------------------------------------------------------------------
# Step definition
# ----------------------------------------------------------------------
class Step:
    """One pipeline step: a name, a human label, and a command to run."""

    def __init__(self, name, label, argv):
        self.name = name        # short id used by --skip / --only
        self.label = label      # human-readable description for logs
        self.argv = argv        # command list passed to subprocess

    def run(self):
        """Run the step's command, streaming its output live. Returns rc."""
        return subprocess.call(self.argv)


# ----------------------------------------------------------------------
# THE PIPELINE  --  edit here to add / reorder / mute steps
# ----------------------------------------------------------------------
# Each step runs the relevant script from setup-files/ using the SAME Python
# interpreter that runs this file (sys.executable), so the active conda env is
# always used.
STEPS = [
    Step(
        "environment",
        "Detect OS+GPU, repair the torch build, install deps, health-check",
        [PY, str(_SETUP_DIR / "setup_env.py")],
    ),
    Step(
        "datasets",
        "Download raw SNLI/MNLI/ANLI into Datasets/<ds>/raw/",
        [PY, str(_SETUP_DIR / "download_datasets.py")],
    ),
    Step(
        "paraphrases",
        "Generate the static paraphrase banks (GPU, one time)",
        [PY, str(_SETUP_DIR / "Paraphrase-Generator" / "generate_paraphrases.py")],
    ),

]


# ----------------------------------------------------------------------
# Logging helpers
# ----------------------------------------------------------------------
def log(msg):
    ts = time.strftime("%H:%M:%S")
    print(f"[{ts}] [main-setup] {msg}", flush=True)


def fmt_dur(seconds):
    """Format a duration like 1h 02m 03s / 4m 05s / 6.1s."""
    if seconds >= 3600:
        h = int(seconds // 3600)
        m = int((seconds % 3600) // 60)
        s = int(seconds % 60)
        return f"{h}h {m:02d}m {s:02d}s"
    if seconds >= 60:
        m = int(seconds // 60)
        s = int(seconds % 60)
        return f"{m}m {s:02d}s"
    return f"{seconds:.1f}s"


# ANSI colors, on only when stdout is a real terminal and NO_COLOR is unset.
_USE_COLOR = sys.stdout.isatty() and os.environ.get('NO_COLOR') is None
_ANSI = {
    'STEP':  '\033[1;36m',    # bold cyan  -- headers / totals
    'OK':    '\033[1;32m',    # bold green -- successes
    'WARN':  '\033[1;33m',    # bold yellow
    'ERROR': '\033[1;31m',    # bold red
    'DIM':   '\033[2m',       # dim        -- separators
    'RESET': '\033[0m',
}


def _color(text, key):
    if not _USE_COLOR:
        return text
    return f"{_ANSI.get(key, '')}{text}{_ANSI['RESET']}"


def _enable_windows_ansi():
    """On Windows, turn on virtual-terminal processing so ANSI colors show."""
    if os.name != 'nt':
        return
    try:
        import ctypes
        k = ctypes.windll.kernel32
        k.SetConsoleMode(k.GetStdHandle(-11), 7)   # ENABLE_VIRTUAL_TERMINAL
    except Exception:
        pass


_enable_windows_ansi()


# ----------------------------------------------------------------------
# Pre-condition checks
# ----------------------------------------------------------------------
def check_conda_active():
    """Refuse to run unless a non-base conda env is active - installing into
    base (or no env) is the easiest way to get the wrong interpreter."""
    env = os.environ.get("CONDA_DEFAULT_ENV")
    if not env:
        log(_color("ERROR: no conda environment is active.", 'ERROR'))
        log("       Activate your project env first, e.g.:")
        log("           conda activate nlu_env")
        return False
    if env == "base":
        log(_color("ERROR: the 'base' conda env is active.", 'ERROR'))
        log("       Create/activate a dedicated env first, e.g.:")
        log("           conda activate nlu_env")
        return False
    log(f"conda env active: '{env}'  (python: {sys.executable})")
    return True


def check_layout():
    """Verify we're in the project root and setup-files/ has the step scripts."""
    if not _SETUP_DIR.is_dir():
        log(_color("ERROR: cannot find setup-files/ next to this script.", 'ERROR'))
        log(f"       Expected at: {_SETUP_DIR}")
        return False
    missing = [
        f for f in ("setup_env.py", "download_datasets.py", "requirements.txt",
                    "Paraphrase-Generator/generate_paraphrases.py")
        if not (_SETUP_DIR / f).exists()
    ]
    if missing:
        log(_color(f"ERROR: setup-files/ is missing: {', '.join(missing)}", 'ERROR'))
        return False
    log(f"layout OK: {_SETUP_DIR}")
    return True


def print_next_steps():
    """After setup, point at the two stages the user launches themselves."""
    log(_color("=" * 64, 'STEP'))
    log(_color("NEXT: two stages remain (you launch these)", 'STEP'))
    log(_color("=" * 64, 'STEP'))
    log(_color("  1/2  Optuna tuning (recommended) - best hyper-parameters per combination:", 'WARN'))
    log("         cd tuning && bash run_all_sbatch.sh      (9 GPU sbatch jobs)")
    log("         products land in configs/tuned/ - nothing to copy by hand")
    log(_color("  2/2  The experiment:", 'OK'))
    log("         python Research-Pipeline/run_pipeline.py")
    log("         fresh random seed per run; results archived + averaged")
    log("  Full picture: setup-files/README.md")


# ----------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description="One-shot environment setup (meta-runner)."
    )
    g = parser.add_mutually_exclusive_group()
    g.add_argument("--skip", nargs="+", default=[], metavar="STEP",
                   help="step name(s) to skip")
    g.add_argument("--only", nargs="+", default=[], metavar="STEP",
                   help="run only these step name(s)")
    parser.add_argument("--list", action="store_true",
                        help="list the pipeline steps and exit")
    args = parser.parse_args()

    if args.list:
        print("Pipeline steps (in order):")
        for s in STEPS:
            print(f"  - {s.name:14s} {s.label}")
        return 0

    # ---- resolve which steps to run --------------------------------
    known = {s.name for s in STEPS}
    for nm in (args.skip + args.only):
        if nm not in known:
            log(f"ERROR: unknown step '{nm}'. Known: {', '.join(sorted(known))}")
            return 2

    if args.only:
        selected = [s for s in STEPS if s.name in args.only]
    else:
        selected = [s for s in STEPS if s.name not in args.skip]

    if not selected:
        log("Nothing to run after applying --skip/--only.")
        return 0

    # ---- pre-conditions --------------------------------------------
    log(_color("=" * 64, 'STEP'))
    log(_color("NLI-DkNN setup -- pre-flight checks", 'STEP'))
    log(_color("=" * 64, 'STEP'))
    if not check_conda_active():
        return 1
    if not check_layout():
        return 1

    log("")
    log(f"Will run {len(selected)} step(s): {', '.join(s.name for s in selected)}")
    log("")

    # ---- run the pipeline ------------------------------------------
    overall_start = time.time()
    durations = []   # (name, seconds, rc)

    for i, step in enumerate(selected, 1):
        log(_color("-" * 64, 'DIM'))
        log(_color(f"STEP {i}/{len(selected)} :: {step.name}", 'STEP'))
        log(f"  {step.label}")
        log(_color("-" * 64, 'DIM'))
        start = time.time()
        rc = step.run()
        elapsed = time.time() - start
        durations.append((step.name, elapsed, rc))

        if rc != 0:
            log("")
            log(_color(f"STEP '{step.name}' FAILED (exit code {rc}) after "
                       f"{fmt_dur(elapsed)}.", 'ERROR'))
            log("Aborting -- later steps depend on this one.")
            _print_summary(durations, time.time() - overall_start, aborted=True)
            return rc

        log(_color(f"STEP '{step.name}' done in {fmt_dur(elapsed)}.", 'OK'))
        log("")

    _print_summary(durations, time.time() - overall_start, aborted=False)
    print_next_steps()
    return 0


def _print_summary(durations, total, aborted):
    bar = _color("=" * 64, 'STEP')
    log(bar)
    title = _color("SETUP SUMMARY", 'STEP')
    if aborted:
        title += _color("  (ABORTED)", 'ERROR')
    log(title)
    log(bar)
    for name, secs, rc in durations:
        status = "OK " if rc == 0 else f"FAIL({rc})"
        tag = _color(f"[{status}]", 'OK' if rc == 0 else 'ERROR')
        log(f"  {tag} {name:14s} {fmt_dur(secs)}")
    log(_color("-" * 64, 'DIM'))
    log(_color(f"  TOTAL{'':11s} {fmt_dur(total)}", 'STEP'))
    log(bar)


if __name__ == "__main__":
    sys.exit(main())
