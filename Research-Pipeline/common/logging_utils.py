"""Colored, structured run logs - know at a glance WHICH model, WHICH dataset,
WHICH step and WHAT is happening right now.

Every subject gets its own color:
    PIPELINE / DATA / BANK   blue      orchestration & sources
    STEP-1 (TRAIN/FILTER/
            ENCODE/EVAL)     cyan      training & data building
    STEP-2                   magenta   paraphrase inference
    STEP-3                   yellow    DkNN + layer distances
    STEP-4 / REPORT          green     metrics, diagnosis, final report
    CLEAN / SKIP / ERROR     red       workspace wipes & problems

The MODEL x DATASET pair is always printed in bold so combinations are easy
to tell apart. Colors auto-disable when stdout is not a terminal (e.g. SLURM
.out files); set FORCE_COLOR=1 to keep them anyway.
"""
import os
import sys

_CODES = {"blue": "34", "cyan": "36", "magenta": "35", "yellow": "33",
          "green": "32", "red": "31", "bold": "1"}

TAG_COLORS = {
    "PIPELINE": "blue", "DATA": "blue", "BANK": "blue", "DEVICE": "green",
    "STEP-1": "cyan", "TRAIN": "cyan", "FILTER": "cyan",
    "ENCODE": "cyan", "EVAL": "cyan",
    "STEP-2": "magenta", "INFER": "magenta",
    "STEP-3": "yellow", "SELECT-K": "yellow", "CRED": "yellow", "DIST": "yellow",
    "STEP-4": "green", "METRICS": "green", "DIAGNOSE": "green", "REPORT": "green",
    "WARN": "yellow",
    "CLEAN": "red", "SKIP": "red", "ERROR": "red",
}


def _enable_windows_ansi():
    """Turn on VT processing so ANSI colors render in PowerShell / cmd."""
    if os.name != "nt":
        return True
    try:
        import ctypes
        kernel32 = ctypes.windll.kernel32
        handle = kernel32.GetStdHandle(-11)              # STD_OUTPUT_HANDLE
        mode = ctypes.c_uint32()
        if not kernel32.GetConsoleMode(handle, ctypes.byref(mode)):
            return False
        return bool(kernel32.SetConsoleMode(handle, mode.value | 0x0004))
    except Exception:                                    # noqa: BLE001
        return False


_ANSI_READY = _enable_windows_ansi()


def _enabled():
    if os.environ.get("FORCE_COLOR") == "1":
        return True
    return sys.stdout.isatty() and _ANSI_READY


def _paint(text, color):
    if not _enabled() or color not in _CODES:
        return text
    return f"\033[{_CODES[color]}m{text}\033[0m"


def _combo(model, dataset):
    if model and dataset:
        return " " + _paint(f"{model} x {dataset}", "bold")
    if dataset:
        return " " + _paint(str(dataset), "bold")
    return ""


def log(tag, message, model=None, dataset=None):
    """One structured line: [TAG] MODEL x DATASET | message."""
    colored_tag = _paint(f"[{tag}]", TAG_COLORS.get(tag, "blue"))
    print(f"{colored_tag}{_combo(model, dataset)} | {message}", flush=True)


def banner(tag, title, model=None, dataset=None):
    """A visually loud section separator."""
    line = _paint("=" * 72, TAG_COLORS.get(tag, "blue"))
    print(f"\n{line}", flush=True)
    log(tag, title, model, dataset)
    print(line, flush=True)
