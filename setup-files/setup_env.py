"""One-command environment bootstrap - any OS, any GPU, zero manual pip work.

    python setup_env.py              (Windows PowerShell / Linux terminal / cluster)
    python setup_env.py --cpu-only   (machine without an NVIDIA GPU, by choice)

Works INSIDE an existing conda/venv environment - nothing is deleted or
recreated; only what is wrong gets repaired.

What it does, in order:
  1. Detects the machine: OS, Python, NVIDIA GPU(s) and the driver's CUDA
     version (via `nvidia-smi`).
  2. Inspects the INSTALLED torch (fresh subprocess) and plans:
       * NVIDIA GPU + torch missing or CPU-ONLY build -> replace it with a
         CUDA wheel. The wheel index is NOT hardcoded: the live cuXXX
         indexes are DISCOVERED from https://download.pytorch.org/whl/ ,
         filtered to what the driver supports, and PROBED (pip --dry-run)
         newest-first until one actually has a wheel for this exact
         Python/OS. Only then is the old torch uninstalled - a failed
         attempt can never leave the environment without torch.
       * NVIDIA GPU + CUDA torch but cuda unavailable -> driver problem;
         prints the diagnosis (nothing to reinstall).
       * No NVIDIA GPU -> plain torch (CPU) is the correct build.
  3. Installs everything else from requirements.txt.
  4. Verifies with the same loud [DEVICE] verdict the pipeline uses, and
     runs the metric sanity check.

Idempotent - safe to re-run any time. ASCII-only output (PowerShell safe).
"""
import argparse
import json
import os
import platform
import re
import subprocess
import sys
import sysconfig
import urllib.request
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent   # setup-files/ -> repo root
SETUP_DIR = Path(__file__).resolve().parent               # setup-files/ (requirements.txt lives here)
INDEX_ROOT = "https://download.pytorch.org/whl/"
# Used only if the live index page cannot be fetched (offline / proxy):
STATIC_FALLBACK_TAGS = [130, 129, 128, 126, 124, 121, 118]


def _enable_ansi():
    """Colors in PowerShell/cmd need VT mode turned on; Linux is ready as-is."""
    if os.name != "nt":
        return True
    try:
        import ctypes
        kernel32 = ctypes.windll.kernel32
        handle = kernel32.GetStdHandle(-11)
        mode = ctypes.c_uint32()
        if not kernel32.GetConsoleMode(handle, ctypes.byref(mode)):
            return False
        return bool(kernel32.SetConsoleMode(handle, mode.value | 0x0004))
    except Exception:                                    # noqa: BLE001
        return False


_ANSI = _enable_ansi() and sys.stdout.isatty()
_KIND_CODES = {"info": "34", "cmd": "36", "good": "32", "warn": "33", "bad": "31"}


def say(msg, kind="info"):
    """One colored line: blue=info, cyan=command, green=good, yellow=heads-up,
    red=problem. Text stays ASCII (code-page safe); only the color is ANSI."""
    line = f"[setup] {msg}"
    if _ANSI and kind in _KIND_CODES:
        line = f"\033[{_KIND_CODES[kind]}m{line}\033[0m"
    print(line, flush=True)


def run(cmd, check=True):
    say("$ " + " ".join(cmd), "cmd")
    result = subprocess.run(cmd)
    if check and result.returncode != 0:
        sys.exit(f"[setup] command failed ({result.returncode}): {' '.join(cmd)}")
    return result.returncode


def pip(*args, check=True):
    return run([sys.executable, "-m", "pip", *args], check=check)


# --------------------------------------------------------------------------
# Detection
# --------------------------------------------------------------------------
def detect_nvidia():
    """(gpu_found, driver_cuda_str | None, [gpu names])."""
    try:
        probe = subprocess.run(["nvidia-smi"], capture_output=True, text=True,
                               timeout=20)
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return False, None, []
    if probe.returncode != 0:
        return False, None, []
    match = re.search(r"CUDA Version:\s*([0-9]+\.[0-9]+)", probe.stdout)
    names = []
    listing = subprocess.run(["nvidia-smi", "-L"], capture_output=True, text=True)
    for line in listing.stdout.splitlines():
        m = re.match(r"GPU \d+: (.+?) \(", line)
        if m:
            names.append(m.group(1))
    return True, (match.group(1) if match else None), names


def driver_number(driver_cuda_str):
    """'13.2' -> 132, '12.1' -> 121; None stays None (comparable to cuXXX tags)."""
    if not driver_cuda_str:
        return None
    major, minor = driver_cuda_str.split(".")
    return int(major) * 10 + int(minor)


def torch_state():
    """Fresh-subprocess probe of the installed torch; None if not installed."""
    code = ("import torch, json\n"
            "d = {'version': torch.__version__, 'cuda': torch.version.cuda,\n"
            "     'available': torch.cuda.is_available()}\n"
            "if d['available']:\n"
            "    p = torch.cuda.get_device_properties(0)\n"
            "    d['name'] = torch.cuda.get_device_name(0)\n"
            "    d['vram_gb'] = round(p.total_memory / 2**30, 1)\n"
            "print(json.dumps(d))")
    probe = subprocess.run([sys.executable, "-c", code],
                           capture_output=True, text=True)
    if probe.returncode != 0:
        return None
    return json.loads(probe.stdout.strip().splitlines()[-1])


# --------------------------------------------------------------------------
# Live discovery of PyTorch's CUDA wheel indexes (nothing hardcoded)
# --------------------------------------------------------------------------
def extract_cuda_tags(index_html):
    """All cuXXX tags on the index page, as ints, newest first (>= cu110)."""
    tags = {int(t) for t in re.findall(r"cu(\d{3})", index_html)}
    return sorted((t for t in tags if t >= 110), reverse=True)


def discover_cuda_tags():
    try:
        with urllib.request.urlopen(INDEX_ROOT, timeout=20) as response:
            html = response.read().decode("utf-8", "replace")
        tags = extract_cuda_tags(html)
        if tags:
            say(f"live CUDA wheel indexes on download.pytorch.org: "
                f"{', '.join('cu%d' % t for t in tags)}")
            return tags
    except Exception as exc:                                    # noqa: BLE001
        say(f"could not fetch the live index list ({exc}) - using a fallback list")
    return list(STATIC_FALLBACK_TAGS)


def probe_index(tag):
    """True if the cu<tag> index has a torch wheel for THIS Python/OS."""
    result = subprocess.run(
        [sys.executable, "-m", "pip", "install", "--dry-run", "--no-deps",
         "torch", "--index-url", f"{INDEX_ROOT}cu{tag}"],
        capture_output=True, text=True)
    return result.returncode == 0


def pick_cuda_tag(driver_num, tags=None, probe=probe_index):
    """Newest live index the driver supports that actually resolves here."""
    tags = tags if tags is not None else discover_cuda_tags()
    usable = [t for t in tags if driver_num is None or t <= driver_num]
    if not usable:
        say(f"driver supports only CUDA {driver_num/10:.1f} - older than every "
            f"live wheel index; update the NVIDIA driver", "bad")
        return None
    for tag in usable:
        say(f"probing cu{tag} for a wheel matching this Python/OS ...")
        if probe(tag):
            say(f"selected: cu{tag}", "good")
            return tag
        say(f"  cu{tag}: no matching wheel")
    return None


# --------------------------------------------------------------------------
# Import health check - installing is not enough; the DLLs must actually LOAD
# --------------------------------------------------------------------------
CORE_IMPORTS = ["numpy", "pandas", "yaml", "matplotlib", "torch",
                "transformers", "datasets", "optuna"]


def check_imports(modules):
    """[(module, last error line)] for every module that fails to import."""
    failures = []
    for module in modules:
        probe = subprocess.run([sys.executable, "-c", f"import {module}"],
                               capture_output=True, text=True)
        if probe.returncode != 0:
            lines = [l for l in probe.stderr.strip().splitlines() if l.strip()]
            failures.append((module, lines[-1] if lines else "unknown error"))
    return failures


def looks_like_app_control(failures):
    """Windows Application Control / Smart App Control DLL blocking?"""
    blob = " ".join(err for _, err in failures)
    return "Application Control" in blob or "DLL load failed" in blob


def unblock_site_packages():
    """Best-effort: strip Mark-of-the-Web from every native DLL in the env
    (helps SmartScreen-style blocks; a strict WDAC policy needs the manual
    fix printed below)."""
    site = sysconfig.get_paths()["purelib"]
    say(f"attempting automatic Unblock-File on {site} ...", "warn")
    subprocess.run(
        ["powershell", "-NoProfile", "-Command",
         f"Get-ChildItem -Path '{site}' -Recurse -Include *.dll,*.pyd "
         f"| Unblock-File"],
        capture_output=True)


def print_app_control_guidance():
    say("Windows 'Smart App Control' (or a corporate WDAC policy) is blocking "
        "unsigned Python DLLs (pyarrow and friends).", "bad")
    say("Pick ONE fix:", "bad")
    say("  1) Settings > Privacy & security > Windows Security > App & browser "
        "control > Smart App Control settings > OFF", "warn")
    say("     (turning it off is permanent until a Windows reinstall - the "
        "standard choice on development machines; native Python DLLs and "
        "Smart App Control do not coexist)", "warn")
    say("  2) Company-managed PC (WDAC set by IT)? Ask IT to allow this "
        "conda environment's folder.", "warn")
    say("  3) Skip the fight entirely: run this step on the Linux cluster - "
        "same commands, no Application Control there.", "warn")
    say("then rerun:  python setup_env.py", "warn")


# --------------------------------------------------------------------------
# The decision brain (pure - unit-tested)
# --------------------------------------------------------------------------
def plan_torch_action(gpu_found, state, cpu_only):
    """-> (action, reason); action in {keep, install_cuda, install_cpu,
    driver_problem}."""
    if cpu_only or not gpu_found:
        if state is None:
            return ("install_cpu",
                    "no NVIDIA GPU (or --cpu-only): the plain torch build is correct")
        return ("keep", "no NVIDIA GPU (or --cpu-only): existing torch is fine")
    if state is None:
        return ("install_cuda", "NVIDIA GPU present, torch missing")
    if state["cuda"] is None:
        return ("install_cuda",
                "NVIDIA GPU present but the installed torch is CPU-ONLY "
                "(the PyPI default on Windows)")
    if not state["available"]:
        return ("driver_problem",
                f"torch was built for CUDA {state['cuda']} but cuda is not "
                f"available - a driver/visibility problem, not a pip problem: "
                f"check `nvidia-smi`, update the NVIDIA driver, or (on SLURM) "
                f"request --gpus=1")
    return ("keep", "installed torch already CUDA-enabled and working")


# --------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cpu-only", action="store_true",
                        help="skip GPU handling; install the plain CPU build")
    args = parser.parse_args()

    say(f"OS: {platform.system()} {platform.release()} | "
        f"Python {platform.python_version()} | {sys.executable}")

    gpu_found, driver_str, names = (False, None, []) if args.cpu_only \
        else detect_nvidia()
    if gpu_found:
        say(f"NVIDIA GPU detected: {', '.join(names) or 'yes'}"
            + (f" | driver supports CUDA {driver_str}" if driver_str else ""), "good")
    else:
        say("no NVIDIA GPU detected (nvidia-smi not found / no devices)"
            + (" [--cpu-only]" if args.cpu_only else ""), "warn")

    state = torch_state()
    say("installed torch: " + (f"{state['version']} (cuda build: {state['cuda']}, "
                               f"available: {state['available']})"
                               if state else "none"))

    action, reason = plan_torch_action(gpu_found, state, args.cpu_only)
    say(f"torch plan: {action} - {reason}",
        "good" if action == "keep" else "warn")

    if action == "install_cuda":
        tag = pick_cuda_tag(driver_number(driver_str))   # probes BEFORE touching anything
        if tag is None:
            say("ERROR: no live CUDA index has a torch wheel for this exact "
                "Python/OS/driver combination.", "bad")
            if state is None:
                say("restoring a plain torch so the environment is not left broken ...")
                pip("install", "torch")
            sys.exit("[setup] could not obtain a CUDA torch build - see the "
                     "messages above (driver too old? proxy blocking "
                     "download.pytorch.org? brand-new Python version?)")
        if state is not None:                       # nothing to remove on a bare env
            pip("uninstall", "-y", "torch", check=False)
        pip("install", "torch", "--index-url", f"{INDEX_ROOT}cu{tag}")
    elif action == "install_cpu":
        pip("install", "torch")

    say("installing the rest of requirements.txt ...")
    pip("install", "-r", str(SETUP_DIR / "requirements.txt"))

    say("verifying that every core library actually LOADS (not just installed) ...")
    failures = check_imports(CORE_IMPORTS)
    if failures and os.name == "nt" and looks_like_app_control(failures):
        say("Windows Application Control blocked native DLLs on import.", "bad")
        unblock_site_packages()
        failures = check_imports([module for module, _ in failures])
    if failures:
        for module, err in failures:
            say(f"import {module} FAILED: {err}", "bad")
        if os.name == "nt" and looks_like_app_control(failures):
            print_app_control_guidance()
        sys.exit("[setup] environment is NOT ready - fix the import failures "
                 "above and rerun python setup_env.py")
    say("all core imports load cleanly.", "good")

    final = torch_state()
    if final and final["available"]:
        say(f"[DEVICE] cuda - {final['name']} ({final['vram_gb']} GB, "
            f"CUDA {final['cuda']})", "good")
    elif final and gpu_found:
        say("[ERROR] GPU is present but torch still cannot use it - "
            + plan_torch_action(True, final, False)[1], "bad")
    else:
        say("[DEVICE] cpu - fine for analysis (steps 3-4); training and the "
            "paraphrase generator want a GPU", "warn")

    say("running the metric sanity check ...")
    run([sys.executable, str(REPO_ROOT / "eval_metrics" / "sanity_check.py")])
    say("environment ready.", "good")


if __name__ == "__main__":
    main()
