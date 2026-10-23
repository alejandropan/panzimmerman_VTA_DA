"""
Draw every panel of the paper.

Runs each figure script in its own process, from the repository root, and prints
one line per script: its status, how long it took, and how many PDFs it wrote.
The console output of each script goes to `logs/<script>.log`. A script that
fails does not stop the rest.

`figure2a.py` runs last. It is interactive and needs the IBL environment, so it
is opened in an IPython session of the environment named `IBL_ENV`, which must
sit beside the current one in conda's `envs` folder. Rotate the scene to the
published view, run `mlab.savefig(str(SAVE_PATH), size=SAVE_SIZE)`, then
`exit()`. It is skipped, with a note, when that environment is not found.

Left out: everything in `pipeline/`, which needs the raw recordings.

Usage (interactive), or:
    python make_all_figures.py

To draw a subset, edit `RUN`.
"""

# %%
import os
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent if "__file__" in globals() else Path.cwd()

# %%
# ============================================================================
# Scripts
# ============================================================================
SCRIPTS = [
    "figure1.py",
    "figure2.py",
    "figure3.py",
    "figure3k.py",
    "figure5.py",
    "figure6.py",
    "figure6_decoding.py",
    "extended_data_figure2.py",
    "extended_data_figure4.py",
    "extended_data_figure5.py",
    "extended_data_figure6.py",
    "extended_data_figure7.py",
    "extended_data_figure10.py",
    "extended_data_figure12.py",
    "extended_data_figure11.py",
]
RUN = list(SCRIPTS)

IBL_ENV = "panzimmerman2026-ibl"
INTERACTIVE = ["figure2a.py"]

LOG_DIR = HERE / "logs"
TAIL_LINES = 6


# %%
# ============================================================================
# Functions
# ============================================================================
def ibl_python():
    """
    The interpreter of the IBL environment, or None if it is not installed.

    Returns:
        Path of the python executable in the environment named `IBL_ENV`,
        looked up beside the current environment.
    """
    interpreter = "python.exe" if os.name == "nt" else "bin/python"
    candidate = Path(sys.prefix).parent / IBL_ENV / interpreter
    return candidate if candidate.exists() else None


# %%
# ============================================================================
# Run
# ============================================================================
missing = [name for name in RUN + INTERACTIVE if not (HERE / name).exists()]
if missing:
    raise FileNotFoundError(f"no such script in {HERE}: {missing}")

LOG_DIR.mkdir(exist_ok=True)
environment = {**os.environ, "MPLBACKEND": "Agg", "PYTHONIOENCODING": "utf-8"}

print(f"{'script':<30} {'status':<8} {'seconds':>8} {'PDFs':>5}")
failed = []
for name in RUN:
    log = LOG_DIR / f"{Path(name).stem}.log"
    start = time.time()
    with open(log, "w") as handle:
        completed = subprocess.run(
            [sys.executable, str(HERE / name)], cwd=HERE, env=environment,
            stdout=handle, stderr=subprocess.STDOUT)
    elapsed = time.time() - start
    written = [p for p in HERE.glob("fig*.pdf") if p.stat().st_mtime >= start]
    status = "OK" if completed.returncode == 0 else "FAILED"
    print(f"{name:<30} {status:<8} {elapsed:>8.0f} {len(written):>5}")
    if completed.returncode != 0:
        failed.append(name)
        lines = log.read_text(encoding="utf-8", errors="replace").splitlines()
        for line in lines[-TAIL_LINES:]:
            print("    " + line.encode("ascii", "replace").decode())

print(f"\n{len(RUN) - len(failed)} of {len(RUN)} scripts finished; logs in {LOG_DIR}")

for name in INTERACTIVE:
    python = ibl_python()
    if python is None:
        print(f"\n{name} skipped: no environment {IBL_ENV} beside {sys.prefix}; "
              "see the README")
        continue
    print(f"\nOpening {name} in {IBL_ENV}. Rotate the scene, then run "
          "mlab.savefig(str(SAVE_PATH), size=SAVE_SIZE) and exit().")
    subprocess.run([str(python), "-m", "IPython", "--gui=qt", "-i",
                    str(HERE / name)], cwd=HERE)

if failed:
    raise SystemExit(f"failed: {', '.join(failed)}")
