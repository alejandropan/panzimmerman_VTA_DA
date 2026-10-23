"""
Write the shipped data files `prepare_figure_data.py` cannot.

Those live in scripts that draw every panel when imported, so they are
executed here rather than imported, and the artifact is taken out of the
resulting namespace. Nothing in the figure scripts is modified.

Run from the repository root, with access to the Witten share:

    python pipeline/prepare_remaining_data.py

Extended Data Figure 2 and Figure 3k are executed in full, so their panels are
drawn and written to their usual output directories; those are ignored by git.
Extended Data Figure 11 is executed only as far as its function definitions,
so the fifty-thousand-trial simulation it ends with is not run.
"""

# %%
import os
import sys

import matplotlib
matplotlib.use("Agg")

import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)
os.chdir(REPO)

import figure6_data as f6

DATA = f6.PAPER_DIR / "data"
OVERWRITE = False

ED11_SPLIT = "# Load - the cheap part"
FIGURE1_SPLIT = 'fulldf["bchoice"] = 1 * (fulldf["choice"] > 0)'
FIGURE1_TRIAL_COLUMNS = ["ses_id", "choice", "fQL", "fQR", "deltaq",
                         "zdecision_time"]

FAILURES = []


def target(*parts):
    """A destination under `data/`, with its parent created."""
    path = DATA.joinpath(*parts)
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def report(path):
    """Print one line about a file that was just written."""
    size = path.stat().st_size
    unit = "KB" if size < 1024 ** 2 else "MB"
    scaled = size / 1024 if unit == "KB" else size / 1024 ** 2
    print(f"    wrote {path.relative_to(f6.PAPER_DIR)}  ({scaled:.1f} {unit})")


def skip(path):
    """True if `path` is already there and OVERWRITE is not set."""
    if path.exists() and not OVERWRITE:
        print(f"    exists, skipped {path.relative_to(f6.PAPER_DIR)}")
        return True
    return False


def run_script(filename, upto=None):
    """
    Execute a figure script and hand back the names it defined.

    Args:
        filename: Script in the repository root.
        upto: Stop at the first line containing this text, so only the
            definitions above it run. None executes the whole file.

    Returns:
        The namespace the script left behind.
    """
    source = open(os.path.join(REPO, filename), encoding="utf-8").read()
    if upto is not None:
        if upto not in source:
            raise ValueError(f"{filename} has no line containing {upto!r}, so "
                             "the cut-off point cannot be found")
        source = source.split(upto)[0]
    namespace = {"__name__": "__main__",
                 "__file__": os.path.join(REPO, filename)}
    exec(compile(source, filename, "exec"), namespace)
    return namespace


# %%
# ============================================================================
# Extended Data Figure 2, panel b
# ============================================================================
print("\n#### extended data figure 2")
out = target("extended_data_figure2", "dv_correlations.npy")
if not skip(out):
    print("    running extended_data_figure2.py ...")
    try:
        ns = run_script("extended_data_figure2.py")
        if "dv_corrs" not in ns:
            raise KeyError("extended_data_figure2.py defined no `dv_corrs`")
        np.save(out, ns["dv_corrs"])
        report(out)
        print(f"    shape {np.asarray(ns['dv_corrs']).shape}")
    except Exception as error:
        FAILURES.append(("extended_data_figure2", error))
        print(f"    FAILED: {type(error).__name__}: {error}")


# %%
# ============================================================================
# Figure 3k
# ============================================================================
print("\n#### figure 3k")
out = target("figure3", "encoding_fractions.csv.gz")
if not skip(out):
    print("    running figure3k.py ...")
    try:
        ns = run_script("figure3k.py")
        if "fractions" not in ns:
            raise KeyError("figure3k.py defined no `fractions`")
        ns["fractions"].to_csv(out, compression="gzip")
        report(out)
        print(f"    {len(ns['fractions'])} subregions")
    except Exception as error:
        FAILURES.append(("figure3k", error))
        print(f"    FAILED: {type(error).__name__}: {error}")


# %%
# ============================================================================
# Figure 1
# ============================================================================
print("\n#### figure 1")
comparison_out = target("figure1", "model_comparison.csv.gz")
example_out = target("figure1", "example_session.csv.gz")
trials_out = target("figure1", "session_trials.csv.gz")
pending = [out for out in (comparison_out, example_out, trials_out)
           if not skip(out)]
if pending:
    print("    running figure1.py up to panel h ...")
    try:
        ns = run_script("figure1.py", upto=FIGURE1_SPLIT)
        if comparison_out in pending:
            summary = ns["compare_models"](ns["LASER_ONLY_ABB"])
            summary["ses"] = (summary["ses"]
                              .str.removeprefix(ns["ROOT_FOLDER"])
                              .str.removesuffix("\\alf"))
            summary.to_csv(comparison_out, index=False, compression="gzip")
            report(comparison_out)
            print(f"    {len(summary)} rows")
        if example_out in pending:
            session = (ns["ROOT_FOLDER"]
                       + ns["LASER_ONLY_ABB"][ns["EXAMPLE_SESSION"]])
            example = ns["prepare_example_session"](session)
            example.to_csv(example_out, compression="gzip")
            report(example_out)
        if trials_out in pending:
            trials = ns["fulldf"][FIGURE1_TRIAL_COLUMNS]
            trials.to_csv(trials_out, index=False, compression="gzip")
            report(trials_out)
            print(f"    {len(trials)} trials, "
                  f"{trials['ses_id'].nunique()} sessions")
    except Exception as error:
        FAILURES.append(("figure1", error))
        print(f"    FAILED: {type(error).__name__}: {error}")


# %%
# ============================================================================
# Extended Data Figure 11, panel l
# ============================================================================
print("\n#### extended data figure 11")
out = target("extended_data_figure11", "lick_table.csv.gz")
if not skip(out):
    print("    building the panel l table ...")
    try:
        ns = run_script("extended_data_figure11.py", upto=ED11_SPLIT)
        table, audit = ns["build_lick_table"]()
        table.to_csv(out, index=False, compression="gzip")
        report(out)
        print(f"    {table['session'].nunique()} sessions, "
              f"{len(table)} CS+ trials")
    except Exception as error:
        FAILURES.append(("extended_data_figure11", error))
        print(f"    FAILED: {type(error).__name__}: {error}")

print()
if FAILURES:
    print(f"{len(FAILURES)} artifact(s) not written:")
    for name, error in FAILURES:
        print(f"  {name}: {type(error).__name__}: {error}")
    print("\nThe rest were written. Re-run after fixing; anything already "
          "there is skipped.")
else:
    print("Done.")
