"""
Write the shipped per-figure data files.

Run once, from a machine with access to the Witten share. Each figure script
reads these from `data/` and then needs no raw data; without them it falls
back to reading the share, which is what happens on the authors' machines.

Run from the repository root:

    python pipeline/prepare_figure_data.py

Existing files are left alone unless OVERWRITE is set.
"""

# %%
import os
import shutil
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ephys_alf_summary import LASER_ONLY_ABB, alf
from model_comparison_accu import load_data_reduced

import figure5_data as f5
import figure6_data as f6

# %%
# ============================================================================
# Configuration
# ============================================================================
REPO = f6.PAPER_DIR
DATA = REPO / "data"
OVERWRITE = False

FIGURE1_ROOT = str(f6.SUBJECTS_ROOT)


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
    print(f"    wrote {path.relative_to(REPO)}  ({scaled:.1f} {unit})")


def skip(path):
    """True if `path` is already there and OVERWRITE is not set."""
    if path.exists() and not OVERWRITE:
        print(f"    exists, skipped {path.relative_to(REPO)}")
        return True
    return False


# %%
# ============================================================================
# Figure 1
# ============================================================================
print("\n#### figure 1")
out = target("figure1", "behaviour.csv.gz")
if not skip(out):
    data = load_data_reduced(ROOT_FOLDER=FIGURE1_ROOT, trial_start=0,
                             trial_end=None, sessions=LASER_ONLY_ABB)
    data.to_csv(out, index=False, compression="gzip")
    report(out)


# %%
# ============================================================================
# Figure 5, and Extended Data Figure 10
# ============================================================================
print("\n#### figure 5")
out = target("figure5", "behaviour.csv.gz")
if not skip(out):
    f5.load_behaviour().to_csv(out, index=False, compression="gzip")
    report(out)

out = target("figure5", "nocs.csv.gz")
out_params = target("figure5", "nocs_params.csv.gz")
if not (skip(out) or skip(out_params)):
    behaviour, parameters = f5.load_nocs_experiment()
    behaviour.to_csv(out, index=False, compression="gzip")
    parameters.to_csv(out_params, index=False, compression="gzip")
    report(out)
    report(out_params)

print("\n#### extended data figure 10")
out = target("extended_data_figure10", "cs_value_posteriors.csv.gz")
if not skip(out):
    posteriors, selected = f5.cs_value_posteriors()
    posteriors.to_csv(out, index=False, compression="gzip")
    report(out)
    print(f"    {len(selected)} sessions: slots {selected}")


# %%
# ============================================================================
# Figure 6, and Extended Data Figure 11
# ============================================================================
print("\n#### figure 6")
out = target("figure6", "mouse_sessions.csv.gz")
if not skip(out):
    f6.load_mouse_sessions().to_csv(out, index=False, compression="gzip")
    report(out)

out = target("figure6", "example_session.csv.gz")
if not skip(out):
    example = alf(str(f6.EXAMPLE_SESSION)).to_df().reset_index(drop=True)
    consumed = np.load(f6.EXAMPLE_SESSION / "alf"
                       / "_ibl_trials.feedbackTypeConsumed.npy")
    if len(consumed) != len(example):
        raise ValueError(
            f"{f6.EXAMPLE_SESSION}: the trials table has {len(example)} rows "
            f"and feedbackTypeConsumed has {len(consumed)}")
    example["feedbackTypeConsumed"] = consumed
    example.to_csv(out, index=False, compression="gzip")
    report(out)

out = target("figure6", "stan_grid.npz")
if not skip(out):
    grid = f6.load_recovery_data()
    np.savez_compressed(out, **{k: np.asarray(v) for k, v in grid.items()})
    report(out)

out = target("figure6", "plefts.npz")
if not skip(out):
    schedules = {}
    for ms_i, alf_path, _ in f6.sessions_from_stan_order():
        schedules[str(ms_i)] = np.load(
            os.path.join(str(alf_path), "alf",
                         "_ibl_trials.probabilityLeft.npy"))
    np.savez_compressed(out, **schedules)
    report(out)
    print(f"    {len(schedules)} block schedules")

out = target("fits", f6.MODEL_NAME, "session_params_median.pkl")
if not skip(out):
    shutil.copy2(f6.SEQAC_PARAMS_PKL, out)
    report(out)


# %%
# ============================================================================
# What this script cannot write
# ============================================================================
print("""
#### still to produce

Three artifacts come from scripts that run every panel on import, so they
cannot be called from here. Run each figure once against the raw data and
append the cell shown, then delete the cell.

  data/extended_data_figure2/dv_correlations.npy
      in extended_data_figure2.py, after the panel b cell:
          np.save(PAPER_DIR / 'data' / 'extended_data_figure2'
                  / 'dv_correlations.npy', corrs)

  data/figure3/encoding_fractions.csv.gz
      in figure3k.py, after load_encoding_fractions has returned:
          fractions.to_csv(ENCODING_CACHE, compression='gzip')

  data/extended_data_figure11/lick_table.csv.gz
      in extended_data_figure11.py, after the panel l table is built:
          lick_table.to_csv(PAPER_DIR / 'data' / 'extended_data_figure11'
                            / 'lick_table.csv.gz', index=False,
                            compression='gzip')

Create the parent directory first in each case.
""")
print("Done.")
