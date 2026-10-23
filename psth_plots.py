"""
Shared drawing code for the example-neuron PSTH panels (Fig. 3b, 3c).

Each panel shows one neuron's firing rate aligned to an event, split into
quintiles of a decision variable. The quintile means and standard errors are
read from a cache written by `pipeline/prepare_example_psths.py`, so the panels
draw without the raw spike data.
"""

# %%
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent if "__file__" in globals() else Path.cwd()

QUINTILE_ALPHAS = {1: 1.0, 2: 0.8, 3: 0.6, 4: 0.4, 5: 0.2}
CACHE_COLUMNS = ["panel", "neuron", "session", "probe", "cluster", "location",
                 "quintile", "n_trials", "time", "mean", "sem"]


def load_example_psths(cache, panel):
    """
    Read the cached PSTHs of one panel.

    Args:
        cache: Path to the `example_psths.csv.gz` of the figure.
        panel: Panel label as written by the cache builder, e.g. `"3b"`.

    Returns:
        One row per neuron, quintile and time bin.
    """
    cache = Path(cache)
    if not cache.exists():
        raise FileNotFoundError(
            f"{cache} is missing. Write it with "
            "`python pipeline/prepare_example_psths.py`, which needs the "
            "Witten share.")
    frame = pd.read_csv(cache)
    missing = [c for c in CACHE_COLUMNS if c not in frame.columns]
    if missing:
        raise ValueError(f"{cache} lacks columns {missing}; "
                         f"found {sorted(frame.columns)}")
    frame = frame.loc[frame["panel"] == panel]
    if frame.empty:
        raise ValueError(f"{cache} has no rows for panel {panel!r}")
    return frame


def plot_quintile_psths(ax, neuron_frame, color):
    """
    Draw the five quintile PSTHs of one neuron, darkest for the highest quintile.

    Args:
        ax: Axes to draw on.
        neuron_frame: Rows of `load_example_psths` for one neuron.
        color: Line and band colour.
    """
    if sorted(neuron_frame["quintile"].unique()) != sorted(QUINTILE_ALPHAS):
        raise ValueError("expected quintiles "
                         f"{sorted(QUINTILE_ALPHAS)}, found "
                         f"{sorted(neuron_frame['quintile'].unique())}")
    for quintile, rows in neuron_frame.groupby("quintile"):
        alpha = QUINTILE_ALPHAS[quintile]
        ax.plot(rows["time"], rows["mean"], color=color, alpha=alpha)
        ax.fill_between(rows["time"], rows["mean"] - rows["sem"],
                        rows["mean"] + rows["sem"], color=color,
                        alpha=alpha / 2)
