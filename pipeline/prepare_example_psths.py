"""
Write the cached PSTHs behind the example-neuron panels (Fig. 3b, 3c).

Needs the Witten share and an environment with iblutil. Run from the repository
root:

    python pipeline/prepare_example_psths.py

Writes `data/figure3/example_psths.csv.gz`: one row per neuron, decision-variable quintile and time bin, holding the mean
firing rate and its standard error across trials. Quintile 1 is the highest.
"""

# %%
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "pipeline"))

from ephys_alf_summary import alf                                   # noqa: E402
from psth_functions import calculate_causal_peths                   # noqa: E402

# %%
# ============================================================================
# Configuration
# ============================================================================
ON_CLUSTER = os.name != "nt"
WITTEN = Path("/jukebox/witten") if ON_CLUSTER else Path(r"\\cup.princeton.edu\witten")
SUBJECTS = WITTEN / "Alex" / "Data" / "Subjects"

OVERWRITE = False

SESSION_A = "dop_53/2022-10-04/001"
SESSION_B = "dop_56/2023-09-30/001"

SETTINGS = {
    "3b": dict(engine=calculate_causal_peths, align="goCue_trigger_times",
               pre_time=0.2, post_time=0.5, bin_size=0.01, smoothing=0.05),
    "3c": dict(engine=calculate_causal_peths, align="response_times",
               pre_time=0.2, post_time=0.5, bin_size=0.01, smoothing=0.05),
}

# `variable` names a key of VARIABLES; trials are those where `field == value`.
NEURONS = [
    dict(panel="3b", session=SESSION_B, probe=1, cluster=918,
         variable="relative", field="choice", value=1),
    dict(panel="3c", session=SESSION_A, probe=0, cluster=1038,
         variable="state", field="outcome", value=1),
]

VARIABLES = {
    "QL": lambda d: d.fQL,
    "QR": lambda d: d.fQR,
    "relative": lambda d: d.fQR - d.fQL,
    "state": lambda d: d.V,
}

OUTPUT = {
    "3b": REPO / "data" / "figure3" / "example_psths.csv.gz",
    "3c": REPO / "data" / "figure3" / "example_psths.csv.gz",
}


# %%
# ============================================================================
# Functions
# ============================================================================
def quintile_masks(values, trials):
    """
    Split the selected trials into five equal-count groups, highest first.

    Args:
        values: Decision variable, one per trial.
        trials: Boolean mask of the trials to split.

    Returns:
        Five boolean masks, quintile 1 holding the highest values.
    """
    edges = np.quantile(values[trials], [0.0, 0.2, 0.4, 0.6, 0.8])
    masks = [trials & (values >= edges[4])]
    for low, high in ((3, 4), (2, 3), (1, 2), (0, 1)):
        masks.append(trials & (values >= edges[low]) & (values < edges[high]))
    return masks


def neuron_rows(data, spec):
    """
    The quintile PSTHs of one neuron as long-format rows.

    Args:
        data: The session's `alf` object.
        spec: One entry of `NEURONS`.

    Returns:
        A list of row dicts, one per quintile and time bin.
    """
    settings = SETTINGS[spec["panel"]]
    probe = data.probe[spec["probe"]]
    values = np.asarray(VARIABLES[spec["variable"]](data))
    trials = np.asarray(getattr(data, spec["field"])) == spec["value"]
    align_times = np.asarray(getattr(data, settings["align"]))
    if not len(values) == len(trials) == len(align_times):
        raise ValueError(
            f"{spec['session']}: {len(values)} values, {len(trials)} trial "
            f"flags and {len(align_times)} alignment times differ in length")
    location = str(probe.cluster_locations[spec["cluster"]])

    rows = []
    for quintile, mask in enumerate(quintile_masks(values, trials), start=1):
        if not mask.any():
            raise ValueError(f"cluster {spec['cluster']}: quintile {quintile} "
                             "holds no trials")
        peths, binned = settings["engine"](
            probe.spike_times, probe.spike_clusters,
            np.array([spec["cluster"]]), align_times[mask],
            pre_time=settings["pre_time"], post_time=settings["post_time"],
            bin_size=settings["bin_size"], smoothing=settings["smoothing"],
            return_fr=True)
        sem = peths.stds[0] / np.sqrt(len(binned))
        for time, mean, error in zip(peths.tscale, peths.means[0], sem):
            rows.append(dict(
                panel=spec["panel"], neuron=f"#{spec['cluster']}",
                session=spec["session"], probe=spec["probe"],
                cluster=spec["cluster"], location=location,
                quintile=quintile, n_trials=len(binned),
                time=time, mean=mean, sem=error))
    return rows


# %%
# ============================================================================
# Run
# ============================================================================
sessions = {}
rows = []
for spec in NEURONS:
    if spec["session"] not in sessions:
        sessions[spec["session"]] = alf(str(SUBJECTS / spec["session"]),
                                        ephys=True)
    print(f"{spec['panel']}  {spec['session']}  probe {spec['probe']}  "
          f"cluster {spec['cluster']}")
    new = neuron_rows(sessions[spec["session"]], spec)
    rows.extend(new)
    frame = pd.DataFrame(new)
    print(f"    {frame['location'].iloc[0]}, "
          f"peak mean rate {frame['mean'].max():.1f} Hz")

frame = pd.DataFrame(rows)
for path in sorted(set(OUTPUT.values())):
    panels = [p for p, target in OUTPUT.items() if target == path]
    if path.exists() and not OVERWRITE:
        print(f"{path} exists; set OVERWRITE to replace it")
        continue
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.loc[frame["panel"].isin(panels)].to_csv(path, index=False)
    print(f"wrote {path}")
