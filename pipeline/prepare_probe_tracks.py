"""
Write the probe-track coordinates behind Fig. 2a.

Needs the Witten share and an environment with ibllib, whose Allen atlas
converts each recorded track into CCF micrometres. Run from the repository root:

    python pipeline/prepare_probe_tracks.py

Writes `data/figure2/probe_tracks.csv`: for every striatal, cortical or pallidal
probe, the two end points of its track (`segment == "track"`) and of the
extension that continues it to the surface (`segment == "extension"`).
"""

# %%
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

import ibllib.atlas as atlas                                        # noqa: E402
from ephys_alf_summary import LASER_ONLY_ABB, alf                   # noqa: E402

# %%
# ============================================================================
# Configuration
# ============================================================================
ON_CLUSTER = os.name != "nt"
WITTEN = Path("/jukebox/witten") if ON_CLUSTER else Path(r"\\cup.princeton.edu\witten")
SUBJECTS = str(WITTEN / "Alex" / "Data" / "Subjects")

OUTPUT = REPO / "data" / "figure2" / "probe_tracks.csv"
OVERWRITE = False

REGIONS = ['DLS', 'DMS', 'TS', 'NAc', 'SS', 'MO', 'PFC', 'OFC', 'VP', 'GPe']
LAST_CHANNEL = 'channel_383'
FIRST_CHANNEL = 'channel_0'
MICROMETRES = 1_000_000


# %%
# ============================================================================
# Functions
# ============================================================================
def point_on_line(start_point, end_point, distance_from_end):
    """
    The point a given distance back from `end_point` along the line to it.

    Args:
        start_point: (x, y, z) of the start of the line.
        end_point: (x, y, z) of the end of the line.
        distance_from_end: Distance back from `end_point`.

    Returns:
        The (x, y, z) of the point.
    """
    start_point = np.array(start_point)
    end_point = np.array(end_point)
    direction = end_point - start_point
    return end_point - distance_from_end * direction / np.linalg.norm(direction)


def probe_segments(channel_locations, brain_atlas):
    """
    The track and its surface extension, in CCF micrometres.

    Args:
        channel_locations: Parsed `channel_locations.json` of one probe.
        brain_atlas: An `ibllib.atlas.AllenAtlas`.

    Returns:
        `{"track": (2, 3), "extension": (2, 3)}`, columns ml, ap, dv.
    """
    def location(channel):
        entry = channel_locations[channel]
        return [-abs(entry['x']), entry['y'], entry['z']]

    short = np.array([location(LAST_CHANNEL), location(FIRST_CHANNEL)]) / MICROMETRES
    extension_start = point_on_line(short[0], short[1], short[1][-1] * -1)
    extension = np.array([extension_start, short[1]])
    return {"track": brain_atlas.xyz2ccf(short),
            "extension": brain_atlas.xyz2ccf(extension)}


# %%
# ============================================================================
# Run
# ============================================================================
if OUTPUT.exists() and not OVERWRITE:
    raise FileExistsError(f"{OUTPUT} exists; set OVERWRITE to replace it")

brain_atlas = atlas.AllenAtlas(25)

rows = []
for session in LASER_ONLY_ABB:
    data = alf(SUBJECTS + session, ephys=True)
    for index, probe in enumerate(data.probe):
        selected = probe.cluster_group_locations[probe.cluster_selection].unique()
        if not np.isin(selected, REGIONS).any():
            continue
        with open(Path(probe.path) / 'channel_locations.json', 'r') as handle:
            channel_locations = json.load(handle)
        for segment, points in probe_segments(channel_locations, brain_atlas).items():
            for point, (ml, ap, dv) in enumerate(points):
                rows.append(dict(session=session, probe=index, segment=segment,
                                 point=point, ml=ml, ap=ap, dv=dv))

tracks = pd.DataFrame(rows)
OUTPUT.parent.mkdir(parents=True, exist_ok=True)
tracks.to_csv(OUTPUT, index=False)
print(f"wrote {len(tracks) // 4} probe tracks to {OUTPUT}")
