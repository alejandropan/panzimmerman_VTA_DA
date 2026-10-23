"""
Build each decoder's job array and submit it.

    python launch_decoders.py                 # dry run - shows, changes nothing
    python launch_decoders.py --write         # write id_dict.csv / areas_summary.npy
    python launch_decoders.py --write --submit    # ... and sbatch

Replaces `master_decoder_paper.py`, `master_decoder_vcs.py`,
`master_decoder_totalq.py` and `master_decoder_QcontraQipsi.py`, which were
the same walk over sessions four times with an `STR=True/False` branch and a
different set of hardcoded output paths pasted into each.

For every spec in RUN it walks that spec's sessions, collects the regions
each recording covers, and numbers the (session, region) pairs. That
numbering IS the SLURM array index: task `i` runs row `i` of `id_dict.csv`.
So the ordering is load-bearing - it has to match the `.npy` files already on
disk - and it is reproduced exactly: sessions in list order, regions sorted
within a session by `np.unique`, counter incrementing across both.

Because of that, **an `id_dict.csv` that already exists is never silently
overwritten**. It is regenerated in memory and compared; if the two disagree
the script refuses and prints the difference, because rewriting it would
break the correspondence between array indices and the decoder outputs
already computed.

Nothing is written or submitted without the flags. The dry run is the default
on purpose: this is the one script here with side effects on the cluster.
"""

import sys
from pathlib import Path
import subprocess

import numpy as np
import pandas as pd

ANALYSIS_CODE_DIR = '/jukebox/witten/Alex/PYTHON/rewardworld/ephys/analysis'
if ANALYSIS_CODE_DIR not in sys.path:
    sys.path.append(ANALYSIS_CODE_DIR)

from ephys_alf_summary import alf                                # noqa: E402
from decoder_specs import (ROOT, SPECS,                          # noqa: E402
                           STRIATAL_AND_CORTICAL_REGIONS, resolve_sessions)

JOB_SCRIPT = (Path(__file__).resolve().parent / 'decoder_job.cmd'
              if '__file__' in globals() else Path.cwd() / 'decoder_job.cmd')

# Which decoders to launch. Subset this to work on one.
RUN = list(SPECS)


# ============================================================================
# Regions per session
# ============================================================================
_PROBE_CACHE = {}


def probe_locations(session):
    """
    Each probe's region labels for one session, under both maps.

    Cached: a session appears in several specs, and `alf(..., ephys=True)`
    opens every probe's spike and cluster files, which is what made the
    original masters slow.

    Args:
        session: Session path relative to the Subjects root.

    Returns:
        List of (group_locations, level_locations), one pair per probe.
    """
    if session not in _PROBE_CACHE:
        alfio = alf(ROOT + session, ephys=True)
        _PROBE_CACHE[session] = [
            (alfio.probe[p].cluster_group_locations,
             alfio.probe[p].cluster_level_locations)
            for p in np.arange(len(alfio.probe[:]))]
    return _PROBE_CACHE[session]


def session_areas(session, area_set):
    """
    The regions one session contributes, sorted.

    The two area sets differ in which location map is read, and that is the
    whole reason only the Q decoders cover cortex and pallidum:

    `striatal` reads `cluster_level_locations` alone. `level_map` is defined
    over striatal sub-labels only, so a cortical or pallidal cluster maps to
    NaN and drops out. It does leave the literal string 'excluded' (ACB, CP,
    STR and the Calleja island all map to it), which the originals ran as
    though it were a region - kept, because those `.npy` files exist and the
    array indices have to line up with them.

    `striatal_cortical_pallidal` reads both maps and keeps the ten named
    regions.

    Args:
        session: Session path relative to the Subjects root.
        area_set: 'striatal' or 'striatal_cortical_pallidal'.

    Returns:
        Sorted unique region labels.
    """
    per_probe = []
    for group_locations, level_locations in probe_locations(session):
        level = level_locations.unique()[~pd.isna(level_locations.unique())]
        if area_set == 'striatal':
            per_probe.append(level)
        else:
            group = group_locations.unique()[~pd.isna(group_locations.unique())]
            per_probe.append(np.concatenate([group, level]))
    return np.unique(np.concatenate(per_probe))


def build_id_dict(sessions, area_set):
    """
    The (session, region) table whose row numbers are the array indices.

    Args:
        sessions: Sessions to walk, in order.
        area_set: Passed to `session_areas`.

    Returns:
        `(id_dict, areas)`: the table with `ses` / `area` / `id`, and the
        per-session region lists as saved to `areas_summary.npy`. Note
        `areas` is NOT filtered even when `id_dict` is - that is what the
        originals saved.
    """
    keep = (STRIATAL_AND_CORTICAL_REGIONS
            if area_set == 'striatal_cortical_pallidal' else None)
    areas, rows, counter = [], [], 0
    for session in sessions:
        found = session_areas(session, area_set)
        areas.append(found)
        for area in found:
            if keep is None or area in keep:
                rows.append({'ses': session, 'area': area, 'id': counter})
                counter += 1
    return pd.DataFrame(rows), areas


# ============================================================================
# Comparing against what is already there
# ============================================================================
def compare_id_dict(new, path):
    """
    Whether a regenerated table matches the one on disk.

    Args:
        new: Freshly built table.
        path: Where `id_dict.csv` would go.

    Returns:
        ('absent' | 'same' | 'differs', message).
    """
    if not Path(path).exists():
        return 'absent', f"no id_dict.csv yet ({len(new)} tasks)"
    old = pd.read_csv(path)[['ses', 'area', 'id']]
    if len(old) != len(new):
        return 'differs', (f"{len(old)} rows on disk, {len(new)} regenerated - "
                           "the array size itself changed")
    merged = old.merge(new, on='id', suffixes=('_disk', '_new'))
    bad = merged[(merged['ses_disk'] != merged['ses_new'])
                 | (merged['area_disk'] != merged['area_new'])]
    if len(bad):
        lines = '; '.join(
            f"id {r.id}: {r.ses_disk} {r.area_disk} -> {r.ses_new} {r.area_new}"
            for r in bad.head(5).itertuples())
        return 'differs', f"{len(bad)} of {len(old)} rows moved. {lines}"
    return 'same', f"matches the {len(old)} rows on disk"


# ============================================================================
# Run
# ============================================================================
WRITE = '--write' in sys.argv
SUBMIT = '--submit' in sys.argv
FORCE = '--force' in sys.argv

if SUBMIT and not WRITE:
    raise SystemExit("--submit needs --write: the array size comes from the "
                     "id_dict this would write")
if not JOB_SCRIPT.exists():
    raise SystemExit(f"job script not found: {JOB_SCRIPT}")

print(f"{'DRY RUN - nothing will be written or submitted' if not WRITE else ''}"
      f"{'WRITING id_dict / areas_summary' if WRITE and not SUBMIT else ''}"
      f"{'WRITING and SUBMITTING' if SUBMIT else ''}")
print(f"{len(RUN)} spec(s): {', '.join(RUN)}\n")

blocked = []
for name in RUN:
    spec = SPECS[name]
    sessions = resolve_sessions(spec['sessions'])
    id_dict, areas = build_id_dict(sessions, spec['areas'])
    state, message = compare_id_dict(id_dict, Path(spec['folder']) / 'id_dict.csv')

    flag = {'same': 'OK  ', 'absent': 'NEW ', 'differs': 'STOP'}[state]
    print(f"{flag}  {name:17s} {len(sessions):3d} sessions, "
          f"{len(id_dict):4d} tasks (array 0-{len(id_dict) - 1})")
    print(f"        {message}")
    print(f"        {spec['folder']}")

    if state == 'differs' and not FORCE:
        blocked.append(name)
        print("        REFUSING to overwrite - the array indices would no "
              "longer match the .npy files already computed.")
        print("        Re-run with --force only if you mean to discard them.\n")
        continue

    if WRITE:
        folder = Path(spec['folder'])
        folder.mkdir(parents=True, exist_ok=True)
        id_dict.to_csv(folder / 'id_dict.csv')
        # Ragged by construction - one array per session, different lengths.
        # Newer numpy refuses to infer that, so the dtype is explicit.
        np.save(folder / 'areas_summary.npy',
                np.array(areas, dtype=object), allow_pickle=True)
        print("        wrote id_dict.csv and areas_summary.npy")

    command = (f"sbatch --export=ALL,SPEC={name} "
               f"--array=0-{len(id_dict) - 1} {JOB_SCRIPT}")
    if SUBMIT:
        result = subprocess.run(command, shell=True, capture_output=True,
                                text=True)
        print(f"        {command}")
        print(f"        {result.stdout.strip() or result.stderr.strip()}")
    else:
        print(f"        would run: {command}")
    print()

if blocked:
    raise SystemExit(f"stopped on {len(blocked)} spec(s) whose id_dict would "
                     f"change: {', '.join(blocked)}")
if not WRITE:
    print("Dry run finished. Nothing was written or submitted.")
