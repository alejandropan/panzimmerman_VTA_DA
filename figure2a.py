"""
Figure 2a - recording tracks and optic fibres in a 3D brain.

Draws the Allen CCF meshes of the striatum, nucleus accumbens, VTA and
substantia nigra, every recorded probe track, and the two fibre positions, then
saves a screenshot of the scene.

Needs the IBL environment (`environment-ibl.yml`: mayavi, ibllib and the
IBL `iblapps` package for `atlaselectrophysiology.rendering`). The tracks are
read from `data/figure2/probe_tracks.csv`, written by
`pipeline/prepare_probe_tracks.py`; the meshes are downloaded once from the Allen
Institute into `data/figure2/meshes`.

Run interactively. Rotate the scene to the view of the published panel, then run
the last cell to save it.
"""

# %%
from pathlib import Path
from urllib.request import urlretrieve

import numpy as np
import pandas as pd
from atlaselectrophysiology import rendering
from mayavi import mlab

# %%
# ============================================================================
# Configuration
# ============================================================================
HERE = Path(__file__).resolve().parent if "__file__" in globals() else Path.cwd()
TRACKS_CSV = HERE / "data" / "figure2" / "probe_tracks.csv"
MESH_DIR = HERE / "data" / "figure2" / "meshes"
MESH_URL = ("https://download.alleninstitute.org/informatics-archive/"
            "current-release/mouse_ccf/annotation/ccf_2017/structure_meshes/")
SAVE_PATH = HERE / "fig2a_probe_tracks.png"
SAVE_SIZE = (2400, 1800)

PURPLE = (0.831, 0.345, 0.831)
BLACK = (0.0, 0.0, 0.0)
GREY = (0.5, 0.5, 0.5)
BLUE = (0.0, 0.231, 1.0)

MESHES = {
    "striatum": dict(structure=477, color=PURPLE, opacity=0.2),
    "nucleus accumbens": dict(structure=56, color=PURPLE, opacity=0.2),
    "VTA": dict(structure=749, color=BLACK, opacity=0.4),
    "SNc": dict(structure=374, color=BLACK, opacity=0.4),
}

SEGMENT_STYLE = {
    "track": dict(color=BLACK, tube_radius=40),
    "extension": dict(color=GREY, tube_radius=20),
}

# Plotted as [x, y, z] = [ap, dv, ml], in CCF micrometres.
FIBERS = [
    dict(ap=[8500.0, 8500.0], dv=[0.0, 5032.0], ml=[4639.0, 5239.0]),
    dict(ap=[8500.0, 8500.0], dv=[0.0, 5032.0], ml=[6839.0, 6339.0]),
]
FIBER_STYLE = dict(color=BLUE, tube_radius=300)
LINE_WIDTH = 3


# %%
# ============================================================================
# Functions
# ============================================================================
def mesh_file(structure):
    """
    Path of an Allen structure mesh, downloaded on first use.

    Args:
        structure: Allen CCF structure id.

    Returns:
        Path of the `.obj` file.
    """
    path = MESH_DIR / f"{structure}.obj"
    if not path.exists():
        MESH_DIR.mkdir(parents=True, exist_ok=True)
        urlretrieve(f"{MESH_URL}{structure}.obj", path)
    return path


# %%
# ============================================================================
# Run
# ============================================================================
if not TRACKS_CSV.exists():
    raise FileNotFoundError(
        f"{TRACKS_CSV} is missing. Write it with "
        "`python pipeline/prepare_probe_tracks.py`, which needs the Witten share.")
tracks = pd.read_csv(TRACKS_CSV)

fig = rendering.figure()
for mesh in MESHES.values():
    rendering.add_mesh(fig, str(mesh_file(mesh["structure"])),
                       color=mesh["color"], opacity=mesh["opacity"])

for (session, probe, segment), points in tracks.groupby(["session", "probe",
                                                         "segment"]):
    points = points.sort_values("point")
    mlab.plot3d(points["ap"].to_numpy(), points["dv"].to_numpy(),
                points["ml"].to_numpy(), line_width=LINE_WIDTH,
                **SEGMENT_STYLE[segment])

for fiber in FIBERS:
    mlab.plot3d(np.array(fiber["ap"]), np.array(fiber["dv"]),
                np.array(fiber["ml"]), line_width=LINE_WIDTH, **FIBER_STYLE)

# %%
mlab.savefig(str(SAVE_PATH), size=SAVE_SIZE)
print(f"wrote {SAVE_PATH}")
