"""
Extended Data Figure 2 - The Q-learning model parameters are recoverable, but the
model's behavior is indistinguishable from that of REINFORCE and Actor-Critic models.

Pan-Vazquez, Zimmerman et al. "VTA dopamine neuron activity produces spatially
organized stimulus and action value representations through conditioned reinforcement."

Consolidated from three scripts:
    plot_loo_arviz.py                     -> panel a
    supp2.py                              -> panels b, c, d
    plot_parameters_discarded_models.py   -> panels e-h

Panel a reads a CSV written by `compute_loo_comparison.py`, which is the only
place that touches the LOO pickles, the reordering corrections and ArviZ. Run
that once (or after refitting a model); this script needs neither.

Run this file top to bottom, or step through it section by section (each `# %%`
marks a cell in VS Code / Spyder). There is no `main()`; every variable stays in
the namespace so panels can be re-plotted or tweaked one at a time.

Panels produced, in manuscript order:
    2a  PSIS-LOO model comparison, ELPD +/- SE and +/- dSE
    2b  Decision-variable similarity between models 1, 4 and 5
    2c  Model confusion matrix
    2d  Parameter recovery for Model 1
    2e  Model 2 parameter distributions
    2f  Model 3 parameter distributions
    2g  Model 4 parameter distributions
    2h  Model 5 parameter distributions

Styling matches the published figure, which was finished in Illustrator: the
violet is the same one as Figure 1, the LOO panel drops the Conditioned
Reinforcer, panel b shows the lower triangle only, and the parameter symbols use
the manuscript's naming throughout.
"""

# %%
# ============================================================================
# Imports and configuration
# ============================================================================
import os
import re
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.patches import Rectangle
from scipy.stats import pearsonr, sem

# --- Paths -------------------------------------------------------------------
# Everything the script reads or writes is declared here. The originals mixed the
# Windows share and the cluster mount; both are the same filesystem.
ON_CLUSTER = os.name != "nt"
WITTEN = Path("/jukebox/witten") if ON_CLUSTER else Path(r"\\cup.princeton.edu\witten")

SUBJECTS_ROOT = WITTEN / "Alex" / "Data" / "Subjects"
PAPER_DIR = (Path(__file__).resolve().parent if "__file__" in globals()
             else Path.cwd())
STAN_ROOT = (PAPER_DIR / "data" / "stan_fits"
             if (PAPER_DIR / "data" / "stan_fits").is_dir()
             else WITTEN / "Alex" / "Data" / "ephys_bandit" / "laser_stan_fits")

LOO_DIR = STAN_ROOT / "loo_results"
FIT_REAL_CSV = STAN_ROOT / "standard_w_forgetting" / "output" / "summary.csv"
FIT_SIM_CSV = (STAN_ROOT / "recovery_models" / "standard_w_forgetting_recovery"
               / "output" / "summary.csv")

# Derived from WITTEN like every other path. Hard-coding the UNC form here was
# a real bug on the cluster: a backslash is a legal filename character on Linux,
# so `Path(r"\\cup...")` is not an absolute path at all - it is a RELATIVE
# single-component name, and `mkdir(parents=True)` silently created a directory
# with backslashes in its name inside the working directory. The panels were
# written, just nowhere anyone would look.
SAVE_DIR = PAPER_DIR
SAVE_DIR.mkdir(parents=True, exist_ok=True)
if not SAVE_DIR.is_absolute():
    raise ValueError(f"SAVE_DIR is relative ({SAVE_DIR}) - panels would land "
                     "in the working directory")
print(f"panels will be written to {SAVE_DIR}")


# The 41 laser-only sessions, as paths relative to SUBJECTS_ROOT. This manifest
# is the record of WHICH sessions the models were fitted on; Subjects is the
# single record of what is in them. It replaces `ephys_bandit/data_laser_only`,
# a partial copy of the Subjects tree that existed only so a folder walk would
# land on this subset - see the note in `confusion_matrix.py`.
from ephys_alf_summary import LASER_ONLY_ABB                        # noqa: E402

# --- Figure style ------------------------------------------------------------
# Same two knobs as figure1.py. Nothing below passes an explicit `fontsize=` or
# `linewidth=`, so changing these moves every panel together.
FONT_SIZE = 12
LINE_WIDTH = 0.8


def sans_serif_stack():
    """
    The font list, preferring Arial but degrading quietly where it is absent.

    Cluster nodes rarely have Arial, and matplotlib then emits one `findfont`
    warning per text object - hundreds of lines that bury the real output.
    Liberation Sans is metrically compatible with Arial and is the usual
    substitute on Linux; DejaVu Sans ships with matplotlib and always exists.

    The chosen font only affects how the PDF looks here. `pdf.fonttype = 42`
    keeps the text editable, so the final typesetting can restyle it.
    """
    installed = {font.name for font in font_manager.fontManager.ttflist}
    stack = ["Arial", "Helvetica", "Liberation Sans", "DejaVu Sans"]
    present = [name for name in stack if name in installed]
    if not present:
        warnings.warn("none of Arial/Helvetica/Liberation Sans/DejaVu Sans is "
                      "installed; matplotlib will fall back on its own")
        return stack
    if present[0] != "Arial":
        warnings.warn(f"Arial not installed, drawing in {present[0]}")
    return present + [name for name in stack if name not in present]


SANS_SERIF = sans_serif_stack()

matplotlib.rcParams.update({
    "font.size": FONT_SIZE,
    "axes.labelsize": FONT_SIZE,
    "axes.titlesize": FONT_SIZE,
    "xtick.labelsize": FONT_SIZE,
    "ytick.labelsize": FONT_SIZE,
    "legend.fontsize": FONT_SIZE,
    "figure.titlesize": FONT_SIZE,
    "font.family": "sans-serif",
    "font.sans-serif": SANS_SERIF,
    # With fontset "custom", matplotlib's mathtext defaults to DejaVu whatever
    # the text font is, so the Greek in the axis labels would not match the rest
    # of the panel. Pin it to the same family.
    "mathtext.fontset": "custom",
    "mathtext.default": "it",
    "mathtext.rm": SANS_SERIF[0],
    "mathtext.it": f"{SANS_SERIF[0]}:italic",
    "mathtext.bf": f"{SANS_SERIF[0]}:bold",
    # `custom` resolves every mathtext slot, including `cal`, which otherwise
    # defaults to "cursive" and warns on any machine without a script face.
    # Nothing here uses calligraphic math; pointing it at the same family is
    # enough to silence it.
    "mathtext.cal": SANS_SERIF[0],
    "mathtext.sf": SANS_SERIF[0],
    "mathtext.tt": "monospace",
    "axes.linewidth": LINE_WIDTH,
    "xtick.major.width": LINE_WIDTH,
    "ytick.major.width": LINE_WIDTH,
    "xtick.major.size": 3,
    "ytick.major.size": 3,
    "lines.linewidth": LINE_WIDTH,
    "patch.linewidth": LINE_WIDTH,
    # Keep text as text rather than outlines, so it stays editable in Illustrator.
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "svg.fonttype": "none",
})

# --- Colours -----------------------------------------------------------------
# Sampled from the published figure. The violins are the same violet as
# Figure 1j - the #8172B2 in the original script was recoloured in Illustrator.
COLOR_INK = "k"              # circles, error bars, violin outlines, points
COLOR_DSE = "#808080"        # dSE triangles in panel a
COLOR_GUIDE = "#B4B4B4"      # identity line in panel d, zero line in e-h
COLOR_VIOLIN = "#CF98E5"     # parameter violins
COLOR_POINT = "#3B3B3B"      # recovery scatter points
HEATMAP_CMAP = "viridis"

# --- Model naming ------------------------------------------------------------
# One place for every model label, so a model reads identically in every panel.
# The published figure writes "CS+=Reward" in panel a and "CS+= Reward" in b, c
# and f; standardised here on the spaced form.
MODEL_LABELS = {
    1: "1. Extended\nQ-learning\n(CS+= Reward)",
    2: "2. Extended\nQ-learning\n(VTA Stim.=RPE)",
    3: "3. Extended\nQ-learning\n(CS+= Reward) -\nVariable α",
    4: "4. REINFORCE",
    5: "5. Actor-Critic",
    6: "6. Conditioned\nReinforcer",
}
# Single-line versions, for titles rather than tick labels.
MODEL_TITLES = {
    2: "Model 2 -  Extended Q-learning (VTA Stim.=RPE)",
    3: "Model 3 -  Extended Q-learning (CS+= Reward) - Variable α",
    4: "Model 4 -  REINFORCE",
    5: "Model 5 -  Actor - Critic",
}


# %%
# ============================================================================
# Panel a - PSIS-LOO model comparison
# ============================================================================
# Panel a reads the table written by compute_loo_comparison.py. That script is
# the only thing that touches the LOO pickles, the *_loo_i_reordered.npy
# corrections and ArviZ; re-run it if a model is refitted. Keeping it separate
# means this figure needs neither the fitting tree nor a matching ArviZ version,
# and the panel can be regenerated by anyone from a six-row CSV.
LOO_COMPARISON_CSV = LOO_DIR / "loo_comparison_5models.csv"
LOO_XTICKS = [-11000, -10900, -10800]
# Both markers for a model sit at the same x (its ELPD) and would overlap, so the
# dSE triangle is nudged up by this fraction of the row spacing. Cosmetic only -
# measured off the published panel, where the triangle sits 26.8 px above the
# circle on a 93.5 px row pitch.
DSE_MARKER_OFFSET = 0.29


def plot_loo(comparison, xticks=LOO_XTICKS, figsize=None):
    """
    Panel a - ELPD per model with two error bars.

    Both markers sit at the same x, the model's ELPD; only the error bar
    differs. Drawn directly rather than through `az.plot_compare`, whose legend
    labels the triangle "ELPD difference" and so implies it is plotted at the
    difference value, which it is not.

    Args:
        comparison: Output of `compare_loos`, best model first.
        xticks: X tick positions, from the published panel. Pass None to let
            matplotlib choose, e.g. if a different set of models shifts the range.
        figsize: Overrides the default, which grows with the number of models so
            the multi-line labels keep their room.
    """
    fig, ax = plt.subplots(figsize=figsize or (4.0, 1.0 + 0.8 * len(comparison)))
    y = np.arange(len(comparison))[::-1]          # best model at the top

    # One row per model. The circle carries the ELPD's own standard error, the
    # triangle the standard error of the trial-wise difference from the best
    # model. Both are drawn at the same x; only the error bar differs.
    ax.errorbar(comparison["elpd"], y, xerr=comparison["se"], fmt="o",
                color=COLOR_INK, elinewidth=LINE_WIDTH, markersize=5,
                label="ELPD\n± SE", zorder=3)
    ax.errorbar(comparison["elpd"], y + DSE_MARKER_OFFSET, xerr=comparison["dse"],
                fmt="^", color=COLOR_DSE, elinewidth=LINE_WIDTH, markersize=5,
                label="ELPD\n± dSE", zorder=2)
    # `comparison` is best-first, so row 0 is the model the dSEs are measured
    # against; the dashed line marks its ELPD.
    ax.axvline(comparison["elpd"].iloc[0], linestyle="--", color=COLOR_DSE,
               linewidth=LINE_WIDTH, zorder=1)

    ax.set_yticks(y)
    ax.set_yticklabels([MODEL_LABELS[n] for n in comparison.index])
    ax.set_ylim(y.min() - 0.7, y.max() + 0.9)
    if xticks is not None:
        ax.set_xticks(xticks)
    ax.set_xlabel("ELPD (LOO)")
    ax.set_title("Model comparison — PSIS-LOO")
    ax.legend(loc="upper left", frameon=True, edgecolor=COLOR_INK, framealpha=1)
    sns.despine(ax=ax)


# %%
# ============================================================================
# Panels b and c - decision-variable similarity and model confusion
# ============================================================================
# Both panels compare the same three models.
MODELS_3 = [1, 4, 5]
# Panel b names the decision variable each model contributes.
DV_LABELS = {
    1: MODEL_LABELS[1] + "\n$(Q^R - Q^L)$",
    4: MODEL_LABELS[4] + "\n$\\pi$(choice)",
    5: MODEL_LABELS[5] + "\n$\\pi$(choice)",
}
# Per-session .npy files holding each model's decision variable. The Q-learning
# decision variable is the Q difference; the other two are stored directly.
DV_FILES = {
    "qfs_QR": "QFS_median_QR.npy",
    "qfs_QL": "QFS_median_QL.npy",
    "reinforce": "REINFORCE_median_reward.npy",
    "ac": "AC_median_reward.npy",
}
# Colour scale limits, from the published panels. The colorbar ticks are set with
# `linspace` rather than `arange` so the last tick is exactly the limit: `arange`
# excludes its stop, and nudging the stop by an epsilon lands the final tick a
# float hair *above* vmax, where the colorbar may drop it.
PANEL_B_VLIM = (0.70, 1.00)
# Panel c spans about 0.04 across its nine cells, and the limits are
# deliberately much wider - the published panel used a scale 3.2x its own data
# range for the same reason. The claim is that the models sit almost on top of
# each other, so a scale fitted to the data would manufacture contrast that is
# not there. At these limits the largest visible gradient is the row effect,
# which is not a model comparison at all: it only says Q-learning's simulated
# dataset is more predictable than the other two. The numbers are printed in
# every cell regardless of the colour.
PANEL_C_VLIM = (-0.60, -0.40)                       # 4.9x the data range
PANEL_C_TICKS = np.linspace(-0.60, -0.40, 5)

# Panel c is computed by `confusion_matrix.py`, which writes this CSV: one row
# per (generative, recovered) cell. It is read rather than pasted, because the
# values that were pasted into the 2026 revision came from a July 2025 pipeline
# that was superseded before submission and differed in three ways - a
# Q-learning model without forgetting, log(mean p) in place of mean(log p), and
# scoring against the real mouse choices rather than the simulated ones. See
# that script's docstring.
CONFUSION_CSV = STAN_ROOT / "recovery_models" / "confusion_matrix.csv"


def session_alf_path(session, filename, subjects_root=SUBJECTS_ROOT):
    """
    Build the path to one file in a session's `alf` folder.

    Args:
        session: Entry from LASER_ONLY_ABB. These use Windows separators and a
            leading separator, so they are normalised rather than joined blindly.
        filename: File inside `alf`.
        subjects_root: Root the relative session paths hang off.

    Returns:
        Path to the file.
    """
    relative = str(session).replace("\\", "/").strip("/")
    return Path(subjects_root).joinpath(*relative.split("/"), "alf", filename)


def compute_dv_correlations(sessions):
    """
    Pearson correlation between each pair of models' decision variables.

    Args:
        sessions: Session paths relative to SUBJECTS_ROOT (LASER_ONLY_ABB).

    Returns:
        Array `[model_i, model_j, session]` of correlation coefficients, with
        NaN for any session whose files are missing.
    """
    cache = PAPER_DIR / "data" / "extended_data_figure2" / "dv_correlations.npy"
    if cache.exists():
        return np.load(cache)
    n = len(MODELS_3)
    corrs = np.full((n, n, len(sessions)), np.nan)
    for k, session in enumerate(sessions):
        try:
            dvs = [
                np.load(session_alf_path(session, DV_FILES["qfs_QR"]))
                - np.load(session_alf_path(session, DV_FILES["qfs_QL"])),
                np.load(session_alf_path(session, DV_FILES["reinforce"])),
                np.load(session_alf_path(session, DV_FILES["ac"])),
            ]
        except FileNotFoundError as err:
            warnings.warn(f"panel b: skipping {session} ({err})")
            continue
        for i in range(n):
            for j in range(n):
                corrs[i, j, k] = pearsonr(dvs[i], dvs[j])[0]
    return corrs


def annotated_heatmap(mean, sem_, labels, vlim, ticks, cbar_label, title,
                      axis_label, ax, lower_only=False):
    """
    A square annotated heatmap, drawn without seaborn.

    seaborn's `annot=` cannot be trusted across versions here. Its
    `_annotate_heatmap` iterates `zip(..., mesh.get_array(), ...)`, and
    matplotlib 3.8 changed `QuadMesh.get_array()` from a flat (N,) array to the
    original 2-D one - so on seaborn 0.12 with a newer matplotlib the loop
    yields ROWS rather than values, `zip` truncates after three, and only the
    first row of a 3x3 panel is annotated. It fails silently: the colours are
    all correct and the numbers simply are not there. That is the fifth
    seaborn-drift bug in this project, so these two panels now use matplotlib
    directly.

    Args:
        mean: 2-D array of cell values.
        sem_: Standard errors, same shape.
        labels: Tick labels, one per row/column.
        vlim: (vmin, vmax) for the colour scale.
        ticks: Colorbar tick positions.
        cbar_label: Colorbar label.
        title: Panel title.
        axis_label: (x label, y label).
        ax: Axes to draw into.
        lower_only: Hide the upper triangle, as panel b does.

    Returns:
        The image, so the caller can attach anything else to it.
    """
    mean = np.asarray(mean, dtype=float)
    n = mean.shape[0]
    hidden = (np.triu(np.ones_like(mean, dtype=bool), k=1) if lower_only
              else np.zeros_like(mean, dtype=bool))

    colours = plt.get_cmap(HEATMAP_CMAP).copy()
    colours.set_bad(alpha=0)          # masked cells show the page, not a colour
    image = ax.imshow(np.ma.masked_where(hidden, mean), cmap=colours,
                      vmin=vlim[0], vmax=vlim[1])

    normalise = plt.Normalize(*vlim)
    for i in range(n):
        for j in range(n):
            if hidden[i, j]:
                continue
            # Same rule seaborn uses: dark text on a light cell, white on a
            # dark one, switching at a relative luminance of 0.408.
            red, green, blue = colours(normalise(mean[i, j]))[:3]
            linear = [c / 12.92 if c <= 0.03928 else ((c + 0.055) / 1.055) ** 2.4
                      for c in (red, green, blue)]
            luminance = 0.2126 * linear[0] + 0.7152 * linear[1] + 0.0722 * linear[2]
            ax.text(j, i, f"{mean[i, j]:.3f}\n$\\pm$ {sem_[i][j]:.3f}",
                    ha="center", va="center",
                    color="0.15" if luminance > 0.408 else "w")
            ax.add_patch(Rectangle((j - 0.5, i - 0.5), 1, 1, fill=False,
                                   edgecolor=COLOR_INK, linewidth=LINE_WIDTH))

    ax.set_xticks(range(n), labels, rotation=45, ha="right")
    ax.set_yticks(range(n), labels, rotation=0)
    ax.set_title(title)
    ax.set_xlabel(axis_label[0])
    ax.set_ylabel(axis_label[1])
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.tick_params(length=0)

    bar = ax.figure.colorbar(image, ax=ax, shrink=0.75, ticks=ticks)
    bar.set_label(cbar_label)
    bar.ax.set_yticklabels([f"{tick:.2f}" for tick in ticks])
    bar.outline.set_linewidth(LINE_WIDTH)
    return image


def plot_dv_similarity(corrs):
    """
    Panel b - session-averaged correlation between models' decision variables.

    Only the lower triangle is drawn, as in the published panel; the matrix is
    symmetric, so the upper half repeats it.

    Args:
        corrs: Output of `compute_dv_correlations`.
    """
    r_mean = np.nanmean(corrs, axis=2)
    r_sem = np.asarray(sem(corrs, axis=2, nan_policy="omit"), dtype=float)
    labels = [DV_LABELS[n] for n in MODELS_3]

    fig, ax = plt.subplots(figsize=(4.4, 4.0))
    annotated_heatmap(r_mean, r_sem, labels, PANEL_B_VLIM,
                      np.linspace(*PANEL_B_VLIM, 7), "r",
                      "Decision-variable similarity", ("Model", "Model"), ax,
                      lower_only=True)


def load_confusion(csv=CONFUSION_CSV):
    """
    The nine panel-c cells, as (mean, s.e.m.) matrices.

    Args:
        csv: Output of `confusion_matrix.py`.

    Returns:
        (mean, sem), each 3x3 with rows the generative model and columns the
        recovered one, in MODELS_3 order.
    """
    if not Path(csv).exists():
        raise FileNotFoundError(
            f"panel c needs {csv}, which `confusion_matrix.py` writes. Run that "
            "first; it takes a few minutes.")
    table = pd.read_csv(csv)
    missing = [(g, r) for g in MODELS_3 for r in MODELS_3
               if not ((table["generative"] == g)
                       & (table["recovered"] == r)).any()]
    if missing:
        raise ValueError(f"{csv} is missing cell(s) {missing}")

    indexed = table.set_index(["generative", "recovered"])
    mean = np.array([[indexed.loc[(g, r), "mean_loglik"] for r in MODELS_3]
                     for g in MODELS_3], dtype=float)
    sem = np.array([[indexed.loc[(g, r), "sem_loglik"] for r in MODELS_3]
                    for g in MODELS_3], dtype=float)
    return mean, sem


def plot_confusion():
    """
    Panel c - mean per-trial log-likelihood, generative model vs recovered model.

    Each row scores the choices its generative model simulated, so the panel
    asks whether a model is identifiable from data it produced. It is not a
    comparison of fit to the observed behaviour - that is panel a.

    The comparison the panel makes is WITHIN a row: the same simulated choices,
    the three models. Row-to-row differences are larger but say only that one
    simulated dataset is more predictable than another.
    """
    mean, sem_ = load_confusion()
    labels = [MODEL_LABELS[n] for n in MODELS_3]

    fig, ax = plt.subplots(figsize=(4.4, 4.0))
    annotated_heatmap(mean, sem_, labels, PANEL_C_VLIM, PANEL_C_TICKS,
                      "Mean log likelihood per choice", "Model confusion",
                      ("Recovered model", "Generated model"), ax)


# %%
# ============================================================================
# Panel d - parameter recovery for Model 1
# ============================================================================
# (Stan parameter stem, short key, axis title, axis limits, x ticks, y ticks).
# The titles use the manuscript's symbols, matching Figure 1j. The published
# panel puts coarser ticks on x than on y, so the two are given separately.
RECOVERY_PARAMS = [
    ("beta_ses",            "beta",      r"$\beta_{rew}$",     (0, 5),
     [0, 2, 4],        [0, 2, 4]),
    ("stay_ses",            "stay",      r"$\beta_{stay}$",    (0, 5),
     [0, 2, 4],        [0, 2, 4]),
    ("sides",               "bias",      "Bias",               (-1, 1),
     [-1, 0, 1],       [-1, -0.5, 0, 0.5, 1]),
    ("alpha_ses",           "alpha",     r"$\alpha_{rew}$",    (0, 1),
     [0, 0.5, 1],      [0, 0.25, 0.5, 0.75, 1]),
    ("alphastay_ses",       "alphastay", r"$\alpha_{stay}$",   (0, 1),
     [0, 0.5, 1],      [0, 0.25, 0.5, 0.75, 1]),
    ("alphaforgetting_ses", "forget",    r"$\gamma_{forget}$", (0, 1),
     [0, 0.5, 1],      [0, 0.25, 0.5, 0.75, 1]),
]


def load_real_session_slots(sessions=LASER_ONLY_ABB, subjects=SUBJECTS_ROOT,
                            expected=41):
    """
    The 1-based Stan indices of the sessions that carry real trials.

    Every model is fitted over a padded mice-by-sessions grid, so the parameter
    vectors are longer than the number of sessions and most slots are empty.
    `NT_all[i] == 0` marks a padding slot, and that is the only authoritative
    record of it - a padding slot still appears in summary.csv, having sampled
    from the prior, so it cannot be spotted from the summary alone.

    Derived rather than written out: a hard-coded list silently goes stale if a
    session is added or dropped.

    Args:
        sessions: Session paths the grid is built from (`LASER_ONLY_ABB`).
        subjects: Root those paths hang off.
        expected: Number of real sessions to insist on. The manuscript reports
            41; a mismatch means the loaded data is not what the figures assume.

    Returns:
        `(slots, standata)`: the 1-based slot numbers with trials, and the Stan
        data dict they were read from.
    """
    grid = PAPER_DIR / "data" / "figure6" / "stan_grid.npz"
    if grid.exists():
        saved = np.load(grid)
        standata = {"NSxNSESS": int(saved["NSxNSESS"]),
                    "NT_all": saved["NT_all"]}
    else:
        import model_comparison_accu as mc

        standata = mc.make_stan_data_reduced(
            mc.load_data_sessions_stan(sessions, ROOT_FOLDER=subjects,
                                       trial_start=0, trial_end=None,
                                       expect_sessions=expected)
        )
    slots = [i + 1 for i in range(standata["NSxNSESS"])
             if standata["NT_all"][i] > 0]
    if expected is not None and len(slots) != expected:
        raise ValueError(
            f"found {len(slots)} sessions with trials in "
            f"{standata['NSxNSESS']} slots, expected {expected}. Either the data "
            f"changed or `expected` is stale. Slots: {slots}")
    return slots, standata


def load_recovery_parameters(standata, real_csv=FIT_REAL_CSV, sim_csv=FIT_SIM_CSV):
    """
    Posterior means for the real fit and the recovery fit, one row per session.

    Args:
        standata: Stan data dict from `load_real_session_slots`, used to skip
            the padded slots.
        real_csv: cmdstan summary for the fit to the animals' choices.
        sim_csv: cmdstan summary for the fit to data simulated from it.

    Returns:
        DataFrame with `<key>_real` and `<key>_sim` columns per parameter.
    """
    real_means = (pd.read_csv(real_csv)
                  .rename(columns={"Unnamed: 0": "name"}).set_index("name")["Mean"])
    sim_means = (pd.read_csv(sim_csv)
                 .rename(columns={"Unnamed: 0": "name"}).set_index("name")["Mean"])

    rows = []
    for i in range(standata["NSxNSESS"]):
        if standata["NT_all"][i] <= 0:
            continue
        row = {"slot": i + 1}                       # 1-based, as in the Stan names
        for stem, key, *_ in RECOVERY_PARAMS:
            name = f"{stem}[{i + 1}]"
            row[f"{key}_real"] = real_means.get(name, np.nan)
            row[f"{key}_sim"] = sim_means.get(name, np.nan)
        rows.append(row)
    return pd.DataFrame(rows)


def plot_recovery(recovery):
    """
    Panel d - fitted parameter from the animals' choices vs from simulated data.

    Args:
        recovery: Output of `load_recovery_parameters`.
    """
    fig, axes = plt.subplots(1, len(RECOVERY_PARAMS), figsize=(15, 2.6),
                             gridspec_kw={"wspace": 0.55})
    for ax, (_, key, title, lim, xticks, yticks) in zip(axes, RECOVERY_PARAMS):
        x = recovery[f"{key}_real"].to_numpy(dtype=float)
        y = recovery[f"{key}_sim"].to_numpy(dtype=float)

        ax.plot(lim, lim, linestyle="--", color=COLOR_GUIDE,
                linewidth=LINE_WIDTH, zorder=0)
        ax.scatter(x, y, s=16, color=COLOR_POINT, alpha=0.75,
                   edgecolor="none", zorder=2)

        # A missing fit means one of the two summary.csv files does not contain
        # this parameter for that session, which is a problem to fix rather than
        # to plot around. Checked explicitly because the alternatives are both
        # bad: `pearsonr` has no `nan_policy`, and a NaN in the input raises an
        # unhelpful "array must not contain infs or NaNs" on scipy < ~1.15 but
        # silently returns NaN on newer versions, printing "r = nan" on the panel.
        missing = ~(np.isfinite(x) & np.isfinite(y))
        if missing.any():
            slots = recovery.loc[missing, "slot"].tolist()
            raise ValueError(
                f"{title}: {missing.sum()} of {len(x)} sessions have no fitted "
                f"value in one or both summaries (session slots {slots}). "
                f"Check that '{key}' is present in both "
                f"{Path(FIT_REAL_CSV).parent.parent.name} and "
                f"{Path(FIT_SIM_CSV).parent.parent.name}.")

        ax.text(0.05, 0.95, f"r = {pearsonr(x, y)[0]:.2f}",
                transform=ax.transAxes, va="top")

        ax.set_xlim(*lim)
        ax.set_ylim(*lim)
        ax.set_xticks(xticks)
        ax.set_yticks(yticks)
        ax.set_aspect("equal", adjustable="box")
        ax.set_title(title)
        ax.set_xlabel("Behavior Fit")
    axes[0].set_ylabel("Simulation Fit")
    sns.despine(fig=fig)


# %%
# ============================================================================
# Panels e-h - parameter distributions for the discarded models
# ============================================================================
# The real session slots come from `load_real_session_slots()`, run in the cell
# below; `REAL_SESSION_SLOTS` is bound there.
SES_PATTERN = re.compile(r"^(.+?)_ses\[(\d+)\]$")   # alpha_ses[7], beta_ses[7], ...

# Symbols as they appear in the published panels.
SYMBOL = {
    "alpha_rew": r"$\alpha_{rew}$",
    "alpha_stay": r"$\alpha_{stay}$",
    "alpha_win": r"$\alpha_{win}$",
    "alpha_loss": r"$\alpha_{loss}$",
    "gamma_forget": r"$\gamma_{forget}$",
    "eta": r"$\eta$",
    "beta_rew": r"$\beta_{rew}$",
    "beta_stay": r"$\beta_{stay}$",
    "beta_win": r"$\beta_{win}$",
    "beta_loss": r"$\beta_{loss}$",
    "bias": "Bias",
}

# One entry per panel. `stems` maps the Stan parameter stem (the part before
# `_ses[n]`; `sides[n]` is normalised to `side_ses[n]`) onto a key in SYMBOL.
#
# The map is per-model on purpose: the same Stan name means different things in
# different models. `alpha` is the reward learning rate in models 2 and 3 but the
# actor's win decay rate in models 4 and 5, and `beta` likewise goes from the
# reward weight to the win weight. Do not hoist this into one shared dict.
# `layout` is the published grouping: "separate" gives each parameter its own
# narrow axis under a shared group title, "shared" puts them on one axis.
#
# NOTE: the stems for models 3-5 are inferred from the parameter symbols in the
# figure and from Figure 1's naming. Any stem found in a summary.csv but absent
# from `stems` is still plotted, under its raw name in a trailing group, and
# named in a warning - so a first run tells you exactly what to correct here.
PARAM_PANELS = {
    "e": {
        "model": 2,
        "summary": STAN_ROOT / "standard_w_forgetting_RPE_laser" / "output" / "summary.csv",
        "stems": {
            "alpha": "alpha_rew",
            "alphastay": "alpha_stay",
            "alphaforgetting": "gamma_forget",
            "beta": "beta_rew",
            "side": "bias",
            "stay": "beta_stay",
        },
        "layout": [
            ("α & γ", "separate", ["alpha_rew", "alpha_stay", "gamma_forget"]),
            ("β & Bias", "shared", ["beta_rew", "bias", "beta_stay"]),
        ],
    },
    "f": {
        "model": 3,
        "summary": STAN_ROOT / "QFS_variable_alpha" / "output" / "summary.csv",
        "stems": {
            "alpha": "alpha_rew",
            "alphastay": "alpha_stay",
            "alphaforgetting": "gamma_forget",
            "eta": "eta",
            "beta": "beta_rew",
            "side": "bias",
            "stay": "beta_stay",
        },
        "layout": [
            ("α & γ", "separate", ["alpha_rew", "alpha_stay", "gamma_forget"]),
            ("Volatility (η)", "separate", ["eta"]),
            ("β & Bias", "shared", ["beta_rew", "bias", "beta_stay"]),
        ],
    },
    "g": {
        "model": 4,
        "summary": STAN_ROOT / "REINFORCE_new" / "output" / "summary.csv",
        "stems": {
            "alpha": "alpha_win",
            "alphaforgetting": "alpha_loss",
            "beta": "beta_win",
            "dmwater": "beta_loss",
            "side": "bias",
        },
        "layout": [
            ("Decay rates", "separate", ["alpha_win", "alpha_loss"]),
            ("β & Bias", "shared", ["beta_win", "beta_loss", "bias"]),
        ],
    },
    "h": {
        "model": 5,
        "summary": STAN_ROOT / "actor_critic" / "output" / "summary.csv",
        "stems": {
            "alpha": "alpha_win",
            "alphaforgetting": "alpha_loss",
            "beta": "beta_win",
            "dmwater": "beta_loss",
            "side": "bias",
            # The critic's learning rate, "Critic α" in the panel. The summary
            # calls it `alpharew_ses`, distinct from the actor's `alpha_ses`.
            "alpharew": "alpha_rew",
        },
        "layout": [
            ("Actor decay rates", "separate", ["alpha_win", "alpha_loss"]),
            ("Critic α", "separate", ["alpha_rew"]),
            ("Actor β & Bias", "shared", ["beta_win", "beta_loss", "bias"]),
        ],
    },
}


def read_session_params(summary_csv, stems, panel):
    """
    Read session-level posterior means from a cmdstan summary.

    Args:
        summary_csv: Path to the model's `summary.csv`.
        stems: Mapping of Stan parameter stem -> key in SYMBOL.
        panel: Panel letter, used in the warning about unmapped stems.

    Returns:
        Long DataFrame with `symbol`, `slot` and `Mean` columns, restricted to
        the real session slots.
    """
    df = pd.read_csv(summary_csv)
    df = df.rename(columns={df.columns[0]: "param"})

    # `sides[n]` is session-level too, it just does not use the `_ses[n]` naming.
    names = df["param"].astype(str).str.replace(r"^sides\[", "side_ses[", regex=True)
    parsed = names.str.extract(SES_PATTERN)
    parsed.columns = ["stem", "slot"]

    out = pd.DataFrame({
        "stem": parsed["stem"],
        "slot": pd.to_numeric(parsed["slot"]),
        "Mean": df["Mean"],
    }).dropna(subset=["stem"])
    out = out[out["slot"].isin(REAL_SESSION_SLOTS)]

    unmapped = sorted(set(out["stem"]) - set(stems))
    if unmapped:
        warnings.warn(
            f"panel {panel}: no symbol for {unmapped} in PARAM_PANELS['{panel}']"
            f"['stems'] - plotted under their raw names. Mapped stems: "
            f"{sorted(stems)}")
    out["symbol"] = out["stem"].map(lambda s: SYMBOL.get(stems.get(s, s), s))
    return out.sort_values(["symbol", "slot"])


def violins(ax, data, symbols, ylabel=""):
    """
    Draw one or more parameter violins on a single axis.

    Args:
        ax: Target axes.
        data: Output of `read_session_params`.
        symbols: Display symbols to draw, in order.
        ylabel: Y-axis label, or "" for none.
    """
    subset = data[data["symbol"].isin(symbols)]
    # saturation=1 keeps the exact hex; seaborn otherwise desaturates to 75%,
    # which turns #CF98E5 into #CBA2DB and no longer matches the other figures.
    sns.violinplot(data=subset, x="symbol", y="Mean", order=symbols, ax=ax,
                   color=COLOR_VIOLIN, saturation=1, cut=0, inner="box",
                   linewidth=LINE_WIDTH)
    style_violins(ax)
    ax.set_xlabel("")
    ax.set_ylabel(ylabel)
    plt.setp(ax.get_xticklabels(), rotation=90)
    sns.despine(ax=ax)


def plot_param_panel(panel):
    """
    Panels e-h - session-level parameter distributions for one discarded model.

    Alpha-family parameters get an axis each because they differ by an order of
    magnitude; the betas share one axis. Group titles sit above their group.

    Args:
        panel: Key into PARAM_PANELS ("e", "f", "g" or "h").

    Returns:
        The DataFrame that was plotted, for inspection.
    """
    spec = PARAM_PANELS[panel]
    data = read_session_params(spec["summary"], spec["stems"], panel)

    # Anything the layout does not mention goes in a trailing group, so a
    # mis-specified stem shows up in the figure rather than vanishing.
    present = set(data["symbol"])
    laid_out = {SYMBOL[k] for _, _, keys in spec["layout"] for k in keys}
    leftover = [s for s in data["symbol"].unique() if s not in laid_out]

    # Drop any group entry the summary does not actually contain, so a stem that
    # is missing or mis-mapped leaves a gap rather than an empty axis.
    layout = []
    for title, kind, keys in spec["layout"]:
        symbols = [SYMBOL[k] for k in keys if SYMBOL[k] in present]
        missing = [SYMBOL[k] for k in keys if SYMBOL[k] not in present]
        if missing:
            warnings.warn(f"panel {panel}: {missing} are in the layout but not in "
                          f"{Path(spec['summary']).parent.parent.name} - skipped")
        if symbols:
            layout.append((title, kind, symbols))

    if leftover:
        warnings.warn(f"panel {panel}: {leftover} are not in the layout - "
                      "appended as 'other'")
        layout = layout + [("other", "shared", leftover)]

    widths, groups = [], []
    for title, kind, symbols in layout:
        n_axes = len(symbols) if kind == "separate" else 1
        groups.append((title, kind, symbols, n_axes))
        widths += [1] * n_axes if kind == "separate" else [len(symbols)]

    fig, axes = plt.subplots(
        1, len(widths), figsize=(0.9 * sum(widths) + 2.0, 2.8),
        gridspec_kw={"width_ratios": widths, "wspace": 0.9},
    )
    axes = np.atleast_1d(axes)

    i = 0
    for title, kind, symbols, n_axes in groups:
        first = i
        if kind == "separate":
            for symbol in symbols:
                violins(axes[i], data, [symbol],
                        ylabel="Coefficient" if i == 0 else "")
                i += 1
        else:
            violins(axes[i], data, symbols,
                    ylabel="Coefficient" if i == 0 else "")
            axes[i].axhline(0, color=COLOR_GUIDE, linewidth=LINE_WIDTH, zorder=0)
            i += 1
        # Centre the group title over the axes it covers.
        left = axes[first].get_position().x0
        right = axes[i - 1].get_position().x1
        fig.text((left + right) / 2, 0.99, title, ha="center", va="top")

    fig.suptitle(MODEL_TITLES[spec["model"]], y=1.20, fontweight="bold")
    return data


# %%
# ============================================================================
# Helpers
# ============================================================================
def style_violins(ax, edgecolor=COLOR_INK):
    """
    Give violins a thin dark outline and keep the median marker white.

    Done after the fact because `linecolor=` only exists in seaborn >= 0.13, and
    because the two versions build the inner box differently: 0.12 draws the
    median as a white PathCollection, 0.13 as a Line2D with a "_" marker. Both
    are left white here; everything else becomes `edgecolor`.

    Args:
        ax: Axes containing the violins.
        edgecolor: Outline colour.
    """
    for collection in ax.collections:
        face = collection.get_facecolor()
        if len(face) and tuple(np.round(face[0][:3], 2)) == (1.0, 1.0, 1.0):
            continue                                   # median dot, seaborn 0.12
        collection.set_edgecolor(edgecolor)
        collection.set_linewidth(LINE_WIDTH)
    for line in ax.lines:
        if line.get_marker() not in ("", "None", None):
            line.set_color("w")                        # median dash, seaborn 0.13
        else:
            line.set_color(edgecolor)


def save_panel(filename, directory=SAVE_DIR):
    """
    Tidy, save the current figure as a vector PDF, and close it.

    Args:
        filename: File name, including the .pdf extension.
        directory: Destination directory. Defaults to SAVE_DIR.
    """
    plt.savefig(Path(directory) / filename, format="pdf", bbox_inches="tight")
    plt.close()


# %%
# ============================================================================
# Fig. ED2a - PSIS-LOO model comparison
# ============================================================================
loo_comparison = pd.read_csv(LOO_COMPARISON_CSV, index_col="model")
print(loo_comparison.round(2).to_string())

plot_loo(loo_comparison)
save_panel("figED2a_loo_comparison.pdf")

# The six-model version is Extended Data Figure 11's panel, and is the other CSV
# compute_loo_comparison.py writes. `plot_loo` handles either size.
# loo_all = pd.read_csv(LOO_DIR / "loo_comparison_6models.csv", index_col="model")
# plot_loo(loo_all, xticks=None)
# save_panel("loo_comparison_all_six_models.pdf")


# %%
# ============================================================================
# Fig. ED2b - Decision-variable similarity
# ============================================================================
dv_corrs = compute_dv_correlations(LASER_ONLY_ABB)
plot_dv_similarity(dv_corrs)
save_panel("figED2b_decision_variable_similarity.pdf")


# %%
# ============================================================================
# Fig. ED2c - Model confusion
# ============================================================================
plot_confusion()
save_panel("figED2c_model_confusion.pdf")


# %%
# ============================================================================
# Session bookkeeping - needed by panel d and panels e-h
# ============================================================================
REAL_SESSION_SLOTS, standata = load_real_session_slots()
print(f"{len(REAL_SESSION_SLOTS)} real sessions of {standata['NSxNSESS']} slots")
print(f"  slots: {REAL_SESSION_SLOTS}")


# %%
# ============================================================================
# Fig. ED2d - Parameter recovery for Model 1
# ============================================================================
recovery = load_recovery_parameters(standata)
plot_recovery(recovery)
save_panel("figED2d_parameter_recovery.pdf")


# %%
# ============================================================================
# Fig. ED2e-h - Parameter distributions for the discarded models
# ============================================================================
param_data = {}
for panel_letter in ("e", "f", "g", "h"):
    model = PARAM_PANELS[panel_letter]["model"]
    param_data[panel_letter] = plot_param_panel(panel_letter)
    save_panel(f"figED2{panel_letter}_model{model}_parameters.pdf")
    print(f"panel {panel_letter} (model {model}): "
          f"{sorted(param_data[panel_letter]['symbol'].unique())}")
