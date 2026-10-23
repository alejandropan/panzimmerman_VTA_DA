"""
Figure 3k - Cortical projection strength against value encoding.

Produces one panel:
    3k  two axes - Relative Value (red) and State Value (blue) - each showing
        r per cortical input, sorted, with bootstrap-SD error bars.

The p-values are printed, not drawn: the published panel carries no
significance markers.
"""

# %%
# ============================================================================
# Imports and configuration
# ============================================================================
import os
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib
import matplotlib.pyplot as plt
from matplotlib import font_manager
from scipy.stats import pearsonr

# --- Paths -------------------------------------------------------------------
ON_CLUSTER = os.name != "nt"
WITTEN = Path("/jukebox/witten") if ON_CLUSTER else Path(r"\\cup.princeton.edu\witten")


HERE = (Path(__file__).resolve().parent if "__file__" in globals()
        else Path.cwd())
PROJECTION_DIR = HERE / "data" / "figure3"
CP_CSV = (PROJECTION_DIR
          / "projections_CP_grouped_allSlices_mean_per_subregion_newGroups20250904.csv")
NAC_CSV = (PROJECTION_DIR
           / "projections_ACB_grouped_allSlices_mean_per_subregion_newGroups20250904.csv")

ENCODING_MODEL_MAT = (
    WITTEN / "Chris" / "matlab" / "cz" / "ibl_da_neuropixels_2022"
    / "archive" / "20261014"
    / "encoding-model" / "plots-RelativeValueStateValue"
    / "encoding-model-output-2026-03-24-05-59-PM.mat")

SAVE_DIR = HERE
SAVE_DIR.mkdir(parents=True, exist_ok=True)
if not SAVE_DIR.is_absolute():
    raise ValueError(f"SAVE_DIR is relative ({SAVE_DIR}) - the panel would land "
                     "in the working directory")
print(f"panel will be written to {SAVE_DIR}")


from encoding_model_summary_to_df import load_encoding_model    # noqa: E402

# --- Analysis parameters -----------------------------------------------------
N_THRESH = 20          # a subregion needs this many neurons to be correlated
SIG_ALPHA = 0.01       # encoding-model p below this counts as "encoding"
N_BOOTSTRAP = 1000     # resamples behind each bar's error bar

# The six cortical inputs, in CSV-column order, mapped onto the label the
# published panel prints. 

CORTICAL_INPUTS = {
    "MeanIntensity_SS": "SS",
    "MeanIntensity_MOp_MOs": "MO",
    "MeanIntensity_PL": "PFC",
    "MeanIntensity_AId_AIp_GU_AIv": "Insula",
    "MeanIntensity_ACA": "ACC",
    "MeanIntensity_ORBvl_ORBl": "OFC",
}

# The two value signals. `column` is the encoding model's p-value column.
ANALYSES = {
    "relative_value": dict(column="overall_policy", title="Relative Value",
                           color="red"),
    "state_value": dict(column="overall_value", title="State Value",
                        color="dodgerblue"),
}

# --- Figure style ------------------------------------------------------------
FONT_SIZE = 12
LINE_WIDTH = 0.8
ERRORBAR_CAPSIZE = 0
XTICK_ROTATION = 90


def apply_style():
    """
    Figure-wide rcParams: font stack, line weights, editable PDF text.
    """

    installed = {font.name for font in font_manager.fontManager.ttflist}
    stack = ["Arial", "Helvetica", "Liberation Sans", "DejaVu Sans"]
    present = [name for name in stack if name in installed]
    if not present:
        warnings.warn("none of Arial/Helvetica/Liberation Sans/DejaVu Sans is "
                      "installed; matplotlib will fall back on its own")
        present = stack
    elif present[0] != "Arial":
        warnings.warn(f"Arial not installed, drawing in {present[0]}")

    matplotlib.rcParams.update({
        "font.size": FONT_SIZE,
        "axes.labelsize": FONT_SIZE,
        "axes.titlesize": FONT_SIZE,
        "xtick.labelsize": FONT_SIZE,
        "ytick.labelsize": FONT_SIZE,
        "figure.titlesize": FONT_SIZE,
        "font.family": "sans-serif",
        "font.sans-serif": present + [n for n in stack if n not in present],
        "axes.linewidth": LINE_WIDTH,
        "xtick.major.width": LINE_WIDTH,
        "ytick.major.width": LINE_WIDTH,
        "xtick.major.size": 3,
        "ytick.major.size": 3,
        "lines.linewidth": LINE_WIDTH,
        "patch.linewidth": LINE_WIDTH,
        # Keep text as text rather than outlines, so it stays editable.
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "svg.fonttype": "none",
    })


def save_panel(filename, directory=SAVE_DIR):
    """
    Save the current figure as a vector PDF and close it.

    Args:
        filename: File name, including the .pdf extension.
        directory: Destination directory. Defaults to SAVE_DIR.
    """
    plt.savefig(Path(directory) / filename, format="pdf", bbox_inches="tight")
    plt.close()


# %%
# ============================================================================
# Loading
# ============================================================================
def load_projection_intensity(cp_csv=CP_CSV, nac_csv=NAC_CSV):
    """
    Mean projection intensity per striatal subregion, CP and NAc together.

    The two CSVs overlap: the ACB file carries CP subregions and the CP file
    carries accumbens ones. Each is cut to the half it is authoritative for, on
    the subregion name, before they are concatenated - so no subregion appears
    twice.

    Args:
        cp_csv: Caudoputamen projection summary.
        nac_csv: Accumbens projection summary.

    Returns:
        One row per subregion, with a `SubregionName` column and one
        `MeanIntensity_*` column per cortical input.
    """
    cp = pd.read_csv(cp_csv)
    nac = pd.read_csv(nac_csv)

    is_accumbens_cp = cp["SubregionName"].str.contains("Accumbens", case=False,
                                                       na=False)
    is_accumbens_nac = nac["SubregionName"].str.contains("Accumbens", case=False,
                                                         na=False)
    data = pd.concat([cp.loc[~is_accumbens_cp].dropna(),
                      nac.loc[is_accumbens_nac].dropna()])

    duplicated = data.loc[data["SubregionName"].duplicated(), "SubregionName"]
    if len(duplicated):
        raise ValueError(f"subregion(s) in both CSVs: {sorted(set(duplicated))}")
    missing = [c for c in CORTICAL_INPUTS if c not in data.columns]
    if missing:
        raise ValueError(f"projection CSVs have no column(s) {missing}. Present: "
                         f"{[c for c in data.columns if c.startswith('MeanIntensity')]}")
    return data


ENCODING_CACHE = HERE / "data" / "figure3" / "encoding_fractions.csv.gz"


def load_encoding_fractions(subregions, mat=ENCODING_MODEL_MAT,
                            n_thresh=N_THRESH, alpha=SIG_ALPHA):
    """
    Fraction of neurons per subregion whose encoding of each value signal is
    significant.

    The encoding model gives one p-value per neuron per signal. Those are
    binarised at `alpha` and averaged within a subregion, so the result is a
    proportion between 0 and 1 - the quantity panel k correlates against
    projection intensity.

    Args:
        subregions: Subregion names to keep, i.e. the ones the projection CSVs
            cover.
        mat: Encoding-model output.
        n_thresh: Minimum neurons for a subregion to be kept.
        alpha: p below which a neuron counts as encoding the signal.

    Returns:
        One row per surviving subregion, indexed by `region_raw`, with one
        column per entry in ANALYSES holding the fraction significant.
    """
    if ENCODING_CACHE.exists():
        return pd.read_csv(ENCODING_CACHE, index_col=0)
    encoding_model = load_encoding_model(model_path=str(mat))
    encoding_model.region_raw = encoding_model.region_raw.explode()

    striatal = encoding_model.loc[
        np.isin(encoding_model.region_raw, np.asarray(subregions))].copy()

    neurons = striatal.groupby("region_raw")["id"].count()
    kept = neurons.index[neurons >= n_thresh]
    dropped = neurons.drop(kept)
    if len(dropped):
        print(f"  dropped {len(dropped)} subregion(s) under {n_thresh} neurons: "
              + ", ".join(f"{name} ({n})" for name, n in dropped.items()))
    striatal = striatal.loc[np.isin(striatal.region_raw, kept)]

    columns = {name: spec["column"] for name, spec in ANALYSES.items()}
    binarised = pd.DataFrame(
        {name: 1 * (striatal[column] <= alpha) for name, column in columns.items()},
        index=striatal.index)
    binarised["region_raw"] = striatal["region_raw"].to_numpy()

    fractions = binarised.groupby("region_raw")[list(columns)].mean()
    print(f"  {len(fractions)} subregions, {len(striatal)} neurons")
    return fractions


# %%
# ============================================================================
# Correlations
# ============================================================================
def bootstrap_pearson(x, y, n_samples=N_BOOTSTRAP):
    """
    Pearson r over `n_samples` resamples of the (x, y) pairs, with replacement.

    Not seeded, so repeated runs differ slightly.

    Args:
        x, y: Paired 1-D arrays, already finite.
        n_samples: Number of resamples.

    Returns:
        Array of `n_samples` correlation coefficients.
    """
    n = len(x)
    correlations = []
    for _ in range(n_samples):
        indices = np.random.choice(n, n, replace=True)
        correlations.append(pearsonr(x[indices], y[indices])[0])
    return np.array(correlations)


def correlate_inputs(frame, value_column, inputs=CORTICAL_INPUTS,
                     n_bootstrap=N_BOOTSTRAP):
    """
    One correlation per cortical input, across subregions.

    Args:
        frame: Output of `build_panel_frame`.
        value_column: Which fraction-significant column to correlate.
        inputs: Mapping of intensity column -> published label.
        n_bootstrap: Resamples per input.

    Returns:
        One row per input: `region`, `r`, `r_sd`, `p`, `n`.
    """
    rows = []
    for column, label in inputs.items():
        x = frame[column].to_numpy(dtype=float)
        y = frame[value_column].to_numpy(dtype=float)

        finite = np.isfinite(x) & np.isfinite(y)
        if finite.sum() < 3:
            raise ValueError(f"{label}: only {finite.sum()} subregion(s) with "
                             "both a projection intensity and an encoding "
                             "fraction - cannot correlate")
        x, y = x[finite], y[finite]

        draws = bootstrap_pearson(x, y, n_bootstrap)
        if not np.isfinite(draws).all():
            raise ValueError(
                f"{label}: {(~np.isfinite(draws)).sum()} of {n_bootstrap} "
                "bootstrap draws gave an undefined r, which happens when a "
                f"resample of the {len(x)} subregions is constant in one "
                "variable. The bar would silently vanish.")
        rows.append({"region": label, "r": draws.mean(), "r_sd": draws.std(),
                     "p": pearsonr(x, y)[1], "n": len(x)})

    return pd.DataFrame(rows)


def build_panel_frame(projections, fractions):
    """
    Join the encoding fractions onto the projection intensities.

    Args:
        projections: Output of `load_projection_intensity`.
        fractions: Output of `load_encoding_fractions`, indexed by subregion.

    Returns:
        One row per subregion, carrying both the fractions and the six
        intensity columns.
    """
    frame = pd.merge(fractions, projections, left_on="region_raw",
                     right_on="SubregionName", how="left")
    unmatched = frame.loc[frame["SubregionName"].isna(), :]
    if len(unmatched):
        raise ValueError(f"{len(unmatched)} subregion(s) have an encoding "
                         "fraction but no projection row")
    return frame


# %%
# ============================================================================
# Plotting
# ============================================================================
def plot_correlation_bars(table, ax, color, title, ylabel=None):
    """
    One axis of panel k - r per cortical input, sorted ascending.

    Draws its own marks and nothing else: no figure, no saving, no panel letter.

    Args:
        table: Output of `correlate_inputs`.
        ax: Axes to draw into.
        color: Bar colour.
        title: Axis title.
        ylabel: Y-axis label, or None for no label. Only the left axis of the
            panel carries one.
    """
    ordered = table.sort_values("r").reset_index(drop=True)
    sns.barplot(data=ordered, x="region", y="r", color=color, ax=ax)
    # A categorical barplot puts its bars at 0..n-1, in the frame's row order.
    ax.errorbar(x=np.arange(len(ordered)), y=ordered["r"], yerr=ordered["r_sd"],
                fmt="none", ecolor="k", elinewidth=LINE_WIDTH,
                capsize=ERRORBAR_CAPSIZE)

    ax.set_title(title)
    ax.set_xlabel("")
    ax.set_ylabel(ylabel if ylabel else "")
    ax.tick_params(axis="x", rotation=XTICK_ROTATION)
    sns.despine(ax=ax)


def report(name, table):
    """
    Print one analysis's correlations, strongest negative first.

    Args:
        name: Key into ANALYSES.
        table: Output of `correlate_inputs`.
    """
    print(f"\n#### {ANALYSES[name]['title']}  "
          f"(n = {table['n'].iloc[0]} subregions, {N_BOOTSTRAP} bootstrap draws)")
    print(table.sort_values("r")[["region", "n", "r", "r_sd", "p"]]
          .to_string(index=False,
                     formatters={"r": "{:+.3f}".format, "r_sd": "{:.3f}".format,
                                 "p": "{:.4f}".format}))


# %%
# ============================================================================
# Run - load
# ============================================================================
apply_style()

print("projection intensities:")
projections = load_projection_intensity()
print(f"  {len(projections)} subregions, inputs {list(CORTICAL_INPUTS.values())}")

print("encoding model:")
fractions = load_encoding_fractions(projections["SubregionName"].unique())

panel = build_panel_frame(projections, fractions)


# %%
# ============================================================================
# Run - correlations
# ============================================================================
correlations = {}
for name, spec in ANALYSES.items():
    correlations[name] = correlate_inputs(panel, value_column=name)
    report(name, correlations[name])


# %%
# ============================================================================
# Fig. 3k - projection strength against value encoding
# ============================================================================
YLABEL = "Projection Strength vs\nValue Encoding (r)"

fig, axes = plt.subplots(1, len(ANALYSES), figsize=(4.2, 2.8))
for ax, (name, spec) in zip(np.atleast_1d(axes), ANALYSES.items()):
    plot_correlation_bars(correlations[name], ax, color=spec["color"],
                          title=spec["title"],
                          ylabel=YLABEL if name == "relative_value" else None)
plt.tight_layout()
save_panel("fig3k_projection_strength_vs_encoding.pdf")