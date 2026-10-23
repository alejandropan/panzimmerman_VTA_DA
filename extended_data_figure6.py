"""
Extended Data Fig. 6 - dopamine release in NAc and DLS during VTA stimulation.

Two panels:

    the example traces   NAc against DLS in one animal (dop87), mean +/- s.e.m.
                         over that session's 20 stimulation trains
    the summary          each fibre's evoked response, one point per animal

Self-contained apart from `extended_data_figure6_traces.csv`, which sits
beside this script and holds every z-scored PSTH the figure uses. The raw
photometry is not needed and is not published.

Usage (interactive), or:
    python extended_data_figure6.py
"""

# %%
import matplotlib
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import warnings
from matplotlib import font_manager
from pathlib import Path
from scipy import stats

# %%
# ============================================================================
# Configuration
# ============================================================================
HERE = Path(__file__).resolve().parent if '__file__' in globals() else Path.cwd()
TRACES = HERE / 'extended_data_figure6_traces.csv'
SAVE_DIR = HERE

REGIONS = ['NAc', 'DLS']
REGION_COLOUR = {'NAc': 'g', 'DLS': 'orange'}

# The animal the example traces come from. The other two contribute to the
# summary only: dop86 to both fibres, dop88 to DLS alone, its NAc fibre
# having given no usable signal.
EXAMPLE_MOUSE = 'dop87'

# The evoked response is the mean of the first second after stimulation minus
# that trial's own pre-stimulation baseline. Defined on bin index rather than
# on time because the displayed axis is offset by one bin from the PSTH's own
# - see the CSV's header.
RESPONSE_BINS = (10, 29)
BASELINE_BINS = (0, 9)

TRACE_YLIM = (-0.9, 3.5)
SUMMARY_YLABEL = 'Evoked response\n(z-scored dF/F)'


def apply_style():
    """The figure style this panel pair uses."""
    sns.set_theme(style="ticks", context="notebook")
    installed = {f.name for f in font_manager.fontManager.ttflist}
    for family in ("Arial", "Helvetica", "Liberation Sans", "DejaVu Sans"):
        if family in installed:
            matplotlib.rcParams["font.sans-serif"] = [family]
            break
    else:
        warnings.warn("no sans-serif font from the preferred list is installed")
    matplotlib.rcParams["font.family"] = "sans-serif"
    matplotlib.rcParams["pdf.fonttype"] = 42      # editable text in Illustrator
    matplotlib.rcParams["ps.fonttype"] = 42


def save_panel(fig, filename, save_dir=None):
    """
    Write one panel as a PDF, with the text left editable.

    Args:
        fig: The figure to write.
        filename: Name including the extension.
        save_dir: Destination. Defaults to `SAVE_DIR`, read at call time.
    """
    save_dir = Path(SAVE_DIR if save_dir is None else save_dir)
    if not save_dir.is_absolute():
        raise ValueError(
            f"save_dir is relative ({save_dir}) - panels would land in the "
            "working directory. On Linux a UNC path is not absolute.")
    save_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_dir / filename, bbox_inches="tight")
    print(f"    wrote {save_dir / filename}")


# %%
# ============================================================================
# Loading
# ============================================================================
def load_traces(path=TRACES):
    """
    The trace table, checked for the shape the rest of this script assumes.

    Args:
        path: The trace table, `extended_data_figure6_traces.csv`.

    Returns:
        Long table of `mouse`, `region`, `trial`, `bin`, `time`, `zdff`.
    """
    if not Path(path).exists():
        raise FileNotFoundError(
            f"{path} not found. It holds the z-scored traces this script "
            "draws from - the recordings themselves are not needed.")
    traces = pd.read_csv(path)

    expected = {'mouse', 'region', 'trial', 'bin', 'time', 'zdff'}
    missing = expected - set(traces.columns)
    if missing:
        raise ValueError(f"{path} is missing column(s): {sorted(missing)}")
    # One `time` per `bin`, or the two ways of addressing a column disagree
    # and the response window means something different per fibre.
    if traces.groupby('bin')['time'].nunique().max() != 1:
        raise ValueError(f"{path}: a bin carries more than one time")

    fibres = traces.groupby(['region', 'mouse'])['trial'].nunique()
    print(f"    {len(traces)} rows, {len(fibres)} fibre(s), "
          f"{traces['bin'].nunique()} bins")
    print(fibres.to_string())
    return traces


def psth_of(traces, mouse, region):
    """
    One fibre's PSTH as a trial x bin matrix, with the time axis.

    Args:
        traces: Output of `load_traces`.
        mouse, region: Which fibre.

    Returns:
        `(binned, times)`: trials down, bins across, and the displayed time
        of each bin.
    """
    fibre = traces.loc[(traces['mouse'] == mouse)
                       & (traces['region'] == region)]
    if fibre.empty:
        raise KeyError(f"no traces for {mouse} {region}")
    wide = fibre.pivot(index='trial', columns='bin', values='zdff')
    times = (fibre.drop_duplicates('bin').sort_values('bin')['time']
             .to_numpy())
    return wide.to_numpy(), times


def evoked_response(traces):
    """
    Each fibre's response: the post-stimulation window minus its baseline.

    Taken per trial and then averaged, which is what the published summary
    plots. NaN bins are ignored throughout - a bin stays NaN when no frame
    fell in it.

    Args:
        traces: Output of `load_traces`.

    Returns:
        One row per fibre: `region`, `mouse`, `evoked_z`.
    """
    # Two groupby means rather than a per-trial `apply`: `mean` already
    # skips NaN, so this is the same arithmetic without depending on
    # `include_groups`, which only exists from pandas 2.2.
    per_trial = ['region', 'mouse', 'trial']
    response = (traces.loc[traces['bin'].between(*RESPONSE_BINS)]
                .groupby(per_trial)['zdff'].mean())
    baseline = (traces.loc[traces['bin'].between(*BASELINE_BINS)]
                .groupby(per_trial)['zdff'].mean())
    summary = ((response - baseline).groupby(['region', 'mouse']).mean()
               .rename('evoked_z').reset_index())
    # Published order: NAc first, then DLS.
    summary['region'] = pd.Categorical(summary['region'], categories=REGIONS,
                                       ordered=True)
    return summary.sort_values(['region', 'mouse']).reset_index(drop=True)


# %%
# ============================================================================
# Plotting
# ============================================================================
def plot_psth(binned, times, ax, color=None, alpha=1, linestyle=None,
              plot_error=True):
    """
    Mean +/- s.e.m. across trials, against time.

    Args:
        binned: Trial x bin matrix, NaN where a bin was empty.
        times: The displayed time of each bin.
        ax: Axes to draw into.
        color, alpha, linestyle: Passed to the line.
        plot_error: Shade the standard error.

    Returns:
        The axes.
    """
    mean = np.nanmean(binned, axis=0)
    ax.plot(times, mean, color=color, alpha=alpha, linestyle=linestyle)
    if plot_error:
        error = stats.sem(binned, nan_policy='omit')
        ax.fill_between(times, mean - error, mean + error, color=color,
                        alpha=alpha / 2, linewidth=0)
    ax.set_xlabel('Time from stimulation (s)')
    ax.set_ylabel('z-score dF/F')
    return ax


# %%
# ============================================================================
# Run
# ============================================================================
apply_style()

print("\n#### ED Fig 6  trace table")
traces = load_traces()


# %%
# ============================================================================
# The example traces - one animal, both fibres
# ============================================================================
print(f"\n#### ED Fig 6  example traces ({EXAMPLE_MOUSE})")
fig, ax = plt.subplots(figsize=(3.2, 2.6))
for region in REGIONS:
    binned, times = psth_of(traces, EXAMPLE_MOUSE, region)
    print(f"    {region}: {binned.shape[0]} trains, "
          f"peak of the mean {np.nanmax(np.nanmean(binned, axis=0)):.2f}")
    plot_psth(binned, times, ax, color=REGION_COLOUR[region])

ax.set_ylim(*TRACE_YLIM)
ax.axvline(0, color='k', linestyle='--')
ax.legend(handles=[mpatches.Patch(color=REGION_COLOUR[r], label=r)
                   for r in REGIONS])
sns.despine(ax=ax)
save_panel(fig, 'figED6_example_traces.pdf')


# %%
# ============================================================================
# The summary - every fibre
# ============================================================================
print("\n#### ED Fig 6  evoked response summary")
summary = evoked_response(traces)
print(summary.to_string(index=False))

fig, ax = plt.subplots(figsize=(2.0, 2.6))
sns.barplot(data=summary, x='region', y='evoked_z', order=REGIONS,
            hue='region', hue_order=REGIONS,
            palette=[REGION_COLOUR[r] for r in REGIONS], errorbar='se',
            legend=False, ax=ax)
# One line per animal, joining its two fibres. dop88 contributed DLS only,
# so it appears as a single point.
sns.pointplot(data=summary, x='region', y='evoked_z', order=REGIONS,
              hue='mouse', palette='Grays', legend=False, ax=ax)
ax.set_ylabel(SUMMARY_YLABEL)
ax.set_xlabel('')
sns.despine(ax=ax)
save_panel(fig, 'figED6_evoked_summary.pdf')

print("\nDone.")
