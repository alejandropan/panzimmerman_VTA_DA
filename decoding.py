"""
Shared machinery for the time-resolved population-decoding panels.

Panels, established from the published legends:

    figure2.py    k,l,m,n   Qcontra/Qipsi action value, contra/ipsi trials
    figure3.py    f         relative value
                  g         state value
    figure6.py    k,l       V_cs, all rewarded trials and the >=5-skip subset
                  p,q       V_choice, same two
    ED figure 5   a-d       the Fig. 2 decoders, cortical and pallidal regions
    ED figure 7   p,q       state value as the unweighted / weighted sum of Qs
    ED figure 12  d         the Conditioned Reinforcer model's choice policy

Each figure script declares its own panel table, imports from here, and writes
its own PDFs.
"""

import os
import warnings
from pathlib import Path

import matplotlib
import matplotlib.transforms as transforms
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import font_manager
from scipy.stats import combine_pvalues, ranksums

# %%
# ============================================================================
# Paths and output
# ============================================================================
ON_CLUSTER = os.name != "nt"
WITTEN = Path("/jukebox/witten") if ON_CLUSTER else Path(r"\\cup.princeton.edu\witten")

# The release folder. A reader who has cloned the repository points this at
# their copy; everything below is relative to it.
RELEASE = (Path(__file__).resolve().parent if "__file__" in globals()
           else Path.cwd())

DECODERS_CONTRA = RELEASE / "decoders_data" / "decoders_contra_results_mua"
DECODERS_REWARDED = RELEASE / "decoders_data" / "decoders_rewarded_results_mua"
SAVE_DIR = RELEASE

SUMMARY_FILE = "summary_102030405060.csv"
SUMMARY_FILE = "summary_102030405060.csv"

# The decoder output folder for each analysis.

DECODER_FOLDERS = {
    'qcontra_contra':    DECODERS_CONTRA / "decoder_output_qcontra_contra_cue_forget" / "location_level",
    'qcontra_ipsi':      DECODERS_CONTRA / "decoder_output_qcontra_ipsi_cue_forget" / "location_level",
    'qipsi_contra':      DECODERS_CONTRA / "decoder_output_qipsi_contra_cue_forget" / "location_level",
    'qipsi_ipsi':        DECODERS_CONTRA / "decoder_output_qipsi_ipsi_cue_forget" / "location_level",
    'relative_value':    DECODERS_CONTRA / "decoder_output_deltaq_cue_forget" / "location_level",
    'pi_choice':         DECODERS_CONTRA / "decoder_output_seqpi" / "location_level",
    'state_value':       DECODERS_REWARDED / "decoder_output_totalq_outcome_forget" / "location_level",
    'state_weighted':    DECODERS_REWARDED / "decoder_output_totalq_weighted_outcome_forget" / "location_level",
    'state_unweighted':  DECODERS_REWARDED / "decoder_output_totalq_unweighted_outcome_forget" / "location_level",
    'vchoice':           DECODERS_REWARDED / "decoder_output_vchoice" / "rewarded_only" / "location_level_all",
    'vcs':               DECODERS_REWARDED / "decoder_output_vcs" / "rewarded_only" / "location_level_all",
    'vchoice_skips':     DECODERS_REWARDED / "decoder_output_vchoice" / "rewarded_only" / "location_level_skips",
    'vcs_skips':         DECODERS_REWARDED / "decoder_output_vcs" / "rewarded_only" / "location_level_skips",
}



def _readable(path):
    """The plain .csv if it is there, else the .csv.gz beside it."""
    if path.exists():
        return path
    gzipped = path.with_name(path.name + '.gz')
    return gzipped if gzipped.exists() else path


def panel_files(folder, n_neurons):
    """`(data_csv, null_csv)` for one decoder folder and neuron count."""
    return (_readable(folder / SUMMARY_FILE),
            _readable(folder / f"nsummary_{n_neurons}.csv"))


def apply_style():
    """
    The figure style every decoder panel uses.
    """
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
        filename: Name including the extension, e.g. `fig2k_qcontra_contra.pdf`.
        save_dir: Destination directory. Defaults to `SAVE_DIR`, read at call
            time rather than bound as a default, so that pointing the module
            somewhere else takes effect.
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
# Configuration
# ============================================================================
# The columns that together identify one decoding cell: one region, in one
# hemisphere of one session, at one time bin, for one neuron count.
KEYS = ['time_bin', 'n_neurons', 'region', 'hemisphere', 'mouse', 'date']

# The decoder pipeline saved variance explained under the column name `mse`.
# It is R2. The name is wrong at the source (saved as mse, from LEGACY version, it's actually R2) and is corrected here, once, so
# that nothing downstream repeats it.
R2_COLUMN_IN_FILE = 'mse'
R2 = 'r2'
R2_CORRECTED = 'r2_corrected'

TIME_BIN = 0.1          # seconds per decoding bin

# Names as the pipeline writes them -> names as the figures use them. The CP
# subregions are abbreviated in the files, and the pallidum is spelled `Gpe`
# there against `GPe` in the manuscript.
REGION_RENAMES = {'Int': 'CP-Int', 'Rostral': 'CP-Ros', 'Caudal': 'CP-Cau',
                  'Gpe': 'GPe'}

# Region sets in the order the published legends use.
STRIATAL = ['NAc', 'CP-Ros', 'CP-Int', 'CP-Cau']
CORTICAL = ['MO', 'OFC', 'PFC', 'SS']
PALLIDAL = ['VP', 'GPe']

# Palletes
PALETTE = {
    'NAc': '#568A42', 'CP-Ros': '#FF7800', 'CP-Int': '#F2A61D', 'CP-Cau': '#D3D726',
}
GREY_RAMP = 'Grays'

# Thresholds for the degenerate-cell exclusion. A decoder occasionally returns
# a catastrophically negative R2 - a fit that failed rather than a fit that
# found nothing - and those values are large enough to distort any mean they
# enter.
NULL_TRIM_FLOOR = -1.0   # a shuffle draw below this is degenerate
REAL_FLOOR      = -1.0   # omit a cell whose real R2 is below this
MAX_BAD_FRAC    = 0.5    # omit a cell if this fraction of its own draws are degenerate
NULL_MEAN_FLOOR = -0.3   # omit a cell whose null_sub is itself this negative

EXCLUSION_LABELS = {
    'REAL':      'real R2 below the floor',
    'NULL_FRAC': 'too many degenerate shuffle draws',
    'NULL_MEAN': 'null mean dragged down by a minority of bad draws',
    'REAL+NULL': 'both',
    'NO_REAL':   'no real R2 in the summary file',
    'NO_MATCH':  'no matching null group',
}


# %%
# ============================================================================
# Loading
# ============================================================================
def load_panel(data_csv, null_csv, n_neurons, regions, max_time_bin=None):
    """
    One panel's real and null decoding results, filtered and named.

    Args:
        data_csv: The `summary_*.csv` of decoders fitted to the observed
            behaviour.
        null_csv: The `nsummary_*.csv` of shuffle draws.
        n_neurons: Keep only decoders trained on this many neurons.
        regions: Keep only these regions.
        max_time_bin: Keep only bins up to and including this index, where only
            part of the recorded window is wanted.

    Returns:
        `(real, null)` frames, with the R2 column named `r2`.
    """
    frames = []
    for path in (data_csv, null_csv):
        frame = pd.read_csv(path)
        frame['region'] = frame['region'].replace(REGION_RENAMES)
        frame = frame.rename(columns={R2_COLUMN_IN_FILE: R2})
        # The null summaries do not always carry an n_neurons column;
        # nulls are normally run only on n=20 or n=40, the real
        # ones always have several ns.
        if 'n_neurons' in frame.columns:
            frame = frame.loc[frame['n_neurons'] == n_neurons]
        frame = frame.loc[frame['region'].isin(regions)].copy()
        if max_time_bin is not None:
            frame = frame.loc[frame['time_bin'] <= max_time_bin].copy()
        frames.append(frame)

    real, null = frames
    print(f"    loaded {len(real)} real cells, {len(null)} null draws "
          f"({n_neurons} neurons"
          + (f", bins <= {max_time_bin}" if max_time_bin is not None else "")
          + ")")
    return real, null


# %%
# ============================================================================
# The null correction
# ============================================================================
def null_correct(real, null):
    """
    Subtract each cell's own null mean, and flag the cells that cannot be used.

    Three conditions retire a cell, and `r2_corrected` is set to NaN for it:

        REAL       its real R2 is below REAL_FLOOR
        NULL_FRAC  more than MAX_BAD_FRAC of its own draws are degenerate
        NULL_MEAN  its null_sub is below NULL_MEAN_FLOOR

    NULL_MEAN is the one that is easy to miss and necessary. A MINORITY
    of catastrophic draws, well under MAX_BAD_FRAC, still drags the untrimmed mean
    far negative: 10% of draws at -20 pulls the mean past -0.5 while 90% of
    draws are fine. Without this check, a real R2 near zero minus a very
    negative null_sub yields a large POSITIVE corrected value with
    no signal behind it.

    Args:
        real: Decoders fitted to the observed behaviour, from `load_panel`.
        null: The matching shuffle draws.

    Returns:
        `real` with `null_sub`, `bad_frac`, `n_draws`, `n_bad_draws`,
        `r2_corrected` and `excl_reason` added. `excl_reason` is the empty
        string for a cell that is kept.
    """
    summary = null.groupby(KEYS)[R2].agg(
        null_sub='mean',
        bad_frac=lambda s: float((s < NULL_TRIM_FLOOR).mean()),
        n_draws='size',
        n_bad_draws=lambda s: int((s < NULL_TRIM_FLOOR).sum()),
    ).reset_index()

    real = real.merge(summary, on=KEYS, how='left')
    real[R2_CORRECTED] = real[R2] - real['null_sub']

    real_bad = real[R2] < REAL_FLOOR
    frac_bad = real['bad_frac'] > MAX_BAD_FRAC
    mean_bad = real['null_sub'] < NULL_MEAN_FLOOR
    null_bad = frac_bad | mean_bad

    # r2_corrected can only be NaN for two reasons, and they are different
    # problems: the real R2 is absent from the summary file, or the left merge
    # found no null group for this cell. Separated so the printed counts say
    # which. Note `real_bad` cannot catch a missing real R2, because
    # `NaN < REAL_FLOOR` is False.
    no_real = real[R2].isna()
    no_match = real[R2_CORRECTED].isna() & ~real_bad & ~null_bad & ~no_real

    # Six mutually exclusive reasons whose union is every cell the original
    # excluded: a cell with both a missing real R2 and a bad null lands on the
    # null reason, as it did before.
    real['excl_reason'] = ''
    real.loc[real_bad & ~null_bad, 'excl_reason'] = 'REAL'
    real.loc[frac_bad & ~real_bad, 'excl_reason'] = 'NULL_FRAC'
    real.loc[mean_bad & ~frac_bad & ~real_bad, 'excl_reason'] = 'NULL_MEAN'
    real.loc[real_bad & null_bad, 'excl_reason'] = 'REAL+NULL'
    real.loc[no_real & ~null_bad, 'excl_reason'] = 'NO_REAL'
    real.loc[no_match, 'excl_reason'] = 'NO_MATCH'

    real.loc[real['excl_reason'] != '', R2_CORRECTED] = np.nan
    return real


def report_exclusions(real):
    """
    Print how many cells were retired and why, then list them.

    A cell is one region x time bin x session-hemisphere. Printed rather than
    returned because it is a methods number, wanted on every run.
    """
    excluded = real.loc[real['excl_reason'] != '']
    print(f"    excluded {len(excluded)}/{len(real)} cells "
          f"(real < {REAL_FLOOR}, > {MAX_BAD_FRAC:.0%} degenerate draws, "
          f"or null_sub < {NULL_MEAN_FLOOR})")
    for reason, label in EXCLUSION_LABELS.items():
        count = int((real['excl_reason'] == reason).sum())
        if count:
            print(f"        {reason:10s} {count:4d}   {label}")
    if len(excluded):
        columns = ['mouse', 'date', 'hemisphere', 'region', 'time_bin', R2,
                   'null_sub', 'excl_reason', 'bad_frac', 'n_bad_draws', 'n_draws']
        print(excluded[columns].sort_values(['region', 'time_bin'])
              .to_string(index=False))


# %%
# ============================================================================
# Time axis
# ============================================================================
def add_time(frame, t0):
    """
    The time in seconds each bin's value refers to.

    A bin's value summarises the window up to and including its right edge, so
    bin 0 of a window starting at t0 is labelled `t0 + TIME_BIN`. That is why
    the published x-axes begin at -0.4 rather than -0.5.

    Rounded to milliseconds because the statistics group rows by this value, and
    `t0 + 6 * 0.1` is not exactly 0.1 * 6 in binary floating point.

    Args:
        frame: Any frame carrying a `time_bin` column.
        t0: Start of the panel's window, in seconds.

    Returns:
        The frame, with a `time` column added.
    """
    frame = frame.copy()
    frame['time'] = np.round(t0 + (frame['time_bin'] + 1) * TIME_BIN, 3)
    return frame


# %%
# ============================================================================
# Is decoding above its own null? (Fisher across recordings)
# ============================================================================
# Identifies one cell within a panel. `n_neurons` is omitted because
# `load_panel` has already restricted the frame to a single neuron count.
CELL_KEYS = ['region', 'time_bin', 'mouse', 'date', 'hemisphere']

MIN_RECORDINGS_FOR_FISHER = 3


def fisher_above_null(real, null):
    """
    Per region and timepoint, is decoding better than its own shuffles?

    Each recording contributes a one-sided rank p-value: where the real R2 sits
    among that cell's own shuffle draws. Those are combined across recordings
    with Fisher's method.

    Args:
        real: Output of `null_correct`.
        null: The matching shuffle draws.

    Returns:
        A frame of `region`, `time_bin`, `sig` - the Fisher-combined p-value, or
        NaN where fewer than MIN_RECORDINGS_FOR_FISHER recordings contributed.
    """
    # One pass over the nulls into an index, rather than a boolean scan of the
    # whole frame per cell as the original did.
    draws = {key: values.to_numpy()
             for key, values in null.groupby(CELL_KEYS)[R2]}

    rows, n_used, n_skipped, n_no_draws = [], 0, 0, 0
    for (region, time_bin), cells in real.groupby(['region', 'time_bin']):
        p_values = []
        for cell in cells.itertuples(index=False):
            if cell.excl_reason != '':
                n_skipped += 1
                continue
            full = draws.get((region, time_bin, cell.mouse, cell.date,
                              cell.hemisphere))
            if full is None or len(full) == 0:
                n_no_draws += 1
                continue
            clean = full[full >= NULL_TRIM_FLOOR]
            if len(clean) == 0:
                clean = full
            beaten = int(np.sum(getattr(cell, R2) > clean))
            p_values.append(1 - beaten / (len(clean) + 1))
            n_used += 1

        sig = (np.nan if len(p_values) < MIN_RECORDINGS_FOR_FISHER
               else combine_pvalues(p_values, method='fisher')[1])
        rows.append({'region': region, 'time_bin': time_bin, 'sig': sig})

    stats = pd.DataFrame(rows)
    print(f"    Fisher vs null: {n_used} cells contributed, "
          f"{n_skipped} already excluded"
          + (f", {n_no_draws} with no null draws" if n_no_draws else ""))
    thin = int(stats['sig'].isna().sum())
    if thin:
        print(f"        {thin} region x bin combination(s) had fewer than "
              f"{MIN_RECORDINGS_FOR_FISHER} recordings, so no p-value")
    return stats


def report_fisher(stats, t0, thresh=0.05):
    """Print the region x timepoint combinations that beat their own null."""
    significant = add_time(stats.loc[stats['sig'] < thresh], t0)
    print(f"    above null at P < {thresh}: {len(significant)} "
          f"region x timepoint combination(s)")
    if len(significant):
        print(significant[['region', 'time', 'sig']]
              .sort_values(['region', 'time']).to_string(index=False))


# %%
# ============================================================================
# Is NAc different from CP? (rank-sum at each timepoint)
# ============================================================================


def ranksum_nac_vs_cp(real, thresh=0.05):
    """
    Two-tailed rank-sum of NAc against the three CP subregions, per timepoint.

    Args:
        real: Output of `null_correct`, with a `time` column from `add_time`.
        thresh: Printed alongside each timepoint as a marker.

    Returns:
        A frame of `time`, `p`, one row per timepoint.
    """
    kept = real.loc[real['excl_reason'] == '']
    if 'NAc' not in set(kept['region']):
        raise ValueError("no NAc rows: this test is only for striatal panels")

    rows = []
    for time in sorted(kept['time'].unique()):
        at_t = kept.loc[kept['time'] == time]
        nac = at_t.loc[at_t['region'] == 'NAc', R2_CORRECTED]
        cp = at_t.loc[at_t['region'] != 'NAc', R2_CORRECTED]
        result = ranksums(nac, cp)
        marker = '  *' if result.pvalue < thresh else ''
        print(f"      t={time:>5.2f}  n(NAc)={len(nac):3d}  n(CP)={len(cp):3d}  "
              f"U={result.statistic:8.3f}  P={result.pvalue:7.4f}{marker}")
        rows.append({'time': time, 'n_nac': len(nac), 'n_cp': len(cp),
                     'p': float(result.pvalue)})
    return pd.DataFrame(rows)

# %%
# ============================================================================
# Plotting
# ============================================================================
def plot_timecourse(real, regions, ax, xlim, ylim, xlabel, title=None,
                    xticks=None, yticks=None, palette=None, legend=True):
    """
    Null-corrected R2 against time, one line per region.

    Mean +/- s.e.m. across sessions.
    Draws its own marks and nothing else: no figure, no saving, no panel letter.

    Args:
        real: Output of `null_correct`, with a `time` column from `add_time`.
        regions: Regions to draw, in legend order.
        ax: Axes to draw into.
        xlim: (start, end) of the time window, in seconds.
        ylim: (bottom, top) for the corrected R2 axis. Set explicitly so that
            panels meant to be compared share a scale.
        xlabel: Time axis label.
        title: Axes title, or None.
        xticks: Tick positions for the time axis, or None to let matplotlib
            choose.
        yticks: Tick positions for the corrected R2 axis, or None.
        palette: Anything seaborn accepts.
        legend: Draw the region legend.

    Returns:
        The axes.
    """
    sns.lineplot(data=real, x='time', y=R2_CORRECTED, hue='region',
                 hue_order=regions, errorbar='se', ax=ax,
                 palette=PALETTE if palette is None else palette)
    ax.axvline(0, linestyle='--', color='k', alpha=0.5)
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    if xticks is not None:
        ax.set_xticks(xticks)
    if yticks is not None:
        ax.set_yticks(yticks)
    ax.set_xlabel(xlabel)
    ax.set_ylabel('Corrected R2')
    if title is not None:
        ax.set_title(title)
    if legend:
        ax.legend(frameon=True, loc='upper left')
    elif ax.get_legend() is not None:
        ax.get_legend().remove()
    sns.despine(ax=ax)
    return ax


def mark_ranksum(ax, ranksum, thresh=0.05, trend=0.06):
    """
    An asterisk above each timepoint where NAc differs from pooled CP, and a
    hash where the p-value falls from `thresh` up to `trend`.

    Placed in blended coordinates - x in data units, y in axes units - so the
    row of marks stays at the top of the panel whatever the y-limits are.

    Args:
        ax: Axes already drawn by `plot_timecourse`.
        ranksum: Output of `ranksum_nac_vs_cp`.
        thresh: Draw an asterisk below this p-value.
        trend: Draw a hash from `thresh` up to this p-value.
    """
    blended = transforms.blended_transform_factory(ax.transData, ax.transAxes)
    for row in ranksum.itertuples(index=False):
        if row.p < thresh:
            marker = '*'
        elif row.p < trend:
            marker = '#'
        else:
            continue
        ax.text(row.time, 0.99, marker, transform=blended, ha='center',
                va='top', color='k', fontsize=14, fontweight='bold',
                clip_on=False)


# %%
# ============================================================================
# One panel, end to end
# ============================================================================
def prepare_panel(label, decoder, n_neurons, regions, t0, max_time_bin=None,
                  fisher=True, ranksum=False):
    """
    Load, correct and test one panel, printing everything as it goes.

    The whole per-panel sequence lives here so that seventeen panels cannot
    drift from each other. Drawing is left to the caller, because Extended Data
    Fig. 5 splits its regions across two axes while every other panel uses one.

    Args:
        label: Printed heading, e.g. `Fig 2k  Qcontra, contra trials`.
        decoder: Key into `DECODER_FOLDERS`.
        n_neurons: Neuron count this panel was built from.
        regions: Every region the panel draws, across all of its axes.
        t0: Start of the window, for the time axis.
        max_time_bin: Passed to `load_panel`.
        fisher: Run and print the above-null test.
        ranksum: Run and print NAc against pooled CP. Striatal panels only.

    Returns:
        `(real, stats, ranksum_result)`. `stats` and `ranksum_result` are None
        where that test was not requested.
    """
    print(f"\n#### {label}")
    data_csv, null_csv = panel_files(DECODER_FOLDERS[decoder], n_neurons)
    real, null = load_panel(data_csv, null_csv, n_neurons, regions,
                            max_time_bin)
    real = null_correct(real, null)
    report_exclusions(real)

    stats = None
    if fisher:
        stats = fisher_above_null(real, null)
        report_fisher(stats, t0)

    real = add_time(real, t0)

    ranksum_result = ranksum_nac_vs_cp(real) if ranksum else None
    return real, stats, ranksum_result


# %%
# ============================================================================
# The second panel style: R2 against neuron count
# ============================================================================
def load_neuron_count_panel(data_csv, regions, time_bin, min_neurons=40):
    """
    One time bin's decoding performance at every neuron count.

    No null is read and none is subtracted: these panels plot the RAW R2. (The
    Extended Data Fig. 4 legend says "corrected R2"; the panels are raw.)

    Only sessions that reached `min_neurons` in that region are kept, so the
    lines are drawn over one fixed set of sessions rather than a set that
    shrinks as the neuron count rises.

    Args:
        data_csv: The `summary_*.csv` of decoders fitted to the observed
            behaviour.
        regions: Keep only these regions.
        time_bin: The single bin these panels summarise.
        min_neurons: A session must have a decoder at this count to be kept.

    Returns:
        The real frame, with the R2 column named `r2`.
    """
    data_csv = _readable(Path(data_csv))
    frame = pd.read_csv(data_csv)
    frame['region'] = frame['region'].replace(REGION_RENAMES)
    frame = frame.rename(columns={R2_COLUMN_IN_FILE: R2})
    frame = frame.loc[frame['region'].isin(regions)]
    frame = frame.loc[frame['time_bin'] == time_bin].copy()

    # `id` is the session-and-region identifier the pipeline writes, so
    # selecting on it is per region, as the legend's counts are.
    if 'id' not in frame.columns:
        raise ValueError(f"{data_csv} has no `id` column, so sessions cannot "
                         f"be selected. Columns: {sorted(frame.columns)}")
    complete = frame.loc[frame['n_neurons'] == min_neurons, 'id'].unique()
    kept = frame.loc[frame['id'].isin(complete)].copy()
    print(f"    bin {time_bin}: {kept['id'].nunique()} of "
          f"{frame['id'].nunique()} sessions reached {min_neurons} neurons")
    for region in regions:
        n = kept.loc[kept['region'] == region, 'id'].nunique()
        print(f"        {region:8s} n={n}")
    return kept


def plot_r2_vs_neurons(real, regions, ax, xlim, ylim, title=None,
                       xticks=None, yticks=None, palette=None, legend=True):
    """
    Raw R2 against the number of neurons the decoder was trained on.

    Mean +/- s.e.m. across sessions at each neuron count.

    Args:
        real: Output of `load_neuron_count_panel`.
        regions: Regions to draw, in legend order.
        ax: Axes to draw into.
        xlim: (min, max) neuron count.
        ylim: (bottom, top) for the R2 axis.
        title: Axes title, or None.
        xticks, yticks: Tick positions, or None for automatic.
        palette: Anything seaborn accepts; defaults to the region colours.
        legend: Draw the region legend.

    Returns:
        The axes.
    """
    sns.lineplot(data=real, x='n_neurons', y=R2, hue='region',
                 hue_order=regions, errorbar='se', ax=ax,
                 palette=PALETTE if palette is None else palette)
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    if xticks is not None:
        ax.set_xticks(xticks)
    if yticks is not None:
        ax.set_yticks(yticks)
    ax.set_xlabel('Number of Neurons')
    ax.set_ylabel('R2')
    if title is not None:
        ax.set_title(title)
    if legend:
        ax.legend(frameon=True, loc='upper left')
    elif ax.get_legend() is not None:
        ax.get_legend().remove()
    sns.despine(ax=ax)
    return ax