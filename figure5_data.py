"""
Shared machinery for Figure 5 and Extended Data Figure 10.

Three independent datasets feed those two figures, and all three are loaded
here so neither figure script contains any data handling:

    the laser-only bandit sessions      Fig 5d, 5e, 5f and ED Fig 10a
    the Stan posteriors                 ED Fig 10b, 10c
    the no-CS+ control sessions         Fig 5h

Nothing here plots or saves anything except `apply_style` and `save_panel`,
which exist so the two figure scripts cannot drift apart.
"""

import os
from pathlib import Path
import warnings
import matplotlib
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import font_manager
from scipy.stats import zscore

from model_comparison_accu import load_data_reduced, make_stan_data_reduced
from ephys_alf_summary import alf

# %%
# ============================================================================
# Paths
# ============================================================================
ON_CLUSTER = os.name != "nt"
WITTEN = Path("/jukebox/witten") if ON_CLUSTER else Path(r"\\cup.princeton.edu\witten")

RELEASE = (Path(__file__).resolve().parent if "__file__" in globals()
           else Path.cwd())
SAVE_DIR = RELEASE

BANDIT_DATA = WITTEN / "Alex" / "Data" / "ephys_bandit" / "data_laser_only"
STAN_FIT = (WITTEN / "Alex" / "Data" / "ephys_bandit" / "laser_stan_fits"
            / "standard_w_forgetting_cs_s")
SUBJECTS = WITTEN / "Alex" / "Data" / "Subjects"

# %%
# ============================================================================
# Configuration
# ============================================================================
# A session needs this many VTA stimulations skipped for statistics
MIN_UNCONSUMED_CS = 5

# Trials kept 10 trials before first blockchange and onwards,
# this prevents artifacts related to spout adjustment during the first few trials.
# In reality having this or not is statistically insignificant.
LEAD_IN_TRIALS = 10

# The three outcome conditions of Fig 5e and ED Fig 10a, in published order
# for each. They are the same three conditions ordered differently, which is
# why both orders are written out rather than one being derived.
CONDITION_LABELS = {
    'after_error': 'CS-',
    'after_laser': 'CS+ w/\nVTA stim',
    'after_cs_no_laser': 'CS+ w/o\nVTA stim',
}
RETURN_ORDER = ['after_error', 'after_laser', 'after_cs_no_laser']   # Fig 5e

# Cyan marks a condition in which the CS+ was present; grey its absence.
CYAN = '#4DC3E8'
GREY = '#8C8C8C'
BLACK = 'k'

# The paired no-CS+ experiment. Each row is one animal: the control session
# and, the day after, the same task with the CS+ omitted.
SESSIONS_CONTROL = [
    SUBJECTS / 'dop_67' / '2024-07-04' / '001',
    SUBJECTS / 'dop_69' / '2024-07-04' / '001',
    SUBJECTS / 'dop_68' / '2024-07-09' / '001',
    SUBJECTS / 'dop_65' / '2024-07-15' / '001',
    SUBJECTS / 'dop_61' / '2024-07-31' / '001',
    SUBJECTS / 'dop_63' / '2024-07-24' / '001',
]
SESSIONS_NOCS = [
    SUBJECTS / 'dop_67' / '2024-07-05' / '001',
    SUBJECTS / 'dop_69' / '2024-07-05' / '001',
    SUBJECTS / 'dop_68' / '2024-07-10' / '001',
    SUBJECTS / 'dop_65' / '2024-07-16' / '001',
    SUBJECTS / 'dop_63' / '2024-07-25' / '001',
    SUBJECTS / 'dop_61' / '2024-08-01' / '001',
]
if len(SESSIONS_CONTROL) != len(SESSIONS_NOCS):
    raise ValueError("the control and no-CS+ session lists must pair up")


# %%
# ============================================================================
# Style and output
# ============================================================================
def apply_style():
    """The figure style both scripts use."""
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
        save_dir: Destination. Defaults to `SAVE_DIR`, read at call time so
            that pointing the module elsewhere takes effect.
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
# The laser-only bandit sessions
# ============================================================================
CACHE_DIR = RELEASE / "data"


def load_behaviour():
    """
    Every laser-only session's trials, labelled by session.

    The first trial is a dud and therfore excluded.

    Returns:
        The trial table, with an `id` column of mouse+date+session.
    """
    cache = CACHE_DIR / "figure5" / "behaviour.csv.gz"
    if cache.exists():
        return pd.read_csv(cache)
    data = load_data_reduced(ROOT_FOLDER=str(BANDIT_DATA), trial_start=0,
                             trial_end=None)
    data['id'] = data.mouse + data.date + data.ses
    data.loc[data.index == 0, 'decision_times'] = np.nan
    data.loc[data.index == 0, 'r_times'] = np.nan
    return data


def build_transition_data(data):
    """
    Per-trial outcome conditions, with each session trimmed to its first block.

    Columns added:
        repeat             did the animal repeat the previous choice
        z_decision_times   go cue to decision, z-scored within session
        after_cs           the previous trial delivered a CS+
        after_laser        the previous trial's CS+ was collected
        after_cs_no_laser  the previous trial's CS+ was NOT collected
        after_error        the previous trial delivered no CS+
        unconsumed_cs+     this trial's CS+ was not collected

    Args:
        data: Output of `load_behaviour`.

    Returns:
        The concatenated per-session tables, index reset.
    """
    sessions = []
    for session_id in data.id.unique():
        session = data.loc[data['id'] == session_id].copy()
        block_change = np.where(np.diff(session.probabilityLeft) != 0)[0] + 1

        session['repeat'] = 1 * (session.choices.to_numpy()
                                 == session.choices.shift(1).to_numpy())
        session['z_decision_times'] = zscore(session.decision_times,
                                             nan_policy='omit')
        session['after_cs'] = session.reward.shift(1)
        session['after_laser'] = session.laser_consumed.shift(1)
        session['after_cs_no_laser'] = ((session['after_laser'] == 0)
                                        * session['after_cs'])
        session['after_error'] = 1 * (session['after_cs'] == 0)

        # Sessions with no block change at all, and those whose first change
        # comes before the lead-in is available, are kept whole.
        if len(block_change) > 0 and block_change[0] - LEAD_IN_TRIALS >= 0:
            session = session.iloc[block_change[0] - LEAD_IN_TRIALS:]
        sessions.append(session)

    transition_data = pd.concat(sessions).reset_index(drop=True)
    transition_data['unconsumed_cs+'] = (transition_data.reward
                                         * (transition_data.laser_consumed == 0))
    return transition_data


def sessions_with_enough_unconsumed(transition_data,
                                    minimum=MIN_UNCONSUMED_CS):
    """
    The sessions in which the animal left at least `minimum` CS+ uncollected.

    Below that, the "CS+ without VTA stim" condition rests on too few trials
    to mean anything, so Fig 5e and ED Fig 10 are restricted to these.

    Args:
        transition_data: Output of `build_transition_data`.
        minimum: Unconsumed CS+ trials a session needs.

    Returns:
        The qualifying session ids.
    """
    per_session = transition_data.groupby('id')['unconsumed_cs+'].sum()
    return per_session.loc[per_session >= minimum].index.to_numpy()


def return_probabilities(transition_data, sessions=None):
    """
    Probability of repeating the previous choice, per session and condition.

    Args:
        transition_data: Output of `build_transition_data`.
        sessions: Restrict to these session ids, or None for all.

    Returns:
        Long table of `id`, `after`, `repeat`, with `after` an ordered
        category in `RETURN_ORDER`.
    """
    parts = []
    for condition in RETURN_ORDER:
        per_session = (transition_data.groupby([condition, 'id'])['repeat']
                       .mean().reset_index())
        per_session = per_session.loc[per_session[condition] == 1].copy()
        per_session['after'] = condition
        parts.append(per_session[['id', 'after', 'repeat']])

    merged = pd.concat(parts, ignore_index=True)
    if sessions is not None:
        merged = merged.loc[np.isin(merged.id, sessions)].copy()

    covered = {name: set(part['id']) for name, part in merged.groupby('after')}
    if len({frozenset(s) for s in covered.values()}) != 1:
        counts = {name: len(s) for name, s in covered.items()}
        everywhere = set.intersection(*covered.values())
        missing = {name: sorted(s - everywhere)
                   for name, s in covered.items() if s - everywhere}
        raise ValueError(f"conditions cover different sessions: {counts}. "
                         f"Not in all three: {missing}")

    merged['after'] = pd.Categorical(merged['after'], categories=RETURN_ORDER,
                                     ordered=True)
    return merged.sort_values(['after', 'id']).reset_index(drop=True)


def mouse_of(session_id):
    """
    The animal a session id belongs to.

    Ids are mouse+date+session and every date starts with '20', so splitting
    there recovers the mouse. Inherited from the original.
    """
    return session_id.split('20')[0]


# %%
# ============================================================================
# The Stan posteriors
# ============================================================================
# Rows of the CmdStan CSV that are not draws: the configuration header, the
# column-name block and the timing footer. Counted from the files these fits
# produced - a different sampler configuration would move them.
STAN_HEADER_ROWS = np.concatenate([np.arange(47), np.arange(48, 52),
                                   np.arange(1551, 1557)])

# The model was fitted with a fixed slot per (mouse, session); this is how
# many sessions each mouse has room for.
SESSIONS_PER_MOUSE = 8


def load_chains():
    """
    The four CmdStan chains, concatenated.

    Takes the most recent file matching each chain number, which is how the
    fits were read when the figure was made.

    Returns:
        Every draw from every chain, with a `chain` column.
    """
    import glob
    output = []
    for number in (1, 2, 3, 4):
        matches = sorted(glob.glob(str(STAN_FIT / 'output' / f'*{number}.csv')))
        if not matches:
            raise FileNotFoundError(
                f"no CmdStan output for chain {number} in {STAN_FIT / 'output'}")
        chain = pd.read_csv(matches[-1], skiprows=STAN_HEADER_ROWS)
        chain['chain'] = number
        output.append(chain)
    return pd.concat(output)


def cs_value_posteriors(minimum=MIN_UNCONSUMED_CS):
    """
    The fitted CS+-without-VTA-stim value, for sessions that can support it.

    One column per session, every draw from every chain as rows. A session
    enters only if the animal left at least `minimum` CS+ uncollected in it -
    otherwise the parameter is not identified by anything.

    Args:
        minimum: Unconsumed CS+ trials a session needs.

    Returns:
        `(posteriors, selected)`: the per-session draws, and the Stan slot
        numbers they came from.
    """
    cache = CACHE_DIR / "extended_data_figure10" / "cs_value_posteriors.csv.gz"
    if cache.exists():
        posteriors = pd.read_csv(cache)
        return posteriors, [int(c.split('.')[1]) for c in posteriors.columns]
    standata = make_stan_data_reduced(
        load_data_reduced(ROOT_FOLDER=str(BANDIT_DATA), trial_start=0,
                          trial_end=None))
    chains = load_chains()

    selected = []
    for slot in np.arange(standata['NSxNSESS']):
        if standata['NT_all'][slot] <= 0:
            continue
        mouse, session = int(slot / SESSIONS_PER_MOUSE), int(slot % SESSIONS_PER_MOUSE)
        unconsumed = sum((standata['l'][mouse][session] == 0)
                         * (standata['r'][mouse][session] == 1))
        if unconsumed >= minimum:
            # +1 because the Stan columns are 1-indexed.
            selected.append(slot + 1)

    columns = [f'laser_ses.{slot}' for slot in selected]
    missing = [c for c in columns if c not in chains.columns]
    if missing:
        raise KeyError(f"{len(missing)} session parameter(s) absent from the "
                       f"CmdStan output, first {missing[0]}")
    print(f"    {len(selected)} session(s) with >= {minimum} unconsumed CS+")
    return chains[columns], selected


# %%
# ============================================================================
# The no-CS+ control sessions
# ============================================================================
def load_nocs_experiment():
    """
    The paired control / no-CS+ sessions and their fitted parameters.

    Each animal contributes one session of the standard task and one the next
    day with the CS+ omitted. 

    Returns:
        `(behaviour, parameters)`. Both carry `cond` ('control' or 'no_cs+')
        and `mouse`.
    """
    cache = CACHE_DIR / "figure5" / "nocs.csv.gz"
    cache_params = CACHE_DIR / "figure5" / "nocs_params.csv.gz"
    if cache.exists() and cache_params.exists():
        return pd.read_csv(cache), pd.read_csv(cache_params)
    behaviour, parameters = [], []
    for index, (control, nocs) in enumerate(zip(SESSIONS_CONTROL,
                                                SESSIONS_NOCS)):
        pair = []
        for session, condition in ((control, 'control'), (nocs, 'no_cs+')):
            params = pd.read_pickle(str(session / 'alf'
                                        / 'forgettingmodel_params.pkl'))
            params['cond'] = condition
            trials = alf(str(session), ephys=False).to_df()
            trials['cond'] = condition
            pair.append((params, trials))

        animal_params = pd.concat([p for p, _ in pair]).copy()
        animal_trials = pd.concat([t for _, t in pair]).copy()
        animal_params['mouse'] = str(index)
        animal_trials['mouse'] = str(index)
        parameters.append(animal_params)
        behaviour.append(animal_trials)

    behaviour = pd.concat(behaviour)
    parameters = pd.concat(parameters)

    # Did the animal choose the side that was currently rewarded more often?
    behaviour['p(high)'] = 0
    behaviour.loc[(behaviour.probabilityLeft == 0.7)
                  & (behaviour.choice == -1), 'p(high)'] = 1
    behaviour.loc[(behaviour.probabilityLeft == 0.1)
                  & (behaviour.choice == 1), 'p(high)'] = 1

    parameters['abs_bias'] = abs(parameters.side_mouse)
    return behaviour.reset_index(), parameters
