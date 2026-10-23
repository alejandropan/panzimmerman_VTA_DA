"""
The thirteen decoders behind the paper's population-decoding panels.

Every per-decoder difference lives in this file and nowhere else.
`run_decoder_job.py` reads one entry and runs it; `launch_decoders.py` reads
the same entry to build the job array; `compile_decoders.py` reads it to know
which CSVs to write. Adding a decoder means adding a row here, and nothing
else.

This replaces thirteen near-identical worker scripts that differed only in the
fields below. Two things the originals did by hand are now fields:

  * the OUTPUT FOLDER. The originals all wrote to `<base>/location_level`
    (from `types[int(sys.argv[2])]`) and the final folder names -
    `location_level_all`, `location_level_skips` - were produced by renaming
    the directory after each run. Here the folder is declared, so the "_all"
    and "_skips" variants can run at the same time and nothing is renamed.
  * the SESSION SET. `master_decoder_vcs.py` switched it with a `SKIP_ONLY`
    flag that had to be edited between runs. Here it is `sessions`.

Read-only: nothing in this module touches disk or runs a decoder.
"""

import sys
from pathlib import Path

import numpy as np

# The analysis folder holds `ephys_alf_summary`. Put it on the path here so
# that importing this module is enough - `compile_decoders.py` and anything
# else that reads the table then works without PYTHONPATH being set. The
# entry-point scripts repeat this block deliberately: each has to work on its
# own, whatever order its imports end up in.
ANALYSIS_CODE_DIR = '/jukebox/witten/Alex/PYTHON/rewardworld/ephys/analysis'
if ANALYSIS_CODE_DIR not in sys.path:
    sys.path.append(ANALYSIS_CODE_DIR)

from ephys_alf_summary import LASER_ONLY_ABB                     # noqa: E402

# ============================================================================
# Paths
# ============================================================================
ROOT = '/jukebox/witten/Alex/Data/Subjects/'
ROOT_NEURAL = '/jukebox/witten/Chris/data/ibl_da_neuropixels/Data/Subjects'
DECODERS_CONTRA = '/jukebox/witten/Alex/decoders_contra_results_mua'
DECODERS_REWARDED = '/jukebox/witten/Alex/decoders_rewarded_results_mua'
NULL_SESSIONS = '/jukebox/witten/Alex/null_sessions'

# ============================================================================
# Session sets
# ============================================================================
def sessions_with_skips(sessions=LASER_ONLY_ABB, min_skips=5):
    """
    The sessions the Seq-AC "skips" decoders run on: those where the animal
    left at least `min_skips` earned rewards unconsumed.

    Derived rather than listed. Verified to reproduce exactly the 24
    sessions that ran, which is what the `.npy` outputs on disk and the
    `id_dict.csv` indices correspond to.

    Read straight from the trial files rather than through `alf`, which loads
    `feedbackTypeConsumed` inside the same `try` as `first_laser_times` and so
    leaves the attribute unset when the laser file is missing.

    Args:
        sessions: Candidate sessions, relative to the Subjects root.
        min_skips: Minimum unconsumed rewards to qualify.

    Returns:
        The qualifying sessions, in the order given.
    """
    qualifying = []
    for session in sessions:
        alf_dir = Path(ROOT) / session.strip('/') / 'alf'
        feedback = np.load(alf_dir / '_ibl_trials.feedbackType.npy')
        consumed = np.load(alf_dir / '_ibl_trials.feedbackTypeConsumed.npy')
        if np.sum((feedback > 0) & (consumed <= 0)) >= min_skips:
            qualifying.append(session)
    return qualifying


def resolve_sessions(which):
    """
    Turn a spec's `sessions` key into the actual list.

    Resolved on demand, not at import: only the launcher needs it, and
    `sessions_with_skips` reads 82 files. An array task looks its session up
    in `id_dict.csv` and never calls this.

    Args:
        which: 'all' or 'with_skips'.

    Returns:
        Session paths relative to the Subjects root.
    """
    if which == 'all':
        return list(LASER_ONLY_ABB)
    if which == 'with_skips':
        return sessions_with_skips()
    raise ValueError(f"unknown session set {which!r}")

# ============================================================================
# Area sets
# ============================================================================
# Which regions get an array task. The two masters differed in which location
# map they read, which is why only the Q decoders cover cortex and pallidum.
#
#   'striatal'  -> cluster_level_locations only. `level_map` is defined over
#                  striatal sub-labels alone, so everything else maps to NaN
#                  and drops out, leaving NAc / Rostral / Caudal / Int.
#                  NOTE it also leaves the literal string 'excluded' (ACB, CP,
#                  STR and the Calleja island all map to it), which the
#                  originals ran as if it were a region. Kept, because the
#                  .npy files for it exist and the array indices must line up
#                  with what ran. It is dropped at plotting time.
#   'striatal_cortical_pallidal'
#               -> cluster_group_locations AND cluster_level_locations,
#                  then filtered to the ten named regions.
STRIATAL_AND_CORTICAL_REGIONS = ['NAc', 'Rostral', 'Caudal', 'Int',
                                 'MO', 'OFC', 'PFC', 'SS', 'VP', 'Gpe']

# ============================================================================
# Constants that are the same for every decoder
# ============================================================================
REAL_NEURON_COMBOS = [10, 20, 30, 40, 50, 60]   # hence summary_102030405060.csv
N_NULLS = 200
MIN_TRIAL_FRACTION = 0.8      # a neuron needs trials on 80% of the session
LOCATION_TYPE = 'location_level'


# ============================================================================
# Variable builders
# ============================================================================
# Each returns `(regressed_variable, trial_filter)` - exactly the pair the
# original worker handed to `run_decoder_for_session_residual`.
#
# When the pair is a two-element LIST, the decoder reads element `h` for
# hemisphere `h`: element 0 is what hemisphere 0 gets. That is how a decoder
# is put into contra/ipsi space.

def q_cue(alfio, q_first, trials_first):
    """
    The four Q decoders: Q-contra or Q-ipsi, on contralateral or ipsilateral
    choice trials. Cue-aligned, previous trial's value.

    Args:
        alfio: Loaded session.
        q_first: 'contra' gives hemisphere 0 the RIGHT Q, which is its
            contralateral one. 'ipsi' gives it the left Q.
        trials_first: 'contra' gives hemisphere 0 the right choices.

    Returns:
        (variable, trial_filter), both two-element lists indexed by hemisphere.
    """
    qr = np.copy(np.roll(alfio.fQR, 1))
    ql = np.copy(np.roll(alfio.fQL, 1))
    qr[0] = 0
    ql[0] = 0
    right, left = np.where(alfio.choice == 1)[0], np.where(alfio.choice == -1)[0]
    return ([qr, ql] if q_first == 'contra' else [ql, qr],
            [right, left] if trials_first == 'contra' else [left, right])


def q_cue_null(null, alfio, q_first, trials_first):
    """
    The same pair from a null session, truncated to the real session's length.

    Args:
        null: One null-session table.
        alfio: The real session, for the lengths to truncate to.
        q_first, trials_first: As in `q_cue`.

    Returns:
        (variable, trial_filter).
    """
    qr = np.roll(np.copy(null['fQR'].to_numpy()), 1)
    ql = np.roll(np.copy(null['fQL'].to_numpy()), 1)
    qr[0] = 0
    ql[0] = 0
    n = len(alfio.fQR)
    n_right = len(np.where(alfio.choice == 1)[0])
    n_left = len(np.where(alfio.choice == -1)[0])
    right = np.where(null.choice == 1)[0][:n_right]
    left = np.where(null.choice == -1)[0][:n_left]
    return ([qr[:n], ql[:n]] if q_first == 'contra' else [ql[:n], qr[:n]],
            [right, left] if trials_first == 'contra' else [left, right])


# The next four put a single signed quantity into contra/ipsi space, and they
# are kept apart rather than shared because the two decoders build the
# negated member DIFFERENTLY, and the difference is visible in the bytes.
#
#   relative value  subtracts twice, `qr - ql` and `ql - qr`
#   pi_choice       negates once, `-1 * x`
#
# Those agree everywhere except at a zeroed first trial, where `-1 * 0.0` is
# NEGATIVE zero and `0.0 - 0.0` is positive zero. Numerically identical - no
# arithmetic can tell them apart - but not the same float, and reproducing
# what ran is the whole point. `test_worker_equivalence.py` caught exactly
# this when the two shared one helper.

def relative_value_cue(alfio):
    """
    Relative value, `Q_contra - Q_ipsi`, on contralateral choice trials.

    Args:
        alfio: Loaded session.

    Returns:
        (variable, trial_filter), both indexed by hemisphere.
    """
    qr = np.copy(np.roll(alfio.fQR, 1))
    ql = np.copy(np.roll(alfio.fQL, 1))
    qr[0] = 0
    ql[0] = 0
    return ([qr - ql, ql - qr],
            [np.where(alfio.choice == 1)[0], np.where(alfio.choice == -1)[0]])


def relative_value_cue_null(null, alfio):
    """
    The null counterpart of `relative_value_cue`.

    The pair comes out REVERSED relative to the real one - `[ql-qr, qr-ql]`
    against `[qr-ql, ql-qr]` - which is what `cue_delta_q.py` did. It changes
    nothing the decoder reports: the two members differ only by a sign,
    Lasso's solution negates exactly with `y`, and r2_score, pearsonr and even
    the CV folds are invariant (StratifiedKFold encodes classes by order of
    first appearance, which negation leaves alone). Tested in
    `check_sign_invariance.py`; every difference came out 0.000e+00. The one
    visible trace is that the saved null coefficients carry the opposite sign
    from the real ones, which no published panel reads.

    Note the first trial is NOT zeroed here, though it is in the real case.
    Also inherited.

    Args:
        null: One null-session table.
        alfio: The real session, for the lengths to truncate to.

    Returns:
        (variable, trial_filter).
    """
    qr = np.roll(np.copy(null['fQR'].to_numpy()), 1)
    ql = np.roll(np.copy(null['fQL'].to_numpy()), 1)
    n = len(alfio.fQR)
    return ([(ql - qr)[:n], (qr - ql)[:n]],
            [np.where(null.choice == 1)[0][:len(np.where(alfio.choice == 1)[0])],
             np.where(null.choice == -1)[0][:len(np.where(alfio.choice == -1)[0])]])


def propensity_cue(alfio):
    """
    The Seq-AC choice propensity, on contralateral choice trials.

    Args:
        alfio: Loaded session.

    Returns:
        (variable, trial_filter), both indexed by hemisphere.
    """
    x = np.copy(np.roll(alfio.seq_ac_propensity, 1))
    return ([x, -1 * x],
            [np.where(alfio.choice == 1)[0], np.where(alfio.choice == -1)[0]])


def propensity_cue_null(null, alfio):
    """
    The null counterpart of `propensity_cue`, pair reversed as in
    `cue_seq_pi.py`. See `relative_value_cue_null` for why that is harmless.

    Args:
        null: One null-session table.
        alfio: The real session, for the lengths to truncate to.

    Returns:
        (variable, trial_filter).
    """
    rl = np.roll(np.copy(null['seq_ac_propensity'].to_numpy()), 1)
    lr = -1 * np.roll(np.copy(null['seq_ac_propensity'].to_numpy()), 1)
    n = len(alfio.seq_ac_propensity)
    return ([lr[:n], rl[:n]],
            [np.where(null.choice == 1)[0][:len(np.where(alfio.choice == 1)[0])],
             np.where(null.choice == -1)[0][:len(np.where(alfio.choice == -1)[0])]])


def scalar_on_rewarded(alfio, attribute):
    """
    One scalar per trial, on rewarded trials, outcome-aligned. No hemisphere
    split - `run_decoder_for_session_residual` gives both hemispheres the same
    variable.

    Args:
        alfio: Loaded session.
        attribute: Name on `alfio` to read.

    Returns:
        (variable, trial_filter).
    """
    x = np.copy(np.roll(getattr(alfio, attribute), 1))
    x[0] = 0
    return x, np.where(alfio.outcome == 1)[0]


def scalar_on_rewarded_null(null, alfio, column):
    """
    The null counterpart of `scalar_on_rewarded`.

    The first trial is NOT zeroed here, though it is in the real case. That
    asymmetry is inherited from the original workers and kept deliberately.

    Args:
        null: One null-session table.
        alfio: The real session, for the length to truncate the filter to.
        column: Column of `null` to read.

    Returns:
        (variable, trial_filter).
    """
    x = np.roll(np.copy(null[column].to_numpy()), 1)
    n_rewarded = len(np.where(alfio.outcome == 1)[0])
    return x, np.where(null.outcome == 1)[0][:n_rewarded]


# ============================================================================
# The table
# ============================================================================
# Fields:
#   folder        where the .npy files and the two summary CSVs go
#   sessions      'all' (the 41 laser-only sessions) or 'with_skips'
#                 (those with >=5 unconsumed rewards); `resolve_sessions`
#                 turns it into a list when the launcher needs one
#   areas         which location map the launcher reads
#   alignment     goCue_time or response_time
#   filetype      what load_all_residuals reads: 'real' is raw spikes,
#                 'residual_crac_*' the CRAC model's residuals
#   criterion     None keeps good + mua units, 'good' keeps only good
#   real          (alfio) -> (variable, trial_filter)
#   null          (null_table, alfio) -> (variable, trial_filter)
#   null_folder   which of the two null-session sets to draw from. Follows
#                 `sessions`: the nulls are built from the same recordings the
#                 decoder ran on, so 'with_skips' pairs with laser_only_skips
#                 and 'all' with laser_only. Asserted below.
#   null_n        neurons per null draw. 40 on residuals, 20 on raw spikes -
#                 asserted below.
#   panels        which published panels this feeds, for the manifest

def _q(name, q_first, trials_first):
    """One of the four Q decoders. They differ only in these two switches."""
    return dict(
        folder=f'{DECODERS_CONTRA}/decoder_output_{name}_cue_forget/{LOCATION_TYPE}',
        sessions='all', areas='striatal_cortical_pallidal',
        alignment='goCue_time', filetype='real', criterion=None,
        real=lambda a: q_cue(a, q_first, trials_first),
        null=lambda n, a: q_cue_null(n, a, q_first, trials_first),
        null_folder='laser_only', null_n=20)


def _outcome_scalar(base, attribute, column, filetype, criterion, null_folder,
                    null_n, sessions='all', folder=None):
    """A scalar decoded on rewarded trials, outcome-aligned."""
    return dict(
        folder=folder or f'{base}/{LOCATION_TYPE}',
        sessions=sessions, areas='striatal',
        alignment='response_time', filetype=filetype, criterion=criterion,
        real=lambda a: scalar_on_rewarded(a, attribute),
        null=lambda n, a: scalar_on_rewarded_null(n, a, column),
        null_folder=null_folder, null_n=null_n)


SPECS = {
    # --- the four Q decoders, Fig 2k-n / ED4f-i / ED5a-d --------------------
    'qcontra_contra': {**_q('qcontra_contra', 'contra', 'contra'),
                       'panels': ['2k', 'ED4f', 'ED5a']},
    'qipsi_contra':   {**_q('qipsi_contra', 'ipsi', 'contra'),
                       'panels': ['2l', 'ED4h', 'ED5b']},
    'qcontra_ipsi':   {**_q('qcontra_ipsi', 'contra', 'ipsi'),
                       'panels': ['2m', 'ED4g', 'ED5d']},
    'qipsi_ipsi':     {**_q('qipsi_ipsi', 'ipsi', 'ipsi'),
                       'panels': ['2n', 'ED4i', 'ED5c']},

    # --- relative value, Fig 3f / ED7i -------------------------------------
    'relative_value': dict(
        folder=f'{DECODERS_CONTRA}/decoder_output_deltaq_cue_forget/{LOCATION_TYPE}',
        sessions='all', areas='striatal',
        alignment='goCue_time', filetype='real', criterion=None,
        real=relative_value_cue,
        null=relative_value_cue_null,
        null_folder='laser_only', null_n=20,
        panels=['3f', 'ED7i']),

    # --- the three state-value decoders, Fig 3g / ED7j,p,q -----------------
    # `fV` is the state value. The folders say "totalq", which is a misnomer
    # kept because that is their name on disk; `unEQ` and `wEQ` really are
    # different variables (the unweighted and weighted sums of Qs).
    'state_value': {**_outcome_scalar(
        f'{DECODERS_REWARDED}/decoder_output_totalq_outcome_forget',
        'fV', 'fV', 'real', None, 'laser_only', 20),
        'panels': ['3g', 'ED7j']},
    'state_unweighted': {**_outcome_scalar(
        f'{DECODERS_REWARDED}/decoder_output_totalq_unweighted_outcome_forget',
        'unEQ', 'unEQ', 'real', None, 'laser_only', 20),
        'panels': ['ED7p']},
    'state_weighted': {**_outcome_scalar(
        f'{DECODERS_REWARDED}/decoder_output_totalq_weighted_outcome_forget',
        'wEQ', 'wEQ', 'real', None, 'laser_only', 20),
        'panels': ['ED7q']},

    # --- the Seq-AC value decoders, Fig 6k,l,p,q ---------------------------
    # Each ran twice: once over every session with the matching null set,
    # once over the skip sessions with the skips nulls. Both wrote to the same
    # folder and were told apart by renaming it afterwards, and the switch
    # needed two manual edits - `SKIP_ONLY` in the master and the null path
    # in the worker. Declaring `sessions` and `folder` here does both at once,
    # and the null set follows from `sessions` (checked below), so the two
    # can no longer disagree.
    'vcs': {**_outcome_scalar(
        None, 'seq_ac_vcs', 'seq_ac_vcs', 'residual_crac_vcs', 'good',
        'laser_only', 40,
        folder=f'{DECODERS_REWARDED}/decoder_output_vcs/rewarded_only/location_level_all'),
        'panels': ['6k']},
    'vcs_skips': {**_outcome_scalar(
        None, 'seq_ac_vcs', 'seq_ac_vcs', 'residual_crac_vcs', 'good',
        'laser_only_skips', 40, sessions='with_skips',
        folder=f'{DECODERS_REWARDED}/decoder_output_vcs/rewarded_only/location_level_skips'),
        'panels': ['6l']},
    'vchoice': {**_outcome_scalar(
        None, 'seq_ac_vchoice', 'seq_ac_vchoice', 'residual_crac_vchoice',
        'good', 'laser_only', 40,
        folder=f'{DECODERS_REWARDED}/decoder_output_vchoice/rewarded_only/location_level_all'),
        'panels': ['6p']},
    'vchoice_skips': {**_outcome_scalar(
        None, 'seq_ac_vchoice', 'seq_ac_vchoice', 'residual_crac_vchoice',
        'good', 'laser_only_skips', 40, sessions='with_skips',
        folder=f'{DECODERS_REWARDED}/decoder_output_vchoice/rewarded_only/location_level_skips'),
        'panels': ['6q']},

    # --- the choice propensity, ED12d --------------------------------------
    # RECONSTRUCTED, not read from a script. The published panel is an earlier
    # run on RAW spikes with 20-neuron nulls; the surviving `cue_seq_pi.py` is
    # a later version reading `residual_crac_pichoice` with 40, and is not
    # what the figure shows. The variable and trial filter are taken from that
    # later script, which is the only record of them.
    'pi_choice': dict(
        folder=f'{DECODERS_CONTRA}/decoder_output_seqpi/{LOCATION_TYPE}',
        sessions='all', areas='striatal',
        alignment='goCue_time', filetype='real', criterion=None,
        real=propensity_cue,
        null=propensity_cue_null,
        null_folder='laser_only', null_n=20,
        panels=['ED12d']),
}


# ============================================================================
# Validation - runs at import, so a bad entry fails before any job is queued
# ============================================================================
RESIDUAL_FILETYPES = {'residual_crac_vcs', 'residual_crac_vchoice',
                      'residual_crac_pichoice'}
REQUIRED = {'folder', 'sessions', 'areas', 'alignment', 'filetype', 'criterion',
            'real', 'null', 'null_folder', 'null_n', 'panels'}

for _name, _spec in SPECS.items():
    _missing = REQUIRED - set(_spec)
    if _missing:
        raise ValueError(f"spec '{_name}' is missing {sorted(_missing)}")
    if _spec['alignment'] not in ('goCue_time', 'response_time'):
        raise ValueError(f"spec '{_name}': unknown alignment "
                         f"{_spec['alignment']!r}")
    if _spec['sessions'] not in ('all', 'with_skips'):
        raise ValueError(f"spec '{_name}': unknown session set "
                         f"{_spec['sessions']!r}")
    if _spec['areas'] not in ('striatal', 'striatal_cortical_pallidal'):
        raise ValueError(f"spec '{_name}': unknown area set {_spec['areas']!r}")
    if _spec['null_folder'] not in ('laser_only', 'laser_only_skips'):
        raise ValueError(f"spec '{_name}': unknown null set "
                         f"{_spec['null_folder']!r}")
    # The null distribution is built from the same recordings the decoder ran
    # on, so the two sets are locked together. In the original scripts this
    # took two separate hand edits - the master's SKIP_ONLY flag and the null
    # path inside the worker - and nothing stopped them disagreeing.
    _expected_nulls = ('laser_only_skips'
                       if _spec['sessions'] == 'with_skips' else 'laser_only')
    if _spec['null_folder'] != _expected_nulls:
        raise ValueError(
            f"spec '{_name}': runs on "
            f"the {_spec['sessions']!r} session set "
            f"but draws nulls from {_spec['null_folder']!r}; expected "
            f"{_expected_nulls!r}. The nulls must come from the same "
            "recordings the decoder ran on.")
    # The rule: nulls run 40 neurons on residuals and 20 on raw spikes. It is
    # written out per spec so a reader sees what ran, and checked here so a
    # new entry cannot silently get the wrong one.
    _expected = 40 if _spec['filetype'] in RESIDUAL_FILETYPES else 20
    if _spec['null_n'] != _expected:
        raise ValueError(
            f"spec '{_name}': null_n is {_spec['null_n']} but filetype "
            f"{_spec['filetype']!r} implies {_expected}. Nulls run 40 neurons "
            "on CRAC residuals and 20 on raw spikes.")

_folders = [s['folder'] for s in SPECS.values()]
if len(set(_folders)) != len(_folders):
    raise ValueError("two specs write to the same folder; they would overwrite "
                     "each other's .npy files")


# ============================================================================
# Reading the table
# ============================================================================
def summary():
    """
    Print the thirteen specs, one line each.

    Not run on import - the worker imports this module once per array task
    and should stay quiet. Call it to review the table:

        python -c "import decoder_specs; decoder_specs.summary()"
    """
    print(f"{len(SPECS)} decoders\n")
    header = (f"{'spec':17s} {'panels':16s} {'sessions':10s} {'areas':10s} "
              f"{'align':9s} {'input':22s} {'units':5s} {'nulls':17s} {'n':>3s}")
    print(header)
    print('-' * len(header))
    for name, spec in SPECS.items():
        print(f"{name:17s} {','.join(spec['panels']):16s} "
              f"{spec['sessions']:10s} "
              f"{'str+ctx' if spec['areas'].startswith('striatal_c') else 'striatal':10s} "
              f"{spec['alignment'].replace('_time', ''):9s} "
              f"{spec['filetype']:22s} "
              f"{'good' if spec['criterion'] == 'good' else '+mua':5s} "
              f"{spec['null_folder']:17s} {spec['null_n']:>3d}")
    print(f"\nreal sweep {REAL_NEURON_COMBOS} and {N_NULLS} nulls for all of them")
    print("\noutput folders:")
    for name, spec in SPECS.items():
        print(f"  {name:17s} {spec['folder']}")
