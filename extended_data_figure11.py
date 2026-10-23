"""
Extended Data Figure 11 - The Conditioned Reinforcer model is recoverable and
reproduces the mice's behavior.

Panels produced, in manuscript order:
    11b  fitted initialization parameters
    11c  latent-trajectory recovery, Fisher-z mean per-session r
    11d  parameter recovery, behaviour fit against simulation fit
    11e  decision-variable similarity, Model 1 against Model 6
    11f  PSIS-LOO model comparison, all six models
    11g  win-stay: P(repeat) after a CS+ or a CS-, mice against the model
    11h  action selection: P(right) against the choice-propensity decile
    11i  block switches: P(high) against trials from the switch
    11j  predicted P(lick) on CS+ trials the mouse did and did not lick for
    11k  fitted V(CS) initialization against the fraction of CS+ skipped,
         drawn twice - against the simulated skip rate and the observed one
    11l  V(S_CS+) against the observed lick, binned within session
    11m  performance with the stimulation removed in silico
    11n  performance against trial number under each manipulation

Panel l is the only one that needs R: its primary statistic is an lme4::glmer
fit reached through rpy2. If rpy2 or lme4 is missing, that cell draws the
panel without the p-value in its title and says so, rather than stopping the
script - every other panel runs with Python alone.
"""

# %%
# ============================================================================
# Imports and configuration
# ============================================================================
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import warnings
from pathlib import Path
from scipy.stats import pearsonr, sem

from figure6_data import (BASELINE_TRIALS, FS_LABEL, FS_LEGEND, FS_TITLE,
                          LINE_WIDTH, MODEL_COLOR, MODEL_NAME, MODE_COLORS,
                          MOUSE_COLOR, PAPER_DIR, STAN_ROOT, SUBJECTS_ROOT,
                          WINDOW, apply_style, bracket,
                          inv_logit, lick_detect_wilcoxon,
                          load_mouse_sessions, load_pkl, load_recovery_data,
                          observed_skip_rate, paired_bar_panel,
                          paired_performance, param_violins, partial_corr,
                          regression_panel, run_manipulation_simulations,
                          save_panel, seqac_params_long, session_alf_path,
                          session_path, simulate_seqac, stan_grid, stars,
                          verify_grid)
from ephys_alf_summary import LASER_ONLY_ABB, alf

# One PDF per panel, beside the other figures of the paper.
SAVE_DIR = PAPER_DIR

# --- Panels c and d: the recovery fits --------------------------------------
# The fit to the animals' choices, whose posterior medians generated the
# simulated sessions, and the fit to those simulated sessions.
BEHAVIOUR_SUMMARY = STAN_ROOT / MODEL_NAME / "output" / "summary" / "summary.csv"

RECOVERY_DIR = STAN_ROOT / "recovery_models" / f"{MODEL_NAME}_recovery"
RECOVERY_SUMMARY = RECOVERY_DIR / "output" / "summary.csv"

# The simulated choices the recovery was fit to, written beside the fit.
SIM_FILES = {'c': 'c_sim.npy', 'o': 'o_sim.npy', 'l': 'l_sim.npy'}

# --- Panel f: the LOO comparison --------------------------------------------
# The six-model table. `compute_loo_comparison.py` writes both this and the
# five-model one Extended Data Fig. 2a uses; dropping a model changes no other
# model's elpd, se or dse, so the two are the same statistics plotted twice.
LOO_COMPARISON_CSV = (STAN_ROOT / "loo_results" / "loo_comparison_6models.csv")
LOO_XTICKS = [-11000, -10900, -10800]
# Both markers for a model sit at the same x, so the dSE triangle is nudged up
# by this fraction of the row spacing. 
DSE_MARKER_OFFSET = 0.29
COLOR_DSE = "#808080"

MODEL_LABELS = {
    1: "1. Extended Q-learning\n(CS+=Reward)",
    2: "2. Extended Q-learning\n(VTA Stim.=RPE)",
    3: "3. Extended Q-learning\n(CS+=Reward) -\nVariable α",
    4: "4. REINFORCE",
    5: "5. Actor-Critic",
    6: "6. Conditioned\nReinforcer",
}

# --- Panels b, c, d: parameters ---------------------------------------------
# Every parameter of the conditioned reinforcer model.
RECOVERY_PARAMS = [
    # Mouse level - panel d's top row, 8 points each.
    dict(name='alpha_v_choice_mouse', scope='mouse', squash=True,
         label=r'$\alpha_{V(S_{choice})}$'),
    dict(name='alpha_v_cs_mouse', scope='mouse', squash=True,
         label=r'$\alpha_{V(S_{CS+})}$'),
    dict(name='beta_lick_mouse', scope='mouse', squash=False,
         label=r'$\beta_{Lick}$'),
    dict(name='alpha_lick_mouse', scope='mouse', squash=True,
         label=r'$\alpha_{\pi(lick)}$'),
    # Session level - the middle and bottom rows, one point per session.
    dict(name='alpha_loss_ses', scope='session', squash=True,
         label=r'$\alpha_{Loss}$'),
    dict(name='beta_win_ses', scope='session', squash=False,
         label=r'$\beta_{Win}$'),
    dict(name='beta_loss_ses', scope='session', squash=False,
         label=r'$\beta_{Loss}$'),
    dict(name='side_ses', scope='session', squash=False, label='Bias'),
    dict(name='V_cs_init_ses', scope='session', squash=False,
         label=r'$V_0(S_{CS+})$'),
    dict(name='V_choice_init_ses', scope='session', squash=False,
         label=r'$V_0(S_{choice})$'),
    dict(name='pi_lick_init_ses', scope='session', squash=False,
         label=r'$\pi_0^{lick}$'),
]
# Parameters of the earlier model that must NOT be present. Their presence
# means the summary belongs to a different fit. In the final model we
# use these at the mouse level.
SUPERSEDED = ['alpha_win_ses', 'alpha_v_choice_ses'] 

D_COLUMNS = 4
D_GROUPS = [(0, 'Mouse Level\nParameters'), (1, 'Session Level\nParameters')]
POINT_COLOR = 'k'
IDENTITY_COLOR = '#B4B4B4'

# Panel c. The latents, in the order the published panel draws them.
LATENTS = [('V_cs', r'$V(S_{cs})$'), ('V_choice', r'$V(S_{choice})$'),
           ('pi_lick', r'$\pi(lick)$'), ('pi_choice', r'$\pi(choice)$')]
C_YLIM = (0, 1.08)
C_BAR_COLOR = '0.65'

# --- Panels e, h, i: panel constants ----------------------------------------
# Panel h splits the propensity into this many equal-count bins.
VALUE_BINS = 10

# Panel i: trials either side of a block switch, and how far back before a
# switch the preceding block contributes.
TRANSITION_XLIM = (-10, 25)
TRIALS_BEFORE_SWITCH = 10

# Panel e. The five decision variables.Model 6 is the Conditioned Reinforcer,
# Model 1 the extended Q-learning model; every column is read from the alf 
# table as saved, i.e. post-update, so they are mutually consistent.
DECISION_VARS = [
    ('Model 6:\n$\\pi$(choice)', lambda d: d['seq_ac_propensity'].to_numpy(float)),
    ('Model 6:\n$V(S_{CS+})$', lambda d: d['seq_ac_vcs'].to_numpy(float)),
    ('Model 6:\n$V(S_{choice})$', lambda d: d['seq_ac_vchoice'].to_numpy(float)),
    ('Model 1:\nRelative Value\n(QR-QL)', lambda d: (d['fQR'] - d['fQL']).to_numpy(float)),
    ('Model 1:\nState Value (V)', lambda d: d['V'].to_numpy(float)),
]
DECISION_VAR_COLS = ['seq_ac_propensity', 'seq_ac_vcs', 'seq_ac_vchoice',
                     'fQR', 'fQL', 'V']
DECISION_MIN_TRIALS = 10
DECISION_TITLE = ('Model 1: Extended Q-learning Model vs\n'
                  'Model 6: Conditioned Reinforcer Model')

# --- Panel l: V(S_CS+) against the observed lick -----------------------------
# The per-trial model arrays `extract_decision_variables` writes beside the
# behaviour in each session's alf folder, under this prefix.
L_PREFIX = 'SeqAC_noawin_m_avchoice_median_'
L_NBINS = 5              # quantile bins of V_cs, within session
L_MIN_BINS = 3           # a session needs this many distinct bins to survive
L_CENTER = True          # subtract each session's own mean from x and y first
# A session is kept only if the mouse forwent at least this many VTA stims.
L_MIN_MISSED_LICKS = 5
# The primary model for statistical analysis. 
L_FORMULA = '{y} ~ V_cs_z + (V_cs_z | mouse/session)'
L_FIGSIZE = (3.4, 3.2)

# --- Panel n: the decay windows ---------------------------------------------
# The two halves of the broken x-axis.
DECAY_LEFT_WINDOW = (0, 3000)
DECAY_RIGHT_WINDOW = (12000, 13000)
DECAY_LABELS = {'intact': 'Control', 'laser_off': 'Laser off',
                'no_cs': 'No CS+'}
CHANCE = 0.5

# %%
# ============================================================================
# Panel g - win-stay
# ============================================================================
def repeat_after_outcome(df):
    """
    Each session's P(repeat the last choice), split by the last outcome.

    Args:
        df: Trial table with `id`, `choice` and `outcome`.

    Returns:
        `id`, `after_cs`, `repeat`.
    """
    g = df.copy()
    previous = g.groupby('id')['choice'].shift(1)
    g['repeat'] = (g['choice'].to_numpy() == previous.to_numpy()).astype(float)
    g.loc[previous.isna(), 'repeat'] = np.nan
    g['after_cs'] = g.groupby('id')['outcome'].shift(1)
    g = g.dropna(subset=['after_cs', 'repeat'])
    return g.groupby(['id', 'after_cs'])['repeat'].mean().reset_index()


# %%
# ============================================================================
# Panel i - block switches
# ============================================================================
def chose_high_side(choice_like, prob_left):
    """Recode a right-choice indicator as a higher-reward-side indicator."""
    chose_right = np.asarray(choice_like, dtype=float)
    return np.where(np.asarray(prob_left) == 0.7, 1.0 - chose_right,
                    chose_right)


def trials_from_switch(df, value_col):
    """
    Re-index every session's trials by their distance from a block switch.

    Trials in the ten before a switch are given negative positions, so the
    panel can show the approach to the switch as well as the recovery after
    it. The first block is dropped up to its own lead-in.

    Args:
        df: Trial table with `id`, `probabilityLeft` and `value_col`.
        value_col: The right-choice indicator to recode as `high`.

    Returns:
        The trials, with `t_w_block` and `high`.
    """
    out = []
    for session_id in df['id'].unique():
        session = df[df['id'] == session_id].copy().reset_index(drop=True)
        prob_left = session['probabilityLeft'].to_numpy()
        switches = np.where(np.diff(prob_left) != 0)[0] + 1
        if len(switches) == 0:
            continue

        position = np.full(len(session), np.nan)
        counter = 0
        for i in range(len(session)):
            counter = 0 if i in switches else counter + 1
            position[i] = counter
        session['t_w_block'] = position
        session['high'] = chose_high_side(session[value_col], prob_left)

        before = []
        for switch in switches:
            start = max(0, switch - TRIALS_BEFORE_SWITCH)
            block = session.iloc[start:switch].copy()
            block['t_w_block'] = np.arange(-(switch - start), 0)
            before.append(block)
        lead_in = (switches[0] - TRIALS_BEFORE_SWITCH
                   if switches[0] >= TRIALS_BEFORE_SWITCH else 0)
        out.append(pd.concat([session.iloc[lead_in:]] + before,
                             ignore_index=True))
    return pd.concat(out, ignore_index=True)


# %%
# ============================================================================
# Panel e - decision-variable similarity
# ============================================================================
def decision_variable_corrs(mouse_df, sessions=LASER_ONLY_ABB,
                            root=SUBJECTS_ROOT):
    """
    Each pair of decision variables correlated within session, then averaged.

    Args:
        mouse_df: Output of `load_mouse_sessions`, reused when it carries the
            columns; otherwise the alf tables are re-read.
        sessions, root: Only used for that fallback.

    Returns:
        `(mean, sem, n, labels)`, each matrix 5 x 5 with a NaN diagonal.
    """
    labels = [label for label, _ in DECISION_VARS]
    k = len(labels)

    if all(column in mouse_df.columns for column in DECISION_VAR_COLS):
        frames = [group for _, group in mouse_df.groupby('id')]
    else:
        print("    decision-variable columns absent; re-reading alf")
        frames = [alf(str(session_path(ses_abb, root))).to_df()
                  for ses_abb in sessions]

    per_session = []
    for df in frames:
        X = np.column_stack([f(df) for _, f in DECISION_VARS])
        X = X[np.all(np.isfinite(X), axis=1)]
        if len(X) < DECISION_MIN_TRIALS:
            continue
        R = np.full((k, k), np.nan)
        for i in range(k):
            for j in range(i + 1, k):
                if np.std(X[:, i]) == 0 or np.std(X[:, j]) == 0:
                    continue
                R[i, j] = R[j, i] = pearsonr(X[:, i], X[:, j])[0]
        per_session.append(R)

    if not per_session:
        raise ValueError("no session had enough finite trials")

    stack = np.dstack(per_session)
    n = np.sum(~np.isnan(stack), axis=2)
    with np.errstate(invalid='ignore'):
        mean = np.nanmean(stack, axis=2)
        sd = np.nanstd(stack, axis=2, ddof=1)
    print(f"    {len(per_session)} sessions in the matrix")
    return mean, sd / np.sqrt(np.maximum(n, 1)), n, labels


# %%
# ============================================================================
# Panel f - the LOO comparison
# ============================================================================
def plot_loo(comparison, xticks=LOO_XTICKS, figsize=None):
    """
    ELPD per model.

    Args:
        comparison: The LOO table, best model first.
        xticks: X tick positions.
        figsize: Overrides the default, which grows with the model count so
            the multi-line labels keep their room.
    """
    fig, ax = plt.subplots(figsize=figsize or (4.0, 1.0 + 0.8 * len(comparison)))
    y = np.arange(len(comparison))[::-1]          # best model at the top

    # One row per model. The circle carries the ELPD's own standard error, the
    # triangle the standard error of the trial-wise difference from the best
    # model. Both are drawn at the same x; only the error bar differs.
    ax.errorbar(comparison["elpd"], y, xerr=comparison["se"], fmt="o",
                color=POINT_COLOR, elinewidth=LINE_WIDTH, markersize=5,
                label="ELPD\n± SE", zorder=3)
    ax.errorbar(comparison["elpd"], y + DSE_MARKER_OFFSET,
                xerr=comparison["dse"], fmt="^", color=COLOR_DSE,
                elinewidth=LINE_WIDTH, markersize=5, label="ELPD\n± dSE",
                zorder=2)
    # `comparison` is best-first, so row 0 is the model the dSEs are measured
    # against (Q-learning with Forgetting and Stay CS+ as reward);
    # the dashed line marks its ELPD. 
    ax.axvline(comparison["elpd"].iloc[0], linestyle="--", color=COLOR_DSE,
               linewidth=LINE_WIDTH, zorder=1)

    ax.set_yticks(y)
    ax.set_yticklabels([MODEL_LABELS[n] for n in comparison.index])
    ax.set_ylim(y.min() - 0.7, y.max() + 0.9)
    if xticks is not None:
        ax.set_xticks(xticks)
    ax.set_xlabel("ELPD (LOO)")
    ax.legend(loc="upper left", frameon=True, edgecolor=POINT_COLOR,
              framealpha=1)
    sns.despine(ax=ax)
    return fig


# %%
# ============================================================================
# Panels c and d - reading the recovery summaries
# ============================================================================
def load_means(path):
    """
    The `Mean` column of a CmdStan summary, indexed by parameter name.

    Args:
        path: A `summary.csv`.

    Returns:
        Series of posterior means.
    """
    if not Path(path).exists():
        raise FileNotFoundError(f"{path} not found")
    table = pd.read_csv(path)
    table = table.rename(columns={table.columns[0]: 'name'})
    if 'Mean' not in table.columns:
        raise ValueError(f"{path} has no 'Mean' column; it has "
                         f"{list(table.columns)}")
    return table.set_index('name')['Mean']


def check_columns(means, label):
    """
    Confirm a summary belongs to the model these panels are written for.

    Args:
        means: Output of `load_means`.
        label: Which summary, for the message.
    """
    present = {name.split('[')[0] for name in means.index}
    missing = [p['name'] for p in RECOVERY_PARAMS if p['name'] not in present]
    superseded = [name for name in SUPERSEDED if name in present]
    if missing or superseded:
        message = [f"{label} is not a fit of {MODEL_NAME}."]
        if missing:
            message.append(f"  missing: {', '.join(missing)}")
        if superseded:
            message.append(
                f"  carries parameters of the earlier model: "
                f"{', '.join(superseded)} - this looks like a fit of plain "
                f"sequential_actor_critic, which has alpha_win and a "
                f"session-level V(choice) rate")
        message.append(f"  the summary does hold: "
                       f"{', '.join(sorted(present)[:12])} ...")
        raise ValueError("\n".join(message))
    print(f"    {label}: all {len(RECOVERY_PARAMS)} parameters present")


def vector(means, base, n):
    """
    One parameter's values as an array, NaN where the summary has no entry.

    Args:
        means: Output of `load_means`.
        base: Parameter name without the subscript.
        n: Expected length.

    Returns:
        Array of length n.
    """
    out = np.full(n, np.nan)
    for i in range(1, n + 1):
        key = f'{base}[{i}]'
        if key in means.index:
            out[i - 1] = means[key]
    return out


def build_params(means, NS, NSxNSESS):
    """Every parameter the forward pass needs, as arrays."""
    return {p['name']: vector(means, p['name'],
                              NS if p['scope'] == 'mouse' else NSxNSESS)
            for p in RECOVERY_PARAMS}


# %%
# ============================================================================
# Panel c - the forward pass and the Fisher-z average
# ============================================================================
def forward_pass(sim, params, ms_i):
    """
    One session's latent trajectories, replayed over its simulated choices.

    Identical updates to `figure6_data.simulate_session` in `intact` mode,
    with the choice, outcome and lick read from the simulated data instead of
    drawn. Each latent is recorded before its update, so the value at trial t
    is the one that acted on trial t.

    Args:
        sim: `NT_all`, `sub_idx`, `sess_idx` and the `c`, `o`, `l` arrays.
        params: Output of `build_params`.
        ms_i: The session slot.

    Returns:
        `V_cs`, `V_choice`, `pi_choice`, `pi_lick`, each one array per trial.
    """
    si = sim['sub_idx'][ms_i] - 1
    sei = sim['sess_idx'][ms_i] - 1
    n_trials = int(sim['NT_all'][ms_i])

    a_v_cs = inv_logit(params['alpha_v_cs_mouse'][si])
    a_v_choice = inv_logit(params['alpha_v_choice_mouse'][si])
    a_lick = inv_logit(params['alpha_lick_mouse'][si])
    beta_lick = params['beta_lick_mouse'][si]

    a_loss = inv_logit(params['alpha_loss_ses'][ms_i])
    beta_win = params['beta_win_ses'][ms_i]
    beta_loss = params['beta_loss_ses'][ms_i]
    side = params['side_ses'][ms_i]

    V_cs = params['V_cs_init_ses'][ms_i]
    V_choice = params['V_choice_init_ses'][ms_i]
    pi_lick = params['pi_lick_init_ses'][ms_i]
    pi_choice = 0.0

    trace = {name: np.zeros(n_trials)
             for name in ('V_cs', 'V_choice', 'pi_choice', 'pi_lick')}

    for t in range(n_trials):
        choice = sim['c'][si, sei, t]
        outcome = sim['o'][si, sei, t]
        lick = sim['l'][si, sei, t]
        action_sign = 2 * choice - 1
        p_right = inv_logit(side + pi_choice)
        p_chosen = p_right if choice == 1 else (1.0 - p_right)

        trace['V_cs'][t] = V_cs
        trace['V_choice'][t] = V_choice
        trace['pi_choice'][t] = pi_choice
        trace['pi_lick'][t] = pi_lick

        if outcome == 1:
            delta_cs = V_cs - V_choice
            V_choice += a_v_choice * delta_cs
            # No decay on the win side: this is the "no_awin" model.
            pi_choice = (pi_choice + action_sign * beta_win * delta_cs
                         * (1.0 - p_chosen))
            p_lick = inv_logit(pi_lick)
            if lick == 1:
                delta_laser = 1.0 - V_cs
                V_cs += a_v_cs * delta_laser
                pi_lick = ((1.0 - a_lick) * pi_lick
                           + beta_lick * delta_laser * (1.0 - p_lick))
            else:
                V_cs += a_v_cs * (0.0 - V_cs)
                pi_lick = (1.0 - a_lick) * pi_lick
        else:
            delta_cs = 0.0 - V_choice
            V_choice += a_v_choice * delta_cs
            pi_choice = ((1.0 - a_loss) * pi_choice
                         + action_sign * beta_loss * delta_cs
                         * (1.0 - p_chosen))
    return trace


def session_correlation(a, b):
    """Pearson r, or NaN where either series is constant or too short."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    finite = np.isfinite(a) & np.isfinite(b)
    a, b = a[finite], b[finite]
    if a.size < 2 or np.std(a) == 0 or np.std(b) == 0:
        return np.nan
    return float(pearsonr(a, b)[0])


def fisher_bar(correlations):
    """
    The Fisher-z mean of a set of Pearson r, with a one-SE interval.

    Averaging r directly is biased; averaging arctanh(r) is not. The interval
    is the empirical SE across sessions in z space, transformed back, so it is
    asymmetric on the r scale.

    Args:
        correlations: Per-session Pearson r.

    Returns:
        `{'mean', 'low', 'high', 'err_low', 'err_high', 'n'}` on the r scale.
    """
    values = np.asarray(correlations, float)
    values = values[np.isfinite(values)]
    if values.size == 0:
        raise ValueError("no session gave a finite correlation")
    z = np.arctanh(np.clip(values, -0.999999, 0.999999))
    mean_z = float(np.mean(z))
    se_z = (float(np.std(z, ddof=1) / np.sqrt(values.size))
            if values.size > 1 else 0.0)
    mean, low, high = (float(np.tanh(mean_z)), float(np.tanh(mean_z - se_z)),
                       float(np.tanh(mean_z + se_z)))
    return {'mean': mean, 'low': low, 'high': high, 'err_low': mean - low,
            'err_high': high - mean, 'n': int(values.size)}


# %%
# ============================================================================
# Panel n - smoothing
# ============================================================================
def rolling_mean(traces, window):
    """Per-row rolling mean, and the trial each value is centred on."""
    kernel = np.ones(window) / window
    rolled = np.array([np.convolve(row, kernel, mode='valid')
                       for row in traces])
    trials = np.arange(window // 2, window // 2 + rolled.shape[1])
    return rolled, trials


# %%
# ============================================================================
# Panel l - the per-CS+-trial table, the binning and the GLMM
# ============================================================================
def load_lick_session(ses_abb, prefix=L_PREFIX, root=SUBJECTS_ROOT):
    """
    One session's CS+ trials, with the model's V_cs and the observed lick.
 
    Args:
        ses_abb: A `LASER_ONLY_ABB` entry.
        prefix: Filename prefix of the model arrays.
        root: Subjects root.
 
    Returns:
        `trial`, `V_cs`, `p_lick`, `lick`, one row per CS+ trial.
 
    Raises:
        FileNotFoundError: If any of the four arrays is absent.
        ValueError: If they do not all hold the same number of trials.
    """
    def read(name, model=False):
        filename = f'{prefix}{name}.npy' if model else name
        path = session_alf_path(ses_abb, filename, root)
        if not path.exists():
            kind = f"model array (prefix '{prefix}')" if model else 'behaviour'
            raise FileNotFoundError(f'[missing] {kind}: {path}')
        return np.load(path, allow_pickle=True).astype(float)
 
    feedback = read('_ibl_trials.feedbackType.npy')
    consumed = read('_ibl_trials.feedbackTypeConsumed.npy')
    v_cs = read('V_cs', model=True)
    p_lick = read('p_lick', model=True)
 
    lengths = {'feedbackType': len(feedback),
               'feedbackTypeConsumed': len(consumed),
               'V_cs': len(v_cs), 'p_lick': len(p_lick)}
    if len(set(lengths.values())) != 1:
        raise ValueError(
            f'{ses_abb}: the trial arrays disagree on length ({lengths}). '
            'Re-run extract_decision_variables for this session.')
 
    n = len(feedback)
    cs_plus = feedback == 1
    return pd.DataFrame({
        'trial': np.arange(n)[cs_plus],
        'V_cs': v_cs[cs_plus],
        'p_lick': p_lick[cs_plus],
        'lick': (consumed[cs_plus] == 1).astype(int),
    })

def build_lick_table(sessions=LASER_ONLY_ABB, prefix=L_PREFIX):
    """
    Every session's CS+ trials in one table, with the inclusion tally printed.

    Args:
        sessions: Session list to attempt.
        prefix: Passed to `load_lick_session`.

    Returns:
        `(table, audit)`. The table carries `session` and `mouse`; the audit
        has one row per attempted session with why it was kept or dropped.
    """
    cache = PAPER_DIR / "data" / "extended_data_figure11" / "lick_table.csv.gz"
    if cache.exists():
        table = pd.read_csv(cache)
        return table, pd.DataFrame({'session': table['session'].unique(),
                                    'status': 'kept', 'reason': ''})
    frames, audit = [], []
    for ses_abb in sessions:
        mouse = str(ses_abb).replace('\\', '/').strip('/').split('/')[0]
        record = {'session': str(ses_abb), 'mouse': mouse, 'n_csplus': np.nan,
                  'n_missed': np.nan, 'lick_rate': np.nan, 'V_cs_std': np.nan,
                  'status': 'kept', 'reason': ''}
        try:
            df = load_lick_session(ses_abb, prefix=prefix)
        except FileNotFoundError as error:
            record['status'], record['reason'] = 'dropped', str(error)
            audit.append(record)
            continue

        n_missed = int((df['lick'] == 0).sum())
        record['n_csplus'] = len(df)
        record['n_missed'] = n_missed
        record['lick_rate'] = df['lick'].mean() if len(df) else np.nan
        record['V_cs_std'] = df['V_cs'].std() if len(df) else np.nan

        if n_missed < L_MIN_MISSED_LICKS:
            record['status'] = 'dropped'
            record['reason'] = (f'only {n_missed} forgone CS+ '
                                f'(< L_MIN_MISSED_LICKS={L_MIN_MISSED_LICKS})')
        elif not np.isfinite(record['V_cs_std']) or record['V_cs_std'] < 1e-12:
            record['status'] = 'dropped'
            record['reason'] = 'V_cs constant - cannot quantile-bin x'
        else:
            df['session'] = str(ses_abb)
            df['mouse'] = mouse
            frames.append(df)
        audit.append(record)

    audit = pd.DataFrame(audit)
    dropped = audit[audit['status'] == 'dropped']
    print(f'    {len(sessions)} sessions attempted, '
          f'{int((audit["status"] == "kept").sum())} kept')
    for _, row in dropped.iterrows():
        print(f'      {row["session"]:<28} {row["reason"]}')
    if not frames:
        raise RuntimeError(
            f"no session loaded - the prefix '{prefix}' is probably wrong, or "
            "extract_decision_variables has not been run over these sessions")

    table = pd.concat(frames, ignore_index=True)
    print(f'    {table["session"].nunique()} sessions, '
          f'{table["mouse"].nunique()} mice, {len(table)} CS+ trials')
    return table, audit


def quantile_bin_sessions(df, xvar='V_cs', yvar='lick', nbins=L_NBINS,
                          center=L_CENTER):
    """
    Bin x into equal-count bins within each session, then average the bins.

    With `center`, each session's own mean is removed from x and y first, so
    what survives is the within-session shape rather than the between-session
    offset. 

    Args:
        df: Output of `build_lick_table`.
        xvar, yvar: Columns to bin and to average.
        nbins: Requested quantile bins per session.
        center: Session-centre x and y before binning.

    Returns:
        `(per_session, summary)`. The summary has one row per bin, with the
        mean and SEM of both axes across sessions.
    """
    per_session, skipped = [], []
    for session, group in df.groupby('session', sort=False):
        x = group[xvar].to_numpy(float)
        y = group[yvar].to_numpy(float)
        if center:
            x = x - x.mean()
            y = y - y.mean()
        if np.std(x) < 1e-12:
            skipped.append((session, f'{xvar} constant'))
            continue
        try:
            bins = pd.qcut(x, nbins, labels=False, duplicates='drop')
        except ValueError as error:
            skipped.append((session, f'qcut failed: {error}'))
            continue
        distinct = pd.Series(bins).nunique()
        if distinct < L_MIN_BINS:
            skipped.append((session, f'only {distinct} distinct bins '
                                     f'(< L_MIN_BINS={L_MIN_BINS})'))
            continue
        binned = (pd.DataFrame({'bin': bins, 'x': x, 'y': y})
                  .groupby('bin').agg(x=('x', 'mean'), y=('y', 'mean'),
                                      n=('y', 'size')).reset_index())
        binned['session'] = session
        binned['mouse'] = group['mouse'].iloc[0]
        per_session.append(binned)

    for session, why in skipped:
        print(f'      binning skipped {session}: {why}')
    if not per_session:
        raise RuntimeError(f'no session survived binning on {xvar!r}')

    per_session = pd.concat(per_session, ignore_index=True)
    summary = per_session.groupby('bin').agg(
        x=('x', 'mean'), x_sem=('x', sem),
        y=('y', 'mean'), y_sem=('y', sem),
        n_ses=('session', 'nunique')).reset_index()
    return per_session, summary


def zscore_within(df, xvar='V_cs', by='session'):
    """
    z-score `xvar` inside each session, so the slope is per within-session SD.

    `ddof=0`, and a session whose SD is numerically zero gets a divisor of 1
    rather than an infinity - it contributes no variance either way.
    """
    out = df.copy()
    grouped = out.groupby(by)[xvar]
    sd = grouped.transform(lambda v: v.std(ddof=0))
    sd = sd.where(sd > 1e-9, 1.0)
    out[f'{xvar}_z'] = (out[xvar] - grouped.transform('mean')) / sd
    return out


def fit_lick_glmm(df, formula=L_FORMULA, yvar='lick'):
    """
    Mixed-effects logistic regression of the lick on V_cs, via lme4::glmer.

    The only panel of this figure that needs R. Run it where rpy2 and lme4
    are reachable; on the cluster `iblenv` already has both.

    Args:
        df: Output of `build_lick_table`. `V_cs_z` is added if absent.
        formula: lme4 formula with `{y}` for the response.
        yvar: Response column.

    Returns:
        The fixed effect of `V_cs_z` as `beta`, `se`, `z`, `p`, `or`, its
        confidence interval, the random-effect variances, and the convergence
        and singularity flags.

    Raises:
        ModuleNotFoundError: If rpy2 is absent.
        RuntimeError: If glmer does not return a fit.
    """
    try:
        import rpy2.robjects as ro
        from rpy2.robjects import pandas2ri
        from rpy2.robjects.packages import importr
    except ModuleNotFoundError as error:
        raise ModuleNotFoundError(
            'rpy2 is not available in this interpreter. Panel l needs R with '
            'lme4; run it on the cluster, where iblenv has both.') from error

    data = df.copy()
    if 'V_cs_z' not in data:
        data = zscore_within(data)
    data = data.replace([np.inf, -np.inf], np.nan).dropna(
        subset=['V_cs_z', yvar])
    data['session'] = data['session'].astype(str)
    data['mouse'] = data['mouse'].astype(str)
    data['resp'] = data[yvar].astype(int)

    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        importr('lme4')
        importr('stats')

        with (ro.default_converter + pandas2ri.converter).context():
            ro.globalenv['dat'] = ro.conversion.get_conversion().py2rpy(
                data[['resp', 'V_cs_z', 'session', 'mouse']])

        ro.r('dat$session <- factor(dat$session); '
             'dat$mouse <- factor(dat$mouse)')
        fitted_formula = formula.format(y='resp')
        ro.globalenv['fml'] = ro.r(f'as.formula("{fitted_formula}")')
        # The default optimiser first; bobyqa only if that one fails to
        # converge, which is lme4's own recommended fallback.
        result = ro.r('''
        suppressWarnings({
            fit <- tryCatch(glmer(fml, data = dat, family = binomial),
                            error = function(e) NULL)
            if (is.null(fit) || fit@optinfo$conv$opt != 0) {
                fit <- tryCatch(
                    glmer(fml, data = dat, family = binomial,
                          control = glmerControl(optimizer = "bobyqa",
                                                 optCtrl = list(maxfun = 1e5))),
                    error = function(e) NULL)
            }
        })
        if (is.null(fit)) {
            list(ok = FALSE)
        } else {
            smry <- summary(fit)
            vc   <- as.data.frame(VarCorr(fit))
            list(ok = TRUE,
                 coef_tab = smry$coefficients,
                 rownames = rownames(smry$coefficients),
                 vc_grp   = as.character(vc$grp),
                 vc_var1  = as.character(vc$var1),
                 vc_var2  = ifelse(is.na(vc$var2), "", as.character(vc$var2)),
                 vc_vcov  = as.numeric(vc$vcov),
                 singular = isSingular(fit, tol = 1e-4),
                 conv     = fit@optinfo$conv$opt,
                 nobs     = nobs(fit),
                 aic      = AIC(fit))
        }
        ''')

    if not bool(result.rx2('ok')[0]):
        raise RuntimeError(f'glmer returned no fit for: {fitted_formula}')

    coefficients = pd.DataFrame(np.array(result.rx2('coef_tab')),
                                index=list(result.rx2('rownames')),
                                columns=['Estimate', 'SE', 'Z-stat', 'P-val'])
    slope = coefficients.loc['V_cs_z']
    random = pd.DataFrame({'group': list(result.rx2('vc_grp')),
                           'term': list(result.rx2('vc_var1')),
                           'term2': list(result.rx2('vc_var2')),
                           'var': list(result.rx2('vc_vcov'))})
    random['kind'] = np.where(random['term2'] == '', 'var', 'cov')

    return {
        'formula': fitted_formula.replace('resp', yvar),
        'beta': float(slope['Estimate']), 'se': float(slope['SE']),
        'z': float(slope['Z-stat']), 'p': float(slope['P-val']),
        'or': float(np.exp(slope['Estimate'])),
        'ci_lo': float(np.exp(slope['Estimate'] - 1.96 * slope['SE'])),
        'ci_hi': float(np.exp(slope['Estimate'] + 1.96 * slope['SE'])),
        'ranef': random,
        'singular': bool(result.rx2('singular')[0]),
        'converged': int(result.rx2('conv')[0]) == 0,
        'n_trials': int(len(data)), 'aic': float(result.rx2('aic')[0]),
        'n_sessions': int(data['session'].nunique()),
        'n_mice': int(data['mouse'].nunique()),
        'coefs': coefficients,
    }


# %%
# ============================================================================
# Load - the cheap part
# ============================================================================
apply_style()
print(f"panels will be written to {SAVE_DIR}")

print("\n#### fitted parameters")
pk = load_pkl()
standata = load_recovery_data()
verify_grid(pk, standata)
params = seqac_params_long(pk, standata)

print("\n#### real behaviour")
mouse_df = load_mouse_sessions()
mouse_choice = mouse_df[mouse_df['choice'] != 0].copy()

print("\n#### on-policy simulation")
sim_df = simulate_seqac(standata, pk)


# %%
# ============================================================================
# Fig. ED11b - fitted initialization parameters
# ============================================================================
print("\n#### ED Fig 11b  initialization parameters")
fig, axes = plt.subplots(1, 2, figsize=(1.9, 2.0))
param_violins(params, ['Critic init.', 'Actor init.'], axes, ylabel=None)
fig.suptitle('Initialization Parameters', fontsize=FS_TITLE, y=1.14)
fig.subplots_adjust(wspace=0.8)
save_panel(fig, 'figED11b_initialization.pdf', save_dir=SAVE_DIR)


# %%
# ============================================================================
# Session grid and the recovery summaries - needed by panels c and d
# ============================================================================
print("\n#### ED Fig 11 c,d  session grid (from the frozen STAN_ORDER)")
_, NT_all, sub_idx, sess_idx, NS, NSESS, NSxNSESS = stan_grid(pk['raw'])
real = NT_all > 0
print(f"    {NS} mice x {NSESS} slots = {NSxNSESS}, "
      f"{int(real.sum())} real sessions")

print("\n#### ED Fig 11 c,d  summaries")
behaviour = load_means(BEHAVIOUR_SUMMARY)
recovered = load_means(RECOVERY_SUMMARY)
check_columns(behaviour, 'the behaviour fit')
check_columns(recovered, 'the recovery fit')


# %%
# ============================================================================
# Fig. ED11c - latent-trajectory recovery
# ============================================================================
print("\n#### ED Fig 11c  latent-trajectory recovery")
sim = {'NT_all': NT_all, 'sub_idx': sub_idx, 'sess_idx': sess_idx}
for key, filename in SIM_FILES.items():
    path = RECOVERY_DIR / filename
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found. Panel c replays both parameter sets over the "
            "simulated choices the recovery was fit to, so those arrays are "
            "needed; they are written beside the fit by the simulation step.")
    sim[key] = np.load(path)
print(f"    simulated data {sim['c'].shape} (mice x slots x trials)")

generating = build_params(behaviour, NS, NSxNSESS)
recovering = build_params(recovered, NS, NSxNSESS)

per_session = {name: [] for name, _ in LATENTS}
for ms_i in range(NSxNSESS):
    if NT_all[ms_i] <= 0:
        continue
    from_behaviour = forward_pass(sim, generating, ms_i)
    from_recovery = forward_pass(sim, recovering, ms_i)
    for name, _ in LATENTS:
        per_session[name].append(session_correlation(from_behaviour[name],
                                                     from_recovery[name]))

bars = [fisher_bar(per_session[name]) for name, _ in LATENTS]
for (name, _), bar in zip(LATENTS, bars):
    print(f"    {name:10s} r={bar['mean']:.3f} "
          f"(-{bar['err_low']:.3f}/+{bar['err_high']:.3f} SE, n={bar['n']})")

fig, ax = plt.subplots(figsize=(1.9, 2.1))
x = np.arange(len(LATENTS))
ax.bar(x, [b['mean'] for b in bars],
       yerr=np.array([[b['err_low'] for b in bars],
                      [b['err_high'] for b in bars]]),
       capsize=3, width=0.7, color=C_BAR_COLOR, edgecolor='black',
       lw=LINE_WIDTH)

for position, bar in zip(x, bars):
    ax.text(position, 0.04, f"{bar['mean']:.3f} (n={bar['n']})",
            rotation=90, ha='center', va='bottom', fontsize=FS_LEGEND)

ax.axhline(1.0, color='grey', ls=':', lw=0.8)
ax.set_xticks(x)
ax.set_xticklabels([label for _, label in LATENTS], rotation=45, ha='right',
                   fontsize=7)
ax.set_ylim(*C_YLIM)
ax.set_ylabel('trajectory recovery\n(Pearson r generating\nvs recovered)',
              fontsize=FS_LABEL)
ax.set_title('Latent-trajectory recovery\n'
             'Fisher-z mean per-session r ± SE', fontsize=FS_LEGEND)
sns.despine(ax=ax)
save_panel(fig, 'figED11c_trajectory_recovery.pdf', save_dir=SAVE_DIR)


# %%
# ============================================================================
# Fig. ED11d - parameter recovery
# ============================================================================
print("\n#### ED Fig 11d  parameter recovery")
rows = D_GROUPS[-1][0] + int(np.ceil(
    len([p for p in RECOVERY_PARAMS if p['scope'] == 'session']) / D_COLUMNS))
fig, axes = plt.subplots(rows, D_COLUMNS,
                         figsize=(1.55 * D_COLUMNS, 1.55 * rows))
axes = np.atleast_1d(axes).ravel()

for ax, param in zip(axes, RECOVERY_PARAMS):
    n = NS if param['scope'] == 'mouse' else NSxNSESS
    x = vector(behaviour, param['name'], n)
    y = vector(recovered, param['name'], n)
    keep = np.isfinite(x) & np.isfinite(y)
    if param['scope'] == 'session':
        keep &= real
    x, y = x[keep], y[keep]
    if param['squash']:
        x, y = inv_logit(x), inv_logit(y)

    r = session_correlation(x, y)
    print(f"    {param['name']:24s} r={r:+.3f}  n={len(x)}")

    ax.scatter(x, y, s=6, color=POINT_COLOR, edgecolor='none')
    low, high = float(min(x.min(), y.min())), float(max(x.max(), y.max()))
    pad = 0.05 * (high - low if high > low else 1.0)
    ax.plot([low - pad, high + pad], [low - pad, high + pad], ls='--',
            color=IDENTITY_COLOR, lw=LINE_WIDTH, zorder=0)
    ax.set_xlim(low - pad, high + pad)
    ax.set_ylim(low - pad, high + pad)
    ax.text(0.05, 0.95, f"r = {r:.2f}", transform=ax.transAxes,
            fontsize=FS_LABEL, va='top')
    ax.set_title(param['label'], fontsize=FS_LABEL)
    ax.tick_params(labelsize=6)
    ax.set_aspect('equal', adjustable='box')
    sns.despine(ax=ax)

for ax in axes[len(RECOVERY_PARAMS):]:
    ax.axis('off')

# Axis labels on the outside only, as the published panel has them.
for row in range(rows):
    axes[row * D_COLUMNS].set_ylabel('Simulation Fit', fontsize=FS_LABEL)
for index, ax in enumerate(axes[:len(RECOVERY_PARAMS)]):
    if index + D_COLUMNS >= len(RECOVERY_PARAMS):
        ax.set_xlabel('Behavior Fit', fontsize=FS_LABEL)

fig.subplots_adjust(left=0.16, hspace=0.75, wspace=0.6)

# The two row-group captions, each spanning the rows it covers. Positions are
# read after the layout is set, so they follow the axes rather than guessing.
group_rows = [(row, next_row) for (row, _), next_row
              in zip(D_GROUPS, [g[0] for g in D_GROUPS[1:]] + [rows])]
for (row, stop), (_, caption) in zip(group_rows, D_GROUPS):
    count = NS if row == 0 else int(real.sum())
    top = axes[row * D_COLUMNS].get_position().y1
    bottom = axes[(stop - 1) * D_COLUMNS].get_position().y0
    fig.text(0.015, (top + bottom) / 2, f"{caption}\n(n = {count})",
             rotation=90, ha='left', va='center', fontsize=FS_LEGEND)

save_panel(fig, 'figED11d_parameter_recovery.pdf', save_dir=SAVE_DIR,
           tight=False)


# %%
# ============================================================================
# Fig. ED11e - decision-variable similarity
# ============================================================================
print("\n#### ED Fig 11e  decision-variable similarity")
mean_r, sem_r, n_r, labels = decision_variable_corrs(mouse_df)

# The published matrix is the strict lower triangle, so the empty first row
# and last column are dropped rather than left as blank margins.
keep = np.tril(np.ones_like(mean_r, dtype=bool), -1)
annotations = np.array(
    [[f"{mean_r[i, j]:.3f}\n±{sem_r[i, j]:.3f}" if keep[i, j] else ''
      for j in range(len(labels))] for i in range(len(labels))], dtype=object)
extreme = np.abs(mean_r[keep]).max()

fig, ax = plt.subplots(figsize=(3.4, 2.8))
sns.heatmap(mean_r[1:, :-1], mask=~keep[1:, :-1], annot=annotations[1:, :-1],
            fmt='', cmap='vlag', vmin=-extreme, vmax=extreme, square=True,
            linewidths=0.6, linecolor='white', annot_kws={'size': 4.5}, ax=ax,
            cbar_kws={'label': 'Correlation between\ndecision variables (r)',
                      'shrink': 0.75, 'pad': 0.02})
ax.set_facecolor('white')          # the masked cells, left blank
ax.set_xticklabels(labels[:-1], rotation=45, ha='right', fontsize=5)
ax.set_yticklabels(labels[1:], rotation=0, fontsize=5)
ax.set_title(DECISION_TITLE, fontsize=FS_LEGEND)
colorbar = ax.collections[0].colorbar
colorbar.ax.tick_params(labelsize=5)
colorbar.ax.yaxis.label.set_size(5.5)
save_panel(fig, 'figED11e_decision_variables.pdf', save_dir=SAVE_DIR)

for i in range(1, len(labels)):
    for j in range(i):
        print(f"    {labels[i].replace(chr(10), ' ')} vs "
              f"{labels[j].replace(chr(10), ' ')}: "
              f"r={mean_r[i, j]:+.3f} ± {sem_r[i, j]:.3f} (n={n_r[i, j]})")


# %%
# ============================================================================
# Fig. ED11f - PSIS-LOO model comparison, all six models
# ============================================================================
print("\n#### ED Fig 11f  PSIS-LOO model comparison")
if not LOO_COMPARISON_CSV.exists():
    raise FileNotFoundError(
        f"{LOO_COMPARISON_CSV} not found. `compute_loo_comparison.py` writes "
        "it alongside the five-model table Extended Data Fig. 2a reads; run "
        "that once.")
loo_comparison = pd.read_csv(LOO_COMPARISON_CSV, index_col="model")
if len(loo_comparison) != 6:
    raise ValueError(
        f"{LOO_COMPARISON_CSV} holds {len(loo_comparison)} models, expected 6. "
        "This panel is the comparison WITH the Conditioned Reinforcer; the "
        "five-model table is Extended Data Fig. 2a's.")
print(loo_comparison.round(2).to_string())

fig = plot_loo(loo_comparison)
save_panel(fig, 'figED11f_loo_comparison.pdf', save_dir=SAVE_DIR)


# %%
# ============================================================================
# Fig. ED11g - win-stay, mice against the model
# ============================================================================
print("\n#### ED Fig 11g  return probability after CS- and CS+")
fig, axes = plt.subplots(1, 2, figsize=(2.2, 2.0), sharey=True)
for ax, table, colour, title in [
        (axes[0], repeat_after_outcome(mouse_choice), 'gray', 'Mouse'),
        (axes[1], repeat_after_outcome(sim_df), MODEL_COLOR, 'Model')]:
    print(f"    {title}: {table['id'].nunique()} sessions, "
          f"after CS- {table.loc[table.after_cs < 1, 'repeat'].mean():.4f}, "
          f"after CS+ {table.loc[table.after_cs > 0, 'repeat'].mean():.4f}")
    # No saturation= here, unlike the violins: the published bars measure
    # seaborn's own 75% desaturation of MODEL_COLOR, so the default is right.
    sns.barplot(data=table, x='after_cs', y='repeat', errorbar='se',
                color=colour, ax=ax)
    sns.lineplot(data=table, x='after_cs', y='repeat', units='id',
                 estimator=None, color='0.45', lw=0.3, alpha=0.3,
                 legend=False, ax=ax)
    ax.set_title(title, fontsize=FS_TITLE)
    ax.set_xlabel('t-1 outcome', fontsize=FS_LABEL)
    ax.set_xticks([0, 1])
    ax.set_xticklabels(['After\nCS-', 'After\nCS+'])
    ax.set_ylim(0, 1)
    ax.set_ylabel('')
    sns.despine(ax=ax)
axes[0].set_ylabel('Return probability', fontsize=FS_LABEL)
save_panel(fig, 'figED11g_return_probability.pdf', save_dir=SAVE_DIR)


# %%
# ============================================================================
# Fig. ED11h - action selection
# ============================================================================
print("\n#### ED Fig 11h  P(right) against the propensity decile")
fig, ax = plt.subplots(figsize=(2.0, 2.0))
for df, column, colour, label in [
        (sim_df, 'pi_choice', MODEL_COLOR, 'Model'),
        (mouse_choice, 'pi_choice_tf', MOUSE_COLOR, 'Mouse')]:
    usable = df.dropna(subset=[column, 'bchoice']).copy()
    usable['decile'] = pd.qcut(usable[column], VALUE_BINS, labels=False,
                               duplicates='drop')
    per_decile = (usable.groupby(['id', 'decile'], observed=True)['bchoice']
                  .mean().reset_index())
    print(f"    {label}: {per_decile['id'].nunique()} sessions, "
          f"{per_decile['decile'].nunique()} bins")
    sns.lineplot(data=per_decile, x='decile', y='bchoice', errorbar='se',
                 err_style='bars', color=colour, label=label, ax=ax)

ax.set_ylim(0, 1)
ax.set_ylabel('P(Right)', fontsize=FS_LABEL)
ax.set_xlabel('Relative value decile', fontsize=FS_LABEL)
ax.legend(frameon=False, fontsize=FS_LEGEND)
sns.despine(ax=ax)
save_panel(fig, 'figED11h_action_selection.pdf', save_dir=SAVE_DIR)


# %%
# ============================================================================
# Fig. ED11i - block switches
# ============================================================================
print("\n#### ED Fig 11i  P(high) around block switches")
fig, ax = plt.subplots(figsize=(2.2, 2.0))
for df, colour, label in [(sim_df, MODEL_COLOR, 'Model'),
                          (mouse_choice, MOUSE_COLOR, 'Mouse')]:
    long = trials_from_switch(df.assign(_right=df['bchoice']), '_right')
    per_position = (long.groupby(['id', 't_w_block'])['high'].mean()
                    .reset_index())
    print(f"    {label}: {per_position['id'].nunique()} sessions")
    sns.lineplot(data=per_position, x='t_w_block', y='high', errorbar='se',
                 err_style='bars', color=colour, label=label, ax=ax)

ax.set_xlim(*TRANSITION_XLIM)
ax.set_ylim(0, 1)
ax.axvline(0, ls='--', color='k', lw=0.8)
ax.set_ylabel('P(High Rew. Prob. Choice)', fontsize=FS_LABEL)
ax.set_xlabel('Trials from block switch', fontsize=FS_LABEL)
ax.legend(frameon=False, fontsize=FS_LEGEND)
sns.despine(ax=ax)
save_panel(fig, 'figED11i_block_switch.pdf', save_dir=SAVE_DIR)


# %%
# ============================================================================
# Fig. ED11j - predicted P(lick) on licked and unlicked CS+ trials
# ============================================================================
print("\n#### ED Fig 11j  predicted P(lick), skipped against collected")
piv, lick = lick_detect_wilcoxon(mouse_df)
print(f"    {lick['n']} sessions; median skip {lick['median_skip']:.4f}, "
      f"median collect {lick['median_collect']:.4f}")
print(f"    Wilcoxon W={lick['W']:.1f}  p={lick['p']:.3g}  {stars(lick['p'])}")

fig, ax = plt.subplots(figsize=(1.5, 2.0))
for _, session in piv.iterrows():
    ax.plot([0, 1], [session[0], session[1]], color='gray', lw=0.4, alpha=0.4,
            zorder=1)
means, sems = piv.mean(), piv.sem()

ax.errorbar([0, 1], [means[0], means[1]], yerr=[sems[0], sems[1]],
            color=MOUSE_COLOR, lw=2, marker='o', ms=5, capsize=3, zorder=3)

top = float(piv.to_numpy().max())
bracket(ax, 0, 1, top + 0.01, stars(lick['p']), tick=0.01)
ax.set_xticks([0, 1])
ax.set_xticklabels(['CS+ No Lick', 'CS+ Lick'], rotation=90)
ax.set_xlim(-0.3, 1.3)
ax.set_ylim(max(0.0, float(piv.to_numpy().min()) - 0.05), top + 0.05)
ax.set_ylabel('P(Lick)', fontsize=FS_LABEL)
sns.despine(ax=ax)
save_panel(fig, 'figED11j_lick_detection.pdf', save_dir=SAVE_DIR)


# %%
# ============================================================================
# Fig. ED11l - V(S_CS+) against observed licking, within session
# ============================================================================
# The only panel here that needs R. If rpy2 or lme4 is missing the figure is
# still drawn, without the p-value in its title, and the cell says so rather
# than failing - so the other eleven panels remain runnable anywhere.
print("\n#### ED Fig 11l  V(S_CS+) against the observed lick")
lick_table, lick_audit = build_lick_table()

try:
    lick_fit = fit_lick_glmm(lick_table)
except ModuleNotFoundError as error:
    print(f"    GLMM skipped: {error}")
    lick_fit = None

if lick_fit is not None:
    print(f"    {lick_fit['formula']}")
    print(f"    beta={lick_fit['beta']:+.3f} (SE {lick_fit['se']:.3f}) "
          f"per within-session SD of V_cs")
    print(f"    OR={lick_fit['or']:.2f} "
          f"[{lick_fit['ci_lo']:.2f}, {lick_fit['ci_hi']:.2f}]  "
          f"z={lick_fit['z']:+.2f}  p={lick_fit['p']:.3g}  "
          f"{stars(lick_fit['p'])}")
    print(f"    n={lick_fit['n_trials']} trials, "
          f"{lick_fit['n_sessions']} sessions, {lick_fit['n_mice']} mice")
    for _, row in lick_fit['ranef'].iterrows():
        kind = 'sigma^2' if row['kind'] == 'var' else 'cov    '
        term = (row['term'] if row['kind'] == 'var'
                else f"{row['term']}~{row['term2']}")
        print(f"      {kind} {row['group']}/{term}: {row['var']:+.3f}")
    if lick_fit['singular']:
        print("    !! SINGULAR - a variance component went to ~0, so this is "
              "effectively a\n       plain logistic regression. Do not report "
              "it as a mixed model.")
    if not lick_fit['converged']:
        print("    !! the optimiser did not converge.")

per_session_bins, lick_summary = quantile_bin_sessions(lick_table)
print(f"    {L_NBINS} quantile bins, "
      f"{int(lick_summary['n_ses'].max())} sessions contributing")

fig, ax = plt.subplots(figsize=L_FIGSIZE)
ax.axhline(0, ls='--', color='0.6', lw=0.8, zorder=0)
ax.axvline(0, ls='--', color='0.6', lw=0.8, zorder=0)
# SEM across sessions on both axes: the bin's x is a session mean too, so an
# x error bar is as earned as the y one.
ax.errorbar(lick_summary['x'], lick_summary['y'],
            xerr=lick_summary['x_sem'], yerr=lick_summary['y_sem'],
            color=MOUSE_COLOR, marker='o', ms=4, lw=1.2, capsize=2, zorder=3)
ax.set_xlabel('$V(S_{CS+})$\nsession-centered', fontsize=FS_LABEL)
ax.set_ylabel('Observed lick\nsession-centered', fontsize=FS_LABEL)
if lick_fit is not None:
    ax.set_title(f"{stars(lick_fit['p'])}  p={lick_fit['p']:.2g}",
                 fontsize=FS_TITLE)
sns.despine(ax=ax)
save_panel(fig, 'figED11l_vcs_lick.pdf', save_dir=SAVE_DIR)


# %%
# ============================================================================
# The manipulation simulations - needed by panels k, m and n
# ============================================================================
# Both simulators are seeded, so these reproduce exactly. The extended horizon
# is the only slow step in this script: EXT_TRIALS x 3 modes x N_REPS per
# session, about fourteen minutes. Pass run_extended=False to skip it while
# working on panels k and m, which only need the standard horizon.
print("\n#### manipulation simulations")
sims = run_manipulation_simulations(pk, run_extended=True)


# %%
# ============================================================================
# Fig. ED11k - fitted V(CS) initialization against the fraction of CS+ skipped
# ============================================================================
# Drawn twice: against the skip rate from the intact simulation and against the
# mice's own skip rate over the same sessions. Both are kept.
print("\n#### ED Fig 11k  V(CS) initialization against skipped CS+")
between = sims['intact'].dropna(subset=['skip_rate', 'V_cs_init',
          'pi_lick_init']).copy() # Currently all sessions pass
between = between.merge(observed_skip_rate(mouse_df), on='session',
                        how='left')
missing = int(between['observed_skip_rate'].isna().sum())
if missing:
    raise ValueError(f"{missing} simulated session(s) did not match a session "
                     "in the behaviour table - STAN_ORDER and LASER_ONLY_ABB "
                     "disagree")

for column, name, filename in [
        ('skip_rate', 'simulated', 'figED11k_between_session_simulated.pdf'),
        ('observed_skip_rate', 'observed',
         'figED11k_between_session_observed.pdf')]:
    r, p = partial_corr(between, 'V_cs_init', column, ['pi_lick_init'])
    plain = pearsonr(between['V_cs_init'], between[column])
    print(f"    {name}: n={len(between)}  partial r={r:+.3f} p={p:.3g}  "
          f"(unadjusted r={plain[0]:+.3f} p={plain[1]:.3g})")
    # Below 1e-4 the published panel reports a bound rather than the value.
    reported = f"{stars(p)}p<0.0001" if p < 1e-4 else f"p={p:.3g}"

    fig, ax = plt.subplots(figsize=(2.0, 2.0))
    regression_panel(ax, between['V_cs_init'], between[column],
                     point_color='k', line_color='0.5', ci=None,
                     linestyle='--',
                     xlabel='$V_0(S_{CS+})$',
                     ylabel='Fraction of CS+ trials\nw/o VTA stim (skipped)',
                     annotation=(f"partial r={r:.2f}, {reported}\n"
                                 f"(control $\\pi_0^{{lick}}$)"))
    save_panel(fig, filename, save_dir=SAVE_DIR)


# %%
# ============================================================================
# Fig. ED11m - performance with the stimulation removed
# ============================================================================
print("\n#### ED Fig 11m  laser-off manipulation")
with_stim, without_stim = paired_performance(sims['sim_mean'], 'laser_off')
print(f"    {len(with_stim)} sessions; w/ stim {with_stim.mean():.4f}, "
      f"w/o stim {without_stim.mean():.4f}")

fig, ax = plt.subplots(figsize=(1.5, 2.0))
w, p = paired_bar_panel(ax, with_stim, without_stim,
                        ['w/ CS+ &\nw/ VTA stim.', 'w/ CS+ &\nw/o VTA stim.'],
                        'P(High Rew. Prob. Choice)', chance=CHANCE)
print(f"    Wilcoxon W={w:.1f}  p={p:.3g}  {stars(p)}")
ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right')
ax.set_title('Laser off\nmanipulation', fontsize=FS_LABEL)
save_panel(fig, 'figED11m_laser_off.pdf', save_dir=SAVE_DIR)


# %%
# ============================================================================
# Fig. ED11n - performance against trial number under each manipulation
# ============================================================================
print("\n#### ED Fig 11n  performance decay")
traces = sims['ext_arr']
if traces is None:
    raise ValueError("the extended horizon was not run, so 11n has no data; "
                     "call run_manipulation_simulations(run_extended=True)")

# A broken x-axis: the transition window on the left, a late window on the
# right, drawn as two axes glued together.
fig, (left, right) = plt.subplots(
    1, 2, figsize=(3.8, 1.9), sharey=True,
    gridspec_kw=dict(
        width_ratios=[DECAY_LEFT_WINDOW[1] - DECAY_LEFT_WINDOW[0],
                      DECAY_RIGHT_WINDOW[1] - DECAY_RIGHT_WINDOW[0]],
        wspace=0.18))
for mode, label in DECAY_LABELS.items():
    rolled, trials = rolling_mean(traces[mode], WINDOW)
    trace_mean = rolled.mean(axis=0)
    trace_sem = rolled.std(axis=0, ddof=1) / np.sqrt(rolled.shape[0])
    late = (trials >= DECAY_RIGHT_WINDOW[0]) & (trials <= DECAY_RIGHT_WINDOW[1])
    print(f"    {label}: {rolled.shape[0]} sessions, "
          f"{rolled.shape[1]} smoothed trials, "
          f"late mean {trace_mean[late].mean():.4f}")
    # Only the trials each half shows are handed to it: drawing the whole
    # horizon into both and relying on the x-limits to crop leaves every
    # point in the PDF and makes the file enormous.
    for ax, window in [(left, DECAY_LEFT_WINDOW), (right, DECAY_RIGHT_WINDOW)]:
        inside = (trials >= window[0]) & (trials <= window[1])
        ax.plot(trials[inside], trace_mean[inside], color=MODE_COLORS[mode],
                lw=1.0, label=label)
        ax.fill_between(trials[inside], (trace_mean - trace_sem)[inside],
                        (trace_mean + trace_sem)[inside],
                        color=MODE_COLORS[mode], alpha=0.2, lw=0)

left.set_xlim(*DECAY_LEFT_WINDOW)
right.set_xlim(*DECAY_RIGHT_WINDOW)
for ax in (left, right):
    ax.axhline(CHANCE, color='black', lw=0.6, alpha=0.4, ls=':')
    # Trial number in thousands for clarity
    ax.set_xticks(np.arange(ax.get_xlim()[0], ax.get_xlim()[1] + 1, 1000))
    ax.set_xticklabels([f"{int(t / 1000)}" for t in ax.get_xticks()])
    sns.despine(ax=ax)

# The manipulation onset
left.axvline(BASELINE_TRIALS, color='0.4', lw=1.0, ls='--')
left.text(BASELINE_TRIALS, -0.22, 'manipulation\nonset', ha='center', va='top',
          fontsize=FS_LEGEND, color='0.4',
          transform=left.get_xaxis_transform())

# Glue the two halves: drop the facing spines and mark the break.
left.spines['right'].set_visible(False)
right.spines['left'].set_visible(False)
right.tick_params(axis='y', left=False, labelleft=False)
for ax, xs in [(left, (1 - 0.02, 1 + 0.02)), (right, (-0.02, 0.02))]:
    ax.plot(xs, (-0.02, 0.02), transform=ax.transAxes, color='k',
            lw=LINE_WIDTH, clip_on=False)

left.set_ylabel('P(High Rew.\nProb. Choice)', fontsize=FS_LABEL)
fig.supxlabel('Trial (x$10^3$)', fontsize=FS_LABEL, y=0.02)
right.legend(frameon=False, fontsize=FS_LEGEND, loc='upper right')
# The onset caption and the x label sit below the axes, so the margins are
# reserved rather than left to a tight box.
fig.subplots_adjust(left=0.17, right=0.97, top=0.95, bottom=0.34)
save_panel(fig, 'figED11n_performance_decay.pdf', save_dir=SAVE_DIR,
           tight=False)

print("\nDone.")