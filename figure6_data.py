"""
Shared loading, simulation and statistics for Fig. 6 and Extended Data Fig. 11.

Both figures draw on the same Conditioned Reinforcer (sequential actor-critic)
fit, so everything that reads data, runs a simulation or computes a test lives
here and the two figure scripts hold only their panels.

    figure6.py                 6b  6c  6d  6e  6f  6g
    extended_data_figure11.py  11b 11e 11g 11h 11i 11j 11k 11m 11n

Extended Data Fig. 11 c, d, f and l come from other scripts and are not part of
this module.

Both simulators are seeded: `simulate_seqac` from `np.random.seed(RNG_SEED)`,
`run_manipulation_simulations` from `np.random.default_rng(RNG_SEED)`, whose
draws seed every per-repeat generator.
"""

# %%
import matplotlib
import numpy as np
import os
import pandas as pd
import seaborn as sns
import warnings
from matplotlib import font_manager
from pathlib import Path
from scipy.stats import pearsonr, wilcoxon
from statsmodels.stats.multitest import multipletests
import statsmodels.formula.api as smf

# %%
# ============================================================================
# Paths
# ============================================================================
ON_CLUSTER = os.name != "nt"
WITTEN = Path("/jukebox/witten") if ON_CLUSTER else Path(r"\\cup.princeton.edu\witten")

SUBJECTS_ROOT = WITTEN / "Alex" / "Data" / "Subjects"
PAPER_DIR = (Path(__file__).resolve().parent if "__file__" in globals()
             else Path.cwd())
STAN_ROOT = (PAPER_DIR / "data" / "stan_fits"
             if (PAPER_DIR / "data" / "stan_fits").is_dir()
             else WITTEN / "Alex" / "Data" / "ephys_bandit" / "laser_stan_fits")

# The fitted model, named once. Every path below hangs off it.
MODEL_NAME = "sequential_actor_critic_no_awin_m_v_choice"
SEQAC_PARAMS_PKL = STAN_ROOT / MODEL_NAME / "session_params_median.pkl"
SEQAC_PARAMS_PKL_CACHE = ("data", "fits", MODEL_NAME,
                          "session_params_median.pkl")

# The 41 fitted sessions, in the order Stan read them: sorted animal, then
# date, then session. 

STAN_DATA_ROOT = WITTEN / "Alex" / "Data" / "ephys_bandit" / "data_laser_only"

STAN_ORDER = [
    ('dop_47', '2022-06-05', '001', 740),
    ('dop_47', '2022-06-06', '001', 634),
    ('dop_47', '2022-06-09', '003', 853),
    ('dop_47', '2022-06-10', '002', 610),
    ('dop_48', '2022-06-19', '002', 559),
    ('dop_48', '2022-06-20', '001', 517),
    ('dop_48', '2022-06-27', '002', 545),
    ('dop_48', '2022-06-28', '001', 547),
    ('dop_49', '2022-06-14', '001', 566),
    ('dop_49', '2022-06-15', '001', 620),
    ('dop_49', '2022-06-16', '001', 530),
    ('dop_49', '2022-06-17', '001', 491),
    ('dop_49', '2022-06-18', '002', 215),
    ('dop_49', '2022-06-19', '001', 452),
    ('dop_49', '2022-06-20', '001', 428),
    ('dop_49', '2022-06-27', '003', 395),
    ('dop_50', '2022-09-12', '001', 706),
    ('dop_50', '2022-09-13', '001', 632),
    ('dop_50', '2022-09-14', '003', 602),
    ('dop_50', '2022-09-16', '003', 662),
    ('dop_50', '2022-09-17', '001', 493),
    ('dop_50', '2022-09-18', '001', 484),
    ('dop_53', '2022-10-02', '001', 430),
    ('dop_53', '2022-10-03', '001', 664),
    ('dop_53', '2022-10-04', '001', 522),
    ('dop_53', '2022-10-05', '001', 480),
    ('dop_53', '2022-10-07', '001', 521),
    ('dop_56', '2023-09-30', '001', 607),
    ('dop_56', '2023-10-01', '001', 531),
    ('dop_56', '2023-10-02', '001', 518),
    ('dop_56', '2023-10-03', '001', 605),
    ('dop_56', '2023-10-05', '001', 526),
    ('dop_57', '2023-10-17', '002', 449),
    ('dop_57', '2023-10-19', '002', 478),
    ('dop_57', '2023-10-20', '001', 469),
    ('dop_59', '2023-10-23', '003', 520),
    ('dop_59', '2023-10-24', '001', 519),
    ('dop_59', '2023-10-25', '001', 527),
    ('dop_59', '2023-10-26', '001', 557),
    ('dop_59', '2023-10-28', '001', 608),
    ('dop_59', '2023-10-29', '001', 594),
]

STAN_NS, STAN_NSESS = 8, 8

EXAMPLE_SESSION = SUBJECTS_ROOT / "dop_48" / "2022-06-27" / "002"

SAVE_DIR = PAPER_DIR


from ephys_alf_summary import alf, LASER_ONLY_ABB      # noqa: E402
from model_comparison_accu import load_sim_data_reduced  # noqa: E402
from session_maker import TrialParamHandler            # noqa: E402

# `ROOT_FOLDER` is the name `load_sim_data_reduced` and the alf loader expect.
ROOT_FOLDER = str(SUBJECTS_ROOT)

# Teacher-forced predictions saved beside each session's alf files.
MEDIAN_PREFIX = "SeqAC_noawin_m_avchoice_median_"
PI_CHOICE_FILE = MEDIAN_PREFIX + "pi_choice.npy"
P_LICK_FILE = MEDIAN_PREFIX + "p_lick.npy"
FEEDBACK_FILE = "_ibl_trials.feedbackType.npy"
FEEDBACK_CONSUMED_FILE = "_ibl_trials.feedbackTypeConsumed.npy"


# %%
# ============================================================================
# Simulation settings
# ============================================================================
RNG_SEED = 0
N_REPS = 20              # stochastic repeats per session and mode
EXT_TRIALS = 50000       # extended horizon, ED Fig 11n only
BASELINE_TRIALS = 1000   # intact trials before the manipulation switches on
WINDOW = 25              # rolling window for ED Fig 11n

MODES = ['intact', 'laser_off', 'no_cs']

# Trial types for the return-probability panel, in published order.
RETURN_ORDER = ['CS-', 'CS+ laser', 'CS+ skip']
RETURN_LABELS = ['CS-', 'CS+ w/\nVTA stim', 'CS+ w/o\nVTA stim']


# %%
# ============================================================================
# Colours, measured from the typeset panels
# ============================================================================
CYAN = '#4CC3E8'          # control, and the CS+ bars
VIOLET = '#CF98E5'        # violins, and the laser-off trace in 11n
GREY = '0.6'              # CS- and manipulation bars
DARK_GREY = '#888888'     # the no-CS+ trace in 11n
MODEL_COLOR = '#A874C9'   # model curves and points - a different violet
MOUSE_COLOR = 'k'

MODE_COLORS = {'intact': CYAN, 'laser_off': VIOLET, 'no_cs': DARK_GREY}

# --- Figure style -----------------------------------------------------------
FONT_SIZE = 12
LINE_WIDTH = 0.8

# Panel-relative sizes, for the few places a smaller size is deliberate.
FS_TITLE, FS_LABEL, FS_LEGEND = 10, 9, 6


def sans_serif_stack():
    """
    The font list, preferring Arial but degrading quietly where it is absent.

    Cluster nodes rarely have Arial, and matplotlib then emits one `findfont`
    warning per text object. Liberation Sans is metrically compatible and is
    the usual Linux substitute; DejaVu Sans ships with matplotlib.
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


def apply_style():
    """The figure style every panel of Fig. 6 and ED Fig. 11 is drawn in."""
    sns.set_theme(style="ticks", context="notebook")
    matplotlib.rcParams.update({
        "font.size": FONT_SIZE,
        "axes.labelsize": FS_LABEL,
        "axes.titlesize": FS_TITLE,
        "xtick.labelsize": 7,
        "ytick.labelsize": 7,
        "legend.fontsize": FS_LEGEND,
        "figure.titlesize": FONT_SIZE,
        "font.family": "sans-serif",
        "font.sans-serif": SANS_SERIF,
        # These panels are full of mathtext. Under fontset "custom" matplotlib
        # otherwise renders it in DejaVu whatever the text font is, so the
        # Greek would not match the rest of the panel.
        "mathtext.fontset": "custom",
        "mathtext.default": "it",
        "mathtext.rm": SANS_SERIF[0],
        "mathtext.it": f"{SANS_SERIF[0]}:italic",
        "mathtext.bf": f"{SANS_SERIF[0]}:bold",
        # "custom" resolves every slot, `cal` included, which otherwise
        # defaults to "cursive" and warns on any machine without a script face.
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
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "svg.fonttype": "none",
    })


def style_violins(ax, edgecolor='k'):
    """
    Give violins a thin dark outline and keep the median marker white.

    Args:
        ax: Axes holding the violins.
        edgecolor: Outline colour.
    """
    for collection in ax.collections:
        face = collection.get_facecolor()
        if len(face) and tuple(np.round(face[0][:3], 2)) == (1.0, 1.0, 1.0):
            continue                             # the median dot, seaborn 0.12
        collection.set_edgecolor(edgecolor)
        collection.set_linewidth(LINE_WIDTH)
    for line in ax.lines:
        if line.get_marker() not in ("", "None", None):
            line.set_color("w")                  # the median dash, seaborn 0.13
        else:
            line.set_color(edgecolor)


def save_panel(fig, filename, save_dir=None, tight=True):
    """
    Write one panel as a PDF, with the text left editable.

    Args:
        fig: The figure to write.
        filename: Name including the extension.
        save_dir: Destination. Defaults to `SAVE_DIR`, read at call time.
        tight: Crop to the drawn content. Pass False where the figure
            reserves its own margins for unclipped artists, which a tight
            box can cut off.
    """
    save_dir = Path(SAVE_DIR if save_dir is None else save_dir)
    if not save_dir.is_absolute():
        raise ValueError(
            f"save_dir is relative ({save_dir}) - the panels would land in the "
            "working directory. On Linux a UNC path is not absolute: a "
            "backslash is a legal filename character, so a hard-coded "
            r"Path(r'\\cup...') is a relative single-component name.")
    save_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_dir / filename, format="pdf",
                **({"bbox_inches": "tight"} if tight else {}))
    print(f"    wrote {save_dir / filename}")


# %%
# ============================================================================
# Small helpers
# ============================================================================
def inv_logit(x):
    return 1.0 / (1.0 + np.exp(-np.clip(x, -500, 500)))


def stars(p):
    """Significance marker, to the four levels the published panels show."""
    if not np.isfinite(p):
        return ''
    if p >= 0.05:
        return 'n.s.'
    if p >= 0.01:
        return '*'
    if p >= 0.001:
        return '**'
    if p >= 0.0001:
        return '***'
    return '****'


def bracket(ax, x1, x2, y, text, tick=0.03):
    """A significance bracket from x1 to x2 at height y, labelled above."""
    ax.plot([x1, x1, x2, x2], [y, y + tick, y + tick, y],
            color='k', lw=0.8, clip_on=False)
    ax.text((x1 + x2) / 2.0, y + tick, text, ha='center', va='bottom',
            fontsize=FS_LEGEND, clip_on=False)


# %%
# ============================================================================
# Statistics
# ============================================================================
def return_prob_wilcoxon(piv, order=RETURN_ORDER):
    """
    Pairwise Wilcoxon signed-rank across the three trial types.

    Args:
        piv: One row per session, one mean per condition column.
        order: The three column names, in display order.

    Returns:
        `{(i, j): {'W', 'p_raw', 'p_fdr', 'reject'}}` for the pairs (0, 1),
        (0, 2) and (1, 2), keyed by position in `order`.
    """
    pairs = [(0, 1), (0, 2), (1, 2)]
    statistic, raw_p = [], []
    for i, j in pairs:
        a, b = piv[order[i]].to_numpy(), piv[order[j]].to_numpy()
        try:
            w, p = wilcoxon(a, b)
        except ValueError:                 # all differences zero
            w, p = np.nan, 1.0
        statistic.append(w)
        raw_p.append(p)
    reject, p_fdr, _, _ = multipletests(raw_p, method="fdr_bh")
    return {pairs[k]: {"W": statistic[k], "p_raw": raw_p[k],
                       "p_fdr": float(p_fdr[k]), "reject": bool(reject[k])}
            for k in range(len(pairs))}


def skip_perf_lmm(df, y="skip_rate", x="reward_rate", mouse="mouse"):
    """
    Mouse random-intercept linear mixed model, `y ~ x` grouped by mouse.

    Args:
        df: One row per session.
        y, x, mouse: Column names.

    Returns:
        `{'slope', 't', 'p', 'n', 'converged'}`.
    """
    d = df.dropna(subset=[y, x, mouse]).copy()
    d[mouse] = d[mouse].astype(str)
    if d[mouse].nunique() < 2 or len(d) < 4:
        raise ValueError(f"{len(d)} rows over {d[mouse].nunique()} mice is too "
                         "little to fit the mixed model")

    model = smf.mixedlm(f"{y} ~ {x}", d, groups=d[mouse])
    results = model.fit()
    test = results.t_test(np.array([[0, 1]]), use_t=True)
    return {"slope": float(results.params[x]),
            "t": float(np.ravel(test.tvalue)[0]),
            "p": float(np.ravel(test.pvalue)[0]),
            "n": len(d), "converged": bool(results.converged)}


def partial_corr(df, x, y, covars):
    """
    Pearson correlation of x and y after regressing both on `covars`.

    Args:
        df: The table holding all three.
        x, y: Column names to correlate.
        covars: Column names to control for.

    Returns:
        `(r, p)`.
    """
    X = np.column_stack([np.ones(len(df))] + [df[c].values for c in covars])

    def residual(name):
        b, *_ = np.linalg.lstsq(X, df[name].values, rcond=None)
        return df[name].values - X @ b

    return pearsonr(residual(x), residual(y))


def lick_detect_wilcoxon(mouse_df):
    """
    Paired Wilcoxon on the per-session mean predicted P(lick), skip vs collect.

    Args:
        mouse_df: Output of `load_mouse_sessions`.

    Returns:
        `(piv, result)`: the per-session table with columns 0 (skip) and 1
        (collect), and `{'W', 'p', 'n', 'median_skip', 'median_collect'}`.
    """
    cs_plus = (mouse_df[mouse_df['cs_plus'] == 1]
               .dropna(subset=['p_lick_tf', 'licked']).copy())
    per = cs_plus.groupby(['id', 'licked'])['p_lick_tf'].mean().reset_index()
    piv = per.pivot(index='id', columns='licked', values='p_lick_tf')
    if not {0, 1}.issubset(set(piv.columns)):
        raise ValueError("no session has both skipped and collected CS+ trials")
    piv = piv[[0, 1]].dropna()

    w, p = wilcoxon(piv[1], piv[0])
    return piv, {"W": float(w), "p": float(p), "n": int(len(piv)),
                 "median_skip": float(piv[0].median()),
                 "median_collect": float(piv[1].median())}


# %%
# ============================================================================
# Real behaviour
# ============================================================================
def session_path(session, root=SUBJECTS_ROOT):
    """
    One session's directory, from a `LASER_ONLY_ABB` entry.

    Those entries use Windows separators and a leading one, so they are
    normalised and joined rather than concatenated: `root + entry` produces a
    path with backslashes inside it on Linux, where a backslash is an
    ordinary filename character and the read simply fails.

    Args:
        session: Entry from `LASER_ONLY_ABB`.
        root: The Subjects tree.

    Returns:
        Path to the session directory.
    """
    relative = str(session).replace("\\", "/").strip("/")
    return Path(root).joinpath(*relative.split("/"))


def session_alf_path(session, filename, root=SUBJECTS_ROOT):
    """One file in a session's `alf` folder. See `session_path`."""
    return session_path(session, root) / "alf" / filename


CACHE_DIR = PAPER_DIR / "data"


def load_mouse_sessions(sessions=LASER_ONLY_ABB, root=SUBJECTS_ROOT):
    """
    Every fitted session's trial table, with the teacher-forced predictions.

    Args:
        sessions: Session suffixes, in the order Stan was given them.
        root: The Subjects share.

    Returns:
        One row per trial. `id` is the position in `sessions`; `pi_choice_tf`
        and `p_lick_tf` are the median-parameter predictions.
    """
    cache = CACHE_DIR / "figure6" / "mouse_sessions.csv.gz"
    if cache.exists():
        return pd.read_csv(cache)
    frames = []
    for sid, ses_abb in enumerate(sessions):
        df = alf(str(session_path(ses_abb, root))).to_df().copy()

        def load(filename, session=ses_abb):
            return np.load(session_alf_path(session, filename, root))

        # pi_choice is stored post-update: the value entering trial t+1. Panel
        # 11h needs the value that drove the choice on t, so shift by +1 within
        # the session. This touches pi_choice_tf alone - the alf columns the
        # similarity matrix reads stay as saved.
        pi_choice = np.asarray(load(PI_CHOICE_FILE), dtype=float)
        pi_choice = np.concatenate(([np.nan], pi_choice[:-1]))
        p_lick = load(P_LICK_FILE)               # already at decision time
        feedback = load(FEEDBACK_FILE)
        consumed = load(FEEDBACK_CONSUMED_FILE)

        n = min(len(df), len(pi_choice), len(p_lick), len(feedback),
                len(consumed))
        if n != len(df):
            print(f"    {ses_abb}: truncating {len(df)} -> {n} for alignment")
        df = df.iloc[:n].copy()

        df['id'] = sid
        df['session'] = ses_abb.replace('\\', '/').strip('/')
        df['bchoice'] = (df['choice'].to_numpy() > 0).astype(int)
        df['pi_choice_tf'] = pi_choice[:n]
        df['p_lick_tf'] = p_lick[:n]
        df['cs_plus'] = (feedback[:n] > 0).astype(int)
        df['licked'] = (consumed[:n] > 0).astype(int)
        frames.append(df)

    mouse_df = pd.concat(frames, ignore_index=True)
    print(f"    {mouse_df['id'].nunique()} sessions, {len(mouse_df)} trials")
    return mouse_df


def observed_skip_rate(mouse_df):
    """
    Each session's fraction of CS+ trials the mouse did not lick for.

    Args:
        mouse_df: Output of `load_mouse_sessions`.

    Returns:
        `session`, `observed_skip_rate`.
    """
    cs_plus = mouse_df.loc[mouse_df['cs_plus'] == 1]
    skipped = 1.0 - cs_plus.groupby('session')['licked'].mean()
    return skipped.rename('observed_skip_rate').reset_index()


# %%
# ============================================================================
# Fitted parameters
# ============================================================================
SESSION_PARAM_LABELS = {
    "beta_win": "βWin", "beta_loss": "βLoss", "side": "Bias",
    "alpha_loss": "αLoss",
    "V_cs_init": "V(CS)init", "V_choice_init": "V(choice)init",
    "pi_lick_init": "π(lick)init",
}
MOUSE_PARAM_LABELS = {
    "beta_lick": "βLick", "alpha_v_cs": "αV(CS)",
    "alpha_v_choice": "αV(choice)", "alpha_lick": "αLick",
}

# Parameters sharing a y-axis, and the panel each group is drawn in.
#   Fig 6c   Actor β & Bias
#   Fig 6d   Actor decay rates, Critic α
#   ED 11b   Critic init., Actor init.
PARAM_GROUPS = {
    "Actor β & Bias":      ["βWin", "βLoss", "βLick", "Bias"],
    "Actor decay rates":   ["αLick", "αLoss"],
    "Critic α":            ["αV(CS)", "αV(choice)"],
    "Critic init.":        ["V(choice)init", "V(CS)init"],
    "Actor init.":         ["π(lick)init"],
}


def load_pkl(pkl_path=SEQAC_PARAMS_PKL):
    """
    The median posterior parameters.

    Args:
        pkl_path: The Stan fit's `session_params_median.pkl`.

    Returns:
        `sess_p`, `mouse_p`, `sub_idx`, `sess_idx`, `NSxNSESS` and the whole
        pickle under `raw`.
    """
    cache = PAPER_DIR.joinpath(*SEQAC_PARAMS_PKL_CACHE)
    p = pd.read_pickle(cache if cache.exists() else pkl_path)
    return {
        "sess_p": p["session"],
        "mouse_p": p["mouse"],
        "sub_idx": np.asarray(p["sub_idx"]),
        "sess_idx": np.asarray(p["sess_idx"]),
        "NSxNSESS": int(p["NSxNSESS"]),
        "raw": p,
    }


def verify_grid(pk, standata):
    """
    Check all three descriptions of the session grid against each other.

    Indexing the parameter arrays by ms_i is only valid if they agree. The
    three are independent: the pickle was frozen when Stan ran, `standata` is
    rebuilt now from Subjects plus the manifest, and `STAN_ORDER` is the
    frozen directory order. `STAN_ORDER` is what the simulators actually index by, so it is the one
    that has to be right.

    Args:
        pk: Output of `load_pkl`.
        standata: Output of `load_sim_data_reduced` over the same sessions.
    """
    if not (pk["NSxNSESS"] == standata["NSxNSESS"]
            and np.array_equal(pk["sub_idx"], standata["sub_idx"])
            and np.array_equal(pk["sess_idx"], standata["sess_idx"])):
        raise ValueError(
            "the pkl grid (sub_idx / sess_idx / NSxNSESS) does not match the "
            "grid from load_sim_data_reduced over LASER_ONLY_ABB - indexing "
            "parameters by ms_i would misalign. Confirm LASER_ONLY_ABB is "
            "exactly the fitted session set, in the same sorted order.")

    _, frozen_nt, frozen_sub, frozen_sess, _, _, frozen_total = stan_grid(
        pk["raw"])
    real = frozen_nt > 0
    if not (frozen_total == pk["NSxNSESS"]
            and np.array_equal(frozen_sub[real],
                               np.asarray(pk["sub_idx"])[real])
            and np.array_equal(frozen_sess[real],
                               np.asarray(pk["sess_idx"])[real])):
        raise ValueError(
            "STAN_ORDER does not reproduce the pickle's grid on the real "
            "sessions. The frozen order belongs to a different fit, or it "
            "has been edited - re-run freeze_stan_order.py against the Stan "
            "tree.")

    standata_nt = np.asarray(standata["NT_all"], dtype=int)
    if not np.array_equal(frozen_nt, standata_nt):
        differing = np.flatnonzero(frozen_nt != standata_nt)
        detail = ", ".join(f"slot {i}: frozen {frozen_nt[i]}, "
                           f"now {standata_nt[i]}" for i in differing[:5])
        raise ValueError(
            f"{len(differing)} session(s) have a different trial count than "
            f"when STAN_ORDER was frozen ({detail}). The fitted parameters "
            "belong to the frozen counts, so the data has drifted from the "
            "fit - resolve before trusting any panel.")
    print(f"    grid matches; {int((standata['NT_all'] > 0).sum())} sessions")


def load_recovery_data(sessions=LASER_ONLY_ABB):
    """The Stan input grid, as `load_sim_data_reduced` builds it."""
    cache = CACHE_DIR / "figure6" / "stan_grid.npz"
    if cache.exists():
        with np.load(cache) as grid:
            return {key: grid[key] for key in grid.files}
    return load_sim_data_reduced(ROOT_FOLDER=ROOT_FOLDER, trial_start=0,
                                 trial_end=None, sessions=sessions)


def seqac_params_long(pk, standata):
    """
    Every fitted parameter as one long table, for the violin panels.

    Session-level parameters contribute one value per real session,
    mouse-level ones one value per mouse.

    Args:
        pk: Output of `load_pkl`.
        standata: Output of `load_recovery_data`.

    Returns:
        `parameter`, `Mean`.
    """
    sess_p, mouse_p, sub_idx = pk["sess_p"], pk["mouse_p"], pk["sub_idx"]
    real = [i for i in range(standata["NSxNSESS"]) if standata["NT_all"][i] > 0]
    real_mice = sorted({int(sub_idx[i]) - 1 for i in real})

    rows = []
    for key, label in SESSION_PARAM_LABELS.items():
        values = np.asarray(sess_p[key], dtype=float)
        rows += [{"parameter": label, "Mean": values[i]} for i in real]
    for key, label in MOUSE_PARAM_LABELS.items():
        values = np.asarray(mouse_p[key], dtype=float)
        rows += [{"parameter": label, "Mean": values[i]} for i in real_mice]
    return pd.DataFrame(rows)


def param_violins(params, groups, axes, ylabel='Coefficient'):
    """
    One violin group per axis, the parameters within a group sharing a y-axis.

    Args:
        params: Output of `seqac_params_long`.
        groups: Group names, as keys of `PARAM_GROUPS`, one per axis.
        axes: As many axes as groups.
        ylabel: Label for the first axis. None leaves every axis unlabelled,
            which is how the initialization panel is drawn.
    """
    for name, ax in zip(groups, axes):
        labels = PARAM_GROUPS[name]
        subset = params[params['parameter'].isin(labels)]
        if subset.empty:
            raise ValueError(f"no fitted values for the {name} parameters")

        sns.violinplot(data=subset, x='parameter', y='Mean', order=labels,
                       color=VIOLET, saturation=1, cut=0, inner='box',
                       linewidth=LINE_WIDTH, ax=ax)
        style_violins(ax)
        ax.axhline(0, color='0.7', lw=0.6, zorder=0)
        ax.set_title(name, fontsize=8)
        ax.set_xlabel('')
        ax.set_ylabel('')
        ax.tick_params(axis='x', rotation=90, labelsize=6)
        ax.tick_params(axis='y', labelsize=6)
        sns.despine(ax=ax)
    if ylabel is not None:
        axes[0].set_ylabel(ylabel, fontsize=FS_LABEL)


# %%
# ============================================================================
# On-policy simulation (ED Fig 11 g, h, i)
# ============================================================================
def simulate_seqac(standata, pk, seed=RNG_SEED):
    """
    Run the model forward on its own choices, one pass per fitted session.

    The block schedule and the reward draws come from `TrialParamHandler`, the
    same generator the rig used, so the simulated sessions see the same task
    the mice did rather than a replay of one animal's blocks.

    Args:
        standata: Output of `load_recovery_data`, for the session lengths.
        pk: Output of `load_pkl`.
        seed: Seeds `np.random`, which drives both the model's own draws and
            the trial generator.

    Returns:
        One row per simulated trial: `choice`, `bchoice`, `outcome`,
        `probabilityLeft`, `tb`, `pi_choice`, `mouse`, `ses`, `id`.
    """
    np.random.seed(seed)
    NT_all = standata["NT_all"]
    sub_idx = standata["sub_idx"] - 1
    sess_idx = standata["sess_idx"] - 1

    sess_p, mouse_p, pk_sub = pk["sess_p"], pk["mouse_p"], pk["sub_idx"]
    sessions = []

    for ms_i in np.arange(standata["NSxNSESS"]):
        if NT_all[ms_i] <= 0:
            continue
        si = int(pk_sub[ms_i]) - 1

        V_cs = float(sess_p["V_cs_init"][ms_i])
        V_choice = float(sess_p["V_choice_init"][ms_i])
        a_loss = float(sess_p["alpha_loss"][ms_i])
        beta_win = float(sess_p["beta_win"][ms_i])
        beta_loss = float(sess_p["beta_loss"][ms_i])
        side = float(sess_p["side"][ms_i])
        pi_lick = float(sess_p["pi_lick_init"][ms_i])
        a_v_choice = float(mouse_p["alpha_v_choice"][si])
        a_v_cs = float(mouse_p["alpha_v_cs"][si])
        a_lick = float(mouse_p["alpha_lick"][si])
        beta_lick = float(mouse_p["beta_lick"][si])

        pi_choice = 0.0
        tph = TrialParamHandler()

        choices, outcome, side_blocks, tb, pic = [], [], [], [], []
        for _ in np.arange(int(NT_all[ms_i])):
            p_right = inv_logit(side + pi_choice)
            pic.append(pi_choice)
            choice = 1 if np.random.random() < p_right else 0
            sign = 2 * choice - 1
            p_chosen = p_right if choice == 1 else (1.0 - p_right)

            tph.response_side_buffer.append(choice)
            p_left = tph.stim_probability_left
            f = tph.right_reward if choice == 1 else tph.left_reward

            if f == 1:                               # CS+
                delta_cs = V_cs - V_choice
                V_choice += a_v_choice * delta_cs
                pi_choice = (pi_choice
                             + sign * beta_win * delta_cs * (1.0 - p_chosen))
                p_lick = inv_logit(pi_lick)
                licked = 1 if np.random.random() < p_lick else 0
                if licked:
                    delta_laser = 1.0 - V_cs
                    V_cs += a_v_cs * delta_laser
                    pi_lick = ((1.0 - a_lick) * pi_lick
                               + beta_lick * delta_laser * (1.0 - p_lick))
                else:
                    delta_laser = 0.0 - V_cs
                    V_cs += a_v_cs * delta_laser
                    pi_lick = (1.0 - a_lick) * pi_lick
            else:                                    # CS-
                delta_cs = 0.0 - V_choice
                V_choice += a_v_choice * delta_cs
                pi_choice = ((1.0 - a_loss) * pi_choice
                             + sign * beta_loss * delta_cs * (1.0 - p_chosen))

            choices.append(choice)
            outcome.append(f)
            side_blocks.append(p_left)
            tb.append(tph.block_trial_num)
            tph = tph.next_trial()

        sessions.append(pd.DataFrame({
            "choice": choices, "outcome": outcome,
            "probabilityLeft": side_blocks, "tb": tb, "pi_choice": pic,
            "mouse": sub_idx[ms_i], "ses": sess_idx[ms_i],
        }))

    sim = pd.concat(sessions, ignore_index=True)
    sim["bchoice"] = sim["choice"].astype(int)
    sim["id"] = sim["mouse"] * 1000 + sim["ses"]
    print(f"    {sim['id'].nunique()} sessions, {len(sim)} trials")
    return sim


# %%
# ============================================================================
# Manipulation simulation (Fig. 6 e, f, g and ED Fig. 11 k, m, n)
# ============================================================================
def stan_grid(params=None):
    """
    The padded grid implied by `STAN_ORDER`, with each session's flat index.

    The index is `ms_i = ns * NSESS + counter`: `ns` counts animals in the
    order they first appear in `STAN_ORDER`, `counter` counts sessions within
    an animal. That reproduces the fitting pipeline's directory walk exactly,
    because the walk sorted every level and `STAN_ORDER` is frozen in that
    sorted order.

    Args:
        params: The raw pickle. When given, its NS and NSxNSESS are checked
            against `STAN_NS` and `STAN_NSESS`; the pickle wins any argument,
            because it is what the parameter arrays were actually indexed by.

    Returns:
        `(rows, NT_all, sub_idx, sess_idx, NS, NSESS, NSxNSESS)`, where `rows`
        is `(ms_i, animal, date, session, n_trials)` in Stan order and the
        three arrays are the full padded grid.
    """
    NS, NSESS = STAN_NS, STAN_NSESS
    if params is not None:
        pickle_total = int(params["NSxNSESS"])
        pickle_ns = params.get("NS", None)
        if pickle_total != STAN_NS * STAN_NSESS:
            raise ValueError(
                f"the pickle says NSxNSESS={pickle_total}, but STAN_ORDER was "
                f"frozen against NS={STAN_NS} x NSESS={STAN_NSESS}="
                f"{STAN_NS * STAN_NSESS}. The frozen list belongs to a "
                "different fit - re-freeze it, or use the matching pickle.")
        if pickle_ns is not None and int(pickle_ns) != STAN_NS:
            raise ValueError(
                f"the pickle says NS={int(pickle_ns)}, STAN_ORDER has "
                f"{STAN_NS} animals.")

    NSxNSESS = NS * NSESS
    rows, ns, counter, current = [], -1, 0, None
    for animal, date, session, n_trials in STAN_ORDER:
        if animal != current:
            ns += 1
            counter = 0
            current = animal
        if counter >= NSESS:
            raise ValueError(
                f"{animal} has more than NSESS={NSESS} sessions in "
                "STAN_ORDER, so its slots would overflow into the next "
                "animal's block and two sessions would share an ms_i.")
        rows.append((ns * NSESS + counter, animal, date, session, n_trials))
        counter += 1

    if ns + 1 != NS:
        raise ValueError(f"STAN_ORDER holds {ns + 1} animals, STAN_NS is {NS}")

    NT_all = np.zeros(NSxNSESS, dtype=int)
    sub_idx = np.zeros(NSxNSESS, dtype=int)
    sess_idx = np.zeros(NSxNSESS, dtype=int)
    for ms_i, _, _, _, n_trials in rows:
        NT_all[ms_i] = n_trials
        sub_idx[ms_i] = ms_i // NSESS + 1
        sess_idx[ms_i] = ms_i % NSESS + 1
    return rows, NT_all, sub_idx, sess_idx, NS, NSESS, NSxNSESS


def sessions_from_stan_order(subjects_root=SUBJECTS_ROOT, params=None):
    """
    Each fitted session's flat index and path, from the frozen order.

    Args:
        subjects_root: Where the alf files live.
        params: The raw pickle, checked against the frozen grid.

    Returns:
        `(ms_i, alf_path, animal)` per session - the same shape the directory
        walk returned.
    """
    rows, *_ = stan_grid(params)
    return [(ms_i, subjects_root / animal / date / session, animal)
            for ms_i, animal, date, session, _ in rows]


def simulate_session(sinfo, ms, mode, rng, sess_p, mouse_p, sub_idx,
                     return_traces=False, n_trials_override=None,
                     baseline_trials=0):
    """
    Replay one session's block schedule under one manipulation.

    Modes:
        intact      the full model; the lick decision gates stimulation.
        laser_off   delta_laser = 0 - V_cs on every CS+; pi_lick only decays.
        no_cs       the CS+ is removed; delta_cs = 0 on CS+ and V_cs is frozen.

    The first `baseline_trials` always run intact, so every mode shares the
    same pre-manipulation dynamics. The realised trials still differ between
    modes, because `run_manipulation_simulations` draws a separate generator
    per session, mode and repeat.

    Args:
        sinfo: `{'ms_i', 'pLeft'}` for one session.
        ms: The session's ms_i, indexing the parameter arrays.
        mode: One of `MODES`.
        rng: A seeded `np.random.Generator`.
        sess_p, mouse_p, sub_idx: From `load_pkl`.
        return_traces: Also return the per-trial arrays.
        n_trials_override: Tile the block schedule out to this many trials.
        baseline_trials: Intact trials before the manipulation starts.

    Returns:
        `performance`, `skip_rate`, `reward_rate`, `n_rewarded`, and with
        `return_traces` the per-trial `choice`, `outcome`, `licked`,
        `correct`, `cs_plus` and `laser_collected`.
    """
    si = int(sub_idx[ms]) - 1

    V_cs = float(sess_p["V_cs_init"][ms])
    V_choice = float(sess_p["V_choice_init"][ms])
    pi_choice = 0.0
    pi_lick = float(sess_p["pi_lick_init"][ms])

    a_loss = float(sess_p["alpha_loss"][ms])
    beta_win = float(sess_p["beta_win"][ms])
    beta_loss = float(sess_p["beta_loss"][ms])
    side = float(sess_p["side"][ms])

    a_v_choice = float(mouse_p["alpha_v_choice"][si])
    a_v_cs = float(mouse_p["alpha_v_cs"][si])
    a_lick = float(mouse_p["alpha_lick"][si])
    beta_lick = float(mouse_p["beta_lick"][si])

    pLeft = sinfo["pLeft"]
    if n_trials_override is not None:
        n_tiles = int(np.ceil(n_trials_override / len(pLeft)))
        pLeft = np.tile(pLeft, n_tiles)[:n_trials_override]
    n_trials = len(pLeft)

    if return_traces:
        tr_choice = np.zeros(n_trials, dtype=np.int8)
        tr_outcome = np.zeros(n_trials, dtype=np.int8)
        tr_licked = np.full(n_trials, -1, dtype=np.int8)
        tr_correct = np.zeros(n_trials, dtype=np.int8)
        tr_cs_plus = np.zeros(n_trials, dtype=bool)
        tr_lcoll = np.zeros(n_trials, dtype=np.int8)

    n_correct = n_rewarded = n_skipped = n_lickdec = 0

    for t in range(n_trials):
        effective_mode = "intact" if t < baseline_trials else mode

        p_right = inv_logit(side + pi_choice)
        choice = 1 if rng.random() < p_right else 0
        action_sign = 2 * choice - 1
        p_reward = pLeft[t] if choice == 0 else (1.0 - pLeft[t])
        outcome = 1 if rng.random() < p_reward else 0
        high_side = 0 if pLeft[t] > 0.5 else 1
        correct = int(choice == high_side)
        n_correct += correct
        p_chosen = p_right if choice == 1 else (1.0 - p_right)
        licked_flag = -1

        if outcome == 1:
            n_rewarded += 1
            if effective_mode == "no_cs":
                delta_cs = 0.0
                pi_choice = (pi_choice + action_sign * beta_win * delta_cs
                             * (1.0 - p_chosen))
            else:
                delta_cs = V_cs - V_choice
                V_choice += a_v_choice * delta_cs
                pi_choice = (pi_choice + action_sign * beta_win * delta_cs
                             * (1.0 - p_chosen))

                p_lick = inv_logit(pi_lick)
                licked = 1 if rng.random() < p_lick else 0
                licked_flag = licked
                n_lickdec += 1

                if effective_mode == "intact":
                    if licked:
                        delta_laser = 1.0 - V_cs
                        V_cs += a_v_cs * delta_laser
                        pi_lick = ((1.0 - a_lick) * pi_lick
                                   + beta_lick * delta_laser * (1.0 - p_lick))
                    else:
                        delta_laser = 0.0 - V_cs
                        V_cs += a_v_cs * delta_laser
                        pi_lick = (1.0 - a_lick) * pi_lick
                        n_skipped += 1
                else:                                # laser_off
                    delta_laser = 0.0 - V_cs
                    V_cs += a_v_cs * delta_laser
                    pi_lick = (1.0 - a_lick) * pi_lick
                    if not licked:
                        n_skipped += 1
        else:
            delta_cs = 0.0 - V_choice
            V_choice += a_v_choice * delta_cs
            pi_choice = ((1.0 - a_loss) * pi_choice
                         + action_sign * beta_loss * delta_cs
                         * (1.0 - p_chosen))

        if return_traces:
            tr_choice[t] = choice
            tr_outcome[t] = outcome
            tr_licked[t] = licked_flag
            tr_correct[t] = correct
            tr_cs_plus[t] = (outcome == 1)
            tr_lcoll[t] = licked_flag if licked_flag in (0, 1) else 0

    skip_rate = (n_skipped / n_lickdec) if n_lickdec > 0 else np.nan
    out = {"performance": n_correct / n_trials, "skip_rate": skip_rate,
           "reward_rate": n_rewarded / n_trials, "n_rewarded": n_rewarded}
    if return_traces:
        out.update(dict(choice=tr_choice, outcome=tr_outcome, licked=tr_licked,
                        correct=tr_correct, cs_plus=tr_cs_plus,
                        laser_collected=tr_lcoll))
    return out


def repeat_by_condition(choice, feedback, consumed):
    """
    P(next choice == this choice), split by what this trial was.

    Args:
        choice: Per-trial choice.
        feedback: +1 on CS+, -1 on CS-.
        consumed: +1 licked on a CS+, -1 skipped on a CS+, 0 on a CS-.

    Returns:
        One mean per entry of `RETURN_ORDER`.
    """
    nxt = np.roll(choice, -1)
    same = (choice == nxt).astype(float)
    same[-1] = np.nan
    masks = {
        "CS+ laser": (feedback == 1) & (consumed == 1),
        "CS+ skip": (feedback == 1) & (consumed != 1),
        "CS-": (feedback == -1),
    }
    out = {}
    for condition, mask in masks.items():
        values = same[mask & ~np.isnan(same)]
        out[condition] = float(np.nanmean(values)) if values.size else np.nan
    return out


def run_manipulation_simulations(pk, seed=RNG_SEED, run_extended=False):
    """
    Replay every fitted session under all three modes.

    The standard horizon runs at the session's own length and takes seconds;
    it feeds Fig. 6e, f and g and ED Fig. 11k and m. `run_extended` adds the
    `EXT_TRIALS`-trial horizon ED Fig. 11n needs, which is the slow part.

    Args:
        pk: Output of `load_pkl`.
        seed: Seeds the generator every per-repeat generator is drawn from.
        run_extended: Also run the long horizon.

    Returns:
        `sim_mean`     per session and mode, mean over repeats of performance,
                       reward rate and skip rate
        `sim_repeat`   per session, intact P(repeat) by trial type
        `intact`       per session, the intact row plus `V_cs_init`,
                       `pi_lick_init`, `mouse`, `session`
        `ext_arr`      per mode, sessions x trials of mean accuracy, or None
    """
    params = pk["raw"]
    sess_p, mouse_p, sub_idx = pk["sess_p"], pk["mouse_p"], pk["sub_idx"]

    stan_sessions = sessions_from_stan_order(params=params)
    print(f"    {len(stan_sessions)} sessions in Stan order")

    cache = CACHE_DIR / "figure6" / "plefts.npz"
    cached = np.load(cache) if cache.exists() else None
    sessions = []
    for ms_i, alf_path, _ in stan_sessions:
        if cached is not None:
            pLeft = cached[str(ms_i)]
        else:
            pLeft = np.load(os.path.join(str(alf_path), "alf",
                                         "_ibl_trials.probabilityLeft.npy"))
        sessions.append({"ms_i": ms_i, "pLeft": pLeft,
                         "session": "/".join(alf_path.parts[-3:])})

    bad = [s["ms_i"] for s in sessions if s["ms_i"] >= params["NSxNSESS"]]
    if bad:
        raise IndexError(
            f"{len(bad)} session(s) have ms_i >= NSxNSESS "
            f"({params['NSxNSESS']}); first is {bad[0]}. The Stan pickle's "
            "layout no longer matches the directory walk.")

    master_rng = np.random.default_rng(seed)

    # --- standard horizon ---------------------------------------------------
    print("    standard horizon ...")
    records, intact_traces = [], []
    for sinfo in sessions:
        ms = sinfo["ms_i"]
        for mode in MODES:
            performance, skips, rewards = [], [], []
            for _ in range(N_REPS):
                rng = np.random.default_rng(
                    int(master_rng.integers(0, 2**31 - 1)))
                need_traces = (mode == "intact")
                out = simulate_session(sinfo, ms, mode, rng, sess_p, mouse_p,
                                       sub_idx, return_traces=need_traces)
                performance.append(out["performance"])
                rewards.append(out["reward_rate"])
                if not np.isnan(out["skip_rate"]):
                    skips.append(out["skip_rate"])
                if need_traces:
                    intact_traces.append({"ms_i": ms, "choice": out["choice"],
                                          "outcome": out["outcome"],
                                          "licked": out["licked"]})
            records.append({
                "ms_i": ms, "mode": mode,
                "performance": float(np.mean(performance)),
                "reward_rate": float(np.mean(rewards)),
                "skip_rate": float(np.mean(skips)) if skips else np.nan})
    sim_mean = pd.DataFrame(records)

    # --- intact P(repeat) by trial type -------------------------------------
    per_rep = []
    for trace in intact_traces:
        choice, outcome, licked = (trace["choice"], trace["outcome"],
                                   trace["licked"])
        feedback = np.where(outcome == 1, 1, -1)
        consumed = np.where(outcome == 1, np.where(licked == 1, 1, -1), 0)
        per_rep.append({"ms_i": trace["ms_i"],
                        **repeat_by_condition(choice, feedback, consumed)})
    sim_repeat = (pd.DataFrame(per_rep)
                  .groupby("ms_i")[RETURN_ORDER].mean().reset_index())

    # --- the intact row, with the parameters the panels correlate against ----
    intact = sim_mean.loc[sim_mean["mode"] == "intact",
                          ["ms_i", "performance", "reward_rate",
                           "skip_rate"]].copy()
    intact["V_cs_init"] = sess_p["V_cs_init"][intact["ms_i"].values]
    intact["pi_lick_init"] = sess_p["pi_lick_init"][intact["ms_i"].values]
    intact["mouse"] = sub_idx[intact["ms_i"].values] - 1
    intact["session"] = intact["ms_i"].map(
        {s["ms_i"]: s["session"] for s in sessions})

    # --- extended horizon ----------------------------------------------------
    ext_arr = None
    if run_extended:
        print(f"    extended horizon, {EXT_TRIALS} trials per run ...")
        trajectories = {mode: [] for mode in MODES}
        for sinfo in sessions:
            ms = sinfo["ms_i"]
            for mode in MODES:
                reps = []
                for _ in range(N_REPS):
                    rng = np.random.default_rng(
                        int(master_rng.integers(0, 2**31 - 1)))
                    out = simulate_session(sinfo, ms, mode, rng, sess_p,
                                           mouse_p, sub_idx,
                                           return_traces=True,
                                           n_trials_override=EXT_TRIALS,
                                           baseline_trials=BASELINE_TRIALS)
                    reps.append(out["correct"].astype(float))
                trajectories[mode].append(np.mean(reps, axis=0))
        ext_arr = {mode: np.array(v) for mode, v in trajectories.items()}

    print(f"    done: {sim_mean['ms_i'].nunique()} sessions")
    return {"sim_mean": sim_mean, "sim_repeat": sim_repeat, "intact": intact,
            "ext_arr": ext_arr}


def paired_performance(sim_mean, mode):
    """
    Intact and manipulated performance for the sessions present in both.

    Args:
        sim_mean: From `run_manipulation_simulations`.
        mode: The manipulation to pair intact against.

    Returns:
        `(intact, manipulated)`, aligned by session.
    """
    a = sim_mean[sim_mean["mode"] == "intact"].set_index("ms_i")["performance"]
    b = sim_mean[sim_mean["mode"] == mode].set_index("ms_i")["performance"]
    shared = sorted(set(a.index) & set(b.index))
    return a.loc[shared].to_numpy(), b.loc[shared].to_numpy()


# %%
# ============================================================================
# Panel helpers used by both figures
# ============================================================================
def regression_panel(ax, x, y, point_color, line_color, xlabel, ylabel,
                     annotation=None, s=18, ci=95, linestyle='-'):
    """
    Scatter with a linear fit, and optionally its confidence band.

    Args:
        ax: Axes to draw into.
        x, y: The two variables.
        point_color, line_color: Marker and line colours.
        xlabel, ylabel: Axis labels.
        annotation: Text placed above the cloud. None draws nothing.
        s: Marker size.
        ci: Confidence level for the band. None draws the line alone.
        linestyle: Style of the fitted line.
    """
    x, y = np.asarray(x, float), np.asarray(y, float)
    finite = np.isfinite(x) & np.isfinite(y)
    x, y = x[finite], y[finite]
    sns.regplot(x=x, y=y, ax=ax, ci=ci, color=line_color,
                scatter_kws=dict(s=s, color=point_color, alpha=0.9,
                                 edgecolor='none'),
                line_kws=dict(lw=1.2, linestyle=linestyle))
    if annotation:
        ax.set_title(annotation, fontsize=FS_LEGEND)
    ax.set_xlabel(xlabel, fontsize=FS_LABEL)
    ax.set_ylabel(ylabel, fontsize=FS_LABEL)
    sns.despine(ax=ax)


def paired_bar_panel(ax, control, manipulated, labels, ylabel, chance=0.5):
    """
    Two bars, mean over sessions, with a line per session joining them.

    Args:
        ax: Axes to draw into.
        control, manipulated: Per-session values, already paired.
        labels: The two tick labels.
        ylabel: y-axis label.
        chance: Height of the dotted reference line. None draws none.

    Returns:
        `(W, p)` from the paired Wilcoxon.
    """
    control = np.asarray(control, float)
    manipulated = np.asarray(manipulated, float)
    both = np.isfinite(control) & np.isfinite(manipulated)
    control, manipulated = control[both], manipulated[both]

    n = len(control)
    sems = [control.std(ddof=1) / np.sqrt(n), manipulated.std(ddof=1) / np.sqrt(n)]
    ax.bar([0, 1], [control.mean(), manipulated.mean()], yerr=sems,
           color=[CYAN, GREY], edgecolor='black', lw=0.6, width=0.7,
           capsize=3, error_kw=dict(lw=0.8), zorder=2)
    for a, b in zip(control, manipulated):
        ax.plot([0, 1], [a, b], color='0.3', lw=0.4, alpha=0.5, zorder=3)
    if chance is not None:
        ax.axhline(chance, ls=':', color='k', lw=0.8, zorder=1)

    w, p = wilcoxon(control, manipulated)
    top = max(control.max(), manipulated.max(),
              control.mean() + sems[0], manipulated.mean() + sems[1])
    bracket(ax, 0, 1, top + 0.03, stars(p))

    ax.set_xticks([0, 1])
    ax.set_xticklabels(labels, fontsize=7)
    ax.set_xlim(-0.6, 1.6)
    ax.set_ylabel(ylabel, fontsize=FS_LABEL)
    sns.despine(ax=ax)
    return float(w), float(p)