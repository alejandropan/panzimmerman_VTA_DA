"""
Figure 1 - Mice adjust their behavior based on trial-and-error to obtain VTA DA stimulation.

Pan-Vazquez, Zimmerman et al. "VTA dopamine neuron activity produces spatially
organized stimulus and action value representations through conditioned reinforcement."

Run this file top to bottom, or step through it section by section (each `# %%`
marks a cell in VS Code / Spyder). There is no `main()`; every variable stays in
the namespace so panels can be re-plotted or tweaked one at a time.

Panels produced, in manuscript order:
    1c  Delta log-likelihood across RL models, relative to Model 1
    1d  Delta accuracy across RL models, relative to Model 1
    1e  Example session: choices and Q-learning choice predictions
    1f  Example session: Q-learning action values (Q_R, Q_L)
    1g  Win-stay calibration: P(return) after CS+ / CS-
    1h  Policy calibration: P(right) vs. percentile of Q_R - Q_L
    1i  Block-switch calibration: P(high reward prob. choice) vs. trials from switch
    1j  Best-fit Q-learning parameter distributions

Colours and fonts are set once, at the top, from the published figure. Text is
saved as editable text (pdf.fonttype 42) so it can still be adjusted in Illustrator.
"""

# %%
# ============================================================================
# Imports and configuration
# ============================================================================
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib
import matplotlib.pyplot as plt
from scipy.stats import zscore

# --- Paths -------------------------------------------------------------------
# Everything the script reads or writes is declared here. To run this on a
# different machine, edit only this block.
ROOT_FOLDER = r"\\cup.princeton.edu\witten\Alex\Data\Subjects"
PAPER_DIR = (Path(__file__).resolve().parent if "__file__" in globals()
             else Path.cwd())
FORGETTING_FIT_SUMMARY = (PAPER_DIR / "data" / "stan_fits"
                          / "standard_w_forgetting" / "output" / "summary.csv")
if not FORGETTING_FIT_SUMMARY.is_file():
    FORGETTING_FIT_SUMMARY = Path(
        r"\\cup.princeton.edu\witten\Alex\Data\ephys_bandit\laser_stan_fits"
        r"\standard_w_forgetting\output\summary.csv")
SAVE_DIR = PAPER_DIR
SAVE_DIR.mkdir(parents=True, exist_ok=True)
# Analyses that are not in the manuscript are kept, but written somewhere they
# cannot be mistaken for a figure panel.
EXPLORATORY_DIR = SAVE_DIR / "exploratory"
EXPLORATORY_DIR.mkdir(parents=True, exist_ok=True)

from ephys_alf_summary import alf, LASER_ONLY_ABB
from model_comparison_accu import (                                    
    phi_approx,
    load_data_reduced,
    make_stan_data_reduced,
    load_sim_data_reduced,
    simulate_q_learning_w_forgetting,
    stan_data_to_df_reduced,
)

# --- Figure style ------------------------------------------------------------
# Two knobs for the whole figure: one for every piece of text, one for every
# stroke. Nothing below passes an explicit `fontsize=` or `linewidth=`, so
# changing these changes all panels together.
FONT_SIZE = 12
LINE_WIDTH = 0.8

matplotlib.rcParams.update({
    "font.size": FONT_SIZE,
    "axes.labelsize": FONT_SIZE,
    "axes.titlesize": FONT_SIZE,
    "xtick.labelsize": FONT_SIZE,
    "ytick.labelsize": FONT_SIZE,
    "legend.fontsize": FONT_SIZE,
    "figure.titlesize": FONT_SIZE,
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial"],
    "mathtext.fontset": "custom",
    "mathtext.default": "it",
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
COLOR_MOUSE = "k"            # animal data: lines, choice rasters, error bars
COLOR_MODEL = "#CF98E5"      # model predictions and on-policy simulations
COLOR_BAR = "#808080"        # grey bars, and Q_L
COLOR_STIM = "#00C7FF"       # VTA DA stimulation ticks
COLOR_BLOCK = "#D9D9D9"      # shading for blocks where right had higher P(reward)

# --- Model comparison --------------------------------------------------------
# Prefix of the saved `<prefix>_median_choice_prediction.npy` files, mapped to
# the model numbering used in the manuscript. Model 1 is the baseline that all
# deltas in Fig. 1c,d are computed against. The x-axis shows the number only;
# the model names live in the figure's legend box.
MODELS = {
    "QFS":                {"number": 1, "name": "Extended Q-learning (CS+=reward)"},
    "QFS_RPE_laser":      {"number": 2, "name": "Extended Q-learning (VTA stim.=RPE)"},
    "QFS_variable_alpha": {"number": 3, "name": "Extended Q-learning (CS+=reward) - Variable α"},
    "REINFORCE":          {"number": 4, "name": "REINFORCE"},
    "AC":                 {"number": 5, "name": "Actor-Critic"},
}
BASELINE_MODEL = "QFS"

# Y ticks read off the published panels, so re-rendered bars line up with the
# existing Illustrator artwork. Both axes run from their lowest tick to 0.
PANEL_C_YTICKS = [0.00, -0.01, -0.02, -0.03, -0.04, -0.05, -0.06]   # Delta LL
PANEL_D_YTICKS = [0.0, -0.2, -0.4, -0.6, -0.8, -1.0, -1.2, -1.4]    # Delta accuracy (%)

# --- Parameter naming for Fig. 1j -------------------------------------------
# Symbols
BETA_REW = r"$\beta_{rew}$"
BETA_STAY = r"$\beta_{stay}$"
BIAS = "Bias"
ALPHA_REW = r"$\alpha_{rew}$"
ALPHA_STAY = r"$\alpha_{stay}$"
GAMMA_FORGET = r"$\gamma_{forget}$"

# Stan parameter name -> symbol. There is no water reward in this task, so the
# "reward" the betas and alphas refer to is the VTA DA stimulation: `beta_ses` is
# the symbol printed as beta_rew. The `standard_w_forgetting` fit used for Fig. 1
# contains `beta_ses` (not `laser_ses`), and its alphas are already on the unit
# interval, so `plot_params_reduced` is called with the default `phi_a=False`.
PARAM_LABELS = {
    "beta_ses": BETA_REW,
    "stay_ses": BETA_STAY,
    "ep_ses": BETA_STAY,
    "sides": BIAS,
    "alpha_ses": ALPHA_REW,
    "alphalaser_ses": ALPHA_REW,
    "alphastay_ses": ALPHA_STAY,
    "alphaforgetting_ses": GAMMA_FORGET,
    "alphalaserforgetting_ses": GAMMA_FORGET,
    # Parameters that appear only in models 2-5 (Extended Data Fig. 2e-h).
    "laser_ses": "βLaser",
    "betalaserbase_ses": "βLaserBase",
    "dmlaser_ses": "βRewardLoss",
    "dmwater_ses": "βRewardLoss", #dmwater naiming is legacy 
    # from a previous task, that's why we are renaming it
    "laserstay_ses": "βRewardStay",
    "alphalaserstay_ses": "αStayReward",
    "alphaloss_ses": "αRewardLoss",
    "alphalaserloss_ses": "αRewardLoss",
    "laserdecay_ses": "αRewardDecay",
    "laserdecayloss_ses": "αRewardDecayLoss",
    "balpha_ses": "αBReward",
    "balphalaser_ses": "αBReward",
    "balphastay_ses": "αBStay",
    "lapse_ses": "ε",
}

# The two parameter groups shown side by side in Fig. 1j. They are split because
# the betas and the learning rates live on very different scales.
BETA_PARAMS = [BETA_REW, BETA_STAY, BIAS]
ALPHA_PARAMS = [ALPHA_REW, ALPHA_STAY, GAMMA_FORGET]

# Number of pre-switch trials shown to the left of a block switch in Fig. 1i.
N_PRE_SWITCH_TRIALS = 10

# --- Example session for Fig. 1e,f -------------------------------------------
EXAMPLE_SESSION = 13                  # index into LASER_ONLY_ABB
# The published panels show a window of the session, not all of it
EXAMPLE_TRIAL_WINDOW = (60, 300)      # (first trial, last trial) as an x limit


# %%
# ============================================================================
# Helpers
# ============================================================================
def save_panel(filename, directory=SAVE_DIR, tight=True):
    """
    Tidy, save the current figure as a vector PDF, and close it.

    Args:
        filename: File name, including the .pdf extension.
        directory: Destination directory. Defaults to SAVE_DIR.
        tight: Run `tight_layout` first. Set False for the multi-row example
            session figures, whose fixed height ratios it cannot handle.
    """
    if tight:
        plt.tight_layout()
    plt.savefig(directory / filename, format="pdf", bbox_inches="tight")
    plt.close()


def style_error_bars(ax, color=COLOR_MOUSE):
    """
    Recolour and thin the error bars seaborn drew on a bar plot.

    Done after the fact rather than through `err_kws`, which only exists in
    seaborn >= 0.13 and is silently forwarded to matplotlib (and raises) on 0.12.

    Args:
        ax: Axes containing the bar plot.
        color: Error bar colour.
    """
    for line in ax.lines:
        line.set_color(color)
        line.set_linewidth(LINE_WIDTH)


def style_violins(ax, edgecolor=COLOR_MOUSE):
    """
    Give violins a thin dark outline.

    Equivalent to `linecolor=`, which only exists in seaborn >= 0.13.

    Args:
        ax: Axes containing the violin plot.
        edgecolor: Outline colour.
    """
    for collection in ax.collections:
        collection.set_edgecolor(edgecolor)
        collection.set_linewidth(LINE_WIDTH)


def align_to_block_switches(data, n_pre=N_PRE_SWITCH_TRIALS):
    """
    Annotate each trial with its position relative to the nearest block switch.

    A block switch is a trial where `probabilityLeft` changes. Trials are indexed
    from 0 at the switch onwards, and from -n_pre to -1 for the trials leading up
    to a switch. Trials before the first switch of a session are dropped.

    Args:
        data: Trial table with `id` (session) and `probabilityLeft` columns,
            ordered by trial within each session.
        n_pre: How many pre-switch trials to label with negative indices.

    Returns:
        Concatenated trial table with two added columns:
            t_w_block                    trial index within block (0 at switch)
            t_w_block_including_neg      -n_pre..-1 for pre-switch trials, else NaN
    """
    aligned = []
    for ses in data["id"].unique():
        ses_data = data.loc[data["id"] == ses].copy().reset_index(drop=True)
        switches = np.where(np.diff(ses_data["probabilityLeft"]) != 0)[0] + 1
        if len(switches) == 0:
            continue

        # Trial index within block: restart the count at every switch.
        block_id = np.cumsum(np.isin(np.arange(len(ses_data)), switches))
        ses_data["t_w_block"] = ses_data.groupby(block_id).cumcount().astype(float)
        first_block = np.flatnonzero(block_id == 0)
        ses_data.loc[first_block, "t_w_block"] = np.arange(1, len(first_block) + 1)

        # Negative indices for the n_pre trials immediately before each switch.
        ses_data["t_w_block_including_neg"] = np.nan
        for switch in switches:
            start = max(switch - n_pre, 0)
            ses_data.loc[start:switch - 1, "t_w_block_including_neg"] = np.arange(
                start - switch, 0
            )

        # Drop the run-in before the first switch, keeping n_pre trials of context.
        aligned.append(ses_data.iloc[max(switches[0] - n_pre, 0):])

    return pd.concat(aligned).reset_index(drop=True)


def compare_models(sessions, root_folder=ROOT_FOLDER, models=MODELS):
    """
    Collect per-session choice-prediction accuracy and log-likelihood per model.

    Reads the pre-computed `<prefix>_median_choice_prediction.npy` written by the
    Stan fitting pipeline, one file per model per session.

    Args:
        sessions: Session paths relative to `root_folder` (e.g. LASER_ONLY_ABB).
        root_folder: Directory the relative session paths hang off.
        models: Mapping of file prefix -> {"number", "name"}.

    Returns:
        DataFrame with one row per (session, model): mean log-likelihood per
        choice (`lchoice`) and mean accuracy (`acc`).
    """
    rows = []
    for ses in sessions:
        # Note: LASER_ONLY_ABB entries begin with a separator, so concatenate
        # rather than using os.path.join.
        ses_path = root_folder + ses + r"\alf"
        for prefix, meta in models.items():
            prediction_file = Path(ses_path) / f"{prefix}_median_choice_prediction.npy"
            if not prediction_file.exists():
                raise FileNotFoundError(
                    f"Missing prediction for model {meta['number']} "
                    f"('{prefix}'): {prediction_file}"
                )
            p_choice = np.load(prediction_file)
            rows.append(
                pd.DataFrame(
                    {
                        "ses": ses_path,
                        "model": prefix,
                        "number": meta["number"],
                        "lchoice": np.log(p_choice),
                        "acc": 1 * (p_choice > 0.5),
                    }
                )
            )

    trials = pd.concat(rows)
    return (
        trials.groupby(["ses", "model", "number"])[["lchoice", "acc"]]
        .mean()
        .reset_index()
    )


def add_deltas_vs_baseline(model_summary, baseline=BASELINE_MODEL):
    """
    Express each model's accuracy and log-likelihood relative to a baseline model.

    Args:
        model_summary: Output of `compare_models`.
        baseline: Model prefix to subtract, per session.

    Returns:
        Copy of `model_summary` with `delta_ll` and `delta_acc` columns added.
        `delta_acc` is in percentage points.
    """
    out = model_summary.copy()
    reference = out.loc[out["model"] == baseline].set_index("ses")
    out["delta_ll"] = out["lchoice"] - out["ses"].map(reference["lchoice"])
    out["delta_acc"] = 100 * (out["acc"] - out["ses"].map(reference["acc"]))
    return out


# %%
# ============================================================================
# Panel plotting functions
# ============================================================================
def plot_model_delta(ll_comparison, metric, ylabel, yticks, tick_format):
    """
    Fig. 1c,d - one model-comparison bar plot.

    Args:
        ll_comparison: Output of `add_deltas_vs_baseline`.
        metric: Column to plot, `delta_ll` or `delta_acc`.
        ylabel: Y-axis label.
        yticks: Y tick positions.
        tick_format: Printf-style format for the tick labels, e.g. "%.2f".
    """
    fig, ax = plt.subplots(figsize=(2.2, 2.2))
    sns.barplot(
        data=ll_comparison, x="number", y=metric,
        order=[m["number"] for m in MODELS.values()],
        errorbar="se", color=COLOR_BAR, ax=ax,
    )
    style_error_bars(ax)
    ax.set_ylim(min(yticks), max(yticks))
    plt.ylabel(ylabel)
    plt.xlabel("Model")
    sns.despine(ax=ax, offset=3)
    ax.spines["left"].set_bounds(min(yticks), max(yticks))
    ax.set_yticks(yticks)
    ax.set_yticklabels([tick_format % tick for tick in yticks])


def draw_example_session(example, ylabel, trial_window=EXAMPLE_TRIAL_WINDOW):
    """
    Set up the shared scaffolding for the two example-session panels, Fig. 1e,f.

    Builds a five-row figure - VTA stimulations and right choices above the trace,
    left choices and their stimulations below - then draws the rasters, the block
    shading and the 50-trial scale bar. The caller plots its own traces on the
    returned main axes.

    Args:
        example: Trial table from `prepare_example_session`.
        ylabel: Y-axis label for the main axes.
        trial_window: `(first, last)` trials to show, or None for the whole
            session. Applied as an x limit on the shared axis, so every row is
            windowed together.

    Returns:
        The main axes, with y limits 0-1 and no x-axis.
    """
    fig, axes = plt.subplots(
        5, 1, figsize=(7, 3.2), sharex=True,
        gridspec_kw={"height_ratios": [0.4, 0.4, 6, 0.4, 0.4], "hspace": 0.4},
    )
    ax_stim_r, ax_choice_r, ax_main, ax_choice_l, ax_stim_l = axes

    raster_rows = [
        (ax_stim_r, "stim_r", COLOR_STIM, "VTA stimulations"),
        (ax_choice_r, "choice_r", COLOR_MOUSE, "Right Choices"),
        (ax_choice_l, "choice_l", COLOR_MOUSE, "Left Choices"),
        (ax_stim_l, "stim_l", COLOR_STIM, "VTA stimulations"),
    ]
    for axis, column, color, label in raster_rows:
        axis.vlines(np.where(example[column] == 1), 0, 1, color=color, linewidth=LINE_WIDTH)
        axis.set_ylabel(label, rotation=0, ha="right", va="center")
        axis.set_yticks([])
        axis.set_xticks([])
        axis.set_axis_off()
        # set_axis_off also hides the label, so re-attach it as free text.
        axis.text(-0.01, 0.5, label, transform=axis.transAxes, ha="right", va="center")

    # Blocks where the right choice had the higher probability of reward.
    ax_main.fill_between(
        x=example.index, y1=0, y2=1,
        where=example["probabilityLeft"] == 0.1,
        color=COLOR_BLOCK, linewidth=0, zorder=0,
    )
    ax_main.set_ylim(0, 1)
    ax_main.set_yticks(np.arange(0, 1.1, 0.2))
    ax_main.set_ylabel(ylabel)
    ax_main.set_xticks([])
    ax_main.spines[["top", "right", "bottom"]].set_visible(False)

    # Show only part of the session. sharex propagates this to the raster rows.
    if trial_window is not None:
        ax_main.set_xlim(*trial_window)

    # 50-trial scale bar, in place of an x-axis, at the right of whatever is shown.
    x_end = ax_main.get_xlim()[1]
    ax_stim_l.plot([x_end - 50, x_end], [-1.5, -1.5], color=COLOR_MOUSE, clip_on=False)
    ax_stim_l.text(x_end - 25, -2.2, "50 Trials", ha="center", va="top")
    return ax_main


def plot_example_choices(example): 
    """
    Fig. 1e - choices and Q-learning choice predictions for one session.

    Args:
        example: Trial table for a single session, from `prepare_example_session`.
    """
    ax = draw_example_session(example, "Fraction of Right\nChoices / Prediction")
    ax.plot(example["choice_r"].rolling(5).mean(), color=COLOR_MOUSE, label="Mouse", zorder=2)
    ax.plot(example["p_right"].rolling(5).mean(), color=COLOR_MODEL, label="Model 1", zorder=3)
    ax.legend(frameon=False, loc="lower left", bbox_to_anchor=(0, 1.35), ncol=2)


def plot_example_action_values(example):
    """
    Fig. 1f - Q-learning action value estimates for the session shown in Fig. 1e.

    Args:
        example: Trial table for a single session, from `prepare_example_session`.
    """
    ax = draw_example_session(example, "Q-Values")
    ax.plot(example["QR"], color=COLOR_MOUSE, label="$Q^R$", zorder=3)
    ax.plot(example["QL"], color=COLOR_BAR, label="$Q^L$", zorder=2)
    ax.legend(frameon=False, loc="lower left", bbox_to_anchor=(0, 1.35), ncol=2)


def plot_return_rate(data, color):
    """
    Fig. 1g - probability of repeating the previous choice, split by previous outcome.

    Args:
        data: Trial table with `id`, `choice`, `outcome`, `probabilityLeft`
            and `tb` (trial within block) columns.
        color: Bar colour. Individual sessions are always drawn in black.
    """
    # "Previous trial" must mean the previous trial in that session, so both
    # shifts happen on the full session, grouped by id, before any trimming.
    data = data.copy()
    by_session = data.groupby("id")
    data["repeat"] = 1 * (data["choice"] == by_session["choice"].shift(1))
    data["prev_outcome"] = by_session["outcome"].shift(1)

    transition_data = align_to_block_switches(data)
    transition_data = transition_data.loc[transition_data["tb"] > 10].copy()

    per_session = (
        transition_data.groupby(["id", "prev_outcome"])["repeat"].mean().reset_index()
    )
    ax = plt.gca()
    sns.barplot(
        data=per_session, x="prev_outcome", y="repeat",
        color=color, errorbar="se", ax=ax,
    )
    # Style the bars' error bars before adding the per-session lines, so the
    # thinner session lines are not caught by it.
    style_error_bars(ax)
    sns.lineplot(
        data=per_session, x="prev_outcome", y="repeat",
        units="id", estimator=None, color=COLOR_MOUSE, linewidth=LINE_WIDTH / 2, ax=ax,
    )
    plt.ylim(0, 1)
    plt.yticks(np.arange(0, 1.1, 0.2))
    plt.ylabel("Return probability")
    plt.xlabel("")
    plt.xticks([0, 1], ["After CS-", "After CS+"], rotation=45, ha="right")
    sns.despine()


def plot_transition(data, color):
    """
    Fig. 1i - probability of choosing the higher-reward-probability action,
    aligned to reward-contingency reversals.

    Args:
        data: Trial table with `id`, `probabilityLeft` and `choice` columns.
        color: Line colour.
    """
    transition_data = align_to_block_switches(data)

    # Duplicate the pre-switch trials under the negative time index so they
    # appear to the left of 0 without being removed from their own block.
    neg_trials = transition_data.loc[
        transition_data["t_w_block_including_neg"].notna()
    ].copy()
    neg_trials["t_w_block"] = neg_trials["t_w_block_including_neg"]
    transition_data = pd.concat([transition_data, neg_trials])

    # Recode "chose right" as "chose the high-reward-probability side".
    left_is_high = transition_data["probabilityLeft"] == 0.7
    transition_data.loc[left_is_high, "choice"] = 1 * (
        transition_data.loc[left_is_high, "choice"] == 0
    )

    per_session = (
        transition_data.groupby(["id", "t_w_block"])[["choice"]].mean().reset_index()
    )
    sns.lineplot(
        data=per_session, x="t_w_block", y="choice",
        errorbar="se", err_style="bars", color=color,
    )
    plt.xlim(-N_PRE_SWITCH_TRIALS, 25)
    plt.ylim(0, 1)
    plt.yticks(np.arange(0, 1.1, 0.2))
    plt.vlines(0, 0, 1, linestyles="dashed", color=COLOR_MOUSE)
    plt.ylabel("P(High Rew. Prob. Choice)")
    plt.xlabel("Trials from block switch")
    sns.despine()


def plot_params_reduced(saved_params, standata, phi_a=False):
    """
    Fig. 1j - distribution of best-fit model parameters across sessions.

    Args:
        saved_params: Stan summary table with `name` and `Mean` columns.
        standata: Stan data dict, used to drop padding (dud) sessions.
        phi_a: If True, map learning-rate parameters through the phi
            approximation to return them to the unit interval.

    Returns:
        Tidy DataFrame of per-session parameter estimates that was plotted.
    """
    # Keep only the session slots that carry real trials.
    real_slots = [
        f"[{i + 1}]" for i in np.arange(standata["NSxNSESS"]) if standata["NT_all"][i] > 0
    ]
    is_real = saved_params["name"].str.endswith(tuple(real_slots))
    is_session_level = saved_params["name"].str.contains(r"s\[", regex=True)

    params = saved_params.loc[is_real & is_session_level, ["name", "Mean"]].copy()
    params["parameter"] = (
        params["name"].str.replace(r"\[\d+\]$", "", regex=True).map(PARAM_LABELS)
    )
    params = params.dropna(subset=["parameter"])

    # Two different Stan parameters mapping to the same symbol get pooled into

    stan_names = params["name"].str.replace(r"\[\d+\]$", "", regex=True)
    collisions = (
        pd.DataFrame({"stan": stan_names, "symbol": params["parameter"]})
        .drop_duplicates()
        .groupby("symbol")["stan"]
        .agg(list)
    )
    collisions = collisions[collisions.map(len) > 1]
    if len(collisions):
        warnings.warn(
            f"These symbols map to more than one Stan parameter and will be "
            f"pooled into a single violin: {collisions.to_dict()}. "
            f"Split them in PARAM_LABELS if that is not intended."
        )

    if phi_a:
        is_alpha = params["parameter"].str.contains(r"alpha|\u03b1", regex=True)
        params.loc[is_alpha, "Mean"] = phi_approx(params.loc[is_alpha, "Mean"] / np.sqrt(2))

    fig, ax = plt.subplots(1, 2, figsize=(4.5, 2.6))
    for axis, group, title in zip(
        ax, [BETA_PARAMS, ALPHA_PARAMS], [r"$\beta$ & $Bias$", r"$\alpha$ & $\gamma$"]
    ):
        present = [p for p in group if p in set(params["parameter"])]
        subset = params.loc[params["parameter"].isin(present)]
        plt.sca(axis)
        sns.violinplot(
            data=subset, x="parameter", y="Mean", order=present, cut=0,
            color=COLOR_MODEL, inner="box", linewidth=LINE_WIDTH, ax=axis,
        )
        style_violins(axis)
        plt.title(title)
        plt.xlabel("")
        plt.ylabel("Coefficients" if axis is ax[0] else "")
    sns.despine()
    return params


def prepare_example_session(session_path, beta_scale=True):
    """
    Load one session and derive the traces plotted in Fig. 1e,f.

    Args:
        session_path: Full path to the session directory containing `alf/`.
        beta_scale: If True, divide the Q-values by the mouse-level inverse
            temperature so they are on the same scale as the reward magnitude.

    Returns:
        Trial table with the columns used by `plot_example_choices` and
        `plot_example_action_values`.
    """
    example = alf(session_path).to_df()
    example_params = pd.read_pickle(session_path + r"\alf\forgettingmodel_params.pkl")

    example["choice_r"] = 1 * (example["choice"] == 1)
    example["choice_l"] = 1 * (example["choice"] == -1)

    # VTA DA stimulation was delivered on rewarded trials in laser blocks, and
    # only counted when the animal actually consumed it.
    laser_consumed = 1 * (
        np.load(session_path + r"\alf\_ibl_trials.feedbackTypeConsumed.npy") > 0
    )
    stim = example["outcome"] * (1 * (example["opto_block"] != 1)) * laser_consumed # * opto_block legacy from when there was different types of blocks
    example["stim_r"] = stim * example["choice_r"]
    example["stim_l"] = stim * example["choice_l"]

    example["QL"] = example["fQLreward"]
    example["QR"] = example["fQRreward"]
    if beta_scale:
        example["QL"] = example["QL"] / example_params.beta_mouse[0]
        example["QR"] = example["QR"] / example_params.beta_mouse[0]

    # `fchoice_prediction` is P(chosen action); convert it to P(right).
    example["p_right"] = example["fchoice_prediction"]
    example.loc[example["choice"] == -1, "p_right"] = (
        1 - example.loc[example["choice"] == -1, "fchoice_prediction"]
    )

    return example


# %%
# ============================================================================
# Load data and run on-policy simulations
# ============================================================================
CACHE_DIR = PAPER_DIR / "data" / "figure1"
CACHE_BEHAVIOUR = CACHE_DIR / "behaviour.csv.gz"
if CACHE_BEHAVIOUR.exists():
    data = pd.read_csv(CACHE_BEHAVIOUR)
else:
    data = load_data_reduced(
        ROOT_FOLDER=ROOT_FOLDER, trial_start=0, trial_end=None, sessions=LASER_ONLY_ABB
    )
standata = make_stan_data_reduced(data)

model_forgetting = pd.read_csv(FORGETTING_FIT_SUMMARY).rename(columns={"Unnamed: 0": "name"})

STAN_GRID = PAPER_DIR / "data" / "figure6" / "stan_grid.npz"
if STAN_GRID.exists():
    with np.load(STAN_GRID) as grid:
        standata_recovery = {key: grid[key] for key in grid.files}
else:
    standata_recovery = load_sim_data_reduced(
        ROOT_FOLDER=ROOT_FOLDER, trial_start=0, trial_end=None,
        sessions=LASER_ONLY_ABB
    )
_, _, sim_forgetting = simulate_q_learning_w_forgetting(
    standata_recovery, saved_params=model_forgetting
)

realdata = stan_data_to_df_reduced(standata_recovery, standata)
# The loader inherits column names from an earlier water-reward version of the
# task; here the positive outcome is the CS+ paired with VTA DA stimulation.
realdata = realdata.rename(columns={"water": "outcome", "choices": "choice"})
sim_forgetting = sim_forgetting.rename(columns={"choices": "choice"})
realdata["id"] = realdata["mouse"] * 1000 + realdata["ses"]
sim_forgetting["id"] = sim_forgetting["mouse"] * 1000 + sim_forgetting["ses"]


# %%
# ============================================================================
# Fig. 1c,d - Model comparison
# ============================================================================
COMPARISON_CACHE = CACHE_DIR / "model_comparison.csv.gz"
if COMPARISON_CACHE.exists():
    model_summary = pd.read_csv(COMPARISON_CACHE)
else:
    model_summary = compare_models(LASER_ONLY_ABB)
ll_comparison = add_deltas_vs_baseline(model_summary)

plot_model_delta(ll_comparison, "delta_ll", "Δ LL", PANEL_C_YTICKS, "%.2f")
save_panel("fig1c_delta_loglikelihood.pdf")

plot_model_delta(ll_comparison, "delta_acc", "Δ Accuracy (%)", PANEL_D_YTICKS, "%.1f")
save_panel("fig1d_delta_accuracy.pdf")


# %%
# ============================================================================
# Fig. 1e,f - Example session
# ============================================================================
EXAMPLE_CACHE = CACHE_DIR / "example_session.csv.gz"
if EXAMPLE_CACHE.exists():
    example = pd.read_csv(EXAMPLE_CACHE, index_col=0)
else:
    example = prepare_example_session(
        ROOT_FOLDER + LASER_ONLY_ABB[EXAMPLE_SESSION])

plot_example_choices(example)
save_panel("fig1e_example_session_choices.pdf", tight=False)

plot_example_action_values(example)
save_panel("fig1f_example_session_action_values.pdf", tight=False)


# %%
# ============================================================================
# Fig. 1g - Win-stay calibration
# ============================================================================
fig, ax = plt.subplots(1, 2, figsize=(3.4, 2.6), sharey=True)
plt.sca(ax[0])
plot_return_rate(realdata, color=COLOR_BAR)
plt.title("Mouse")
plt.sca(ax[1])
plot_return_rate(sim_forgetting, color=COLOR_MODEL)
plt.title("Model")
plt.ylabel("")
save_panel("fig1g_return_rate.pdf")


# %%
# ============================================================================
# Fig. 1h - Choice policy vs. relative action value
# ============================================================================
TRIALS_CACHE = CACHE_DIR / "session_trials.csv.gz"
if TRIALS_CACHE.exists():
    fulldf = pd.read_csv(TRIALS_CACHE)
else:
    fulldf = pd.DataFrame()
    for idx, ses_abb in enumerate(LASER_ONLY_ABB):
        df = alf(ROOT_FOLDER + ses_abb).to_df()
        df["ses_id"] = idx
        df["deltaq"] = df["fQR"] - df["fQL"]
        df["decision_time"] = df["response_times"] - df["goCue_trigger_times"]
        df["zdecision_time"] = zscore(df["decision_time"], nan_policy="omit")
        fulldf = pd.concat([fulldf, df])
    fulldf = fulldf.reset_index(drop=True)

fulldf["bchoice"] = 1 * (fulldf["choice"] > 0)
# QR-QL at the end of trial t drives the choice on trial t+1, hence the shift.
fulldf["deltaq_decile"] = pd.qcut(
    fulldf["deltaq"], 10, labels=np.arange(10)
).shift(1)
sim_forgetting["deltaq"] = sim_forgetting["QR"] - sim_forgetting["QL"]
sim_forgetting["deltaq_decile"] = pd.qcut(sim_forgetting["deltaq"], 10, labels=np.arange(10))

fig, ax = plt.subplots(figsize=(2.6, 2.6))
sns.lineplot(
    data=sim_forgetting.groupby(["id", "deltaq_decile"])["choice"].mean().reset_index(),
    x="deltaq_decile", y="choice", errorbar="se", color=COLOR_MODEL, label="Model",
)
sns.lineplot(
    data=fulldf.groupby(["ses_id", "deltaq_decile"])["bchoice"].mean().reset_index(),
    x="deltaq_decile", y="bchoice", errorbar="se", color=COLOR_MOUSE, label="Mouse",
)
plt.ylabel("P(Right)")
plt.xlabel("$Q^R - Q^L$ Percentile")
# Deciles are plotted at 0-9; label them by the percentile they close.
plt.xticks([0, 4, 9], ["10", "50", "100"])
plt.legend(frameon=False, loc="upper left")
sns.despine()
save_panel("fig1h_policy_vs_relative_value.pdf")


# %%
# ============================================================================
# Fig. 1i - Block-switch calibration
# ============================================================================
fig, ax = plt.subplots(figsize=(3.0, 2.6))
plot_transition(sim_forgetting, color=COLOR_MODEL)
plot_transition(realdata, color=COLOR_MOUSE)
plt.legend(
    handles=[
        plt.Line2D([], [], color=COLOR_MODEL, label="Model"),
        plt.Line2D([], [], color=COLOR_MOUSE, label="Mouse"),
    ],
    frameon=False, loc="lower right",
)
save_panel("fig1i_transition_plot.pdf")


# %%
# ============================================================================
# Fig. 1j - Best-fit model parameters
# ============================================================================
params = plot_params_reduced(model_forgetting, standata)
save_panel("fig1j_model_parameters.pdf")


# %%
# ============================================================================
# Exploratory - not in the manuscript
# ============================================================================
# Chosen action value vs. decision time. This panel is not in Figure 1 or in any
# Extended Data figure; Extended Data Fig. 10a shows decision time split by
# outcome, not by Q-chosen decile. Delete this cell if it is not needed.
fulldf["qchosen"] = np.where(fulldf["bchoice"] == 1, fulldf["fQR"], fulldf["fQL"])
fulldf["qchosen_decile"] = pd.qcut(fulldf["qchosen"], 10, labels=np.arange(10))

fig, ax = plt.subplots(figsize=(2.6, 2.6))
sns.lineplot(
    data=fulldf.groupby(["ses_id", "qchosen_decile"])["zdecision_time"].mean().reset_index(),
    x="qchosen_decile", y="zdecision_time", errorbar="se", color=COLOR_BAR,
)
plt.ylabel("Decision Time (σ)")
plt.xlabel("$Q^{chosen}$ Percentile")
plt.xticks([0, 4, 9], ["10", "50", "100"])
sns.despine()
save_panel("qchosen_decision_time.pdf", directory=EXPLORATORY_DIR)