"""
Figure 6 b-g - the Conditioned Reinforcer model's fit and its predictions.

    b  model variables over one representative session
    c  fitted actor weights and bias
    d  fitted critic learning rates and actor decay rates
    e  return probability by trial type, from the on-policy simulation
    f  skipped CS+ against CS+ rate, across simulated sessions
    g  performance with the CS+ removed in silico

Panels h-q of the published figure are the ephys encoding maps and are not
part of this script.

Everything substantive is in `figure6_data.py`.

Usage (interactive), or:
    python figure6.py
"""

# %%
import matplotlib.lines as mlines
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from figure6_data import (CYAN, EXAMPLE_SESSION, FS_LABEL, FS_LEGEND, GREY,
                          PAPER_DIR, RETURN_LABELS, RETURN_ORDER, VIOLET,
                          apply_style, bracket, load_pkl, load_recovery_data,
                          paired_bar_panel, paired_performance, param_violins,
                          regression_panel, return_prob_wilcoxon,
                          run_manipulation_simulations, save_panel,
                          seqac_params_long, skip_perf_lmm, stars,
                          verify_grid)
from ephys_alf_summary import alf

# %%
# ============================================================================
# Panel 6b configuration
# ============================================================================
B_XLIM = (0, 331)
B_SCALE_BAR_TRIALS = 50

B_TRACES = [
    ('seq_ac_propensity', 'red',       r'$\pi^{choice}$ - Left / Right',
     (-3.08, 3.08), [-2, 0, 2]),
    ('seq_ac_vcs',        CYAN,        r'$V(S_{CS+})$',
     (0.900, 0.984), [0.90, 0.92, 0.94, 0.96, 0.98]),
    ('seq_ac_vchoice',    VIOLET, r'$V(S_{choice})$',
     (0.200, 0.505), [0.2, 0.3, 0.4, 0.5]),
]
B_SPINE_STEP = 34        # points between the stacked right-hand axes

# The event rasters, drawn outside the axes. Each entry is
# (column, colour, offset from the axis edge, tick length), in axis-fraction
# units; the choice raster sits closest to the axes, then the CS+, then the
# stimulation, as the published panel stacks them.
B_RASTERS = [
    ('choice_side', 'k',  0.04, 0.09),
    ('cs_plus',     VIOLET, 0.15, 0.05),
    ('vta_stim',    CYAN, 0.22, 0.05),
]

# Blocks are shaded where the right choice carries the higher reward
# probability, which is where probabilityLeft is the low value.
B_LOW_P_LEFT = 0.1


# %%
# ============================================================================
# Load
# ============================================================================
apply_style()

print("\n#### Fig 6  fitted parameters")
pk = load_pkl()
standata = load_recovery_data()
verify_grid(pk, standata)
params = seqac_params_long(pk, standata)
print(f"    {params['parameter'].nunique()} parameters, {len(params)} values")

print("\n#### Fig 6  manipulation simulations")
sims = run_manipulation_simulations(pk, run_extended=False)


# %%
# ============================================================================
# 6b - model variables over one session
# ============================================================================
print(f"\n#### Fig 6b  example session {EXAMPLE_SESSION}")
# The x-axis is a trial number, so the index has to be the trial number.
EXAMPLE_CACHE = PAPER_DIR / "data" / "figure6" / "example_session.csv.gz"
if EXAMPLE_CACHE.exists():
    example = pd.read_csv(EXAMPLE_CACHE)
    consumed = example['feedbackTypeConsumed'].to_numpy()
else:
    example = alf(str(EXAMPLE_SESSION)).to_df().reset_index(drop=True)
    consumed = np.load(EXAMPLE_SESSION / 'alf'
                       / '_ibl_trials.feedbackTypeConsumed.npy')

# Right choices are drawn above the axes, left choices below.
example['choice_side'] = np.where(example['choice'] == 1, 1,
                                  np.where(example['choice'] == -1, -1, 0))
# A CS+ is a rewarded trial; the stimulation is the subset the animal licked
# for. Both rasters are split by the side chosen on that trial.
example['cs_plus'] = np.where(example['outcome'] == 1,
                              example['choice_side'], 0)
example['vta_stim'] = np.where(consumed[:len(example)] > 0,
                               example['choice_side'], 0)

n_drawn = ((example.index >= B_XLIM[0]) & (example.index <= B_XLIM[1])).sum()
print(f"    {len(example)} trials, {n_drawn} in the drawn window")
print(f"    {int((example['cs_plus'] != 0).sum())} CS+, "
      f"{int((example['vta_stim'] != 0).sum())} with stimulation")

fig, host = plt.subplots(figsize=(5.4, 2.6))

# One axis per variable: the first on the host, the rest on twins whose right
# spines step outward so the three scales read separately.
twins = []
for index, (column, colour, label, ylim, yticks) in enumerate(B_TRACES):
    ax = host if index == 0 else host.twinx()
    if index > 1:
        ax.spines['right'].set_position(('outward', (index - 1) * B_SPINE_STEP))
    ax.plot(example.index, example[column], color=colour, lw=0.9, zorder=2)
    ax.set_ylim(*ylim)
    ax.set_yticks(yticks)
    # The axis label carries the colour; the tick labels stay black, as the
    # published panel has them.
    ax.set_ylabel(label, fontsize=FS_LABEL, color=colour)
    ax.tick_params(axis='y', labelsize=7)
    if index:
        twins.append(ax)

# The blocks, behind the traces and spanning the full height. Drawn in
# axis-fraction y so the shading does not depend on any trace's limits.
host.fill_between(example.index, 0, 1,
                  where=example['probabilityLeft'] == B_LOW_P_LEFT,
                  color='gray', alpha=0.3, lw=0, zorder=0,
                  transform=host.get_xaxis_transform())

# The rasters, in axis-fraction coordinates so they sit outside the data.
# clip_on=False puts them beyond the axes, so the trials have to be cut to
# the drawn window by hand.
in_window = (example.index >= B_XLIM[0]) & (example.index <= B_XLIM[1])
for column, colour, offset, length in B_RASTERS:
    for side, base in [(1, 1.0 + offset), (-1, -offset - length)]:
        trials = example.index[in_window & (example[column] == side)]
        host.vlines(trials, base, base + length, color=colour, lw=0.5,
                    transform=host.get_xaxis_transform(), clip_on=False)
    if column == 'choice_side':          # the row nearest the axes is labelled
        for side, base, label in [(1, 1.0 + offset, 'Right\nChoices'),
                                  (-1, -offset - length, 'Left\nChoices')]:
            # Both coordinates are axis fractions, so transAxes, not the
            # y-axis transform (which takes y in data units).
            host.text(-0.01, base + length / 2, label, ha='right', va='center',
                      fontsize=FS_LABEL, transform=host.transAxes)

# A scale bar instead of an x-axis.
bar_left = B_XLIM[0] + 0.02 * (B_XLIM[1] - B_XLIM[0])
host.plot([bar_left, bar_left + B_SCALE_BAR_TRIALS], [-0.36, -0.36],
          color='k', lw=1.4, transform=host.get_xaxis_transform(),
          clip_on=False)
host.text(bar_left + B_SCALE_BAR_TRIALS / 2, -0.39,
          f"{B_SCALE_BAR_TRIALS} trials", ha='center', va='top',
          fontsize=FS_LABEL, transform=host.get_xaxis_transform())

host.set_xlim(*B_XLIM)
host.set_xticks([])
host.legend(handles=[
    mpatches.Patch(color='gray', alpha=0.3, label='Block'),
    mlines.Line2D([], [], color='k', lw=1, label='Choice'),
    mlines.Line2D([], [], color=VIOLET, lw=1, label='CS+'),
    mlines.Line2D([], [], color=CYAN, lw=1, label='VTA stimulation'),
], loc='lower center', bbox_to_anchor=(0.6, -0.42), ncol=4,
    fontsize=FS_LEGEND, frameon=True, handlelength=1.2)

# The host keeps its left spine only; each twin keeps its own right spine.
sns.despine(ax=host, bottom=True)
for ax in twins:
    sns.despine(ax=ax, left=True, bottom=True, right=False)

# The rasters, scale bar and legend are drawn unclipped outside the axes, so
# the margins are reserved here and the panel is saved without a tight box -
# a tight box is computed from the axes and would cut them off.
fig.subplots_adjust(left=0.13, right=0.80, top=0.72, bottom=0.30)
save_panel(fig, 'fig6b_example_session.pdf', tight=False)


# %%
# ============================================================================
# 6c - fitted actor weights and bias
# ============================================================================
print("\n#### Fig 6c  actor weights and bias")
fig, ax = plt.subplots(figsize=(1.3, 2.0))
param_violins(params, ['Actor β & Bias'], [ax])
ax.set_xlabel('Parameter', fontsize=FS_LABEL)
save_panel(fig, 'fig6c_actor_weights.pdf')


# %%
# ============================================================================
# 6d - fitted learning and decay rates
# ============================================================================
print("\n#### Fig 6d  learning and decay rates")
fig, axes = plt.subplots(1, 2, figsize=(1.9, 2.0))
param_violins(params, ['Actor decay rates', 'Critic α'], axes)
fig.supxlabel('Parameter', fontsize=FS_LABEL, y=-0.06)
fig.subplots_adjust(wspace=0.7)
save_panel(fig, 'fig6d_learning_rates.pdf')


# %%
# ============================================================================
# 6e - return probability by trial type
# ============================================================================
print("\n#### Fig 6e  return probability, simulated")
returns = sims['sim_repeat'][RETURN_ORDER].dropna()
print(f"    {len(returns)} sessions")
print(returns.mean().to_string())

result = return_prob_wilcoxon(returns)
for (i, j), d in result.items():
    print(f"    {RETURN_ORDER[i]:>9} vs {RETURN_ORDER[j]:<9} W={d['W']:7.1f} "
          f"p={d['p_raw']:.3g}  p_fdr={d['p_fdr']:.3g}  {stars(d['p_fdr'])}")

fig, ax = plt.subplots(figsize=(1.4, 2.2))
x = np.arange(len(RETURN_ORDER))
ax.bar(x, returns.mean().to_numpy(), yerr=returns.sem().to_numpy(),
       color=[GREY, CYAN, CYAN], edgecolor='black', lw=0.6, width=0.7,
       capsize=3, zorder=2)
for _, session in returns.iterrows():
    ax.plot(x, session.to_numpy(), color='0.5', lw=0.25, alpha=0.5, zorder=1)

# Two brackets between the neighbouring pairs, then the outer pair above them.
lower = min(float(returns.to_numpy().max()) + 0.05, 1.02)
upper = lower + 0.13
bracket(ax, 0, 1, lower, stars(result[(0, 1)]['p_fdr']))
bracket(ax, 1, 2, lower, stars(result[(1, 2)]['p_fdr']))
bracket(ax, 0, 2, upper, stars(result[(0, 2)]['p_fdr']))

ax.set_xticks(x)
ax.set_xticklabels(RETURN_LABELS, fontsize=6, rotation=90)
ax.set_yticks(np.arange(0, 1.01, 0.2))
ax.set_ylim(0, upper + 0.14)
ax.set_ylabel('Return probability', fontsize=FS_LABEL)
sns.despine(ax=ax)
save_panel(fig, 'fig6e_return_probability.pdf')


# %%
# ============================================================================
# 6f - skipped CS+ against CS+ rate
# ============================================================================
print("\n#### Fig 6f  skipped CS+ against CS+ rate, simulated")
intact = sims['intact'].dropna(subset=['reward_rate', 'skip_rate', 'mouse'])
lmm = skip_perf_lmm(intact, y='skip_rate', x='reward_rate')
print(f"    slope={lmm['slope']:+.4f}  t={lmm['t']:+.3f}  p={lmm['p']:.3g}  "
      f"n={lmm['n']} sessions over {intact['mouse'].nunique()} mice  "
      f"converged={lmm['converged']}")

annotation = (f"{stars(lmm['p'])}p={lmm['p']:.3f}" if lmm['p'] < 0.05
              else f"p={lmm['p']:.3f}")
fig, ax = plt.subplots(figsize=(1.9, 2.0))
regression_panel(ax, intact['reward_rate'], intact['skip_rate'],
                 point_color='k', line_color=CYAN,
                 xlabel='Fraction of trials\nw/ CS+',
                 ylabel='Fraction of CS+\ntrials w/o VTA stim',
                 annotation=annotation)
save_panel(fig, 'fig6f_skipped_vs_cs_rate.pdf')


# %%
# ============================================================================
# 6g - performance with the CS+ removed
# ============================================================================
print("\n#### Fig 6g  no-CS+ manipulation")
with_cs, without_cs = paired_performance(sims['sim_mean'], 'no_cs')
print(f"    {len(with_cs)} sessions; "
      f"w/ CS+ {with_cs.mean():.4f}, w/o CS+ {without_cs.mean():.4f}")

fig, ax = plt.subplots(figsize=(1.3, 2.0))
w, p = paired_bar_panel(ax, with_cs, without_cs, ['w/ CS+', 'w/o CS+'],
                        'P(High Rew. Prob. Choice)', chance=0.5)
print(f"    Wilcoxon W={w:.1f}  p={p:.3g}  {stars(p)}")
ax.set_title('No CS+\nmanipulation', fontsize=FS_LABEL)
save_panel(fig, 'fig6g_no_cs_manipulation.pdf')

print("\nDone.")