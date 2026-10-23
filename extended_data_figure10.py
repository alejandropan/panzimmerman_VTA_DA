"""
Extended Data Fig. 10a,b,c - the fitted value of a CS+ without VTA stim.

    a  decision times after each outcome, per session
    b  the pooled posterior over the CS+-without-stim value parameter
    c  the same posterior, one session at a time

Everything substantive is in `figure5_data.py`.

Usage (interactive), or:
    python extended_data_figure10.py
"""

# %%
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.stats import gaussian_kde

from figure5_data import (BLACK, CONDITION_LABELS, CYAN, RETURN_ORDER, GREY,
                          MIN_UNCONSUMED_CS, apply_style,
                          build_transition_data, cs_value_posteriors,
                          load_behaviour, save_panel)

# %%
# ============================================================================
# Configuration
# ============================================================================
# Limits and ticks measured from the typeset figure.
A_YLIM = (-0.5, 1.15)
A_YTICKS = [-0.2, 0.0, 0.2, 0.4, 0.6, 0.8, 1.0]

# Both posterior panels share this axis, so b and c can be read together.
VALUE_XLIM = (-2, 4)
VALUE_XTICKS = [-2, -1, 0, 1, 2, 3, 4]
B_YTICKS = [0.0, 0.2, 0.4, 0.6, 0.8]

# The CS+ WITH VTA stim is fixed at 1 by construction, so 1 is the reference
# the fitted value is read against, and 0 is "worth nothing".
REFERENCE_VALUE = 1.0

# Panel c. Each session's density is drawn on its own baseline, spaced by
# this fraction of the tallest density.
RIDGE_SPACING = 0.7
RIDGE_GRID = np.linspace(*VALUE_XLIM, 512)


# %%
# ============================================================================
# 10a - decision times by outcome condition
# ============================================================================
apply_style()

print("\n#### ED Fig 10a  decision times by outcome")
behaviour = load_behaviour()
transition_data = build_transition_data(behaviour)

# A session is drawn only where every condition rests on enough trials, which
# for the two CS+ conditions means the uncollected one.
after_uncollected = transition_data.loc[
    (transition_data['after_cs_no_laser'] == 1)
    & (transition_data['after_cs'] == 1)]
enough = (after_uncollected.groupby('id')
          .filter(lambda part: len(part) >= MIN_UNCONSUMED_CS)['id'].unique())
drawn = transition_data.loc[transition_data['id'].isin(enough)]
print(f"    {len(enough)} of {transition_data.id.nunique()} sessions")

# One mean per session per condition. T
per_condition = (drawn.groupby(['id', 'after_cs', 'after_cs_no_laser'])
                 ['z_decision_times'].mean().reset_index())
per_condition['after'] = np.where(
    per_condition['after_cs'] == 0, 'after_error',
    np.where(per_condition['after_cs_no_laser'] == 1,
             'after_cs_no_laser', 'after_laser'))
per_condition['after'] = pd.Categorical(per_condition['after'],
                                        categories=RETURN_ORDER, ordered=True)
per_condition = per_condition.sort_values(['after', 'id'])

fig, ax = plt.subplots(figsize=(2.4, 2.8))
sns.barplot(data=per_condition, x='after', y='z_decision_times',
            order=RETURN_ORDER, errorbar='se', color=GREY, ax=ax)
sns.lineplot(data=per_condition, x='after', y='z_decision_times', hue='id',
             palette=[BLACK] * per_condition.id.nunique(), errorbar=None,
             legend=False, linewidth=0.6, ax=ax)
ax.set_ylim(*A_YLIM)
ax.set_yticks(A_YTICKS)
ax.set_ylabel('z-scored\ndecision times')
ax.set_xlabel('')
ax.set_xticks(range(len(RETURN_ORDER)))
ax.set_xticklabels([CONDITION_LABELS[c].replace('\n', ' ')
                    for c in RETURN_ORDER], rotation=45, ha='right')
sns.despine(ax=ax)
save_panel(fig, 'figED10a_decision_times.pdf')


# %%
# ============================================================================
# 10b - the pooled posterior
# ============================================================================
# Same data but the order of sessions (rows) differs from figure

print("\n#### ED Fig 10b  pooled posterior")
posteriors, selected = cs_value_posteriors()
pooled = posteriors.to_numpy().flatten()
print(f"    {posteriors.shape[1]} sessions, {len(pooled)} draws")
print(f"    P(value > 0) = {np.mean(pooled > 0):.4f}")
print(f"    5th percentile = {np.percentile(pooled, 5):.3f}, "
      f"median = {np.median(pooled):.3f}")

fig, ax = plt.subplots(figsize=(3.2, 2.6))
sns.kdeplot(x=pooled, fill=True, color=CYAN, linewidth=1, ax=ax)
ax.axvline(0, color=BLACK, linewidth=1.5)
ax.axvline(REFERENCE_VALUE, color=BLACK, linestyle='--', linewidth=1.5)
ax.set_xlim(*VALUE_XLIM)
ax.set_xticks(VALUE_XTICKS)
ax.set_yticks(B_YTICKS)
ax.set_xlabel('Parameter Value')
ax.set_ylabel('Density')
sns.despine(ax=ax)
save_panel(fig, 'figED10b_pooled_posterior.pdf')


# %%
# ============================================================================
# 10c - one session at a time
# ============================================================================
print("\n#### ED Fig 10c  per-session posteriors")

densities = []
for column in posteriors.columns:
    draws = posteriors[column].to_numpy()
    densities.append(gaussian_kde(draws)(RIDGE_GRID))
densities = np.array(densities)
step = RIDGE_SPACING * densities.max()

fig, ax = plt.subplots(figsize=(3.2, 0.22 * len(densities) + 1.0))
for row, density in enumerate(densities):
    baseline = row * step
    # Later sessions are drawn on top, so each fill hides the one behind it.
    ax.fill_between(RIDGE_GRID, baseline, baseline + density, color=CYAN,
                    alpha=0.85, linewidth=0, zorder=row * 2)
    ax.plot(RIDGE_GRID, baseline + density, color='white', linewidth=0.8,
            zorder=row * 2 + 1)

ax.axvline(0, color=BLACK, linewidth=1.5, zorder=len(densities) * 2 + 1)
ax.axvline(REFERENCE_VALUE, color=BLACK, linestyle='--', linewidth=1,
           alpha=0.6, zorder=len(densities) * 2 + 1)
ax.set_xlim(*VALUE_XLIM)
ax.set_xticks(VALUE_XTICKS)
ax.set_ylim(-0.05 * step, (len(densities) - 1) * step + densities.max() * 1.1)
ax.set_yticks([])
ax.set_xlabel('Parameter Value')
ax.set_ylabel('Individual sessions (density)')
sns.despine(ax=ax, left=True)
save_panel(fig, 'figED10c_per_session_posteriors.pdf')

# Which sessions clear zero, the claim the panel supports.
above_zero = [column for column in posteriors.columns
              if np.percentile(posteriors[column], 5) > 0]
print(f"    {len(above_zero)} of {posteriors.shape[1]} sessions have their "
      "5th percentile above zero")

print("\nDone.")
