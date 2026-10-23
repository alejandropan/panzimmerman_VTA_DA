"""
Figure 5d,e,f,h - the CS+ as a conditioned reinforcer.

    d  fraction of CS+ trials the animal left uncollected, per session
    e  probability of repeating the previous choice, by outcome condition
    f  that fraction against overall CS+ rate, across sessions
    h  the no-CS+ control experiment: behaviour and fitted parameters

Panels a, b (encoding-model kernels), c, g (task schematics) and i come from
elsewhere. Everything substantive is in `figure5_data.py`.

Usage (interactive), or:
    python figure5.py
"""

# %%
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import statsmodels.formula.api as smf
from scipy.stats import ttest_rel, wilcoxon
from statsmodels.stats.multitest import multipletests

from figure5_data import (BLACK, CONDITION_LABELS, CYAN, GREY, RETURN_ORDER,
                          apply_style, build_transition_data, load_behaviour,
                          load_nocs_experiment, mouse_of, return_probabilities,
                          save_panel, sessions_with_enough_unconsumed)

# %%
# ============================================================================
# Configuration
# ============================================================================
# Limits and ticks measured from the typeset figure.
D_YLIM, D_YTICKS = (0.0, 0.46), [0.0, 0.1, 0.2, 0.3, 0.4]
E_YLIM, E_YTICKS = (0.0, 1.05), [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
F_XLIM, F_XTICKS = (0.36, 0.57), [0.4, 0.5]
F_YLIM, F_YTICKS = (-0.03, 0.45), [0.0, 0.1, 0.2, 0.3, 0.4]

CORRECTION = 'holm'
ALPHA = 0.05

# Fig 5h. Eight panels over two rows, in published order: the two behavioural
# measures first, then the six fitted parameters.
NOCS_PANELS = [
    dict(source='behaviour', column='outcome', label='Reward Rate',
         test='wilcoxon', ylim=(0.0, 0.65), yticks=[0.0, 0.2, 0.4, 0.6],
         reference=0.4),
    dict(source='behaviour', column='p(high)',
         label='P(High Rew.\nProb. Choice)', test='wilcoxon',
         ylim=(0.0, 0.72), yticks=[0.0, 0.2, 0.4, 0.6], reference=0.5),
    dict(source='parameters', column='beta_mouse', label=r'$\beta_{rew}$',
         test='ttest', ylim=(0.0, 2.2), yticks=[0, 1, 2]),
    dict(source='parameters', column='stay_mouse', label=r'$\beta_{stay}$',
         test='ttest', ylim=(0.4, 1.7), yticks=[0.4, 0.8, 1.2, 1.6]),
    dict(source='parameters', column='abs_bias', label='|Bias|',
         test='wilcoxon', ylim=(0.0, 2.5), yticks=[0, 1, 2]),
    dict(source='parameters', column='alpha', label=r'$\alpha_{rew}$',
         test='wilcoxon', ylim=(0.0, 0.5), yticks=[0.0, 0.2, 0.4]),
    dict(source='parameters', column='alphastay', label=r'$\alpha_{stay}$',
         test='wilcoxon', ylim=(0.0, 0.65), yticks=[0.0, 0.2, 0.4, 0.6]),
    dict(source='parameters', column='forget', label=r'$\gamma_{forget}$',
         test='wilcoxon', ylim=(0.0, 0.45), yticks=[0.0, 0.2, 0.4]),
]

# 'control' is the session with the CS+; 'no_cs+' the next day without it.
CONDITION_ORDER = ['control', 'no_cs+']
CONDITION_TICKS = ['w/ CS+', 'w/o CS+']


def bracket(ax, left, right, height, label):
    """
    A significance bracket spanning two bars.

    Args:
        ax: Axes to draw into.
        left, right: Bar positions to span.
        height: Vertical position, 0-1 of the axes.
        label: Text above the bracket, e.g. '***' or 'n.s.'.
    """
    ax.plot([left, left, right, right],
            [height, height + 0.02, height + 0.02, height],
            transform=ax.get_xaxis_transform(), color=BLACK, linewidth=1,
            clip_on=False)
    ax.text((left + right) / 2, height + 0.03, label,
            transform=ax.get_xaxis_transform(), ha='center', va='bottom',
            fontsize=9 if label == 'n.s.' else 12)


def stars(p_value):
    """The conventional marker for one p-value."""
    if p_value < 0.0001:
        return '****'
    if p_value < 0.001:
        return '***'
    if p_value < 0.01:
        return '**'
    if p_value < ALPHA:
        return '*'
    return 'n.s.'


# %%
# ============================================================================
# The data
# ============================================================================
apply_style()

behaviour = load_behaviour()
transition_data = build_transition_data(behaviour)
enough = sessions_with_enough_unconsumed(transition_data)
print(f"{behaviour.id.nunique()} sessions, {len(enough)} with at least 5 "
      "unconsumed CS+")


# %%
# ============================================================================
# 5d - how often the CS+ went uncollected
# ============================================================================
print("\n#### Fig 5d  fraction of CS+ trials without VTA stim")
rewarded = behaviour.loc[behaviour.reward == 1]
uncollected = 1 - rewarded.groupby('id')['laser_consumed'].mean()
print(f"    {len(uncollected)} sessions, mean {uncollected.mean():.3f}, "
      f"max {uncollected.max():.3f}")

fig, ax = plt.subplots(figsize=(1.6, 2.6))
sns.barplot(y=uncollected.values, errorbar='se', color=CYAN, ax=ax)
sns.swarmplot(y=uncollected.values, color=BLACK, size=3, ax=ax)
ax.set_ylim(*D_YLIM)
ax.set_yticks(D_YTICKS)
ax.set_ylabel('Fraction of CS+ trials\nw/o VTA stim')
ax.set_xlabel(' ')
sns.despine(ax=ax)
save_panel(fig, 'fig5d_uncollected_cs.pdf')


# %%
# ============================================================================
# 5e - return probability by outcome condition
# ============================================================================
print("\n#### Fig 5e  return probability")
returns = return_probabilities(transition_data, sessions=enough)
print(f"    {returns.id.nunique()} sessions")

per_session = returns.pivot(index='id', columns='after', values='repeat')
comparisons = [('after_error', 'after_laser'),
               ('after_error', 'after_cs_no_laser'),
               ('after_cs_no_laser', 'after_laser')]
raw = [wilcoxon(per_session[a], per_session[b])[1] for a, b in comparisons]
reject, adjusted, _, _ = multipletests(raw, method=CORRECTION)
for (a, b), p, q in zip(comparisons, raw, adjusted):
    print(f"    {a} vs {b}: p={p:.3e}, {CORRECTION}-adjusted q={q:.3e} "
          f"{stars(q)}")

fig, ax = plt.subplots(figsize=(2.4, 2.6))
sns.barplot(data=returns, x='after', y='repeat', order=RETURN_ORDER,
            errorbar='se', palette=[GREY, CYAN, CYAN], hue='after',
            hue_order=RETURN_ORDER, legend=False, ax=ax)
sns.lineplot(data=returns, x='after', y='repeat', hue='id', color=BLACK,
             palette=[BLACK] * returns.id.nunique(), legend=False,
             linewidth=0.5, alpha=0.6, ax=ax)
ax.set_ylim(*E_YLIM)
ax.set_yticks(E_YTICKS)
ax.set_ylabel('Return probability')
ax.set_xlabel('')
ax.set_xticks(range(len(RETURN_ORDER)))
ax.set_xticklabels([CONDITION_LABELS[c] for c in RETURN_ORDER])
# Positions follow `comparisons`: CS- vs w/ stim, CS- vs w/o stim, and the
# two CS+ conditions against each other.
for (left, right), height, q in zip([(0, 1), (0, 2), (1, 2)],
                                    [0.88, 0.97, 0.88], adjusted):
    bracket(ax, left, right, height, stars(q))
sns.despine(ax=ax)
save_panel(fig, 'fig5e_return_probability.pdf')

# The mixed model behind the same comparison.
returns_lmm = returns.copy()
returns_lmm['mouse'] = returns_lmm['id'].apply(mouse_of)
model = smf.mixedlm("repeat ~ C(after)", returns_lmm, groups="mouse",
                    re_formula="~1", vc_formula={"id_re": "0 + C(id)"})
results = model.fit(maxiter=1000)
reference = RETURN_ORDER[0]
print(f"\n    Linear mixed model, reference category {reference}:")
if any(f'C(after)[T.{reference}]' == name for name in results.params.index):
    raise ValueError(f"{reference} is not the reference level - the category "
                     "order was lost, so patsy fell back to alphabetical")
contrasts = {
    f'after_laser vs {reference}': "C(after)[T.after_laser]",
    f'after_cs_no_laser vs {reference}': "C(after)[T.after_cs_no_laser]",
    'after_laser vs after_cs_no_laser':
        "C(after)[T.after_laser] - C(after)[T.after_cs_no_laser]",
}
for description, constraint in contrasts.items():
    print(f"    --- {description} ---")
    print(results.f_test(constraint))


# %%
# ============================================================================
# 5f - uncollected CS+ against overall CS+ rate
# ============================================================================
print("\n#### Fig 5f  uncollected CS+ vs CS+ rate")

per_session_rate = pd.DataFrame({
    'cs_rate': transition_data.groupby('id')['reward'].mean(),
    'uncollected': (transition_data.loc[transition_data.reward == 1]
                    .groupby('id')['unconsumed_cs+'].mean()),
}).reset_index()
if per_session_rate.isna().any().any():
    raise ValueError("a session has no rewarded trials, so its uncollected "
                     "fraction is undefined")
per_session_rate['mouse'] = per_session_rate['id'].apply(mouse_of)
print(f"    {len(per_session_rate)} sessions, "
      f"{per_session_rate.mouse.nunique()} mice")

model_f = smf.mixedlm("uncollected ~ cs_rate", per_session_rate,
                      groups=per_session_rate["mouse"])
results_f = model_f.fit(method='cg')
slope_test = results_f.t_test(np.array([[0, 1]]), use_t=True)
print(slope_test)
print(f"    slope p = {float(np.squeeze(slope_test.pvalue)):.3e}")

fig, ax = plt.subplots(figsize=(2.4, 2.6))
sns.regplot(data=per_session_rate, x='cs_rate', y='uncollected', color=CYAN,
            scatter_kws=dict(color=BLACK, s=6), ax=ax)
slope_p = float(np.squeeze(slope_test.pvalue))
ax.text(0.03, 0.97, 'p<0.001' if slope_p < 0.001 else f'p={slope_p:.3f}',
        transform=ax.transAxes, ha='left', va='top')
ax.set_xlim(*F_XLIM)
ax.set_xticks(F_XTICKS)
ax.set_ylim(*F_YLIM)
ax.set_yticks(F_YTICKS)
ax.set_xlabel('Fraction of trials\nw/ CS+')
ax.set_ylabel('Fraction of CS+ trials\nw/o VTA stim')
sns.despine(ax=ax)
save_panel(fig, 'fig5f_uncollected_vs_rate.pdf')


# %%
# ============================================================================
# 5h - the no-CS+ control experiment
# ============================================================================
print("\n#### Fig 5h  no-CS+ control")
nocs_behaviour, nocs_parameters = load_nocs_experiment()
# One value per animal per condition, which is what both the bars and the
# tests are over (there is only one session pair per animal).
per_animal = (nocs_behaviour.groupby(['mouse', 'cond'])[['outcome', 'p(high)']]
              .mean().reset_index())
print(f"    {per_animal.mouse.nunique()} animals")

sources = {'behaviour': per_animal, 'parameters': nocs_parameters}

fig, axes = plt.subplots(2, 4, figsize=(9.0, 5.2))
for spec, ax in zip(NOCS_PANELS, axes.ravel()):
    table = sources[spec['source']]
    paired = table.pivot(index='mouse', columns='cond',
                         values=spec['column'])[CONDITION_ORDER]
    test = wilcoxon if spec['test'] == 'wilcoxon' else ttest_rel
    result = test(paired[CONDITION_ORDER[0]], paired[CONDITION_ORDER[1]])
    print(f"    {spec['column']:12s} {spec['test']:8s} "
          f"stat={result.statistic:7.3f}  p={result.pvalue:.4f} "
          f"{stars(result.pvalue)}")

    sns.barplot(data=table, x='cond', y=spec['column'], order=CONDITION_ORDER,
                hue='cond', hue_order=CONDITION_ORDER, palette=[CYAN, GREY],
                errorbar='se', legend=False, ax=ax)
    sns.lineplot(data=table, x='cond', y=spec['column'], hue='mouse',
                 palette=[BLACK] * table.mouse.nunique(), errorbar=None,
                 legend=False, linewidth=0.5, ax=ax)
    ax.set_ylim(*spec['ylim'])
    ax.set_yticks(spec['yticks'])
    ax.set_ylabel(spec['label'])
    ax.set_xlabel('')
    ax.set_xticks(range(len(CONDITION_ORDER)))
    ax.set_xticklabels(CONDITION_TICKS, rotation=45, ha='right')
    if spec.get('reference') is not None:
        ax.axhline(spec['reference'], color=BLACK, linestyle=':', linewidth=1)
    if result.pvalue < ALPHA:
        bracket(ax, 0, 1, 0.92, '*')
    sns.despine(ax=ax)

fig.tight_layout()
save_panel(fig, 'fig5h_nocs_control.pdf')

print("\nDone.")
