"""
Figure 6k,l,p,q - decoding the Conditioned Reinforcer model's value signals.

Four panels, all from groups of 40 randomly chosen neurons on rewarded trials,
aligned to reward:

    k  V_cs,     all rewarded trials
    l  V_cs,     sessions with >= 5 CS+ trials without VTA stimulation
    p  V_choice, all rewarded trials
    q  V_choice, that same subset

Usage (interactive), or:
    python figure6.py
"""

# %%
import matplotlib.pyplot as plt

from decoding import (STRIATAL, apply_style, mark_ranksum, plot_timecourse,
                      prepare_panel, save_panel)

# %%
# ============================================================================
# Panels
# ============================================================================
XLABEL = "Time From Reward (s)"
T0, T1 = 0.0, 1.0
XLIM = (T0 + 0.1, T1)

# Only the first second is shown; bins 0-9 at 100 ms each.
MAX_TIME_BIN = 9

# PROVISIONAL - confirm against the typeset figure.
YLIM = (-0.10, 0.21)
XTICKS = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
YTICKS = [-0.10, 0.00, 0.05, 0.10, 0.15, 0.20]

PANELS = [
    dict(panel='k', decoder='vcs', title=r"$V^{CS}$"),
    dict(panel='l', decoder='vcs_skips', title=r"$V^{CS}$, $\geq$5 skips"),
    dict(panel='p', decoder='vchoice', title=r"$V^{choice}$"),
    dict(panel='q', decoder='vchoice_skips',
         title=r"$V^{choice}$, $\geq$5 skips"),
]

N_NEURONS = 40


# %%
# ============================================================================
# Run
# ============================================================================
apply_style()

for spec in PANELS:
    real, stats, ranksum = prepare_panel(
        label=f"Fig 6{spec['panel']}  {spec['title']}",
        decoder=spec['decoder'], n_neurons=N_NEURONS, regions=STRIATAL,
        t0=T0, max_time_bin=MAX_TIME_BIN, fisher=True, ranksum=True)

    fig, ax = plt.subplots(figsize=(3.0, 2.6))
    plot_timecourse(real, STRIATAL, ax, xlim=XLIM, ylim=YLIM, xlabel=XLABEL,
                    title=spec['title'], xticks=XTICKS, yticks=YTICKS)
    mark_ranksum(ax, ranksum)
    save_panel(fig, f"fig6{spec['panel']}_{spec['decoder']}.pdf")

print("\nDone.")
