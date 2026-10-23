"""
Figure 2k-n - time-resolved action-value decoding in the striatum.

Four panels: the contralateral and ipsilateral action value, each decoded in
trials where the animal chose contralaterally or ipsilaterally. Groups of 20
randomly chosen neurons, aligned to the go cue.

Everything substantive is in `decoding.py`. This file declares the panels and
lays them out.

Usage (interactive), or:
    python figure2.py
"""

# %%
import matplotlib.pyplot as plt

from decoding import (STRIATAL, apply_style, mark_ranksum, plot_timecourse,
                      prepare_panel, save_panel)

# %%
# ============================================================================
# Panels
# ============================================================================
XLABEL = "Time From Go Cue (s)"

# The first bin is labelled at its right edge, so the data start one bin after
# the window opens: a window from -0.5 has its first point at -0.4.
T0, T1 = -0.5, 0.5
XLIM = (T0 + 0.1, T1)

# Shared across the four panels so they can be read against each other.
# PROVISIONAL - confirm against the typeset figure.
YLIM = (-0.02, 0.13)
XTICKS = [-0.4, -0.2, 0.0, 0.2, 0.4]
YTICKS = [0.00, 0.04, 0.08, 0.12]

PANELS = [
    dict(panel='k', decoder='qcontra_contra',
         title=r"$Q^{contra}$ (Contra Trials)"),
    dict(panel='l', decoder='qipsi_contra',
         title=r"$Q^{ipsi}$ (Contra Trials)"),
    dict(panel='m', decoder='qcontra_ipsi',
         title=r"$Q^{contra}$ (Ipsi Trials)"),
    dict(panel='n', decoder='qipsi_ipsi',
         title=r"$Q^{ipsi}$ (Ipsi Trials)"),
]

N_NEURONS = 20


# %%
# ============================================================================
# Run
# ============================================================================
apply_style()

for spec in PANELS:
    real, stats, ranksum = prepare_panel(
        label=f"Fig 2{spec['panel']}  {spec['title']}",
        decoder=spec['decoder'], n_neurons=N_NEURONS, regions=STRIATAL,
        t0=T0, fisher=True, ranksum=True)

    fig, ax = plt.subplots(figsize=(3.0, 2.6))
    plot_timecourse(real, STRIATAL, ax, xlim=XLIM, ylim=YLIM, xlabel=XLABEL,
                    title=spec['title'], xticks=XTICKS, yticks=YTICKS)
    mark_ranksum(ax, ranksum)
    save_panel(fig, f"fig2{spec['panel']}_{spec['decoder']}.pdf")

print("\nDone.")
