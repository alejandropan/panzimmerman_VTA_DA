"""
Extended Data Fig. 12d - decoding the Conditioned Reinforcer model's choice policy.

One panel: pi_choice decoded from groups of 20 randomly chosen neurons in
contralateral-choice trials, aligned to the go cue.

Usage (interactive), or:
    python extended_data_figure12.py
"""

# %%
import matplotlib.pyplot as plt

from decoding import (STRIATAL, apply_style, mark_ranksum, plot_timecourse,
                      prepare_panel, save_panel)

# %%
# ============================================================================
# The panel
# ============================================================================
XLABEL = "Time From Go Cue (s)"
T0, T1 = -0.5, 0.5
XLIM = (T0 + 0.1, T1)
TITLE = r"$\pi^{choice}$"

# PROVISIONAL - confirm against the typeset figure.
YLIM = (-0.02, 0.08)
XTICKS = [-0.4, 0.0, 0.4]
YTICKS = [0.00, 0.02, 0.04, 0.06]

N_NEURONS = 20


# %%
# ============================================================================
# Run
# ============================================================================
apply_style()

real, stats, ranksum = prepare_panel(
    label=f"ED Fig 12d  {TITLE}", decoder='pi_choice', n_neurons=N_NEURONS,
    regions=STRIATAL, t0=T0, fisher=True, ranksum=True)

fig, ax = plt.subplots(figsize=(3.0, 2.6))
plot_timecourse(real, STRIATAL, ax, xlim=XLIM, ylim=YLIM, xlabel=XLABEL,
                title=TITLE, xticks=XTICKS, yticks=YTICKS)
mark_ranksum(ax, ranksum)
save_panel(fig, "figED12d_pi_choice.pdf")

print("\nDone.")
