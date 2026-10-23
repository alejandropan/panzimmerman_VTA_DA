"""
Extended Data Fig. 5a-d - action-value decoding from striatal inputs and outputs.

The same four decoders as Fig. 2k-n, read out from cortical and pallidal
regions instead of the striatum. Each panel letter is TWO axes side by side -
cortex (MO, OFC, PFC, SS) and pallidum (VP, GPe) - sharing a y-scale.

Usage (interactive), or:
    python extended_data_figure5.py
"""

# %%
import matplotlib.pyplot as plt

from decoding import (CORTICAL, GREY_RAMP, PALLIDAL, apply_style,
                      plot_timecourse, prepare_panel, save_panel)

# %%
# ============================================================================
# Panels
# ============================================================================
XLABEL = "Time From Go Cue (s)"
T0, T1 = -0.5, 0.5
XLIM = (T0 + 0.1, T1)

# Measured from the published panels: every one of the eight axes shares these.
YLIM = (-0.02, 0.21)
XTICKS = [-0.4, 0.0, 0.5]
YTICKS = [0.00, 0.05, 0.10, 0.15, 0.20]

# (axes title, regions) for the two axes each panel letter carries.
REGION_GROUPS = [("Cortex", CORTICAL), ("Pallidum", PALLIDAL)]
ALL_REGIONS = CORTICAL + PALLIDAL

PANELS = [
    dict(panel='a', decoder='qcontra_contra',
         title=r"$Q^{contra}$ (Contra Trials)"),
    dict(panel='b', decoder='qipsi_contra',
         title=r"$Q^{ipsi}$ (Contra Trials)"),
    dict(panel='c', decoder='qipsi_ipsi',
         title=r"$Q^{ipsi}$ (Ipsi Trials)"),
    dict(panel='d', decoder='qcontra_ipsi',
         title=r"$Q^{contra}$ (Ipsi Trials)"),
]

N_NEURONS = 20


# %%
# ============================================================================
# Run
# ============================================================================
apply_style()

for spec in PANELS:
    # Loaded once over all six regions, so the exclusion and Fisher printouts
    # cover the whole panel; the split is only for drawing.
    real, stats, _ = prepare_panel(
        label=f"ED Fig 5{spec['panel']}  {spec['title']}",
        decoder=spec['decoder'], n_neurons=N_NEURONS, regions=ALL_REGIONS,
        t0=T0, fisher=True, ranksum=False)

    fig, axes = plt.subplots(1, 2, figsize=(5.2, 2.6), sharey=True)
    for ax, (group_title, regions) in zip(axes, REGION_GROUPS):
        plot_timecourse(real.loc[real['region'].isin(regions)], regions, ax,
                        xlim=XLIM, ylim=YLIM, xlabel=XLABEL, title=group_title,
                        xticks=XTICKS, yticks=YTICKS, palette=GREY_RAMP)
    # The panel's own title sits above both axes; the axes carry Cortex and
    # Pallidum.
    fig.suptitle(spec['title'])
    fig.tight_layout()
    save_panel(fig, f"figED5{spec['panel']}_{spec['decoder']}_cortex_pallidum.pdf")

print("\nDone.")
