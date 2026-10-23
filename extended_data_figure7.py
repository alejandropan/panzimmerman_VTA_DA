"""
Extended Data Fig. 7i,j and 7p,q - relative and state value decoding.

Four panels in two styles:

    i  relative value, R2 against NEURON COUNT, one time bin
    j  state value, same
    p  state value as the unweighted sum of Qs, time-resolved
    q  state value as the weighted sum, time-resolved

Usage (interactive), or:
    python extended_data_figure7.py
"""

# %%
import matplotlib.pyplot as plt

from decoding import (DECODER_FOLDERS, STRIATAL, SUMMARY_FILE, apply_style,
                      load_neuron_count_panel, plot_r2_vs_neurons,
                      plot_timecourse, prepare_panel, save_panel)

# %%
# ============================================================================
# Panels
# ============================================================================
XLABEL = "Time From CS+ (s)"
T0, T1 = -0.5, 2.0
XLIM = (T0 + 0.1, T1)

YLIM = (-0.02, 0.21)
XTICKS = [-0.4, 0.0, 1.0, 2.0]
YTICKS = [0.00, 0.05, 0.10, 0.15, 0.20]

TIMECOURSE_PANELS = [
    dict(panel='p', decoder='state_unweighted',
         title="State Value (unweighted sum)"),
    dict(panel='q', decoder='state_weighted',
         title="State Value (weighted sum)"),
]

N_NEURONS = 20

# --- the neuron-count panels, i and j --------------------------------------
# CONFIRM: the bin each of these summarises. Bin 7 is what the Fig. 2 decoders
# use, but relative value is cue-aligned over -0.5..0.5 while state value is
# CS+-aligned over -0.5..2, so the same index is a different time in each.
NEURON_COUNT_PANELS = [
    dict(panel='i', decoder='relative_value', title="Relative Value",
         time_bin=7),
    dict(panel='j', decoder='state_value', title="State Value",
         time_bin=5),
]

MIN_NEURONS = 40
NEURON_XLIM = (10, 40)
NEURON_XTICKS = [10, 20, 30, 40]


# %%
# ============================================================================
# Run
# ============================================================================
apply_style()

for spec in TIMECOURSE_PANELS:
    real, stats, _ = prepare_panel(
        label=f"ED Fig 7{spec['panel']}  {spec['title']}",
        decoder=spec['decoder'], n_neurons=N_NEURONS, regions=STRIATAL,
        t0=T0, fisher=True, ranksum=False)

    fig, ax = plt.subplots(figsize=(3.0, 2.6))
    plot_timecourse(real, STRIATAL, ax, xlim=XLIM, ylim=(-0.02, 0.16), xlabel=XLABEL,
                    title=spec['title'], xticks=XTICKS, yticks=[0.00, 0.05, 0.10, 0.15])
    save_panel(fig, f"figED7{spec['panel']}_{spec['decoder']}.pdf")

# %%
# ============================================================================
# Run - i,j, R2 against neuron count
# ============================================================================
for spec in NEURON_COUNT_PANELS:
    print(f"\n#### ED Fig 7{spec['panel']}  {spec['title']}")
    real = load_neuron_count_panel(
        DECODER_FOLDERS[spec['decoder']] / SUMMARY_FILE,
        regions=STRIATAL, time_bin=spec['time_bin'],
        min_neurons=MIN_NEURONS)

    fig, ax = plt.subplots(figsize=(3.0, 2.6))
    plot_r2_vs_neurons(real, STRIATAL, ax, xlim=NEURON_XLIM, ylim=YLIM,
                       title=spec['title'], xticks=NEURON_XTICKS,
                       yticks=YTICKS)
    save_panel(fig, f"figED7{spec['panel']}_{spec['decoder']}_vs_neurons.pdf")

print("\nDone.")
