"""
Extended Data Fig. 4f-i - action-value decoding against neuron count.

The second decoder panel style. The same four decoders as Fig. 2k-n, but read
at ONE time bin and plotted against the number of neurons the decoder was
trained on, rather than against time.

Usage (interactive), or:
    python extended_data_figure4.py
"""

# %%
import matplotlib.pyplot as plt

from decoding import (STRIATAL, SUMMARY_FILE, DECODER_FOLDERS, apply_style,
                      load_neuron_count_panel, plot_r2_vs_neurons, save_panel)

# %%
# ============================================================================
# Panels
# ============================================================================
# The single bin these panels summarise. Bin 7 of a window opening at -0.5 s
TIME_BIN = 7

# A session is drawn only if it has a decoder at this neuron count.
MIN_NEURONS = 40

XLIM = (10, 40)
XTICKS = [10, 20, 30, 40]

# PROVISIONAL - confirm against the typeset figure.
YLIM = (-0.02, 0.21)
YTICKS = [0.00, 0.05, 0.10, 0.15, 0.20]

PANELS = [
    dict(panel='f', decoder='qcontra_contra',
         title=r"$Q^{contra}$ (Contra Trials)"),
    dict(panel='g', decoder='qcontra_ipsi',
         title=r"$Q^{contra}$ (Ipsi Trials)"),
    dict(panel='h', decoder='qipsi_contra',
         title=r"$Q^{ipsi}$ (Contra Trials)"),
    dict(panel='i', decoder='qipsi_ipsi',
         title=r"$Q^{ipsi}$ (Ipsi Trials)"),
]


# %%
# ============================================================================
# Run
# ============================================================================
apply_style()

for spec in PANELS:
    print(f"\n#### ED Fig 4{spec['panel']}  {spec['title']}")
    real = load_neuron_count_panel(
        DECODER_FOLDERS[spec['decoder']] / SUMMARY_FILE,
        regions=STRIATAL, time_bin=TIME_BIN, min_neurons=MIN_NEURONS)

    fig, ax = plt.subplots(figsize=(3.0, 2.6))
    plot_r2_vs_neurons(real, STRIATAL, ax, xlim=XLIM, ylim=YLIM,
                       title=spec['title'], xticks=XTICKS, yticks=YTICKS)
    save_panel(fig, f"figED4{spec['panel']}_{spec['decoder']}_vs_neurons.pdf")

print("\nDone.")
