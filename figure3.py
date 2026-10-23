"""
Figure 3b,c and 3f,g - example neurons and value decoding in the striatum.

Two example neurons, drawn from `data/figure3/example_psths.csv.gz` after the
decoding panels:

    b  relative value, contralateral-choice trials, aligned to the go cue
    c  state value, rewarded trials only, aligned to the CS+

Two decoding panels, both from groups of 20 randomly chosen neurons:

    f  relative value, contralateral-choice trials, aligned to the go cue
    g  state value, rewarded trials only, aligned to the CS+

Usage (interactive), or:
    python figure3.py
"""

# %%
import matplotlib.pyplot as plt

from decoding import (STRIATAL, apply_style, mark_ranksum, plot_timecourse,
                      prepare_panel, save_panel)

# %%
# ============================================================================
# Panels
# ============================================================================
# PROVISIONAL y-limits and x-ticks - confirm against the typeset figure.
YTICKS = [0.00, 0.05, 0.10, 0.15, 0.20]
PANELS = [
    dict(panel='f', decoder='relative_value', title="Relative Value",
         t0=-0.5, t1=0.5, xlabel="Time From Go Cue (s)", ylim=(-0.02, 0.10),
         xticks=[-0.4, 0.0, 0.4]),
    dict(panel='g', decoder='state_value', title="State Value",
         t0=-0.5, t1=2.0, xlabel="Time From CS+ (s)", ylim=(-0.02, 0.18),
         xticks=[-0.4, 0.0, 1.0, 2.0]),
]

N_NEURONS = 20


# %%
# ============================================================================
# Run
# ============================================================================
apply_style()

for spec in PANELS:
    real, stats, ranksum = prepare_panel(
        label=f"Fig 3{spec['panel']}  {spec['title']}",
        decoder=spec['decoder'], n_neurons=N_NEURONS, regions=STRIATAL,
        t0=spec['t0'], fisher=True, ranksum=True)

    fig, ax = plt.subplots(figsize=(3.0, 2.6))
    plot_timecourse(real, STRIATAL, ax,
                    xlim=(spec['t0'] + 0.1, spec['t1']), ylim=spec['ylim'],
                    xlabel=spec['xlabel'], title=spec['title'],
                    xticks=spec['xticks'], yticks=YTICKS)
    mark_ranksum(ax, ranksum)
    save_panel(fig, f"fig3{spec['panel']}_{spec['decoder']}.pdf")

print("\nDone.")


# %%
# ============================================================================
# Example neurons, 3b and 3c
# ============================================================================
import seaborn as sns

from psth_plots import HERE, load_example_psths, plot_quintile_psths

CACHE = HERE / "data" / "figure3" / "example_psths.csv.gz"

FONT_SIZE = 12
YLABEL = "Firing Rate (Hz)"
XLIM = (-0.2, 0.5)
XTICKS = [-0.2, 0.0, 0.2, 0.4]

EXAMPLES = [
    dict(panel="3b", neuron="#918", title="#918\n(Contra Trials)",
         color="r", xlabel="Time from go cue (s)", zero_line="--",
         yticks=[0, 20, 40, 60], filename="fig3b_example_relative_value.pdf"),
    dict(panel="3c", neuron="#1038", title="#1038\n(Rewarded Trials)",
         color="dodgerblue", xlabel="Time from CS+ (s)", zero_line="-",
         yticks=[0, 10, 20, 30], filename="fig3c_example_state_value.pdf"),
]


# %%
# ============================================================================
# Run - 3b, 3c
# ============================================================================
for spec in EXAMPLES:
    frame = load_example_psths(CACHE, spec["panel"])
    neuron_frame = frame.loc[frame["neuron"] == spec["neuron"]]
    if neuron_frame.empty:
        raise ValueError(f"no cached PSTH for neuron {spec['neuron']} in "
                         f"panel {spec['panel']}; cached: "
                         f"{sorted(frame['neuron'].unique())}")

    fig, ax = plt.subplots(figsize=(2.6, 2.4))
    plot_quintile_psths(ax, neuron_frame, spec["color"])
    ax.axvline(0, color="gray", linestyle=spec["zero_line"], linewidth=0.8)
    ax.set_title(spec["title"], fontsize=FONT_SIZE)
    ax.set_xlabel(spec["xlabel"], fontsize=FONT_SIZE)
    ax.set_ylabel(YLABEL, fontsize=FONT_SIZE)
    ax.set_xlim(*XLIM)
    ax.set_xticks(XTICKS)
    ax.set_ylim(bottom=0)
    ax.set_yticks(spec["yticks"])
    sns.despine(ax=ax)
    fig.tight_layout()
    save_panel(fig, spec["filename"])
    plt.close(fig)

print("\nDone.")
