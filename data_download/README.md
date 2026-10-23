# Data release: IBL recordings, download and conversion

Code for retrieving the electrophysiology and behaviour of the 41 sessions in
**[VTA dopamine neuron activity produces spatially organized value representations](https://www.biorxiv.org/content/10.1101/2025.11.04.685995v1)**
from the International Brain Laboratory OpenAlyx server, and for converting the
public files into the layout that `ephys_alf_summary.alf()` reads.

None of this is needed to draw the figures. Every figure script in the repository
root reads the tables and caches shipped in `data/` and `decoders_data/`. Use this
folder to rebuild those caches from the recordings, or to analyse the sessions
yourself.

## Prerequisites

Work in the IBL environment defined at the repository root:

```
conda env create -f environment-ibl.yml
conda activate panzimmerman2026-ibl
```

It provides `one-api`, `ibllib`, `iblatlas`, `scikit-image`, `numpy` and `pandas`.
See the [IBL installation instructions](https://int-brain-lab.github.io/iblenv/installation.html)
if you prefer to use your own IBL environment.

The cluster histology is relabelled with the Chon et al. 2019 striatal atlas
(https://www.nature.com/articles/s41467-019-13057-w). Create a folder named
`histology_files` beside `download_data.py` and place in it `allen_subdivisions.tif`
(extracted from `concatenated_subdivisions_from_Chonetal2019_NatComms.zip`),
`41467_2019_13057_MOESM4_ESM.csv` and `nonstr.csv`:

```text
data_download/
├── download_data.py
├── convert_to_alf.py
└── histology_files/
    ├── allen_subdivisions.tif
    ├── 41467_2019_13057_MOESM4_ESM.csv
    └── nonstr.csv
```

## Workflow

Run both scripts from any directory. Neither takes arguments.

1. `python download_data.py` downloads every session from OpenAlyx into the ONE
   cache (by default `~/Downloads/ONE/openalyx.internationalbrainlab.org`, under
   `Subjects/<mouse>/<date>/<session>/`). Video, raw ephys and other heavy files
   are skipped. For each session it then writes, beside the downloaded files, the
   trial arrays (`_ibl_trials.choice.npy`, `feedbackType`, `goCue_times`,
   `response_times`, `feedback_times`, `firstMovement_times`,
   `feedbackTypeConsumed`, `first_laser_times`) and, for each probe, the cluster
   selections (`clusters_selection.npy`, `clusters_goodmua_selection.npy`) and the
   channel histology (`channels.locations.npy`, `channels.hemisphere.npy`).
2. `python convert_to_alf.py` adds the remaining files the loader expects:
   `_ibl_trials.left_reward`, `right_reward`, `probabilityLeft`, `rewardVolume`,
   `stimOn_times` and `intervals`, and for each probe `clusters.metrics.csv` and
   `channel_locations.json`. It converts every session found in the ONE cache.

After this, `ephys_alf_summary.alf(<session folder>, ephys=True)` loads a session.
To rebuild the shipped caches, point `SUBJECTS` at the cache's `Subjects` folder in
the scripts under `pipeline/` (`prepare_figure_data.py`,
`prepare_remaining_data.py`, `prepare_example_psths.py`, `prepare_probe_tracks.py`,
the decoder jobs) and run them.

## Model variables

The decision variables of the fitted models (`QFS_median_*`, `REINFORCE_median_*`,
`AC_median_*`, `SeqAC_noawin_m_avchoice_median_*`) are **not** part of the public
download. They are produced from the Stan posteriors by
`pipeline/extract_decision_variables.py`. The Stan chains are not distributed:
the models are in `data/stan/` and can be refitted, then `STAN_ROOT` in the
extractor points at the new chains. Without them, `alf()` still loads the trials
and spikes and skips the model blocks, and every figure still draws from the
cached tables in `data/`.
