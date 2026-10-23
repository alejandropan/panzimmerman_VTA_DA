# VTA dopamine neuron activity produces spatially organized stimulus and action value representations through conditioned reinforcement

Analysis code for Pan-Vazquez, Zimmerman et al.

## Installation

```
conda env create -f environment.yml
conda activate panzimmerman2026
```

A second environment is needed only for the compute pipeline and for Extended
Data Figure 11l, which fits a GLMM through R:

```
conda env create -f environment-compute.yml
conda activate panzimmerman2026-compute
```

Figure 2a, the 3D brain, and every script that reads the raw recordings
(`data_download/` and the example-neuron and probe-track builders in
`pipeline/`) need a third environment with the IBL packages and Mayavi, plus the
IBL `iblapps` repository, which provides `atlaselectrophysiology.rendering` and
is not on PyPI. Clone it and put it on `PYTHONPATH`:

```
conda env create -f environment-ibl.yml
conda activate panzimmerman2026-ibl
git clone https://github.com/int-brain-lab/iblapps
git -C iblapps checkout 74e6fd58dc5ba623470151d7c78aa95dea4b230a
set PYTHONPATH=%CD%\iblapps
```

`figure2a.py` is interactive: rotate the scene to the published view, then run
the last cell to save the screenshot.

## Drawing a figure

Run any script from the repository root:

```
python figure6_decoding.py
```

Every script is written to run cell by cell as well. Cells are marked `# %%`,
so the same file can be stepped through in VS Code or Spyder. There are no
`main()` functions and no command-line arguments; the configuration is a block
of constants at the top of each file.

Each panel is written as a PDF to the repository root (`fig2k_…pdf`, `figED5a_…pdf`).

To draw every figure in one go:

```
python make_all_figures.py
```

Each script runs in its own process and its console output is kept in
`logs/`. The summary lists, per script, whether it finished and how many PDFs it
wrote. A failure does not stop the rest. Extended Data Figure 11 runs last of the batch
and takes about fourteen minutes. Then `figure2a.py` opens, because it is
interactive: it opens in the IBL environment (`panzimmerman2026-ibl`) as an
IPython session with the 3D scene. Rotate it to the published view, run
`mlab.savefig(str(SAVE_PATH), size=SAVE_SIZE)`, then `exit()`. It is skipped when
that environment is not installed.

## What runs from this repository

Every script listed in `PANELS.md` runs as cloned, without the raw recordings.
They read:

- the decoder summaries in `decoders_data/` (Figures 2, 3 and 6, Extended Data
  Figures 4, 5, 7 and 12);
- the photometry traces in `extended_data_figure6_traces.csv` (Extended Data
  Figure 6);
- the tables, Stan fits and model-comparison artifacts in `data/` (Figures 1,
  3k, 5 and 6, Extended Data Figures 2, 10 and 11).

The one exception is the GLMM in Extended Data Figure 11l, which is fitted with
lme4 through R and needs the compute environment above. Without it that cell
prints a message and the rest of the figure is drawn.

The raw behavioural and electrophysiological recordings are not in this
repository. The tables under `data/<figure>/` were derived from them by
`pipeline/prepare_figure_data.py` and `pipeline/prepare_remaining_data.py`,
which need those recordings. The recordings of the 41 sessions are public on the
IBL OpenAlyx server. `data_download/` retrieves them and converts them to the
layout the loader reads: download, convert, then the builders in `pipeline/`
(see `data_download/README.md`). Model decision variables are not downloaded;
`pipeline/extract_decision_variables.py` produces them from the Stan posteriors.

## Layout

```
figure*.py                        one script per figure
extended_data_figure*.py          one script per extended data figure
figure5_data.py, figure6_data.py  loaders and statistics shared within a figure
decoding.py                       shared by the decoder panels
model_comparison_accu.py          behavioural models and their forward passes
ephys_alf_summary.py              the trials loader and the session list
session_maker.py                  the task's trial generator
encoding_model_summary_to_df.py   the encoding model reader

data_download/                    retrieves the public recordings from IBL
pipeline/                         the compute that produced the artifacts
data/                             Stan models, fits and summary tables
decoders_data/                    decoder outputs, read by the decoder panels
```

`ephys_alf_summary.py` is a reduced copy of the laboratory module: the classes
and functions no published panel reaches have been removed, so that drawing a
figure does not require the full electrophysiology stack.

## Models

`data/stan/` holds the six Stan models compared in Extended Data Figure 2a,
unmodified, each with the script that launched its fit. The model numbering
follows the paper.

## Reproducing the compute

The pipeline was run under `environment-compute.yml` on a cluster. The decoder
jobs in particular take many hours across many nodes. Re-running them is not
necessary to draw any figure, and the shipped artifacts were produced by the
versions pinned in that file; a re-run under different versions may not
reproduce them exactly.

## License

The code and the derived tables, fits and decoder outputs in this repository are
released under the MIT License (`LICENSE`). Third-party material keeps its own
terms: the Chon et al. 2019 striatal atlas files in
`data_download/histology_files/`, and the raw recordings, which are served by the
International Brain Laboratory on OpenAlyx under IBL's terms.

## Citation

Please cite the paper.
