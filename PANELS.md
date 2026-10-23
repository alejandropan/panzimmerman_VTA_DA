# Panel manifest

Every published panel, the script that draws it, and whether it runs from the
data in this repository.

`repo` — runs from files included here.
`repo, R` — as `repo`, plus R with lme4 for one statistic (see the README).
`repo, IBL environment` — as `repo`, run in `environment-ibl.yml` (see the README).

## Figure 1

| panel | script | data |
|---|---|---|
| 1c | `figure1.py` | repo |
| 1d | `figure1.py` | repo |
| 1e | `figure1.py` | repo |
| 1f | `figure1.py` | repo |
| 1g | `figure1.py` | repo |
| 1h | `figure1.py` | repo |
| 1i | `figure1.py` | repo |
| 1j | `figure1.py` | repo |

## Figure 2

| panel | script | data |
|---|---|---|
| 2a | `figure2a.py` | repo, IBL environment |
| 2k–2n | `figure2.py` | repo |

## Figure 3

| panel | script | data |
|---|---|---|
| 3b, 3c | `figure3.py` | repo |
| 3f, 3g | `figure3.py` | repo |
| 3k | `figure3k.py` | repo |

## Figure 5

| panel | script | data |
|---|---|---|
| 5d | `figure5.py` | repo |
| 5e | `figure5.py` | repo |
| 5f | `figure5.py` | repo |
| 5h | `figure5.py` | repo |

## Figure 6

| panel | script | data |
|---|---|---|
| 6b–6g | `figure6.py` | repo |
| 6k, 6l, 6p, 6q | `figure6_decoding.py` | repo |

## Extended Data Figure 2

| panel | script | data |
|---|---|---|
| a | `extended_data_figure2.py` | repo |
| b | `extended_data_figure2.py` | repo |
| c | `extended_data_figure2.py` | repo |
| d–h | `extended_data_figure2.py` | repo |

## Extended Data Figure 4

| panel | script | data |
|---|---|---|
| f–i | `extended_data_figure4.py` | repo |

## Extended Data Figure 5

| panel | script | data |
|---|---|---|
| a–d | `extended_data_figure5.py` | repo |

## Extended Data Figure 6

| panel | script | data |
|---|---|---|
| example traces, evoked summary | `extended_data_figure6.py` | repo |

## Extended Data Figure 7

| panel | script | data |
|---|---|---|
| i, j, p, q | `extended_data_figure7.py` | repo |

## Extended Data Figure 10

| panel | script | data |
|---|---|---|
| a, b, c | `extended_data_figure10.py` | repo |

## Extended Data Figure 11

| panel | script | data |
|---|---|---|
| b, c, d | `extended_data_figure11.py` | repo |
| e, g, h, i, j | `extended_data_figure11.py` | repo |
| f | `extended_data_figure11.py` | repo |
| k, m, n | `extended_data_figure11.py` | repo |
| l | `extended_data_figure11.py` | repo, R |

Panel a of several figures is a schematic and has no code.

Panels 2b and 2d were made by a collaborator and are not in this repository.

## Extended Data Figure 12

| panel | script | data |
|---|---|---|
| d | `extended_data_figure12.py` | repo |

## Compute pipeline

The scripts under `pipeline/` produce the artifacts the figure scripts read.
They are not needed to draw any panel from the files included here.

| script | produces |
|---|---|
| `compute_loo.py` | per-model PSIS-LOO pickles |
| `compute_loo_comparison.py` | `loo_comparison_5models.csv`, `loo_comparison_6models.csv` |
| `confusion_matrix.py` | `confusion_matrix.csv` for ED Fig 2c |
| `extract_decision_variables.py` | the per-trial decision-variable arrays |
| `launch_decoders.py`, `run_decoder_job.py`, `decoder_specs.py`, `decoding_functions.py` | the decoder outputs under `decoders_data/` |
| `compile_decoders.py` | the decoder summary tables |
| `null_distributions.py` | the decoder null distributions |
| `prepare_figure_data.py`, `prepare_remaining_data.py` | the tables under `data/<figure>/`; need the raw recordings |
| `prepare_example_psths.py` | `data/figure3/example_psths.csv.gz` (Fig 3b,c); needs the raw recordings |
| `prepare_probe_tracks.py` | `data/figure2/probe_tracks.csv` (Fig 2a); needs the raw recordings |
