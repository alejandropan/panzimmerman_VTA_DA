"""
Run one decoder for one (session, area) task.

    python run_decoder_job.py <spec name> <array index>

One SLURM array task = one row of the spec's `id_dict.csv` = one recording in
one region. The real fit runs first, then 200 nulls. Everything that differs
between decoders comes from `decoder_specs.py`; nothing decoder-specific is
written here.

This replaces thirteen worker scripts that were identical apart from the
fields in that table. Three things they did by hand are gone:

  * the location type is no longer `types[int(sys.argv[2])]` with the output
    folder built from it, so no folder needs renaming after a run;
  * there is no `NULLS` flag - the real fit and the nulls always both run,
    since commenting one out was how re-runs were staged;
  * there is no `missing_nulls.pkl` resume branch.

The numerics are untouched: this calls the same
`run_decoder_for_session_residual` with the same arguments the originals
passed, which `test_worker_equivalence.py` checks call by call.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ANALYSIS_CODE_DIR = '/jukebox/witten/Alex/PYTHON/rewardworld/ephys/analysis'
if ANALYSIS_CODE_DIR not in sys.path:
    sys.path.append(ANALYSIS_CODE_DIR)

from ephys_alf_summary import alf                                # noqa: E402
from encoding_model_summary_to_df import (load_all_residuals,    # noqa: E402
                                          common_neural_data)
from decoding_functions import run_decoder_for_session_residual  # noqa: E402
from decoder_specs import (MIN_TRIAL_FRACTION, N_NULLS,          # noqa: E402
                           NULL_SESSIONS, REAL_NEURON_COMBOS, ROOT,
                           ROOT_NEURAL, SPECS)

import warnings                                                  # noqa: E402
warnings.filterwarnings('ignore')


def load_session(session_path):
    """
    Load one session's behaviour and label it.

    Args:
        session_path: Absolute path to the session.

    Returns:
        The `alf` object, with mouse / date / ses / path set - the decoder
        builds its output filenames from those.
    """
    alfio = alf(session_path, ephys=True)
    alfio.mouse = Path(session_path).parent.parent.name
    alfio.date = Path(session_path).parent.name
    alfio.ses = Path(session_path).name
    alfio.path = session_path
    return alfio


def load_region_neurons(session, area, filetype, criterion):
    """
    Every usable neuron this task's region contributes.

    Args:
        session: Session path relative to the Subjects root, as `id_dict`
            stores it.
        area: The region for this array task.
        filetype: What `load_all_residuals` reads - 'real' for raw spikes,
            `residual_crac_*` for the CRAC model's residuals.
        criterion: None keeps good and mua units, 'good' keeps only good.

    Returns:
        The neurons in `area`, restricted to those present on enough trials.
    """
    neurons = load_all_residuals(
        f'{ROOT_NEURAL}/{session}/alf/encodingmodels/inputs/neurons/',
        filetype=filetype, criterion=criterion)
    # A region outside the striatal `level_map` has no level label; fall back
    # to its group label so non-striatal areas can still be selected.
    unlabelled = neurons.location_level.isna()
    neurons.loc[unlabelled, 'location_level'] = neurons.loc[unlabelled, 'location']
    return neurons.loc[neurons['location_level'] == area]


def load_null_session(null_folder, index):
    """
    One surrogate session, as the decoders read it.

    Args:
        null_folder: 'laser_only' or 'laser_only_skips'.
        index: Which of the 200.

    Returns:
        The table from `choice` rightwards. The originals sliced positionally
        like this and then read columns by name; kept as-is.
    """
    table = pd.read_csv(f'{NULL_SESSIONS}/{null_folder}/{index}.csv')
    return table.iloc[:, np.where(table.columns == 'choice')[0][0]:]


# ============================================================================
# Run
# ============================================================================
if len(sys.argv) < 3:
    raise SystemExit(f"usage: {Path(sys.argv[0]).name} <spec name> "
                     f"<array index>\nspecs: {', '.join(SPECS)}")

SPEC_NAME, ARRAY_INDEX = sys.argv[1], int(sys.argv[2])
if SPEC_NAME not in SPECS:
    raise SystemExit(f"unknown spec {SPEC_NAME!r}. Known: {', '.join(SPECS)}")
spec = SPECS[SPEC_NAME]
output_folder = spec['folder']

# --- which session and region this task is ----------------------------------
id_dict = pd.read_csv(f'{output_folder}/id_dict.csv')
row = id_dict.loc[id_dict['id'] == ARRAY_INDEX]
if len(row) != 1:
    raise SystemExit(f"{SPEC_NAME}: id_dict.csv has {len(row)} rows with id "
                     f"{ARRAY_INDEX}, expected exactly 1")
session, area = row['ses'].iloc[0], row['area'].iloc[0]
print(f"{SPEC_NAME}  task {ARRAY_INDEX}: {session}  {area}")

# --- the data ---------------------------------------------------------------
alfio = load_session(ROOT + session)
neurons = load_region_neurons(session, area, spec['filetype'], spec['criterion'])
c_neural_data = common_neural_data(
    neurons, n_trials_minimum=int(MIN_TRIAL_FRACTION * len(alfio.choice)))
print(f"  {len(c_neural_data)} neurons, {len(alfio.choice)} trials")

# Sample weighting was available but never used; every decoder ran unweighted.
weights = None

# --- the real fit -----------------------------------------------------------
regressed_variable, trial_filter = spec['real'](alfio)
run_decoder_for_session_residual(
    c_neural_data, area, alfio, regressed_variable, weights, spec['alignment'],
    etype='real', output_folder=output_folder, trial_filter=trial_filter,
    n_neurons_minimum=REAL_NEURON_COMBOS)

# --- the nulls --------------------------------------------------------------
# Drawn from the surrogate set built on the same recordings this spec runs on;
# `decoder_specs` pairs the two and refuses a mismatch.
print(f"  {N_NULLS} nulls from {spec['null_folder']}, "
      f"{spec['null_n']} neurons each")
for null_index in range(N_NULLS):
    null_table = load_null_session(spec['null_folder'], null_index)
    null_variable, null_filter = spec['null'](null_table, alfio)
    run_decoder_for_session_residual(
        c_neural_data, area, alfio, null_variable, weights, spec['alignment'],
        etype='null', n=null_index, output_folder=output_folder,
        trial_filter=null_filter, n_neurons_minimum=spec['null_n'])

print(f"{SPEC_NAME}  task {ARRAY_INDEX}: done")
