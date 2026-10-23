"""
Build the surrogate sessions the null decoders are fitted to.

Each one is the whole set of recordings concatenated and then circularly
shifted by a random amount, so behaviour and neural data keep their own
structure but lose their correspondence. `run_decoder_job.py` reads
`<index>.csv` from one of these folders for each of its 200 null fits.

The two sets differ only in which recordings go into the concatenation:

    laser_only          all 41 laser-only sessions
    laser_only_skips    the 24 with at least 5 unconsumed rewards
"""

import sys
from pathlib import Path
import random

import numpy as np
import pandas as pd

ANALYSIS_CODE_DIR = '/jukebox/witten/Alex/PYTHON/rewardworld/ephys/analysis'
if ANALYSIS_CODE_DIR not in sys.path:
    sys.path.append(ANALYSIS_CODE_DIR)

from ephys_alf_summary import alf                                 # noqa: E402
from decoder_specs import (N_NULLS, NULL_SESSIONS, ROOT,          # noqa: E402
                           resolve_sessions)

# Output folder -> which session set fills it.
SETS = {
    'laser_only': 'all',
    'laser_only_skips': 'with_skips',
}

SEED = 10
TRIALS_PER_NULL = 1000     # rows kept from the shifted concatenation
MIN_SHIFT = 700
DROP_FIRST = 10            # trials dropped from the start of each session

# Variables that have to run continuously across a session boundary rather
# than restarting: the sequential actor-critic's values accumulate.
CONTINUOUS_VARS = ['seq_ac_vcs', 'seq_ac_vchoice']


for folder_name, session_set in SETS.items():
    sessions = resolve_sessions(session_set)
    destination = Path(NULL_SESSIONS) / folder_name
    destination.mkdir(parents=True, exist_ok=True)
    print(f"{folder_name}: {len(sessions)} sessions -> {destination}")

    # --- concatenate, stitching the accumulating variables across joins ----
    prev_end = {v: None for v in CONTINUOUS_VARS}
    data = pd.DataFrame()

    for ses in [ROOT + s for s in sessions]:
        ses_data = alf(ses, ephys=False)
        ses_data.mouse = Path(ses).parent.parent.name
        ses_data.date = Path(ses).parent.name
        ses_data.ses = Path(ses).name
        data_temp = ses_data.to_df()[DROP_FIRST:]
        try:
            video_value_rl = np.load(ses + '/alf/decoder_output_deltaq_cue_forget_contra_videoPCs_rl_100_residual.npy')[DROP_FIRST:, :].astype(float)
            data_temp['video_value_rl'] = [video_value_rl[l, :] for l in np.arange(len(video_value_rl))]
            video_value_lr = np.load(ses + '/alf/decoder_output_deltaq_cue_forget_contra_videoPCs_lr_100_residual.npy')[DROP_FIRST:, :].astype(float)
            data_temp['video_value_lr'] = [video_value_lr[l, :] for l in np.arange(len(video_value_lr))]
            try:
                video_total_value = np.load(ses + '/alf/decoder_output_totalq_outcome_forget_rewarded_videoPCs_rl_100_residual.npy')[DROP_FIRST:, :].astype(float)
            except:
                video_total_value = np.load(ses + '/alf/decoder_output_totalq_outcome_forget_rewarded_videoPCs_100_residual.npy')[DROP_FIRST:, :].astype(float)
            data_temp['video_total_value'] = [video_total_value[l, :] for l in np.arange(len(video_total_value))]
        except:
            []

        for v in CONTINUOUS_VARS:
            if v not in data_temp.columns or not data_temp[v].notna().any():
                continue
            first_val = data_temp[v].dropna().iloc[0]
            if prev_end[v] is not None:
                data_temp[v] = data_temp[v] + (prev_end[v] - first_val)
            prev_end[v] = data_temp[v].dropna().iloc[-1]

        data = pd.concat([data, data_temp])
    data = data.reset_index()
    print(f"  {len(data)} trials concatenated")

    # --- shift and save ----------------------------------------------------
    random.seed(SEED)
    for n_null in np.arange(N_NULLS):
        shift = random.randint(MIN_SHIFT, len(data))
        data_t = data.reindex(index=np.roll(data.index, shift)).reset_index().copy()
        null = data_t.iloc[:TRIALS_PER_NULL, :]
        null.to_csv(destination / f'{n_null}.csv')
    print(f"  wrote {N_NULLS} surrogate sessions")