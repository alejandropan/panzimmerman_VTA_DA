import numpy as np
import pandas as pd
from pathlib import Path
from one.api import ONE

one = ONE(base_url='https://openalyx.internationalbrainlab.org/', password='international', mode='local')
subjects_dir = Path(one.cache_dir) / 'Subjects'

# %% Session-level trial files
alf_dirs = sorted(p.parent for p in subjects_dir.glob('*/*/*/alf/_ibl_trials.table.pqt'))
if not alf_dirs:
    raise FileNotFoundError(f'No downloaded sessions under {subjects_dir}; run download_data.py first.')

for alf_dir in alf_dirs:
    table = pd.read_parquet(alf_dir / '_ibl_trials.table.pqt')
    np.save(alf_dir / '_ibl_trials.left_reward.npy', np.load(alf_dir / '_av_trials.leftReward.npy'))
    np.save(alf_dir / '_ibl_trials.right_reward.npy', np.load(alf_dir / '_av_trials.rightReward.npy'))
    np.save(alf_dir / '_ibl_trials.probabilityLeft.npy', table.probabilityLeft.to_numpy())
    np.save(alf_dir / '_ibl_trials.rewardVolume.npy', table.rewardVolume.to_numpy())
    np.save(alf_dir / '_ibl_trials.stimOn_times.npy', table.stimOn_times.to_numpy())
    np.save(alf_dir / '_ibl_trials.intervals.npy', table[['intervals_0', 'intervals_1']].to_numpy())
    print(f'Trials converted: {alf_dir}')

# %% Probe-level cluster and channel files
for alf_dir in alf_dirs:
    for probe_dir in sorted(alf_dir.glob('probe[0-9][0-9]/pykilosort')):
        if not (probe_dir / 'clusters_selection.npy').is_file():
            print(f'Skipped (no cluster selection): {probe_dir}')
            continue
        metrics_file = probe_dir / '_av_clusters.metrics.pqt'
        if not metrics_file.is_file():
            metrics_file = probe_dir / 'clusters.metrics.pqt'
        metrics = pd.read_parquet(metrics_file)
        labels = pd.read_table(probe_dir / '_av_clusters.curatedLabels.tsv').set_index('Unnamed: 0')['group']
        metrics['group'] = labels.reindex(metrics['cluster_id']).to_numpy()
        metrics.to_csv(probe_dir / 'clusters.metrics.csv', index=False)

        mlapdv = np.load(probe_dir / 'channels.mlapdv.npy')
        sites = pd.DataFrame(mlapdv, columns=['x', 'y', 'z'], index=[f'channel_{i}' for i in range(len(mlapdv))])
        sites.loc['origin'] = np.nan
        sites.to_json(probe_dir / 'channel_locations.json', orient='index')
        print(f'Probe converted: {probe_dir}')
