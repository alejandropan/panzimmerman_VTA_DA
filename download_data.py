from iblutil.util import Bunch
import sys 
import numpy as np
import pandas as pd
from one.api import ONE
from brainbox.io.one import SpikeSortingLoader
import os
from skimage import io
from iblatlas.atlas import AllenAtlas

#List of sessions
LASER_ONLY_ABB = ['/dop_56/2023-09-30/001',
 '/dop_56/2023-10-01/001',
 '/dop_56/2023-10-02/001',
 '/dop_56/2023-10-03/001',
 '/dop_56/2023-10-05/001',
 '/dop_57/2023-10-17/002',
 '/dop_57/2023-10-19/002',
 '/dop_57/2023-10-20/001',
 '/dop_59/2023-10-23/003',
 '/dop_59/2023-10-24/001',
 '/dop_59/2023-10-25/001',
 '/dop_59/2023-10-26/001',
 '/dop_59/2023-10-28/001',
 '/dop_59/2023-10-29/001',
 '/dop_49/2022-06-27/003',
 '/dop_49/2022-06-20/001',
 '/dop_49/2022-06-19/001',
 '/dop_49/2022-06-18/002',
 '/dop_49/2022-06-17/001',
 '/dop_49/2022-06-16/001',
 '/dop_49/2022-06-15/001',
 '/dop_49/2022-06-14/001',
 '/dop_48/2022-06-28/001',
 '/dop_48/2022-06-27/002',
 '/dop_48/2022-06-20/001',
 '/dop_48/2022-06-19/002',
 '/dop_47/2022-06-10/002',
 '/dop_47/2022-06-09/003',
 '/dop_47/2022-06-06/001',
 '/dop_47/2022-06-05/001',
 '/dop_50/2022-09-18/001',
 '/dop_50/2022-09-17/001',
 '/dop_50/2022-09-16/003',
 '/dop_50/2022-09-14/003',
 '/dop_50/2022-09-13/001',
 '/dop_50/2022-09-12/001',
 '/dop_53/2022-10-07/001',
 '/dop_53/2022-10-05/001',
 '/dop_53/2022-10-04/001',
 '/dop_53/2022-10-03/001',
 '/dop_53/2022-10-02/001']

# Load Histology files and parameters
ba = AllenAtlas()
BREGMA_ALLEN =  np.array([5400,332,5739])
img_path = r"histology_files\allen_subdivisions.tif"
subdivisions = pd.read_csv(r'histology_files\41467_2019_13057_MOESM4_ESM.csv')
nonstr_regions = pd.read_csv(r"histology_files\nonstr.csv")
im = io.imread(img_path)
imarray = np.array(im)

# Datasets to exclude from download due to size
heavy_datasets = [ 
    'alf/_ibl_bodyCamera.times.npy',
    'alf/_ibl_leftCamera.times.npy',
    'alf/_ibl_rightCamera.times.npy',
    'raw_ephys_data/probe01/_spikeglx_ephysData_g0_t0.imec1.sync.npy',
    'raw_ephys_data/probe01/_spikeglx_ephysData_g0_t0.imec1.timestamps.npy',
    'raw_ephys_data/probe01/_spikeglx_sync.channels.probe01.npy',
    'raw_ephys_data/probe01/_spikeglx_sync.polarities.probe01.npy',
    'raw_ephys_data/probe01/_spikeglx_sync.times.probe01.npy',
    'raw_video_data/_iblrig_bodyCamera.GPIO.bin',
    'raw_video_data/_iblrig_bodyCamera.frame_counter.bin',
    'raw_video_data/_iblrig_bodyCamera.raw.mp4',
    'raw_video_data/_iblrig_bodyCamera.timestamps.ssv',
    'raw_video_data/_iblrig_leftCamera.GPIO.bin',
    'raw_video_data/_iblrig_leftCamera.frame_counter.bin',
    'raw_video_data/_iblrig_leftCamera.raw.mp4',
    'raw_video_data/_iblrig_leftCamera.timestamps.ssv',
    'raw_video_data/_iblrig_rightCamera.GPIO.bin',
    'raw_video_data/_iblrig_rightCamera.frame_counter.bin',
    'raw_video_data/_iblrig_rightCamera.raw.mp4',
    'raw_video_data/_iblrig_rightCamera.timestamps.ssv'
    'spike_sorters/pykilosort/probe00/_kilosort_raw.output.tar',
    'spike_sorters/pykilosort/probe01/_kilosort_raw.output.tar',
    'raw_behavior_data/_iblrig_ambientSensorData.raw.jsonable',
    'raw_behavior_data/_iblrig_codeFiles.raw.zip',
    'raw_behavior_data/_iblrig_encoderEvents.raw.ssv',
    'raw_behavior_data/_iblrig_encoderPositions.raw.ssv',
    'raw_behavior_data/_iblrig_encoderTrialInfo.raw.ssv',
    'raw_behavior_data/_iblrig_stimPositionScreen.raw.csv',
    'raw_behavior_data/_iblrig_syncSquareUpdate.raw.csv',
    'raw_behavior_data/_iblrig_taskData.raw.jsonable',
    'raw_behavior_data/_iblrig_taskSettings.raw.json',
    'raw_ephys_data/_spikeglx_sync.channels.npy',
    'raw_ephys_data/_spikeglx_sync.polarities.npy',
    'raw_ephys_data/_spikeglx_sync.times.npy',
    'raw_ephys_data/probe00/_iblqc_ephysChannels.apRMS.npy',
    'raw_ephys_data/probe00/_iblqc_ephysChannels.labels.npy',
    'raw_ephys_data/probe00/_iblqc_ephysChannels.rawSpikeRates.npy',
    'raw_ephys_data/probe00/_iblqc_ephysSpectralDensityAP.freqs.npy',
    'raw_ephys_data/probe00/_iblqc_ephysSpectralDensityAP.power.npy',
    'raw_ephys_data/probe00/_iblqc_ephysSpectralDensityLF.freqs.npy',
    'raw_ephys_data/probe00/_iblqc_ephysSpectralDensityLF.power.npy',
    'raw_ephys_data/probe00/_iblqc_ephysTimeRmsLF.rms.npy',
    'raw_ephys_data/probe00/_spikeglx_ephysData_g0_t0.imec0.sync.npy',
    'raw_ephys_data/probe00/_spikeglx_ephysData_g0_t0.imec0.timestamps.npy',
    'raw_ephys_data/probe00/_spikeglx_sync.channels.probe00.npy',
    'raw_ephys_data/probe00/_spikeglx_sync.polarities.probe00.npy',
    'raw_ephys_data/probe00/_spikeglx_sync.times.probe00.npy',
    'raw_ephys_data/probe01/_iblqc_ephysChannels.apRMS.npy',
    'raw_ephys_data/probe01/_iblqc_ephysChannels.labels.npy',
    'raw_ephys_data/probe01/_iblqc_ephysChannels.rawSpikeRates.npy',
    'raw_ephys_data/probe01/_iblqc_ephysSpectralDensityAP.freqs.npy',
    'raw_ephys_data/probe01/_iblqc_ephysSpectralDensityAP.power.npy',
    'raw_ephys_data/probe01/_iblqc_ephysSpectralDensityLF.freqs.npy',
    'raw_ephys_data/probe01/_iblqc_ephysSpectralDensityLF.power.npy',
    'raw_ephys_data/probe01/_iblqc_ephysTimeRmsLF.rms.npy',
    'raw_ephys_data/probe01/_iblqc_ephysTimeRmsLF.timestamps.npy',
    'raw_video_data/_iblrig_rightCamera.timestamps.ssv',
    'spike_sorters/pykilosort/probe00/_kilosort_raw.output.tar']

one=ONE(base_url='https://openalyx.internationalbrainlab.org/', password='international')
org_root = r"\\cup.princeton.edu\witten\Alex\Data\Subjects"

for sess in LASER_ONLY_ABB:
    # Start ONE for download
    org_path = org_root + sess

    eid = one.path2eid(sess)
    datasets_to_download = one.list_datasets(eid)


    EXCLUDED_SUFFIXES = (".ch", ".bin", ".cbin", ".meta", ".mp4", ".tar")
    datasets_to_download = [item for item in datasets_to_download if not item.endswith(EXCLUDED_SUFFIXES)]


    datasets_to_download = np.array(datasets_to_download)[~np.isin(datasets_to_download,heavy_datasets)]

    print(f"Found {len(datasets_to_download)} datasets for session {eid}.")
    #print("Starting download... (This may take a while)")

    downloaded_file_paths = []
    failed_downloads = []

    # Loop through the list of dataset names
    for dset_name in datasets_to_download:
        try:
            # Use one.load_dataset() with download_only=True
            # This downloads the file and returns its local path
            # without loading it into memory.
            print(f"Downloading: {dset_name}")
            file_path = one.load_dataset(eid, dset_name, download_only=True)
            downloaded_file_paths.append(file_path)
            
        except Exception as e:
            print(f"ERROR: Failed to download {dset_name}: {e}")
            failed_downloads.append(dset_name)

    print("\n--- Download Complete ---")
    print(f"Successfully downloaded {len(downloaded_file_paths)} datasets.")
    if failed_downloads:
        print(f"Failed to download {len(failed_downloads)} datasets:")
        for failed in failed_downloads:
            print(f"  - {failed}")

    download_dir = os.path.dirname(downloaded_file_paths[0])

    insertions = one.alyx.rest('insertions', 'list', session=eid)
    probe_eids = [ins['id'] for ins in insertions]
    for ins in insertions:
        if ins['id'] == 'ea2ccafc-12c1-4c58-9ca4-ec2b827548f3': # this the id of a bad probe 
            continue
        try:
            spike_data_metrics = pd.read_parquet(download_dir + '/' + ins['name'] + r"/pykilosort/_av_clusters.metrics.pqt")
        except:
            spike_data_metrics = pd.read_parquet(download_dir + '/' + ins['name'] + r"/pykilosort/clusters.metrics.pqt")
        spike_data_labels = pd.read_table(download_dir + '/' + ins['name'] + r'/pykilosort/_av_clusters.curatedLabels.tsv')
        min_amp=30
        labels_select = spike_data_labels.loc[np.isin(spike_data_labels['group'],['good','mua']),'Unnamed: 0'].to_numpy()
        fr_select = spike_data_metrics.loc[spike_data_metrics['firing_rate']>=0.01, 'cluster_id'].to_numpy()
        RP_select = spike_data_metrics.loc[spike_data_metrics['contamination']<=0.1, 'cluster_id'].to_numpy()
        amp_select = spike_data_metrics.loc[spike_data_metrics['amp_median']>(min_amp/1e6),'cluster_id'].to_numpy()
        selection = np.intersect1d(labels_select,RP_select)
        selection = np.intersect1d(selection, amp_select)
        selection = np.intersect1d(selection, fr_select)
        P_select = spike_data_metrics.loc[spike_data_metrics['presence_ratio']>=0.75, 'cluster_id'].to_numpy()
        mua_selection = np.intersect1d(amp_select, P_select) # for decoders only use single or mua present for at least 75% of the recording
        mua_selection = np.intersect1d(mua_selection, fr_select)
        mua_selection = np.intersect1d(mua_selection, labels_select) #addition jan 25
        np.save(download_dir + '/' + ins['name'] + r'/pykilosort/clusters_selection.npy', selection)
        np.save(download_dir + '/' + ins['name'] + r'/pykilosort/clusters_goodmua_selection.npy', mua_selection)
        ### Now make the cluster histology files
        loc_codes = np.load(download_dir + '/' + ins['name'] + r'/pykilosort/channels.brainLocationIds_ccf_2017.npy')
        coords = np.load(download_dir + '/' + ins['name'] + r'/pykilosort/channels.mlapdv.npy')
        allen_channels = ba.regions.id2acronym(loc_codes)
        striatal_channels = np.where(np.isin(allen_channels,['ACB','CP','STR']))[0]
        allen_w_kim_channels = np.copy(allen_channels)
        for i in striatal_channels:
            subarea=[]
            coord = [coords[i,1], coords[i,2],coords[i,0]] # AP, DV, ML
            # Coordinates in Allen_10 space
            coord[0] = -1*(coord[0]) # AP every 10um, positive is more posterior
            coord[1] = -1*(coord[1]) # DV every 10um, down is positive
            coord[2] = coord[2] # ML every 10um
            coords_10 = BREGMA_ALLEN + coord
            # Kim atlas AP offset (950um)
            coords_offset = np.array([coords_10[0]-950,coords_10[1],coords_10[2]]) #950 because of python numbering -850-100
            # Now round to closest voxel
            label_idx =  np.array([int(np.round(coords_offset[0]/100)), int(coords_offset[1]/10),
                                int(coords_offset[2]/10)])
            label = imarray[label_idx[0], label_idx[1], label_idx[2]]
            if label>0:
                subarea = subdivisions.loc[subdivisions['Structural ID']==label, 'Franklin-Paxinos Full name'].to_list()[0]
            if np.isin(subarea,nonstr_regions):
                if int(np.round(coords_offset[0]/100)) != int(coords_offset[0]/100):
                    label_idx[0] = int(coords_offset[0]/100)
                else:
                    label_idx[0] = int(np.ceil(coords_offset[0]/100))
                label = imarray[label_idx[0], label_idx[1], label_idx[2]]
                if label>0:
                    subarea = subdivisions.loc[subdivisions['Structural ID']==label, 'Franklin-Paxinos Full name'].to_list()[0]
            if ~np.isin(subarea,nonstr_regions):
                allen_w_kim_channels[i] = subarea
        np.save(download_dir + '/' + ins['name'] + r'/pykilosort/channels.locations.npy', allen_w_kim_channels)
        np.save(download_dir + '/' + ins['name'] + r'/pykilosort/channels.hemisphere.npy', 1*(coords[:,0]>0))


    print(download_dir)
    laserOnset_times_one = np.load(download_dir + "\_av_trials.laserOnset_times.npy")
    feedback_one = np.load(download_dir + "\_av_trials.feedbackType.npy")
    leftReward_one = np.load(download_dir + "\_av_trials.leftReward.npy")
    rightReward_one = np.load(download_dir + "\_av_trials.rightReward.npy")
    goCueTrigger_times_one = np.load(download_dir + "\_ibl_trials.goCueTrigger_times.npy")
    table_one = pd.read_parquet(download_dir + "\_ibl_trials.table.pqt")
    position_one = np.load(download_dir + "\_ibl_wheel.position.npy")
    timestamps_one = np.load(download_dir + "\_ibl_wheel.timestamps.npy")
    intervals_one = np.load(download_dir + "\_ibl_wheelMoves.intervals.npy")
    peakAmplitude_one = np.load(download_dir + "\_ibl_wheelMoves.peakAmplitude.npy")
    probabilityRewardLeft_one = np.load(download_dir + "\_av_trials.probabilityRewardLeft.npy")

    goCue_times_one = table_one.goCue_times
    response_times_one = table_one.response_times
    choice_one = table_one.choice
    feedback_times_one = table_one.feedback_times
    firstMovement_times_one = table_one.firstMovement_times
    feedback_s_one = table_one.feedbackType
    feedbackTypeConsumed_one = 1*(feedback_one==1)
    feedbackTypeConsumed_one[feedbackTypeConsumed_one==0] = -1
    first_laser_times_one = laserOnset_times_one[~np.isnan(laserOnset_times_one)]

    np.save(download_dir + "\_ibl_trials.goCue_times.npy", goCue_times_one)
    np.save(download_dir + "\_ibl_trials.response_times.npy", response_times_one)
    np.save(download_dir + "\_ibl_trials.choice.npy", choice_one)
    np.save(download_dir + "\_ibl_trials.feedback_times.npy", feedback_times_one)
    np.save(download_dir + "\_ibl_trials.firstMovement_times.npy", firstMovement_times_one)
    np.save(download_dir + "\_ibl_trials.feedbackType.npy", feedback_s_one)
    np.save(download_dir + "\_ibl_trials.feedbackTypeConsumed.npy", feedbackTypeConsumed_one)
    np.save(download_dir + "\_ibl_trials.first_laser_times.npy", first_laser_times_one)

    # Value variables
    modelpred = pd.read_parquet(download_dir + "\_av_trials.modelPredictions.pqt")
    np.save(download_dir + "\QFS_median_choice_prediction.npy", modelpred.QFS_median_choice_prediction.to_numpy())
    np.save(download_dir + "\QFS_median_QL.npy", modelpred.QFS_median_QL.to_numpy())
    np.save(download_dir + "\QFS_median_QLreward.npy", modelpred.QFS_median_QLreward.to_numpy())
    np.save(download_dir + "\QFS_median_QLstay.npy", modelpred.QFS_median_QLstay.to_numpy())
    np.save(download_dir + "\QFS_median_QR.npy", modelpred.QFS_median_QR.to_numpy())
    np.save(download_dir + "\QFS_median_QRreward.npy", modelpred.QFS_median_QRreward.to_numpy())
    np.save(download_dir + "\QFS_median_QRstay.npy", modelpred.QFS_median_QRstay.to_numpy())
    np.save(download_dir + "\QFS_median_V.npy", modelpred.QFS_median_V.to_numpy())
    #np.save(download_dir + "\QFS_medianmodel_params.pkl", modelpred.QFS_median_choice_prediction.to_numpy())
    np.save(download_dir + "\QFS_RPE_median_choice_prediction.npy", modelpred.QFS_RPE_median_choice_prediction.to_numpy())
    np.save(download_dir + "\QFS_RPE_median_QL.npy", modelpred.QFS_RPE_median_QL.to_numpy())
    np.save(download_dir + "\QFS_RPE_median_QLreward.npy", modelpred.QFS_RPE_median_QLreward.to_numpy())
    np.save(download_dir + "\QFS_RPE_median_QLstay.npy", modelpred.QFS_RPE_median_QLstay.to_numpy())
    np.save(download_dir + "\QFS_RPE_median_QR.npy", modelpred.QFS_RPE_median_QR.to_numpy())
    np.save(download_dir + "\QFS_RPE_median_QRreward.npy", modelpred.QFS_RPE_median_QRreward.to_numpy())
    np.save(download_dir + "\QFS_RPE_median_QRstay.npy", modelpred.QFS_RPE_median_QRstay.to_numpy())
    np.save(download_dir + "\QFS_RPE_median_V.npy", modelpred.QFS_RPE_median_V.to_numpy())
    np.save(download_dir + "\AC_median_reward.npy", modelpred.AC_median_reward.to_numpy())
    np.save(download_dir + "\AC_median_choice_prediction.npy", modelpred.AC_median_choice_prediction.to_numpy())
    np.save(download_dir + "\AC_median_V.npy", modelpred.AC_median_V.to_numpy())
    np.save(download_dir + "\REINFORCE_median_choice_prediction.npy", modelpred.REINFORCE_median_choice_prediction.to_numpy())
    np.save(download_dir + "\REINFORCE_median_reward.npy", modelpred.REINFORCE_median_reward.to_numpy())