import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib import gridspec
from pathlib import Path
from glob import glob
import os
from scipy.stats import ttest_rel as pairedt
from scipy.stats import zscore
from functools import reduce
import copy
from scipy.stats import ttest_rel as pttest
from decoders_yield_availability import (community_map, custom_map,
                                          level_map)
from itertools import permutations 

HERE = (Path(__file__).resolve().parent if "__file__" in globals()
        else Path.cwd())
SIMPLIFIED_REGIONS = (HERE / "decoders_data" / "histology_files"
                      / "simplified_regions.csv")

LASER_ONLY_ABB = [
'/dop_56/2023-09-30/001',
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
'/dop_53/2022-10-02/001'
]


def add_transition_info(ses_d, trials_forward=10):
    trials_back=5 # Current cannot be changed
    ses_df = ses_d.copy()
    ses_df['choice_1'] = ses_df['choice']>0
    ses_df['transition_analysis'] = np.nan
    ses_df['transition_type'] = np.nan
    ses_df['transition_analysis_real'] = np.nan
    ses_df['transition_type_real'] = np.nan
    ses_df.loc[ses_df['trial_within_block']<trials_forward,'transition_analysis']=1
    ses_df['transition_analysis_real']=1
    for i in np.arange(len(ses_df['block_number'].unique())):
        if i>0:
            ses_ses_past = ses_df.loc[(ses_df['block_number']==i) &
            (ses_df['transition_analysis']==1) &
            (ses_df['trial_within_block']<0)]

            ses_ses_next = ses_df.loc[(ses_df['block_number']==i) &
            (ses_df['transition_analysis']==1) &
            (ses_df['trial_within_block']>=0)]

            ses_df.loc[(ses_df['block_number']==i) &
            (ses_df['transition_analysis']==1) &
            (ses_df['trial_within_block']<0) ,'transition_type'] = \
                                    ses_ses_past['probabilityLeft'].astype(str)+ \
                                    ' to '\
                                   +ses_ses_past['probabilityLeft_next'].astype(str)

            ses_df.loc[(ses_df['block_number']==i) &
            (ses_df['transition_analysis']==1) &
            (ses_df['trial_within_block']>=0) ,'transition_type'] = \
                                    ses_ses_next['probabilityLeft_past'].astype(str)+ \
                                     ' to '\
                                   +ses_ses_next['probabilityLeft'].astype(str)

            blocks = np.array([0.1,0.7])
            ses_ses_next_real = ses_df.loc[(ses_df['block_number']==i) &
            (ses_df['transition_analysis_real']==1) &
            (ses_df['trial_within_block_real']>=0)]
            past_block = blocks[blocks!=ses_ses_next_real['probabilityLeft'].iloc[0]]

            ses_df.loc[(ses_df['block_number_real']==i) &
            (ses_df['transition_analysis_real']==1) &
            (ses_df['trial_within_block_real']>=0) ,'transition_type_real'] = \
                                    ses_ses_next_real['probabilityLeft'].iloc[0].astype(str) + \
                                     ' to '\
                                   + str(past_block[0])

    return ses_df

def trial_within_block(behav):
    behav['trial_within_block'] = np.nan
    behav['block_number'] = np.nan
    behav['trial_within_block_real'] = np.nan
    behav['probabilityLeft_next'] = np.nan # this is for plotting trials before block change
    behav['opto_block_next'] = np.nan # this is for plotting trials before block change
    behav['probabilityLeft_past'] = np.nan # this is for plotting trials before block change
    behav['opto_block_past'] = np.nan # this is for plotting trials before block change
    behav['block_change'] = np.concatenate([np.zeros(1),
                                            1*(np.diff(behav['probabilityLeft'])!=0)])
    block_switches = np.concatenate([np.zeros(1),
                                     behav.loc[behav['block_change']==1].index]).astype(int)
    col_trial_within_block = np.where(behav.columns == 'trial_within_block')[0][0]
    col_probabilityLeft_next = np.where(behav.columns == 'probabilityLeft_next')[0][0]
    col_block_number = np.where(behav.columns == 'block_number')[0][0]
    col_opto_block_next = np.where(behav.columns == 'opto_block_next')[0][0]
    col_opto_block_past = np.where(behav.columns == 'opto_block_past')[0][0]
    col_opto_probabilityLeft_past = np.where(behav.columns == 'probabilityLeft_past')[0][0]
    col_trial_within_block_real = np.where(behav.columns == 'trial_within_block_real')[0][0]
    for i in np.arange(len(block_switches)):
        if i == 0:
            # Trial within block and probability for plotting
            behav.iloc[block_switches[i]:block_switches[i+1], col_probabilityLeft_next] = \
                behav['probabilityLeft'][block_switches[i+1]]
            behav.iloc[block_switches[i]:block_switches[i+1], col_opto_block_next] = \
                        behav['opto_block'][block_switches[i]]
            behav.iloc[block_switches[i]:block_switches[i+1], col_opto_block_next] = \
            np.arange(block_switches[i+1] - block_switches[i])
            # Block number
            behav.iloc[block_switches[i]:block_switches[i+1], col_block_number] = i
            behav.iloc[block_switches[i]:block_switches[i+1], col_trial_within_block_real] = \
            np.arange(len(behav.iloc[block_switches[i]:block_switches[i+1], col_block_number]))
        elif i == len(block_switches)-1:
            # Trial within block and probability for plotting
            behav.iloc[block_switches[i]-5:, col_probabilityLeft_next] = \
                behav['probabilityLeft'][block_switches[i]]
            behav.iloc[block_switches[i]-5:, col_opto_block_next] = \
                behav['opto_block'][block_switches[i]]
            behav.iloc[block_switches[i]:, col_opto_probabilityLeft_past] = \
                behav['probabilityLeft'][block_switches[i-1]]
            behav.iloc[block_switches[i]-5:, col_opto_block_past] = \
                behav['opto_block'][block_switches[i-1]]
            behav.iloc[block_switches[i]-5:, col_trial_within_block] = \
                np.arange(-5, len(behav) - block_switches[i])
            # Block number
            behav.iloc[block_switches[i]:, col_block_number] = i
            behav.iloc[block_switches[i]:, col_trial_within_block_real] = np.arange(len(behav.iloc[block_switches[i]:, col_block_number]))
        else:
            # Trial within block and probability for plotting
            behav.iloc[block_switches[i]-5:block_switches[i+1], col_probabilityLeft_next] = \
                behav['probabilityLeft'][block_switches[i]]
            behav.iloc[block_switches[i]-5:block_switches[i+1], col_opto_block_next] = \
                behav['opto_block'][block_switches[i]]
            behav.iloc[block_switches[i]:block_switches[i+1], col_opto_probabilityLeft_past] = \
                behav['probabilityLeft'][block_switches[i-1]]
            behav.iloc[block_switches[i]:block_switches[i+1], col_opto_block_past] = \
                behav['opto_block'][block_switches[i-1]]
            behav.iloc[block_switches[i]-5:block_switches[i+1], col_trial_within_block] = \
                np.arange(-5, block_switches[i+1] - block_switches[i])
            # Block number
            behav.iloc[block_switches[i]:block_switches[i+1], col_block_number] = i
            behav.iloc[block_switches[i]:block_switches[i+1], col_trial_within_block_real] = np.arange(len(behav.iloc[block_switches[i]:block_switches[i+1], col_block_number]))
    #Assign next block to negative within trial
    behav['block_number_real'] = behav['block_number'].copy()
    behav.loc[behav['trial_within_block']<0,'block_number'] = \
                behav.loc[behav['trial_within_block']<0,'block_number']+1
    behav.loc[behav['trial_within_block']>=0,'opto_block_next'] = np.nan
    behav.loc[behav['trial_within_block']>=0,'probabilityLeft_next'] = np.nan
    behav.loc[behav['trial_within_block']<0,'opto_block_past'] = np.nan
    behav.loc[behav['trial_within_block']<0,'probabilityLeft_past'] = np.nan
    return behav


def average_hz_trial(cluster_id, spike_clusters, spike_times, epoch, window=-1):
    # Epoch - e.g go_cue_times
    # Window - window from epoch in seconds to calculate rate
    spikes_from_cluster = spike_times[np.where(spike_clusters==cluster_id)[0]]
    rate_trial = []
    for i in np.arange(len(epoch)):
        if window>0:
            rate = np.sum(np.where((spikes_from_cluster >= epoch[i])
                                    & (spikes_from_cluster <= epoch[i]+window)))/abs(window)
            rate_trial.append(rate)
        if window<0:
            rate = len(np.where((spikes_from_cluster <= epoch[i])
                                            & (spikes_from_cluster >= epoch[i]+window))[0])/abs(window)
            rate_trial.append(rate)
    return np.array(rate_trial)

def NormalizeData(data):
    return (data - np.min(data)) / (np.max(data) - np.min(data))




######## Classes
class probe:
    def __init__(self,path):
        self.path = path
        self.spike_times = np.load(path+'/spikes.times.npy')
        self.spike_clusters = np.load(path+'/spikes.clusters.npy')
        self.cluster_channels = np.load(path+'/clusters.channels.npy')
        self.channel_locations = np.load(path+'/channels.locations.npy', allow_pickle = True)
        self.channel_xyz = pd.read_json(path + '/channel_locations.json', orient='index').iloc[:-1,:3].to_numpy()/1000000
        self.cluster_xyz = np.zeros([len(self.cluster_channels),3])
        self.cluster_xyz[:] = np.nan
        self.cluster_xyz[~np.isnan(self.cluster_channels),:]=  self.channel_xyz[self.cluster_channels[~np.isnan(self.cluster_channels)].astype(int),:]
        self.channel_xyz  = np.load(path+'/channels.localCoordinates.npy')
        self.channel_hem = np.load(path+'/channels.hemisphere.npy')
        self.cluster_hem = np.zeros([len(self.cluster_channels)])
        self.cluster_hem[:] = np.nan
        self.cluster_hem[~np.isnan(self.cluster_channels)] = \
            self.channel_hem[self.cluster_channels[~np.isnan(self.cluster_channels)].astype(int)]
        self.cluster_locations = np.array(['nan']*len(self.cluster_channels), dtype='<U100')
        self.cluster_locations[~np.isnan(self.cluster_channels)] = self.channel_locations[self.cluster_channels[~np.isnan(self.cluster_channels)].astype(int)]
        metrics = pd.read_csv(path+'/clusters.metrics.csv')
        self.cluster_metrics = [None] * (metrics['cluster_id'].max()+1)
        for i in metrics.cluster_id.to_numpy():
            self.cluster_metrics[i] = metrics.loc[metrics['cluster_id']==i, 'group'].to_list()[0]
        self.cluster_selection = np.load(path+'/clusters_selection.npy')
        self.cluster_goodmuaselection = np.load(path+'/clusters_goodmua_selection.npy')
        try:
            self.cluster_id = pd.read_csv(path+'/clusters.metrics.csv')['cluster_id']
        except:
            self.cluster_id = pd.read_csv(path+'/clusters.metrics.csv')['id']           
        try:
            groups = pd.read_csv(SIMPLIFIED_REGIONS)
            groups = groups.iloc[:,1:3]
            groups = groups.set_index('original')
            group_dict = groups.to_dict()['group']
            self.cluster_group_locations = \
                pd.Series(self.cluster_locations).map(group_dict)
            self.channel_group_locations = \
                pd.Series(self.channel_locations).map(group_dict)
            self.cluster_com_locations = \
                pd.Series(self.cluster_locations).map(community_map)
            self.channel_com_locations = \
                pd.Series(self.channel_locations).map(community_map)
            self.cluster_custom_locations = \
                pd.Series(self.cluster_locations).map(custom_map)
            self.channel_custom_locations = \
                pd.Series(self.channel_locations).map(custom_map)
            self.cluster_level_locations = \
                pd.Series(self.cluster_locations).map(level_map)
            self.channel_level_locations = \
                pd.Series(self.channel_locations).map(level_map)
        except:
            print("Simplified locations not available")

class alf_ephys:
    def __init__(self, n_probes):
        self.probes = [None]*n_probes
    def __setitem__(self, probe_number, data):
        self.probes[probe_number] = data
    def __getitem__(self, probe_number):
        return self.probes[probe_number]

class alf:
    def __init__(self,path, ephys=False):
        try:
            self.choice  = -1*(np.load(path+'/alf/_ibl_trials.choice.npy'))
            self.outcome  = (np.load(path+'/alf/_ibl_trials.feedbackType.npy')>0)*1
            self.left_reward  = np.load(path+'/alf/_ibl_trials.left_reward.npy')
            self.right_reward  = np.load(path+'/alf/_ibl_trials.right_reward.npy')
            self.probabilityLeft  = np.load(path+'/alf/_ibl_trials.probabilityLeft.npy')
            self.no_reward_block = np.array([0])
            if Path(path+'/alf/QFS_median_QRlaser.npy').is_file()==True:
                self.opto_block = np.load(path+'/alf/_ibl_trials.opto_block.npy')
            else:
                self.opto_block = np.zeros(len(self.choice))
                print('No reward type blocks')
                self.no_reward_block = np.array([1])
            self.laser_reward  = (1*(np.load(path+'/alf/_ibl_trials.rewardVolume.npy')>0)) * self.opto_block
            self.water_reward = (1*(np.load(path+'/alf/_ibl_trials.rewardVolume.npy')>0)) \
                                * (1*(self.opto_block==0))
            self.goCue_trigger_times = np.load(path+'/alf/_ibl_trials.goCue_times.npy')
            self.stimOn_times = np.load(path+'/alf/_ibl_trials.stimOn_times.npy')
            self.response_times = np.load(path+'/alf/_ibl_trials.response_times.npy')
            self.feedback_times = np.load(path+'/alf/_ibl_trials.feedback_times.npy')
            self.start_time = np.load(path+'/alf/_ibl_trials.intervals.npy')[:,0]
            self.end_time = np.load(path+'/alf/_ibl_trials.intervals.npy')[:,1]
            self.first_move = np.load(path+'/alf/_ibl_trials.firstMovement_times.npy')
            try:
                self.firstlaser_times = np.load(path+'/alf/_ibl_trials.first_laser_times.npy')
                self.feedback_consumed = np.load(path+'/alf/_ibl_trials.feedbackTypeConsumed.npy')
            except:
                self.firstlaser_times = np.array([np.nan])
            if os.path.isfile(path+'/alf/standard_QL.npy'):
                self.QL = np.load(path+'/alf/standard_QL.npy')
                self.QR = np.load(path+'/alf/standard_QR.npy')
                if self.no_reward_block==0:
                    self.QLlaser = np.load(path+'/alf/standard_QLlaser.npy')
                    self.QRlaser = np.load(path+'/alf/standard_QRlaser.npy')
                self.QLstay  = np.load(path+'/alf/standard_QLstay.npy')
                self.QRstay  = np.load(path+'/alf/standard_QRstay.npy')
                self.QLreward  = np.load(path+'/alf/standard_QLreward.npy')
                self.QRreward  = np.load(path+'/alf/standard_QRreward.npy')
                self.choice_prediction  = np.load(path+'/alf/standard_choice_prediction.npy')
                self.accuracy =[np.mean((1*(self.choice>0))==(1*(self.choice_prediction>0.5)))]
            
            if os.path.isfile(path+'/alf/QFS_median_QL.npy'):
                self.fQL = np.load(path+'/alf/QFS_median_QL.npy')
                self.fQR = np.load(path+'/alf/QFS_median_QR.npy')
                self.fV = np.load(path+'/alf/QFS_median_V.npy')
                self.seq_ac_propensity = np.load(path+'/alf/SeqAC_noawin_m_avchoice_median_pi_choice.npy')
                self.seq_ac_vchoice = np.load(path+'/alf/SeqAC_noawin_m_avchoice_median_V_choice.npy')
                self.seq_ac_pi_lick = np.load(path+'/alf/SeqAC_noawin_m_avchoice_median_pi_lick.npy')
                self.seq_ac_vcs = np.load(path+'/alf/SeqAC_noawin_m_avchoice_median_V_cs.npy')
                self.seq_ac_choice_prediction = np.load(path+'/alf/SeqAC_noawin_m_avchoice_median_choice_prediction.npy')
                self.seq_ac_lick_prediction = np.load(path+'/alf/SeqAC_noawin_m_avchoice_median_lick_prediction.npy')
                if self.no_reward_block==0:
                    self.fQLlaser = np.load(path+'/alf/QFS_median_QLlaser.npy')
                    self.fQRlaser = np.load(path+'/alf/QFS_median_QRlaser.npy')
                self.fQLstay  = np.load(path+'/alf/QFS_median_QLstay.npy')
                self.fQRstay  = np.load(path+'/alf/QFS_median_QRstay.npy')
                self.fQLreward  = np.load(path+'/alf/QFS_median_QLreward.npy')
                self.fQRreward  = np.load(path+'/alf/QFS_median_QRreward.npy')
                self.fchoice_prediction  = np.load(path+'/alf/QFS_median_choice_prediction.npy')
                self.V =  np.load(path+'/alf/QFS_median_V.npy') # redundtant, it is the same as fV
                p_right = np.where(self.choice==-1, 1-self.fchoice_prediction, self.fchoice_prediction)
                self.wEQ = p_right*self.fQR + (1-p_right)*self.fQL
                self.unEQ = self.fQR + self.fQL
                self.faccuracy = [np.mean((1*(self.choice>0))==(1*(self.fchoice_prediction>0.5)))]
            if os.path.isfile(path+'/alf/REINFORCE_median_reward.npy'):
                self.propensity = np.load(path+'/alf/REINFORCE_median_reward.npy')
                self.reinforce_choice_prediction  = np.load(path+'/alf/REINFORCE_median_choice_prediction.npy')
                self.raccuracy = [np.mean((1*(self.choice>0))==(1*(self.reinforce_choice_prediction>0.5)))]

            if os.path.isfile(path+'/alf/ideal_predicted_state.npy'):
                self.ideal_pstate_rl = np.load(path+'/alf/ideal_predicted_state.npy')[1,:]
                self.ideal_preward_rl  = np.load(path+'/alf/ideal_predicted_reward.npy')[1,:]
                self.ideal_pstate_lr = np.load(path+'/alf/ideal_predicted_state.npy')[0,:]
                self.ideal_preward_lr  = np.load(path+'/alf/ideal_predicted_reward.npy')[0,:]
                self.ideal_accuracy = [np.mean((1*(self.choice>0))==(1*(self.ideal_pstate_rl>0.5)))]
        except:
                print('No task data, passive stim session?')
        if ephys==True: # Clunky section to be streamlined
            probe_paths = sorted(glob(path +'/alf/*[0-9]*/'))
            self.probe = alf_ephys(len(probe_paths))
            for p, p_path in enumerate(probe_paths):
                if Path(p_path+'/pykilosort').is_dir():
                    p_path=p_path+'/pykilosort'
                self.probe[p] = probe(p_path)

    def to_df(self):
        dl = len(self.choice)
        ses_dict = self.__dict__
        include_var = []
        for i in np.arange(len(ses_dict)):
            if len(list(ses_dict.values())[i])==dl:
                include_var.append(list(ses_dict)[i])
        df = pd.DataFrame()
        for var in include_var:
            df[var] = ses_dict[var]     
        return df
    
    def fr_bytrial(self, cluster_id, aligning_var,
                        probe_name='probe00', window=-1):
        prb = getattr(self,probe_name)
        bytrial = average_hz_trial(cluster_id, prb.spike_clusters, prb.spike_times, aligning_var,
                         window=window)
        return bytrial

    def plot_session(self):
        example = self.to_df()
        example['choice_r'] = (example['choice']==1)*1
        example['choice_l'] = (example['choice']==-1)*1
        if self.no_reward_block==0: 
            example['value_laser'] = example['fQRlaser']-example['fQLlaser'] 
        example['value_reward'] = example['fQRreward']-example['fQLreward']
        example['value_stay'] = example['fQRstay']-example['fQLstay']
        example['probabilityRight']=0.1
        example.loc[example['probabilityLeft']==0.1, 'probabilityRight'] = 0.7
        example['reward_r'] = example['outcome']*example['choice_r']*(1*(example['opto_block']!=1))
        example['reward_l'] = example['outcome']*example['choice_l']*(1*(example['opto_block']!=1))
        if self.no_reward_block==0:
            example['laser_r'] = example['outcome']*example['choice_r']*example['opto_block']
            example['laser_l'] = example['outcome']*example['choice_l']*example['opto_block']
        fig = plt.figure()
        spec = gridspec.GridSpec(ncols=1, nrows=2,
                                 height_ratios=[2, 5])
        spec.update(wspace=0.025, hspace=0.05)
        ax1 = fig.add_subplot(spec[1])
        ax1.plot(example['choice_r'].rolling(10, center=False).mean(),color='k')
        ax1.plot(example['fchoice_prediction'].rolling(10,center=False).mean(),color='k', linestyle='dashed', linewidth=2)
        ax1.spines['top'].set_visible(False)
        ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis
        if self.no_reward_block==0:
            ax2.plot(example['value_laser'],color='orange', linestyle='dashed', linewidth=2)
        ax2.plot(example['value_reward'],color='green', linestyle='dashed', linewidth=2)
        ax2.spines['top'].set_visible(False)
        ax2.plot(example['value_stay'],color='gray', linestyle='dashed', linewidth=2)
        # Only show ticks on the left and bottom spines
        ax1.set_ylim(-0.1,1.1)
        ax1.set_ylabel('Fraction of right choices')
        ax2.set_ylabel('QR-QL')
        ax3 = fig.add_subplot(spec[0], sharex=ax1)
        ax3.plot(example['probabilityRight'],color='k',
                 linestyle='--', alpha =0.5)
        #plt.xlim(0,400)
        plt.sca(ax3)
        plt.vlines(np.where(example['choice']==1),0.925,1.0, color='k')
        plt.vlines(np.where(example['choice']==-1),-0.225,-0.15, color='k')
        plt.vlines(np.where(example['reward_l']==1),-0.225,-0.15, color='green')
        plt.vlines(np.where(example['reward_r']==1),0.925,1.0, color='green')
        if self.no_reward_block==0:
            plt.vlines(np.where(example['laser_r']==1),0.925,1.0, color='orange')
            plt.vlines(np.where(example['laser_l']==1),-0.225,-0.15, color='orange')
        plt.axis('off')
        plt.ylabel('Reward probability')
        return fig

    def plot_session_REINFORCE(self):
        example = self.to_df()
        example['choice_r'] = (example['choice']==1)*1
        example['choice_l'] = (example['choice']==-1)*1
        example['value_laser'] = example['DQlaser']
        example['value_reward'] = example['DQwater']
        example['value_stay'] = example['RQRstay']-example['RQLstay']
        example['probabilityRight']=0.1
        example.loc[example['probabilityLeft']==0.1, 'probabilityRight'] = 0.7
        example['reward_r'] = example['outcome']*example['choice_r']*(1*(example['opto_block']!=1))
        example['reward_l'] = example['outcome']*example['choice_l']*(1*(example['opto_block']!=1))
        example['laser_r'] = example['outcome']*example['choice_r']*example['opto_block']
        example['laser_l'] = example['outcome']*example['choice_l']*example['opto_block']
        fig = plt.figure()
        spec = gridspec.GridSpec(ncols=1, nrows=2,
                                 height_ratios=[2, 5])
        spec.update(wspace=0.025, hspace=0.05)
        ax1 = fig.add_subplot(spec[1])
        ax1.plot(example['choice_r'].rolling(10, center=False).mean(),color='k')
        ax1.plot(example['reinforce_choice_prediction'].rolling(10, center=False).mean(),color='k', linestyle='dashed', linewidth=2)
        ax1.spines['top'].set_visible(False)
        ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis
        ax2.plot(example['value_laser'].rolling(10, center=False).mean(),color='orange', linestyle='dashed', linewidth=2)
        ax2.plot(example['value_reward'].rolling(10, center=False).mean(),color='dodgerblue', linestyle='dashed', linewidth=2)
        ax2.spines['top'].set_visible(False)
        ax2.plot(example['value_stay'].rolling(10, center=False).mean(),color='gray', linestyle='dashed', linewidth=2)
        # Only show ticks on the left and bottom spines
        ax1.set_ylim(-0.1,1.1)
        ax1.set_ylabel('Fraction of right choices')
        ax2.set_ylabel('DQ')
        ax3 = fig.add_subplot(spec[0], sharex=ax1)
        ax3.plot(example['probabilityRight'],color='k',
                 linestyle='--', alpha =0.5)
        #plt.xlim(0,400)
        plt.sca(ax3)
        plt.vlines(np.where(example['choice']==1),0.925,1.0, color='k')
        plt.vlines(np.where(example['choice']==-1),-0.225,-0.15, color='k')
        plt.vlines(np.where(example['reward_l']==1),-0.225,-0.15, color='dodgerblue')
        plt.vlines(np.where(example['reward_r']==1),0.925,1.0, color='dodgerblue')
        plt.vlines(np.where(example['laser_r']==1),0.925,1.0, color='orange')
        plt.vlines(np.where(example['laser_l']==1),-0.225,-0.15, color='orange')
        plt.axis('off')
        plt.ylabel('Reward probability')
        return fig

    def plot_correlations(self):
        QR = ses.QRreward
        QRl = ses.QRlaser
        QRs = ses.QRstay
        QL = ses.QLreward
        QLl = ses.QLlaser
        QLs = ses.QLstay
        QRQL = QR-QL
        QRlQLl = QRl - QLl
        QRsQLs = QRs - QLs
        fig, ax  = plt.subplots(4,3)
        plt.sca(ax[0,0])
        plt.scatter(QR, QRl)
        plt.xlabel('QR reward')
        plt.ylabel('QR Laser')
        plt.sca(ax[0,1])
        plt.scatter(QR, QRs)
        plt.xlabel('QR reward')
        plt.ylabel('QR stay')
        plt.sca(ax[0,2])
        plt.scatter(QRl, QRs)
        plt.xlabel('QR Laser')
        plt.ylabel('QR stay')
        plt.sca(ax[1,0])
        plt.scatter(QL, QLl)
        plt.xlabel('QL reward')
        plt.ylabel('QL Laser')
        plt.sca(ax[1,1])
        plt.scatter(QL, QLs)
        plt.xlabel('QL reward')
        plt.ylabel('QL stay')
        plt.sca(ax[1,2])
        plt.scatter(QLl, QLs)
        plt.xlabel('QL Laser')
        plt.ylabel('QL stay')
        plt.sca(ax[2,0])
        plt.scatter(QRQL, QRlQLl)
        plt.xlabel('QR-QL reward')
        plt.ylabel('QR-QL Laser')
        plt.sca(ax[2,1])
        plt.scatter(QRQL, QRsQLs)
        plt.xlabel('QR-QL reward')
        plt.ylabel('QR-QL stay')
        plt.sca(ax[2,2])
        plt.scatter(QRlQLl, QRsQLs)
        plt.xlabel('QR-QL Laser')
        plt.ylabel('QR-QL stay')
        plt.sca(ax[3,0])
        plt.scatter(np.arange(len(QR)), QR+QL)
        plt.ylabel('QR+QL reward')
        plt.xlabel('Trial')
        plt.sca(ax[3,1])
        plt.scatter(np.arange(len(QR)), QRl+QLl)
        plt.ylabel('QR+QL laser')
        plt.xlabel('Trial')
        plt.sca(ax[3,2])
        plt.scatter(np.arange(len(QR)), QR+QL+QRl+QLl+QRs+QLs)
        plt.ylabel('QR+QL')
        plt.xlabel('Trial')
        plt.tight_layout()
        sns.despine()
