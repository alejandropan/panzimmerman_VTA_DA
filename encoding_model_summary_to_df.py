"""
Readers for the MATLAB encoding-model output: the per-neuron residual files
the decoders are fitted to, and the model summary table.

CURATED BY DELETION ONLY. Nothing that ran has been rewritten. Removed:

    load_tuning, load_all_tuning   pretest tuning files; no caller in the
                                   release.
    common_trials_old              superseded by `common_trials`.
    summary_firing_rate            a plotting helper that could never have
                                   run - it calls `plt` and `sns`, neither
                                   of which this module imports.
    the 'yreal_crac' filetype      provably dead: its branch assigns only
                                   `outcome_data_crac`, and the block below
                                   asserts `gocue_data_crac` as well, so it
                                   raises NameError on the first trial. This
                                   is why `pi_choice` ran on raw spikes
                                   rather than CRAC residuals.
    ~29 commented-out column
    assignments in
    load_encoding_model            alternative kernel sets, left behind as
                                   the model changed.
"""

from scipy.io import loadmat
import pandas as pd
import mat73
import glob
import numpy as np
from pathlib import Path
from decoders_yield_availability import *

HERE = (Path(__file__).resolve().parent if "__file__" in globals()
        else Path.cwd())
SIMPLIFIED_REGIONS = (HERE / "decoders_data" / "histology_files"
                      / "simplified_regions.csv")


def load_encoding_model(model_path):
    model_dict = loadmat(model_path)
    encoding_model = pd.DataFrame()
    mouse=[]
    date=[]
    ses=[]
    id=[]
    probe=[]
    [mouse.append(i[0][:6]) for i in model_dict['EncodingSummary']['recording_name'][0][0][0]]
    [date.append(i[0][7:17]) for i in model_dict['EncodingSummary']['recording_name'][0][0][0]]
    [ses.append(i[0][18:21]) for i in model_dict['EncodingSummary']['recording_name'][0][0][0]]
    [id.append(i[0]) for i in model_dict['EncodingSummary']['recording_name'][0][0][0]]
    [probe.append(int(i[0][6])) for i in model_dict['EncodingSummary']['probe'][0][0][0]]
    encoding_model['overall_policy_or_value'] = model_dict['EncodingSummary']['pvalues'][0][0]['AllKernels'][0][0]['remove_AllValues'][0][0][0]
    encoding_model['overall_policy'] = model_dict['EncodingSummary']['pvalues'][0][0]['AllKernels'][0][0]['remove_RelativeValue'][0][0][0]
    encoding_model['overall_value'] = model_dict['EncodingSummary']['pvalues'][0][0]['AllKernels'][0][0]['remove_StateValue'][0][0][0]
    encoding_model['dr2_overall_policy_or_value'] = model_dict['EncodingSummary']['delta_r2'][0][0]['AllKernels'][0][0]['remove_AllValues'][0][0][0]
    encoding_model['dr2_overall_policy'] = model_dict['EncodingSummary']['delta_r2'][0][0]['AllKernels'][0][0]['remove_RelativeValue'][0][0][0]
    encoding_model['dr2_overall_value'] = model_dict['EncodingSummary']['delta_r2'][0][0]['AllKernels'][0][0]['remove_StateValue'][0][0][0]    
    alphas = model_dict['EncodingSummary']['varyalpha'][0][0]['alpha'][0][0]['R2'][0][0]['alphas'][0][0]
    deltar2 = model_dict['EncodingSummary']['varyalpha'][0][0]['alpha'][0][0]['R2'][0][0]['AllValues'][0][0] - \
            model_dict['EncodingSummary']['varyalpha'][0][0]['alpha'][0][0]['R2'][0][0]['remove_AllValues_AllKernels'][0][0]   
    encoding_model['best_value_alpha'] = np.concatenate(alphas[np.argmax(deltar2, axis=1)])
    encoding_model['cluster_id'] = model_dict['EncodingSummary']['cluster'][0][0][0].astype(int)
    encoding_model['laser_cue_gain'] = model_dict['EncodingSummary']['gains'][0][0]['cue'][0][0][:,5]
    encoding_model['laser_contra_gain'] = model_dict['EncodingSummary']['gains'][0][0]['movement_contra'][0][0][:,5]
    encoding_model['laser_ipsi_gain'] = model_dict['EncodingSummary']['gains'][0][0]['movement_ipsi'][0][0][:,5]
    encoding_model['outcome_laser_gain'] = model_dict['EncodingSummary']['gains'][0][0]['outcome_water'][0][0][:,5]
    encoding_model['dlaser_cue_gain'] = model_dict['EncodingSummary']['gains'][0][0]['cue'][0][0][:,2]
    encoding_model['dlaser_contra_gain'] = model_dict['EncodingSummary']['gains'][0][0]['movement_contra'][0][0][:,2]
    encoding_model['dlaser_ipsi_gain'] = model_dict['EncodingSummary']['gains'][0][0]['movement_ipsi'][0][0][:,2]
    encoding_model['doutcome_laser_gain'] = model_dict['EncodingSummary']['gains'][0][0]['outcome_water'][0][0][:,2]
    encoding_model['firing_rate'] =  model_dict['EncodingSummary']['firing_rate'][0][0][0]
    encoding_model['region_raw'] = model_dict['EncodingSummary']['region_name'][0][0][0]
    encoding_model['region'] =  model_dict['EncodingSummary']['region_key'][0][0][0][model_dict['EncodingSummary']['region_id'][0][0][0]-1]
    encoding_model['mouse'] = mouse
    encoding_model['date'] = date
    encoding_model['ses'] = ses
    encoding_model['probe'] = probe
    encoding_model['id'] = id
    return encoding_model

def resample(psth_array,final_bins, model_bin_size):
    resampling_factor = final_bins/model_bin_size
    new_n_col = int(psth_array.shape[1]/resampling_factor)
    new_psth_array = np.zeros([psth_array.shape[0], new_n_col])
    for j in np.arange(new_n_col):
        new_psth_array[:,j] = np.mean(psth_array[:,int(resampling_factor*j):int(resampling_factor*(j+1))], 
                                        axis = 1)
    return new_psth_array

def load_residual(neuron_file, model_bin_size=10, final_bins=100, pre_time = -500, post_time = 500, 
                 post_time_outcome=2000, filetype='residual', criterion='good'): # bin sizes in ms
    # Alternative 20 / -500 /3000 /3000
    if criterion == 'good':
        acceptable_labels = 1
    elif criterion == 'mua':
        acceptable_labels = 2
    else:
        acceptable_labels = [1,2]
    residual_struct = mat73.loadmat(neuron_file)
    if ~np.isin(residual_struct['data']['cluster']['label'], acceptable_labels):
        return None
    # Start_dataframe
    neuron = pd.DataFrame()
    n_trials = int(len(residual_struct['data']['Y_pred']))
    # General info
    try:
        neuron['lambd'] = [residual_struct['data']['lambda'].tolist()]
    except:
        neuron['lambd'] = np.nan    
    neuron['trials_included'] = [residual_struct['data']['trials'] - 1]
    neuron['cluster_id_org'] =  int(residual_struct['data']['cluster']['clusterid'])
    neuron['probe_id'] = int(residual_struct['data']['cluster']['probe'][6])
    neuron['cluster_id'] =  (neuron['probe_id']*10000)+ neuron['cluster_id_org']
    neuron['animal'] =  residual_struct['data']['cluster']['session'][:6]
    neuron['date'] = residual_struct['data']['cluster']['session'][7:17]
    neuron['session'] = residual_struct['data']['cluster']['session'][18:21]
    neuron['area'] = residual_struct['data']['cluster']['location']
    neuron['hem'] = residual_struct['data']['cluster']['hem']
    # Make matrix
    pre_window = int(abs(pre_time/model_bin_size))
    post_window = int(abs(post_time/model_bin_size))
    post_window_outcome = int(abs(post_time_outcome/model_bin_size))
    residuals_goCue = np.zeros([n_trials, pre_window+post_window])
    residuals_choice =  np.zeros([n_trials, pre_window+post_window])
    residuals_outcome = np.zeros([n_trials, pre_window+post_window_outcome])
    residuals_goCue_crac = np.zeros([n_trials, 100])
    residuals_choice_crac = np.zeros([n_trials, 100])
    residuals_outcome_crac = np.zeros([n_trials, 200])
    if (filetype=='residual_crac_vcs') | (filetype=='residual_crac_vchoice') | (filetype=='residual_crac_pichoice'): # Load output from encoding CRAC model
        base_path = Path(neuron_file).parents[2] 
        neuron_id = Path(neuron_file).stem.split('_residuals')[0]
        crac_file = base_path / 'outputs' / 'CRAC_median' / neuron_id / f"{neuron_id}_session_000.mat"
        crac_file = str(crac_file)
        try:
            crac_model = loadmat(crac_file)
        except:
            print('No residual available for ' + neuron_id + ' in ' + neuron['area'].tolist()[0])
            return None
    for i in range(n_trials):
        go = int(residual_struct['data']['times'][i][0] - 1) # -1 to account for matlab indexing
        choice =  int(residual_struct['data']['times'][i][1] - 1) # -1 to account for matlab indexing
        outcome = int(residual_struct['data']['times'][i][2] - 1) # -1 to account for matlab indexing
        if filetype=='residual_crac_vcs':
            mapping = crac_model['output']['remove_Vcs_AllKernels'][0][0]['fit2'][0][0]['stats'][0][0]['bins_include'][0][0][0][i][0] - 1 
            go_idx = np.where(mapping==go)[0][0]
            epoch_idxs = mapping[go_idx:go_idx+100]
            assert(all(np.diff(epoch_idxs)==1))
            gocue_data_crac = crac_model['output']['remove_Vcs_AllKernels'][0][0]['fit2'][0][0]['stats'][0][0]['Y_resd'][0][0][0][i][0][go_idx:go_idx+100]
            choice_idx = np.where(mapping==choice)[0][0]
            epoch_idxs = mapping[choice_idx-50:choice_idx+50]
            assert(all(np.diff(epoch_idxs)==1))
            choice_data_crac = crac_model['output']['remove_Vcs_AllKernels'][0][0]['fit2'][0][0]['stats'][0][0]['Y_resd'][0][0][0][i][0][choice_idx:choice_idx+100]
            outcome_idx = np.where(mapping==outcome)[0][0]
            epoch_idxs = mapping[outcome_idx:outcome_idx+200]
            assert(all(np.diff(epoch_idxs)==1))
            outcome_data_crac = crac_model['output']['remove_Vcs_AllKernels'][0][0]['fit2'][0][0]['stats'][0][0]['Y_resd'][0][0][0][i][0][outcome_idx:outcome_idx+200]
        if filetype=='residual_crac_vchoice':
            mapping = crac_model['output']['remove_Vchoice_AllKernels'][0][0]['fit2'][0][0]['stats'][0][0]['bins_include'][0][0][0][i][0] - 1 
            go_idx = np.where(mapping==go)[0][0]
            epoch_idxs = mapping[go_idx:go_idx+100]
            assert(all(np.diff(epoch_idxs)==1))
            gocue_data_crac = crac_model['output']['remove_Vchoice_AllKernels'][0][0]['fit2'][0][0]['stats'][0][0]['Y_resd'][0][0][0][i][0][go_idx:go_idx+100]
            choice_idx = np.where(mapping==choice)[0][0]
            epoch_idxs = mapping[choice_idx-50:choice_idx+50]
            assert(all(np.diff(epoch_idxs)==1))
            choice_data_crac = crac_model['output']['remove_Vchoice_AllKernels'][0][0]['fit2'][0][0]['stats'][0][0]['Y_resd'][0][0][0][i][0][choice_idx:choice_idx+100]
            outcome_idx = np.where(mapping==outcome)[0][0]
            epoch_idxs = mapping[outcome_idx:outcome_idx+200]
            assert(all(np.diff(epoch_idxs)==1))
            outcome_data_crac = crac_model['output']['remove_Vchoice_AllKernels'][0][0]['fit2'][0][0]['stats'][0][0]['Y_resd'][0][0][0][i][0][outcome_idx:outcome_idx+200]
        if filetype=='residual_crac_pichoice':
            mapping = crac_model['output']['remove_PIchoice_AllKernels'][0][0]['fit2'][0][0]['stats'][0][0]['bins_include'][0][0][0][i][0] - 1 
            go_idx = np.where(mapping==go)[0][0]
            epoch_idxs = mapping[go_idx:go_idx+100]
            assert(all(np.diff(epoch_idxs)==1))
            gocue_data_crac = crac_model['output']['remove_PIchoice_AllKernels'][0][0]['fit2'][0][0]['stats'][0][0]['Y_resd'][0][0][0][i][0][go_idx:go_idx+100]
            choice_idx = np.where(mapping==choice)[0][0]
            epoch_idxs = mapping[choice_idx-50:choice_idx+50]
            assert(all(np.diff(epoch_idxs)==1))
            choice_data_crac = crac_model['output']['remove_PIchoice_AllKernels'][0][0]['fit2'][0][0]['stats'][0][0]['Y_resd'][0][0][0][i][0][choice_idx:choice_idx+100]
            outcome_idx = np.where(mapping==outcome)[0][0]
            epoch_idxs = mapping[outcome_idx:outcome_idx+200]
            assert(all(np.diff(epoch_idxs)==1))
            outcome_data_crac = crac_model['output']['remove_PIchoice_AllKernels'][0][0]['fit2'][0][0]['stats'][0][0]['Y_resd'][0][0][0][i][0][outcome_idx:outcome_idx+200]
        if filetype=='residual':
            go_data = residual_struct['data']['Y_resd'][i][go-pre_window:go+post_window] #
            choice_data = residual_struct['data']['Y_resd'][i][choice-pre_window:choice+post_window] #
            outcome_data = residual_struct['data']['Y_resd'][i][outcome-pre_window:outcome+post_window_outcome] #
        if filetype=='real':
            go_data = residual_struct['data']['Y_real'][i][go-pre_window:go+post_window] #
            choice_data = residual_struct['data']['Y_real'][i][choice-pre_window:choice+post_window] #
            outcome_data = residual_struct['data']['Y_real'][i][outcome-pre_window:outcome+post_window_outcome] # 
        if filetype=='prediction':
            go_data = residual_struct['data']['Y_pred'][i][go-pre_window:go+post_window] #
            choice_data = residual_struct['data']['Y_pred'][i][choice-pre_window:choice+post_window] #
            outcome_data = residual_struct['data']['Y_pred'][i][outcome-pre_window:outcome+post_window_outcome] #

        if (filetype=='residual_crac_vcs') | (filetype=='residual_crac_vchoice') | (filetype=='residual_crac_pichoice'): 
            assert len(outcome_data_crac) == len(residuals_outcome_crac[i,:])
            residuals_outcome_crac[i,:] = outcome_data_crac
            assert len(gocue_data_crac) == len(residuals_goCue_crac[i,:])
            residuals_goCue_crac[i,:] = gocue_data_crac
            assert len(choice_data_crac) == len(residuals_choice_crac[i,:])
            residuals_choice_crac[i,:] = choice_data_crac

        else:    
            assert len(go_data) == len(residuals_goCue[i,:])
            assert len(choice_data) == len(residuals_choice[i,:])
            assert len(outcome_data) == len(residuals_outcome[i,:])
            residuals_goCue[i,:] = go_data
            residuals_choice[i,:] = choice_data
            residuals_outcome[i,:] = outcome_data
    # Resample
    if (filetype=='residual_crac_vcs') | (filetype=='residual_crac_vchoice') | (filetype=='residual_crac_pichoice'): 
        neuron['residuals_outcome'] = [resample(residuals_outcome_crac,final_bins, model_bin_size)]
        neuron['residuals_goCue'] = [resample(residuals_goCue_crac,final_bins, model_bin_size)]
        neuron['residuals_choice'] =  [resample(residuals_choice_crac,final_bins, model_bin_size)]
    else:
        neuron['residuals_goCue'] = [resample(residuals_goCue,final_bins, model_bin_size)]
        neuron['residuals_choice'] =  [resample(residuals_choice,final_bins, model_bin_size)]
        neuron['residuals_outcome'] = [resample(residuals_outcome,final_bins, model_bin_size)]
    return neuron

def homogenize_neural_data(residuals, trials_included):
    reduced_residuals = pd.DataFrame()
    for i in np.arange(len(residuals)):
        new_neuron = pd.DataFrame()
        neuron = residuals.iloc[i].copy()
        assert np.array_equal(neuron['trials_included'], sorted(neuron['trials_included']))
        trials_idx = np.searchsorted(neuron['trials_included'], trials_included)
        new_neuron['cluster_id'] = [neuron['cluster_id']]
        new_neuron['animal'] = [neuron['animal']]
        new_neuron['date'] = [neuron['date']]
        new_neuron['session'] = [neuron['session']]
        new_neuron['area'] = [neuron['area']]
        new_neuron['hem'] = [neuron['hem']]
        new_neuron['location'] = [neuron['location']]
        new_neuron['trials_included'] = [neuron['trials_included'][trials_idx].copy()]
        new_neuron['residuals_goCue'] = [neuron['residuals_goCue'][trials_idx].copy()]
        new_neuron['residuals_choice'] = [neuron['residuals_choice'][trials_idx].copy()]
        new_neuron['residuals_outcome'] = [neuron['residuals_outcome'][trials_idx].copy()]
        assert new_neuron['residuals_goCue'][0].shape[0]==len(trials_included)
        reduced_residuals = pd.concat([reduced_residuals,new_neuron])
    return reduced_residuals

def load_all_residuals(root_path, filetype='residual', criterion='good',fbins=100):
    path_to_all_residuals = glob.glob(root_path+'/*_residuals.mat')
    residuals = pd.DataFrame()
    for p in path_to_all_residuals:
        residuals = pd.concat([residuals, load_residual(p, filetype=filetype, criterion=criterion, final_bins=fbins)])
    groups = pd.read_csv(SIMPLIFIED_REGIONS)
    groups = groups.iloc[:,1:3]
    groups = groups.set_index('original')
    group_dict = groups.to_dict()['group']
    residuals['location'] = pd.Series(residuals.area).map(group_dict)
    residuals['location_level'] = pd.Series(residuals.area).map(level_map)
    residuals['location_community'] = pd.Series(residuals.area).map(community_map)
    residuals['location_custom'] = pd.Series(residuals.area).map(custom_map)
    return residuals

def common_neural_data(residuals,n_trials_minimum=100):
    root =  '/jukebox/witten/Alex/Data/Subjects/'
    residuals['include'] = 0
    for probe in residuals.probe_id.unique().astype(str):
        goodmua = np.load(root + residuals.animal.unique()[0] + '/' + residuals.date.unique()[0] \
        +  '/' + residuals.session.unique()[0] + '/alf/probe0' + probe + '/pykilosort/' \
        + 'clusters_goodmua_selection.npy')
        residuals.loc[(residuals.probe_id==int(probe))&
                (np.isin(residuals.cluster_id_org,goodmua)), 'include'] = 1
    residuals = residuals.loc[residuals.include==1].reset_index()
    included = []
    for i in np.arange(len(residuals)):
        if len(residuals.iloc[i]['trials_included'])>=n_trials_minimum:
             included.append(i)
    return residuals.iloc[included,:]

def common_trials(neural_data, subsample):
    hem_neural_data = neural_data.iloc[subsample].copy()
    for i in np.arange(len(hem_neural_data)-1):    
        if i == 0:
            trials_included = hem_neural_data['trials_included'].iloc[i]
        t = hem_neural_data['trials_included'].iloc[i+1]
        trials_included = np.intersect1d(trials_included,t)
    reduced_residuals = homogenize_neural_data(hem_neural_data, trials_included)
    return reduced_residuals, trials_included
