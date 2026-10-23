"""
The decoder itself. Six functions, called from `run_decoder_job.py`.

CURATED BY DELETION ONLY. The decoders will not be re-run, so no line that
executed on the path to the `.npy` files on disk has been rewritten. What
was removed was either never called at all, or sits behind a parameter value
the release never passes:

    etype        only 'real' and 'null' are ever passed, so the 'time' and
                 'lags' branches are gone, and with them `run_decoder_time`,
                 `weighted_decoder_time` and `summary_score_time`.
    decoder      always Lasso, so the LogisticRegression and ElasticNet
                 paths are gone, and with them `summary_score_f1`.
    weights      always None; every decoder ran unweighted. The parameter is
                 kept on the entry point, whose signature is pinned by the
                 equivalence test, and dropped below it.
    binevolving  never True: the regressed variable is one value per trial,
                 never one per trial and bin. The branch is gone, and with it
                 the 2-D case of `shuffled_Xval_stratified`.

Also removed, having no callers at all: `z_score_peth`,
`get_session_sample_weights`, `get_sample_weighting`, `shuffled_Xval`
(only the stratified version is used), `makeXvalpartitions` and
`summary_score_log`.

**The results are not reproducible.** The neuron subsample below uses
`np.random.choice` with no seed, and none is set anywhere. Re-running any
job gives different R2 values. The summary CSVs, not these functions, are
the record of what was published.
"""

import sys
sys.path.append('/jukebox/witten/Alex/PYTHON/rewardworld/ephys/analysis')
from sklearn.linear_model import Lasso as LR
from sklearn.preprocessing import KBinsDiscretizer
from sklearn.model_selection import StratifiedKFold
from scipy.stats import pearsonr
import numpy as np
from sklearn.metrics import r2_score
from encoding_model_summary_to_df import common_trials

def reshape_psth_array(binned_spikes):
    # n_trials x n_neurons x n_bins
    spike_data = np.zeros([binned_spikes.iloc[0].shape[0],len(binned_spikes),binned_spikes.iloc[0].shape[1]])
    for n in np.arange(spike_data.shape[1]):
        assert spike_data[:,n,:].shape ==  binned_spikes.iloc[n].shape
        spike_data[:,n,:] = binned_spikes.iloc[n]
    return spike_data

def shuffled_Xval_stratified(data, k_folds):
    if len(np.unique(data)) == 2: # If variable is binary
        n_trials = len(data)
        x = np.arange(n_trials)
        # Initialize StratifiedKFold
        # Using a fixed random_state for reproducibility
        k = StratifiedKFold(n_splits=k_folds, shuffle=True, random_state=n_trials)
        outer_training=[]
        outer_test=[]
        for otrainingi, otesti in k.split(x, data):
            outer_training.append(otrainingi)
            outer_test.append(otesti)
        outer_folds = [outer_training, outer_test]
        # inner_folds
        all_inner_training=[]
        all_inner_test=[]
        for f in outer_folds[0]: # go over outer training folds
            inner_training=[]
            inner_test=[]
            # We use data[f] to ensure the inner split respects the 10/90 ratio 
            # present within that specific outer training fold
            for itrainingi, itesti in k.split(f, data[f]):
                inner_training.append(f[itrainingi]) 
                inner_test.append(f[itesti])
            all_inner_training.append(inner_training)
            all_inner_test.append(inner_test)        
        inner_folds = [all_inner_training, all_inner_test]
    else:
        nbins = len(np.histogram_bin_edges(data, bins='fd'))
        data_c = data.reshape((len(data),1))
        kbins =  KBinsDiscretizer(n_bins=nbins, encode='ordinal', strategy='uniform')
        data_bins = kbins.fit_transform(data_c)
        n_trials = len(data)
        x = np.arange(n_trials)
        # outer_folds
        k = StratifiedKFold(n_splits=k_folds, shuffle=True, random_state=len(data))
        outer_training=[]
        outer_test=[]
        for otrainingi, otesti in k.split(x,data_bins):
            outer_training.append(otrainingi)
            outer_test.append(otesti)
        outer_folds=[outer_training,outer_test]
        # inner_folds
        all_inner_training=[]
        all_inner_test=[]
        for f in outer_folds[0]: # go over outer training folds
            inner_training=[]
            inner_test=[]
            for itrainingi, itesti in k.split(f,data_bins[f]):
                inner_training.append(f[itrainingi]) # is already an index, we dont want the index of the index hence f[it..i]
                inner_test.append(f[itesti])
            all_inner_training.append(inner_training) # is already an index, we dont want the index of the index hence f[it..i]
            all_inner_test.append(inner_test)        
        inner_folds = [all_inner_training,all_inner_test]
    return outer_folds, inner_folds


def run_decoder(hem_neural_data , alignment_time, regressed_variable, number_folds=5, decoder = LR,
                n_neurons_minimum = 20, lambdas = None, max_n_combinations=100,
                trial_filter=None, null=False):
    '''
    Params:
    xs : This is the binned firing rates for the population of interest
    regressed_variable : the y, one value per trial
    max_n_combinations: Number of combination run for each set of (10,20 etc neurons)

    `weights` and `binevolvingvariable` are gone: sample weighting was never
    used, and the regressed variable was never a (trials x bins) array.
    '''
    if type(n_neurons_minimum)==list:
        if len(n_neurons_minimum)>1:
            n_neurons_combos = np.array(n_neurons_minimum).copy()
            n_neurons_minimum = n_neurons_minimum[0]

    if lambdas is None:
        lambdas = np.array([0.000001,0.00001,0.0001,0.001,0.01,0.1,1,10,100])
    n_neurons_available = len(hem_neural_data) 
    if n_neurons_available < n_neurons_minimum:
        print('Not enough neurons')
        return None
    try:
        neuron_samples_n = n_neurons_combos
    except:
        neuron_samples_n = np.array([n_neurons_minimum])
    neuron_samples_n = neuron_samples_n[np.where(neuron_samples_n<=n_neurons_available)] # Ignore samples with higher n that possible neurons
    neuron_combinations = []
    for samples in neuron_samples_n:
        neuron_selections = []
        for i in np.arange(max_n_combinations):
            neuron_selections.append(np.random.choice(np.arange(n_neurons_available), samples, replace=False))
        neuron_combinations.append(neuron_selections) # neuron_combinations[neuron_samples_n][n_combinations_until_using_every_neurons]
    if alignment_time=='response_time':
        n_bins = hem_neural_data['residuals_outcome'].iloc[0].shape[1]
    if alignment_time=='goCue_time':
        n_bins = hem_neural_data['residuals_goCue'].iloc[0].shape[1]
    if alignment_time=='action_time':
        n_bins = hem_neural_data['residuals_choice'].iloc[0].shape[1]    
    pearson_summary = np.zeros([number_folds,len(lambdas), n_bins, len(neuron_combinations), max_n_combinations])
    mse_summary = np.zeros([number_folds,len(lambdas), n_bins, len(neuron_combinations), max_n_combinations]) 
    pearson_summary[:] = np.nan
    mse_summary[:] = np.nan
    pred_summary = np.zeros([len(regressed_variable),n_bins,len(neuron_combinations),max_n_combinations])
    ytest_summary = np.zeros([len(regressed_variable),n_bins,len(neuron_combinations),max_n_combinations])
    pred_summary[:] = np.nan
    ytest_summary[:] = np.nan
    lambdas_store = []
    # --- coef/intercept accumulators: mean over all outer fits (draws x folds) per neuron ---
    n_combos = len(neuron_combinations)
    coef_sum        = np.zeros([n_neurons_available, n_combos, n_bins])
    coef_count      = np.zeros([n_neurons_available, n_combos, n_bins])
    intercept_sum   = np.zeros([n_combos, n_bins])
    intercept_count = np.zeros([n_combos, n_bins])
    for nc, nsample in enumerate(neuron_combinations): # loop over combos of 10, 20 ... neurons
        for s, subsample in enumerate(nsample):
            reduced_residuals, trials_included = common_trials(hem_neural_data, subsample) #remake function with this new arguments
            if alignment_time=='response_time':
                binned_spikes = reduced_residuals['residuals_outcome']
            if alignment_time=='goCue_time':
                binned_spikes = reduced_residuals['residuals_goCue']
            if alignment_time=='action_time':
                binned_spikes = reduced_residuals['residuals_choice']
            xs_s = reshape_psth_array(binned_spikes) # turn into array
            regressed_variable_s = regressed_variable[trials_included.astype(int)].copy()

            if trial_filter is not None: # In case we want to subdivide by trial tipe
                trials_included_filtered = np.intersect1d(trials_included, trial_filter).astype(int)
                regressed_variable_s = regressed_variable[trials_included_filtered].copy()
                xs_s = xs_s[np.searchsorted(trials_included,trials_included_filtered)]
            folds  = shuffled_Xval_stratified(regressed_variable_s,
                                                number_folds)
            for f in np.arange(number_folds):
                outer_training = folds[0][0][f] 
                outer_test = folds[0][1][f] 
                inner_training = folds[1][0][f]
                inner_test = folds[1][1][f]
                inner_l_performance = np.zeros([len(inner_training),len(lambdas), n_bins])
                inner_l_performance[:] = np.nan
                # Start of inner XVal
                inner_real = np.zeros([len(regressed_variable_s),len(lambdas),n_bins])
                inner_real[:] = np.nan
                inner_pred = np.zeros([len(regressed_variable_s),len(lambdas),n_bins])
                inner_pred[:] = np.nan
                for inner_f in np.arange(number_folds):
                    training_trials = inner_training[inner_f].astype(int)
                    testing_trials = inner_test[inner_f].astype(int)
                    for i_l, l in enumerate(lambdas):
                        for b in np.arange(n_bins):
                                pred, y_test = weighted_decoder(b, regressed_variable_d=regressed_variable_s, xs=xs_s,
                                                        training_trials=training_trials, 
                                                        testing_trials=testing_trials, 
                                                        decoder=decoder, l=l)
                                inner_real[testing_trials,i_l,b] = y_test
                                inner_pred[testing_trials,i_l,b] = pred
                                #print(r2_score(y_test,pred))
                inner_real = inner_real[sorted(np.hstack(inner_test)),:,:]
                inner_pred = inner_pred[sorted(np.hstack(inner_test)),:,:] 
                inner_l_performance = np.zeros([len(lambdas), n_bins])
                inner_l_performance[:] = np.nan
                for ti in np.arange(len(lambdas)):
                    for tb in np.arange(n_bins):
                        #inner_l_performance[ti,tb] = pearsonr(inner_real[:,ti,tb], inner_pred[:,ti,tb])[0]
                        inner_l_performance[ti,tb] = r2_score(inner_real[:,ti,tb], inner_pred[:,ti,tb])
                #fold_lambda = np.argmin(inner_l_performance, axis=0) # select best lambda for each bin
                fold_lambda = np.argmax(inner_l_performance, axis=0) # select best lambda for each bin
                lambdas_store.append(lambdas[fold_lambda])
                # End of inner XVal
                for b in np.arange(n_bins):
                    pred, y_test, coef, intercept = weighted_decoder(b, regressed_variable_d=regressed_variable_s, xs=xs_s, 
                                                training_trials=outer_training.astype(int), 
                                                testing_trials=outer_test.astype(int), 
                                                decoder=decoder, l=lambdas[fold_lambda[b]], 
                                                return_pred = True)
                    coef_sum[subsample, nc, b] += coef
                    coef_count[subsample, nc, b] += 1
                    intercept_sum[nc, b] += intercept
                    intercept_count[nc, b] += 1
                    if trial_filter is not None:
                        pred_summary[trials_included_filtered[outer_test].astype(int),b,nc,s] = pred
                        ytest_summary[trials_included_filtered[outer_test].astype(int),b,nc,s] = y_test
                    else:
                        pred_summary[trials_included[outer_test].astype(int),b,nc,s] = pred
                        ytest_summary[trials_included[outer_test].astype(int),b,nc,s] = y_test
    if trial_filter is not None:
        pearson_summary , mse_summary, mpearson_summary , mmse_summary = summary_score(ytest_summary,pred_summary,trials_included_filtered)
    else:
        pearson_summary , mse_summary, mpearson_summary , mmse_summary = summary_score(ytest_summary,pred_summary,list(trials_included.astype(int)))
    
    with np.errstate(invalid='ignore', divide='ignore'):
        coef_mean      = coef_sum / coef_count            # NaN where a neuron was never drawn in that combo
        intercept_mean = intercept_sum / intercept_count
    if null==False:
        return pearson_summary, mse_summary, pred_summary, lambdas_store , mpearson_summary , mmse_summary, coef_mean, intercept_mean
    else:                     
        return pearson_summary, mse_summary, lambdas_store , mpearson_summary , mmse_summary, coef_mean, intercept_mean

def summary_score(ytest_summary,pred_summary,trials_included_filtered):
    r2 = np.zeros([ytest_summary.shape[1], ytest_summary.shape[2]])
    r = np.zeros([ytest_summary.shape[1], ytest_summary.shape[2]])
    r2m = np.zeros([ytest_summary.shape[1], ytest_summary.shape[2]])
    rm = np.zeros([ytest_summary.shape[1], ytest_summary.shape[2]])
    for j in np.arange(ytest_summary.shape[1]):
        for nneurons in np.arange(ytest_summary.shape[2]):
            r2bcs=[]
            rbcs=[]
            for sample in np.arange(ytest_summary.shape[3]):
                r2bcs.append(r2_score(ytest_summary[trials_included_filtered,j,0,0], 
                            pred_summary[trials_included_filtered,j,nneurons,sample], 
                            multioutput='uniform_average'))
                rbcs.append(pearsonr(ytest_summary[:,-1,0,0][trials_included_filtered], 
                          pred_summary[trials_included_filtered,j,nneurons,sample])[0])
            r2[j,nneurons] = np.mean(r2bcs)
            r[j,nneurons] = np.mean(rbcs)    
            r2m[j,nneurons] = np.max(r2bcs)
            rm[j,nneurons] = np.max(rbcs)  
    return r, r2, rm, r2m

def weighted_decoder(b, regressed_variable_d=None, xs=None,
    training_trials=None,
    testing_trials=None,
    decoder=None, l=None, return_pred = False):
    # `history` (concatenating earlier bins onto X) and `weights`
    # (sample_weight) were never used: no caller passed either. The
    # ElasticNet and LogisticRegression branches are gone with them - the
    # decoder is always Lasso. `l` is always supplied, so the unpenalised
    # `l is None` fit is gone too.
    spike_data = xs[:,:,b]
    X_train = spike_data[training_trials]
    X_test = spike_data[testing_trials]
    y_train = regressed_variable_d[training_trials]
    y_test = regressed_variable_d[testing_trials]
    reg = decoder(alpha=l).fit(X_train, y_train)
    y_pred = reg.predict(X_test)
    if return_pred:
        coef = reg.coef_
        intercept = reg.intercept_
        return y_pred, y_test, coef, intercept
    return y_pred, y_test

def run_decoder_for_session_residual(c_neural_data, area, alfio, regressed_variable_org, weights, alignment_time, etype = 'real', 
                            output_folder='/jukebox/witten/Alex/decoder_output', n_neurons_minimum = 20, 
                            n=None, decoder = 'lasso', lambdas=None,
                            trial_filter=None):
    """
    Run decoder for session residual data.

    Args:
        weights: Kept, and ignored. Sample weighting was available but never
            used - every decoder ran unweighted - and this signature is what
            the equivalence test pins, so the parameter stays where it is.
        trial_filter: An array of trials containing a particular variable. It does not need to be the same
            length as the trials in c_neural_data (in other words it can just be any set of trials in
            standard trial indexing). The run_decoder function will find the common trials between the
            set in "trial_filter", e.g. rewarded trials, and the set for which we have binned data available
            after all criterion (presence etc).
        etype: 'real' or 'null'. The 'time' and 'lags' variants were never
            run for this paper and have been removed.
    """
    if decoder != 'lasso':
        raise ValueError("only 'lasso' ran; the logistic path has been removed")
    decoder_type = LR
    # Regressed variable can be a list of arrays in the case of delta q (QR-QL and QL-QR)
    hem = c_neural_data['hem']
    for h in np.unique(hem): # Only analyze ensambles from a single hemisphere, even if from multiple probes, they should be from the same hem
        try:
            hem_neural_data = c_neural_data.loc[c_neural_data['hem']==h].copy()
            if len(regressed_variable_org)==2: #then we are decoding deltaq or qcontra/ipsi, need to choose R-L or L-R so that it matches contra-ipsi 
                regressed_variable = regressed_variable_org[int(h)].copy()
                if trial_filter is not None:
                    trial_filter_h = trial_filter[int(h)]
                else:
                    trial_filter_h = None
            else:
                regressed_variable = regressed_variable_org.copy()
                if trial_filter is not None:
                    trial_filter_h = trial_filter.copy()
                else:
                    trial_filter_h = None
            if etype=='real':
                p_summary, mse_summary, pred_summary, lambdas_store, mpearson_summary , mmse_summary, coef, intercept  = run_decoder(hem_neural_data, alignment_time, regressed_variable, 
                                                                    n_neurons_minimum=n_neurons_minimum, decoder = decoder_type, 
                                                                    lambdas=lambdas, trial_filter=trial_filter_h, null=False)
                np.save(output_folder+'/'+str(etype)+'_'+str(area)+'_'+str(alfio.mouse)+'_'+str(alfio.date)+'_'+str(int(h))+'_p_summary.npy', p_summary)
                np.save(output_folder+'/'+str(etype)+'_'+str(area)+'_'+str(alfio.mouse)+'_'+str(alfio.date)+'_'+str(int(h))+'_neural_pred_value.npy', pred_summary)
                np.save(output_folder+'/'+str(etype)+'_'+str(area)+'_'+str(alfio.mouse)+'_'+str(alfio.date)+'_'+str(int(h))+'_mse_summary.npy', mse_summary)
                np.save(output_folder+'/'+str(etype)+'_'+str(area)+'_'+str(alfio.mouse)+'_'+str(alfio.date)+'_'+str(int(h))+'_mmse_summary.npy', mmse_summary)
                np.save(output_folder+'/'+str(etype)+'_'+str(area)+'_'+str(alfio.mouse)+'_'+str(alfio.date)+'_'+str(int(h))+'_lambdas_summary.npy', lambdas_store)
                np.save(output_folder+'/'+str(etype)+'_'+str(area)+'_'+str(alfio.mouse)+'_'+str(alfio.date)+'_'+str(int(h))+'_coef.npy', coef)
                np.save(output_folder+'/'+str(etype)+'_'+str(area)+'_'+str(alfio.mouse)+'_'+str(alfio.date)+'_'+str(int(h))+'_coef_cluster_ids.npy', hem_neural_data['cluster_id'].to_numpy())
                np.save(output_folder+'/'+str(etype)+'_'+str(area)+'_'+str(alfio.mouse)+'_'+str(alfio.date)+'_'+str(int(h))+'_intercept.npy', intercept)
            else:
                p_summary, mse_summary, lambdas_store, mpearson_summary , mmse_summary, coef, intercept = run_decoder(hem_neural_data, alignment_time, regressed_variable, n_neurons_minimum=n_neurons_minimum, 
                                                    decoder = decoder_type,  lambdas=lambdas,
                                                    trial_filter=trial_filter_h, null=True)
                np.save(output_folder+'/'+str(n)+'_'+str(etype)+'_'+str(area)+'_'+str(alfio.mouse)+'_'+str(alfio.date)+'_'+str(int(h))+'_p_summary.npy', p_summary)
                np.save(output_folder+'/'+str(n)+'_'+str(etype)+'_'+str(area)+'_'+str(alfio.mouse)+'_'+str(alfio.date)+'_'+str(int(h))+'_mse_summary.npy', mse_summary)
                np.save(output_folder+'/'+str(n)+'_'+str(etype)+'_'+str(area)+'_'+str(alfio.mouse)+'_'+str(alfio.date)+'_'+str(int(h))+'_mmse_summary.npy', mmse_summary)
                np.save(output_folder+'/'+str(n)+'_'+str(etype)+'_'+str(area)+'_'+str(alfio.mouse)+'_'+str(alfio.date)+'_'+str(int(h))+'_coef.npy', coef)
                np.save(output_folder+'/'+str(n)+'_'+str(etype)+'_'+str(area)+'_'+str(alfio.mouse)+'_'+str(alfio.date)+'_'+str(int(h))+'_intercept.npy', intercept)
        
        except Exception as ex:
            print(ex)
            continue