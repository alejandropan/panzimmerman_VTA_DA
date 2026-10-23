if __name__ == "__main__":
    import numpy as np
    import sys
    sys.path.append('/jukebox/witten/Alex/PYTHON/rewardworld/ephys/analysis')
    from session_maker import TrialParamHandler   
    from cmdstanpy import CmdStanModel
    import pandas as pd
    from pathlib import Path

    LOAD_DF = False
    REEXTRACT = False
    ROOT_FOLDER =  '/jukebox/witten/Alex/Data/ephys_bandit/data_laser_only'
    SPOCK_FOLDER =  '/jukebox/witten/Alex/Data/ephys_bandit/laser_stan_fits/recovery_models/REINFORCE_recovery/'
    OUTPUT = '/jukebox/witten/Alex/Data/ephys_bandit/laser_stan_fits/recovery_models/REINFORCE_recovery/output/'
    saved_params = pd.read_csv('/jukebox/witten/Alex/Data/ephys_bandit/laser_stan_fits/REINFORCE_new/output/summary.csv')
    saved_params = saved_params.rename(columns={'Unnamed: 0':'name'})

    # Load data
    def inv_logit(arr):
        '''Elementwise inverse logit (logistic) function.'''
        return 1 / (1 + np.exp(-arr))

    def load_sim_data_reduced(ROOT_FOLDER = '/jukebox/witten/Alex/Data/ephys_bandit/data_laser_only', LOAD_DF = False, REEXTRACT=False, trial_start=0,trial_end=None):
        if LOAD_DF==False:
            root_path = Path(ROOT_FOLDER)
            psy = pd.DataFrame()
            for animal in sorted(root_path.iterdir()):
                if animal.is_dir():
                    print(animal)
                    for day in sorted(animal.iterdir()):
                        if day.is_dir():
                            for ses in sorted(day.iterdir()):
                                if ses.is_dir():
                                    if REEXTRACT==True:
                                        full_bandit_fix(ses.as_posix())
                                    mouse_psy = pd.DataFrame()
                                    mouse_psy['feedback'] = \
                                        1*(np.load(ses.joinpath('alf', '_ibl_trials.feedbackType.npy'))>0)[trial_start:trial_end]
                                    mouse_psy['choices'] = \
                                        1*((-1*np.load(ses.joinpath('alf', '_ibl_trials.choice.npy')))>0)[trial_start:trial_end]
                                    mouse_psy['probabilityLeft'] = \
                                        np.load(ses.joinpath('alf', '_ibl_trials.probabilityLeft.npy'))[trial_start:trial_end]
                                    mouse_psy['reward'] = mouse_psy['feedback']
                                    mouse_psy['mouse'] = animal.name
                                    mouse_psy['ses'] = ses.name
                                    mouse_psy['date'] = day.name
                                    psy = pd.concat([psy, mouse_psy])
                                else:
                                    continue
                        else:
                            continue
                else:
                    continue
        else:
            psy = pd.read_pickle("psy.pkl")
        # Prepara data for Stan​
        # 1) optomization variables
        NS = len(psy['mouse'].unique())
        NSESS = psy.groupby(['mouse','date','ses']).size().reset_index()['mouse'].value_counts().max() # Maximum number of sessions
        NT = psy.groupby(['mouse','date','ses']).size().reset_index()[0].max()
        NSxNSESS = NS*NSESS
        NT_all = np.zeros(NSxNSESS)
        sub_idx = np.zeros(NSxNSESS) #  Mapping parameter to subject
        for i in np.arange(psy['mouse'].unique().shape[0]):
            sub_idx[i*NSESS:(i+1)*NSESS]=i
        sess_idx =  np.zeros(NSxNSESS) # Mapping parameter to session
        folder_dic = pd.DataFrame(columns=['mouse', 'day', 'ses', 'sess_idx', 'NT_all']) # Will hold relation between arrays and folders
        # 2) trial variables
        r = np.zeros([NS,NSESS,NT])
        c = np.zeros([NS,NSESS,NT])
        p = np.zeros([NS,NSESS,NT])
        tb = np.zeros([NS,NSESS,NT])
        for ns, mouse in enumerate(sorted(psy['mouse'].unique())):
            animal = psy.loc[psy['mouse']==mouse]
            counter=0
            for d, day in enumerate(sorted(animal['date'].unique())):
                day_s = animal.loc[animal['date']==day]
                for nsess, ses in enumerate(sorted(day_s['ses'].unique())):
                    session = day_s.loc[day_s['ses']==ses]
                    r[ns, counter, :len(session)] = session['reward']
                    c[ns, counter, :len(session)] = session['choices']
                    p[ns, counter, :len(session)] = session['probabilityLeft']
                    bchange = 1*(np.diff(session['probabilityLeft'])!=0)
                    for t in np.arange(len(session)):
                        if t==0:
                            twb = 1
                            tb[ns, counter, t] = twb
                            twb+=1
                        elif t==len(session)-1:
                            tb[ns, counter, t] = twb
                        else:
                            if bchange[t]==1:
                                tb[ns, counter, t] = twb
                                twb = 1
                            if bchange[t]==0:
                                tb[ns, counter, t] = twb
                                twb+=1
                    sess_idx[(ns*NSESS)+counter] = counter
                    NT_all[(ns*NSESS)+counter] = len(session)
                    counter+=1
                    ses_df = pd.DataFrame()
                    ses_df['mouse'] = [mouse]
                    ses_df['day'] = [day]
                    ses_df['ses'] = [ses]
                    ses_df['sess_idx'] = [counter]
                    ses_df['NT_all'] = [len(session)]
                    folder_dic = pd.concat([folder_dic, ses_df])

        standata_recovery = {'NS': int(NS) ,'NT': int(NT),'NSESS': int(NSESS),
                'p':p.astype(float), 'tb':tb.astype(int),
                'NT_all':np.array(NT_all).astype(int), 'NSxNSESS':int(NSxNSESS),
                'sess_idx':np.array(sess_idx).astype(int)+1,
                'sub_idx':np.array(sub_idx).astype(int)+1}

        return standata_recovery

    def simulate_REINFORCE(standata_recovery,saved_params=None, fit=None, csv=True):
        NS =standata_recovery['NS']
        NSESS=standata_recovery['NSESS']
        NT =standata_recovery['NT']
        NT_all =standata_recovery['NT_all']
        c_sim = np.zeros([NS,NSESS,NT])
        o_sim = np.zeros([NS,NSESS,NT])
        sub_idx =  standata_recovery['sub_idx']-1
        sess_idx = standata_recovery['sess_idx']-1
        NSxNSESS = standata_recovery['NSxNSESS']
        sim_data = pd.DataFrame()
        for ms_i in np.arange(NSxNSESS):
            if NT_all[ms_i]>0:
                if saved_params is not None:
                    if ms_i==0:
                        print('Using saved parameters')
                    if csv==True:
                        dmwater_ses = float(saved_params.loc[saved_params['name']=='dmwater_ses['+str(ms_i+1)+']', 'Mean'])
                        beta_ses = float(saved_params.loc[saved_params['name']=='beta_ses['+str(ms_i+1)+']', 'Mean'])
                        side_ses = float(saved_params.loc[saved_params['name']=='sides['+str(ms_i+1)+']', 'Mean'])
                        alpha = float(saved_params.loc[saved_params['name']=='alpha_ses['+str(ms_i+1)+']', 'Mean'])
                        alphaforgetting= float(saved_params.loc[saved_params['name']=='alphaforgetting_ses['+str(ms_i+1)+']', 'Mean'])
                    else:
                        print('Error only CSV version available')

                tph = TrialParamHandler()
                q = 0
                predicted_choices=[]
                rewards = []
                outcome=[]
                side_blocks=[]
                trial_within_block=[]
                right_reward=[]
                left_reward=[]
                buffer_qlaser=[]
                buffer_qwater=[]

                for t in np.arange(NT_all[ms_i]):
                    t = int(t)    
                    p_choice = inv_logit(side_ses + q)
                    choice =  np.random.choice(2,p= [1-p_choice,p_choice])
                    working_choice = 2 * choice -1
                    tph.response_side_buffer.append(choice)  # Add choice to response buffer
                    p_left= tph.stim_probability_left
                    if choice==1:
                        f = tph.right_reward
                    if choice==0:
                        f = tph.left_reward
                    if f == 1:
                        q = (1 - alpha) * q + working_choice * beta_ses * (1-inv_logit(working_choice*(side_ses + q)))
                    if f == 0:
                        q = (1 - alphaforgetting) * q + working_choice * dmwater_ses * (1-inv_logit(working_choice*(side_ses + q)))
                    
                    # Store variables
                    rewards.append(f)
                    outcome.append(f)
                    side_blocks.append(p_left)
                    predicted_choices.append(choice)
                    trial_within_block.append(tph.block_trial_num)
                    right_reward.append(tph.right_reward)
                    left_reward.append(tph.left_reward)
                    buffer_qwater.append(q)
                    # Init next trial
                    tph =  tph.next_trial()

                if NT_all[ms_i]!=0:
                    c_sim[sub_idx[ms_i], sess_idx[ms_i], :NT_all[ms_i]]= predicted_choices
                    o_sim[sub_idx[ms_i], sess_idx[ms_i], :NT_all[ms_i]] = rewards
                    ses_data = pd.DataFrame()
                    ses_data['choices'] = predicted_choices
                    ses_data['water'] = rewards
                    ses_data['tb'] = trial_within_block
                    ses_data['probabilityLeft'] = side_blocks
                    ses_data['outcome'] = outcome
                    ses_data['mouse'] = sub_idx[ms_i]
                    ses_data['ses'] = sess_idx[ms_i]
                    ses_data['right_reward'] = right_reward
                    ses_data['left_reward'] = left_reward
                    ses_data['qwater'] = buffer_qwater
                    sim_data = pd.concat([sim_data,ses_data])
        return c_sim, o_sim, sim_data


    # Simulate data
    standata_recovery  = load_sim_data_reduced()
    c_sim, o_sim, sim_data = simulate_REINFORCE(standata_recovery,saved_params)
    standata_recovery['o'] = o_sim.astype(int)
    standata_recovery['c'] = c_sim.astype(int)
    np.save('/jukebox/witten/Alex/Data/ephys_bandit/laser_stan_fits/recovery_models/REINFORCE_recovery/o_sim.npy', o_sim)
    np.save('/jukebox/witten/Alex/Data/ephys_bandit/laser_stan_fits/recovery_models/REINFORCE_recovery/c_sim.npy', c_sim)
    bernoulli_model = CmdStanModel(
        stan_file="/jukebox/witten/Alex/Data/ephys_bandit/laser_stan_fits/recovery_models/REINFORCE_recovery/REINFORCE_new.stan")
    # Compile the model
    fit = bernoulli_model.sample(data=standata_recovery, output_dir=OUTPUT,
                    iter_sampling=1500,iter_warmup=350,chains=4,adapt_delta=0.99,
                    max_treedepth= 13)
    fit.summary().to_csv(OUTPUT+"summary.csv")