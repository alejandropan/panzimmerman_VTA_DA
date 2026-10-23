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
    ROOT_FOLDER  = '/jukebox/witten/Alex/Data/ephys_bandit/data_laser_only'
    SPOCK_FOLDER = '/jukebox/witten/Alex/Data/ephys_bandit/laser_stan_fits/recovery_models/sequential_actor_critic_no_awin_m_v_choice_recovery/'
    OUTPUT       = '/jukebox/witten/Alex/Data/ephys_bandit/laser_stan_fits/recovery_models/sequential_actor_critic_no_awin_m_v_choice_recovery/output/'
    saved_params = pd.read_csv('/jukebox/witten/Alex/Data/ephys_bandit/laser_stan_fits/sequential_actor_critic_m_v_choice/output/summary.csv')
    saved_params = saved_params.rename(columns={'Unnamed: 0': 'name'})

    # Load data
    def inv_logit(arr):
        '''Elementwise inverse logit (logistic) function.'''
        return 1 / (1 + np.exp(-arr))

    def load_sim_data_reduced(ROOT_FOLDER='/jukebox/witten/Alex/Data/ephys_bandit/data_laser_only', LOAD_DF=False, REEXTRACT=False, trial_start=0, trial_end=None):
        if LOAD_DF == False:
            root_path = Path(ROOT_FOLDER)
            psy = pd.DataFrame()
            for animal in sorted(root_path.iterdir()):
                if animal.is_dir():
                    print(animal)
                    for day in sorted(animal.iterdir()):
                        if day.is_dir():
                            for ses in sorted(day.iterdir()):
                                if ses.is_dir():
                                    if REEXTRACT == True:
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
        # Prepara data for Stan
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

    def simulate_sequential_actor_critic(standata_recovery, saved_params=None, fit=None, csv=True):
        '''
        Fully generative simulation of the Sequential Actor-Critic (dual REINFORCE).

        IMPORTANT distinction between the two outcome variables:
          - o (reward / CS outcome): drawn by the bandit environment via the
            TrialParamHandler (tph), exactly as in the standard AC and
            standard_w_forgetting recovery scripts. o==1 means the chosen side
            was rewarded on this trial (the CS tone indicating reward); there is
            no water in this task.
          - l (laser consumed): a SEPARATE Bernoulli event that can only occur on
            o==1 trials. A positive (rewarded) outcome does NOT automatically produce a
            consumed laser; the animal must "lick", drawn from inv_logit(pi_lick).
            This mirrors the Stan model, where the lick likelihood is only
            evaluated on reward (o==1) trials and l drives the V_cs / pi_lick
            (laser critic + lick policy) updates.

        State variables (match the Stan likelihood exactly):
          V_cs       — CS/laser critic value
          V_choice   — choice critic value
          pi_choice  — choice policy log-odds (actor)
          pi_lick    — lick policy log-odds (actor)
        '''
        NS       = standata_recovery['NS']
        NSESS    = standata_recovery['NSESS']
        NT       = standata_recovery['NT']
        NT_all   = standata_recovery['NT_all']
        c_sim    = np.zeros([NS, NSESS, NT])
        o_sim    = np.zeros([NS, NSESS, NT])
        l_sim    = np.zeros([NS, NSESS, NT])
        sub_idx  = standata_recovery['sub_idx'] - 1
        sess_idx = standata_recovery['sess_idx'] - 1
        NSxNSESS = standata_recovery['NSxNSESS']
        sim_data = pd.DataFrame()

        for ms_i in np.arange(NSxNSESS):
            if NT_all[ms_i] > 0:
                if saved_params is not None:
                    if ms_i == 0:
                        print('Using saved parameters')
                    if csv == True:
                        sub_i = int(sub_idx[ms_i]) + 1   # 1-indexed mouse for param lookup

                        # --- mouse-level parameters (no session subscript) --- #
                        # alpha_v_cs alpha_v_choice and alpha_lick are mouse-level only in this model
                        alpha_v_choice = inv_logit(float(saved_params.loc[saved_params['name']=='alpha_v_choice_mouse['+str(sub_i)+']', 'Mean']))
                        alpha_v_cs = inv_logit(float(saved_params.loc[saved_params['name']=='alpha_v_cs_mouse['+str(sub_i)+']', 'Mean']))
                        alpha_lick = inv_logit(float(saved_params.loc[saved_params['name']=='alpha_lick_mouse['+str(sub_i)+']', 'Mean']))
                        beta_lick  = float(saved_params.loc[saved_params['name']=='beta_lick_mouse['+str(sub_i)+']', 'Mean'])

                        # --- session-level parameters --- #
                        alpha_loss     = inv_logit(float(saved_params.loc[saved_params['name']=='alpha_loss_ses['+str(ms_i+1)+']', 'Mean']))
                        beta_win       = float(saved_params.loc[saved_params['name']=='beta_win_ses['+str(ms_i+1)+']', 'Mean'])
                        beta_loss      = float(saved_params.loc[saved_params['name']=='beta_loss_ses['+str(ms_i+1)+']', 'Mean'])
                        side_ses       = float(saved_params.loc[saved_params['name']=='side_ses['+str(ms_i+1)+']', 'Mean'])
                        V_cs_init      = float(saved_params.loc[saved_params['name']=='V_cs_init_ses['+str(ms_i+1)+']', 'Mean'])
                        V_choice_init  = float(saved_params.loc[saved_params['name']=='V_choice_init_ses['+str(ms_i+1)+']', 'Mean'])
                        pi_lick_init   = float(saved_params.loc[saved_params['name']=='pi_lick_init_ses['+str(ms_i+1)+']', 'Mean'])
                    else:
                        print('Error only CSV version available')

                tph = TrialParamHandler()

                # --- initialise state (matches Stan transformed/likelihood init) --- #
                V_cs      = V_cs_init
                V_choice  = V_choice_init
                pi_choice = 0.0
                pi_lick   = pi_lick_init

                predicted_choices = []
                outcome           = []        # reward / CS outcome (o): rewarded trial (tone) or not
                licks             = []        # laser consumed (l)
                side_blocks       = []
                trial_within_block= []
                right_reward      = []
                left_reward       = []
                buffer_V_cs       = []
                buffer_V_choice   = []
                buffer_pi_choice  = []
                buffer_pi_lick    = []

                for t in np.arange(NT_all[ms_i]):
                    t = int(t)

                    # --- choice (actor) --- #
                    p_choice = inv_logit(side_ses + pi_choice)
                    choice   = np.random.choice(2, p=[1-p_choice, p_choice])
                    action_sign = 2 * choice - 1
                    tph.response_side_buffer.append(choice)  # Add choice to response buffer
                    p_left = tph.stim_probability_left

                    # p_chosen for the REINFORCE (1 - p_chosen) advantage term
                    if choice == 1:
                        p_chosen = inv_logit(side_ses + pi_choice)
                    else:
                        p_chosen = 1.0 - inv_logit(side_ses + pi_choice)

                    # --- reward / CS outcome from the bandit (tph): was the trial rewarded? --- #
                    if choice == 1:
                        f = tph.right_reward
                    if choice == 0:
                        f = tph.left_reward

                    # --- learning updates (match Stan likelihood exactly) --- #
                    if f == 1:
                        # win: choice critic update + win-REINFORCE on pi_choice
                        delta_cs  = V_cs - V_choice
                        V_choice += alpha_v_choice * delta_cs
                        pi_choice = pi_choice \
                                    + action_sign * beta_win * delta_cs * (1.0 - p_chosen)

                        # lick decision (laser consumed) — separate Bernoulli, only on win trials
                        p_lick = inv_logit(pi_lick)
                        lick   = int(np.random.choice(2, p=[1-p_lick, p_lick]))

                        if lick == 1:
                            delta_laser = 1.0 - V_cs
                            V_cs       += alpha_v_cs * delta_laser
                            pi_lick     = (1.0 - alpha_lick) * pi_lick \
                                          + beta_lick * delta_laser * (1.0 - p_lick)
                        else:
                            delta_laser = 0.0 - V_cs
                            V_cs       += alpha_v_cs * delta_laser
                            pi_lick     = (1.0 - alpha_lick) * pi_lick
                    else:
                        # no reward: choice critic update + loss-REINFORCE on pi_choice
                        lick = 0   # lick not observed / not delivered on no-reward trials
                        delta_cs  = 0.0 - V_choice
                        V_choice += alpha_v_choice * delta_cs
                        pi_choice = (1.0 - alpha_loss) * pi_choice \
                                    + action_sign * beta_loss * delta_cs * (1.0 - p_chosen)

                    # --- store --- #
                    predicted_choices.append(choice)
                    outcome.append(f)
                    licks.append(lick)
                    side_blocks.append(p_left)
                    trial_within_block.append(tph.block_trial_num)
                    right_reward.append(tph.right_reward)
                    left_reward.append(tph.left_reward)
                    buffer_V_cs.append(V_cs)
                    buffer_V_choice.append(V_choice)
                    buffer_pi_choice.append(pi_choice)
                    buffer_pi_lick.append(pi_lick)

                    # Init next trial
                    tph = tph.next_trial()

                if NT_all[ms_i] != 0:
                    c_sim[sub_idx[ms_i], sess_idx[ms_i], :NT_all[ms_i]] = predicted_choices
                    o_sim[sub_idx[ms_i], sess_idx[ms_i], :NT_all[ms_i]] = outcome
                    l_sim[sub_idx[ms_i], sess_idx[ms_i], :NT_all[ms_i]] = licks
                    ses_data = pd.DataFrame()
                    ses_data['choices'] = predicted_choices
                    ses_data['outcome'] = outcome          # reward / CS outcome (1 = rewarded trial)
                    ses_data['lick'] = licks               # laser consumed
                    ses_data['tb'] = trial_within_block
                    ses_data['probabilityLeft'] = side_blocks
                    ses_data['mouse'] = sub_idx[ms_i]
                    ses_data['ses'] = sess_idx[ms_i]
                    ses_data['right_reward'] = right_reward
                    ses_data['left_reward'] = left_reward
                    ses_data['V_cs'] = buffer_V_cs
                    ses_data['V_choice'] = buffer_V_choice
                    ses_data['pi_choice'] = buffer_pi_choice
                    ses_data['pi_lick'] = buffer_pi_lick
                    sim_data = pd.concat([sim_data, ses_data])
        return c_sim, o_sim, l_sim, sim_data

    # Simulate data
    standata_recovery = load_sim_data_reduced()
    c_sim, o_sim, l_sim, sim_data = simulate_sequential_actor_critic(standata_recovery, saved_params)
    standata_recovery['o'] = o_sim.astype(int)
    standata_recovery['c'] = c_sim.astype(int)
    standata_recovery['l'] = l_sim.astype(int)
    np.save(SPOCK_FOLDER + 'o_sim.npy', o_sim)
    np.save(SPOCK_FOLDER + 'c_sim.npy', c_sim)
    np.save(SPOCK_FOLDER + 'l_sim.npy', l_sim)
    sim_data.to_csv(SPOCK_FOLDER + 'sim_data.csv', index=False)  # per-trial DVs for pre-fit diagnostics
    bernoulli_model = CmdStanModel(
        stan_file=SPOCK_FOLDER + "sequential_actor_critic_no_awin_m_v_choice.stan")
    # Compile the model
    fit = bernoulli_model.sample(data=standata_recovery, output_dir=OUTPUT,
                    iter_sampling=6000,iter_warmup=2000,chains=4,adapt_delta=0.99,
                    max_treedepth= 13, seed = 87096)
    fit.summary().to_csv(OUTPUT+"summary.csv")
