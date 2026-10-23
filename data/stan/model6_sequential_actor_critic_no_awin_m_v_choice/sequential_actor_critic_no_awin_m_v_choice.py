if __name__ == "__main__":
    import numpy as np
    #cmdstanpy.install_cmdstan()
    from cmdstanpy import cmdstan_path, CmdStanModel
    import pandas as pd
    from pathlib import Path
    import os
    import pickle
    import _pickle as cPickle

    LOAD_DF = False
    REEXTRACT = False
    ROOT_FOLDER =  '/jukebox/witten/Alex/Data/ephys_bandit/data_laser_only'
    SPOCK_FOLDER =  '/jukebox/witten/Alex/Data/ephys_bandit/laser_stan_fits/sequential_actor_critic_no_awin_m_v_choice'
    OUTPUT =  '/jukebox/witten/Alex/Data/ephys_bandit/laser_stan_fits/sequential_actor_critic_no_awin_m_v_choice/output/'

    # Load data
    if LOAD_DF==False:
        root_path = Path(ROOT_FOLDER)
        psy = pd.DataFrame()
        for animal in sorted(root_path.iterdir()):
            if animal.is_dir():
                for day in sorted(animal.iterdir()):
                    if day.is_dir():
                        for ses in sorted(day.iterdir()):
                            if ses.is_dir():
                                if REEXTRACT==True:
                                    full_bandit_fix(ses.as_posix())
                                mouse_psy = pd.DataFrame()
                                mouse_psy['reward'] = \
                                    1*(np.load(ses.joinpath('alf', '_ibl_trials.feedbackType.npy'))>0)
                                mouse_psy['choices'] = \
                                    1*((-1*np.load(ses.joinpath('alf', '_ibl_trials.choice.npy')))>0)
                                mouse_psy['laser_consumed'] = \
                                    1*(np.load(ses.joinpath('alf', '_ibl_trials.feedbackTypeConsumed.npy'))>0)
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

    # 2) trial variables
    o = np.zeros([NS,NSESS,NT])
    c = np.zeros([NS,NSESS,NT])
    l = np.zeros([NS,NSESS,NT])
    for ns, mouse in enumerate(sorted(psy['mouse'].unique())):
        animal = psy.loc[psy['mouse']==mouse]
        counter=0
        for d, day in enumerate(sorted(animal['date'].unique())):
            day_s = animal.loc[animal['date']==day]
            for nsess, ses in enumerate(sorted(day_s['ses'].unique())):
                session = day_s.loc[day_s['ses']==ses]
                o[ns, counter, :len(session)] = session['reward']
                c[ns, counter, :len(session)] = session['choices']
                l[ns, counter, :len(session)] = session['laser_consumed']
                sess_idx[(ns*NSESS)+counter] = counter
                NT_all[(ns*NSESS)+counter] = len(session)
                counter+=1

    standata = {'NS': int(NS) ,'NT': int(NT),'NSESS': int(NSESS),
               'o':o.astype(int), 'c':c.astype(int), 'l':l.astype(int),
               'NT_all':np.array(NT_all).astype(int), 'NSxNSESS':int(NSxNSESS),
               'sess_idx':np.array(sess_idx).astype(int)+1,
               'sub_idx':np.array(sub_idx).astype(int)+1}

    bernoulli_model = CmdStanModel(
        stan_file="/jukebox/witten/Alex/Data/ephys_bandit/laser_stan_fits/sequential_actor_critic_no_awin_m_v_choice/sequential_actor_critic_no_awin_m_v_choice.stan")
    # Compile the model
    fit = bernoulli_model.sample(data=standata, output_dir=OUTPUT,
                    iter_sampling=6000,iter_warmup=2000,chains=4,adapt_delta=0.99,
                    max_treedepth= 13, seed = 17096)
    fit.summary().to_csv(OUTPUT+"summary.csv")
