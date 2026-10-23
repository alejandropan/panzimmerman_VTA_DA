# Import packages
import numpy as np
import seaborn as sns
import pandas as pd
from pathlib import Path
from matplotlib import pyplot as plt
import glob
from session_maker import TrialParamHandler   
from scipy.stats import zscore
import scipy.stats as ss
# Common Functions

def inv_logit(arr):
    '''Elementwise inverse logit (logistic) function.'''
    return 1.0 / (1.0 + np.exp(-arr))

def phi_approx(arr):
    '''Elementwise fast approximation of the cumulative unit normal.
    For details, see Bowling et al. (2009). "A logistic approximation
    to the cumulative normal distribution."'''
    return inv_logit(0.07056 * arr ** 3 + 1.5976 * arr)

def  num_to_name(psy):
    ses_id=[]
    for ns, mouse in enumerate(psy['mouse'].unique()):
        animal = psy.loc[psy['mouse']==mouse]
        counter=0
        for d, day in enumerate(animal['date'].unique()):
            day_s = animal.loc[animal['date']==day]
            for nsess, ses in enumerate(day_s['ses'].unique()):
                ses_id.append(mouse+'_'+day+'_'+ses)
    return ses_id



 
def load_data_sessions_stan(sessions, ROOT_FOLDER, trial_start=0, trial_end=None,
                            consumed=False, REEXTRACT=False, expect_sessions=None):

    root = Path(ROOT_FOLDER)
 
    # The same normalisation `load_data_reduced`'s `sessions=` branch applies,
    # repeated here only so the paths can be checked and named before the load
    # starts. The list handed on is the normalised one, which that branch
    # accepts unchanged.
    relative = [str(s).replace('\\', '/').strip('/') for s in sessions]
 
    duplicates = sorted({p for p in relative if relative.count(p) > 1})
    if duplicates:
        raise ValueError(
            f"{len(duplicates)} session(s) listed more than once: "
            + ", ".join(duplicates))
 
    missing = [p for p in relative if not (root / p).is_dir()]
    if missing:
        raise FileNotFoundError(
            f"{len(missing)} of {len(relative)} sessions are not under {root}:\n  "
            + "\n  ".join(missing)
            + "\n(load_data_reduced skips missing sessions silently, which "
              "would return a grid with the wrong number of slots.)")
 
    psy = load_data_reduced(ROOT_FOLDER=str(root), sessions=relative,
                            trial_start=trial_start, trial_end=trial_end,
                            consumed=consumed, REEXTRACT=REEXTRACT)
 
    loaded = psy.groupby(['mouse', 'date', 'ses']).ngroups
    wanted = len(relative) if expect_sessions is None else expect_sessions
    if loaded != wanted:
        raise ValueError(
            f"asked for {wanted} sessions, got {loaded}. A session directory "
            "exists but produced no trials - check its alf folder.")
 
    return psy

def load_data(ROOT_FOLDER = '/Volumes/witten/Alex/Data/ephys_bandit/data_reduced', REEXTRACT=False, trial_start=0,trial_end=None):
    root_path = Path(ROOT_FOLDER)
    psy = pd.DataFrame()
    for animal in root_path.iterdir():
        if animal.is_dir():
            for day in animal.iterdir():
                if day.is_dir():
                    for ses in day.iterdir():
                        if ses.is_dir():
                            if REEXTRACT==True:
                                full_bandit_fix(ses.as_posix())
                            mouse_psy = pd.DataFrame()
                            mouse_psy['feedback'] = \
                                1*(np.load(ses.joinpath('alf', '_ibl_trials.feedbackType.npy'))>0)[trial_start:trial_end]
                            mouse_psy['choices'] = \
                                1*((-1*np.load(ses.joinpath('alf', '_ibl_trials.choice.npy')))>0)[trial_start:trial_end]
                            mouse_psy['laser_block'] = \
                                np.load(ses.joinpath('alf', '_ibl_trials.opto_block.npy'))[trial_start:trial_end]
                            mouse_psy['laser'] = mouse_psy['feedback']*mouse_psy['laser_block']
                            mouse_psy['reward'] = mouse_psy['feedback']*(1*(mouse_psy['laser_block']==0))
                            mouse_psy['mouse'] = animal.name
                            mouse_psy['ses'] = ses.name
                            mouse_psy['date'] = day.name
                            mouse_psy['probabilityLeft'] = \
                                np.load(ses.joinpath('alf', '_ibl_trials.probabilityLeft.npy'))[trial_start:trial_end]

                            psy = pd.concat([psy, mouse_psy])
                        else:
                            continue
                else:
                    continue
        else:
            continue
    return psy

def load_data_reduced(ROOT_FOLDER = '/Volumes/witten/Alex/Data/ephys_bandit/data_reduced', 
                        REEXTRACT=False, trial_start=0,trial_end=None,consumed=False, sessions=None):
    """
    Load reduced data from session folders.
    
    Args:
        ROOT_FOLDER: Root directory containing session data
        REEXTRACT: Whether to re-extract data
        trial_start: Starting trial index
        trial_end: Ending trial index (None for all)
        consumed: Whether to use consumed feedback type
        sessions: Optional list of session paths (relative to ROOT_FOLDER, e.g., '/dop_59/2023-10-29/001')
                  If None, loads all sessions. If provided, only loads specified sessions directly (faster).
    
    Returns:
        DataFrame with trial data
    """
    root_path = Path(ROOT_FOLDER)
    psy = pd.DataFrame()
    
    # If sessions list is provided, load directly from those paths (much faster)
    if sessions is not None:
        for ses_path in sessions:
            # Normalize path separators and ensure it starts with /
            normalized = ses_path.replace('\\', '/')
            if not normalized.startswith('/'):
                normalized = '/' + normalized
            
            # Construct full path to session
            ses_full_path = root_path / normalized.lstrip('/')
            
            # Skip if session doesn't exist
            if not ses_full_path.exists() or not ses_full_path.is_dir():
                continue
            
            # Extract animal, date, and session name from path
            parts = normalized.strip('/').split('/')
            if len(parts) >= 3:
                animal_name = parts[0]
                date_name = parts[1]
                ses_name = parts[2]
            else:
                continue
            
            if REEXTRACT==True:
                full_bandit_fix(ses_full_path.as_posix())
            mouse_psy = pd.DataFrame()
            if consumed==True:
                mouse_psy['feedback'] = \
                    1*(np.load(ses_full_path.joinpath('alf', '_ibl_trials.feedbackTypeConsumed.npy'))>0)[trial_start:trial_end]
            else:
                mouse_psy['feedback'] = \
                    1*(np.load(ses_full_path.joinpath('alf', '_ibl_trials.feedbackType.npy'))>0)[trial_start:trial_end]
            mouse_psy['choices'] = \
                1*((-1*np.load(ses_full_path.joinpath('alf', '_ibl_trials.choice.npy')))>0)[trial_start:trial_end]
            mouse_psy['laser_consumed'] = \
                    1*(np.load(ses_full_path.joinpath('alf', '_ibl_trials.feedbackTypeConsumed.npy'))>0)
            mouse_psy['reward'] = mouse_psy['feedback']
            mouse_psy['decision_times'] = np.load(ses_full_path.joinpath('alf', '_ibl_trials.response_times.npy')) - \
                                                    np.load(ses_full_path.joinpath('alf', '_ibl_trials.goCueTrigger_times.npy'))
            mouse_psy['r_times'] = np.load(ses_full_path.joinpath('alf', '_ibl_trials.firstMovement_times.npy')) - \
                                                    np.load(ses_full_path.joinpath('alf', '_ibl_trials.goCueTrigger_times.npy'))
            mouse_psy['mouse'] = animal_name
            mouse_psy['ses'] = ses_name
            mouse_psy['date'] = date_name
            mouse_psy['probabilityLeft'] = \
                np.load(ses_full_path.joinpath('alf', '_ibl_trials.probabilityLeft.npy'))[trial_start:trial_end]
            psy = pd.concat([psy, mouse_psy])
    else:
        # Original behavior: iterate through all directories
        for animal in sorted(root_path.iterdir()):
            if animal.is_dir():
                for day in sorted(animal.iterdir()):
                    if day.is_dir():
                        for ses in sorted(day.iterdir()):
                            if ses.is_dir():
                                if REEXTRACT==True:
                                    full_bandit_fix(ses.as_posix())
                                mouse_psy = pd.DataFrame()
                                if consumed==True:
                                    mouse_psy['feedback'] = \
                                        1*(np.load(ses.joinpath('alf', '_ibl_trials.feedbackTypeConsumed.npy'))>0)[trial_start:trial_end]
                                else:
                                    mouse_psy['feedback'] = \
                                        1*(np.load(ses.joinpath('alf', '_ibl_trials.feedbackType.npy'))>0)[trial_start:trial_end]
                                mouse_psy['choices'] = \
                                    1*((-1*np.load(ses.joinpath('alf', '_ibl_trials.choice.npy')))>0)[trial_start:trial_end]
                                mouse_psy['laser_consumed'] = \
                                        1*(np.load(ses.joinpath('alf', '_ibl_trials.feedbackTypeConsumed.npy'))>0)
                                mouse_psy['reward'] = mouse_psy['feedback']
                                mouse_psy['decision_times'] = np.load(ses.joinpath('alf', '_ibl_trials.response_times.npy')) - \
                                                                    np.load(ses.joinpath('alf', '_ibl_trials.goCueTrigger_times.npy'))
                                mouse_psy['r_times'] = np.load(ses.joinpath('alf', '_ibl_trials.firstMovement_times.npy')) - \
                                                                    np.load(ses.joinpath('alf', '_ibl_trials.goCueTrigger_times.npy'))
                                mouse_psy['mouse'] = animal.name
                                mouse_psy['ses'] = ses.name
                                mouse_psy['date'] = day.name
                                mouse_psy['probabilityLeft'] = \
                                    np.load(ses.joinpath('alf', '_ibl_trials.probabilityLeft.npy'))[trial_start:trial_end]
                                psy = pd.concat([psy, mouse_psy])
                            else:
                                continue
                    else:
                        continue
            else:
                continue
    return psy


def make_stan_data(psy, n_trials=None):
    # Prepara data for Stan​
    # 0) First check if there is a limit on trials
    if n_trials is not None:
        psy = psy.reset_index()
        psy = psy.loc[psy['index']<=n_trials]
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
    r = np.zeros([NS,NSESS,NT])
    c = np.zeros([NS,NSESS,NT])
    l = np.zeros([NS,NSESS,NT])
    b = np.zeros([NS,NSESS,NT])
    tb = np.zeros([NS,NSESS,NT])
    for ns, mouse in enumerate(psy['mouse'].unique()):
        animal = psy.loc[psy['mouse']==mouse]
        counter=0
        for d, day in enumerate(animal['date'].unique()):
            day_s = animal.loc[animal['date']==day]
            for nsess, ses in enumerate(day_s['ses'].unique()):
                session = day_s.loc[day_s['ses']==ses]
                r[ns, counter, :len(session)] = session['reward']
                c[ns, counter, :len(session)] = session['choices']
                l[ns, counter, :len(session)] = session['laser']
                b[ns, counter, :len(session)] = session['laser_block']
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

    standata = {'NS': int(NS) ,'NT': int(NT),'NSESS': int(NSESS),
               'r':r.astype(int), 'c':c.astype(int), 'l':l.astype(int), 'b':b.astype(int),
               'tb':tb.astype(int),
               'NT_all':np.array(NT_all).astype(int), 'NSxNSESS':int(NSxNSESS),
               'sess_idx':np.array(sess_idx).astype(int)+1,
               'sub_idx':np.array(sub_idx).astype(int)+1}

    return standata

def make_stan_data_reduced(psy, n_trials=None):
    # Prepara data for Stan​
    # 0) First check if there is a limit on trials
    if n_trials is not None:
        psy = psy.reset_index()
        psy = psy.loc[psy['index']<=n_trials]
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
    r = np.zeros([NS,NSESS,NT])
    c = np.zeros([NS,NSESS,NT])
    l = np.zeros([NS,NSESS,NT])
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
                l[ns, counter, :len(session)] = session['laser_consumed']
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

    standata = {'NS': int(NS) ,'NT': int(NT),'NSESS': int(NSESS),
               'r':r.astype(int), 'c':c.astype(int),'l':l.astype(int),
               'tb':tb.astype(int),
               'NT_all':np.array(NT_all).astype(int), 'NSxNSESS':int(NSxNSESS),
               'sess_idx':np.array(sess_idx).astype(int)+1,
               'sub_idx':np.array(sub_idx).astype(int)+1}

    return standata

def load_sim_data_reduced(ROOT_FOLDER = '/Volumes/witten/Alex/Data/ephys_bandit/data_reduced', LOAD_DF = False, REEXTRACT=False, trial_start=0,trial_end=-150, sessions=None):
    if LOAD_DF==False:
        root_path = Path(ROOT_FOLDER)
        psy = pd.DataFrame()
        
        # If sessions list is provided, load directly from those paths (much faster)
        if sessions is not None:
            for ses_path in sessions:
                # Normalize path separators and ensure it starts with /
                normalized = ses_path.replace('\\', '/')
                if not normalized.startswith('/'):
                    normalized = '/' + normalized
                
                # Construct full path to session
                ses_full_path = root_path / normalized.lstrip('/')
                
                # Skip if session doesn't exist
                if not ses_full_path.exists() or not ses_full_path.is_dir():
                    continue
                
                # Skip passive sessions
                if ses_full_path.name.lower() == 'passive':
                    continue
                
                # Extract animal, date, and session name from path
                parts = normalized.strip('/').split('/')
                if len(parts) >= 3:
                    animal_name = parts[0]
                    date_name = parts[1]
                    ses_name = parts[2]
                else:
                    continue
                
                if REEXTRACT==True:
                    full_bandit_fix(ses_full_path.as_posix())
                mouse_psy = pd.DataFrame()
                mouse_psy['feedback'] = \
                    1*(np.load(ses_full_path.joinpath('alf', '_ibl_trials.feedbackType.npy'))>0)[trial_start:trial_end]
                mouse_psy['choices'] = \
                    1*((-1*np.load(ses_full_path.joinpath('alf', '_ibl_trials.choice.npy')))>0)[trial_start:trial_end]
                mouse_psy['probabilityLeft'] = \
                    np.load(ses_full_path.joinpath('alf', '_ibl_trials.probabilityLeft.npy'))[trial_start:trial_end]
                mouse_psy['reward'] = mouse_psy['feedback']
                mouse_psy['mouse'] = animal_name
                mouse_psy['ses'] = ses_name
                mouse_psy['date'] = date_name
                psy = pd.concat([psy, mouse_psy])
        else:
            # Original behavior: iterate through all directories
            for animal in sorted(root_path.iterdir()):
                if animal.is_dir():
                    for day in sorted(animal.iterdir()):
                        if day.is_dir():
                            for ses in sorted(day.iterdir()):
                                if ses.is_dir():
                                    # Skip passive sessions
                                    if ses.name.lower() == 'passive':
                                        continue
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



def load_sim_data(ROOT_FOLDER = '/Volumes/witten/Alex/Data/ephys_bandit/data_reduced', LOAD_DF = False, REEXTRACT=False, trial_start=0,trial_end=-150):
    if LOAD_DF==False:
        root_path = Path(ROOT_FOLDER)
        psy = pd.DataFrame()
        for animal in root_path.iterdir():
            if animal.is_dir():
                for day in animal.iterdir():
                    if day.is_dir():
                        for ses in day.iterdir():
                            if ses.is_dir():
                                if REEXTRACT==True:
                                    full_bandit_fix(ses.as_posix())
                                mouse_psy = pd.DataFrame()
                                mouse_psy['feedback'] = \
                                    1*(np.load(ses.joinpath('alf', '_ibl_trials.feedbackType.npy'))>0)[trial_start:trial_end]
                                mouse_psy['choices'] = \
                                    1*((-1*np.load(ses.joinpath('alf', '_ibl_trials.choice.npy')))>0)[trial_start:trial_end]
                                mouse_psy['laser_block'] = \
                                    np.load(ses.joinpath('alf', '_ibl_trials.opto_block.npy'))[trial_start:trial_end]
                                mouse_psy['probabilityLeft'] = \
                                    np.load(ses.joinpath('alf', '_ibl_trials.probabilityLeft.npy'))[trial_start:trial_end]
                                mouse_psy['laser'] = mouse_psy['feedback']*mouse_psy['laser_block']
                                mouse_psy['reward'] = mouse_psy['feedback']*(1*(mouse_psy['laser_block']==0))
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
    l = np.zeros([NS,NSESS,NT])
    p = np.zeros([NS,NSESS,NT])
    b = np.zeros([NS,NSESS,NT])
    tb = np.zeros([NS,NSESS,NT])
    for ns, mouse in enumerate(psy['mouse'].unique()):
        animal = psy.loc[psy['mouse']==mouse]
        counter=0
        for d, day in enumerate(animal['date'].unique()):
            day_s = animal.loc[animal['date']==day]
            for nsess, ses in enumerate(day_s['ses'].unique()):
                session = day_s.loc[day_s['ses']==ses]
                r[ns, counter, :len(session)] = session['reward']
                c[ns, counter, :len(session)] = session['choices']
                l[ns, counter, :len(session)] = session['laser']
                b[ns, counter, :len(session)] = session['laser_block']
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
                'b':b.astype(int), 'p':p.astype(float), 'tb':tb.astype(int),
               'NT_all':np.array(NT_all).astype(int), 'NSxNSESS':int(NSxNSESS),
               'sess_idx':np.array(sess_idx).astype(int)+1,
               'sub_idx':np.array(sub_idx).astype(int)+1,}

    return standata_recovery

def load_qdata_from_list(SESSION_LIST,  prefix='forgetting_'):
    psy = pd.DataFrame()
    for s in SESSION_LIST:
        ses = Path(s)
        if ses.joinpath('alf', prefix+'QR.npy').is_file():
            mouse_psy = pd.DataFrame()
            try:
                mouse_psy['DQwater'] =  np.load(ses.joinpath('alf', prefix+'QRreward.npy')) - \
                                        np.load(ses.joinpath('alf', prefix+'QLreward.npy'))
                mouse_psy['DQlaser'] = np.load(ses.joinpath('alf', prefix+'QRlaser.npy')) - \
                                        np.load(ses.joinpath('alf', prefix+'QLlaser.npy'))
            except: 
                mouse_psy['DQreward'] =  np.load(ses.joinpath('alf', prefix+'QRreward.npy')) - \
                                    np.load(ses.joinpath('alf', prefix+'QLreward.npy'))
            mouse_psy['choice_prediction'] = np.load(ses.joinpath('alf', prefix+'choice_prediction.npy'))
            try:                     
                mouse_psy['QLstay'] = np.load(ses.joinpath('alf', prefix+'QLstay.npy'))
                mouse_psy['QRstay'] = np.load(ses.joinpath('alf', prefix+'QRstay.npy'))
                mouse_psy['DQstay'] =  mouse_psy['QRstay']- mouse_psy['QLstay']
            except:
                print('No stay model')
            mouse_psy['ses'] = str(ses)[-14:]
            mouse_psy['mouse']= str(ses)[-21:-15]
            psy = pd.concat([psy, mouse_psy])
    return psy

def load_REINFORCE_from_list(SESSION_LIST,  prefix='forgetting_'):
    psy = pd.DataFrame()
    for s in SESSION_LIST:
        ses = Path(s)
        if ses.joinpath('alf', prefix+'reward.npy').is_file():
            mouse_psy = pd.DataFrame()
            try:
                mouse_psy['DQlaser'] = np.load(ses.joinpath('alf', prefix+'laser.npy'))
                mouse_psy['DQwater'] = np.load(ses.joinpath('alf', prefix+'water.npy'))
            except: 
                mouse_psy['DQreward'] =  np.load(ses.joinpath('alf', prefix+'reward.npy'))
            mouse_psy['choice_prediction'] = np.load(ses.joinpath('alf', prefix+'choice_prediction.npy'))
            try:                     
                mouse_psy['QLstay'] = np.load(ses.joinpath('alf', prefix+'QLstay.npy'))
                mouse_psy['QRstay'] = np.load(ses.joinpath('alf', prefix+'QRstay.npy'))
                mouse_psy['DQstay'] =  mouse_psy['QRstay']- mouse_psy['QLstay']
            except:
                print('No stay model')
            mouse_psy['ses'] = str(ses)[-14:]
            mouse_psy['mouse']= str(ses)[-21:-15]
            psy = pd.concat([psy, mouse_psy])
    return psy

def load_qdata_from_file(ROOT_FOLDER = '/Volumes/witten/Alex/Data/Subjects', prefix='QLearning_laserdecay_'):
    root_path = Path(ROOT_FOLDER)
    psy = pd.DataFrame()
    for animal in root_path.iterdir():
        if animal.is_dir():
            for day in animal.iterdir():
                if day.is_dir():
                    for ses in day.iterdir():
                        if ses.is_dir():
                            if ses.joinpath('alf', prefix+'water.npy').is_file():
                                mouse_psy = pd.DataFrame()
                                mouse_psy['DQwater'] = np.load(ses.joinpath('alf', prefix+'water.npy'))
                                mouse_psy['DQlaser'] = np.load(ses.joinpath('alf', prefix+'laser.npy'))
                                mouse_psy['choice_prediction'] = np.load(ses.joinpath('alf', prefix+'choice_prediction.npy'))
                                mouse_psy['QLstay'] = np.load(ses.joinpath('alf', prefix+'QLstay.npy'))
                                mouse_psy['QRstay'] = np.load(ses.joinpath('alf', prefix+'QRstay.npy'))
                                mouse_psy['DQstay'] =  mouse_psy['QRstay']- mouse_psy['QLstay']
                                mouse_psy['ses'] = str(ses)[-14:]
                                mouse_psy['mouse']= str(animal)[-6:]
                                psy = pd.concat([psy, mouse_psy])
                                try:
                                    mouse_psy['QL'] = np.load(ses.joinpath('alf', prefix+'QL.npy'))
                                    mouse_psy['QR'] = np.load(ses.joinpath('alf', prefix+'QR.npy'))
                                    mouse_psy['QLlaser'] = np.load(ses.joinpath('alf', prefix+'QLlaser.npy'))
                                    mouse_psy['QRlaser'] = np.load(ses.joinpath('alf', prefix+'QRlaser.npy'))
                                    mouse_psy['QLreward'] = np.load(ses.joinpath('alf', prefix+'QLreward.npy'))
                                    mouse_psy['QRreward'] = np.load(ses.joinpath('alf', prefix+'QRreward.npy'))
                                except:
                                    continue
                            else:
                                continue
                        else:
                            continue
                else:
                    continue
        else:
            continue
    return psy
def stan_data_to_df_reduced(standata_recovery,standata):
    r =  standata['r']
    c =  standata['c']
    tb = standata_recovery['tb']
    p = standata_recovery['p']
    sub_idx =  standata_recovery['sub_idx']-1
    sess_idx = standata_recovery['sess_idx']-1
    NSxNSESS = standata_recovery['NSxNSESS']
    NT_all = standata_recovery['NT_all']
    data = pd.DataFrame()
    for ms_i in np.arange(NSxNSESS):
        if NT_all[ms_i]>0:
                ses_data = pd.DataFrame()
                ses_data['choices'] = c[sub_idx[ms_i], sess_idx[ms_i],:int(NT_all[ms_i])]
                ses_data['water'] = r[sub_idx[ms_i], sess_idx[ms_i],:int(NT_all[ms_i])]
                ses_data['tb'] = tb[sub_idx[ms_i], sess_idx[ms_i],:int(NT_all[ms_i])]
                ses_data['probabilityLeft'] = p[sub_idx[ms_i], sess_idx[ms_i],:int(NT_all[ms_i])]
                ses_data['mouse'] = sub_idx[ms_i]
                ses_data['ses'] = sess_idx[ms_i]
                data = pd.concat([data,ses_data])
    return data


def stan_data_to_df(standata_recovery,standata):
    r =  standata['r']
    c =  standata['c']
    l =  standata['l']
    b =  standata_recovery['b']
    tb = standata_recovery['tb']
    p = standata_recovery['p']
    sub_idx =  standata_recovery['sub_idx']-1
    sess_idx = standata_recovery['sess_idx']-1
    NSxNSESS = standata_recovery['NSxNSESS']
    NT_all = standata_recovery['NT_all']
    data = pd.DataFrame()
    for ms_i in np.arange(NSxNSESS):
        if NT_all[ms_i]>0:
                ses_data = pd.DataFrame()
                ses_data['choices'] = c[sub_idx[ms_i], sess_idx[ms_i],:int(NT_all[ms_i])]
                ses_data['water'] = r[sub_idx[ms_i], sess_idx[ms_i],:int(NT_all[ms_i])]
                ses_data['laser'] = l[sub_idx[ms_i], sess_idx[ms_i],:int(NT_all[ms_i])]
                ses_data['laser_block'] = b[sub_idx[ms_i], sess_idx[ms_i],:int(NT_all[ms_i])]
                ses_data['tb'] = tb[sub_idx[ms_i], sess_idx[ms_i],:int(NT_all[ms_i])]
                ses_data['probabilityLeft'] = p[sub_idx[ms_i], sess_idx[ms_i],:int(NT_all[ms_i])]
                ses_data['mouse'] = sub_idx[ms_i]
                ses_data['ses'] = sess_idx[ms_i]
                data = pd.concat([data,ses_data])
    return data
def load_cmd_stan_output(folder):
    chains = glob.glob(folder + '/*.csv')
    output = pd.DataFrame()
    for c,i in enumerate(chains):
        chain = pd.read_csv(i
            , skiprows= \
            np.concatenate([np.arange(42), np.array([44,45,46,43]),
            np.arange(1547,1552)]))
        chain['chain'] = c+1
        output = pd.concat([output,chain])
    return output



def chains_2_summary(chain_folder, mode=True):
    output = load_cmd_stan_output(chain_folder)
    if mode==True:
        output = pd.DataFrame(output.apply(ss.mode).iloc[0,:]).reset_index()
    else:
        output = pd.DataFrame(output.mean()).reset_index()
    output.rename(columns={'index':'name', 0:'Mean'}, inplace=True)
    output.loc[(output['name'].str.contains('\d')),'name']=\
        output.loc[(output['name'].str.contains('\d')),'name']+']'
    output['name']=output['name'].str.replace('.','[')
    return output
def make_deltas(model_data):
    model_data['deltaQ'] = model_data['QR']-model_data['QL']
    model_data['Qwater'] = model_data['QRreward']-model_data['QLreward']
    try:
        model_data['Qstay'] = model_data['QRstay']-model_data['QLstay']
    except:
        print('No stay')
    model_data['Qlaser'] = model_data['QRlaser']-model_data['QLlaser']
    return model_data
def plot_params(saved_params, standata, save=False, output_path=None, phi_a=False):
    var_names = {'sides':'Bias', 'alphalaser_ses':'αLaser', 'alpha_ses':'αWater',
                'alphastay_ses':'αStay', 'laser_ses':'βLaser',
                'stay_ses':'βStay', 'beta_ses': 'βWater', 'alphaforgetting_ses':'αWaterForget',
                'alphalaserforgetting_ses':'αLaserForget',
                'dmlaser_ses':'βLaserLoss', 'dmwater_ses':'βWaterLoss',
                'laserstay_ses':'βLaserStay', 'alphalaserstay_ses': 'αStayLaser',
                'lapse_ses':'ε', 'balphalaser_ses':'αBLaser', 'balpha_ses':'αBWater',
                'balphastay_ses':'αBStay', 'alphalaserloss_ses':'αLaserLoss', 'alphaloss_ses':'αWaterLoss',
                'laserdecay_ses':'αLaserDecay', 'laserdecayloss_ses':'αLaserDecayLoss',
                'betalaserlossbase_ses':'βLaserLossBase', 'betalaserbase_ses':'βLaserBase', 'ep_ses':'βStay'}
    var_order = ['βWater','βLaser','βStay', 'βWaterLoss', 'βLaserLoss','βLaserStay','βLaserLossBase','βLaserBase',
                'αWater','αLaser','αStay','αWaterForget','αLaserForget','αWaterLoss','αLaserLoss','αStayLaser',
                'αBWater','αBLaser','αBStay','αLaserDecay','αLaserDecayLoss','ε','εStay','Bias']
    pal = ['dodgerblue','orange','gray', 'dodgerblue', 'orange','gray','orange','orange',
                'dodgerblue','orange','gray','dodgerblue','orange','dodgerblue','orange','gray',
                'dodgerblue','orange','gray','orange','orange','violet','gray','salmon']

    NSxNSESS = standata['NSxNSESS']
    NT_all = standata['NT_all']
    saved_params_new = pd.DataFrame()
    for ms_i in np.arange(NSxNSESS):
        if NT_all[ms_i]>0:
            saved_params_new = pd.concat([saved_params_new, saved_params.loc[(saved_params['name'].str.endswith('['+str(ms_i+1)+']'))]])
    saved_params = saved_params_new
    params=saved_params.loc[(saved_params['name'].str.endswith(']')) &
        (saved_params['name'].str.contains('s\[')),['name','Mean']]
    params['parameter']=params['name'].str.rstrip('[123456789102030]')
    params['parameter'] = params['parameter'].map(var_names)
    if phi_a==True:
        params.loc[(params['parameter'].str.contains('α')), 'Mean'] = \
                phi_approx(params.loc[(params['parameter'].str.contains('α')), 'Mean']/np.sqrt(2))
    sns.swarmplot(data=params, x='parameter', y='Mean',
                order=np.array(var_order)[np.isin(var_order, params['parameter'].unique())],
                color='k')
    sns.barplot(data=params, x='parameter', y='Mean', color='gray',
                order=np.array(var_order)[np.isin(var_order, params['parameter'].unique())],
                palette=np.array(pal)[np.isin(var_order, params['parameter'].unique())])
    plt.ylabel('Coefficient')
    plt.xticks(rotation=90)
    sns.despine()
    if save==True:
        plt.savefig(filepath+'/params.pdf')

def plot_params_reduced(saved_params, standata, save=False, filepath=None, phi_a=False):
    var_names = {'sides':'Bias', 'alphalaser_ses':'αReward', 'alpha_ses':'αReward',
                'alphastay_ses':'αStay', 'laser_ses':'βReward',
                'stay_ses':'βStay', 'beta_ses': 'βReward', 'alphaforgetting_ses':'αRewardForget',
                'alphalaserforgetting_ses':'αRewardForget',
                'dmlaser_ses':'βRewardLoss', 'dmwater_ses':'βRewardLoss',
                'laserstay_ses':'βRewardStay', 'alphalaserstay_ses': 'αStayReward',
                'lapse_ses':'ε', 'balphalaser_ses':'αBReward', 'balpha_ses':'αBReward',
                'balphastay_ses':'αBStay', 'alphalaserloss_ses':'αRewardLoss', 'alphaloss_ses':'αRewardLoss',
                'laserdecay_ses':'αRewardDecay', 'laserdecayloss_ses':'αRewardDecayLoss',
                'betalaserlossbase_ses':'βLaserLossBase', 'betalaserbase_ses':'βLaserBase', 'ep_ses':'βStay'}
    var_order = ['βReward','βLaser','βStay', 'βRewardLoss', 'βLaserLoss','βLaserStay','βLaserLossBase','βLaserBase',
                'αReward','αLaser','αStay','αRewardForget','αLaserForget','αRewardLoss','αLaserLoss','αStayLaser',
                'αBReward','αBLaser','αBStay','αLaserDecay','αLaserDecayLoss','ε','εStay','Bias']
    pal = ['green','orange','gray', 'green', 'orange','gray','orange','orange',
                'green','orange','gray','green','orange','green','orange','gray',
                'green','orange','gray','orange','orange','violet','gray','salmon']
    NSxNSESS = standata['NSxNSESS']
    NT_all = standata['NT_all']
    saved_params_new = pd.DataFrame()
    for ms_i in np.arange(NSxNSESS):
        if NT_all[ms_i]>0:
            saved_params_new = pd.concat([saved_params_new, saved_params.loc[(saved_params['name'].str.endswith('['+str(ms_i+1)+']'))]])
    saved_params = saved_params_new
    params=saved_params.loc[(saved_params['name'].str.endswith(']')) &
        (saved_params['name'].str.contains('s\[')),['name','Mean']]
    params['parameter']=params['name'].str.rstrip('[123456789102030]')
    params['parameter'] = params['parameter'].map(var_names)
    if phi_a==True:
        params.loc[(params['parameter'].str.contains('α')), 'Mean'] = \
                phi_approx(params.loc[(params['parameter'].str.contains('α')), 'Mean']/np.sqrt(2))
    sns.swarmplot(data=params, x='parameter', y='Mean',
                order=np.array(var_order)[np.isin(var_order, params['parameter'].unique())],
                color='k')
    sns.barplot(data=params, x='parameter', y='Mean', color='gray',
                order=np.array(var_order)[np.isin(var_order, params['parameter'].unique())],
                palette=np.array(pal)[np.isin(var_order, params['parameter'].unique())])
    plt.ylabel('Coefficient')
    plt.xticks(rotation=90)
    sns.despine()
    if save==True:
        plt.savefig(filepath+'/params.pdf')
    return params


def compare_two_models(saved_params, stan_data, save=False, output_path=None, phi_a=True):
        var_names = {'sides':'Bias', 'alphalaser_ses':'αLaser', 'alpha_ses':'αWater',
                    'alphastay_ses':'αStay', 'laser_ses':'βLaser',
                    'stay_ses':'βStay', 'beta_ses': 'βWater', 'alphaforgetting_ses':'αWaterForget',
                    'alphalaserforgetting_ses':'αLaserForget',
                    'dmlaser_ses':'βLaserLoss', 'dmwater_ses':'βWaterLoss',
                    'laserstay_ses':'βLaserStay', 'alphalaserstay_ses': 'αStayLaser',
                    'lapse_ses':'ε', 'balphalaser_ses':'αBLaser', 'balpha_ses':'αBWater',
                    'balphastay_ses':'αBStay'}
        var_order = ['βWater','βLaser','βStay', 'βWaterLoss', 'βLaserLoss','βLaserStay',
                    'αWater','αLaser','αStay','αWaterForget','αLaserForget','αStayLaser',
                    'αBWater','αBLaser','αBStay','ε','Bias']
        pal = ['dodgerblue','orange','gray', 'dodgerblue', 'orange','gray',
                    'dodgerblue','orange','gray','dodgerblue','orange','gray',
                    'dodgerblue','orange','gray','violet','salmon']
        NSxNSESS = standata['NSxNSESS']
        NT_all = standata['NT_all']
        saved_params_new = pd.DataFrame()
        for ms_i in np.arange(NSxNSESS):
            if NT_all[ms_i]>0:
                saved_params_new = pd.concat([saved_params_new, saved_params.loc[(saved_params['name'].str.endswith('['+str(ms_i+1)+']'))]])
        saved_params = saved_params_new
        params=saved_params.loc[(saved_params['name'].str.endswith(']')) &
            (saved_params['name'].str.contains('s\[')),['name','Mean','model_name']]
        params['parameter']=params['name'].str.rstrip('[123456789102030]')
        params['parameter'] = params['parameter'].map(var_names)
        if phi_a==True:
            params.loc[(params['parameter'].str.contains('α')), 'Mean'] = \
                    phi_approx(params.loc[(params['parameter'].str.contains('α')), 'Mean']/np.sqrt(2))
        sns.catplot(data=params, x='model_name', col='parameter' ,y='Mean', hue='name',kind='point',legend=False, color='k')
        plt.legend().remove()
        plt.ylabel('Coefficients')
        sns.despine()
        plt.tight_layout()
def plot_params_diff_learning(saved_params, stan_data, save=False, output_path=None, phi_a=True):
    var_names = {'sides':'Bias', 'alphalaser_ses':'αLaser', 'alpha_ses':'αWater',
                'alphastay_ses':'αStay', 'laser_ses':'βLaser',
                'stay_ses':'βStay', 'beta_ses': 'βWater', 'alphaforgetting_ses':'αWaterForget',
                'alphalaserforgetting_ses':'αLaserForget',
                'dmlaser_ses':'βLaserLoss', 'dmwater_ses':'βWaterLoss',
                'laserstay_ses':'βLaserStay', 'alphalaserstay_ses': 'αStayLaser',
                'lapse_ses':'ε', 'balphalaser_ses':'αBLaser', 'balpha_ses':'αBWater',
                'balphastay_ses':'αBStay'}
    var_order = ['βWater','βLaser','βStay', 'βWaterLoss', 'βLaserLoss','βLaserStay',
                'αWater','αLaser','αStay','αWaterForget','αLaserForget','αStayLaser',
                'αBWater','αBLaser','αBStay','ε','Bias']
    pal = ['dodgerblue','orange','gray', 'dodgerblue', 'orange','gray',
                'dodgerblue','orange','gray','dodgerblue','orange','gray',
                'dodgerblue','orange','gray','violet','salmon']
    NSxNSESS = standata['NSxNSESS']
    NT_all = standata['NT_all']
    saved_params_new = pd.DataFrame()
    for ms_i in np.arange(NSxNSESS):
        if NT_all[ms_i]>0:
            saved_params_new = pd.concat([saved_params_new, saved_params.loc[(saved_params['name'].str.endswith('['+str(ms_i+1)+']'))]])
    saved_params = saved_params_new
    params=saved_params.loc[(saved_params['name'].str.endswith(']')) &
        (saved_params['name'].str.contains('s\[')),['name','Mean']]
    params['parameter']=params['name'].str.rstrip('[123456789102030]')
    params['parameter'] = params['parameter'].map(var_names)
    params = params.loc[np.isin(params['parameter'], ['αWater','αLaser','αStay','αBWater','αBLaser','αBStay'])]
    params['type'] = 'End'
    params.loc[np.isin(params['parameter'], ['αBWater','αBLaser','αBStay']), 'type'] = 'Beggining'
    params.loc[params['parameter']=='αBWater','parameter'] = 'αWater'
    params.loc[params['parameter']=='αBLaser','parameter'] ='αLaser'
    params.loc[params['parameter']=='αBStay','parameter'] = 'αStay'

    if phi_a==True:
        params.loc[(params['parameter'].str.contains('α')), 'Mean'] = \
                phi_approx(params.loc[(params['parameter'].str.contains('α')), 'Mean']/np.sqrt(2))
    sns.barplot(data=params, x='parameter', y='Mean', hue='type',
                order=np.array(var_order)[np.isin(var_order, params['parameter'].unique())],
                palette='inferno')
    plt.ylabel('Coefficient')
    sns.despine()
    plt.ylim(0.3,0.6)
    if save==True:
        plt.savefig(filepath+'/params.pdf')
def plot_model_comparison(models_summary, save=False, output_path=None):
    sns.pointplot(data=models_summary, x='Model', y='Accuracy', color='r', hue=models_summary.index,
                scale=0.7, order=models_summary.groupby(['Model']).mean().sort_values('Accuracy').index)
    sns.barplot(data=models_summary, x='Model', y='Accuracy', color='gray',
                order=models_summary.groupby(['Model']).mean().sort_values('Accuracy').index)
    plt.xticks(rotation=45)
    plt.ylim(0.47,0.85)
    plt.legend().remove()
    sns.despine()
    plt.tight_layout()
    if save==True:
        plt.savefig(filepath+'/model_comparison.pdf')
def compare_accuracy_reinforce(reinforce_summary):
    sns.pointplot(data=reinforce_summary, x='Model', y='Accuracy', color='r', hue=reinforce_summary.index,
                scale=0.7, order=reinforce_summary.groupby(['Model']).mean().sort_values('Accuracy').index,
                palette='gray')
    plt.legend().remove()
    sns.barplot(data=reinforce_summary, x='Model', y='Accuracy', color='gray',
                order=reinforce_summary.groupby(['Model']).mean().sort_values('Accuracy').index)
    plt.ylim(0.5,0.85)
    sns.despine()
def check_model_behavior(sim_data):
    sim_data_with_rep = sim_data.reset_index().copy()
    sim_data_plotting = pd.DataFrame()
    for mouse in sim_data_with_rep['mouse'].unique():
        select_1 = sim_data_with_rep.loc[sim_data_with_rep['mouse']==mouse]
        for ses in select_1['ses'].unique():
            select2 = select_1.loc[select_1['ses']==ses]
            for i in np.where(select2['probabilityLeft'].diff()!=0)[0]:
                reps = select2.iloc[i-5:i].copy()
                reps['tb'] = np.arange(-5,0)
                if reps['probabilityLeft'].iloc[0]==0.1:
                    reps['probabilityLeft'] = 0.7
                else:
                    reps['probabilityLeft'] = 0.1
                sim_data_plotting = pd.concat([sim_data_plotting, reps])
    sim_data_plotting = pd.concat([sim_data_plotting, sim_data_with_rep])
    sns.lineplot(data=sim_data_plotting.reset_index(), x='tb', y= 'choices',
                  ci=68, hue='laser_block', err_style='bars', 
                  style='probabilityLeft', palette = ['dodgerblue','orange'])
    plt.legend().remove()
    plt.xlim(-5,15)
    plt.xlabel('Trials back')
    plt.vlines(0,0,1,linestyles='dashed', color='k')
    plt.ylabel('Fraction of Right Choices')
    sns.despine()
# Models
def reduced_uchida_model(standata,saved_params):
    r =  standata['r']
    c =  standata['c']
    sub_idx =  standata['sub_idx']-1
    sess_idx = standata['sess_idx']-1
    NSxNSESS = standata['NSxNSESS']
    NT_all = standata['NT_all']
    data = pd.DataFrame()
    for ms_i in np.arange(NSxNSESS):
        if NT_all[ms_i]>0:
            beta_mouse = float(saved_params.loc[saved_params['name']=='beta_ses['+str(ms_i+1)+']', 'Mean'])
            stay_mouse = float(saved_params.loc[saved_params['name']=='stay_ses['+str(ms_i+1)+']', 'Mean'])
            side_mouse = float(saved_params.loc[saved_params['name']=='sides['+str(ms_i+1)+']', 'Mean'])
            alpha = float(saved_params.loc[saved_params['name']=='alpha_ses['+str(ms_i+1)+']', 'Mean'])
            alphastay = float(saved_params.loc[saved_params['name']=='alphastay_ses['+str(ms_i+1)+']', 'Mean'])
            alphaloss = float(saved_params.loc[saved_params['name']=='alphaloss_ses['+str(ms_i+1)+']', 'Mean'])
            q = np.zeros(2)
            qloss = np.zeros(2)
            qstay = np.zeros(2)
            predicted_choices=[]
            QL=[]
            QR=[]
            QLloss=[]
            QRloss=[]           
            QLstay=[]
            QRstay=[]
            choices=[]
            QLgeneral=[]
            QRgeneral=[]
            for t in np.arange(NT_all[ms_i]):
                t = int(t)
                predicted_choice = inv_logit(side_mouse
                    + beta_mouse  * ((q[1] - qloss[1])-(q[0] - qloss[0]))
                    + stay_mouse * (qstay[1] - qstay[0]))
                choice = c[sub_idx[ms_i], sess_idx[ms_i], t]
                q[choice] = (1-alpha) * q[choice] + alpha * (r[sub_idx[ms_i], sess_idx[ms_i],t]) # Join r+l for reduced model
                qloss[choice] = (1-alphaloss) * qloss[choice] + alphaloss * (1 - r[sub_idx[ms_i], sess_idx[ms_i],t]) # Join r+l for reduced model               
                qstay = qstay * (1 - alphastay)
                qstay[choice] = qstay[choice] + alphastay

                # Store variables
                predicted_choices.append(predicted_choice)
                choices.append(choice)
                QL.append(q[0])
                QR.append(q[1])
                QLloss.append(qloss[0])
                QRloss.append(qloss[1])
                QLstay.append(qstay[0])
                QRstay.append(qstay[1])
                QLgeneral.append((beta_mouse*(q[0]-qloss[0]))+(stay_mouse*qstay[0]))
                QRgeneral.append((beta_mouse*(q[1]-qloss[1]))+(stay_mouse*qstay[1]))
            acc = []
            for i in np.arange(len(choices)):
                acc.append(1*(choices[i]==(1*(predicted_choices[i]>0.5))))
            #print(np.mean(acc))

            if NT_all[ms_i]!=0:
                ses_data = pd.DataFrame()
                ses_data['predicted_choice'] = predicted_choices
                ses_data['QL'] = QLgeneral
                ses_data['QR'] = QRgeneral
                ses_data['QLreward'] = np.array(QL) * beta_mouse 
                ses_data['QRreward'] =  np.array(QR) * beta_mouse
                ses_data['QLloss'] = np.array(QLloss) * beta_mouse 
                ses_data['QRloss'] =  np.array(QRloss) * beta_mouse
                ses_data['QLstay'] =  np.array(QLstay) * stay_mouse
                ses_data['QRstay'] =  np.array(QRstay) * stay_mouse
                ses_data['deltaQ'] = ses_data['QR'] - ses_data['QL']
                ses_data['choices'] = choices
                ses_data['mouse'] = sub_idx[ms_i]
                ses_data['ses'] = sess_idx[ms_i]
                ses_data['acc'] = np.mean(acc)
                ses_data['id'] = ses_data['mouse']*100 + ses_data['ses']
                data = pd.concat([data,ses_data])
    return data

def double_update_model(standata,saved_params):
    r =  standata['r']
    c =  standata['c']
    sub_idx =  standata['sub_idx']-1
    sess_idx = standata['sess_idx']-1
    NSxNSESS = standata['NSxNSESS']
    NT_all = standata['NT_all']
    data = pd.DataFrame()
    for ms_i in np.arange(NSxNSESS):
        if NT_all[ms_i]>0:
            beta_mouse = float(saved_params.loc[saved_params['name']=='beta_ses['+str(ms_i+1)+']', 'Mean'])
            stay_mouse = float(saved_params.loc[saved_params['name']=='stay_ses['+str(ms_i+1)+']', 'Mean'])
            side_mouse = float(saved_params.loc[saved_params['name']=='sides['+str(ms_i+1)+']', 'Mean'])
            alpha = float(saved_params.loc[saved_params['name']=='alpha_ses['+str(ms_i+1)+']', 'Mean'])
            alphastay = float(saved_params.loc[saved_params['name']=='alphastay_ses['+str(ms_i+1)+']', 'Mean'])
            alphaloss = float(saved_params.loc[saved_params['name']=='alphaloss_ses['+str(ms_i+1)+']', 'Mean'])
            q = np.zeros(2)
            qstay = np.zeros(2)
            predicted_choices=[]
            QL=[]
            QR=[]
            QLstay=[]
            QRstay=[]
            choices=[]
            QLgeneral=[]
            QRgeneral=[]
            for t in np.arange(NT_all[ms_i]):
                t = int(t)
                predicted_choice = inv_logit(side_mouse
                    + beta_mouse  * (q[1] - q[0])
                    + stay_mouse * (qstay[1] - qstay[0]))
                choice = c[sub_idx[ms_i], sess_idx[ms_i], t]
                unchoice = 1*(choice==0)
                q[unchoice] = q[unchoice] - alphaloss * (r[sub_idx[ms_i], sess_idx[ms_i],t]-q[choice]) 
                q[choice] = (1-alpha) * q[choice] + alpha * (r[sub_idx[ms_i], sess_idx[ms_i],t])
                qstay = qstay * (1 - alphastay)
                qstay[choice] = qstay[choice] + alphastay
                # Store variables
                predicted_choices.append(predicted_choice)
                choices.append(choice)
                QL.append(q[0])
                QR.append(q[1])
                QLstay.append(qstay[0])
                QRstay.append(qstay[1])
                QLgeneral.append((beta_mouse*q[0])+(stay_mouse*qstay[0]))
                QRgeneral.append((beta_mouse*q[1])+(stay_mouse*qstay[1]))
            acc = []
            for i in np.arange(len(choices)):
                acc.append(1*(choices[i]==(1*(predicted_choices[i]>0.5))))
            #print(np.mean(acc))

            if NT_all[ms_i]!=0:
                ses_data = pd.DataFrame()
                ses_data['predicted_choice'] = predicted_choices
                ses_data['QL'] = QLgeneral
                ses_data['QR'] = QRgeneral
                ses_data['QLreward'] = np.array(QL) * beta_mouse # Note this QL is the reward  - TODO change name
                ses_data['QRreward'] =  np.array(QR) * beta_mouse
                ses_data['QLstay'] =  np.array(QLstay) * stay_mouse
                ses_data['QRstay'] =  np.array(QRstay) * stay_mouse
                ses_data['deltaQ'] = ses_data['QR'] - ses_data['QL']
                ses_data['choices'] = choices
                ses_data['mouse'] = sub_idx[ms_i]
                ses_data['ses'] = sess_idx[ms_i]
                ses_data['acc'] = np.mean(acc)
                ses_data['id'] = ses_data['mouse']*100 + ses_data['ses']
                data = pd.concat([data,ses_data])

def simulate_double_update_model(standata_recovery,saved_params=None, fit=None, csv=True):
    NS =standata_recovery['NS']
    NSESS=standata_recovery['NSESS']
    NT =standata_recovery['NT']
    NT_all =standata_recovery['NT_all']
    r_sim = np.zeros([NS,NSESS,NT])
    c_sim = np.zeros([NS,NSESS,NT])
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
                    beta_mouse = float(saved_params.loc[saved_params['name']=='beta_ses['+str(ms_i+1)+']', 'Mean'])
                    stay_mouse = float(saved_params.loc[saved_params['name']=='stay_ses['+str(ms_i+1)+']', 'Mean'])
                    side_mouse = float(saved_params.loc[saved_params['name']=='sides['+str(ms_i+1)+']', 'Mean'])
                    alpha = float(saved_params.loc[saved_params['name']=='alpha_ses['+str(ms_i+1)+']', 'Mean'])
                    alphastay = float(saved_params.loc[saved_params['name']=='alphastay_ses['+str(ms_i+1)+']', 'Mean'])
                    alphaloss = float(saved_params.loc[saved_params['name']=='alphaloss_ses['+str(ms_i+1)+']', 'Mean'])
                else:
                    print('Error: Only csv version available at the moment!')

            tph = TrialParamHandler()
            q = np.zeros(2)
            qstay = np.zeros(2)
            predicted_choices=[]
            rewards = []
            outcome=[]
            side_blocks=[]
            trial_within_block=[]
            right_reward=[]
            left_reward=[]
            for t in np.arange(NT_all[ms_i]):
                t = int(t)
                p_choice = inv_logit(side_mouse
                    + beta_mouse  * (q[1] - q[0])
                    + stay_mouse * (qstay[1] - qstay[0]))
                choice =  np.random.choice(2,p= [1-p_choice,p_choice])
                unchoice = 1*(choice==0)
                tph.response_side_buffer.append(choice)  # Add choice to response buffer
                p_left= tph.stim_probability_left
                if choice==1:
                    f = tph.right_reward
                if choice==0:
                    f = tph.left_reward
                q[unchoice] = q[unchoice] - alphaloss * (f-q[choice]) 
                q[choice] = (1-alpha) * q[choice] + alpha * f
                qstay = qstay * (1 - alphastay)
                qstay[choice] = qstay[choice] + alphastay
            
                # Store variables
                rewards.append(f)
                outcome.append(f)
                side_blocks.append(p_left)
                predicted_choices.append(choice)
                trial_within_block.append(tph.block_trial_num)
                right_reward.append(tph.right_reward)
                left_reward.append(tph.left_reward)
                # Init next trial
                tph =  tph.next_trial()

            if NT_all[ms_i]!=0:
                c_sim[sub_idx[ms_i], sess_idx[ms_i], :int(NT_all[ms_i])]= predicted_choices
                r_sim[sub_idx[ms_i], sess_idx[ms_i], :int(NT_all[ms_i])] = rewards
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
                sim_data = pd.concat([sim_data,ses_data])
    return c_sim, r_sim, sim_data



def simulate_reduced_uchida_model(standata_recovery,saved_params=None, fit=None, csv=True):
    NS =standata_recovery['NS']
    NSESS=standata_recovery['NSESS']
    NT =standata_recovery['NT']
    NT_all =standata_recovery['NT_all']
    r_sim = np.zeros([NS,NSESS,NT])
    c_sim = np.zeros([NS,NSESS,NT])
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
                    beta_mouse = float(saved_params.loc[saved_params['name']=='beta_ses['+str(ms_i+1)+']', 'Mean'])
                    stay_mouse = float(saved_params.loc[saved_params['name']=='stay_ses['+str(ms_i+1)+']', 'Mean'])
                    side_mouse = float(saved_params.loc[saved_params['name']=='sides['+str(ms_i+1)+']', 'Mean'])
                    alpha = float(saved_params.loc[saved_params['name']=='alpha_ses['+str(ms_i+1)+']', 'Mean'])
                    alphastay = float(saved_params.loc[saved_params['name']=='alphastay_ses['+str(ms_i+1)+']', 'Mean'])
                    alphaloss = float(saved_params.loc[saved_params['name']=='alphaloss_ses['+str(ms_i+1)+']', 'Mean'])
                else:
                    print('Error: Only csv version available at the moment!')

            tph = TrialParamHandler()
            q = np.zeros(2)
            qstay = np.zeros(2)
            qloss = np.zeros(2)
            predicted_choices=[]
            rewards = []
            outcome=[]
            side_blocks=[]
            trial_within_block=[]
            right_reward=[]
            left_reward=[]
            for t in np.arange(NT_all[ms_i]):
                t = int(t)
                p_choice = inv_logit(side_mouse
                    + beta_mouse  * ((q[1] - qloss[1])-(q[0] - qloss[0]))
                    + stay_mouse * (qstay[1] - qstay[0]))
                choice =  np.random.choice(2,p= [1-p_choice,p_choice])
                tph.response_side_buffer.append(choice)  # Add choice to response buffer
                p_left= tph.stim_probability_left
                if choice==1:
                    f = tph.right_reward
                if choice==0:
                    f = tph.left_reward
                q[choice] = (1-alpha) * q[choice] + alpha * f
                qloss[choice] = (1-alphaloss) * qloss[choice] + alphaloss * (1 - f) # Join r+l for reduced model               
                qstay = qstay * (1 - alphastay)
                qstay[choice] = qstay[choice] + alphastay
            
                # Store variables
                rewards.append(f)
                outcome.append(f)
                side_blocks.append(p_left)
                predicted_choices.append(choice)
                trial_within_block.append(tph.block_trial_num)
                right_reward.append(tph.right_reward)
                left_reward.append(tph.left_reward)
                # Init next trial
                tph =  tph.next_trial()

            if NT_all[ms_i]!=0:
                c_sim[sub_idx[ms_i], sess_idx[ms_i], :int(NT_all[ms_i])]= predicted_choices
                r_sim[sub_idx[ms_i], sess_idx[ms_i], :int(NT_all[ms_i])] = rewards
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
                sim_data = pd.concat([sim_data,ses_data])
    return c_sim, r_sim, sim_data

def q_learning_model_reduced(standata,saved_params=None):
    r =  standata['r']
    c =  standata['c']
    l =  standata['l']
    sub_idx =  standata['sub_idx']-1
    sess_idx = standata['sess_idx']-1
    NSxNSESS = standata['NSxNSESS']
    NT_all = standata['NT_all']
    data = pd.DataFrame()
    for ms_i in np.arange(NSxNSESS):
        if NT_all[ms_i]>0:
            beta_mouse = float(saved_params.loc[saved_params['name']=='beta_ses['+str(ms_i+1)+']', 'Mean'])
            side_mouse = float(saved_params.loc[saved_params['name']=='sides['+str(ms_i+1)+']', 'Mean'])
            alpha = float(saved_params.loc[saved_params['name']=='alpha_ses['+str(ms_i+1)+']', 'Mean'])
            q = np.zeros(2)
            predicted_choices=[]
            QL=[]
            QR=[]
            choices=[]
            QLgeneral=[]
            QRgeneral=[]
            for t in np.arange(NT_all[ms_i]):
                t = int(t)
                predicted_choice = inv_logit(side_mouse
                    + beta_mouse  * (q[1] - q[0]))
                choice = c[sub_idx[ms_i], sess_idx[ms_i], t]
                q[choice] = (1-alpha) * q[choice] + alpha * (r[sub_idx[ms_i], sess_idx[ms_i],t]+
                                                             l[sub_idx[ms_i], sess_idx[ms_i],t]) # Join r+l for reduced model
                # Store variables
                predicted_choices.append(predicted_choice)
                choices.append(choice)
                QL.append(q[0])
                QR.append(q[1])
                QLgeneral.append((beta_mouse*q[0]))
                QRgeneral.append((beta_mouse*q[1]))
            acc = []
            for i in np.arange(len(choices)):
                acc.append(1*(choices[i]==(1*(predicted_choices[i]>0.5))))
            #print(np.mean(acc))

            if NT_all[ms_i]!=0:
                ses_data = pd.DataFrame()
                ses_data['predicted_choice'] = predicted_choices
                ses_data['QL'] = QLgeneral
                ses_data['QR'] = QRgeneral
                ses_data['QLreward'] = np.array(QL) * beta_mouse # Note this QL is the reward  - TODO change name
                ses_data['QRreward'] =  np.array(QR) * beta_mouse
                ses_data['deltaQ'] = ses_data['QR'] - ses_data['QL']
                ses_data['choices'] = choices
                ses_data['mouse'] = sub_idx[ms_i]
                ses_data['ses'] = sess_idx[ms_i]
                ses_data['acc'] = np.mean(acc)
                data = pd.concat([data,ses_data])
    return data

def pearce_hall_model(standata,saved_params=None, fit=None, csv=True):
    if saved_params is not None:
        r =  standata['r']
        c =  standata['c']
        l =  standata['l']
        sub_idx =  standata['sub_idx']-1
        sess_idx = standata['sess_idx']-1
        NSxNSESS = standata['NSxNSESS']
        NT_all = standata['NT_all']
    else:
        r =  fit.data['r']
        c =  fit.data['c']
        l =  fit.data['l']
        sub_idx =  fit.data['sub_idx']-1
        sess_idx = fit.data['sess_idx']-1
    data = pd.DataFrame()

    for ms_i in np.arange(NSxNSESS):
        if NT_all[ms_i]>0:
            if saved_params is not None:
                if ms_i==0:
                    print('Using saved parameters')
                if csv==True:
                    beta_mouse = float(saved_params.loc[saved_params['name']=='beta_ses['+str(ms_i+1)+']', 'Mean'])
                    stay_mouse = float(saved_params.loc[saved_params['name']=='stay_ses['+str(ms_i+1)+']', 'Mean'])
                    side_mouse = float(saved_params.loc[saved_params['name']=='sides['+str(ms_i+1)+']', 'Mean'])
                    laser_mouse = float(saved_params.loc[saved_params['name']=='laser_ses['+str(ms_i+1)+']', 'Mean'])
                    alphastay = float(saved_params.loc[saved_params['name']=='alphastay_ses['+str(ms_i+1)+']', 'Mean'])
                    eta = float(saved_params.loc[saved_params['name']=='eta_ses['+str(ms_i+1)+']', 'Mean'])
                    kappa_laser_ses = float(saved_params.loc[saved_params['name']=='kappa_laser_ses['+str(ms_i+1)+']', 'Mean'])
                    kappa_ses = float(saved_params.loc[saved_params['name']=='kappa_ses['+str(ms_i+1)+']', 'Mean'])
                else:
                    print('Error: Only csv version available at the moment!')
            q = np.zeros(2)
            qstay = np.zeros(2)
            qlaser = np.zeros(2)
            predicted_choices=[]
            QL=[]
            QR=[]
            QLstay=[]
            QRstay=[]
            QLlaser=[]
            QRlaser=[]
            choices=[]
            QLgeneral=[]
            QRgeneral=[]
            alpha_laser=1
            alpha_water=1
            for t in np.arange(NT_all[ms_i]):
                t = int(t)
                predicted_choice = inv_logit(side_mouse
                    + beta_mouse  * (q[1] - q[0])
                    + stay_mouse * (qstay[1] - qstay[0])
                    + laser_mouse * (qlaser[1] - qlaser[0]))
                choice = c[sub_idx[ms_i], sess_idx[ms_i], t]

                delta_water =  r[sub_idx[ms_i], sess_idx[ms_i],t] - q[choice]
                delta_laser =  l[sub_idx[ms_i], sess_idx[ms_i],t] - qlaser[choice]
                q[choice] = q[choice] + kappa_ses*alpha_water*delta_water
                qlaser[choice] = qlaser[choice] + kappa_laser_ses*alpha_laser*delta_laser
                qstay = qstay * (1 - alphastay)
                qstay[choice] = qstay[choice] + alphastay
                alpha_water = (eta * abs(delta_water)) + ((1-eta)*alpha_water)
                alpha_laser = (eta * abs(delta_laser)) + ((1-eta)*alpha_laser)

                # Store variables
                predicted_choices.append(predicted_choice)
                choices.append(choice)
                QL.append(q[0])
                QR.append(q[1])
                QLstay.append(qstay[0])
                QRstay.append(qstay[1])
                QLlaser.append(qlaser[0])
                QRlaser.append(qlaser[1])
                QLgeneral.append((beta_mouse*q[0])+(laser_mouse*qlaser[0])+(stay_mouse*qstay[0]))
                QRgeneral.append((beta_mouse*q[1])+(laser_mouse*qlaser[1])+(stay_mouse*qstay[1]))
            acc = []
            for i in np.arange(len(choices)):
                acc.append(1*(choices[i]==(1*(predicted_choices[i]>0.5))))
            print(np.mean(acc))

            if NT_all[ms_i]!=0:
                ses_data = pd.DataFrame()
                ses_data['predicted_choice'] = predicted_choices
                ses_data['QL'] = QLgeneral
                ses_data['QR'] = QRgeneral
                ses_data['QLreward'] = np.array(QL) * beta_mouse # Note this QL is the reward  - TODO change name
                ses_data['QRreward'] =  np.array(QR) * beta_mouse
                ses_data['QLstay'] =  np.array(QLstay) * stay_mouse
                ses_data['QRstay'] =  np.array(QRstay) * stay_mouse
                ses_data['QLlaser'] =  np.array(QLlaser) * laser_mouse
                ses_data['QRlaser'] =  np.array(QRlaser) * laser_mouse
                ses_data['deltaQ'] = ses_data['QR'] - ses_data['QL']
                ses_data['choices'] = choices
                ses_data['mouse'] = sub_idx[ms_i]
                ses_data['ses'] = sess_idx[ms_i]
                ses_data['acc'] = np.mean(acc)
                data = pd.concat([data,ses_data])
    return make_deltas(data)
def q_learning_model_asymmetric(standata,saved_params=None, fit=None, csv=True):
    if saved_params is not None:
        r =  standata['r']
        c =  standata['c']
        l =  standata['l']
        sub_idx =  standata['sub_idx']-1
        sess_idx = standata['sess_idx']-1
        NSxNSESS = standata['NSxNSESS']
        NT_all = standata['NT_all']
    else:
        r =  fit.data['r']
        c =  fit.data['c']
        l =  fit.data['l']
        sub_idx =  fit.data['sub_idx']-1
        sess_idx = fit.data['sess_idx']-1
    data = pd.DataFrame()

    for ms_i in np.arange(NSxNSESS):
        if NT_all[ms_i]>0:
            if saved_params is not None:
                if ms_i==0:
                    print('Using saved parameters')
                if csv==True:
                    beta_mouse = float(saved_params.loc[saved_params['name']=='beta_ses['+str(ms_i+1)+']', 'Mean'])
                    stay_mouse = float(saved_params.loc[saved_params['name']=='stay_ses['+str(ms_i+1)+']', 'Mean'])
                    side_mouse = float(saved_params.loc[saved_params['name']=='sides['+str(ms_i+1)+']', 'Mean'])
                    laser_mouse = float(saved_params.loc[saved_params['name']=='laser_ses['+str(ms_i+1)+']', 'Mean'])
                    alpha = float(saved_params.loc[saved_params['name']=='alpha_ses['+str(ms_i+1)+']', 'Mean'])
                    alphalaser = float(saved_params.loc[saved_params['name']=='alphalaser_ses['+str(ms_i+1)+']', 'Mean'])
                    alphaloss = float(saved_params.loc[saved_params['name']=='alphaloss_ses['+str(ms_i+1)+']', 'Mean'])
                    alphalaserloss = float(saved_params.loc[saved_params['name']=='alphalaserloss_ses['+str(ms_i+1)+']', 'Mean'])
                    alphastay = float(saved_params.loc[saved_params['name']=='alphastay_ses['+str(ms_i+1)+']', 'Mean'])
                else:
                    print('Error: Only csv version available at the moment!')
            q = np.zeros(2)
            qstay = np.zeros(2)
            qlaser = np.zeros(2)
            predicted_choices=[]
            QL=[]
            QR=[]
            QLstay=[]
            QRstay=[]
            QLlaser=[]
            QRlaser=[]
            choices=[]
            QLgeneral=[]
            QRgeneral=[]
            for t in np.arange(NT_all[ms_i]):
                t = int(t)
                predicted_choice = inv_logit(side_mouse
                    + beta_mouse  * (q[1] - q[0])
                    + stay_mouse * (qstay[1] - qstay[0])
                    + laser_mouse * (qlaser[1] - qlaser[0]))   
                choice = c[sub_idx[ms_i], sess_idx[ms_i], t]
                rt = r[sub_idx[ms_i], sess_idx[ms_i],t]
                lt = l[sub_idx[ms_i], sess_idx[ms_i],t]
                if rt==1:
                    q[choice] = (1-alpha) * q[choice] + alpha * r[sub_idx[ms_i], sess_idx[ms_i],t]
                if rt==0:
                    q[choice] = (1-alphaloss) * q[choice] + alphaloss * r[sub_idx[ms_i], sess_idx[ms_i],t]
                if lt==1:
                    qlaser[choice] = (1-alphalaser) * qlaser[choice] + alphalaser * l[sub_idx[ms_i], sess_idx[ms_i],t]
                if lt==0:
                    qlaser[choice] = (1-alphalaserloss) * qlaser[choice] + alphalaserloss * l[sub_idx[ms_i], sess_idx[ms_i],t]
                qstay = qstay * (1 - alphastay)
                qstay[choice] = qstay[choice] + alphastay
                # Store variables
                predicted_choices.append(predicted_choice)
                choices.append(choice)
                QL.append(q[0])
                QR.append(q[1])
                QLstay.append(qstay[0])
                QRstay.append(qstay[1])
                QLlaser.append(qlaser[0])
                QRlaser.append(qlaser[1])
                QLgeneral.append((beta_mouse*q[0])+(laser_mouse*qlaser[0])+(stay_mouse*qstay[0]))
                QRgeneral.append((beta_mouse*q[1])+(laser_mouse*qlaser[1])+(stay_mouse*qstay[1]))
            acc = []
            for i in np.arange(len(choices)):
                acc.append(1*(choices[i]==(1*(predicted_choices[i]>0.5))))
            print(np.mean(acc))

            if NT_all[ms_i]!=0:
                ses_data = pd.DataFrame()
                ses_data['predicted_choice'] = predicted_choices
                ses_data['QL'] = QLgeneral
                ses_data['QR'] = QRgeneral
                ses_data['QLreward'] = np.array(QL) * beta_mouse # Note this QL is the reward  - TODO change name
                ses_data['QRreward'] =  np.array(QR) * beta_mouse
                ses_data['QLstay'] =  np.array(QLstay) * stay_mouse
                ses_data['QRstay'] =  np.array(QRstay) * stay_mouse
                ses_data['QLlaser'] =  np.array(QLlaser) * laser_mouse
                ses_data['QRlaser'] =  np.array(QRlaser) * laser_mouse
                ses_data['deltaQ'] = ses_data['QR'] - ses_data['QL']
                ses_data['choices'] = choices
                ses_data['mouse'] = sub_idx[ms_i]
                ses_data['ses'] = sess_idx[ms_i]
                ses_data['acc'] = np.mean(acc)
                data = pd.concat([data,ses_data])
    return make_deltas(data)
def q_learning_model_asymmetric_nostay(standata,saved_params=None, fit=None, csv=True):
    if saved_params is not None:
        r =  standata['r']
        c =  standata['c']
        l =  standata['l']
        sub_idx =  standata['sub_idx']-1
        sess_idx = standata['sess_idx']-1
        NSxNSESS = standata['NSxNSESS']
        NT_all = standata['NT_all']
    else:
        r =  fit.data['r']
        c =  fit.data['c']
        l =  fit.data['l']
        sub_idx =  fit.data['sub_idx']-1
        sess_idx = fit.data['sess_idx']-1
    
    data = pd.DataFrame()

    for ms_i in np.arange(NSxNSESS):
        if NT_all[ms_i]>0:
            if saved_params is not None:
                if ms_i==0:
                    print('Using saved parameters')
                if csv==True:
                    beta_mouse = float(saved_params.loc[saved_params['name']=='beta_ses['+str(ms_i+1)+']', 'Mean'])
                    side_mouse = float(saved_params.loc[saved_params['name']=='sides['+str(ms_i+1)+']', 'Mean'])
                    laser_mouse = float(saved_params.loc[saved_params['name']=='laser_ses['+str(ms_i+1)+']', 'Mean'])
                    alpha = float(saved_params.loc[saved_params['name']=='alpha_ses['+str(ms_i+1)+']', 'Mean'])
                    alphalaser = float(saved_params.loc[saved_params['name']=='alphalaser_ses['+str(ms_i+1)+']', 'Mean'])
                    alphaloss = float(saved_params.loc[saved_params['name']=='alphaloss_ses['+str(ms_i+1)+']', 'Mean'])
                    alphalaserloss = float(saved_params.loc[saved_params['name']=='alphalaserloss_ses['+str(ms_i+1)+']', 'Mean'])
                else:
                    print('Error: Only csv version available at the moment!')
            q = np.zeros(2)
            qlaser = np.zeros(2)
            predicted_choices=[]
            QL=[]
            QR=[]
            QLlaser=[]
            QRlaser=[]
            choices=[]
            QLgeneral=[]
            QRgeneral=[]
            for t in np.arange(NT_all[ms_i]):
                t = int(t)
                predicted_choice = inv_logit(side_mouse
                    + beta_mouse  * (q[1] - q[0])
                    + laser_mouse * (qlaser[1] - qlaser[0]))    
                choice = c[sub_idx[ms_i], sess_idx[ms_i], t]    
                rt = r[sub_idx[ms_i], sess_idx[ms_i],t]
                lt = l[sub_idx[ms_i], sess_idx[ms_i],t]
                if rt==1:
                    q[choice] = (1-alpha) * q[choice] + alpha * r[sub_idx[ms_i], sess_idx[ms_i],t]
                if rt==0:
                    q[choice] = (1-alphaloss) * q[choice] + alphaloss * r[sub_idx[ms_i], sess_idx[ms_i],t]
                if lt==1:
                    qlaser[choice] = (1-alphalaser) * qlaser[choice] + alphalaser * l[sub_idx[ms_i], sess_idx[ms_i],t]
                if lt==0:
                    qlaser[choice] = (1-alphalaserloss) * qlaser[choice] + alphalaserloss * l[sub_idx[ms_i], sess_idx[ms_i],t]
                # Store variables
                predicted_choices.append(predicted_choice)
                choices.append(choice)
                QL.append(q[0])
                QR.append(q[1])
                QLlaser.append(qlaser[0])
                QRlaser.append(qlaser[1])
                QLgeneral.append((beta_mouse*q[0])+(laser_mouse*qlaser[0]))
                QRgeneral.append((beta_mouse*q[1])+(laser_mouse*qlaser[1]))
            acc = []
            for i in np.arange(len(choices)):
                acc.append(1*(choices[i]==(1*(predicted_choices[i]>0.5))))
            print(np.mean(acc))

            if NT_all[ms_i]!=0:
                ses_data = pd.DataFrame()
                ses_data['predicted_choice'] = predicted_choices
                ses_data['QL'] = QLgeneral
                ses_data['QR'] = QRgeneral
                ses_data['QLreward'] = np.array(QL) * beta_mouse # Note this QL is the reward  - TODO change name
                ses_data['QRreward'] =  np.array(QR) * beta_mouse
                ses_data['QLlaser'] =  np.array(QLlaser) * laser_mouse
                ses_data['QRlaser'] =  np.array(QRlaser) * laser_mouse
                ses_data['deltaQ'] = ses_data['QR'] - ses_data['QL']
                ses_data['choices'] = choices
                ses_data['mouse'] = sub_idx[ms_i]
                ses_data['ses'] = sess_idx[ms_i]
                ses_data['acc'] = np.mean(acc)
                data = pd.concat([data,ses_data])
    return make_deltas(data)

def q_learning_model_reduced_forgetting(standata,saved_params=None):
    r =  standata['r']
    c =  standata['c']
    sub_idx =  standata['sub_idx']-1
    sess_idx = standata['sess_idx']-1
    NSxNSESS = standata['NSxNSESS']
    NT_all = standata['NT_all']
    data = pd.DataFrame()
    for ms_i in np.arange(NSxNSESS):
        if NT_all[ms_i]>0:
            beta_mouse = float(saved_params.loc[saved_params['name']=='beta_ses['+str(ms_i+1)+']', 'Mean'])
            side_mouse = float(saved_params.loc[saved_params['name']=='sides['+str(ms_i+1)+']', 'Mean'])
            alpha = float(saved_params.loc[saved_params['name']=='alpha_ses['+str(ms_i+1)+']', 'Mean'])
            forget = float(saved_params.loc[saved_params['name']=='alphaforgetting_ses['+str(ms_i+1)+']', 'Mean'])
            q = np.zeros(2)
            predicted_choices=[]
            QL=[]
            QR=[]
            choices=[]
            QLgeneral=[]
            QRgeneral=[]
            for t in np.arange(NT_all[ms_i]):
                t = int(t)
                predicted_choice = inv_logit(side_mouse
                    + beta_mouse  * (q[1] - q[0]))
                choice = c[sub_idx[ms_i], sess_idx[ms_i], t]
                unchoice = 1*(choice==0)
                q[choice] = (1-alpha) * q[choice] + alpha * (r[sub_idx[ms_i], sess_idx[ms_i],t]) # Join r+l for reduced model
                q[unchoice] =  q[unchoice] - forget* q[unchoice]
                # Store variables
                predicted_choices.append(predicted_choice)
                choices.append(choice)
                QL.append(q[0])
                QR.append(q[1])
                QLgeneral.append((beta_mouse*q[0]))
                QRgeneral.append((beta_mouse*q[1]))
            acc = []
            for i in np.arange(len(choices)):
                acc.append(1*(choices[i]==(1*(predicted_choices[i]>0.5))))
            #print(np.mean(acc))

            if NT_all[ms_i]!=0:
                ses_data = pd.DataFrame()
                ses_data['predicted_choice'] = predicted_choices
                ses_data['QL'] = QLgeneral
                ses_data['QR'] = QRgeneral
                ses_data['QLreward'] = np.array(QL) * beta_mouse # Note this QL is the reward  - TODO change name
                ses_data['QRreward'] =  np.array(QR) * beta_mouse
                ses_data['deltaQ'] = ses_data['QR'] - ses_data['QL']
                ses_data['choices'] = choices
                ses_data['mouse'] = sub_idx[ms_i]
                ses_data['ses'] = sess_idx[ms_i]
                ses_data['acc'] = np.mean(acc)
                ses_data['id'] = ses_data['mouse']*100 + ses_data['ses']
                data = pd.concat([data,ses_data])
    return data


def q_learning_model_reduced_stay_forgetting(standata,saved_params=None):
    r =  standata['r']
    c =  standata['c']
    sub_idx =  standata['sub_idx']-1
    sess_idx = standata['sess_idx']-1
    NSxNSESS = standata['NSxNSESS']
    NT_all = standata['NT_all']
    data = pd.DataFrame()
    for ms_i in np.arange(NSxNSESS):
        if NT_all[ms_i]>0:
            beta_mouse = float(saved_params.loc[saved_params['name']=='beta_ses['+str(ms_i+1)+']', 'Mean'])
            stay_mouse = float(saved_params.loc[saved_params['name']=='stay_ses['+str(ms_i+1)+']', 'Mean'])
            side_mouse = float(saved_params.loc[saved_params['name']=='sides['+str(ms_i+1)+']', 'Mean'])
            alpha = float(saved_params.loc[saved_params['name']=='alpha_ses['+str(ms_i+1)+']', 'Mean'])
            alphastay = float(saved_params.loc[saved_params['name']=='alphastay_ses['+str(ms_i+1)+']', 'Mean'])
            forget = float(saved_params.loc[saved_params['name']=='alphaforgetting_ses['+str(ms_i+1)+']', 'Mean'])
            q = np.zeros(2)
            qstay = np.zeros(2)
            V = 0
            predicted_choices=[]
            QL=[]
            QR=[]
            QLstay=[]
            QRstay=[]
            choices=[]
            QLgeneral=[]
            QRgeneral=[]
            QRew = []
            for t in np.arange(NT_all[ms_i]):
                t = int(t)
                predicted_choice = inv_logit(side_mouse
                    + beta_mouse * (q[1] - q[0])
                    + stay_mouse * (qstay[1] - qstay[0]))
                choice = c[sub_idx[ms_i], sess_idx[ms_i], t]
                unchoice = 1*(choice==0)
                q[choice] = (1-alpha) * q[choice] + alpha * (r[sub_idx[ms_i], sess_idx[ms_i],t]) # Join r+l for reduced model
                q[unchoice] =  q[unchoice] - forget* q[unchoice]
                qstay = qstay * (1 - alphastay)
                qstay[choice] = qstay[choice] + alphastay
                V = V + alpha * (r[sub_idx[ms_i], sess_idx[ms_i],t] - V)
                # Store variables
                predicted_choices.append(predicted_choice)
                choices.append(choice)
                QL.append(q[0])
                QR.append(q[1])
                QRew.append(V)
                QLstay.append(qstay[0])
                QRstay.append(qstay[1])
                QLgeneral.append((beta_mouse*q[0])+(stay_mouse*qstay[0]))
                QRgeneral.append((beta_mouse*q[1])+(stay_mouse*qstay[1]))
            acc = []
            for i in np.arange(len(choices)):
                acc.append(1*(choices[i]==(1*(predicted_choices[i]>0.5))))
            #print(np.mean(acc))

            if NT_all[ms_i]!=0:
                ses_data = pd.DataFrame()
                ses_data['predicted_choice'] = predicted_choices
                ses_data['QL'] = QLgeneral
                ses_data['QR'] = QRgeneral
                ses_data['V'] = QRew
                ses_data['QLreward'] = np.array(QL) * beta_mouse # Note this QL is the reward  - TODO change name
                ses_data['QRreward'] =  np.array(QR) * beta_mouse
                ses_data['QLstay'] =  np.array(QLstay) * stay_mouse
                ses_data['QRstay'] =  np.array(QRstay) * stay_mouse
                ses_data['deltaQ'] = ses_data['QR'] - ses_data['QL']
                ses_data['choices'] = choices
                ses_data['mouse'] = sub_idx[ms_i]
                ses_data['ses'] = sess_idx[ms_i]
                ses_data['acc'] = np.mean(acc)
                ses_data['id'] = ses_data['mouse']*100 + ses_data['ses']
                ses_data['beta_mouse'] = beta_mouse
                ses_data['stay_mouse'] = stay_mouse
                ses_data['side_mouse'] = side_mouse
                ses_data['alpha'] = alpha
                ses_data['alphastay'] = alphastay
                ses_data['forget'] = forget
                data = pd.concat([data,ses_data])
    return data


            
def q_learning_model_reduced_stay_forgetting_variable_alpha(standata,saved_params=None):
    r =  standata['r']
    c =  standata['c']
    sub_idx =  standata['sub_idx']-1
    sess_idx = standata['sess_idx']-1
    NSxNSESS = standata['NSxNSESS']
    NT_all = standata['NT_all']
    data = pd.DataFrame()
    for ms_i in np.arange(NSxNSESS):
        if NT_all[ms_i]>0:
            beta_mouse = float(saved_params.loc[saved_params['name']=='beta_ses['+str(ms_i+1)+']', 'Mean'])
            stay_mouse = float(saved_params.loc[saved_params['name']=='stay_ses['+str(ms_i+1)+']', 'Mean'])
            side_mouse = float(saved_params.loc[saved_params['name']=='sides['+str(ms_i+1)+']', 'Mean'])
            alpha = float(saved_params.loc[saved_params['name']=='alpha_ses['+str(ms_i+1)+']', 'Mean'])
            alphastay = float(saved_params.loc[saved_params['name']=='alphastay_ses['+str(ms_i+1)+']', 'Mean'])
            forget = float(saved_params.loc[saved_params['name']=='alphaforgetting_ses['+str(ms_i+1)+']', 'Mean'])
            eta = float(saved_params.loc[saved_params['name']=='eta_ses['+str(ms_i+1)+']', 'Mean'])

            q = np.zeros(2)
            qstay = np.zeros(2)
            V = 0
            predicted_choices=[]
            QL=[]
            QR=[]
            QLstay=[]
            QRstay=[]
            choices=[]
            QLgeneral=[]
            QRgeneral=[]
            QRew = []
            alphas_t = []
            var_alpha = 0.5

            for t in np.arange(NT_all[ms_i]):
                t = int(t)
                predicted_choice = inv_logit(side_mouse
                    + beta_mouse * (q[1] - q[0])
                    + stay_mouse * (qstay[1] - qstay[0]))
                choice = c[sub_idx[ms_i], sess_idx[ms_i], t]
                unchoice = 1*(choice==0)
                delta = r[sub_idx[ms_i], sess_idx[ms_i],t] - q[choice]
                q[choice] = q[choice] + alpha * alpha * var_alpha * delta
                q[unchoice] =  q[unchoice] - forget* q[unchoice]
                qstay = qstay * (1 - alphastay)
                qstay[choice] = qstay[choice] + alphastay
                var_alpha = (eta * abs(delta)) + ((1-eta)*var_alpha)
                V = V + alpha * var_alpha * (r[sub_idx[ms_i], sess_idx[ms_i],t] - V)
                # Store variables
                predicted_choices.append(predicted_choice)
                choices.append(choice)
                QL.append(q[0])
                QR.append(q[1])
                QRew.append(V)
                QLstay.append(qstay[0])
                QRstay.append(qstay[1])
                QLgeneral.append((beta_mouse*q[0])+(stay_mouse*qstay[0]))
                QRgeneral.append((beta_mouse*q[1])+(stay_mouse*qstay[1]))
                alphas_t.append(var_alpha)
            acc = []
            for i in np.arange(len(choices)):
                acc.append(1*(choices[i]==(1*(predicted_choices[i]>0.5))))
            #print(np.mean(acc))

            if NT_all[ms_i]!=0:
                ses_data = pd.DataFrame()
                ses_data['predicted_choice'] = predicted_choices
                ses_data['QL'] = QLgeneral
                ses_data['QR'] = QRgeneral
                ses_data['V'] = QRew
                ses_data['QLreward'] = np.array(QL) * beta_mouse # Note this QL is the reward  - TODO change name
                ses_data['QRreward'] =  np.array(QR) * beta_mouse
                ses_data['QLstay'] =  np.array(QLstay) * stay_mouse
                ses_data['QRstay'] =  np.array(QRstay) * stay_mouse
                ses_data['deltaQ'] = ses_data['QR'] - ses_data['QL']
                ses_data['choices'] = choices
                ses_data['mouse'] = sub_idx[ms_i]
                ses_data['ses'] = sess_idx[ms_i]
                ses_data['acc'] = np.mean(acc)
                ses_data['id'] = ses_data['mouse']*100 + ses_data['ses']
                ses_data['beta_mouse'] = beta_mouse
                ses_data['stay_mouse'] = stay_mouse
                ses_data['side_mouse'] = side_mouse
                ses_data['alpha'] = alpha
                ses_data['var_alpha'] = alphas_t
                ses_data['effective_alpha'] = ses_data['alpha'] * ses_data['var_alpha']
                ses_data['alphastay'] = alphastay
                ses_data['forget'] = forget
                data = pd.concat([data,ses_data])
    return data

def q_learning_model_reduced_stay_forgetting_laser(standata,saved_params=None):
    r =  standata['r']
    c =  standata['c']
    l =  standata['l']
    sub_idx =  standata['sub_idx']-1
    sess_idx = standata['sess_idx']-1
    NSxNSESS = standata['NSxNSESS']
    NT_all = standata['NT_all']
    data = pd.DataFrame()
    for ms_i in np.arange(NSxNSESS):
        if NT_all[ms_i]>0:
            beta_mouse = float(saved_params.loc[saved_params['name']=='beta_ses['+str(ms_i+1)+']', 'Mean'])
            stay_mouse = float(saved_params.loc[saved_params['name']=='stay_ses['+str(ms_i+1)+']', 'Mean'])
            laser_mouse = float(saved_params.loc[saved_params['name']=='laser_ses['+str(ms_i+1)+']', 'Mean'])
            side_mouse = float(saved_params.loc[saved_params['name']=='sides['+str(ms_i+1)+']', 'Mean'])
            alpha = float(saved_params.loc[saved_params['name']=='alpha_ses['+str(ms_i+1)+']', 'Mean'])
            alphastay = float(saved_params.loc[saved_params['name']=='alphastay_ses['+str(ms_i+1)+']', 'Mean'])
            forget = float(saved_params.loc[saved_params['name']=='alphaforgetting_ses['+str(ms_i+1)+']', 'Mean'])
            q = np.zeros(2)
            qstay = np.zeros(2)
            V = 0
            predicted_choices=[]
            QL=[]
            QR=[]
            QLstay=[]
            QRstay=[]
            choices=[]
            QLgeneral=[]
            QRgeneral=[]
            QRew = []
            for t in np.arange(NT_all[ms_i]):
                t = int(t)
                predicted_choice = inv_logit(side_mouse
                    + beta_mouse * (q[1] - q[0])
                    + stay_mouse * (qstay[1] - qstay[0]))
                choice = c[sub_idx[ms_i], sess_idx[ms_i], t]
                unchoice = 1*(choice==0)
                if (r[sub_idx[ms_i], sess_idx[ms_i],t] == 1) & (l[sub_idx[ms_i], sess_idx[ms_i],t] == 0):
                        outcome_modified_q = laser_mouse
                elif (r[sub_idx[ms_i], sess_idx[ms_i],t] == 1) & (l[sub_idx[ms_i], sess_idx[ms_i],t] == 1):
                        outcome_modified_q = 1.0
                else:
                    outcome_modified_q = 0.0
                q[choice] = (1-alpha) * q[choice] + alpha * outcome_modified_q # Join r+l for reduced model
                q[unchoice] =  q[unchoice] - forget* q[unchoice]
                qstay = qstay * (1 - alphastay)
                qstay[choice] = qstay[choice] + alphastay
                V = V + alpha * (r[sub_idx[ms_i], sess_idx[ms_i],t] - V)
                # Store variables
                predicted_choices.append(predicted_choice)
                choices.append(choice)
                QL.append(q[0])
                QR.append(q[1])
                QRew.append(V)
                QLstay.append(qstay[0])
                QRstay.append(qstay[1])
                QLgeneral.append((beta_mouse*q[0])+(stay_mouse*qstay[0]))
                QRgeneral.append((beta_mouse*q[1])+(stay_mouse*qstay[1]))
            acc = []
            for i in np.arange(len(choices)):
                acc.append(1*(choices[i]==(1*(predicted_choices[i]>0.5))))
            #print(np.mean(acc))

            if NT_all[ms_i]!=0:
                ses_data = pd.DataFrame()
                ses_data['predicted_choice'] = predicted_choices
                ses_data['QL'] = QLgeneral
                ses_data['QR'] = QRgeneral
                ses_data['V'] = QRew
                ses_data['QLreward'] = np.array(QL) * beta_mouse # Note this QL is the reward  - TODO change name
                ses_data['QRreward'] =  np.array(QR) * beta_mouse
                ses_data['QLstay'] =  np.array(QLstay) * stay_mouse
                ses_data['QRstay'] =  np.array(QRstay) * stay_mouse
                ses_data['deltaQ'] = ses_data['QR'] - ses_data['QL']
                ses_data['choices'] = choices
                ses_data['mouse'] = sub_idx[ms_i]
                ses_data['ses'] = sess_idx[ms_i]
                ses_data['acc'] = np.mean(acc)
                ses_data['id'] = ses_data['mouse']*100 + ses_data['ses']
                ses_data['beta_mouse'] = beta_mouse
                ses_data['stay_mouse'] = stay_mouse
                ses_data['side_mouse'] = side_mouse
                ses_data['alpha'] = alpha
                ses_data['alphastay'] = alphastay
                ses_data['forget'] = forget
                data = pd.concat([data,ses_data])
    return data


def q_learning_model_reduced_stay_forgetting_RPE_laser(standata,saved_params=None):
    r =  standata['r'] * standata['l']
    c =  standata['c']
    sub_idx =  standata['sub_idx']-1
    sess_idx = standata['sess_idx']-1
    NSxNSESS = standata['NSxNSESS']
    NT_all = standata['NT_all']
    data = pd.DataFrame()
    for ms_i in np.arange(NSxNSESS):
        if NT_all[ms_i]>0:
            beta_mouse = float(saved_params.loc[saved_params['name']=='beta_ses['+str(ms_i+1)+']', 'Mean'])
            stay_mouse = float(saved_params.loc[saved_params['name']=='stay_ses['+str(ms_i+1)+']', 'Mean'])
            side_mouse = float(saved_params.loc[saved_params['name']=='sides['+str(ms_i+1)+']', 'Mean'])
            alpha = float(saved_params.loc[saved_params['name']=='alpha_ses['+str(ms_i+1)+']', 'Mean'])
            alphastay = float(saved_params.loc[saved_params['name']=='alphastay_ses['+str(ms_i+1)+']', 'Mean'])
            forget = float(saved_params.loc[saved_params['name']=='alphaforgetting_ses['+str(ms_i+1)+']', 'Mean'])
            q = np.zeros(2)
            qstay = np.zeros(2)
            V = 0
            predicted_choices=[]
            QL=[]
            QR=[]
            QLstay=[]
            QRstay=[]
            choices=[]
            QLgeneral=[]
            QRgeneral=[]
            QLgeneral_raw=[]
            QRgeneral_raw=[]
            QRew = []
            for t in np.arange(NT_all[ms_i]):
                t = int(t)
                predicted_choice = inv_logit(side_mouse
                    + beta_mouse * (q[1] - q[0])
                    + stay_mouse * (qstay[1] - qstay[0]))
                choice = c[sub_idx[ms_i], sess_idx[ms_i], t]
                unchoice = 1*(choice==0)
                if r[sub_idx[ms_i], sess_idx[ms_i],t]==0:
                    q[choice] = (1-alpha) * q[choice] + alpha * (r[sub_idx[ms_i], sess_idx[ms_i],t]) # Join r+l for reduced model
                else:
                    q[choice] = q[choice] + alpha * (r[sub_idx[ms_i], sess_idx[ms_i],t]) # Join r+l for reduced model
                q[unchoice] =  q[unchoice] - forget* q[unchoice]
                qstay = qstay * (1 - alphastay)
                qstay[choice] = qstay[choice] + alphastay
                V = V + alpha * (r[sub_idx[ms_i], sess_idx[ms_i],t] - V)
                # Store variables
                predicted_choices.append(predicted_choice)
                choices.append(choice)
                QL.append(q[0])
                QR.append(q[1])
                QRew.append(V)
                QLstay.append(qstay[0])
                QRstay.append(qstay[1])
                QLgeneral.append((beta_mouse*q[0])+(stay_mouse*qstay[0]))
                QRgeneral.append((beta_mouse*q[1])+(stay_mouse*qstay[1]))
                QLgeneral_raw.append((q[0])+(qstay[0]))
                QRgeneral_raw.append((q[1])+(qstay[1]))
            acc = []
            for i in np.arange(len(choices)):
                acc.append(1*(choices[i]==(1*(predicted_choices[i]>0.5))))
            #print(np.mean(acc))

            if NT_all[ms_i]!=0:
                ses_data = pd.DataFrame()
                ses_data['predicted_choice'] = predicted_choices
                ses_data['QL'] = QLgeneral
                ses_data['QR'] = QRgeneral
                ses_data['QL_raw'] = QLgeneral_raw
                ses_data['QR_raw'] = QRgeneral_raw
                ses_data['V'] = QRew
                ses_data['QLreward'] = np.array(QL)  # Note this QL is the reward  - TODO change name
                ses_data['QRreward'] =  np.array(QR) 
                ses_data['QLstay'] =  np.array(QLstay) 
                ses_data['QRstay'] =  np.array(QRstay) 
                ses_data['deltaQ'] = ses_data['QR'] - ses_data['QL']
                ses_data['choices'] = choices
                ses_data['mouse'] = sub_idx[ms_i]
                ses_data['ses'] = sess_idx[ms_i]
                ses_data['acc'] = np.mean(acc)
                ses_data['id'] = ses_data['mouse']*100 + ses_data['ses']
                ses_data['beta_mouse'] = beta_mouse
                ses_data['stay_mouse'] = stay_mouse
                ses_data['side_mouse'] = side_mouse
                ses_data['alpha'] = alpha
                ses_data['alphastay'] = alphastay
                ses_data['forget'] = forget
                data = pd.concat([data,ses_data])
    return data



def q_learning_model_reduced_stay_forgetting_RPE(standata,saved_params=None):
    r =  standata['r']
    c =  standata['c']
    sub_idx =  standata['sub_idx']-1
    sess_idx = standata['sess_idx']-1
    NSxNSESS = standata['NSxNSESS']
    NT_all = standata['NT_all']
    data = pd.DataFrame()
    for ms_i in np.arange(NSxNSESS):
        if NT_all[ms_i]>0:
            beta_mouse = float(saved_params.loc[saved_params['name']=='beta_ses['+str(ms_i+1)+']', 'Mean'])
            stay_mouse = float(saved_params.loc[saved_params['name']=='stay_ses['+str(ms_i+1)+']', 'Mean'])
            side_mouse = float(saved_params.loc[saved_params['name']=='sides['+str(ms_i+1)+']', 'Mean'])
            alpha = float(saved_params.loc[saved_params['name']=='alpha_ses['+str(ms_i+1)+']', 'Mean'])
            alphastay = float(saved_params.loc[saved_params['name']=='alphastay_ses['+str(ms_i+1)+']', 'Mean'])
            forget = float(saved_params.loc[saved_params['name']=='alphaforgetting_ses['+str(ms_i+1)+']', 'Mean'])
            q = np.zeros(2)
            qstay = np.zeros(2)
            V = 0
            predicted_choices=[]
            QL=[]
            QR=[]
            QLstay=[]
            QRstay=[]
            choices=[]
            QLgeneral=[]
            QRgeneral=[]
            QLgeneral_raw=[]
            QRgeneral_raw=[]
            QRew = []
            for t in np.arange(NT_all[ms_i]):
                t = int(t)
                predicted_choice = inv_logit(side_mouse
                    + beta_mouse * (q[1] - q[0])
                    + stay_mouse * (qstay[1] - qstay[0]))
                choice = c[sub_idx[ms_i], sess_idx[ms_i], t]
                unchoice = 1*(choice==0)
                if r[sub_idx[ms_i], sess_idx[ms_i],t]==0:
                    q[choice] = (1-alpha) * q[choice] + alpha * (r[sub_idx[ms_i], sess_idx[ms_i],t]) # Join r+l for reduced model
                else:
                    q[choice] = q[choice] + alpha * (r[sub_idx[ms_i], sess_idx[ms_i],t]) # Join r+l for reduced model
                q[unchoice] =  q[unchoice] - forget* q[unchoice]
                qstay = qstay * (1 - alphastay)
                qstay[choice] = qstay[choice] + alphastay
                V = V + alpha * (r[sub_idx[ms_i], sess_idx[ms_i],t] - V)
                # Store variables
                predicted_choices.append(predicted_choice)
                choices.append(choice)
                QL.append(q[0])
                QR.append(q[1])
                QRew.append(V)
                QLstay.append(qstay[0])
                QRstay.append(qstay[1])
                QLgeneral.append((beta_mouse*q[0])+(stay_mouse*qstay[0]))
                QRgeneral.append((beta_mouse*q[1])+(stay_mouse*qstay[1]))
                QLgeneral_raw.append((q[0])+(qstay[0]))
                QRgeneral_raw.append((q[1])+(qstay[1]))
            acc = []
            for i in np.arange(len(choices)):
                acc.append(1*(choices[i]==(1*(predicted_choices[i]>0.5))))
            #print(np.mean(acc))

            if NT_all[ms_i]!=0:
                ses_data = pd.DataFrame()
                ses_data['predicted_choice'] = predicted_choices
                ses_data['QL'] = QLgeneral
                ses_data['QR'] = QRgeneral
                ses_data['QL_raw'] = QLgeneral_raw
                ses_data['QR_raw'] = QRgeneral_raw
                ses_data['V'] = QRew
                ses_data['QLreward'] = np.array(QL)  # Note this QL is the reward  - TODO change name
                ses_data['QRreward'] =  np.array(QR) 
                ses_data['QLstay'] =  np.array(QLstay) 
                ses_data['QRstay'] =  np.array(QRstay) 
                ses_data['deltaQ'] = ses_data['QR'] - ses_data['QL']
                ses_data['choices'] = choices
                ses_data['mouse'] = sub_idx[ms_i]
                ses_data['ses'] = sess_idx[ms_i]
                ses_data['acc'] = np.mean(acc)
                ses_data['id'] = ses_data['mouse']*100 + ses_data['ses']
                ses_data['beta_mouse'] = beta_mouse
                ses_data['stay_mouse'] = stay_mouse
                ses_data['side_mouse'] = side_mouse
                ses_data['alpha'] = alpha
                ses_data['alphastay'] = alphastay
                ses_data['forget'] = forget
                data = pd.concat([data,ses_data])
    return data


def q_learning_model_reduced_stay(standata,saved_params=None):
    r =  standata['r']
    c =  standata['c']
    sub_idx =  standata['sub_idx']-1
    sess_idx = standata['sess_idx']-1
    NSxNSESS = standata['NSxNSESS']
    NT_all = standata['NT_all']
    data = pd.DataFrame()
    for ms_i in np.arange(NSxNSESS):
        if NT_all[ms_i]>0:
            beta_mouse = float(saved_params.loc[saved_params['name']=='beta_ses['+str(ms_i+1)+']', 'Mean'])
            stay_mouse = float(saved_params.loc[saved_params['name']=='stay_ses['+str(ms_i+1)+']', 'Mean'])
            side_mouse = float(saved_params.loc[saved_params['name']=='sides['+str(ms_i+1)+']', 'Mean'])
            alpha = float(saved_params.loc[saved_params['name']=='alpha_ses['+str(ms_i+1)+']', 'Mean'])
            alphastay = float(saved_params.loc[saved_params['name']=='alphastay_ses['+str(ms_i+1)+']', 'Mean'])
            q = np.zeros(2)
            qstay = np.zeros(2)
            predicted_choices=[]
            QL=[]
            QR=[]
            QLstay=[]
            QRstay=[]
            choices=[]
            QLgeneral=[]
            QRgeneral=[]
            for t in np.arange(NT_all[ms_i]):
                t = int(t)
                predicted_choice = inv_logit(side_mouse
                    + beta_mouse  * (q[1] - q[0])
                    + stay_mouse * (qstay[1] - qstay[0]))
                choice = c[sub_idx[ms_i], sess_idx[ms_i], t]
                q[choice] = (1-alpha) * q[choice] + alpha * (r[sub_idx[ms_i], sess_idx[ms_i],t]) # Join r+l for reduced model
                qstay = qstay * (1 - alphastay)
                qstay[choice] = qstay[choice] + alphastay
                # Store variables
                predicted_choices.append(predicted_choice)
                choices.append(choice)
                QL.append(q[0])
                QR.append(q[1])
                QLstay.append(qstay[0])
                QRstay.append(qstay[1])
                QLgeneral.append((beta_mouse*q[0])+(stay_mouse*qstay[0]))
                QRgeneral.append((beta_mouse*q[1])+(stay_mouse*qstay[1]))
            acc = []
            for i in np.arange(len(choices)):
                acc.append(1*(choices[i]==(1*(predicted_choices[i]>0.5))))
            #print(np.mean(acc))

            if NT_all[ms_i]!=0:
                ses_data = pd.DataFrame()
                ses_data['predicted_choice'] = predicted_choices
                ses_data['QL'] = QLgeneral
                ses_data['QR'] = QRgeneral
                ses_data['QLreward'] = np.array(QL) # Note this QL is the reward  - TODO change name
                ses_data['QRreward'] =  np.array(QR) 
                ses_data['QLstay'] =  np.array(QLstay) * stay_mouse
                ses_data['QRstay'] =  np.array(QRstay) * stay_mouse
                ses_data['deltaQ'] = ses_data['QR'] - ses_data['QL']
                ses_data['choices'] = choices
                ses_data['mouse'] = sub_idx[ms_i]
                ses_data['ses'] = sess_idx[ms_i]
                ses_data['acc'] = np.mean(acc)
                ses_data['id'] = ses_data['mouse']*100 + ses_data['ses']
                data = pd.concat([data,ses_data])
    return data

def q_learning_model_reduced_nostay(standata,saved_params=None):
    r =  standata['r']
    c =  standata['c']
    sub_idx =  standata['sub_idx']-1
    sess_idx = standata['sess_idx']-1
    NSxNSESS = standata['NSxNSESS']
    NT_all = standata['NT_all']
    data = pd.DataFrame()
    for ms_i in np.arange(NSxNSESS):
        if NT_all[ms_i]>0:
            beta_mouse = float(saved_params.loc[saved_params['name']=='beta_ses['+str(ms_i+1)+']', 'Mean'])
            side_mouse = float(saved_params.loc[saved_params['name']=='sides['+str(ms_i+1)+']', 'Mean'])
            alpha = float(saved_params.loc[saved_params['name']=='alpha_ses['+str(ms_i+1)+']', 'Mean'])
            q = np.zeros(2)
            predicted_choices=[]
            QL=[]
            QR=[]
            choices=[]
            QLgeneral=[]
            QRgeneral=[]
            for t in np.arange(NT_all[ms_i]):
                t = int(t)
                predicted_choice = inv_logit(side_mouse
                    + beta_mouse  * (q[1] - q[0]))
                choice = c[sub_idx[ms_i], sess_idx[ms_i], t]
                q[choice] = (1-alpha) * q[choice] + alpha * (r[sub_idx[ms_i], sess_idx[ms_i],t]) # Join r+l for reduced model
                # Store variables
                predicted_choices.append(predicted_choice)
                choices.append(choice)
                QL.append(q[0])
                QR.append(q[1])
                QLgeneral.append((beta_mouse*q[0]))
                QRgeneral.append((beta_mouse*q[1]))
            acc = []
            for i in np.arange(len(choices)):
                acc.append(1*(choices[i]==(1*(predicted_choices[i]>0.5))))
            #print(np.mean(acc))

            if NT_all[ms_i]!=0:
                ses_data = pd.DataFrame()
                ses_data['predicted_choice'] = predicted_choices
                ses_data['QL'] = QLgeneral
                ses_data['QR'] = QRgeneral
                ses_data['QLreward'] = np.array(QL) * beta_mouse # Note this QL is the reward  - TODO change name
                ses_data['QRreward'] =  np.array(QR) * beta_mouse
                ses_data['deltaQ'] = ses_data['QR'] - ses_data['QL']
                ses_data['choices'] = choices
                ses_data['mouse'] = sub_idx[ms_i]
                ses_data['ses'] = sess_idx[ms_i]
                ses_data['acc'] = np.mean(acc)
                ses_data['id'] = ses_data['mouse']*100 + ses_data['ses']
                data = pd.concat([data,ses_data])
    return data

def q_learning_lasernostay(standata,saved_params=None, fit=None, csv=True):
    if saved_params is not None:
        r =  standata['r']
        c =  standata['c']
        l =  standata['l']
        sub_idx =  standata['sub_idx']-1
        sess_idx = standata['sess_idx']-1
        NSxNSESS = standata['NSxNSESS']
        NT_all = standata['NT_all']
    else:
        r =  fit.data['r']
        c =  fit.data['c']
        l =  fit.data['l']
        sub_idx =  fit.data['sub_idx']-1
        sess_idx = fit.data['sess_idx']-1
    data = pd.DataFrame()

    for ms_i in np.arange(NSxNSESS):
        if NT_all[ms_i]>0:
            if saved_params is not None:
            #    if ms_i==0:
            #        print('Using saved parameters')
                if csv==True:
                    beta_mouse = float(saved_params.loc[saved_params['name']=='beta_ses['+str(ms_i+1)+']', 'Mean'])
                    side_mouse = float(saved_params.loc[saved_params['name']=='sides['+str(ms_i+1)+']', 'Mean'])
                    laser_mouse = float(saved_params.loc[saved_params['name']=='laser_ses['+str(ms_i+1)+']', 'Mean'])
                    alpha = float(saved_params.loc[saved_params['name']=='alpha_ses['+str(ms_i+1)+']', 'Mean'])
                    alphalaser = float(saved_params.loc[saved_params['name']=='alphalaser_ses['+str(ms_i+1)+']', 'Mean'])
                else:
                    print('Error: Only csv version available at the moment!')
            q = np.zeros(2)
            qlaser = np.zeros(2)
            predicted_choices=[]
            QL=[]
            QR=[]
            QLlaser=[]
            QRlaser=[]
            choices=[]
            QLgeneral=[]
            QRgeneral=[]
            for t in np.arange(NT_all[ms_i]):
                t = int(t)
                predicted_choice = inv_logit(side_mouse
                    + beta_mouse  * (q[1] - q[0])
                    + laser_mouse * (qlaser[1] - qlaser[0]))

                choice = c[sub_idx[ms_i], sess_idx[ms_i], t]
                q[choice] = (1-alpha) * q[choice] + alpha * r[sub_idx[ms_i], sess_idx[ms_i],t]
                qlaser[choice] = (1-alphalaser) * qlaser[choice] + alphalaser * l[sub_idx[ms_i], sess_idx[ms_i],t]
                # Store variables
                predicted_choices.append(predicted_choice)
                choices.append(choice)
                QL.append(q[0])
                QR.append(q[1])
                QLlaser.append(qlaser[0])
                QRlaser.append(qlaser[1])
                QLgeneral.append((beta_mouse*q[0])+(laser_mouse*qlaser[0]))
                QRgeneral.append((beta_mouse*q[1])+(laser_mouse*qlaser[1]))
            acc = []
            for i in np.arange(len(choices)):
                acc.append(1*(choices[i]==(1*(predicted_choices[i]>0.5))))
            #print(np.mean(acc))

            if NT_all[ms_i]!=0:
                ses_data = pd.DataFrame()
                ses_data['predicted_choice'] = predicted_choices
                ses_data['QL'] = QLgeneral
                ses_data['QR'] = QRgeneral
                ses_data['QLreward'] = np.array(QL) * beta_mouse # Note this QL is the reward  - TODO change name
                ses_data['QRreward'] =  np.array(QR) * beta_mouse
                ses_data['QLlaser'] =  np.array(QLlaser) * laser_mouse
                ses_data['QRlaser'] =  np.array(QRlaser) * laser_mouse
                ses_data['deltaQ'] = ses_data['QR'] - ses_data['QL']
                ses_data['choices'] = choices
                ses_data['mouse'] = sub_idx[ms_i]
                ses_data['ses'] = sess_idx[ms_i]
                ses_data['acc'] = np.mean(acc)
                data = pd.concat([data,ses_data])
    return data
def q_learning_model(standata,saved_params=None, fit=None, csv=True):
    if saved_params is not None:
        r =  standata['r']
        c =  standata['c']
        l =  standata['l']
        sub_idx =  standata['sub_idx']-1
        sess_idx = standata['sess_idx']-1
        NSxNSESS = standata['NSxNSESS']
        NT_all = standata['NT_all']
    else:
        r =  fit.data['r']
        c =  fit.data['c']
        l =  fit.data['l']
        sub_idx =  fit.data['sub_idx']-1
        sess_idx = fit.data['sess_idx']-1
    data = pd.DataFrame()

    for ms_i in np.arange(NSxNSESS):
        if NT_all[ms_i]>0:
            if saved_params is not None:
            #    if ms_i==0:
            #        print('Using saved parameters')
                if csv==True:
                    beta_mouse = float(saved_params.loc[saved_params['name']=='beta_ses['+str(ms_i+1)+']', 'Mean'])
                    stay_mouse = float(saved_params.loc[saved_params['name']=='stay_ses['+str(ms_i+1)+']', 'Mean'])
                    side_mouse = float(saved_params.loc[saved_params['name']=='sides['+str(ms_i+1)+']', 'Mean'])
                    laser_mouse = float(saved_params.loc[saved_params['name']=='laser_ses['+str(ms_i+1)+']', 'Mean'])
                    alpha = float(saved_params.loc[saved_params['name']=='alpha_ses['+str(ms_i+1)+']', 'Mean'])
                    alphalaser = float(saved_params.loc[saved_params['name']=='alphalaser_ses['+str(ms_i+1)+']', 'Mean'])
                    alphastay = float(saved_params.loc[saved_params['name']=='alphastay_ses['+str(ms_i+1)+']', 'Mean'])
            #    else:
            #        print('Error: Only csv version available at the moment!')
            else:
                if ms_i==0:
                    print('No saved parameters')
            q = np.zeros(2)
            qstay = np.zeros(2)
            qlaser = np.zeros(2)
            predicted_choices=[]
            QL=[]
            QR=[]
            QLstay=[]
            QRstay=[]
            QLlaser=[]
            QRlaser=[]
            choices=[]
            QLgeneral=[]
            QRgeneral=[]
            for t in np.arange(NT_all[ms_i]):
                t = int(t)
                predicted_choice = inv_logit(side_mouse
                    + beta_mouse  * (q[1] - q[0])
                    + stay_mouse * (qstay[1] - qstay[0])
                    + laser_mouse * (qlaser[1] - qlaser[0]))

                choice = c[sub_idx[ms_i], sess_idx[ms_i], t]
                q[choice] = (1-alpha) * q[choice] + alpha * r[sub_idx[ms_i], sess_idx[ms_i],t]
                qlaser[choice] = (1-alphalaser) * qlaser[choice] + alphalaser * l[sub_idx[ms_i], sess_idx[ms_i],t]
                qstay = qstay * (1 - alphastay)
                qstay[choice] = qstay[choice] + alphastay
                # Store variables
                predicted_choices.append(predicted_choice)
                choices.append(choice)
                QL.append(q[0])
                QR.append(q[1])
                QLstay.append(qstay[0])
                QRstay.append(qstay[1])
                QLlaser.append(qlaser[0])
                QRlaser.append(qlaser[1])
                QLgeneral.append((beta_mouse*q[0])+(laser_mouse*qlaser[0])+(stay_mouse*qstay[0]))
                QRgeneral.append((beta_mouse*q[1])+(laser_mouse*qlaser[1])+(stay_mouse*qstay[1]))
            acc = []
            for i in np.arange(len(choices)):
                acc.append(1*(choices[i]==(1*(predicted_choices[i]>0.5))))
            #print(np.mean(acc))

            if NT_all[ms_i]!=0:
                ses_data = pd.DataFrame()
                ses_data['predicted_choice'] = predicted_choices
                ses_data['QL'] = QLgeneral
                ses_data['QR'] = QRgeneral
                ses_data['QLreward'] = np.array(QL) * beta_mouse # Note this QL is the reward  - TODO change name
                ses_data['QRreward'] =  np.array(QR) * beta_mouse
                ses_data['QLstay'] =  np.array(QLstay) * stay_mouse
                ses_data['QRstay'] =  np.array(QRstay) * stay_mouse
                ses_data['QLlaser'] =  np.array(QLlaser) * laser_mouse
                ses_data['QRlaser'] =  np.array(QRlaser) * laser_mouse
                ses_data['deltaQ'] = ses_data['QR'] - ses_data['QL']
                ses_data['choices'] = choices
                ses_data['mouse'] = sub_idx[ms_i]
                ses_data['ses'] = sess_idx[ms_i]
                ses_data['acc'] = np.mean(acc)
                data = pd.concat([data,ses_data])
    return make_deltas(data)


def simulate_q_learning_model_alphalaserdecay(standata_recovery,saved_params=None, fit=None, csv=True):
    NS =standata_recovery['NS']
    NSESS=standata_recovery['NSESS']
    NT =standata_recovery['NT']
    NT_all =standata_recovery['NT_all']
    r_sim = np.zeros([NS,NSESS,NT])
    c_sim = np.zeros([NS,NSESS,NT])
    l_sim = np.zeros([NS,NSESS,NT])
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
                    beta_mouse = float(saved_params.loc[saved_params['name']=='beta_ses['+str(ms_i+1)+']', 'Mean'])
                    stay_mouse = float(saved_params.loc[saved_params['name']=='stay_ses['+str(ms_i+1)+']', 'Mean'])
                    side_mouse = float(saved_params.loc[saved_params['name']=='sides['+str(ms_i+1)+']', 'Mean'])
                    laser_mouse = float(saved_params.loc[saved_params['name']=='laser_ses['+str(ms_i+1)+']', 'Mean'])
                    alpha = float(saved_params.loc[saved_params['name']=='alpha_ses['+str(ms_i+1)+']', 'Mean'])
                    alphalaser_init = float(saved_params.loc[saved_params['name']=='alphalaser_ses['+str(ms_i+1)+']', 'Mean'])
                    alphastay = float(saved_params.loc[saved_params['name']=='alphastay_ses['+str(ms_i+1)+']', 'Mean'])
                    alphalaserbase= float(saved_params.loc[saved_params['name']=='alphalaserbase_ses['+str(ms_i+1)+']', 'Mean'])
                    laserdecay= float(saved_params.loc[saved_params['name']=='laserdecay_ses['+str(ms_i+1)+']', 'Mean'])
                else:
                    print('Error: Only csv version available at the moment!')

            tph = TrialParamHandler()
            q = np.zeros(2)
            qlaser = np.zeros(2)
            qstay = np.zeros(2)
            predicted_choices=[]
            rewards = []
            lasers = []
            outcome=[]
            opto_blocks=[]
            side_blocks=[]
            trial_within_block=[]
            right_reward=[]
            left_reward=[]
            alphalaser_buffer=[]
            buffer_qlaser=[]
            buffer_qwater=[]

            alphalaser = alphalaser_init

            for t in np.arange(NT_all[ms_i]):
                t = int(t)
                p_choice = inv_logit(side_mouse
                    + beta_mouse  * (q[1] - q[0])
                    + laser_mouse  * (qlaser[1] - qlaser[0])
                    + stay_mouse * (qstay[1] - qstay[0]))
                choice =  np.random.choice(2,p= [1-p_choice,p_choice])
                tph.response_side_buffer.append(choice)  # Add choice to response buffer
                p_left= tph.stim_probability_left
                l = tph.opto_block
                if choice==1:
                    f = tph.right_reward
                if choice==0:
                    f = tph.left_reward
                q[choice] = (1-alpha) * q[choice] + alpha * (1*(f*(l==0)))
                qlaser[choice] = (1-(alphalaser+alphalaserbase)) * qlaser[choice] + ((alphalaser+alphalaserbase) * f * l)
                qstay = qstay * (1 - alphastay)
                qstay[choice] = qstay[choice] + alphastay
                if (f*l) == 1:
                    alphalaser = (1-laserdecay)*alphalaser
                # Store variables
                alphalaser_buffer.append(alphalaser)
                rewards.append((1*(f*(l==0))))
                lasers.append(f*l)
                outcome.append(f)
                opto_blocks.append(l) 
                side_blocks.append(p_left)
                predicted_choices.append(choice)
                trial_within_block.append(tph.block_trial_num)
                right_reward.append(tph.right_reward)
                left_reward.append(tph.left_reward)
                buffer_qlaser.append(qlaser[choice])
                buffer_qwater.append(q[choice])
                # Init next trial
                tph =  tph.next_trial()

            if NT_all[ms_i]!=0:
                c_sim[sub_idx[ms_i], sess_idx[ms_i], :int(NT_all[ms_i])]= predicted_choices
                l_sim[sub_idx[ms_i], sess_idx[ms_i], :int(NT_all[ms_i])] = lasers
                r_sim[sub_idx[ms_i], sess_idx[ms_i], :int(NT_all[ms_i])] = rewards
                ses_data = pd.DataFrame()
                ses_data['choices'] = predicted_choices
                ses_data['a_init_laser'] = alphalaser_init
                ses_data['alpha_laser'] = alphalaser_buffer
                ses_data['water'] = rewards
                ses_data['laser'] = lasers
                ses_data['laser_block'] = opto_blocks
                ses_data['tb'] = trial_within_block
                ses_data['probabilityLeft'] = side_blocks
                ses_data['outcome'] = outcome
                ses_data['mouse'] = sub_idx[ms_i]
                ses_data['ses'] = sess_idx[ms_i]
                ses_data['right_reward'] = right_reward
                ses_data['left_reward'] = left_reward
                ses_data['qwater'] = buffer_qwater
                ses_data['qlaser'] = buffer_qlaser
                sim_data = pd.concat([sim_data,ses_data])
    return c_sim, l_sim, r_sim, sim_data

def simulate_q_learning_model_alphalaserdecay_everytrial(standata_recovery,saved_params=None, fit=None, csv=True):
    NS =standata_recovery['NS']
    NSESS=standata_recovery['NSESS']
    NT =standata_recovery['NT']
    NT_all =standata_recovery['NT_all']
    r_sim = np.zeros([NS,NSESS,NT])
    c_sim = np.zeros([NS,NSESS,NT])
    l_sim = np.zeros([NS,NSESS,NT])
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
                    beta_mouse = float(saved_params.loc[saved_params['name']=='beta_ses['+str(ms_i+1)+']', 'Mean'])
                    stay_mouse = float(saved_params.loc[saved_params['name']=='stay_ses['+str(ms_i+1)+']', 'Mean'])
                    side_mouse = float(saved_params.loc[saved_params['name']=='sides['+str(ms_i+1)+']', 'Mean'])
                    laser_mouse = float(saved_params.loc[saved_params['name']=='laser_ses['+str(ms_i+1)+']', 'Mean'])
                    alpha = float(saved_params.loc[saved_params['name']=='alpha_ses['+str(ms_i+1)+']', 'Mean'])
                    alphalaser_init = float(saved_params.loc[saved_params['name']=='alphalaser_ses['+str(ms_i+1)+']', 'Mean'])
                    alphastay = float(saved_params.loc[saved_params['name']=='alphastay_ses['+str(ms_i+1)+']', 'Mean'])
                    alphalaserbase= float(saved_params.loc[saved_params['name']=='alphalaserbase_ses['+str(ms_i+1)+']', 'Mean'])
                    laserdecay= float(saved_params.loc[saved_params['name']=='laserdecay_ses['+str(ms_i+1)+']', 'Mean'])
                else:
                    print('Error: Only csv version available at the moment!')

            tph = TrialParamHandler()
            q = np.zeros(2)
            qlaser = np.zeros(2)
            qstay = np.zeros(2)
            predicted_choices=[]
            rewards = []
            lasers = []
            outcome=[]
            opto_blocks=[]
            side_blocks=[]
            trial_within_block=[]
            right_reward=[]
            left_reward=[]
            alphalaser_buffer=[]
            buffer_qlaser=[]
            buffer_qwater=[]

            alphalaser = alphalaser_init

            for t in np.arange(NT_all[ms_i]):
                t = int(t)
                p_choice = inv_logit(side_mouse
                    + beta_mouse  * (q[1] - q[0])
                    + laser_mouse  * (qlaser[1] - qlaser[0])
                    + stay_mouse * (qstay[1] - qstay[0]))
                choice =  np.random.choice(2,p= [1-p_choice,p_choice])
                tph.response_side_buffer.append(choice)  # Add choice to response buffer
                p_left= tph.stim_probability_left
                l = tph.opto_block
                if choice==1:
                    f = tph.right_reward
                if choice==0:
                    f = tph.left_reward
                q[choice] = (1-alpha) * q[choice] + alpha * (1*(f*(l==0)))
                qlaser[choice] = (1-(alphalaser+alphalaserbase)) * qlaser[choice] + (alphalaser+alphalaserbase) * f * l
                qstay = qstay * (1 - alphastay)
                qstay[choice] = qstay[choice] + alphastay
                alphalaser = (1-laserdecay)*alphalaser
                # Store variables
                alphalaser_buffer.append(alphalaser)
                rewards.append((1*(f*(l==0))))
                lasers.append(f*l)
                outcome.append(f)
                opto_blocks.append(l) 
                side_blocks.append(p_left)
                predicted_choices.append(choice)
                trial_within_block.append(tph.block_trial_num)
                right_reward.append(tph.right_reward)
                left_reward.append(tph.left_reward)
                buffer_qlaser.append(qlaser[choice])
                buffer_qwater.append(q[choice])
                # Init next trial
                tph =  tph.next_trial()

            if NT_all[ms_i]!=0:
                c_sim[sub_idx[ms_i], sess_idx[ms_i], :int(NT_all[ms_i])]= predicted_choices
                l_sim[sub_idx[ms_i], sess_idx[ms_i], :int(NT_all[ms_i])] = lasers
                r_sim[sub_idx[ms_i], sess_idx[ms_i], :int(NT_all[ms_i])] = rewards
                ses_data = pd.DataFrame()
                ses_data['choices'] = predicted_choices
                ses_data['a_init_laser'] = alphalaser_init
                ses_data['alpha_laser'] = alphalaser_buffer
                ses_data['water'] = rewards
                ses_data['laser'] = lasers
                ses_data['laser_block'] = opto_blocks
                ses_data['tb'] = trial_within_block
                ses_data['probabilityLeft'] = side_blocks
                ses_data['outcome'] = outcome
                ses_data['mouse'] = sub_idx[ms_i]
                ses_data['ses'] = sess_idx[ms_i]
                ses_data['right_reward'] = right_reward
                ses_data['left_reward'] = left_reward
                ses_data['qwater'] = buffer_qwater
                ses_data['qlaser'] = buffer_qlaser
                sim_data = pd.concat([sim_data,ses_data])
    return c_sim, l_sim, r_sim, sim_data
def q_learning_model_alphalaserdecay_everytrial(standata,saved_params=None, fit=None, csv=True):
    if saved_params is not None:
        r =  standata['r']
        c =  standata['c']
        l =  standata['l']
        sub_idx =  standata['sub_idx']-1
        sess_idx = standata['sess_idx']-1
        NSxNSESS = standata['NSxNSESS']
        NT_all = standata['NT_all']
    else:
        r =  fit.data['r']
        c =  fit.data['c']
        l =  fit.data['l']
        sub_idx =  fit.data['sub_idx']-1
        sess_idx = fit.data['sess_idx']-1
    data = pd.DataFrame()

    for ms_i in np.arange(NSxNSESS):
        if NT_all[ms_i]>0:
            if saved_params is not None:
            #    if ms_i==0:
            #        print('Using saved parameters')
                if csv==True:
                    beta_mouse = float(saved_params.loc[saved_params['name']=='beta_ses['+str(ms_i+1)+']', 'Mean'])
                    stay_mouse = float(saved_params.loc[saved_params['name']=='stay_ses['+str(ms_i+1)+']', 'Mean'])
                    side_mouse = float(saved_params.loc[saved_params['name']=='sides['+str(ms_i+1)+']', 'Mean'])
                    laser_mouse = float(saved_params.loc[saved_params['name']=='laser_ses['+str(ms_i+1)+']', 'Mean'])
                    alpha = float(saved_params.loc[saved_params['name']=='alpha_ses['+str(ms_i+1)+']', 'Mean'])
                    alphalaser = float(saved_params.loc[saved_params['name']=='alphalaser_ses['+str(ms_i+1)+']', 'Mean'])
                    alphastay = float(saved_params.loc[saved_params['name']=='alphastay_ses['+str(ms_i+1)+']', 'Mean'])
                    alphalaserbase= float(saved_params.loc[saved_params['name']=='alphalaserbase_ses['+str(ms_i+1)+']', 'Mean'])
                    laserdecay= float(saved_params.loc[saved_params['name']=='laserdecay_ses['+str(ms_i+1)+']', 'Mean'])
            #    else:
            #        print('Error: Only csv version available at the moment!')
            q = np.zeros(2)
            qstay = np.zeros(2)
            qlaser = np.zeros(2)
            predicted_choices=[]
            QL=[]
            QR=[]
            QLstay=[]
            QRstay=[]
            QLlaser=[]
            QRlaser=[]
            choices=[]
            QLgeneral=[]
            QRgeneral=[]
            alphalaser_buffer=[]
            alphalaser = alphalaser
            for t in np.arange(NT_all[ms_i]):
                t = int(t)
                predicted_choice = inv_logit(side_mouse
                    + beta_mouse  * (q[1] - q[0])
                    + stay_mouse * (qstay[1] - qstay[0])
                    + laser_mouse * (qlaser[1] - qlaser[0]))
                choice = c[sub_idx[ms_i], sess_idx[ms_i], t]
                q[choice] = (1-alpha) * q[choice] + alpha * r[sub_idx[ms_i], sess_idx[ms_i],t]
                qlaser[choice] = (1-(alphalaser+alphalaserbase)) * qlaser[choice] + (alphalaser+alphalaserbase) * l[sub_idx[ms_i], sess_idx[ms_i],t]
                alphalaser = (1-laserdecay)*alphalaser
                qstay = qstay * (1 - alphastay)
                qstay[choice] = qstay[choice] + alphastay
                # Store variables
                predicted_choices.append(predicted_choice)
                choices.append(choice)
                QL.append(q[0])
                QR.append(q[1])
                QLstay.append(qstay[0])
                QRstay.append(qstay[1])
                QLlaser.append(qlaser[0])
                QRlaser.append(qlaser[1])
                QLgeneral.append((beta_mouse*q[0])+(laser_mouse*qlaser[0])+(stay_mouse*qstay[0]))
                QRgeneral.append((beta_mouse*q[1])+(laser_mouse*qlaser[1])+(stay_mouse*qstay[1]))
                alphalaser_buffer.append(alphalaser+alphalaserbase)

            acc = []
            for i in np.arange(len(choices)):
                acc.append(1*(choices[i]==(1*(predicted_choices[i]>0.5))))
            #print(np.mean(acc))

            if NT_all[ms_i]!=0:
                ses_data = pd.DataFrame()
                ses_data['predicted_choice'] = predicted_choices
                ses_data['QL'] = QLgeneral
                ses_data['QR'] = QRgeneral
                ses_data['QLreward'] = np.array(QL) * beta_mouse # Note this QL is the reward  - TODO change name
                ses_data['QRreward'] =  np.array(QR) * beta_mouse
                ses_data['QLstay'] =  np.array(QLstay) * stay_mouse
                ses_data['QRstay'] =  np.array(QRstay) * stay_mouse
                ses_data['QLlaser'] =  np.array(QLlaser) * laser_mouse
                ses_data['QRlaser'] =  np.array(QRlaser) * laser_mouse
                ses_data['deltaQ'] = ses_data['QR'] - ses_data['QL']
                ses_data['choices'] = choices
                ses_data['mouse'] = sub_idx[ms_i]
                ses_data['ses'] = sess_idx[ms_i]
                ses_data['acc'] = np.mean(acc)
                ses_data['alphalaser'] = np.array(alphalaser_buffer)
                data = pd.concat([data,ses_data])
    return make_deltas(data)
def q_learning_model_alphalaserdecay(standata,saved_params=None, fit=None, csv=True):
    if saved_params is not None:
        r =  standata['r']
        c =  standata['c']
        l =  standata['l']
        sub_idx =  standata['sub_idx']-1
        sess_idx = standata['sess_idx']-1
        NSxNSESS = standata['NSxNSESS']
        NT_all = standata['NT_all']
    else:
        r =  fit.data['r']
        c =  fit.data['c']
        l =  fit.data['l']
        sub_idx =  fit.data['sub_idx']-1
        sess_idx = fit.data['sess_idx']-1
    data = pd.DataFrame()

    for ms_i in np.arange(NSxNSESS):
        if NT_all[ms_i]>0:
            if saved_params is not None:
            #    if ms_i==0:
            #        print('Using saved parameters')
                if csv==True:
                    beta_mouse = float(saved_params.loc[saved_params['name']=='beta_ses['+str(ms_i+1)+']', 'Mean'])
                    stay_mouse = float(saved_params.loc[saved_params['name']=='stay_ses['+str(ms_i+1)+']', 'Mean'])
                    side_mouse = float(saved_params.loc[saved_params['name']=='sides['+str(ms_i+1)+']', 'Mean'])
                    laser_mouse = float(saved_params.loc[saved_params['name']=='laser_ses['+str(ms_i+1)+']', 'Mean'])
                    alpha = float(saved_params.loc[saved_params['name']=='alpha_ses['+str(ms_i+1)+']', 'Mean'])
                    alphalaser_init = float(saved_params.loc[saved_params['name']=='alphalaser_ses['+str(ms_i+1)+']', 'Mean'])
                    alphastay = float(saved_params.loc[saved_params['name']=='alphastay_ses['+str(ms_i+1)+']', 'Mean'])
                    alphalaserbase= float(saved_params.loc[saved_params['name']=='alphalaserbase_ses['+str(ms_i+1)+']', 'Mean'])
                    laserdecay= float(saved_params.loc[saved_params['name']=='laserdecay_ses['+str(ms_i+1)+']', 'Mean'])
            #    else:
            #        print('Error: Only csv version available at the moment!')
            q = np.zeros(2)
            qstay = np.zeros(2)
            qlaser = np.zeros(2)
            predicted_choices=[]
            QL=[]
            QR=[]
            QLstay=[]
            QRstay=[]
            QLlaser=[]
            QRlaser=[]
            choices=[]
            QLgeneral=[]
            QRgeneral=[]
            alphalaser_buffer=[]
            alphalaser = alphalaser_init
            for t in np.arange(NT_all[ms_i]):
                t = int(t)
                predicted_choice = inv_logit(side_mouse
                    + beta_mouse  * (q[1] - q[0])
                    + stay_mouse * (qstay[1] - qstay[0])
                    + laser_mouse * (qlaser[1] - qlaser[0]))
                choice = c[sub_idx[ms_i], sess_idx[ms_i], t]
                q[choice] = (1-alpha) * q[choice] + alpha * r[sub_idx[ms_i], sess_idx[ms_i],t]
                qlaser[choice] = (1-(alphalaser+alphalaserbase)) * qlaser[choice] + (alphalaser+alphalaserbase) * l[sub_idx[ms_i], sess_idx[ms_i],t]
                if l[sub_idx[ms_i], sess_idx[ms_i], t] == 1:
                    alphalaser = (1-laserdecay)*alphalaser
                qstay = qstay * (1 - alphastay)
                qstay[choice] = qstay[choice] + alphastay
                # Store variables
                predicted_choices.append(predicted_choice)
                choices.append(choice)
                QL.append(q[0])
                QR.append(q[1])
                QLstay.append(qstay[0])
                QRstay.append(qstay[1])
                QLlaser.append(qlaser[0])
                QRlaser.append(qlaser[1])
                QLgeneral.append((beta_mouse*q[0])+(laser_mouse*qlaser[0])+(stay_mouse*qstay[0]))
                QRgeneral.append((beta_mouse*q[1])+(laser_mouse*qlaser[1])+(stay_mouse*qstay[1]))
                alphalaser_buffer.append(alphalaser+alphalaserbase)

            acc = []
            for i in np.arange(len(choices)):
                acc.append(1*(choices[i]==(1*(predicted_choices[i]>0.5))))
            #print(np.mean(acc))

            if NT_all[ms_i]!=0:
                ses_data = pd.DataFrame()
                ses_data['predicted_choice'] = predicted_choices
                ses_data['QL'] = QLgeneral
                ses_data['QR'] = QRgeneral
                ses_data['QLreward'] = np.array(QL) * beta_mouse # Note this QL is the reward  - TODO change name
                ses_data['QRreward'] =  np.array(QR) * beta_mouse
                ses_data['QLstay'] =  np.array(QLstay) * stay_mouse
                ses_data['QRstay'] =  np.array(QRstay) * stay_mouse
                ses_data['QLlaser'] =  np.array(QLlaser) * laser_mouse
                ses_data['QRlaser'] =  np.array(QRlaser) * laser_mouse
                ses_data['deltaQ'] = ses_data['QR'] - ses_data['QL']
                ses_data['choices'] = choices
                ses_data['mouse'] = sub_idx[ms_i]
                ses_data['ses'] = sess_idx[ms_i]
                ses_data['acc'] = np.mean(acc)
                ses_data['alphalaser'] = np.array(alphalaser_buffer)
                data = pd.concat([data,ses_data])
    return make_deltas(data)
def q_learning_model_laserdecay(standata,saved_params=None, fit=None, csv=True):
    if saved_params is not None:
        r =  standata['r']
        c =  standata['c']
        l =  standata['l']
        sub_idx =  standata['sub_idx']-1
        sess_idx = standata['sess_idx']-1
        NSxNSESS = standata['NSxNSESS']
        NT_all = standata['NT_all']
    else:
        r =  fit.data['r']
        c =  fit.data['c']
        l =  fit.data['l']
        sub_idx =  fit.data['sub_idx']-1
        sess_idx = fit.data['sess_idx']-1
    data = pd.DataFrame()

    for ms_i in np.arange(NSxNSESS):
        if NT_all[ms_i]>0:
            if saved_params is not None:
            #    if ms_i==0:
            #        print('Using saved parameters')
                if csv==True:
                    beta_mouse = float(saved_params.loc[saved_params['name']=='beta_ses['+str(ms_i+1)+']', 'Mean'])
                    stay_mouse = float(saved_params.loc[saved_params['name']=='stay_ses['+str(ms_i+1)+']', 'Mean'])
                    side_mouse = float(saved_params.loc[saved_params['name']=='sides['+str(ms_i+1)+']', 'Mean'])
                    laser_mouse = float(saved_params.loc[saved_params['name']=='laser_ses['+str(ms_i+1)+']', 'Mean'])
                    alpha = float(saved_params.loc[saved_params['name']=='alpha_ses['+str(ms_i+1)+']', 'Mean'])
                    alphalaser = float(saved_params.loc[saved_params['name']=='alphalaser_ses['+str(ms_i+1)+']', 'Mean'])
                    alphastay = float(saved_params.loc[saved_params['name']=='alphastay_ses['+str(ms_i+1)+']', 'Mean'])
                    betalaserbase= float(saved_params.loc[saved_params['name']=='betalaserbase_ses['+str(ms_i+1)+']', 'Mean'])
                    laserdecay= float(saved_params.loc[saved_params['name']=='laserdecay_ses['+str(ms_i+1)+']', 'Mean'])
            #    else:
            #        print('Error: Only csv version available at the moment!')
            q = np.zeros(2)
            qstay = np.zeros(2)
            qlaser = np.zeros(2)
            predicted_choices=[]
            QL=[]
            QR=[]
            QLstay=[]
            QRstay=[]
            QLlaser=[]
            QRlaser=[]
            choices=[]
            QLgeneral=[]
            QRgeneral=[]
            betalaser_buffer=[]
            betalaser = laser_mouse
            for t in np.arange(NT_all[ms_i]):
                t = int(t)
                predicted_choice = inv_logit(side_mouse
                    + beta_mouse  * (q[1] - q[0])
                    + stay_mouse * (qstay[1] - qstay[0])
                    + (betalaser+betalaserbase) * (qlaser[1] - qlaser[0]))
                choice = c[sub_idx[ms_i], sess_idx[ms_i], t]
                q[choice] = (1-alpha) * q[choice] + alpha * r[sub_idx[ms_i], sess_idx[ms_i],t]
                qlaser[choice] = (1-alphalaser) * qlaser[choice] + alphalaser * l[sub_idx[ms_i], sess_idx[ms_i],t]
                if l[sub_idx[ms_i], sess_idx[ms_i], t] == 1:
                    betalaser = (1-laserdecay)*betalaser
                qstay = qstay * (1 - alphastay)
                qstay[choice] = qstay[choice] + alphastay
                # Store variables
                predicted_choices.append(predicted_choice)
                choices.append(choice)
                QL.append(q[0])
                QR.append(q[1])
                QLstay.append(qstay[0])
                QRstay.append(qstay[1])
                QLlaser.append(qlaser[0])
                QRlaser.append(qlaser[1])
                QLgeneral.append((beta_mouse*q[0])+(laser_mouse*qlaser[0])+(stay_mouse*qstay[0]))
                QRgeneral.append((beta_mouse*q[1])+(laser_mouse*qlaser[1])+(stay_mouse*qstay[1]))
                betalaser_buffer.append(betalaser+betalaserbase)
            acc = []
            for i in np.arange(len(choices)):
                acc.append(1*(choices[i]==(1*(predicted_choices[i]>0.5))))
            #print(np.mean(acc))

            if NT_all[ms_i]!=0:
                ses_data = pd.DataFrame()
                ses_data['predicted_choice'] = predicted_choices
                ses_data['QL'] = QLgeneral
                ses_data['QR'] = QRgeneral
                ses_data['QLreward'] = np.array(QL) * beta_mouse # Note this QL is the reward  - TODO change name
                ses_data['QRreward'] =  np.array(QR) * beta_mouse
                ses_data['QLstay'] =  np.array(QLstay) * stay_mouse
                ses_data['QRstay'] =  np.array(QRstay) * stay_mouse
                ses_data['QLlaser'] =  np.array(QLlaser) * laser_mouse
                ses_data['QRlaser'] =  np.array(QRlaser) * laser_mouse
                ses_data['deltaQ'] = ses_data['QR'] - ses_data['QL']
                ses_data['choices'] = choices
                ses_data['mouse'] = sub_idx[ms_i]
                ses_data['ses'] = sess_idx[ms_i]
                ses_data['acc'] = np.mean(acc)
                ses_data['betalaser'] = np.array(betalaser_buffer)
                data = pd.concat([data,ses_data])
    return make_deltas(data)
def simulate_q_learning_model_new(standata_recovery,saved_params=None, fit=None, csv=True):
    NS =standata_recovery['NS']
    NSESS=standata_recovery['NSESS']
    NT =standata_recovery['NT']
    NT_all =standata_recovery['NT_all']
    r_sim = np.zeros([NS,NSESS,NT])
    c_sim = np.zeros([NS,NSESS,NT])
    l_sim = np.zeros([NS,NSESS,NT])
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
                    beta_mouse = float(saved_params.loc[saved_params['name']=='beta_ses['+str(ms_i+1)+']', 'Mean'])
                    stay_mouse = float(saved_params.loc[saved_params['name']=='stay_ses['+str(ms_i+1)+']', 'Mean'])
                    side_mouse = float(saved_params.loc[saved_params['name']=='sides['+str(ms_i+1)+']', 'Mean'])
                    laser_mouse = float(saved_params.loc[saved_params['name']=='laser_ses['+str(ms_i+1)+']', 'Mean'])
                    alpha = float(saved_params.loc[saved_params['name']=='alpha_ses['+str(ms_i+1)+']', 'Mean'])
                    alphalaser = float(saved_params.loc[saved_params['name']=='alphalaser_ses['+str(ms_i+1)+']', 'Mean'])
                    alphastay = float(saved_params.loc[saved_params['name']=='alphastay_ses['+str(ms_i+1)+']', 'Mean'])
                else:
                    print('Error: Only csv version available at the moment!')
            else:
                if ms_i==0:
                    print('No saved parameters')
                beta_mouse = fit.extract()['beta_ses'][:,int(ms_i)].mean()
                stay_mouse = fit.extract()['stay_ses'][:,int(ms_i)].mean()
                side_mouse = fit.extract()['sides'][:,int(ms_i)].mean()
                laser_mouse = fit.extract()['laser_ses'][:,int(ms_i)].mean()
                alpha = phi_approx(fit.extract()['alpha_ses'][:,int(ms_i)].mean())
                alphalaser =phi_approx(fit.extract()['alphalaser_ses'][:,int(ms_i)].mean())
                alphastay = phi_approx(fit.extract()['alphastay_ses'][:,int(ms_i)].mean())

            tph = TrialParamHandler()
            q = np.zeros(2)
            qstay = np.zeros(2)
            qlaser = np.zeros(2)
            predicted_choices=[]
            rewards = []
            lasers = []
            outcome=[]
            opto_blocks=[]
            side_blocks=[]
            trial_within_block=[]
            right_reward=[]
            left_reward=[]

            for t in np.arange(NT_all[ms_i]):
                t = int(t)
                p_choice = inv_logit(side_mouse
                    + beta_mouse  * (q[1] - q[0])
                    + stay_mouse * (qstay[1] - qstay[0])
                    + laser_mouse * (qlaser[1] - qlaser[0]))
                choice =  np.random.choice(2,p= [1-p_choice,p_choice])
                tph.response_side_buffer.append(choice)  # Add choice to response buffer
                p_left= tph.stim_probability_left
                l = tph.opto_block
                if choice==1:
                    f = tph.right_reward
                if choice==0:
                    f = tph.left_reward
                q[choice] = (1-alpha) * q[choice] + alpha * (1*(f*(l==0)))
                qlaser[choice] = (1-alphalaser) * qlaser[choice] + alphalaser * f * l
                qstay = qstay * (1 - alphastay)
                qstay[choice] = qstay[choice] + alphastay
                
                # Store variables
                rewards.append((1*(f*(l==0))))
                lasers.append(f*l)
                outcome.append(f)
                opto_blocks.append(l) 
                side_blocks.append(p_left)
                predicted_choices.append(choice)
                trial_within_block.append(tph.block_trial_num)
                right_reward.append(tph.right_reward)
                left_reward.append(tph.left_reward)
                # Init next trial
                tph =  tph.next_trial()

            if NT_all[ms_i]!=0:
                c_sim[sub_idx[ms_i], sess_idx[ms_i], :int(NT_all[ms_i])]= predicted_choices
                l_sim[sub_idx[ms_i], sess_idx[ms_i], :int(NT_all[ms_i])] = lasers
                r_sim[sub_idx[ms_i], sess_idx[ms_i], :int(NT_all[ms_i])] = rewards
                ses_data = pd.DataFrame()
                ses_data['choices'] = predicted_choices
                ses_data['water'] = rewards
                ses_data['laser'] = lasers
                ses_data['laser_block'] = opto_blocks
                ses_data['tb'] = trial_within_block
                ses_data['probabilityLeft'] = side_blocks
                ses_data['outcome'] = outcome
                ses_data['mouse'] = sub_idx[ms_i]
                ses_data['ses'] = sess_idx[ms_i]
                ses_data['right_reward'] = right_reward
                ses_data['left_reward'] = left_reward
                sim_data = pd.concat([sim_data,ses_data])
    return c_sim, l_sim, r_sim, sim_data

def simulate_q_learning_model_reduced_stay(standata_recovery,saved_params=None, fit=None, csv=True):
    NS =standata_recovery['NS']
    NSESS=standata_recovery['NSESS']
    NT =standata_recovery['NT']
    NT_all =standata_recovery['NT_all']
    r_sim = np.zeros([NS,NSESS,NT])
    c_sim = np.zeros([NS,NSESS,NT])
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
                    beta_mouse = float(saved_params.loc[saved_params['name']=='beta_ses['+str(ms_i+1)+']', 'Mean'])
                    stay_mouse = float(saved_params.loc[saved_params['name']=='stay_ses['+str(ms_i+1)+']', 'Mean'])
                    side_mouse = float(saved_params.loc[saved_params['name']=='sides['+str(ms_i+1)+']', 'Mean'])
                    alpha = float(saved_params.loc[saved_params['name']=='alpha_ses['+str(ms_i+1)+']', 'Mean'])
                    alphastay = float(saved_params.loc[saved_params['name']=='alphastay_ses['+str(ms_i+1)+']', 'Mean'])
                else:
                    print('Error: Only csv version available at the moment!')
            else:
                if ms_i==0:
                    print('No saved parameters')
                beta_mouse = fit.extract()['beta_ses'][:,int(ms_i)].mean()
                stay_mouse = fit.extract()['stay_ses'][:,int(ms_i)].mean()
                side_mouse = fit.extract()['sides'][:,int(ms_i)].mean()
                alpha = phi_approx(fit.extract()['alpha_ses'][:,int(ms_i)].mean())
                alphastay = phi_approx(fit.extract()['alphastay_ses'][:,int(ms_i)].mean())

            tph = TrialParamHandler()
            q = np.zeros(2)
            qstay = np.zeros(2)
            predicted_choices=[]
            rewards = []
            outcome=[]
            side_blocks=[]
            trial_within_block=[]
            right_reward=[]
            left_reward=[]

            for t in np.arange(NT_all[ms_i]):
                t = int(t)
                p_choice = inv_logit(side_mouse
                    + beta_mouse  * (q[1] - q[0])
                    + stay_mouse * (qstay[1] - qstay[0]))
                choice =  np.random.choice(2,p= [1-p_choice,p_choice])
                tph.response_side_buffer.append(choice)  # Add choice to response buffer
                p_left= tph.stim_probability_left
                if choice==1:
                    f = tph.right_reward
                if choice==0:
                    f = tph.left_reward
                q[choice] = (1-alpha) * q[choice] + alpha * (1*f)
                qstay = qstay * (1 - alphastay)
                qstay[choice] = qstay[choice] + alphastay
                
                # Store variables
                rewards.append(1*f)
                outcome.append(f)
                side_blocks.append(p_left)
                predicted_choices.append(choice)
                trial_within_block.append(tph.block_trial_num)
                right_reward.append(tph.right_reward)
                left_reward.append(tph.left_reward)
                # Init next trial
                tph =  tph.next_trial()

            if NT_all[ms_i]!=0:
                c_sim[sub_idx[ms_i], sess_idx[ms_i], :int(NT_all[ms_i])]= predicted_choices
                r_sim[sub_idx[ms_i], sess_idx[ms_i], :int(NT_all[ms_i])] = rewards
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
                sim_data = pd.concat([sim_data,ses_data])
    return c_sim, r_sim, sim_data

def simulate_q_learning_model_noqlaser_new(standata_recovery,saved_params=None, fit=None, csv=True):
    NS =standata_recovery['NS']
    NSESS=standata_recovery['NSESS']
    NT =standata_recovery['NT']
    NT_all =standata_recovery['NT_all']
    r_sim = np.zeros([NS,NSESS,NT])
    c_sim = np.zeros([NS,NSESS,NT])
    l_sim = np.zeros([NS,NSESS,NT])
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
                    beta_mouse = float(saved_params.loc[saved_params['name']=='beta_ses['+str(ms_i+1)+']', 'Mean'])
                    stay_mouse = float(saved_params.loc[saved_params['name']=='stay_ses['+str(ms_i+1)+']', 'Mean'])
                    side_mouse = float(saved_params.loc[saved_params['name']=='sides['+str(ms_i+1)+']', 'Mean'])
                    laser_mouse = float(saved_params.loc[saved_params['name']=='laser_ses['+str(ms_i+1)+']', 'Mean'])
                    alpha = float(saved_params.loc[saved_params['name']=='alpha_ses['+str(ms_i+1)+']', 'Mean'])
                    alphalaser = float(saved_params.loc[saved_params['name']=='alphalaser_ses['+str(ms_i+1)+']', 'Mean'])
                    alphastay = float(saved_params.loc[saved_params['name']=='alphastay_ses['+str(ms_i+1)+']', 'Mean'])
                else:
                    print('Error: Only csv version available at the moment!')
            else:
                if ms_i==0:
                    print('No saved parameters')
                beta_mouse = fit.extract()['beta_ses'][:,int(ms_i)].mean()
                stay_mouse = fit.extract()['stay_ses'][:,int(ms_i)].mean()
                side_mouse = fit.extract()['sides'][:,int(ms_i)].mean()
                laser_mouse = fit.extract()['laser_ses'][:,int(ms_i)].mean()
                alpha = phi_approx(fit.extract()['alpha_ses'][:,int(ms_i)].mean())
                alphalaser =phi_approx(fit.extract()['alphalaser_ses'][:,int(ms_i)].mean())
                alphastay = phi_approx(fit.extract()['alphastay_ses'][:,int(ms_i)].mean())

            tph = TrialParamHandler()
            q = np.zeros(2)
            qstay = np.zeros(2)
            qlaser = np.zeros(2)
            predicted_choices=[]
            rewards = []
            lasers = []
            outcome=[]
            opto_blocks=[]
            side_blocks=[]
            trial_within_block=[]

            for t in np.arange(NT_all[ms_i]):
                t = int(t)
                p_choice = inv_logit(side_mouse
                    + beta_mouse  * (q[1] - q[0])
                    + stay_mouse * (qstay[1] - qstay[0]))
                choice =  np.random.choice(2,p= [1-p_choice,p_choice])
                tph.response_side_buffer.append(choice)  # Add choice to response buffer
                p_left= tph.stim_probability_left
                l = tph.opto_block
                if choice==1:
                    f = tph.right_reward
                if choice==0:
                    f = tph.left_reward
                q[choice] = (1-alpha) * q[choice] + alpha * (1*(f*(l==0)))
                qstay = qstay * (1 - alphastay)
                qstay[choice] = qstay[choice] + alphastay
                
                # Store variables
                rewards.append((1*(f*(l==0))))
                lasers.append(f*l)
                outcome.append(f)
                opto_blocks.append(l) 
                side_blocks.append(p_left)
                predicted_choices.append(choice)
                trial_within_block.append(tph.block_trial_num)

                # Init next trial
                tph =  tph.next_trial()

            if NT_all[ms_i]!=0:
                c_sim[sub_idx[ms_i], sess_idx[ms_i], :int(NT_all[ms_i])]= predicted_choices
                l_sim[sub_idx[ms_i], sess_idx[ms_i], :int(NT_all[ms_i])] = lasers
                r_sim[sub_idx[ms_i], sess_idx[ms_i], :int(NT_all[ms_i])] = rewards
                ses_data = pd.DataFrame()
                ses_data['choices'] = predicted_choices
                ses_data['water'] = rewards
                ses_data['laser'] = lasers
                ses_data['laser_block'] = opto_blocks
                ses_data['tb'] = trial_within_block
                ses_data['probabilityLeft'] = side_blocks
                ses_data['outcome'] = outcome
                ses_data['mouse'] = sub_idx[ms_i]
                ses_data['ses'] = sess_idx[ms_i]
                sim_data = pd.concat([sim_data,ses_data])
    return c_sim, l_sim, r_sim, sim_data
    
def simulate_q_learning_model_noqwater_new(standata_recovery,saved_params=None, fit=None, csv=True):
    NS =standata_recovery['NS']
    NSESS=standata_recovery['NSESS']
    NT =standata_recovery['NT']
    NT_all =standata_recovery['NT_all']
    r_sim = np.zeros([NS,NSESS,NT])
    c_sim = np.zeros([NS,NSESS,NT])
    l_sim = np.zeros([NS,NSESS,NT])
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
                    beta_mouse = float(saved_params.loc[saved_params['name']=='beta_ses['+str(ms_i+1)+']', 'Mean'])
                    stay_mouse = float(saved_params.loc[saved_params['name']=='stay_ses['+str(ms_i+1)+']', 'Mean'])
                    side_mouse = float(saved_params.loc[saved_params['name']=='sides['+str(ms_i+1)+']', 'Mean'])
                    laser_mouse = float(saved_params.loc[saved_params['name']=='laser_ses['+str(ms_i+1)+']', 'Mean'])
                    alpha = float(saved_params.loc[saved_params['name']=='alpha_ses['+str(ms_i+1)+']', 'Mean'])
                    alphalaser = float(saved_params.loc[saved_params['name']=='alphalaser_ses['+str(ms_i+1)+']', 'Mean'])
                    alphastay = float(saved_params.loc[saved_params['name']=='alphastay_ses['+str(ms_i+1)+']', 'Mean'])
                else:
                    print('Error: Only csv version available at the moment!')
            else:
                if ms_i==0:
                    print('No saved parameters')
                beta_mouse = fit.extract()['beta_ses'][:,int(ms_i)].mean()
                stay_mouse = fit.extract()['stay_ses'][:,int(ms_i)].mean()
                side_mouse = fit.extract()['sides'][:,int(ms_i)].mean()
                laser_mouse = fit.extract()['laser_ses'][:,int(ms_i)].mean()
                alpha = phi_approx(fit.extract()['alpha_ses'][:,int(ms_i)].mean())
                alphalaser =phi_approx(fit.extract()['alphalaser_ses'][:,int(ms_i)].mean())
                alphastay = phi_approx(fit.extract()['alphastay_ses'][:,int(ms_i)].mean())

            tph = TrialParamHandler()
            q = np.zeros(2)
            qstay = np.zeros(2)
            qlaser = np.zeros(2)
            predicted_choices=[]
            rewards = []
            lasers = []
            outcome=[]
            opto_blocks=[]
            side_blocks=[]
            trial_within_block=[]

            for t in np.arange(NT_all[ms_i]):
                t = int(t)
                p_choice = inv_logit(side_mouse
                    + stay_mouse * (qstay[1] - qstay[0])
                    + laser_mouse * (qlaser[1] - qlaser[0]))
                choice =  np.random.choice(2,p= [1-p_choice,p_choice])
                tph.response_side_buffer.append(choice)  # Add choice to response buffer
                p_left= tph.stim_probability_left
                l = tph.opto_block
                if choice==1:
                    f = tph.right_reward
                if choice==0:
                    f = tph.left_reward
                qlaser[choice] = (1-alphalaser) * qlaser[choice] + alphalaser * f * l
                qstay = qstay * (1 - alphastay)
                qstay[choice] = qstay[choice] + alphastay
                
                # Store variables
                rewards.append((1*(f*(l==0))))
                lasers.append(f*l)
                outcome.append(f)
                opto_blocks.append(l) 
                side_blocks.append(p_left)
                predicted_choices.append(choice)
                trial_within_block.append(tph.block_trial_num)

                # Init next trial
                tph =  tph.next_trial()

            if NT_all[ms_i]!=0:
                c_sim[sub_idx[ms_i], sess_idx[ms_i], :int(NT_all[ms_i])]= predicted_choices
                l_sim[sub_idx[ms_i], sess_idx[ms_i], :int(NT_all[ms_i])] = lasers
                r_sim[sub_idx[ms_i], sess_idx[ms_i], :int(NT_all[ms_i])] = rewards
                ses_data = pd.DataFrame()
                ses_data['choices'] = predicted_choices
                ses_data['water'] = rewards
                ses_data['laser'] = lasers
                ses_data['laser_block'] = opto_blocks
                ses_data['tb'] = trial_within_block
                ses_data['probabilityLeft'] = side_blocks
                ses_data['outcome'] = outcome
                ses_data['mouse'] = sub_idx[ms_i]
                ses_data['ses'] = sess_idx[ms_i]
                sim_data = pd.concat([sim_data,ses_data])
    return c_sim, l_sim, r_sim, sim_data
def simulate_q_learning_model_noqwaterorlaser_new(standata_recovery,saved_params=None, fit=None, csv=True):
    NS =standata_recovery['NS']
    NSESS=standata_recovery['NSESS']
    NT =standata_recovery['NT']
    NT_all =standata_recovery['NT_all']
    r_sim = np.zeros([NS,NSESS,NT])
    c_sim = np.zeros([NS,NSESS,NT])
    l_sim = np.zeros([NS,NSESS,NT])
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
                    beta_mouse = float(saved_params.loc[saved_params['name']=='beta_ses['+str(ms_i+1)+']', 'Mean'])
                    stay_mouse = float(saved_params.loc[saved_params['name']=='stay_ses['+str(ms_i+1)+']', 'Mean'])
                    side_mouse = float(saved_params.loc[saved_params['name']=='sides['+str(ms_i+1)+']', 'Mean'])
                    laser_mouse = float(saved_params.loc[saved_params['name']=='laser_ses['+str(ms_i+1)+']', 'Mean'])
                    alpha = float(saved_params.loc[saved_params['name']=='alpha_ses['+str(ms_i+1)+']', 'Mean'])
                    alphalaser = float(saved_params.loc[saved_params['name']=='alphalaser_ses['+str(ms_i+1)+']', 'Mean'])
                    alphastay = float(saved_params.loc[saved_params['name']=='alphastay_ses['+str(ms_i+1)+']', 'Mean'])
                else:
                    print('Error: Only csv version available at the moment!')
            else:
                if ms_i==0:
                    print('No saved parameters')
                beta_mouse = fit.extract()['beta_ses'][:,int(ms_i)].mean()
                stay_mouse = fit.extract()['stay_ses'][:,int(ms_i)].mean()
                side_mouse = fit.extract()['sides'][:,int(ms_i)].mean()
                laser_mouse = fit.extract()['laser_ses'][:,int(ms_i)].mean()
                alpha = phi_approx(fit.extract()['alpha_ses'][:,int(ms_i)].mean())
                alphalaser =phi_approx(fit.extract()['alphalaser_ses'][:,int(ms_i)].mean())
                alphastay = phi_approx(fit.extract()['alphastay_ses'][:,int(ms_i)].mean())

            tph = TrialParamHandler()
            q = np.zeros(2)
            qstay = np.zeros(2)
            qlaser = np.zeros(2)
            predicted_choices=[]
            rewards = []
            lasers = []
            outcome=[]
            opto_blocks=[]
            side_blocks=[]
            trial_within_block=[]

            for t in np.arange(NT_all[ms_i]):
                t = int(t)
                p_choice = inv_logit(side_mouse
                    + stay_mouse * (qstay[1] - qstay[0]))
                choice =  np.random.choice(2,p= [1-p_choice,p_choice])
                tph.response_side_buffer.append(choice)  # Add choice to response buffer
                p_left= tph.stim_probability_left
                l = tph.opto_block
                if choice==1:
                    f = tph.right_reward
                if choice==0:
                    f = tph.left_reward
                qlaser[choice] = (1-alphalaser) * qlaser[choice] + alphalaser * f * l
                qstay = qstay * (1 - alphastay)
                qstay[choice] = qstay[choice] + alphastay
                
                # Store variables
                rewards.append((1*(f*(l==0))))
                lasers.append(f*l)
                outcome.append(f)
                opto_blocks.append(l) 
                side_blocks.append(p_left)
                predicted_choices.append(choice)
                trial_within_block.append(tph.block_trial_num)

                # Init next trial
                tph =  tph.next_trial()

            if NT_all[ms_i]!=0:
                c_sim[sub_idx[ms_i], sess_idx[ms_i], :int(NT_all[ms_i])]= predicted_choices
                l_sim[sub_idx[ms_i], sess_idx[ms_i], :int(NT_all[ms_i])] = lasers
                r_sim[sub_idx[ms_i], sess_idx[ms_i], :int(NT_all[ms_i])] = rewards
                ses_data = pd.DataFrame()
                ses_data['choices'] = predicted_choices
                ses_data['water'] = rewards
                ses_data['laser'] = lasers
                ses_data['laser_block'] = opto_blocks
                ses_data['tb'] = trial_within_block
                ses_data['probabilityLeft'] = side_blocks
                ses_data['outcome'] = outcome
                ses_data['mouse'] = sub_idx[ms_i]
                ses_data['ses'] = sess_idx[ms_i]
                sim_data = pd.concat([sim_data,ses_data])
    return c_sim, l_sim, r_sim, sim_data
def simulate_reinforce_laserdecay_winloss_stay_same_trials(standata_recovery,saved_params=None, fit=None, csv=True):
    b =  standata_recovery['b']
    p =  standata_recovery['p']
    tb = standata_recovery['tb']
    NS =standata_recovery['NS']
    NSESS=standata_recovery['NSESS']
    NT =standata_recovery['NT']
    NT_all =standata_recovery['NT_all']
    r_sim = np.zeros([NS,NSESS,NT])
    c_sim = np.zeros([NS,NSESS,NT])
    l_sim = np.zeros([NS,NSESS,NT])
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
                    beta_ses = float(saved_params.loc[saved_params['name']=='beta_ses['+str(ms_i+1)+']', 'Mean'])
                    side_ses = float(saved_params.loc[saved_params['name']=='sides['+str(ms_i+1)+']', 'Mean'])
                    laser_ses = float(saved_params.loc[saved_params['name']=='laser_ses['+str(ms_i+1)+']', 'Mean'])
                    epses = float(saved_params.loc[saved_params['name']=='ep_ses['+str(ms_i+1)+']', 'Mean'])
                    alpha = float(saved_params.loc[saved_params['name']=='alpha_ses['+str(ms_i+1)+']', 'Mean'])
                    alphalaser = float(saved_params.loc[saved_params['name']=='alphalaser_ses['+str(ms_i+1)+']', 'Mean'])
                    alphastay = float(saved_params.loc[saved_params['name']=='alphastay_ses['+str(ms_i+1)+']', 'Mean'])
                    alphaforgetting= float(saved_params.loc[saved_params['name']=='alphaforgetting_ses['+str(ms_i+1)+']', 'Mean'])
                    alphalaserforgetting= float(saved_params.loc[saved_params['name']=='alphalaserforgetting_ses['+str(ms_i+1)+']', 'Mean'])
                    betalaserbase= float(saved_params.loc[saved_params['name']=='betalaserbase_ses['+str(ms_i+1)+']', 'Mean'])
                    dmwater_ses = float(saved_params.loc[saved_params['name']=='dmwater_ses['+str(ms_i+1)+']', 'Mean'])
                    dmlaser_ses = float(saved_params.loc[saved_params['name']=='dmlaser_ses['+str(ms_i+1)+']', 'Mean'])
                    betalaserlossbase= float(saved_params.loc[saved_params['name']=='betalaserlossbase_ses['+str(ms_i+1)+']', 'Mean'])
                    laserdecayloss= float(saved_params.loc[saved_params['name']=='laserdecayloss_ses['+str(ms_i+1)+']', 'Mean'])
                    laserdecay = float(saved_params.loc[saved_params['name']=='laserdecay_ses['+str(ms_i+1)+']', 'Mean'])

                else:
                    print('Error: Only csv version available at the moment!')
            q = 0
            qlaser = 0
            qstay = np.zeros(2)
            betalaser = laser_ses
            predicted_choices=[]
            rewards = []
            lasers = []
            outcome=[]
            opto_blocks=[]
            side_blocks=[]
            trial_within_block=[]
            betalaser_buffer=[]
            buffer_qlaser=[]
            buffer_qwater=[]
            betalaser = laser_ses
            betalaserloss = dmlaser_ses
            print(betalaser)
            print(betalaserloss)
            for t in np.arange(NT_all[ms_i]):
                t = int(t)    
                p_choice = (1-epses) * inv_logit(side_ses + (q + qlaser)) + epses*(inv_logit(qstay[1]-qstay[0]))
                choice =  np.random.choice(2,p= [1-p_choice,p_choice])
                working_choice = 2 * choice -1
                assert np.isin(p[sub_idx[ms_i], sess_idx[ms_i],t], [0.7,0.1])
                p_left= p[sub_idx[ms_i], sess_idx[ms_i],t]
                if p_left==0.7:
                    p_right=0.1
                else:
                    p_right=0.7
                l = b[sub_idx[ms_i], sess_idx[ms_i],t]
                if choice==1:
                    f = np.random.choice([0,1], p=[1-p_right,p_right])
                if choice==0:
                    f = np.random.choice([0,1], p=[1-p_left,p_left])
                if (f*(l==0)) == 1:
                    q = (1 - alpha) * q + working_choice * beta_ses
                if (f*(l==0)) == 0:
                    q = (1 - alphaforgetting) * q + working_choice * dmwater_ses
                if (f*l) == 1:
                    qlaser = (1 - alphalaser) * qlaser + working_choice * (betalaser+betalaserbase)
                    betalaser = (1-laserdecay)*betalaser
                if (f*l) == 0:
                    qlaser = (1 - alphalaserforgetting) * qlaser + working_choice * (betalaserloss+betalaserlossbase)               
                    betalaserloss = (1-laserdecayloss)*betalaserloss
                qstay = qstay * (1 - alphastay)
                qstay[choice] = qstay[choice] + alphastay

                # Store variables
                betalaser_buffer.append(betalaser)
                rewards.append((1*(f*(l==0))))
                lasers.append(f*l)
                outcome.append(f)
                opto_blocks.append(l) 
                side_blocks.append(p_left)
                predicted_choices.append(choice)
                trial_within_block.append(tb[sub_idx[ms_i], sess_idx[ms_i],t])
                buffer_qlaser.append(qlaser)
                buffer_qwater.append(q)

            if NT_all[ms_i]!=0:
                c_sim[sub_idx[ms_i], sess_idx[ms_i], :int(NT_all[ms_i])]= predicted_choices
                l_sim[sub_idx[ms_i], sess_idx[ms_i], :int(NT_all[ms_i])] = lasers
                r_sim[sub_idx[ms_i], sess_idx[ms_i], :int(NT_all[ms_i])] = rewards
                ses_data = pd.DataFrame()
                ses_data['choices'] = predicted_choices
                ses_data['b_init_laser'] = laser_ses
                ses_data['beta_laser'] = betalaser_buffer
                ses_data['water'] = rewards
                ses_data['laser'] = lasers
                ses_data['laser_block'] = opto_blocks
                ses_data['tb'] = trial_within_block
                ses_data['probabilityLeft'] = side_blocks
                ses_data['outcome'] = outcome
                ses_data['mouse'] = sub_idx[ms_i]
                ses_data['ses'] = sess_idx[ms_i]
                ses_data['qwater'] = buffer_qwater
                ses_data['qlaser'] = buffer_qlaser
                sim_data = pd.concat([sim_data,ses_data])
    return c_sim, l_sim, r_sim, sim_data
def simulate_reinforce_laserdecay_winloss_stay(standata_recovery,saved_params=None, fit=None, csv=True):
    NS =standata_recovery['NS']
    NSESS=standata_recovery['NSESS']
    NT =standata_recovery['NT']
    NT_all =standata_recovery['NT_all']
    r_sim = np.zeros([NS,NSESS,NT])
    c_sim = np.zeros([NS,NSESS,NT])
    l_sim = np.zeros([NS,NSESS,NT])
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
                    beta_ses = float(saved_params.loc[saved_params['name']=='beta_ses['+str(ms_i+1)+']', 'Mean'])
                    side_ses = float(saved_params.loc[saved_params['name']=='sides['+str(ms_i+1)+']', 'Mean'])
                    laser_ses = float(saved_params.loc[saved_params['name']=='laser_ses['+str(ms_i+1)+']', 'Mean'])
                    epses = float(saved_params.loc[saved_params['name']=='ep_ses['+str(ms_i+1)+']', 'Mean'])
                    alpha = float(saved_params.loc[saved_params['name']=='alpha_ses['+str(ms_i+1)+']', 'Mean'])
                    alphalaser = float(saved_params.loc[saved_params['name']=='alphalaser_ses['+str(ms_i+1)+']', 'Mean'])
                    alphastay = float(saved_params.loc[saved_params['name']=='alphastay_ses['+str(ms_i+1)+']', 'Mean'])
                    alphaforgetting= float(saved_params.loc[saved_params['name']=='alphaforgetting_ses['+str(ms_i+1)+']', 'Mean'])
                    alphalaserforgetting= float(saved_params.loc[saved_params['name']=='alphalaserforgetting_ses['+str(ms_i+1)+']', 'Mean'])
                    betalaserbase= float(saved_params.loc[saved_params['name']=='betalaserbase_ses['+str(ms_i+1)+']', 'Mean'])
                    dmwater_ses = float(saved_params.loc[saved_params['name']=='dmwater_ses['+str(ms_i+1)+']', 'Mean'])
                    dmlaser_ses = float(saved_params.loc[saved_params['name']=='dmlaser_ses['+str(ms_i+1)+']', 'Mean'])
                    betalaserlossbase= float(saved_params.loc[saved_params['name']=='betalaserlossbase_ses['+str(ms_i+1)+']', 'Mean'])
                    laserdecayloss= float(saved_params.loc[saved_params['name']=='laserdecayloss_ses['+str(ms_i+1)+']', 'Mean'])
                    laserdecay = float(saved_params.loc[saved_params['name']=='laserdecay_ses['+str(ms_i+1)+']', 'Mean'])

                else:
                    print('Error: Only csv version available at the moment!')

            tph = TrialParamHandler()
            q = 0
            qlaser = 0
            qstay = np.zeros(2)
            betalaser = laser_ses
            predicted_choices=[]
            rewards = []
            lasers = []
            outcome=[]
            opto_blocks=[]
            side_blocks=[]
            trial_within_block=[]
            right_reward=[]
            left_reward=[]
            betalaser_buffer=[]
            buffer_qlaser=[]
            buffer_qwater=[]
            betalaser = laser_ses
            betalaserloss = dmlaser_ses
            for t in np.arange(NT_all[ms_i]):
                t = int(t)    
                p_choice = (1-epses) * inv_logit(side_ses + (q + qlaser)) + epses*(inv_logit(qstay[1]-qstay[0]))
                choice =  np.random.choice(2,p= [1-p_choice,p_choice])
                working_choice = 2 * choice -1
                tph.response_side_buffer.append(choice)  # Add choice to response buffer
                p_left= tph.stim_probability_left
                l = tph.opto_block
                if choice==1:
                    f = tph.right_reward
                if choice==0:
                    f = tph.left_reward
                if (f*(l==0)) == 1:
                    q = (1 - alpha) * q + working_choice * beta_ses
                if (f*(l==0)) == 0:
                    q = (1 - alphaforgetting) * q + working_choice * dmwater_ses
                if (f*l) == 1:
                    betalaser = (1-laserdecay)*betalaser
                    qlaser = (1 - alphalaser) * qlaser + working_choice * (betalaser+betalaserbase)
                if (f*l) == 0:
                    betalaserloss = (1-laserdecayloss)*betalaserloss
                    qlaser = (1 - alphalaserforgetting) * qlaser + working_choice * (betalaserloss+betalaserlossbase)               
                qstay = qstay * (1 - alphastay)
                qstay[choice] = qstay[choice] + alphastay

                # Store variables
                betalaser_buffer.append(betalaser)
                rewards.append((1*(f*(l==0))))
                lasers.append(f*l)
                outcome.append(f)
                opto_blocks.append(l) 
                side_blocks.append(p_left)
                predicted_choices.append(choice)
                trial_within_block.append(tph.block_trial_num)
                right_reward.append(tph.right_reward)
                left_reward.append(tph.left_reward)
                buffer_qlaser.append(qlaser)
                buffer_qwater.append(q)
                # Init next trial
                tph =  tph.next_trial()

            if NT_all[ms_i]!=0:
                c_sim[sub_idx[ms_i], sess_idx[ms_i], :int(NT_all[ms_i])]= predicted_choices
                l_sim[sub_idx[ms_i], sess_idx[ms_i], :int(NT_all[ms_i])] = lasers
                r_sim[sub_idx[ms_i], sess_idx[ms_i], :int(NT_all[ms_i])] = rewards
                ses_data = pd.DataFrame()
                ses_data['choices'] = predicted_choices
                ses_data['b_init_laser'] = laser_ses
                ses_data['beta_laser'] = betalaser_buffer
                ses_data['water'] = rewards
                ses_data['laser'] = lasers
                ses_data['laser_block'] = opto_blocks
                ses_data['tb'] = trial_within_block
                ses_data['probabilityLeft'] = side_blocks
                ses_data['outcome'] = outcome
                ses_data['mouse'] = sub_idx[ms_i]
                ses_data['ses'] = sess_idx[ms_i]
                ses_data['right_reward'] = right_reward
                ses_data['left_reward'] = left_reward
                ses_data['qwater'] = buffer_qwater
                ses_data['qlaser'] = buffer_qlaser
                sim_data = pd.concat([sim_data,ses_data])
    return c_sim, l_sim, r_sim, sim_data


def simulate_reinforce_reduced_stay(standata_recovery,saved_params=None, fit=None, csv=True):
    NS =standata_recovery['NS']
    NSESS=standata_recovery['NSESS']
    NT =standata_recovery['NT']
    NT_all =standata_recovery['NT_all']
    r_sim = np.zeros([NS,NSESS,NT])
    c_sim = np.zeros([NS,NSESS,NT])
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
                    epses = float(saved_params.loc[saved_params['name']=='ep_ses['+str(ms_i+1)+']', 'Mean'])
                    alpha = float(saved_params.loc[saved_params['name']=='alpha_ses['+str(ms_i+1)+']', 'Mean'])
                    alphastay = float(saved_params.loc[saved_params['name']=='alphastay_ses['+str(ms_i+1)+']', 'Mean'])
                    alphaforgetting= float(saved_params.loc[saved_params['name']=='alphaforgetting_ses['+str(ms_i+1)+']', 'Mean'])
                else:
                    print('Error: Only csv version available at the moment!')

            tph = TrialParamHandler()
            q = 0
            qstay = np.zeros(2)
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
                p_choice = (1-epses) * inv_logit(side_ses + q) + epses*(inv_logit(qstay[1]-qstay[0]))
                choice =  np.random.choice(2,p= [1-p_choice,p_choice])
                working_choice = 2 * choice -1
                tph.response_side_buffer.append(choice)  # Add choice to response buffer
                p_left= tph.stim_probability_left
                if choice==1:
                    f = tph.right_reward
                if choice==0:
                    f = tph.left_reward
                if f == 1:
                    q = (1 - alpha) * q + working_choice * beta_ses
                if f == 0:
                    q = (1 - alphaforgetting) * q + working_choice * dmwater_ses
                qstay = qstay * (1 - alphastay)
                qstay[choice] = qstay[choice] + alphastay

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
                c_sim[sub_idx[ms_i], sess_idx[ms_i], :int(NT_all[ms_i])]= predicted_choices
                r_sim[sub_idx[ms_i], sess_idx[ms_i], :int(NT_all[ms_i])] = rewards
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
    return c_sim, r_sim, sim_data

def simulate_reinforce_reduced(standata_recovery,saved_params=None, fit=None, csv=True):
    NS =standata_recovery['NS']
    NSESS=standata_recovery['NSESS']
    NT =standata_recovery['NT']
    NT_all =standata_recovery['NT_all']
    r_sim = np.zeros([NS,NSESS,NT])
    c_sim = np.zeros([NS,NSESS,NT])
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
                    print('Error: Only csv version available at the moment!')

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
                    q = (1 - alpha) * q + working_choice * beta_ses
                if f == 0:
                    q = (1 - alphaforgetting) * q + working_choice * dmwater_ses

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
                c_sim[sub_idx[ms_i], sess_idx[ms_i], :int(NT_all[ms_i])]= predicted_choices
                r_sim[sub_idx[ms_i], sess_idx[ms_i], :int(NT_all[ms_i])] = rewards
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
    return c_sim, r_sim, sim_data





def simulate_reinforce_alphalaserdecay_win_stay(standata_recovery,saved_params=None, fit=None, csv=True):
    NS =standata_recovery['NS']
    NSESS=standata_recovery['NSESS']
    NT =standata_recovery['NT']
    NT_all =standata_recovery['NT_all']
    r_sim = np.zeros([NS,NSESS,NT])
    c_sim = np.zeros([NS,NSESS,NT])
    l_sim = np.zeros([NS,NSESS,NT])
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
                    beta_ses = float(saved_params.loc[saved_params['name']=='beta_ses['+str(ms_i+1)+']', 'Mean'])
                    side_ses = float(saved_params.loc[saved_params['name']=='sides['+str(ms_i+1)+']', 'Mean'])
                    laser_ses = float(saved_params.loc[saved_params['name']=='laser_ses['+str(ms_i+1)+']', 'Mean'])
                    epses = float(saved_params.loc[saved_params['name']=='ep_ses['+str(ms_i+1)+']', 'Mean'])
                    alpha = float(saved_params.loc[saved_params['name']=='alpha_ses['+str(ms_i+1)+']', 'Mean'])
                    alphalaser = float(saved_params.loc[saved_params['name']=='alphalaser_ses['+str(ms_i+1)+']', 'Mean'])
                    alphastay = float(saved_params.loc[saved_params['name']=='alphastay_ses['+str(ms_i+1)+']', 'Mean'])
                    alphaforgetting= float(saved_params.loc[saved_params['name']=='alphaforgetting_ses['+str(ms_i+1)+']', 'Mean'])
                    alphalaserforgetting= float(saved_params.loc[saved_params['name']=='alphalaserforgetting_ses['+str(ms_i+1)+']', 'Mean'])
                    betalaserbase= float(saved_params.loc[saved_params['name']=='betalaserbase_ses['+str(ms_i+1)+']', 'Mean'])
                    dmwater_ses = float(saved_params.loc[saved_params['name']=='dmwater_ses['+str(ms_i+1)+']', 'Mean'])
                    dmlaser_ses = float(saved_params.loc[saved_params['name']=='dmlaser_ses['+str(ms_i+1)+']', 'Mean'])
                    laserdecay = float(saved_params.loc[saved_params['name']=='laserdecay_ses['+str(ms_i+1)+']', 'Mean'])

                else:
                    print('Error: Only csv version available at the moment!')

            tph = TrialParamHandler()
            q = 0
            qlaser = 0
            qstay = np.zeros(2)
            predicted_choices=[]
            rewards = []
            lasers = []
            outcome=[]
            opto_blocks=[]
            side_blocks=[]
            trial_within_block=[]
            right_reward=[]
            left_reward=[]
            betalaser_buffer=[]
            buffer_qlaser=[]
            buffer_qwater=[]
            betalaser = laser_ses
            betalaserloss = dmlaser_ses
            for t in np.arange(NT_all[ms_i]):
                t = int(t)    
                p_choice = (1-epses) * inv_logit(side_ses + (q + qlaser)) + epses*(inv_logit(qstay[1]-qstay[0]))
                choice =  np.random.choice(2,p= [1-p_choice,p_choice])
                working_choice = 2 * choice -1
                tph.response_side_buffer.append(choice)  # Add choice to response buffer
                p_left= tph.stim_probability_left
                l = tph.opto_block
                if choice==1:
                    f = tph.right_reward
                if choice==0:
                    f = tph.left_reward
                if (f*(l==0)) == 1:
                    q = (1 - alpha) * q + working_choice * beta_ses
                if (f*(l==0)) == 0:
                    q = (1 - alphaforgetting) * q + working_choice * dmwater_ses
                if (f*l) == 1:
                    betalaser = (1-laserdecay)*betalaser
                    qlaser = (1 - (betalaser+betalaserbase)) * qlaser + working_choice * alphalaser
                if (f*l) == 0:
                    qlaser = (1 - alphalaserforgetting) * qlaser + working_choice * (betalaserloss)               
                qstay = qstay * (1 - alphastay)
                qstay[choice] = qstay[choice] + alphastay

                # Store variables
                betalaser_buffer.append(betalaser)
                rewards.append((1*(f*(l==0))))
                lasers.append(f*l)
                outcome.append(f)
                opto_blocks.append(l) 
                side_blocks.append(p_left)
                predicted_choices.append(choice)
                trial_within_block.append(tph.block_trial_num)
                right_reward.append(tph.right_reward)
                left_reward.append(tph.left_reward)
                buffer_qlaser.append(qlaser)
                buffer_qwater.append(q)
                # Init next trial
                tph =  tph.next_trial()

            if NT_all[ms_i]!=0:
                c_sim[sub_idx[ms_i], sess_idx[ms_i], :int(NT_all[ms_i])]= predicted_choices
                l_sim[sub_idx[ms_i], sess_idx[ms_i], :int(NT_all[ms_i])] = lasers
                r_sim[sub_idx[ms_i], sess_idx[ms_i], :int(NT_all[ms_i])] = rewards
                ses_data = pd.DataFrame()
                ses_data['choices'] = predicted_choices
                ses_data['b_init_laser'] = laser_ses
                ses_data['beta_laser'] = betalaser_buffer
                ses_data['water'] = rewards
                ses_data['laser'] = lasers
                ses_data['laser_block'] = opto_blocks
                ses_data['tb'] = trial_within_block
                ses_data['probabilityLeft'] = side_blocks
                ses_data['outcome'] = outcome
                ses_data['mouse'] = sub_idx[ms_i]
                ses_data['ses'] = sess_idx[ms_i]
                ses_data['right_reward'] = right_reward
                ses_data['left_reward'] = left_reward
                ses_data['qwater'] = buffer_qwater
                ses_data['qlaser'] = buffer_qlaser
                sim_data = pd.concat([sim_data,ses_data])
    return c_sim, l_sim, r_sim, sim_data
def simulate_reinforce_laserdecay_winloss_stay_nolaser(standata_recovery,saved_params=None, fit=None, csv=True):
    NS =standata_recovery['NS']
    NSESS=standata_recovery['NSESS']
    NT =standata_recovery['NT']
    NT_all =standata_recovery['NT_all']
    r_sim = np.zeros([NS,NSESS,NT])
    c_sim = np.zeros([NS,NSESS,NT])
    l_sim = np.zeros([NS,NSESS,NT])
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
                    beta_ses = float(saved_params.loc[saved_params['name']=='beta_ses['+str(ms_i+1)+']', 'Mean'])
                    side_ses = float(saved_params.loc[saved_params['name']=='sides['+str(ms_i+1)+']', 'Mean'])
                    laser_ses = float(saved_params.loc[saved_params['name']=='laser_ses['+str(ms_i+1)+']', 'Mean'])
                    epses = float(saved_params.loc[saved_params['name']=='ep_ses['+str(ms_i+1)+']', 'Mean'])
                    alpha = float(saved_params.loc[saved_params['name']=='alpha_ses['+str(ms_i+1)+']', 'Mean'])
                    alphalaser = float(saved_params.loc[saved_params['name']=='alphalaser_ses['+str(ms_i+1)+']', 'Mean'])
                    alphastay = float(saved_params.loc[saved_params['name']=='alphastay_ses['+str(ms_i+1)+']', 'Mean'])
                    alphaforgetting= float(saved_params.loc[saved_params['name']=='alphaforgetting_ses['+str(ms_i+1)+']', 'Mean'])
                    alphalaserforgetting= float(saved_params.loc[saved_params['name']=='alphalaserforgetting_ses['+str(ms_i+1)+']', 'Mean'])
                    laserdecay= float(saved_params.loc[saved_params['name']=='laserdecay_ses['+str(ms_i+1)+']', 'Mean'])
                    betalaserbase= float(saved_params.loc[saved_params['name']=='betalaserbase_ses['+str(ms_i+1)+']', 'Mean'])
                    dmwater_ses = float(saved_params.loc[saved_params['name']=='dmwater_ses['+str(ms_i+1)+']', 'Mean'])
                    dmlaser_ses = float(saved_params.loc[saved_params['name']=='dmlaser_ses['+str(ms_i+1)+']', 'Mean'])
                    betalaserlossbase= float(saved_params.loc[saved_params['name']=='betalaserlossbase_ses['+str(ms_i+1)+']', 'Mean'])
                    laserdecayloss= float(saved_params.loc[saved_params['name']=='laserdecayloss_ses['+str(ms_i+1)+']', 'Mean'])
                else:
                    print('Error: Only csv version available at the moment!')

            tph = TrialParamHandler()
            q = 0
            qlaser = 0
            qstay = np.zeros(2)
            betalaser = laser_ses
            predicted_choices=[]
            rewards = []
            lasers = []
            outcome=[]
            opto_blocks=[]
            side_blocks=[]
            trial_within_block=[]
            right_reward=[]
            left_reward=[]
            betalaser_buffer=[]
            buffer_qlaser=[]
            buffer_qwater=[]
            betalaser = laser_ses
            betalaserloss = dmlaser_ses
            for t in np.arange(NT_all[ms_i]):
                t = int(t)    
                p_choice = (1-epses) * inv_logit(side_ses + q) + epses*(inv_logit(qstay[1]-qstay[0]))
                choice =  np.random.choice(2,p= [1-p_choice,p_choice])
                working_choice = 2 * choice -1
                tph.response_side_buffer.append(choice)  # Add choice to response buffer
                p_left= tph.stim_probability_left
                l = tph.opto_block
                if choice==1:
                    f = tph.right_reward
                if choice==0:
                    f = tph.left_reward
                if (f*(l==0)) == 1:
                    q = (1 - alpha) * q + working_choice * beta_ses
                if (f*(l==0)) == 0:
                    q = (1 - alphaforgetting) * q + working_choice * dmwater_ses           
                qstay = qstay * (1 - alphastay)
                qstay[choice] = qstay[choice] + alphastay

                # Store variables
                betalaser_buffer.append(betalaser)
                rewards.append((1*(f*(l==0))))
                lasers.append(f*l)
                outcome.append(f)
                opto_blocks.append(l) 
                side_blocks.append(p_left)
                predicted_choices.append(choice)
                trial_within_block.append(tph.block_trial_num)
                right_reward.append(tph.right_reward)
                left_reward.append(tph.left_reward)
                buffer_qlaser.append(qlaser)
                buffer_qwater.append(q)
                # Init next trial
                tph =  tph.next_trial()

            if NT_all[ms_i]!=0:
                c_sim[sub_idx[ms_i], sess_idx[ms_i], :int(NT_all[ms_i])]= predicted_choices
                l_sim[sub_idx[ms_i], sess_idx[ms_i], :int(NT_all[ms_i])] = lasers
                r_sim[sub_idx[ms_i], sess_idx[ms_i], :int(NT_all[ms_i])] = rewards
                ses_data = pd.DataFrame()
                ses_data['choices'] = predicted_choices
                ses_data['b_init_laser'] = laser_ses
                ses_data['beta_laser'] = betalaser_buffer
                ses_data['water'] = rewards
                ses_data['laser'] = lasers
                ses_data['laser_block'] = opto_blocks
                ses_data['tb'] = trial_within_block
                ses_data['probabilityLeft'] = side_blocks
                ses_data['outcome'] = outcome
                ses_data['mouse'] = sub_idx[ms_i]
                ses_data['ses'] = sess_idx[ms_i]
                ses_data['right_reward'] = right_reward
                ses_data['left_reward'] = left_reward
                ses_data['qwater'] = buffer_qwater
                ses_data['qlaser'] = buffer_qlaser
                sim_data = pd.concat([sim_data,ses_data])
    return c_sim, l_sim, r_sim, sim_data
def simulate_reinforce_winloss_stay(standata_recovery,saved_params=None, fit=None, csv=True):
    NS =standata_recovery['NS']
    NSESS=standata_recovery['NSESS']
    NT =standata_recovery['NT']
    NT_all =standata_recovery['NT_all']
    r_sim = np.zeros([NS,NSESS,NT])
    c_sim = np.zeros([NS,NSESS,NT])
    l_sim = np.zeros([NS,NSESS,NT])
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
                    dmlaser_ses = float(saved_params.loc[saved_params['name']=='dmlaser_ses['+str(ms_i+1)+']', 'Mean'])
                    beta_ses = float(saved_params.loc[saved_params['name']=='beta_ses['+str(ms_i+1)+']', 'Mean'])
                    side_ses = float(saved_params.loc[saved_params['name']=='sides['+str(ms_i+1)+']', 'Mean'])
                    laser_ses = float(saved_params.loc[saved_params['name']=='laser_ses['+str(ms_i+1)+']', 'Mean'])
                    epses = float(saved_params.loc[saved_params['name']=='ep_ses['+str(ms_i+1)+']', 'Mean'])
                    alpha = float(saved_params.loc[saved_params['name']=='alpha_ses['+str(ms_i+1)+']', 'Mean'])
                    alphalaser = float(saved_params.loc[saved_params['name']=='alphalaser_ses['+str(ms_i+1)+']', 'Mean'])
                    alphastay = float(saved_params.loc[saved_params['name']=='alphastay_ses['+str(ms_i+1)+']', 'Mean'])
                    alphaforgetting= float(saved_params.loc[saved_params['name']=='alphaforgetting_ses['+str(ms_i+1)+']', 'Mean'])
                    alphalaserforgetting= float(saved_params.loc[saved_params['name']=='alphalaserforgetting_ses['+str(ms_i+1)+']', 'Mean'])
                else:
                    print('Error: Only csv version available at the moment!')

            tph = TrialParamHandler()
            q = 0
            qlaser = 0
            qstay = np.zeros(2)
            betalaser = laser_ses
            predicted_choices=[]
            rewards = []
            lasers = []
            outcome=[]
            opto_blocks=[]
            side_blocks=[]
            trial_within_block=[]
            right_reward=[]
            left_reward=[]
            buffer_qlaser=[]
            buffer_qwater=[]
            for t in np.arange(NT_all[ms_i]):
                t = int(t)    
                p_choice = (1-epses) * inv_logit(side_ses + (q + qlaser)) + epses*(inv_logit(qstay[1]-qstay[0]))
                choice =  np.random.choice(2,p= [1-p_choice,p_choice])
                working_choice = 2 * choice -1
                tph.response_side_buffer.append(choice)  # Add choice to response buffer
                p_left= tph.stim_probability_left
                l = tph.opto_block
                if choice==1:
                    f = tph.right_reward
                if choice==0:
                    f = tph.left_reward
                if (f*(l==0)) == 1:
                    q = (1 - alpha) * q + working_choice * beta_ses
                if (f*(l==0)) == 0:
                    q = (1 - alphaforgetting) * q + working_choice * dmwater_ses
                if (f*l) == 1:
                    qlaser = (1 - alphalaser) * qlaser + working_choice * laser_ses
                if (f*l) == 0:
                    qlaser = (1 - alphalaserforgetting) * qlaser + working_choice * dmlaser_ses
                qstay = qstay * (1 - alphastay)
                qstay[choice] = qstay[choice] + alphastay

                # Store variables
                rewards.append((1*(f*(l==0))))
                lasers.append(f*l)
                outcome.append(f)
                opto_blocks.append(l) 
                side_blocks.append(p_left)
                predicted_choices.append(choice)
                trial_within_block.append(tph.block_trial_num)
                right_reward.append(tph.right_reward)
                left_reward.append(tph.left_reward)
                buffer_qlaser.append(qlaser)
                buffer_qwater.append(q)
                # Init next trial
                tph =  tph.next_trial()

            if NT_all[ms_i]!=0:
                c_sim[sub_idx[ms_i], sess_idx[ms_i], :int(NT_all[ms_i])]= predicted_choices
                l_sim[sub_idx[ms_i], sess_idx[ms_i], :int(NT_all[ms_i])] = lasers
                r_sim[sub_idx[ms_i], sess_idx[ms_i], :int(NT_all[ms_i])] = rewards
                ses_data = pd.DataFrame()
                ses_data['choices'] = predicted_choices
                ses_data['water'] = rewards
                ses_data['laser'] = lasers
                ses_data['laser_block'] = opto_blocks
                ses_data['tb'] = trial_within_block
                ses_data['probabilityLeft'] = side_blocks
                ses_data['outcome'] = outcome
                ses_data['mouse'] = sub_idx[ms_i]
                ses_data['ses'] = sess_idx[ms_i]
                ses_data['right_reward'] = right_reward
                ses_data['left_reward'] = left_reward
                ses_data['qwater'] = buffer_qwater
                ses_data['qlaser'] = buffer_qlaser
                sim_data = pd.concat([sim_data,ses_data])
    return c_sim, l_sim, r_sim, sim_data
def simulate_reinforce_laserdecay_winloss_stay(standata_recovery,saved_params=None, fit=None, csv=True):
    NS =standata_recovery['NS']
    NSESS=standata_recovery['NSESS']
    NT =standata_recovery['NT']
    NT_all =standata_recovery['NT_all']
    r_sim = np.zeros([NS,NSESS,NT])
    c_sim = np.zeros([NS,NSESS,NT])
    l_sim = np.zeros([NS,NSESS,NT])
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
                    beta_ses = float(saved_params.loc[saved_params['name']=='beta_ses['+str(ms_i+1)+']', 'Mean'])
                    side_ses = float(saved_params.loc[saved_params['name']=='sides['+str(ms_i+1)+']', 'Mean'])
                    laser_ses = float(saved_params.loc[saved_params['name']=='laser_ses['+str(ms_i+1)+']', 'Mean'])
                    epses = float(saved_params.loc[saved_params['name']=='ep_ses['+str(ms_i+1)+']', 'Mean'])
                    alpha = float(saved_params.loc[saved_params['name']=='alpha_ses['+str(ms_i+1)+']', 'Mean'])
                    alphalaser = float(saved_params.loc[saved_params['name']=='alphalaser_ses['+str(ms_i+1)+']', 'Mean'])
                    alphastay = float(saved_params.loc[saved_params['name']=='alphastay_ses['+str(ms_i+1)+']', 'Mean'])
                    alphaforgetting= float(saved_params.loc[saved_params['name']=='alphaforgetting_ses['+str(ms_i+1)+']', 'Mean'])
                    alphalaserforgetting= float(saved_params.loc[saved_params['name']=='alphalaserforgetting_ses['+str(ms_i+1)+']', 'Mean'])
                    betalaserbase= float(saved_params.loc[saved_params['name']=='betalaserbase_ses['+str(ms_i+1)+']', 'Mean'])
                    dmwater_ses = float(saved_params.loc[saved_params['name']=='dmwater_ses['+str(ms_i+1)+']', 'Mean'])
                    dmlaser_ses = float(saved_params.loc[saved_params['name']=='dmlaser_ses['+str(ms_i+1)+']', 'Mean'])
                    betalaserlossbase= float(saved_params.loc[saved_params['name']=='betalaserlossbase_ses['+str(ms_i+1)+']', 'Mean'])
                    laserdecayloss= float(saved_params.loc[saved_params['name']=='laserdecayloss_ses['+str(ms_i+1)+']', 'Mean'])
                    laserdecay = float(saved_params.loc[saved_params['name']=='laserdecay_ses['+str(ms_i+1)+']', 'Mean'])

                else:
                    print('Error: Only csv version available at the moment!')

            tph = TrialParamHandler()
            q = 0
            qlaser = 0
            qstay = np.zeros(2)
            betalaser = laser_ses
            predicted_choices=[]
            rewards = []
            lasers = []
            outcome=[]
            opto_blocks=[]
            side_blocks=[]
            trial_within_block=[]
            right_reward=[]
            left_reward=[]
            betalaser_buffer=[]
            buffer_qlaser=[]
            buffer_qwater=[]
            betalaser = laser_ses
            betalaserloss = dmlaser_ses
            for t in np.arange(NT_all[ms_i]):
                t = int(t)    
                p_choice = (1-epses) * inv_logit(side_ses + (q + qlaser)) + epses*(inv_logit(qstay[1]-qstay[0]))
                choice =  np.random.choice(2,p= [1-p_choice,p_choice])
                working_choice = 2 * choice -1
                tph.response_side_buffer.append(choice)  # Add choice to response buffer
                p_left= tph.stim_probability_left
                l = tph.opto_block
                if choice==1:
                    f = tph.right_reward
                if choice==0:
                    f = tph.left_reward
                if (f*(l==0)) == 1:
                    q = (1 - alpha) * q + working_choice * beta_ses
                if (f*(l==0)) == 0:
                    q = (1 - alphaforgetting) * q + working_choice * dmwater_ses
                if (f*l) == 1:
                    betalaser = (1-laserdecay)*betalaser
                    qlaser = (1 - alphalaser) * qlaser + working_choice * (betalaser+betalaserbase)
                if (f*l) == 0:
                    betalaserloss = (1-laserdecayloss)*betalaserloss
                    qlaser = (1 - alphalaserforgetting) * qlaser + working_choice * (betalaserloss+betalaserlossbase)               
                qstay = qstay * (1 - alphastay)
                qstay[choice] = qstay[choice] + alphastay

                # Store variables
                betalaser_buffer.append(betalaser)
                rewards.append((1*(f*(l==0))))
                lasers.append(f*l)
                outcome.append(f)
                opto_blocks.append(l) 
                side_blocks.append(p_left)
                predicted_choices.append(choice)
                trial_within_block.append(tph.block_trial_num)
                right_reward.append(tph.right_reward)
                left_reward.append(tph.left_reward)
                buffer_qlaser.append(qlaser)
                buffer_qwater.append(q)
                # Init next trial
                tph =  tph.next_trial()

            if NT_all[ms_i]!=0:
                c_sim[sub_idx[ms_i], sess_idx[ms_i], :int(NT_all[ms_i])]= predicted_choices
                l_sim[sub_idx[ms_i], sess_idx[ms_i], :int(NT_all[ms_i])] = lasers
                r_sim[sub_idx[ms_i], sess_idx[ms_i], :int(NT_all[ms_i])] = rewards
                ses_data = pd.DataFrame()
                ses_data['choices'] = predicted_choices
                ses_data['b_init_laser'] = laser_ses
                ses_data['beta_laser'] = betalaser_buffer
                ses_data['water'] = rewards
                ses_data['laser'] = lasers
                ses_data['laser_block'] = opto_blocks
                ses_data['tb'] = trial_within_block
                ses_data['probabilityLeft'] = side_blocks
                ses_data['outcome'] = outcome
                ses_data['mouse'] = sub_idx[ms_i]
                ses_data['ses'] = sess_idx[ms_i]
                ses_data['right_reward'] = right_reward
                ses_data['left_reward'] = left_reward
                ses_data['qwater'] = buffer_qwater
                ses_data['qlaser'] = buffer_qlaser
                sim_data = pd.concat([sim_data,ses_data])
                
    return c_sim, l_sim, r_sim, sim_data
def simulate_q_learning_model_laserdecay_nolaser(standata_recovery,saved_params=None, fit=None, csv=True):
    NS =standata_recovery['NS']
    NSESS=standata_recovery['NSESS']
    NT =standata_recovery['NT']
    NT_all =standata_recovery['NT_all']
    r_sim = np.zeros([NS,NSESS,NT])
    c_sim = np.zeros([NS,NSESS,NT])
    l_sim = np.zeros([NS,NSESS,NT])
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
                    beta_mouse = float(saved_params.loc[saved_params['name']=='beta_ses['+str(ms_i+1)+']', 'Mean'])
                    stay_mouse = float(saved_params.loc[saved_params['name']=='stay_ses['+str(ms_i+1)+']', 'Mean'])
                    side_mouse = float(saved_params.loc[saved_params['name']=='sides['+str(ms_i+1)+']', 'Mean'])
                    laser_mouse = float(saved_params.loc[saved_params['name']=='laser_ses['+str(ms_i+1)+']', 'Mean'])
                    alpha = float(saved_params.loc[saved_params['name']=='alpha_ses['+str(ms_i+1)+']', 'Mean'])
                    alphalaser = float(saved_params.loc[saved_params['name']=='alphalaser_ses['+str(ms_i+1)+']', 'Mean'])
                    alphastay = float(saved_params.loc[saved_params['name']=='alphastay_ses['+str(ms_i+1)+']', 'Mean'])
                    betalaserbase= float(saved_params.loc[saved_params['name']=='betalaserbase_ses['+str(ms_i+1)+']', 'Mean'])
                    laserdecay= float(saved_params.loc[saved_params['name']=='laserdecay_ses['+str(ms_i+1)+']', 'Mean'])
                else:
                    print('Error: Only csv version available at the moment!')

            tph = TrialParamHandler()
            q = np.zeros(2)
            qlaser = np.zeros(2)
            qstay = np.zeros(2)
            predicted_choices=[]
            rewards = []
            lasers = []
            outcome=[]
            opto_blocks=[]
            side_blocks=[]
            trial_within_block=[]
            right_reward=[]
            left_reward=[]
            betalaser_buffer=[]
            buffer_qlaser=[]
            buffer_qwater=[]
            betalaser = laser_mouse

            for t in np.arange(NT_all[ms_i]):
                t = int(t)
                p_choice = inv_logit(side_mouse
                    + beta_mouse  * (q[1] - q[0])
                    + stay_mouse * (qstay[1] - qstay[0]))
                choice =  np.random.choice(2,p= [1-p_choice,p_choice])
                tph.response_side_buffer.append(choice)  # Add choice to response buffer
                p_left= tph.stim_probability_left
                l = tph.opto_block
                if choice==1:
                    f = tph.right_reward
                if choice==0:
                    f = tph.left_reward
                q[choice] = (1-alpha) * q[choice] + alpha * (1*(f*(l==0)))
                qstay = qstay * (1 - alphastay)
                qstay[choice] = qstay[choice] + alphastay
                if (f*l) == 1:
                    betalaser = (1-laserdecay)*betalaser
                # Store variables
                betalaser_buffer.append(betalaser)
                rewards.append((1*(f*(l==0))))
                lasers.append(f*l)
                outcome.append(f)
                opto_blocks.append(l) 
                side_blocks.append(p_left)
                predicted_choices.append(choice)
                trial_within_block.append(tph.block_trial_num)
                right_reward.append(tph.right_reward)
                left_reward.append(tph.left_reward)
                buffer_qlaser.append(qlaser[choice])
                buffer_qwater.append(q[choice])
                # Init next trial
                tph =  tph.next_trial()

            if NT_all[ms_i]!=0:
                c_sim[sub_idx[ms_i], sess_idx[ms_i], :int(NT_all[ms_i])]= predicted_choices
                l_sim[sub_idx[ms_i], sess_idx[ms_i], :int(NT_all[ms_i])] = lasers
                r_sim[sub_idx[ms_i], sess_idx[ms_i], :int(NT_all[ms_i])] = rewards
                ses_data = pd.DataFrame()
                ses_data['choices'] = predicted_choices
                ses_data['b_init_laser'] = laser_mouse
                ses_data['beta_laser'] = betalaser_buffer
                ses_data['water'] = rewards
                ses_data['laser'] = lasers
                ses_data['laser_block'] = opto_blocks
                ses_data['tb'] = trial_within_block
                ses_data['probabilityLeft'] = side_blocks
                ses_data['outcome'] = outcome
                ses_data['mouse'] = sub_idx[ms_i]
                ses_data['ses'] = sess_idx[ms_i]
                ses_data['right_reward'] = right_reward
                ses_data['left_reward'] = left_reward
                ses_data['qwater'] = buffer_qwater
                ses_data['qlaser'] = buffer_qlaser
                sim_data = pd.concat([sim_data,ses_data])
    return c_sim, l_sim, r_sim, sim_data

def simulate_q_learning_model_alphalaserdecay_same_trials(standata_recovery,saved_params=None, fit=None, csv=True):
    b =  standata_recovery['b']
    p =  standata_recovery['p']
    tb = standata_recovery['tb']
    NS =standata_recovery['NS']
    NSESS=standata_recovery['NSESS']
    NT =standata_recovery['NT']
    NT_all =standata_recovery['NT_all']
    r_sim = np.zeros([NS,NSESS,NT])
    c_sim = np.zeros([NS,NSESS,NT])
    l_sim = np.zeros([NS,NSESS,NT])
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
                    beta_mouse = float(saved_params.loc[saved_params['name']=='beta_ses['+str(ms_i+1)+']', 'Mean'])
                    stay_mouse = float(saved_params.loc[saved_params['name']=='stay_ses['+str(ms_i+1)+']', 'Mean'])
                    side_mouse = float(saved_params.loc[saved_params['name']=='sides['+str(ms_i+1)+']', 'Mean'])
                    laser_mouse = float(saved_params.loc[saved_params['name']=='laser_ses['+str(ms_i+1)+']', 'Mean'])
                    alpha = float(saved_params.loc[saved_params['name']=='alpha_ses['+str(ms_i+1)+']', 'Mean'])
                    alphalaser_init = float(saved_params.loc[saved_params['name']=='alphalaser_ses['+str(ms_i+1)+']', 'Mean'])
                    alphastay = float(saved_params.loc[saved_params['name']=='alphastay_ses['+str(ms_i+1)+']', 'Mean'])
                    alphalaserbase= float(saved_params.loc[saved_params['name']=='alphalaserbase_ses['+str(ms_i+1)+']', 'Mean'])
                    laserdecay= float(saved_params.loc[saved_params['name']=='laserdecay_ses['+str(ms_i+1)+']', 'Mean'])
                else:
                    print('Error: Only csv version available at the moment!')

            q = np.zeros(2)
            qlaser = np.zeros(2)
            qstay = np.zeros(2)
            predicted_choices=[]
            rewards = []
            lasers = []
            outcome=[]
            opto_blocks=[]
            side_blocks=[]
            trial_within_block=[]
            alphalaser_buffer=[]
            buffer_qlaser=[]
            buffer_qwater=[]
            alphalaser = alphalaser_init

            for t in np.arange(NT_all[ms_i]):
                t = int(t)
                p_choice = inv_logit(side_mouse
                    + beta_mouse  * (q[1] - q[0])
                    + laser_mouse * (qlaser[1] - qlaser[0])
                    + stay_mouse * (qstay[1] - qstay[0]))
                choice =  np.random.choice(2,p= [1-p_choice,p_choice])
                assert np.isin(p[sub_idx[ms_i], sess_idx[ms_i],t], [0.7,0.1])
                p_left= p[sub_idx[ms_i], sess_idx[ms_i],t]
                if p_left==0.7:
                    p_right=0.1
                else:
                    p_right=0.7                
                l = b[sub_idx[ms_i], sess_idx[ms_i],t]
                if choice==1:
                    f = np.random.choice([0,1], p=[1-p_right,p_right])
                if choice==0:
                    f = np.random.choice([0,1], p=[1-p_left,p_left])
                q[choice] = (1-alpha) * q[choice] + alpha * (1*(f*(l==0)))
                qlaser[choice] = (1-(alphalaser+alphalaserbase) ) * qlaser[choice] + (alphalaser+alphalaserbase)  * f * l
                qstay = qstay * (1 - alphastay)
                qstay[choice] = qstay[choice] + alphastay
                if (f*l) == 1:
                    alphalaser = (1-laserdecay)*alphalaser
                # Store variables
                alphalaser_buffer.append(alphalaser)
                rewards.append((1*(f*(l==0))))
                lasers.append(f*l)
                outcome.append(f)
                opto_blocks.append(l) 
                side_blocks.append(p_left)
                predicted_choices.append(choice)
                trial_within_block.append(tb[sub_idx[ms_i], sess_idx[ms_i],t])
                buffer_qlaser.append(qlaser[choice])
                buffer_qwater.append(q[choice])

            if NT_all[ms_i]!=0:
                c_sim[sub_idx[ms_i], sess_idx[ms_i], :int(NT_all[ms_i])]= predicted_choices
                l_sim[sub_idx[ms_i], sess_idx[ms_i], :int(NT_all[ms_i])] = lasers
                r_sim[sub_idx[ms_i], sess_idx[ms_i], :int(NT_all[ms_i])] = rewards
                ses_data = pd.DataFrame()
                ses_data['choices'] = predicted_choices
                ses_data['a_init_laser'] = alphalaser_init
                ses_data['beta_laser'] = alphalaser_buffer
                ses_data['water'] = rewards
                ses_data['laser'] = lasers
                ses_data['laser_block'] = opto_blocks
                ses_data['tb'] = trial_within_block
                ses_data['probabilityLeft'] = side_blocks
                ses_data['outcome'] = outcome
                ses_data['mouse'] = sub_idx[ms_i]
                ses_data['ses'] = sess_idx[ms_i]
                ses_data['qwater'] = buffer_qwater
                ses_data['qlaser'] = buffer_qlaser
                sim_data = pd.concat([sim_data,ses_data])
    return c_sim, l_sim, r_sim, sim_data
def simulate_q_learning_model_laserdecay_same_trials(standata_recovery,saved_params=None, fit=None, csv=True):
    b =  standata_recovery['b']
    p =  standata_recovery['p']
    tb = standata_recovery['tb']
    NS =standata_recovery['NS']
    NSESS=standata_recovery['NSESS']
    NT =standata_recovery['NT']
    NT_all =standata_recovery['NT_all']
    r_sim = np.zeros([NS,NSESS,NT])
    c_sim = np.zeros([NS,NSESS,NT])
    l_sim = np.zeros([NS,NSESS,NT])
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
                    beta_mouse = float(saved_params.loc[saved_params['name']=='beta_ses['+str(ms_i+1)+']', 'Mean'])
                    stay_mouse = float(saved_params.loc[saved_params['name']=='stay_ses['+str(ms_i+1)+']', 'Mean'])
                    side_mouse = float(saved_params.loc[saved_params['name']=='sides['+str(ms_i+1)+']', 'Mean'])
                    laser_mouse = float(saved_params.loc[saved_params['name']=='laser_ses['+str(ms_i+1)+']', 'Mean'])
                    alpha = float(saved_params.loc[saved_params['name']=='alpha_ses['+str(ms_i+1)+']', 'Mean'])
                    alphalaser = float(saved_params.loc[saved_params['name']=='alphalaser_ses['+str(ms_i+1)+']', 'Mean'])
                    alphastay = float(saved_params.loc[saved_params['name']=='alphastay_ses['+str(ms_i+1)+']', 'Mean'])
                    betalaserbase= float(saved_params.loc[saved_params['name']=='betalaserbase_ses['+str(ms_i+1)+']', 'Mean'])
                    laserdecay= float(saved_params.loc[saved_params['name']=='laserdecay_ses['+str(ms_i+1)+']', 'Mean'])
                else:
                    print('Error: Only csv version available at the moment!')

            q = np.zeros(2)
            qlaser = np.zeros(2)
            qstay = np.zeros(2)
            predicted_choices=[]
            rewards = []
            lasers = []
            outcome=[]
            opto_blocks=[]
            side_blocks=[]
            trial_within_block=[]
            betalaser_buffer=[]
            buffer_qlaser=[]
            buffer_qwater=[]
            betalaser = laser_mouse

            for t in np.arange(NT_all[ms_i]):
                t = int(t)
                p_choice = inv_logit(side_mouse
                    + beta_mouse  * (q[1] - q[0])
                    + (betalaser+betalaserbase)  * (qlaser[1] - qlaser[0])
                    + stay_mouse * (qstay[1] - qstay[0]))
                choice =  np.random.choice(2,p= [1-p_choice,p_choice])
                assert np.isin(p[sub_idx[ms_i], sess_idx[ms_i],t], [0.7,0.1])
                p_left= p[sub_idx[ms_i], sess_idx[ms_i],t]
                if p_left==0.7:
                    p_right=0.1
                else:
                    p_right=0.7                
                l = b[sub_idx[ms_i], sess_idx[ms_i],t]
                if choice==1:
                    f = np.random.choice([0,1], p=[1-p_right,p_right])
                if choice==0:
                    f = np.random.choice([0,1], p=[1-p_left,p_left])
                q[choice] = (1-alpha) * q[choice] + alpha * (1*(f*(l==0)))
                qlaser[choice] = (1-alphalaser) * qlaser[choice] + alphalaser * f * l
                qstay = qstay * (1 - alphastay)
                qstay[choice] = qstay[choice] + alphastay
                if (f*l) == 1:
                    betalaser = (1-laserdecay)*betalaser
                # Store variables
                betalaser_buffer.append(betalaser)
                rewards.append((1*(f*(l==0))))
                lasers.append(f*l)
                outcome.append(f)
                opto_blocks.append(l) 
                side_blocks.append(p_left)
                predicted_choices.append(choice)
                trial_within_block.append(tb[sub_idx[ms_i], sess_idx[ms_i],t])
                buffer_qlaser.append(qlaser[choice])
                buffer_qwater.append(q[choice])

            if NT_all[ms_i]!=0:
                c_sim[sub_idx[ms_i], sess_idx[ms_i], :int(NT_all[ms_i])]= predicted_choices
                l_sim[sub_idx[ms_i], sess_idx[ms_i], :int(NT_all[ms_i])] = lasers
                r_sim[sub_idx[ms_i], sess_idx[ms_i], :int(NT_all[ms_i])] = rewards
                ses_data = pd.DataFrame()
                ses_data['choices'] = predicted_choices
                ses_data['b_init_laser'] = laser_mouse
                ses_data['beta_laser'] = betalaser_buffer
                ses_data['water'] = rewards
                ses_data['laser'] = lasers
                ses_data['laser_block'] = opto_blocks
                ses_data['tb'] = trial_within_block
                ses_data['probabilityLeft'] = side_blocks
                ses_data['outcome'] = outcome
                ses_data['mouse'] = sub_idx[ms_i]
                ses_data['ses'] = sess_idx[ms_i]
                ses_data['qwater'] = buffer_qwater
                ses_data['qlaser'] = buffer_qlaser
                sim_data = pd.concat([sim_data,ses_data])
    return c_sim, l_sim, r_sim, sim_data

def simulate_q_learning_model_laserdecay(standata_recovery,saved_params=None, fit=None, csv=True):
    NS =standata_recovery['NS']
    NSESS=standata_recovery['NSESS']
    NT =standata_recovery['NT']
    NT_all =standata_recovery['NT_all']
    r_sim = np.zeros([NS,NSESS,NT])
    c_sim = np.zeros([NS,NSESS,NT])
    l_sim = np.zeros([NS,NSESS,NT])
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
                    beta_mouse = float(saved_params.loc[saved_params['name']=='beta_ses['+str(ms_i+1)+']', 'Mean'])
                    stay_mouse = float(saved_params.loc[saved_params['name']=='stay_ses['+str(ms_i+1)+']', 'Mean'])
                    side_mouse = float(saved_params.loc[saved_params['name']=='sides['+str(ms_i+1)+']', 'Mean'])
                    laser_mouse = float(saved_params.loc[saved_params['name']=='laser_ses['+str(ms_i+1)+']', 'Mean'])
                    alpha = float(saved_params.loc[saved_params['name']=='alpha_ses['+str(ms_i+1)+']', 'Mean'])
                    alphalaser = float(saved_params.loc[saved_params['name']=='alphalaser_ses['+str(ms_i+1)+']', 'Mean'])
                    alphastay = float(saved_params.loc[saved_params['name']=='alphastay_ses['+str(ms_i+1)+']', 'Mean'])
                    betalaserbase= float(saved_params.loc[saved_params['name']=='betalaserbase_ses['+str(ms_i+1)+']', 'Mean'])
                    laserdecay= float(saved_params.loc[saved_params['name']=='laserdecay_ses['+str(ms_i+1)+']', 'Mean'])
                else:
                    print('Error: Only csv version available at the moment!')

            tph = TrialParamHandler()
            q = np.zeros(2)
            qlaser = np.zeros(2)
            qstay = np.zeros(2)
            predicted_choices=[]
            rewards = []
            lasers = []
            outcome=[]
            opto_blocks=[]
            side_blocks=[]
            trial_within_block=[]
            right_reward=[]
            left_reward=[]
            betalaser_buffer=[]
            buffer_qlaser=[]
            buffer_qwater=[]
            betalaser = laser_mouse

            for t in np.arange(NT_all[ms_i]):
                t = int(t)
                p_choice = inv_logit(side_mouse
                    + beta_mouse  * (q[1] - q[0])
                    + (betalaser+betalaserbase)  * (qlaser[1] - qlaser[0])
                    + stay_mouse * (qstay[1] - qstay[0]))
                choice =  np.random.choice(2,p= [1-p_choice,p_choice])
                tph.response_side_buffer.append(choice)  # Add choice to response buffer
                p_left= tph.stim_probability_left
                l = tph.opto_block
                if choice==1:
                    f = tph.right_reward
                if choice==0:
                    f = tph.left_reward
                q[choice] = (1-alpha) * q[choice] + alpha * (1*(f*(l==0)))
                qlaser[choice] = (1-alphalaser) * qlaser[choice] + alphalaser * f * l
                qstay = qstay * (1 - alphastay)
                qstay[choice] = qstay[choice] + alphastay
                if (f*l) == 1:
                    betalaser = (1-laserdecay)*betalaser
                # Store variables
                betalaser_buffer.append(betalaser)
                rewards.append((1*(f*(l==0))))
                lasers.append(f*l)
                outcome.append(f)
                opto_blocks.append(l) 
                side_blocks.append(p_left)
                predicted_choices.append(choice)
                trial_within_block.append(tph.block_trial_num)
                right_reward.append(tph.right_reward)
                left_reward.append(tph.left_reward)
                buffer_qlaser.append(qlaser[choice])
                buffer_qwater.append(q[choice])
                # Init next trial
                tph =  tph.next_trial()

            if NT_all[ms_i]!=0:
                c_sim[sub_idx[ms_i], sess_idx[ms_i], :int(NT_all[ms_i])]= predicted_choices
                l_sim[sub_idx[ms_i], sess_idx[ms_i], :int(NT_all[ms_i])] = lasers
                r_sim[sub_idx[ms_i], sess_idx[ms_i], :int(NT_all[ms_i])] = rewards
                ses_data = pd.DataFrame()
                ses_data['choices'] = predicted_choices
                ses_data['b_init_laser'] = laser_mouse
                ses_data['beta_laser'] = betalaser_buffer
                ses_data['water'] = rewards
                ses_data['laser'] = lasers
                ses_data['laser_block'] = opto_blocks
                ses_data['tb'] = trial_within_block
                ses_data['probabilityLeft'] = side_blocks
                ses_data['outcome'] = outcome
                ses_data['mouse'] = sub_idx[ms_i]
                ses_data['ses'] = sess_idx[ms_i]
                ses_data['right_reward'] = right_reward
                ses_data['left_reward'] = left_reward
                ses_data['qwater'] = buffer_qwater
                ses_data['qlaser'] = buffer_qlaser
                sim_data = pd.concat([sim_data,ses_data])
    return c_sim, l_sim, r_sim, sim_data

def q_learning_model_w_forgetting(standata,saved_params=None, fit=None, csv=True):
    if saved_params is not None:
        r =  standata['r']
        c =  standata['c']
        l =  standata['l']
        sub_idx =  standata['sub_idx']-1
        sess_idx = standata['sess_idx']-1
        NSxNSESS = standata['NSxNSESS']
        NT_all = standata['NT_all']
    else:
        r =  fit.data['r']
        c =  fit.data['c']
        l =  fit.data['l']
        sub_idx =  fit.data['sub_idx']-1
        sess_idx = fit.data['sess_idx']-1
    data = pd.DataFrame()

    for ms_i in np.arange(NSxNSESS):
        if NT_all[ms_i]>0:
            if saved_params is not None:
                if ms_i==0:
                    print('Using saved parameters')
                if csv==True:
                    beta_mouse = float(saved_params.loc[saved_params['name']=='beta_ses['+str(ms_i+1)+']', 'Mean'])
                    stay_mouse = float(saved_params.loc[saved_params['name']=='stay_ses['+str(ms_i+1)+']', 'Mean'])
                    side_mouse = float(saved_params.loc[saved_params['name']=='sides['+str(ms_i+1)+']', 'Mean'])
                    laser_mouse = float(saved_params.loc[saved_params['name']=='laser_ses['+str(ms_i+1)+']', 'Mean'])
                    alpha = float(saved_params.loc[saved_params['name']=='alpha_ses['+str(ms_i+1)+']', 'Mean']/np.sqrt(2))
                    alphalaser = float(saved_params.loc[saved_params['name']=='alphalaser_ses['+str(ms_i+1)+']', 'Mean']/np.sqrt(2))
                    alphastay = float(saved_params.loc[saved_params['name']=='alphastay_ses['+str(ms_i+1)+']', 'Mean']/np.sqrt(2))
                    alphaforgetting= float(saved_params.loc[saved_params['name']=='alphaforgetting_ses['+str(ms_i+1)+']', 'Mean']/np.sqrt(2))
                    alphalaserforgetting= float(saved_params.loc[saved_params['name']=='alphalaserforgetting_ses['+str(ms_i+1)+']', 'Mean']/np.sqrt(2))
                else:
                    print('Error only CSV version available')

            q = np.zeros(2)
            qstay = np.zeros(2)
            qlaser = np.zeros(2)
            predicted_choices=[]
            QL=[]
            QR=[]
            QLstay=[]
            QRstay=[]
            QLlaser=[]
            QRlaser=[]
            choices=[]
            QLgeneral=[]
            QRgeneral=[]
            for t in np.arange(NT_all[ms_i]):
                t = int(t)
                predicted_choice = inv_logit(side_mouse
                    + beta_mouse  * (q[1] - q[0])
                    + stay_mouse * (qstay[1] - qstay[0])
                    + laser_mouse * (qlaser[1] - qlaser[0]))

                choice = c[sub_idx[ms_i], sess_idx[ms_i], t]
                q[choice] = (1-alpha) * q[choice] + alpha * r[sub_idx[ms_i], sess_idx[ms_i],t]
                q[choice^1] = q[choice^1] - (q[choice^1] * alphaforgetting)
                qlaser[choice] = (1-alphalaser) * qlaser[choice] + alphalaser * l[sub_idx[ms_i], sess_idx[ms_i],t]
                qlaser[choice^1] = qlaser[choice^1] - (alphalaserforgetting * qlaser[choice^1])
                qstay = qstay * (1 - alphastay)
                qstay[choice] = qstay[choice] + alphastay
                # Store variables
                predicted_choices.append(predicted_choice)
                choices.append(choice)
                QL.append(q[0])
                QR.append(q[1])
                QLstay.append(qstay[0])
                QRstay.append(qstay[1])
                QLlaser.append(qlaser[0])
                QRlaser.append(qlaser[1])
                QLgeneral.append((beta_mouse*q[0])+(laser_mouse*qlaser[0])+(stay_mouse*qstay[0]))
                QRgeneral.append((beta_mouse*q[1])+(laser_mouse*qlaser[1])+(stay_mouse*qstay[1]))
            acc = []
            for i in np.arange(len(choices)):
                acc.append(1*(choices[i]==(1*(predicted_choices[i]>0.5))))
            print(np.mean(acc))

            if NT_all[ms_i]!=0:
                ses_data = pd.DataFrame()
                ses_data['predicted_choice'] = predicted_choices
                ses_data['QL'] = QLgeneral
                ses_data['QR'] = QRgeneral
                ses_data['QLreward'] = np.array(QL) * beta_mouse # Note this QL is the reward  - TODO change name
                ses_data['QRreward'] =  np.array(QR) * beta_mouse
                ses_data['QLstay'] =  np.array(QLstay) * stay_mouse
                ses_data['QRstay'] =  np.array(QRstay) * stay_mouse
                ses_data['QLlaser'] =  np.array(QLlaser) * laser_mouse
                ses_data['QRlaser'] =  np.array(QRlaser) * laser_mouse
                ses_data['deltaQ'] = ses_data['QR'] - ses_data['QL']
                ses_data['choices'] = choices
                ses_data['mouse'] = sub_idx[ms_i]
                ses_data['ses'] = sess_idx[ms_i]
                ses_data['acc'] = np.mean(acc)
                data = pd.concat([data,ses_data])
    return make_deltas(data)


def simulate_q_learning_w_forgetting(standata_recovery,saved_params=None, fit=None, csv=True):
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
                    beta_mouse = float(saved_params.loc[saved_params['name']=='beta_ses['+str(ms_i+1)+']', 'Mean'])
                    stay_mouse = float(saved_params.loc[saved_params['name']=='stay_ses['+str(ms_i+1)+']', 'Mean'])
                    side_mouse = float(saved_params.loc[saved_params['name']=='sides['+str(ms_i+1)+']', 'Mean'])
                    alpha = float((saved_params.loc[saved_params['name']=='alpha_ses['+str(ms_i+1)+']', 'Mean']))
                    alphastay = float((saved_params.loc[saved_params['name']=='alphastay_ses['+str(ms_i+1)+']', 'Mean']))
                    alphaforgetting= float((saved_params.loc[saved_params['name']=='alphaforgetting_ses['+str(ms_i+1)+']', 'Mean']))
                else:
                    print('Error only CSV version available')

            tph = TrialParamHandler()
            qrewstay = np.zeros(2)
            q = np.zeros(2)
            qstay = np.zeros(2)
            predicted_choices=[]
            outcome=[]
            side_blocks=[]
            trial_within_block=[]
            right_reward=[]
            left_reward=[]
            QR = [0]
            QL = [0]

            for t in np.arange(NT_all[ms_i]): #np.arange(NT_all[ms_i])
                t = int(t)
                p_choice = inv_logit(side_mouse
                    + beta_mouse  * (q[1] - q[0])
                    + stay_mouse * (qstay[1] - qstay[0]))
                choice =  np.random.choice(2,p= [1-p_choice,p_choice])
                tph.response_side_buffer.append(choice)  # Add choice to response buffer
                p_left= tph.stim_probability_left
                if choice==1:
                    f = tph.right_reward
                if choice==0:
                    f = tph.left_reward
                q[choice] = (1-alpha) * q[choice] + alpha * (1*f)
                q[choice^1] = q[choice^1] - (q[choice^1] * alphaforgetting)
                qstay = qstay * (1 - alphastay)
                qstay[choice] = qstay[choice] + alphastay
                # Store variables
                outcome.append(f)
                side_blocks.append(p_left)
                predicted_choices.append(choice)
                trial_within_block.append(tph.block_trial_num)
                right_reward.append(tph.right_reward)
                left_reward.append(tph.left_reward)
                QR.append(beta_mouse  * q[1] + stay_mouse * qstay[1])
                QL.append(beta_mouse  * q[0] + stay_mouse * qstay[0])
                # Init next trial
                tph =  tph.next_trial()

            if NT_all[ms_i]!=0:
                c_sim[sub_idx[ms_i], sess_idx[ms_i], :len(predicted_choices)]= predicted_choices
                o_sim[sub_idx[ms_i], sess_idx[ms_i], :len(outcome)] = outcome
                ses_data = pd.DataFrame()
                ses_data['choices'] = predicted_choices
                ses_data['QR'] = QR[:-1]
                ses_data['QL'] = QL[:-1]
                ses_data['tb'] = trial_within_block
                ses_data['probabilityLeft'] = side_blocks
                ses_data['outcome'] = outcome
                ses_data['mouse'] = sub_idx[ms_i]
                ses_data['ses'] = sess_idx[ms_i]
                ses_data['right_reward'] = right_reward
                ses_data['left_reward'] = left_reward
                sim_data = pd.concat([sim_data,ses_data])
    return c_sim, o_sim, sim_data

 
def reinforce_model_mixed_perseveration(standata,saved_params=None, fit=None, csv=True):
    if saved_params is not None:
        r =  standata['r']
        c =  standata['c']
        l =  standata['l']
        sub_idx =  standata['sub_idx']-1
        sess_idx = standata['sess_idx']-1
        NSxNSESS = standata['NSxNSESS']
        NT_all = standata['NT_all']
    else:
        r =  fit.data['r']
        c =  fit.data['c']
        l =  fit.data['l']
        sub_idx =  fit.data['sub_idx']-1
        sess_idx = fit.data['sess_idx']-1
    data = pd.DataFrame()

    for ms_i in np.arange(NSxNSESS):
        if NT_all[ms_i]>0:
            if saved_params is not None:
            #    if ms_i==0:
            #        print('Using saved parameters')
                if csv==True:
                    dmwater_ses = float(saved_params.loc[saved_params['name']=='dmwater_ses['+str(ms_i+1)+']', 'Mean'])
                    dmlaser_ses = float(saved_params.loc[saved_params['name']=='dmlaser_ses['+str(ms_i+1)+']', 'Mean'])
                    beta_ses = float(saved_params.loc[saved_params['name']=='beta_ses['+str(ms_i+1)+']', 'Mean'])
                    side_ses = float(saved_params.loc[saved_params['name']=='sides['+str(ms_i+1)+']', 'Mean'])
                    laser_ses = float(saved_params.loc[saved_params['name']=='laser_ses['+str(ms_i+1)+']', 'Mean'])
                    epses = float(saved_params.loc[saved_params['name']=='ep_ses['+str(ms_i+1)+']', 'Mean'])
                    alpha = float(saved_params.loc[saved_params['name']=='alpha_ses['+str(ms_i+1)+']', 'Mean'])
                    alphalaser = float(saved_params.loc[saved_params['name']=='alphalaser_ses['+str(ms_i+1)+']', 'Mean'])
                    alphastay = float(saved_params.loc[saved_params['name']=='alphastay_ses['+str(ms_i+1)+']', 'Mean'])
                    alphaforgetting= float(saved_params.loc[saved_params['name']=='alphaforgetting_ses['+str(ms_i+1)+']', 'Mean'])
                    alphalaserforgetting= float(saved_params.loc[saved_params['name']=='alphalaserforgetting_ses['+str(ms_i+1)+']', 'Mean'])
                else:
                    Print('Non csv version not avaialble atm')
            q = 0
            qlaser = 0
            qstay = np.zeros(2)
            predicted_choices=[]
            Q=[]
            Qlaser=[]
            QLstay=[]
            QRstay=[]
            choices=[]
            for t in np.arange(NT_all[ms_i]):
                t = int(t)
                choices.append(c[sub_idx[ms_i], sess_idx[ms_i], t])
                predicted_choice = (1-epses) * inv_logit(side_ses + (q + qlaser)) + epses*(inv_logit(qstay[1]-qstay[0]))
                choice_10 = c[sub_idx[ms_i], sess_idx[ms_i], t]
                choice = 2 * c[sub_idx[ms_i], sess_idx[ms_i], t] -1
                if r[sub_idx[ms_i], sess_idx[ms_i], t] == 1:
                    q = (1 - alpha) * q + choice * beta_ses
                if r[sub_idx[ms_i], sess_idx[ms_i], t] == 0:
                    q = (1 - alphaforgetting) * q + choice * dmwater_ses
                if l[sub_idx[ms_i], sess_idx[ms_i], t] == 1:
                    qlaser = (1 - alphalaser) * qlaser + choice * laser_ses
                if l[sub_idx[ms_i], sess_idx[ms_i], t] == 0:
                    qlaser = (1 - alphalaserforgetting) * qlaser + choice * dmlaser_ses
                qstay = qstay * (1 - alphastay)
                qstay[choice_10] = qstay[choice_10] + alphastay
                # Store variables
                predicted_choices.append(predicted_choice)
                Q.append(q)
                Qlaser.append(qlaser)
                QLstay.append(qstay[0])
                QRstay.append(qstay[1])

            acc = []
            for i in np.arange(len(choices)):
                acc.append(1*(choices[i]==(1*(predicted_choices[i]>0.5))))
            #print(np.mean(acc))

            if NT_all[ms_i]!=0:
                ses_data = pd.DataFrame()
                ses_data['predicted_choice'] = predicted_choices
                ses_data['Qwater'] = Q
                ses_data['Qlaser'] = Qlaser
                ses_data['QLstay'] =  np.array(QLstay)
                ses_data['QRstay'] =  np.array(QRstay)
                ses_data['choices'] = choices
                ses_data['mouse'] = sub_idx[ms_i]
                ses_data['ses'] = sess_idx[ms_i]
                ses_data['acc'] = np.mean(acc)
                data = pd.concat([data,ses_data])
    return data
def reinforce_model_mixed_perseveration_noloss(standata,saved_params=None, fit=None, csv=True):
    if saved_params is not None:
        r =  standata['r']
        c =  standata['c']
        l =  standata['l']
        sub_idx =  standata['sub_idx']-1
        sess_idx = standata['sess_idx']-1
        NSxNSESS = standata['NSxNSESS']
        NT_all = standata['NT_all']
    else:
        r =  fit.data['r']
        c =  fit.data['c']
        l =  fit.data['l']
        sub_idx =  fit.data['sub_idx']-1
        sess_idx = fit.data['sess_idx']-1
    data = pd.DataFrame()

    for ms_i in np.arange(NSxNSESS):
        if NT_all[ms_i]>0:
            if saved_params is not None:
            #    if ms_i==0:
            #        print('Using saved parameters')
                if csv==True:
                    beta_ses = float(saved_params.loc[saved_params['name']=='beta_ses['+str(ms_i+1)+']', 'Mean'])
                    side_ses = float(saved_params.loc[saved_params['name']=='sides['+str(ms_i+1)+']', 'Mean'])
                    laser_ses = float(saved_params.loc[saved_params['name']=='laser_ses['+str(ms_i+1)+']', 'Mean'])
                    epses = float(saved_params.loc[saved_params['name']=='ep_ses['+str(ms_i+1)+']', 'Mean'])
                    alpha = float(saved_params.loc[saved_params['name']=='alpha_ses['+str(ms_i+1)+']', 'Mean'])
                    alphalaser = float(saved_params.loc[saved_params['name']=='alphalaser_ses['+str(ms_i+1)+']', 'Mean'])
                    alphastay = float(saved_params.loc[saved_params['name']=='alphastay_ses['+str(ms_i+1)+']', 'Mean'])
                    alphaforgetting= float(saved_params.loc[saved_params['name']=='alphaforgetting_ses['+str(ms_i+1)+']', 'Mean'])
                    alphalaserforgetting= float(saved_params.loc[saved_params['name']=='alphalaserforgetting_ses['+str(ms_i+1)+']', 'Mean'])
                else:
                    Print('Non csv version not avaialble atm')
            q = 0
            qlaser = 0
            qstay = np.zeros(2)
            predicted_choices=[]
            Q=[]
            Qlaser=[]
            QLstay=[]
            QRstay=[]
            choices=[]
            for t in np.arange(NT_all[ms_i]):
                t = int(t)
                choices.append(c[sub_idx[ms_i], sess_idx[ms_i], t])
                predicted_choice = (1-epses) * inv_logit(side_ses + (q + qlaser)) + epses*(inv_logit(qstay[1]-qstay[0]))
                choice_10 = c[sub_idx[ms_i], sess_idx[ms_i], t]
                choice = 2 * c[sub_idx[ms_i], sess_idx[ms_i], t] -1
                if r[sub_idx[ms_i], sess_idx[ms_i], t] == 1:
                    q = (1 - alpha) * q + choice * beta_ses
                if r[sub_idx[ms_i], sess_idx[ms_i], t] == 0:
                    q = (1 - alphaforgetting) * q
                if l[sub_idx[ms_i], sess_idx[ms_i], t] == 1:
                    qlaser = (1 - alphalaser) * qlaser + choice * laser_ses
                if l[sub_idx[ms_i], sess_idx[ms_i], t] == 0:
                    qlaser = (1 - alphalaserforgetting) * qlaser
                qstay = qstay * (1 - alphastay)
                qstay[choice_10] = qstay[choice_10] + alphastay
                # Store variables
                predicted_choices.append(predicted_choice)
                Q.append(q)
                Qlaser.append(qlaser)
                QLstay.append(qstay[0])
                QRstay.append(qstay[1])

            acc = []
            for i in np.arange(len(choices)):
                acc.append(1*(choices[i]==(1*(predicted_choices[i]>0.5))))
            #print(np.mean(acc))

            if NT_all[ms_i]!=0:
                ses_data = pd.DataFrame()
                ses_data['predicted_choice'] = predicted_choices
                ses_data['Qwater'] = Q
                ses_data['Qlaser'] = Qlaser
                ses_data['QLstay'] =  np.array(QLstay)
                ses_data['QRstay'] =  np.array(QRstay)
                ses_data['choices'] = choices
                ses_data['mouse'] = sub_idx[ms_i]
                ses_data['ses'] = sess_idx[ms_i]
                ses_data['acc'] = np.mean(acc)
                data = pd.concat([data,ses_data])
    return data
def reinforce_model_nolaser_mixed_perseveration(standata,saved_params=None, fit=None, csv=True):
    if saved_params is not None:
        r =  standata['r']
        c =  standata['c']
        l =  standata['l']
        sub_idx =  standata['sub_idx']-1
        sess_idx = standata['sess_idx']-1
        NSxNSESS = standata['NSxNSESS']
        NT_all = standata['NT_all']
    else:
        r =  fit.data['r']
        c =  fit.data['c']
        l =  fit.data['l']
        sub_idx =  fit.data['sub_idx']-1
        sess_idx = fit.data['sess_idx']-1
    o=r+l
    data = pd.DataFrame()

    for ms_i in np.arange(NSxNSESS):
        if NT_all[ms_i]>0:
            if saved_params is not None:
            #    if ms_i==0:
            #        print('Using saved parameters')
                if csv==True:
                    dmwater_ses = float(saved_params.loc[saved_params['name']=='dmwater_ses['+str(ms_i+1)+']', 'Mean'])
                    beta_ses = float(saved_params.loc[saved_params['name']=='beta_ses['+str(ms_i+1)+']', 'Mean'])
                    side_ses = float(saved_params.loc[saved_params['name']=='sides['+str(ms_i+1)+']', 'Mean'])
                    epses = float(saved_params.loc[saved_params['name']=='ep_ses['+str(ms_i+1)+']', 'Mean'])
                    alpha = float(saved_params.loc[saved_params['name']=='alpha_ses['+str(ms_i+1)+']', 'Mean'])
                    alphastay = float(saved_params.loc[saved_params['name']=='alphastay_ses['+str(ms_i+1)+']', 'Mean'])
                    alphaforgetting= float(saved_params.loc[saved_params['name']=='alphaforgetting_ses['+str(ms_i+1)+']', 'Mean'])
                else:
                    Print('Non csv version not avaialble atm')
            q = 0
            qstay = np.zeros(2)
            predicted_choices=[]
            Q=[]
            QLstay=[]
            QRstay=[]
            choices=[]
            for t in np.arange(NT_all[ms_i]):
                t = int(t)
                choices.append(c[sub_idx[ms_i], sess_idx[ms_i], t])
                predicted_choice = (1-epses) * inv_logit(side_ses + q) + epses*(inv_logit(qstay[1]-qstay[0]))
                choice_10 = c[sub_idx[ms_i], sess_idx[ms_i], t]
                choice = 2 * c[sub_idx[ms_i], sess_idx[ms_i], t] -1
                if o[sub_idx[ms_i], sess_idx[ms_i], t] == 1:
                    q = (1 - alpha) * q + choice * beta_ses
                if o[sub_idx[ms_i], sess_idx[ms_i], t] == 0:
                    q = (1 - alphaforgetting) * q + choice * dmwater_ses
                qstay = qstay * (1 - alphastay)
                qstay[choice_10] = qstay[choice_10] + alphastay
                # Store variables
                predicted_choices.append(predicted_choice)
                Q.append(q)
                QLstay.append(qstay[0])
                QRstay.append(qstay[1])

            acc = []
            for i in np.arange(len(choices)):
                acc.append(1*(choices[i]==(1*(predicted_choices[i]>0.5))))
            #print(np.mean(acc))

            if NT_all[ms_i]!=0:
                ses_data = pd.DataFrame()
                ses_data['predicted_choice'] = predicted_choices
                ses_data['Qwater'] = Q
                ses_data['QLstay'] =  np.array(QLstay)
                ses_data['QRstay'] =  np.array(QRstay)
                ses_data['choices'] = choices
                ses_data['mouse'] = sub_idx[ms_i]
                ses_data['ses'] = sess_idx[ms_i]
                ses_data['acc'] = np.mean(acc)
                data = pd.concat([data,ses_data])
    return data

def reinforce_model(standata,saved_params=None, fit=None, csv=True):
    if saved_params is not None:
        r =  standata['r']
        c =  standata['c']
        l =  standata['l']
        sub_idx =  standata['sub_idx']-1
        sess_idx = standata['sess_idx']-1
        NSxNSESS = standata['NSxNSESS']
        NT_all = standata['NT_all']
    else:
        r =  fit.data['r']
        c =  fit.data['c']
        l =  fit.data['l']
        sub_idx =  fit.data['sub_idx']-1
        sess_idx = fit.data['sess_idx']-1
    data = pd.DataFrame()

    for ms_i in np.arange(NSxNSESS):
        if NT_all[ms_i]>0:
            if saved_params is not None:
            #    if ms_i==0:
            #        print('Using saved parameters')
                if csv==True:
                    dmwater_ses = float(saved_params.loc[saved_params['name']=='dmwater_ses['+str(ms_i+1)+']', 'Mean'])
                    dmlaser_ses = float(saved_params.loc[saved_params['name']=='dmlaser_ses['+str(ms_i+1)+']', 'Mean'])
                    beta_ses = float(saved_params.loc[saved_params['name']=='beta_ses['+str(ms_i+1)+']', 'Mean'])
                    side_ses = float(saved_params.loc[saved_params['name']=='sides['+str(ms_i+1)+']', 'Mean'])
                    laser_ses = float(saved_params.loc[saved_params['name']=='laser_ses['+str(ms_i+1)+']', 'Mean'])
                    alpha = float(saved_params.loc[saved_params['name']=='alpha_ses['+str(ms_i+1)+']', 'Mean'])
                    alphalaser = float(saved_params.loc[saved_params['name']=='alphalaser_ses['+str(ms_i+1)+']', 'Mean'])
                    alphaforgetting= float(saved_params.loc[saved_params['name']=='alphaforgetting_ses['+str(ms_i+1)+']', 'Mean'])
                    alphalaserforgetting= float(saved_params.loc[saved_params['name']=='alphalaserforgetting_ses['+str(ms_i+1)+']', 'Mean'])
                else:
                    Print('Non csv version not avaialble atm')
            q = 0
            qlaser = 0
            predicted_choices=[]
            Q=[]
            Qlaser=[]
            choices=[]
            for t in np.arange(NT_all[ms_i]):
                t = int(t)
                choices.append(c[sub_idx[ms_i], sess_idx[ms_i], t])
                predicted_choice = inv_logit(side_ses + (q + qlaser))
                choice = 2 * c[sub_idx[ms_i], sess_idx[ms_i], t] -1
                if r[sub_idx[ms_i], sess_idx[ms_i], t] == 1:
                    q = (1 - alpha) * q + choice * beta_ses
                if r[sub_idx[ms_i], sess_idx[ms_i], t] == 0:
                    q = (1 - alphaforgetting) * q + choice * dmwater_ses
                if l[sub_idx[ms_i], sess_idx[ms_i], t] == 1:
                    qlaser = (1 - alphalaser) * qlaser + choice * laser_ses
                if l[sub_idx[ms_i], sess_idx[ms_i], t] == 0:
                    qlaser = (1 - alphalaserforgetting) * qlaser + choice * dmlaser_ses
                # Store variables
                predicted_choices.append(predicted_choice)
                Q.append(q)
                Qlaser.append(qlaser)

            acc = []
            for i in np.arange(len(choices)):
                acc.append(1*(choices[i]==(1*(predicted_choices[i]>0.5))))
            #print(np.mean(acc))

            if NT_all[ms_i]!=0:
                ses_data = pd.DataFrame()
                ses_data['predicted_choice'] = predicted_choices
                ses_data['Qwater'] = Q
                ses_data['Qlaser'] = Qlaser
                ses_data['deltaQ'] = ses_data['Qwater'] + ses_data['Qlaser']
                ses_data['choices'] = choices
                ses_data['mouse'] = sub_idx[ms_i]
                ses_data['ses'] = sess_idx[ms_i]
                ses_data['acc'] = np.mean(acc)
                data = pd.concat([data,ses_data])
    return data

def reinforce_model_reduced_new(standata,saved_params=None, fit=None, csv=True):
    if saved_params is not None:
        r =  standata['r']
        c =  standata['c']
        sub_idx =  standata['sub_idx']-1
        sess_idx = standata['sess_idx']-1
        NSxNSESS = standata['NSxNSESS']
        NT_all = standata['NT_all']
    else:
        r =  fit.data['r']
        c =  fit.data['c']
        sub_idx =  fit.data['sub_idx']-1
        sess_idx = fit.data['sess_idx']-1
    data = pd.DataFrame()

    for ms_i in np.arange(NSxNSESS):
        if NT_all[ms_i]>0:
            if saved_params is not None:
            #    if ms_i==0:
            #        print('Using saved parameters')
                if csv==True:
                    dmwater_ses = float(saved_params.loc[saved_params['name']=='dmwater_ses['+str(ms_i+1)+']', 'Mean'])
                    beta_ses = float(saved_params.loc[saved_params['name']=='beta_ses['+str(ms_i+1)+']', 'Mean'])
                    side_ses = float(saved_params.loc[saved_params['name']=='sides['+str(ms_i+1)+']', 'Mean'])
                    alpha = float(saved_params.loc[saved_params['name']=='alpha_ses['+str(ms_i+1)+']', 'Mean'])
                    alphaforgetting= float(saved_params.loc[saved_params['name']=='alphaforgetting_ses['+str(ms_i+1)+']', 'Mean'])
                else:
                    print('Non csv version not avaialble atm')
            q = 0
            predicted_choices=[]
            Q=[]
            choices=[]
            for t in np.arange(NT_all[ms_i]):
                t = int(t)
                choices.append(c[sub_idx[ms_i], sess_idx[ms_i], t])
                predicted_choice = inv_logit(side_ses + q)
                choice = 2 * c[sub_idx[ms_i], sess_idx[ms_i], t] -1
                o = r[sub_idx[ms_i], sess_idx[ms_i], t]
                if o == 1:
                    q = (1 - alpha) * q + choice * beta_ses * (1-inv_logit(choice*(side_ses + q)))
                if o == 0:
                    q = (1 - alphaforgetting) * q + choice * dmwater_ses * (1-inv_logit(choice*(side_ses + q)))
                # Store variables
                predicted_choices.append(predicted_choice)
                Q.append(q)

            acc = []
            for i in np.arange(len(choices)):
                acc.append(1*(choices[i]==(1*(predicted_choices[i]>0.5))))
            #print(np.mean(acc))

            if NT_all[ms_i]!=0:
                ses_data = pd.DataFrame()
                ses_data['predicted_choice'] = predicted_choices
                ses_data['Qwater'] = Q
                ses_data['deltaQ'] = ses_data['Qwater']
                ses_data['choices'] = choices
                ses_data['mouse'] = sub_idx[ms_i]
                ses_data['ses'] = sess_idx[ms_i]
                ses_data['id'] = ses_data['mouse']*100 + ses_data['ses']
                ses_data['acc'] = np.mean(acc)
                data = pd.concat([data,ses_data])
    return data

def actor_critic(standata,saved_params=None, fit=None, csv=True):
    if saved_params is not None:
        r =  standata['r']
        c =  standata['c']
        sub_idx =  standata['sub_idx']-1
        sess_idx = standata['sess_idx']-1
        NSxNSESS = standata['NSxNSESS']
        NT_all = standata['NT_all']
    else:
        r =  fit.data['r']
        c =  fit.data['c']
        sub_idx =  fit.data['sub_idx']-1
        sess_idx = fit.data['sess_idx']-1
    data = pd.DataFrame()

    for ms_i in np.arange(NSxNSESS):
        if NT_all[ms_i]>0:
            if saved_params is not None:
            #    if ms_i==0:
            #        print('Using saved parameters')
                if csv==True:
                    dmwater_ses = float(saved_params.loc[saved_params['name']=='dmwater_ses['+str(ms_i+1)+']', 'Mean'])
                    beta_ses = float(saved_params.loc[saved_params['name']=='beta_ses['+str(ms_i+1)+']', 'Mean'])
                    side_ses = float(saved_params.loc[saved_params['name']=='sides['+str(ms_i+1)+']', 'Mean'])
                    alpha = float(saved_params.loc[saved_params['name']=='alpha_ses['+str(ms_i+1)+']', 'Mean'])
                    alpha_rew = float(saved_params.loc[saved_params['name']=='alpharew_ses['+str(ms_i+1)+']', 'Mean'])
                    alphaforgetting= float(saved_params.loc[saved_params['name']=='alphaforgetting_ses['+str(ms_i+1)+']', 'Mean'])
                else:
                    print('Non csv version not avaialble atm')
            q = 0
            v = 0
            predicted_choices=[]
            Q=[]
            V=[]
            choices=[]
            for t in np.arange(NT_all[ms_i]):
                t = int(t)
                choices.append(c[sub_idx[ms_i], sess_idx[ms_i], t])
                predicted_choice = inv_logit(side_ses + q)
                choice = 2 * c[sub_idx[ms_i], sess_idx[ms_i], t] -1
                o = r[sub_idx[ms_i], sess_idx[ms_i], t]
                
                #Critic
                rpe  = o - v
                v = v + (alpha_rew * rpe)

                #Actor
                if o == 1:
                    q = (1 - alpha) * q + choice * beta_ses * (1-inv_logit(choice*(side_ses + q)))*rpe
                if o == 0:
                    q = (1 - alphaforgetting) * q + choice * dmwater_ses * (1-inv_logit(choice*(side_ses + q)))*rpe
                # Store variables
                predicted_choices.append(predicted_choice)
                Q.append(q)
                V.append(v)

            acc = []
            for i in np.arange(len(choices)):
                acc.append(1*(choices[i]==(1*(predicted_choices[i]>0.5))))
            #print(np.mean(acc))

            if NT_all[ms_i]!=0:
                ses_data = pd.DataFrame()
                ses_data['predicted_choice'] = predicted_choices
                ses_data['Qwater'] = Q
                ses_data['V'] = V
                ses_data['deltaQ'] = ses_data['Qwater']
                ses_data['choices'] = choices
                ses_data['mouse'] = sub_idx[ms_i]
                ses_data['ses'] = sess_idx[ms_i]
                ses_data['id'] = ses_data['mouse']*100 + ses_data['ses']
                ses_data['acc'] = np.mean(acc)
                data = pd.concat([data,ses_data])
    return data


def reinforce_model_reduced(standata,saved_params=None, fit=None, csv=True):
    if saved_params is not None:
        r =  standata['r']
        c =  standata['c']
        sub_idx =  standata['sub_idx']-1
        sess_idx = standata['sess_idx']-1
        NSxNSESS = standata['NSxNSESS']
        NT_all = standata['NT_all']
    else:
        r =  fit.data['r']
        c =  fit.data['c']
        sub_idx =  fit.data['sub_idx']-1
        sess_idx = fit.data['sess_idx']-1
    data = pd.DataFrame()

    for ms_i in np.arange(NSxNSESS):
        if NT_all[ms_i]>0:
            if saved_params is not None:
            #    if ms_i==0:
            #        print('Using saved parameters')
                if csv==True:
                    dmwater_ses = float(saved_params.loc[saved_params['name']=='dmwater_ses['+str(ms_i+1)+']', 'Mean'])
                    beta_ses = float(saved_params.loc[saved_params['name']=='beta_ses['+str(ms_i+1)+']', 'Mean'])
                    side_ses = float(saved_params.loc[saved_params['name']=='sides['+str(ms_i+1)+']', 'Mean'])
                    alpha = float(saved_params.loc[saved_params['name']=='alpha_ses['+str(ms_i+1)+']', 'Mean'])
                    alphaforgetting= float(saved_params.loc[saved_params['name']=='alphaforgetting_ses['+str(ms_i+1)+']', 'Mean'])
                else:
                    print('Non csv version not avaialble atm')
            q = 0
            predicted_choices=[]
            Q=[]
            choices=[]
            for t in np.arange(NT_all[ms_i]):
                t = int(t)
                choices.append(c[sub_idx[ms_i], sess_idx[ms_i], t])
                predicted_choice = inv_logit(side_ses + q)
                choice = 2 * c[sub_idx[ms_i], sess_idx[ms_i], t] -1
                o = r[sub_idx[ms_i], sess_idx[ms_i], t]
                if o == 1:
                    q = (1 - alpha) * q + choice * beta_ses
                if o == 0:
                    q = (1 - alphaforgetting) * q + choice * dmwater_ses
                # Store variables
                predicted_choices.append(predicted_choice)
                Q.append(q)

            acc = []
            for i in np.arange(len(choices)):
                acc.append(1*(choices[i]==(1*(predicted_choices[i]>0.5))))
            #print(np.mean(acc))

            if NT_all[ms_i]!=0:
                ses_data = pd.DataFrame()
                ses_data['predicted_choice'] = predicted_choices
                ses_data['Qwater'] = Q
                ses_data['deltaQ'] = ses_data['Qwater']
                ses_data['choices'] = choices
                ses_data['mouse'] = sub_idx[ms_i]
                ses_data['ses'] = sess_idx[ms_i]
                ses_data['id'] = ses_data['mouse']*100 + ses_data['ses']
                ses_data['acc'] = np.mean(acc)
                data = pd.concat([data,ses_data])
    return data

def reinforce_model_reduced_stay(standata,saved_params=None, fit=None, csv=True):
    if saved_params is not None:
        r =  standata['r']
        c =  standata['c']
        sub_idx =  standata['sub_idx']-1
        sess_idx = standata['sess_idx']-1
        NSxNSESS = standata['NSxNSESS']
        NT_all = standata['NT_all']
    else:
        r =  fit.data['r']
        c =  fit.data['c']
        sub_idx =  fit.data['sub_idx']-1
        sess_idx = fit.data['sess_idx']-1
    data = pd.DataFrame()

    for ms_i in np.arange(NSxNSESS):
        if NT_all[ms_i]>0:
            if saved_params is not None:
            #    if ms_i==0:
            #        print('Using saved parameters')
                if csv==True:
                    dmwater_ses = float(saved_params.loc[saved_params['name']=='dmwater_ses['+str(ms_i+1)+']', 'Mean'])
                    beta_ses = float(saved_params.loc[saved_params['name']=='beta_ses['+str(ms_i+1)+']', 'Mean'])
                    side_ses = float(saved_params.loc[saved_params['name']=='sides['+str(ms_i+1)+']', 'Mean'])
                    epses = float(saved_params.loc[saved_params['name']=='ep_ses['+str(ms_i+1)+']', 'Mean'])
                    alpha = float(saved_params.loc[saved_params['name']=='alpha_ses['+str(ms_i+1)+']', 'Mean'])
                    alphastay = float(saved_params.loc[saved_params['name']=='alphastay_ses['+str(ms_i+1)+']', 'Mean'])
                    alphaforgetting= float(saved_params.loc[saved_params['name']=='alphaforgetting_ses['+str(ms_i+1)+']', 'Mean'])
                else:
                    print('Non csv version not avaialble atm')
            q = 0
            qstay = np.zeros(2)
            predicted_choices=[]
            Q=[]
            QLstay=[]
            QRstay=[]
            choices=[]
            for t in np.arange(NT_all[ms_i]):
                t = int(t)
                choices.append(c[sub_idx[ms_i], sess_idx[ms_i], t])
                predicted_choice = (1-epses) * inv_logit(side_ses + q) + epses*(inv_logit(qstay[1]-qstay[0]))
                choice_10 = c[sub_idx[ms_i], sess_idx[ms_i], t]
                choice = 2 * c[sub_idx[ms_i], sess_idx[ms_i], t] -1
                if r[sub_idx[ms_i], sess_idx[ms_i], t] == 1:
                    q = (1 - alpha) * q + choice * beta_ses
                if r[sub_idx[ms_i], sess_idx[ms_i], t] == 0:
                    q = (1 - alphaforgetting) * q + choice * dmwater_ses
                qstay = qstay * (1 - alphastay)
                qstay[choice_10] = qstay[choice_10] + alphastay
                # Store variables
                predicted_choices.append(predicted_choice)
                Q.append(q)
                QLstay.append(qstay[0])
                QRstay.append(qstay[1])

            acc = []
            for i in np.arange(len(choices)):
                acc.append(1*(choices[i]==(1*(predicted_choices[i]>0.5))))
            #print(np.mean(acc))

            if NT_all[ms_i]!=0:
                ses_data = pd.DataFrame()
                ses_data['predicted_choice'] = predicted_choices
                ses_data['Qwater'] = Q
                ses_data['QLstay'] =  np.array(QLstay)
                ses_data['QRstay'] =  np.array(QRstay)
                ses_data['choices'] = choices
                ses_data['mouse'] = sub_idx[ms_i]
                ses_data['ses'] = sess_idx[ms_i]
                ses_data['acc'] = np.mean(acc)
                ses_data['id'] = ses_data['mouse']*100 + ses_data['ses']
                data = pd.concat([data,ses_data])
    return data

def reinforce_model_laserdecay_linear(standata,saved_params=None, fit=None, csv=True):
    if saved_params is not None:
        r =  standata['r']
        c =  standata['c']
        l =  standata['l']
        sub_idx =  standata['sub_idx']-1
        sess_idx = standata['sess_idx']-1
        NSxNSESS = standata['NSxNSESS']
        NT_all = standata['NT_all']
    else:
        r =  fit.data['r']
        c =  fit.data['c']
        l =  fit.data['l']
        sub_idx =  fit.data['sub_idx']-1
        sess_idx = fit.data['sess_idx']-1
    data = pd.DataFrame()

    for ms_i in np.arange(NSxNSESS):
        if NT_all[ms_i]>0:
            if saved_params is not None:
                if ms_i==0:
                    print('Using saved parameters')
                if csv==True:
                    dmwater_ses = float(saved_params.loc[saved_params['name']=='dmwater_ses['+str(ms_i+1)+']', 'Mean'])
                    dmlaser_ses = float(saved_params.loc[saved_params['name']=='dmlaser_ses['+str(ms_i+1)+']', 'Mean'])
                    beta_ses = float(saved_params.loc[saved_params['name']=='beta_ses['+str(ms_i+1)+']', 'Mean'])
                    side_ses = float(saved_params.loc[saved_params['name']=='sides['+str(ms_i+1)+']', 'Mean'])
                    laser_ses = float(saved_params.loc[saved_params['name']=='laser_ses['+str(ms_i+1)+']', 'Mean'])
                    alpha = float(saved_params.loc[saved_params['name']=='alpha_ses['+str(ms_i+1)+']', 'Mean'])
                    alphalaser = float(saved_params.loc[saved_params['name']=='alphalaser_ses['+str(ms_i+1)+']', 'Mean'])
                    alphaforgetting= float(saved_params.loc[saved_params['name']=='alphaforgetting_ses['+str(ms_i+1)+']', 'Mean'])
                    alphalaserforgetting= float(saved_params.loc[saved_params['name']=='alphalaserforgetting_ses['+str(ms_i+1)+']', 'Mean'])
                    laserdecay= float(saved_params.loc[saved_params['name']=='laserdecay_ses['+str(ms_i+1)+']', 'Mean'])

                else:
                    Print('Non csv version not avaialble atm')
            q = 0
            qlaser = 0
            predicted_choices=[]
            Q=[]
            Qlaser=[]
            choices=[]
            betalaser = laser_ses
            for t in np.arange(NT_all[ms_i]):
                t = int(t)
                choices.append(c[sub_idx[ms_i], sess_idx[ms_i], t])
                predicted_choice = inv_logit(side_ses + (q + qlaser))
                choice = 2 * c[sub_idx[ms_i], sess_idx[ms_i], t] -1
                if r[sub_idx[ms_i], sess_idx[ms_i], t] == 1:
                    q = (1 - alpha) * q + choice * beta_ses
                if r[sub_idx[ms_i], sess_idx[ms_i], t] == 0:
                    q = (1 - alphaforgetting) * q + choice * dmwater_ses
                if l[sub_idx[ms_i], sess_idx[ms_i], t] == 1:
                    qlaser = (1 - alphalaser) * qlaser + choice * betalaser
                    betalaser = betalaser - laserdecay
                if l[sub_idx[ms_i], sess_idx[ms_i], t] == 0:
                    qlaser = (1 - alphalaserforgetting) * qlaser + choice * dmlaser_ses
                # Store variables
                predicted_choices.append(predicted_choice)
                Q.append(q)
                Qlaser.append(qlaser)

            acc = []
            print(laser_ses-betalaser)
            for i in np.arange(len(choices)):
                acc.append(1*(choices[i]==(1*(predicted_choices[i]>0.5))))

            if NT_all[ms_i]!=0:
                ses_data = pd.DataFrame()
                ses_data['predicted_choice'] = predicted_choices
                ses_data['Qwater'] = Q
                ses_data['Qlaser'] = Qlaser
                ses_data['deltaQ'] = ses_data['Qwater'] + ses_data['Qlaser']
                ses_data['choices'] = choices
                ses_data['mouse'] = sub_idx[ms_i]
                ses_data['ses'] = sess_idx[ms_i]
                ses_data['acc'] = np.mean(acc)
                data = pd.concat([data,ses_data])
    return data
def reinforce_model_alldecay_exp(standata,saved_params=None, fit=None, csv=True):
    if saved_params is not None:
        r =  standata['r']
        c =  standata['c']
        l =  standata['l']
        sub_idx =  standata['sub_idx']-1
        sess_idx = standata['sess_idx']-1
        NSxNSESS = standata['NSxNSESS']
        NT_all = standata['NT_all']
    else:
        r =  fit.data['r']
        c =  fit.data['c']
        l =  fit.data['l']
        sub_idx =  fit.data['sub_idx']-1
        sess_idx = fit.data['sess_idx']-1
    data = pd.DataFrame()

    for ms_i in np.arange(NSxNSESS):
        if NT_all[ms_i]>0:
            if saved_params is not None:
                if ms_i==0:
                    print('Using saved parameters')
                if csv==True:
                    dmwater_ses = float(saved_params.loc[saved_params['name']=='dmwater_ses['+str(ms_i+1)+']', 'Mean'])
                    dmlaser_ses = float(saved_params.loc[saved_params['name']=='dmlaser_ses['+str(ms_i+1)+']', 'Mean'])
                    beta_ses = float(saved_params.loc[saved_params['name']=='beta_ses['+str(ms_i+1)+']', 'Mean'])
                    side_ses = float(saved_params.loc[saved_params['name']=='sides['+str(ms_i+1)+']', 'Mean'])
                    laser_ses = float(saved_params.loc[saved_params['name']=='laser_ses['+str(ms_i+1)+']', 'Mean'])
                    alpha = float(saved_params.loc[saved_params['name']=='alpha_ses['+str(ms_i+1)+']', 'Mean'])
                    alphalaser = float(saved_params.loc[saved_params['name']=='alphalaser_ses['+str(ms_i+1)+']', 'Mean'])
                    alphaforgetting= float(saved_params.loc[saved_params['name']=='alphaforgetting_ses['+str(ms_i+1)+']', 'Mean'])
                    alphalaserforgetting= float(saved_params.loc[saved_params['name']=='alphalaserforgetting_ses['+str(ms_i+1)+']', 'Mean'])
                    laserdecay= float(saved_params.loc[saved_params['name']=='laserdecay_ses['+str(ms_i+1)+']', 'Mean'])
                    waterdecay= float(saved_params.loc[saved_params['name']=='waterdecay_ses['+str(ms_i+1)+']', 'Mean'])
                else: 
                    Print('Non csv version not avaialble atm')
            q = 0
            qlaser = 0
            predicted_choices=[]
            Q=[]
            Qlaser=[]
            choices=[]
            betalaser = laser_ses
            betawater = beta_ses
            for t in np.arange(NT_all[ms_i]):
                t = int(t)
                choices.append(c[sub_idx[ms_i], sess_idx[ms_i], t])
                predicted_choice = inv_logit(side_ses + (q + qlaser))
                choice = 2 * c[sub_idx[ms_i], sess_idx[ms_i], t] -1
                if r[sub_idx[ms_i], sess_idx[ms_i], t] == 1:
                    betawater = (1-waterdecay)*betawater
                    q = (1 - alpha) * q + choice * betawater
                if r[sub_idx[ms_i], sess_idx[ms_i], t] == 0:
                    q = (1 - alphaforgetting) * q + choice * dmwater_ses
                if l[sub_idx[ms_i], sess_idx[ms_i], t] == 1:
                    betalaser = (1-laserdecay)*betalaser
                    qlaser = (1 - alphalaser) * qlaser + choice * betalaser
                if l[sub_idx[ms_i], sess_idx[ms_i], t] == 0:
                    qlaser = (1 - alphalaserforgetting) * qlaser + choice * dmlaser_ses
                # Store variables
                predicted_choices.append(predicted_choice)
                Q.append(q)
                Qlaser.append(qlaser)

            acc = []
            print(laser_ses-betalaser)

            for i in np.arange(len(choices)):
                acc.append(1*(choices[i]==(1*(predicted_choices[i]>0.5))))

            if NT_all[ms_i]!=0:
                ses_data = pd.DataFrame()
                ses_data['predicted_choice'] = predicted_choices
                ses_data['Qwater'] = Q
                ses_data['Qlaser'] = Qlaser
                ses_data['deltaQ'] = ses_data['Qwater'] + ses_data['Qlaser']
                ses_data['choices'] = choices
                ses_data['mouse'] = sub_idx[ms_i]
                ses_data['ses'] = sess_idx[ms_i]
                ses_data['acc'] = np.mean(acc)
                data = pd.concat([data,ses_data])
    return data
def reinforce_model_laserdecay_mixed_perseveration(standata,saved_params=None, fit=None, csv=True):
    if saved_params is not None:
        r =  standata['r']
        c =  standata['c']
        l =  standata['l']
        sub_idx =  standata['sub_idx']-1
        sess_idx = standata['sess_idx']-1
        NSxNSESS = standata['NSxNSESS']
        NT_all = standata['NT_all']
    else:
        r =  fit.data['r']
        c =  fit.data['c']
        l =  fit.data['l']
        sub_idx =  fit.data['sub_idx']-1
        sess_idx = fit.data['sess_idx']-1
    data = pd.DataFrame()

    for ms_i in np.arange(NSxNSESS):
        if NT_all[ms_i]>0:
            if saved_params is not None:
            #    if ms_i==0:
            #        print('Using saved parameters')
                if csv==True:
                    beta_ses = float(saved_params.loc[saved_params['name']=='beta_ses['+str(ms_i+1)+']', 'Mean'])
                    side_ses = float(saved_params.loc[saved_params['name']=='sides['+str(ms_i+1)+']', 'Mean'])
                    laser_ses = float(saved_params.loc[saved_params['name']=='laser_ses['+str(ms_i+1)+']', 'Mean'])
                    epses = float(saved_params.loc[saved_params['name']=='ep_ses['+str(ms_i+1)+']', 'Mean'])
                    alpha = float(saved_params.loc[saved_params['name']=='alpha_ses['+str(ms_i+1)+']', 'Mean'])
                    alphalaser = float(saved_params.loc[saved_params['name']=='alphalaser_ses['+str(ms_i+1)+']', 'Mean'])
                    alphastay = float(saved_params.loc[saved_params['name']=='alphastay_ses['+str(ms_i+1)+']', 'Mean'])
                    alphaforgetting= float(saved_params.loc[saved_params['name']=='alphaforgetting_ses['+str(ms_i+1)+']', 'Mean'])
                    alphalaserforgetting= float(saved_params.loc[saved_params['name']=='alphalaserforgetting_ses['+str(ms_i+1)+']', 'Mean'])
                    laserdecay= float(saved_params.loc[saved_params['name']=='laserdecay_ses['+str(ms_i+1)+']', 'Mean'])
                    betalaserbase= float(saved_params.loc[saved_params['name']=='betalaserbase_ses['+str(ms_i+1)+']', 'Mean'])
                    dmwater_ses = float(saved_params.loc[saved_params['name']=='dmwater_ses['+str(ms_i+1)+']', 'Mean'])
                    dmlaser_ses = float(saved_params.loc[saved_params['name']=='dmlaser_ses['+str(ms_i+1)+']', 'Mean'])
                    betalaserlossbase= float(saved_params.loc[saved_params['name']=='betalaserlossbase_ses['+str(ms_i+1)+']', 'Mean'])
                    laserdecayloss= float(saved_params.loc[saved_params['name']=='laserdecayloss_ses['+str(ms_i+1)+']', 'Mean'])
                else:
                    Print('Non csv version not avaialble atm')
            q = 0
            qlaser = 0
            qstay = np.zeros(2)
            predicted_choices=[]
            Q=[]
            Qlaser=[]
            QLstay=[]
            QRstay=[]
            choices=[]
            betalaser_buffer=[]
            betalaserloss_buffer=[]
            betalaser = laser_ses
            betalaserloss = dmlaser_ses
            for t in np.arange(NT_all[ms_i]):
                t = int(t)
                choices.append(c[sub_idx[ms_i], sess_idx[ms_i], t])
                predicted_choice = (1-epses) * inv_logit(side_ses + (q + qlaser)) + epses*(inv_logit(qstay[1]-qstay[0]))
                choice_10 = c[sub_idx[ms_i], sess_idx[ms_i], t]
                choice = 2 * c[sub_idx[ms_i], sess_idx[ms_i], t] -1
                if r[sub_idx[ms_i], sess_idx[ms_i], t] == 1:
                    q = (1 - alpha) * q + choice * beta_ses
                if r[sub_idx[ms_i], sess_idx[ms_i], t] == 0:
                    q = (1 - alphaforgetting) * q + choice * dmwater_ses
                if l[sub_idx[ms_i], sess_idx[ms_i], t] == 1:
                    betalaser = (1-laserdecay)*betalaser
                    qlaser = (1 - alphalaser) * qlaser + choice * (betalaser+betalaserbase)
                if l[sub_idx[ms_i], sess_idx[ms_i], t] == 0:
                    betalaserloss = (1-laserdecayloss)*betalaserloss
                    qlaser = (1 - alphalaserforgetting) * qlaser + choice * (betalaserloss+betalaserlossbase)               
                qstay = qstay * (1 - alphastay)
                qstay[choice_10] = qstay[choice_10] + alphastay
                # Store variables
                predicted_choices.append(predicted_choice)
                Q.append(q)
                betalaser_buffer.append(betalaser+betalaserbase)
                betalaserloss_buffer.append(betalaserloss+betalaserlossbase)
                Qlaser.append(qlaser)
                QLstay.append(qstay[0])
                QRstay.append(qstay[1])

            acc = []
            for i in np.arange(len(choices)):
                acc.append(1*(choices[i]==(1*(predicted_choices[i]>0.5))))
            #print(np.mean(acc))

            if NT_all[ms_i]!=0:
                ses_data = pd.DataFrame()
                ses_data['predicted_choice'] = predicted_choices
                ses_data['Qwater'] = Q
                ses_data['Qlaser'] = Qlaser
                ses_data['QLstay'] =  np.array(QLstay)
                ses_data['QRstay'] =  np.array(QRstay)
                ses_data['choices'] = choices
                ses_data['betalaser'] = betalaser_buffer
                ses_data['betalaserloss'] = betalaserloss_buffer
                ses_data['mouse'] = sub_idx[ms_i]
                ses_data['ses'] = sess_idx[ms_i]
                ses_data['acc'] = np.mean(acc)
                data = pd.concat([data,ses_data])
    return data
def reinforce_model_alphalaserdecay_mixed_perseveration(standata,saved_params=None, fit=None, csv=True):
    if saved_params is not None:
        r =  standata['r']
        c =  standata['c']
        l =  standata['l']
        sub_idx =  standata['sub_idx']-1
        sess_idx = standata['sess_idx']-1
        NSxNSESS = standata['NSxNSESS']
        NT_all = standata['NT_all']
    else:
        r =  fit.data['r']
        c =  fit.data['c']
        l =  fit.data['l']
        sub_idx =  fit.data['sub_idx']-1
        sess_idx = fit.data['sess_idx']-1
    data = pd.DataFrame()

    for ms_i in np.arange(NSxNSESS):
        if NT_all[ms_i]>0:
            if saved_params is not None:
            #    if ms_i==0:
            #        print('Using saved parameters')
                if csv==True:
                    beta_ses = float(saved_params.loc[saved_params['name']=='beta_ses['+str(ms_i+1)+']', 'Mean'])
                    side_ses = float(saved_params.loc[saved_params['name']=='sides['+str(ms_i+1)+']', 'Mean'])
                    laser_ses = float(saved_params.loc[saved_params['name']=='laser_ses['+str(ms_i+1)+']', 'Mean'])
                    epses = float(saved_params.loc[saved_params['name']=='ep_ses['+str(ms_i+1)+']', 'Mean'])
                    alpha = float(saved_params.loc[saved_params['name']=='alpha_ses['+str(ms_i+1)+']', 'Mean'])
                    alphalaser = float(saved_params.loc[saved_params['name']=='alphalaser_ses['+str(ms_i+1)+']', 'Mean'])
                    alphastay = float(saved_params.loc[saved_params['name']=='alphastay_ses['+str(ms_i+1)+']', 'Mean'])
                    alphaforgetting= float(saved_params.loc[saved_params['name']=='alphaforgetting_ses['+str(ms_i+1)+']', 'Mean'])
                    alphalaserforgetting= float(saved_params.loc[saved_params['name']=='alphalaserforgetting_ses['+str(ms_i+1)+']', 'Mean'])
                    laserdecay= float(saved_params.loc[saved_params['name']=='laserdecay_ses['+str(ms_i+1)+']', 'Mean'])
                    betalaserbase= float(saved_params.loc[saved_params['name']=='betalaserbase_ses['+str(ms_i+1)+']', 'Mean'])
                    dmwater_ses = float(saved_params.loc[saved_params['name']=='dmwater_ses['+str(ms_i+1)+']', 'Mean'])
                    dmlaser_ses = float(saved_params.loc[saved_params['name']=='dmlaser_ses['+str(ms_i+1)+']', 'Mean'])
                else:
                    Print('Non csv version not avaialble atm')
            q = 0
            qlaser = 0
            qstay = np.zeros(2)
            predicted_choices=[]
            Q=[]
            Qlaser=[]
            QLstay=[]
            QRstay=[]
            choices=[]
            betalaser_buffer=[]
            betalaserloss_buffer=[]
            betalaser = laser_ses
            betalaserloss = dmlaser_ses
            for t in np.arange(NT_all[ms_i]):
                t = int(t)
                choices.append(c[sub_idx[ms_i], sess_idx[ms_i], t])
                predicted_choice = (1-epses) * inv_logit(side_ses + (q + qlaser)) + epses*(inv_logit(qstay[1]-qstay[0]))
                choice_10 = c[sub_idx[ms_i], sess_idx[ms_i], t]
                choice = 2 * c[sub_idx[ms_i], sess_idx[ms_i], t] -1
                if r[sub_idx[ms_i], sess_idx[ms_i], t] == 1:
                    q = (1 - alpha) * q + choice * beta_ses
                if r[sub_idx[ms_i], sess_idx[ms_i], t] == 0:
                    q = (1 - alphaforgetting) * q + choice * dmwater_ses
                if l[sub_idx[ms_i], sess_idx[ms_i], t] == 1:
                    qlaser = (1 - (betalaser+betalaserbase)) * qlaser + choice * alphalaser
                    betalaser = (1-laserdecay)*betalaser
                if l[sub_idx[ms_i], sess_idx[ms_i], t] == 0:
                    qlaser = (1 - alphalaserforgetting) * qlaser + choice * betalaserloss               
                qstay = qstay * (1 - alphastay)
                qstay[choice_10] = qstay[choice_10] + alphastay
                # Store variables
                predicted_choices.append(predicted_choice)
                Q.append(q)
                betalaser_buffer.append(betalaser+betalaserbase)
                betalaserloss_buffer.append(betalaserloss)
                Qlaser.append(qlaser)
                QLstay.append(qstay[0])
                QRstay.append(qstay[1])

            acc = []
            for i in np.arange(len(choices)):
                acc.append(1*(choices[i]==(1*(predicted_choices[i]>0.5))))
            #print(np.mean(acc))

            if NT_all[ms_i]!=0:
                ses_data = pd.DataFrame()
                ses_data['predicted_choice'] = predicted_choices
                ses_data['Qwater'] = Q
                ses_data['Qlaser'] = Qlaser
                ses_data['QLstay'] =  np.array(QLstay)
                ses_data['QRstay'] =  np.array(QRstay)
                ses_data['choices'] = choices
                ses_data['betalaser'] = betalaser_buffer
                ses_data['betalaserloss'] = betalaserloss_buffer
                ses_data['mouse'] = sub_idx[ms_i]
                ses_data['ses'] = sess_idx[ms_i]
                ses_data['acc'] = np.mean(acc)
                data = pd.concat([data,ses_data])
    return data
def reinforce_model_laserdecay_mixed_perseveration_noloss(standata,saved_params=None, fit=None, csv=True):
    if saved_params is not None:
        r =  standata['r']
        c =  standata['c']
        l =  standata['l']
        sub_idx =  standata['sub_idx']-1
        sess_idx = standata['sess_idx']-1
        NSxNSESS = standata['NSxNSESS']
        NT_all = standata['NT_all']
    else:
        r =  fit.data['r']
        c =  fit.data['c']
        l =  fit.data['l']
        sub_idx =  fit.data['sub_idx']-1
        sess_idx = fit.data['sess_idx']-1
    data = pd.DataFrame()

    for ms_i in np.arange(NSxNSESS):
        if NT_all[ms_i]>0:
            if saved_params is not None:
            #    if ms_i==0:
            #        print('Using saved parameters')
                if csv==True:
                    beta_ses = float(saved_params.loc[saved_params['name']=='beta_ses['+str(ms_i+1)+']', 'Mean'])
                    side_ses = float(saved_params.loc[saved_params['name']=='sides['+str(ms_i+1)+']', 'Mean'])
                    laser_ses = float(saved_params.loc[saved_params['name']=='laser_ses['+str(ms_i+1)+']', 'Mean'])
                    epses = float(saved_params.loc[saved_params['name']=='ep_ses['+str(ms_i+1)+']', 'Mean'])
                    alpha = float(saved_params.loc[saved_params['name']=='alpha_ses['+str(ms_i+1)+']', 'Mean'])
                    alphalaser = float(saved_params.loc[saved_params['name']=='alphalaser_ses['+str(ms_i+1)+']', 'Mean'])
                    alphastay = float(saved_params.loc[saved_params['name']=='alphastay_ses['+str(ms_i+1)+']', 'Mean'])
                    alphaforgetting= float(saved_params.loc[saved_params['name']=='alphaforgetting_ses['+str(ms_i+1)+']', 'Mean'])
                    alphalaserforgetting= float(saved_params.loc[saved_params['name']=='alphalaserforgetting_ses['+str(ms_i+1)+']', 'Mean'])
                    laserdecay= float(saved_params.loc[saved_params['name']=='laserdecay_ses['+str(ms_i+1)+']', 'Mean'])
                    betalaserbase= float(saved_params.loc[saved_params['name']=='betalaserbase_ses['+str(ms_i+1)+']', 'Mean'])
                else:
                    Print('Non csv version not avaialble atm')
            q = 0
            qlaser = 0
            qstay = np.zeros(2)
            predicted_choices=[]
            Q=[]
            Qlaser=[]
            QLstay=[]
            QRstay=[]
            choices=[]
            betalaser = laser_ses
            for t in np.arange(NT_all[ms_i]):
                t = int(t)
                choices.append(c[sub_idx[ms_i], sess_idx[ms_i], t])
                predicted_choice = (1-epses) * inv_logit(side_ses + (q + qlaser)) + epses*(inv_logit(qstay[1]-qstay[0]))
                choice_10 = c[sub_idx[ms_i], sess_idx[ms_i], t]
                choice = 2 * c[sub_idx[ms_i], sess_idx[ms_i], t] -1
                if r[sub_idx[ms_i], sess_idx[ms_i], t] == 1:
                    q = (1 - alpha) * q + choice * beta_ses
                if r[sub_idx[ms_i], sess_idx[ms_i], t] == 0:
                    q = (1 - alphaforgetting) * q
                if l[sub_idx[ms_i], sess_idx[ms_i], t] == 1:
                    betalaser = (1-laserdecay)*betalaser
                    qlaser = (1 - alphalaser) * qlaser + choice * (betalaser+betalaserbase)
                if l[sub_idx[ms_i], sess_idx[ms_i], t] == 0:
                    qlaser = (1 - alphalaserforgetting) * qlaser
                qstay = qstay * (1 - alphastay)
                qstay[choice_10] = qstay[choice_10] + alphastay
                # Store variables
                predicted_choices.append(predicted_choice)
                Q.append(q)
                Qlaser.append(qlaser)
                QLstay.append(qstay[0])
                QRstay.append(qstay[1])

            acc = []
            for i in np.arange(len(choices)):
                acc.append(1*(choices[i]==(1*(predicted_choices[i]>0.5))))
            #print(np.mean(acc))

            if NT_all[ms_i]!=0:
                ses_data = pd.DataFrame()
                ses_data['predicted_choice'] = predicted_choices
                ses_data['Qwater'] = Q
                ses_data['Qlaser'] = Qlaser
                ses_data['QLstay'] =  np.array(QLstay)
                ses_data['QRstay'] =  np.array(QRstay)
                ses_data['choices'] = choices
                ses_data['mouse'] = sub_idx[ms_i]
                ses_data['ses'] = sess_idx[ms_i]
                ses_data['acc'] = np.mean(acc)
                data = pd.concat([data,ses_data])
    return data
def reinforce_model_laserdecaywinloss_exp(standata,saved_params=None, fit=None, csv=True):
    if saved_params is not None:
        r =  standata['r']
        c =  standata['c']
        l =  standata['l']
        sub_idx =  standata['sub_idx']-1
        sess_idx = standata['sess_idx']-1
        NSxNSESS = standata['NSxNSESS']
        NT_all = standata['NT_all']
    else:
        r =  fit.data['r']
        c =  fit.data['c']
        l =  fit.data['l']
        sub_idx =  fit.data['sub_idx']-1
        sess_idx = fit.data['sess_idx']-1
    data = pd.DataFrame()

    for ms_i in np.arange(NSxNSESS):
        if NT_all[ms_i]>0:
            if saved_params is not None:
            #    if ms_i==0:
            #        print('Using saved parameters')
                if csv==True:
                    dmwater_ses = float(saved_params.loc[saved_params['name']=='dmwater_ses['+str(ms_i+1)+']', 'Mean'])
                    dmlaser_ses = float(saved_params.loc[saved_params['name']=='dmlaser_ses['+str(ms_i+1)+']', 'Mean'])
                    beta_ses = float(saved_params.loc[saved_params['name']=='beta_ses['+str(ms_i+1)+']', 'Mean'])
                    side_ses = float(saved_params.loc[saved_params['name']=='sides['+str(ms_i+1)+']', 'Mean'])
                    laser_ses = float(saved_params.loc[saved_params['name']=='laser_ses['+str(ms_i+1)+']', 'Mean'])
                    alpha = float(saved_params.loc[saved_params['name']=='alpha_ses['+str(ms_i+1)+']', 'Mean'])
                    alphalaser = float(saved_params.loc[saved_params['name']=='alphalaser_ses['+str(ms_i+1)+']', 'Mean'])
                    alphaforgetting= float(saved_params.loc[saved_params['name']=='alphaforgetting_ses['+str(ms_i+1)+']', 'Mean'])
                    alphalaserforgetting= float(saved_params.loc[saved_params['name']=='alphalaserforgetting_ses['+str(ms_i+1)+']', 'Mean'])
                    laserdecay= float(saved_params.loc[saved_params['name']=='laserdecay_ses['+str(ms_i+1)+']', 'Mean'])
                    betalaserlossbase= float(saved_params.loc[saved_params['name']=='betalaserlossbase_ses['+str(ms_i+1)+']', 'Mean'])
                    betalaserbase= float(saved_params.loc[saved_params['name']=='betalaserbase_ses['+str(ms_i+1)+']', 'Mean'])
                    laserdecayloss= float(saved_params.loc[saved_params['name']=='laserdecayloss_ses['+str(ms_i+1)+']', 'Mean'])
                else:
                    Print('Non csv version not avaialble atm')
            q = 0
            qlaser = 0
            predicted_choices=[]
            Q=[]
            Qlaser=[]
            choices=[]
            betalaser = laser_ses
            betalaserloss = dmlaser_ses
            for t in np.arange(NT_all[ms_i]):
                t = int(t)
                choices.append(c[sub_idx[ms_i], sess_idx[ms_i], t])
                predicted_choice = inv_logit(side_ses + (q + qlaser))
                choice = 2 * c[sub_idx[ms_i], sess_idx[ms_i], t] -1
                if r[sub_idx[ms_i], sess_idx[ms_i], t] == 1:
                    q = (1 - alpha) * q + choice * beta_ses
                if r[sub_idx[ms_i], sess_idx[ms_i], t] == 0:
                    q = (1 - alphaforgetting) * q + choice * dmwater_ses
                if l[sub_idx[ms_i], sess_idx[ms_i], t] == 1:
                    betalaser = (1-laserdecay)*betalaser
                    qlaser = (1 - alphalaser) * qlaser + choice * (betalaser+betalaserbase)
                if l[sub_idx[ms_i], sess_idx[ms_i], t] == 0:
                    betalaserloss = (1-laserdecayloss)*betalaserloss
                    qlaser = (1 - alphalaserforgetting) * qlaser + choice * (betalaserloss+betalaserlossbase)
                # Store variables
                predicted_choices.append(predicted_choice)
                Q.append(q)
                Qlaser.append(qlaser)

            acc = []
            #print(laser_ses-betalaser)

            for i in np.arange(len(choices)):
                acc.append(1*(choices[i]==(1*(predicted_choices[i]>0.5))))

            if NT_all[ms_i]!=0:
                ses_data = pd.DataFrame()
                ses_data['predicted_choice'] = predicted_choices
                ses_data['Qwater'] = Q
                ses_data['Qlaser'] = Qlaser
                ses_data['deltaQ'] = ses_data['Qwater'] + ses_data['Qlaser']
                ses_data['choices'] = choices
                ses_data['mouse'] = sub_idx[ms_i]
                ses_data['ses'] = sess_idx[ms_i]
                ses_data['acc'] = np.mean(acc)
                data = pd.concat([data,ses_data])
    return data
def reinforce_model_waterlaserdecaywinloss_exp(standata,saved_params=None, fit=None, csv=True):
    if saved_params is not None:
        r =  standata['r']
        c =  standata['c']
        l =  standata['l']
        sub_idx =  standata['sub_idx']-1
        sess_idx = standata['sess_idx']-1
        NSxNSESS = standata['NSxNSESS']
        NT_all = standata['NT_all']
    else:
        r =  fit.data['r']
        c =  fit.data['c']
        l =  fit.data['l']
        sub_idx =  fit.data['sub_idx']-1
        sess_idx = fit.data['sess_idx']-1
    data = pd.DataFrame()

    for ms_i in np.arange(NSxNSESS):
        if NT_all[ms_i]>0:
            if saved_params is not None:
            #    if ms_i==0:
            #        print('Using saved parameters')
                if csv==True:
                    dmwater_ses = float(saved_params.loc[saved_params['name']=='dmwater_ses['+str(ms_i+1)+']', 'Mean'])
                    dmlaser_ses = float(saved_params.loc[saved_params['name']=='dmlaser_ses['+str(ms_i+1)+']', 'Mean'])
                    beta_ses = float(saved_params.loc[saved_params['name']=='beta_ses['+str(ms_i+1)+']', 'Mean'])
                    side_ses = float(saved_params.loc[saved_params['name']=='sides['+str(ms_i+1)+']', 'Mean'])
                    laser_ses = float(saved_params.loc[saved_params['name']=='laser_ses['+str(ms_i+1)+']', 'Mean'])
                    alpha = float(saved_params.loc[saved_params['name']=='alpha_ses['+str(ms_i+1)+']', 'Mean'])
                    alphalaser = float(saved_params.loc[saved_params['name']=='alphalaser_ses['+str(ms_i+1)+']', 'Mean'])
                    alphaforgetting= float(saved_params.loc[saved_params['name']=='alphaforgetting_ses['+str(ms_i+1)+']', 'Mean'])
                    alphalaserforgetting= float(saved_params.loc[saved_params['name']=='alphalaserforgetting_ses['+str(ms_i+1)+']', 'Mean'])
                    laserdecay= float(saved_params.loc[saved_params['name']=='laserdecay_ses['+str(ms_i+1)+']', 'Mean'])
                    betalaserlossbase= float(saved_params.loc[saved_params['name']=='betalaserlossbase_ses['+str(ms_i+1)+']', 'Mean'])
                    betalaserbase= float(saved_params.loc[saved_params['name']=='betalaserbase_ses['+str(ms_i+1)+']', 'Mean'])
                    laserdecayloss= float(saved_params.loc[saved_params['name']=='laserdecayloss_ses['+str(ms_i+1)+']', 'Mean'])
                    waterdecay= float(saved_params.loc[saved_params['name']=='waterdecay_ses['+str(ms_i+1)+']', 'Mean'])
                    betawaterlossbase= float(saved_params.loc[saved_params['name']=='betawaterlossbase_ses['+str(ms_i+1)+']', 'Mean'])
                    betawaterbase= float(saved_params.loc[saved_params['name']=='betawaterbase_ses['+str(ms_i+1)+']', 'Mean'])
                    waterdecayloss= float(saved_params.loc[saved_params['name']=='waterdecayloss_ses['+str(ms_i+1)+']', 'Mean'])

                else:
                    Print('Non csv version not avaialble atm')
            q = 0
            qlaser = 0
            predicted_choices=[]
            Q=[]
            Qlaser=[]
            choices=[]
            betawater = beta_ses
            betalaser = laser_ses
            betalaserloss = dmlaser_ses
            betawaterloss = dmwater_ses
            for t in np.arange(NT_all[ms_i]):
                t = int(t)
                choices.append(c[sub_idx[ms_i], sess_idx[ms_i], t])
                predicted_choice = inv_logit(side_ses + (q + qlaser))
                choice = 2 * c[sub_idx[ms_i], sess_idx[ms_i], t] -1
                if r[sub_idx[ms_i], sess_idx[ms_i], t] == 1:
                    betawater = (1-waterdecay)*betawater
                    q = (1 - alpha) * q + choice * (betawater+betawaterbase)
                if r[sub_idx[ms_i], sess_idx[ms_i], t] == 0:
                    betawaterloss = (1-waterdecayloss)*betawaterloss
                    q = (1 - alphaforgetting) * q + choice * (betawaterloss+betawaterlossbase)
                if l[sub_idx[ms_i], sess_idx[ms_i], t] == 1:
                    betalaser = (1-laserdecay)*betalaser
                    qlaser = (1 - alphalaser) * qlaser + choice * (betalaser+betalaserbase)
                if l[sub_idx[ms_i], sess_idx[ms_i], t] == 0:
                    betalaserloss = (1-laserdecayloss)*betalaserloss
                    qlaser = (1 - alphalaserforgetting) * qlaser + choice * (betalaserloss+betalaserlossbase)
                # Store variables
                predicted_choices.append(predicted_choice)
                Q.append(q)
                Qlaser.append(qlaser)

            acc = []
            #print(laser_ses-betalaser)

            for i in np.arange(len(choices)):
                acc.append(1*(choices[i]==(1*(predicted_choices[i]>0.5))))

            if NT_all[ms_i]!=0:
                ses_data = pd.DataFrame()
                ses_data['predicted_choice'] = predicted_choices
                ses_data['Qwater'] = Q
                ses_data['Qlaser'] = Qlaser
                ses_data['deltaQ'] = ses_data['Qwater'] + ses_data['Qlaser']
                ses_data['choices'] = choices
                ses_data['mouse'] = sub_idx[ms_i]
                ses_data['ses'] = sess_idx[ms_i]
                ses_data['acc'] = np.mean(acc)
                data = pd.concat([data,ses_data])
    return data
def reinforce_model_laserdecay_exp(standata,saved_params=None, fit=None, csv=True):
    if saved_params is not None:
        r =  standata['r']
        c =  standata['c']
        l =  standata['l']
        sub_idx =  standata['sub_idx']-1
        sess_idx = standata['sess_idx']-1
        NSxNSESS = standata['NSxNSESS']
        NT_all = standata['NT_all']
    else:
        r =  fit.data['r']
        c =  fit.data['c']
        l =  fit.data['l']
        sub_idx =  fit.data['sub_idx']-1
        sess_idx = fit.data['sess_idx']-1
    data = pd.DataFrame()

    for ms_i in np.arange(NSxNSESS):
        if NT_all[ms_i]>0:
            if saved_params is not None:
                if ms_i==0:
                    print('Using saved parameters')
                if csv==True:
                    dmwater_ses = float(saved_params.loc[saved_params['name']=='dmwater_ses['+str(ms_i+1)+']', 'Mean'])
                    dmlaser_ses = float(saved_params.loc[saved_params['name']=='dmlaser_ses['+str(ms_i+1)+']', 'Mean'])
                    beta_ses = float(saved_params.loc[saved_params['name']=='beta_ses['+str(ms_i+1)+']', 'Mean'])
                    side_ses = float(saved_params.loc[saved_params['name']=='sides['+str(ms_i+1)+']', 'Mean'])
                    laser_ses = float(saved_params.loc[saved_params['name']=='laser_ses['+str(ms_i+1)+']', 'Mean'])
                    alpha = float(saved_params.loc[saved_params['name']=='alpha_ses['+str(ms_i+1)+']', 'Mean'])
                    alphalaser = float(saved_params.loc[saved_params['name']=='alphalaser_ses['+str(ms_i+1)+']', 'Mean'])
                    alphaforgetting= float(saved_params.loc[saved_params['name']=='alphaforgetting_ses['+str(ms_i+1)+']', 'Mean'])
                    alphalaserforgetting= float(saved_params.loc[saved_params['name']=='alphalaserforgetting_ses['+str(ms_i+1)+']', 'Mean'])
                    laserdecay= float(saved_params.loc[saved_params['name']=='laserdecay_ses['+str(ms_i+1)+']', 'Mean'])
                else:
                    Print('Non csv version not avaialble atm')
            q = 0
            qlaser = 0
            predicted_choices=[]
            Q=[]
            Qlaser=[]
            choices=[]
            betalaser = laser_ses
            for t in np.arange(NT_all[ms_i]):
                t = int(t)
                choices.append(c[sub_idx[ms_i], sess_idx[ms_i], t])
                predicted_choice = inv_logit(side_ses + (q + qlaser))
                choice = 2 * c[sub_idx[ms_i], sess_idx[ms_i], t] -1
                if r[sub_idx[ms_i], sess_idx[ms_i], t] == 1:
                    q = (1 - alpha) * q + choice * beta_ses
                if r[sub_idx[ms_i], sess_idx[ms_i], t] == 0:
                    q = (1 - alphaforgetting) * q + choice * dmwater_ses
                if l[sub_idx[ms_i], sess_idx[ms_i], t] == 1:
                    qlaser = (1 - alphalaser) * qlaser + choice * betalaser
                    betalaser = (1-laserdecay)*betalaser
                if l[sub_idx[ms_i], sess_idx[ms_i], t] == 0:
                    qlaser = (1 - alphalaserforgetting) * qlaser + choice * dmlaser_ses
                # Store variables
                predicted_choices.append(predicted_choice)
                Q.append(q)
                Qlaser.append(qlaser)

            acc = []
            print(laser_ses-betalaser)

            for i in np.arange(len(choices)):
                acc.append(1*(choices[i]==(1*(predicted_choices[i]>0.5))))

            if NT_all[ms_i]!=0:
                ses_data = pd.DataFrame()
                ses_data['predicted_choice'] = predicted_choices
                ses_data['Qwater'] = Q
                ses_data['Qlaser'] = Qlaser
                ses_data['deltaQ'] = ses_data['Qwater'] + ses_data['Qlaser']
                ses_data['choices'] = choices
                ses_data['mouse'] = sub_idx[ms_i]
                ses_data['ses'] = sess_idx[ms_i]
                ses_data['acc'] = np.mean(acc)
                data = pd.concat([data,ses_data])
    return data
def simulate_reinforce_model(standata_recovery,saved_params=None, fit=None, csv=True):
    b =  standata_recovery['b']
    p =  standata_recovery['p']
    tb = standata_recovery['tb']
    NS =standata_recovery['NS']
    NSESS=standata_recovery['NSESS']
    NT =standata_recovery['NT']
    NT_all =standata_recovery['NT_all']
    r_sim = np.zeros([NS,NSESS,NT])
    c_sim = np.zeros([NS,NSESS,NT])
    l_sim = np.zeros([NS,NSESS,NT])
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
                    dmlaser_ses = float(saved_params.loc[saved_params['name']=='dmlaser_ses['+str(ms_i+1)+']', 'Mean'])
                    beta_ses = float(saved_params.loc[saved_params['name']=='beta_ses['+str(ms_i+1)+']', 'Mean'])
                    side_ses = float(saved_params.loc[saved_params['name']=='sides['+str(ms_i+1)+']', 'Mean'])
                    laser_ses = float(saved_params.loc[saved_params['name']=='laser_ses['+str(ms_i+1)+']', 'Mean'])
                    alpha = float(saved_params.loc[saved_params['name']=='alpha_ses['+str(ms_i+1)+']', 'Mean'])
                    alphalaser = float(saved_params.loc[saved_params['name']=='alphalaser_ses['+str(ms_i+1)+']', 'Mean'])
                    alphaforgetting= float(saved_params.loc[saved_params['name']=='alphaforgetting_ses['+str(ms_i+1)+']', 'Mean'])
                    alphalaserforgetting= float(saved_params.loc[saved_params['name']=='alphalaserforgetting_ses['+str(ms_i+1)+']', 'Mean'])
                else:
                    Print('Non csv version not avaialble atm')

            q = 0
            qlaser = 0
            predicted_choices=[]
            rewards = []
            lasers = []
            outcome=[]

            for t in np.arange(NT_all[ms_i]):
                t = int(t)
                p_choice = inv_logit(side_ses + (q + qlaser))
                choice =  np.random.choice(2,p= [1-p_choice,p_choice])
                predicted_choices.append(choice)
                choice = 2 * choice - 1
                assert np.isin(p[sub_idx[ms_i], sess_idx[ms_i],t], [0.7,0.1])
                p_left= p[sub_idx[ms_i], sess_idx[ms_i],t]
                l = b[sub_idx[ms_i], sess_idx[ms_i],t]
                if p_left==0.7:
                    p_right=0.1
                else:
                    p_right=0.7
                if choice==1:
                    f = np.random.choice([0,1], p=[1-p_right,p_right])
                else:
                    f = np.random.choice([0,1], p=[1-p_left,p_left])

                if (1*(f*(l==0))) == 1:
                    q = (1 - alpha) * q + choice * beta_ses
                if (1*(f*(l==0))) == 0:
                    q = (1 - alphaforgetting) * q + choice * dmwater_ses
                if (f * l) == 1:
                    qlaser = (1 - alphalaser) * qlaser + choice * laser_ses
                if (f * l)== 0:
                    qlaser = (1 - alphalaserforgetting) * qlaser + choice * dmlaser_ses

                rewards.append((1*(f*(l==0))))
                lasers.append(f*l)
                outcome.append(f)
                # Store variables

            if NT_all[ms_i]!=0:
                c_sim[sub_idx[ms_i], sess_idx[ms_i], :int(NT_all[ms_i])]= predicted_choices
                l_sim[sub_idx[ms_i], sess_idx[ms_i], :int(NT_all[ms_i])] = lasers
                r_sim[sub_idx[ms_i], sess_idx[ms_i], :int(NT_all[ms_i])] = rewards
                ses_data = pd.DataFrame()
                ses_data['choices'] = predicted_choices
                ses_data['water'] = rewards
                ses_data['laser'] = lasers
                ses_data['laser_block'] = b[sub_idx[ms_i], sess_idx[ms_i],:int(NT_all[ms_i])]
                ses_data['tb'] = tb[sub_idx[ms_i], sess_idx[ms_i],:int(NT_all[ms_i])]
                ses_data['probabilityLeft'] = p[sub_idx[ms_i], sess_idx[ms_i],:int(NT_all[ms_i])]
                ses_data['outcome'] = outcome
                ses_data['mouse'] = sub_idx[ms_i]
                ses_data['ses'] = sess_idx[ms_i]
                sim_data = pd.concat([sim_data,ses_data])

    return c_sim, l_sim, r_sim, sim_data

def reinforce_model_w_stay(standata,saved_params=None, fit=None, csv=True):
    if saved_params is not None:
        r =  standata['r']
        c =  standata['c']
        l =  standata['l']
        sub_idx =  standata['sub_idx']-1
        sess_idx = standata['sess_idx']-1
        NSxNSESS = standata['NSxNSESS']
        NT_all = standata['NT_all']
    else:
        r =  fit.data['r']
        c =  fit.data['c']
        l =  fit.data['l']
        sub_idx =  fit.data['sub_idx']-1
        sess_idx = fit.data['sess_idx']-1
    data = pd.DataFrame()

    for ms_i in np.arange(NSxNSESS):
        if NT_all[ms_i]>0:
            if saved_params is not None:
            #    if ms_i==0:
            #        print('Using saved parameters')
                if csv==True:
                    stay_ses = float(saved_params.loc[saved_params['name']=='stay_ses['+str(ms_i+1)+']', 'Mean'])
                    beta_ses = float(saved_params.loc[saved_params['name']=='beta_ses['+str(ms_i+1)+']', 'Mean'])
                    side_ses = float(saved_params.loc[saved_params['name']=='sides['+str(ms_i+1)+']', 'Mean'])
                    laser_ses = float(saved_params.loc[saved_params['name']=='laser_ses['+str(ms_i+1)+']', 'Mean'])
                    alpha = float(saved_params.loc[saved_params['name']=='alpha_ses['+str(ms_i+1)+']', 'Mean'])
                    alphalaser = float(saved_params.loc[saved_params['name']=='alphalaser_ses['+str(ms_i+1)+']', 'Mean'])
                    alphastay= float(saved_params.loc[saved_params['name']=='alphastay_ses['+str(ms_i+1)+']', 'Mean'])
                else:
                    Print('Non csv version not avaialble atm')
            q = 0
            qlaser = 0
            qstay = 0
            predicted_choices=[]
            Q=[]
            Qlaser=[]
            Qstay=[]
            choices=[]
            for t in np.arange(NT_all[ms_i]):
                t = int(t)
                choices.append(c[sub_idx[ms_i], sess_idx[ms_i], t])
                predicted_choice = inv_logit(side_ses + (q + qlaser + qstay))
                choice = 2 * c[sub_idx[ms_i], sess_idx[ms_i], t] -1
                if r[sub_idx[ms_i], sess_idx[ms_i], t] == 1:
                    q = (1 - alpha) * q + choice * beta_ses
                if l[sub_idx[ms_i], sess_idx[ms_i], t] == 1:
                    qlaser = (1 - alphalaser) * qlaser + choice * laser_ses
                qstay = (1 - alphastay) * qstay + choice * stay_ses
                # Store variables
                predicted_choices.append(predicted_choice)
                Q.append(q)
                Qlaser.append(qlaser)
                Qstay.append(qstay)
            acc = []
            for i in np.arange(len(choices)):
                acc.append(1*(choices[i]==(1*(predicted_choices[i]>0.5))))
            #print(np.mean(acc))

            if NT_all[ms_i]!=0:
                ses_data = pd.DataFrame()
                ses_data['predicted_choice'] = predicted_choices
                ses_data['Qwater'] = Q
                ses_data['bias'] = side_ses
                ses_data['Qlaser'] = Qlaser
                ses_data['Qstay'] = Qstay
                ses_data['deltaQ'] = ses_data['Qwater'] + ses_data['Qlaser'] + ses_data['Qstay']
                ses_data['choices'] = choices
                ses_data['mouse'] = sub_idx[ms_i]
                ses_data['ses'] = sess_idx[ms_i]
                ses_data['acc'] = np.mean(acc)
                data = pd.concat([data,ses_data])
    return data
def simulate_reinforce_model_w_stay(standata_recovery,saved_params=None, fit=None, csv=True):
    b =  standata_recovery['b']
    p =  standata_recovery['p']
    tb = standata_recovery['tb']
    NS =standata_recovery['NS']
    NSESS=standata_recovery['NSESS']
    NT =standata_recovery['NT']
    NT_all =standata_recovery['NT_all']
    r_sim = np.zeros([NS,NSESS,NT])
    c_sim = np.zeros([NS,NSESS,NT])
    l_sim = np.zeros([NS,NSESS,NT])
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
                    stay_ses = float(saved_params.loc[saved_params['name']=='stay_ses['+str(ms_i+1)+']', 'Mean'])
                    beta_ses = float(saved_params.loc[saved_params['name']=='beta_ses['+str(ms_i+1)+']', 'Mean'])
                    side_ses = float(saved_params.loc[saved_params['name']=='sides['+str(ms_i+1)+']', 'Mean'])
                    laser_ses = float(saved_params.loc[saved_params['name']=='laser_ses['+str(ms_i+1)+']', 'Mean'])
                    alpha = float(saved_params.loc[saved_params['name']=='alpha_ses['+str(ms_i+1)+']', 'Mean'])
                    alphalaser = float(saved_params.loc[saved_params['name']=='alphalaser_ses['+str(ms_i+1)+']', 'Mean'])
                    alphastay= float(saved_params.loc[saved_params['name']=='alphastay_ses['+str(ms_i+1)+']', 'Mean'])
                else:
                    Print('Non csv version not avaialble atm')
            q = 0
            qlaser = 0
            qstay = 0
            predicted_choices=[]
            rewards = []
            lasers = []
            outcome=[]

            for t in np.arange(NT_all[ms_i]):
                t = int(t)
                p_choice = inv_logit(side_ses + (q + qlaser + qstay))
                choice =  np.random.choice(2,p= [1-p_choice,p_choice])
                predicted_choices.append(choice)
                choice = 2 * choice - 1
                assert np.isin(p[sub_idx[ms_i], sess_idx[ms_i],t], [0.7,0.1])
                p_left= p[sub_idx[ms_i], sess_idx[ms_i],t]
                l = b[sub_idx[ms_i], sess_idx[ms_i],t]
                if p_left==0.7:
                    p_right=0.1
                else:
                    p_right=0.7
                if choice==1:
                    f = np.random.choice([0,1], p=[1-p_right,p_right])
                else:
                    f = np.random.choice([0,1], p=[1-p_left,p_left])

                if (1*(f*(l==0))) == 1:
                    q = (1 - alpha) * q + choice * beta_ses
                if (f * l) == 1:
                    qlaser = (1 - alphalaser) * qlaser + choice * laser_ses
                qstay = (1 - alphastay) * qstay + choice * stay_ses

                rewards.append((1*(f*(l==0))))
                lasers.append(f*l)
                outcome.append(f)
                # Store variables

            if NT_all[ms_i]!=0:
                c_sim[sub_idx[ms_i], sess_idx[ms_i], :int(NT_all[ms_i])]= predicted_choices
                l_sim[sub_idx[ms_i], sess_idx[ms_i], :int(NT_all[ms_i])] = lasers
                r_sim[sub_idx[ms_i], sess_idx[ms_i], :int(NT_all[ms_i])] = rewards
                ses_data = pd.DataFrame()
                ses_data['choices'] = predicted_choices
                ses_data['water'] = rewards
                ses_data['laser'] = lasers
                ses_data['tb'] = tb[sub_idx[ms_i], sess_idx[ms_i],:int(NT_all[ms_i])]
                ses_data['laser_block'] = b[sub_idx[ms_i], sess_idx[ms_i],:int(NT_all[ms_i])]
                ses_data['probabilityLeft'] = p[sub_idx[ms_i], sess_idx[ms_i],:int(NT_all[ms_i])]
                ses_data['outcome'] = outcome
                ses_data['mouse'] = sub_idx[ms_i]
                ses_data['ses'] = sess_idx[ms_i]
                sim_data = pd.concat([sim_data,ses_data])

    return c_sim, l_sim, r_sim, sim_data
def reinforce_model_w_2stay(standata,saved_params=None, fit=None, csv=True):
    if saved_params is not None:
        r =  standata['r']
        c =  standata['c']
        l =  standata['l']
        b =  standata['b']
        tb = standata['tb']
        sub_idx =  standata['sub_idx']-1
        sess_idx = standata['sess_idx']-1
        NSxNSESS = standata['NSxNSESS']
        NT_all = standata['NT_all']
    else:
        r =  fit.data['r']
        c =  fit.data['c']
        l =  fit.data['l']
        sub_idx =  fit.data['sub_idx']-1
        sess_idx = fit.data['sess_idx']-1
    data = pd.DataFrame()

    for ms_i in np.arange(NSxNSESS):
        if NT_all[ms_i]>0:
            if saved_params is not None:
                if ms_i==0:
                    print('Using saved parameters')
                if csv==True:
                    stay_ses = float(saved_params.loc[saved_params['name']=='stay_ses['+str(ms_i+1)+']', 'Mean'])
                    beta_ses = float(saved_params.loc[saved_params['name']=='beta_ses['+str(ms_i+1)+']', 'Mean'])
                    side_ses = float(saved_params.loc[saved_params['name']=='sides['+str(ms_i+1)+']', 'Mean'])
                    laser_ses = float(saved_params.loc[saved_params['name']=='laser_ses['+str(ms_i+1)+']', 'Mean'])
                    alpha = float(saved_params.loc[saved_params['name']=='alpha_ses['+str(ms_i+1)+']', 'Mean'])
                    alphalaser = float(saved_params.loc[saved_params['name']=='alphalaser_ses['+str(ms_i+1)+']', 'Mean'])
                    alphastay= float(saved_params.loc[saved_params['name']=='alphastay_ses['+str(ms_i+1)+']', 'Mean'])
                    alphalaserstay= float(saved_params.loc[saved_params['name']=='alphalaserstay_ses['+str(ms_i+1)+']', 'Mean'])
                    laserstay_ses = float(saved_params.loc[saved_params['name']=='laserstay_ses['+str(ms_i+1)+']', 'Mean'])

                else:
                    Print('Non csv version not avaialble atm')
            q = 0
            qlaser = 0
            qstay = 0
            predicted_choices=[]
            Q=[]
            Qlaser=[]
            Qstay=[]
            choices=[]
            for t in np.arange(NT_all[ms_i]):
                t = int(t)
                choices.append(c[sub_idx[ms_i], sess_idx[ms_i], t])
                predicted_choice = inv_logit(side_ses + (q + qlaser + qstay))
                choice = 2 * c[sub_idx[ms_i], sess_idx[ms_i], t] -1
                if r[sub_idx[ms_i], sess_idx[ms_i], t] == 1:
                    q = (1 - alpha) * q + choice * beta_ses
                if l[sub_idx[ms_i], sess_idx[ms_i], t] == 1:
                    qlaser = (1 - alphalaser) * qlaser + choice * laser_ses
                if b[sub_idx[ms_i], sess_idx[ms_i], t] == 0:
                    qstay = (1 - alphastay) * qstay + choice * stay_ses
                if b[sub_idx[ms_i], sess_idx[ms_i], t] == 1:
                    qstay = (1 - alphalaserstay) * qstay + choice * laserstay_ses
                # Store variables
                predicted_choices.append(predicted_choice)
                Q.append(q)
                Qlaser.append(qlaser)
                Qstay.append(qstay)
            acc = []
            for i in np.arange(len(choices)):
                acc.append(1*(choices[i]==(1*(predicted_choices[i]>0.5))))
            print(np.mean(acc))

            if NT_all[ms_i]!=0:
                ses_data = pd.DataFrame()
                ses_data['predicted_choice'] = predicted_choices
                ses_data['Qwater'] = Q
                ses_data['laser_block'] = b[sub_idx[ms_i], sess_idx[ms_i],:int(NT_all[ms_i])]
                ses_data['trial_block'] = tb[sub_idx[ms_i], sess_idx[ms_i],:int(NT_all[ms_i])]
                ses_data['Qlaser'] = Qlaser
                ses_data['Qstay'] = Qstay
                ses_data['deltaQ'] = ses_data['Qwater'] + ses_data['Qlaser'] + ses_data['Qstay']
                ses_data['choices'] = choices
                ses_data['mouse'] = sub_idx[ms_i]
                ses_data['ses'] = sess_idx[ms_i]
                ses_data['acc'] = np.mean(acc)
                data = pd.concat([data,ses_data])
    return data
def simulate_reinforce_model_w_2stay(standata_recovery,saved_params=None, fit=None, csv=True):
    b =  standata_recovery['b']
    p =  standata_recovery['p']
    tb = standata_recovery['tb']
    NS =standata_recovery['NS']
    NSESS=standata_recovery['NSESS']
    NT =standata_recovery['NT']
    NT_all =standata_recovery['NT_all']
    r_sim = np.zeros([NS,NSESS,NT])
    c_sim = np.zeros([NS,NSESS,NT])
    l_sim = np.zeros([NS,NSESS,NT])
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
                    stay_ses = float(saved_params.loc[saved_params['name']=='stay_ses['+str(ms_i+1)+']', 'Mean'])
                    laserstay_ses = float(saved_params.loc[saved_params['name']=='laserstay_ses['+str(ms_i+1)+']', 'Mean'])
                    beta_ses = float(saved_params.loc[saved_params['name']=='beta_ses['+str(ms_i+1)+']', 'Mean'])
                    side_ses = float(saved_params.loc[saved_params['name']=='sides['+str(ms_i+1)+']', 'Mean'])
                    laser_ses = float(saved_params.loc[saved_params['name']=='laser_ses['+str(ms_i+1)+']', 'Mean'])
                    alpha = float(saved_params.loc[saved_params['name']=='alpha_ses['+str(ms_i+1)+']', 'Mean'])
                    alphalaser = float(saved_params.loc[saved_params['name']=='alphalaser_ses['+str(ms_i+1)+']', 'Mean'])
                    alphastay= float(saved_params.loc[saved_params['name']=='alphastay_ses['+str(ms_i+1)+']', 'Mean'])
                    alphalaserstay= float(saved_params.loc[saved_params['name']=='alphalaserstay_ses['+str(ms_i+1)+']', 'Mean'])

                else:
                    Print('Non csv version not avaialble atm')
            q = 0
            qlaser = 0
            qstay = 0
            predicted_choices=[]
            rewards = []
            lasers = []
            outcome=[]

            for t in np.arange(NT_all[ms_i]):
                t = int(t)
                p_choice = inv_logit(side_ses + (q + qlaser + qstay))
                choice =  np.random.choice(2,p= [1-p_choice,p_choice])
                predicted_choices.append(choice)
                choice = 2 * choice - 1
                assert np.isin(p[sub_idx[ms_i], sess_idx[ms_i],t], [0.7,0.1])
                p_left= p[sub_idx[ms_i], sess_idx[ms_i],t]
                l = b[sub_idx[ms_i], sess_idx[ms_i],t]
                if p_left==0.7:
                    p_right=0.1
                else:
                    p_right=0.7
                if choice==1:
                    f = np.random.choice([0,1], p=[1-p_right,p_right])
                else:
                    f = np.random.choice([0,1], p=[1-p_left,p_left])

                if (1*(f*(l==0))) == 1:
                    q = (1 - alpha) * q + choice * beta_ses
                if (f * l) == 1:
                    qlaser = (1 - alphalaser) * qlaser + choice * laser_ses
                if l == 1:
                    qstay = (1 - alphalaserstay) * qstay + choice * laserstay_ses
                if l == 0:
                    qstay = (1 - alphastay) * qstay + choice * stay_ses

                rewards.append((1*(f*(l==0))))
                lasers.append(f*l)
                outcome.append(f)
                # Store variables

            if NT_all[ms_i]!=0:
                c_sim[sub_idx[ms_i], sess_idx[ms_i], :int(NT_all[ms_i])]= predicted_choices
                l_sim[sub_idx[ms_i], sess_idx[ms_i], :int(NT_all[ms_i])] = lasers
                r_sim[sub_idx[ms_i], sess_idx[ms_i], :int(NT_all[ms_i])] = rewards
                ses_data = pd.DataFrame()
                ses_data['choices'] = predicted_choices
                ses_data['water'] = rewards
                ses_data['laser'] = lasers
                ses_data['tb'] = tb[sub_idx[ms_i], sess_idx[ms_i],:int(NT_all[ms_i])]
                ses_data['laser_block'] = b[sub_idx[ms_i], sess_idx[ms_i],:int(NT_all[ms_i])]
                ses_data['probabilityLeft'] = p[sub_idx[ms_i], sess_idx[ms_i],:int(NT_all[ms_i])]
                ses_data['outcome'] = outcome
                ses_data['mouse'] = sub_idx[ms_i]
                ses_data['ses'] = sess_idx[ms_i]
                sim_data = pd.concat([sim_data,ses_data])

    return c_sim, l_sim, r_sim, sim_data
def q_learning_model_w_2learning(standata,saved_params=None, fit=None, csv=True):
    if saved_params is not None:
        r =  standata['r']
        c =  standata['c']
        l =  standata['l']
        tb =  standata['tb']
        sub_idx =  standata['sub_idx']-1
        sess_idx = standata['sess_idx']-1
        NSxNSESS = standata['NSxNSESS']
        NT_all = standata['NT_all']
    else:
        r =  fit.data['r']
        c =  fit.data['c']
        l =  fit.data['l']
        sub_idx =  fit.data['sub_idx']-1
        sess_idx = fit.data['sess_idx']-1
    data = pd.DataFrame()

    for ms_i in np.arange(NSxNSESS):
        if NT_all[ms_i]>0:
            if saved_params is not None:
                if ms_i==0:
                    print('Using saved parameters')
                if csv==True:
                    beta_mouse = float(saved_params.loc[saved_params['name']=='beta_ses['+str(ms_i+1)+']', 'Mean'])
                    stay_mouse = float(saved_params.loc[saved_params['name']=='stay_ses['+str(ms_i+1)+']', 'Mean'])
                    side_mouse = float(saved_params.loc[saved_params['name']=='sides['+str(ms_i+1)+']', 'Mean'])
                    laser_mouse = float(saved_params.loc[saved_params['name']=='laser_ses['+str(ms_i+1)+']', 'Mean'])
                    alpha = float(saved_params.loc[saved_params['name']=='alpha_ses['+str(ms_i+1)+']', 'Mean'])
                    alphalaser = float(saved_params.loc[saved_params['name']=='alphalaser_ses['+str(ms_i+1)+']', 'Mean'])
                    alphastay = float(saved_params.loc[saved_params['name']=='alphastay_ses['+str(ms_i+1)+']', 'Mean'])
                    balpha = float(saved_params.loc[saved_params['name']=='balpha_ses['+str(ms_i+1)+']', 'Mean'])
                    balphalaser = float(saved_params.loc[saved_params['name']=='balphalaser_ses['+str(ms_i+1)+']', 'Mean'])
                    balphastay = float(saved_params.loc[saved_params['name']=='balphastay_ses['+str(ms_i+1)+']', 'Mean'])
                else:
                    print('Error: Only csv version available at the moment!')
            else:
                if ms_i==0:
                    print('No saved parameters')
                beta_mouse = fit.extract()['beta_ses'][:,int(ms_i)].mean()
                stay_mouse = fit.extract()['stay_ses'][:,int(ms_i)].mean()
                side_mouse = fit.extract()['sides'][:,int(ms_i)].mean()
                laser_mouse = fit.extract()['laser_ses'][:,int(ms_i)].mean()
                alpha = phi_approx(fit.extract()['alpha_ses'][:,int(ms_i)].mean())
                alphalaser =phi_approx(fit.extract()['alphalaser_ses'][:,int(ms_i)].mean())
                alphastay = phi_approx(fit.extract()['alphastay_ses'][:,int(ms_i)].mean())
            q = np.zeros(2)
            qstay = np.zeros(2)
            qlaser = np.zeros(2)
            predicted_choices=[]
            QL=[]
            QR=[]
            QLstay=[]
            QRstay=[]
            QLlaser=[]
            QRlaser=[]
            choices=[]
            QLgeneral=[]
            QRgeneral=[]
            for t in np.arange(NT_all[ms_i]):
                t = int(t)
                predicted_choice = inv_logit(side_mouse
                    + beta_mouse  * (q[1] - q[0])
                    + stay_mouse * (qstay[1] - qstay[0])
                    + laser_mouse * (qlaser[1] - qlaser[0]))
                choice = c[sub_idx[ms_i], sess_idx[ms_i], t]
                if tb[sub_idx[ms_i], sess_idx[ms_i],t]>10:
                    q[choice] = (1-alpha) * q[choice] + alpha * r[sub_idx[ms_i], sess_idx[ms_i],t]
                    qlaser[choice] = (1-alphalaser) * qlaser[choice] + alphalaser * l[sub_idx[ms_i], sess_idx[ms_i],t]
                    qstay = qstay * (1 - alphastay)
                    qstay[choice] = qstay[choice] + alphastay
                if tb[sub_idx[ms_i], sess_idx[ms_i],t]<11:
                    q[choice] = (1-balpha) * q[choice] + balpha * r[sub_idx[ms_i], sess_idx[ms_i],t]
                    qlaser[choice] = (1-balphalaser) * qlaser[choice] + balphalaser * l[sub_idx[ms_i], sess_idx[ms_i],t]
                    qstay = qstay * (1 - balphastay)
                    qstay[choice] = qstay[choice] + balphastay
                # Store variables
                predicted_choices.append(predicted_choice)
                choices.append(choice)
                QL.append(q[0])
                QR.append(q[1])
                QLstay.append(qstay[0])
                QRstay.append(qstay[1])
                QLlaser.append(qlaser[0])
                QRlaser.append(qlaser[1])
                QLgeneral.append((beta_mouse*q[0])+(laser_mouse*qlaser[0])+(stay_mouse*qstay[0]))
                QRgeneral.append((beta_mouse*q[1])+(laser_mouse*qlaser[1])+(stay_mouse*qstay[1]))
            acc = []
            for i in np.arange(len(choices)):
                acc.append(1*(choices[i]==(1*(predicted_choices[i]>0.5))))
            print(np.mean(acc))
            if NT_all[ms_i]!=0:
                ses_data = pd.DataFrame()
                ses_data['predicted_choice'] = predicted_choices
                ses_data['QL'] = QLgeneral
                ses_data['QR'] = QRgeneral
                ses_data['QLreward'] = np.array(QL) * beta_mouse # Note this QL is the reward  - TODO change name
                ses_data['QRreward'] =  np.array(QR) * beta_mouse
                ses_data['QLstay'] =  np.array(QLstay) * stay_mouse
                ses_data['QRstay'] =  np.array(QRstay) * stay_mouse
                ses_data['QLlaser'] =  np.array(QLlaser) * laser_mouse
                ses_data['QRlaser'] =  np.array(QRlaser) * laser_mouse
                ses_data['deltaQ'] = ses_data['QR'] - ses_data['QL']
                ses_data['choices'] = choices
                ses_data['mouse'] = sub_idx[ms_i]
                ses_data['ses'] = sess_idx[ms_i]
                ses_data['acc'] = np.mean(acc)
                data = pd.concat([data,ses_data])
    return make_deltas(data)
def simulate_q_learning_w_2learning(standata_recovery,saved_params=None, fit=None, csv=True):
    b =  standata_recovery['b']
    p =  standata_recovery['p']
    tb =  standata_recovery['tb']
    NS =standata_recovery['NS']
    NSESS=standata_recovery['NSESS']
    NT =standata_recovery['NT']
    NT_all =standata_recovery['NT_all']
    r_sim = np.zeros([NS,NSESS,NT])
    c_sim = np.zeros([NS,NSESS,NT])
    l_sim = np.zeros([NS,NSESS,NT])
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
                    beta_mouse = float(saved_params.loc[saved_params['name']=='beta_ses['+str(ms_i+1)+']', 'Mean'])
                    stay_mouse = float(saved_params.loc[saved_params['name']=='stay_ses['+str(ms_i+1)+']', 'Mean'])
                    side_mouse = float(saved_params.loc[saved_params['name']=='sides['+str(ms_i+1)+']', 'Mean'])
                    laser_mouse = float(saved_params.loc[saved_params['name']=='laser_ses['+str(ms_i+1)+']', 'Mean'])
                    alpha = float(saved_params.loc[saved_params['name']=='alpha_ses['+str(ms_i+1)+']', 'Mean'])
                    alphalaser = float(saved_params.loc[saved_params['name']=='alphalaser_ses['+str(ms_i+1)+']', 'Mean'])
                    alphastay = float(saved_params.loc[saved_params['name']=='alphastay_ses['+str(ms_i+1)+']', 'Mean'])
                    balpha = float(saved_params.loc[saved_params['name']=='balpha_ses['+str(ms_i+1)+']', 'Mean'])
                    balphalaser = float(saved_params.loc[saved_params['name']=='balphalaser_ses['+str(ms_i+1)+']', 'Mean'])
                    balphastay = float(saved_params.loc[saved_params['name']=='balphastay_ses['+str(ms_i+1)+']', 'Mean'])
                else:
                    print('Error: Only csv version available at the moment!')
            else:
                if ms_i==0:
                    print('No saved parameters')
                beta_mouse = fit.extract()['beta_ses'][:,int(ms_i)].mean()
                stay_mouse = fit.extract()['stay_ses'][:,int(ms_i)].mean()
                side_mouse = fit.extract()['sides'][:,int(ms_i)].mean()
                laser_mouse = fit.extract()['laser_ses'][:,int(ms_i)].mean()
                alpha = phi_approx(fit.extract()['alpha_ses'][:,int(ms_i)].mean())
                alphalaser =phi_approx(fit.extract()['alphalaser_ses'][:,int(ms_i)].mean())
                alphastay = phi_approx(fit.extract()['alphastay_ses'][:,int(ms_i)].mean())
            q = np.zeros(2)
            qstay = np.zeros(2)
            qlaser = np.zeros(2)
            predicted_choices=[]
            rewards = []
            lasers = []
            outcome=[]

            for t in np.arange(NT_all[ms_i]):
                t = int(t)
                p_choice = inv_logit(side_mouse
                    + beta_mouse  * (q[1] - q[0])
                    + stay_mouse * (qstay[1] - qstay[0])
                    + laser_mouse * (qlaser[1] - qlaser[0]))
                choice =  np.random.choice(2,p= [1-p_choice,p_choice])
                assert np.isin(p[sub_idx[ms_i], sess_idx[ms_i],t], [0.7,0.1])
                p_left= p[sub_idx[ms_i], sess_idx[ms_i],t]
                l = b[sub_idx[ms_i], sess_idx[ms_i],t]
                if p_left==0.7:
                    p_right=0.1
                else:
                    p_right=0.7
                if choice==1:
                    f = np.random.choice([0,1], p=[1-p_right,p_right])
                if choice==0:
                    f = np.random.choice([0,1], p=[1-p_left,p_left])
                if tb[sub_idx[ms_i], sess_idx[ms_i],t]>10:
                    q[choice] = (1-alpha) * q[choice] + alpha * (1*(f*(l==0)))
                    qlaser[choice] = (1-alphalaser) * qlaser[choice] + alphalaser * f * l
                    qstay = qstay * (1 - alphastay)
                    qstay[choice] = qstay[choice] + alphastay
                if tb[sub_idx[ms_i], sess_idx[ms_i],t]<11:
                    q[choice] = (1-balpha) * q[choice] + balpha * (1*(f*(l==0)))
                    qlaser[choice] = (1-balphalaser) * qlaser[choice] + balphalaser * f * l
                    qstay = qstay * (1 - balphastay)
                    qstay[choice] = qstay[choice] + balphastay
                rewards.append((1*(f*(l==0))))
                lasers.append(f*l)
                outcome.append(f)
                # Store variables
                predicted_choices.append(choice)

            if NT_all[ms_i]!=0:
                c_sim[sub_idx[ms_i], sess_idx[ms_i], :int(NT_all[ms_i])]= predicted_choices
                l_sim[sub_idx[ms_i], sess_idx[ms_i], :int(NT_all[ms_i])] = lasers
                r_sim[sub_idx[ms_i], sess_idx[ms_i], :int(NT_all[ms_i])] = rewards
                ses_data = pd.DataFrame()
                ses_data['choices'] = predicted_choices
                ses_data['water'] = rewards
                ses_data['laser'] = lasers
                ses_data['tb'] = tb[sub_idx[ms_i], sess_idx[ms_i],:int(NT_all[ms_i])]
                ses_data['probabilityLeft'] = p[sub_idx[ms_i], sess_idx[ms_i],:int(NT_all[ms_i])]
                ses_data['laser_block'] = b[sub_idx[ms_i], sess_idx[ms_i],:int(NT_all[ms_i])]
                ses_data['probabilityLeft'] = p[sub_idx[ms_i], sess_idx[ms_i],:int(NT_all[ms_i])]
                ses_data['outcome'] = outcome
                ses_data['mouse'] = sub_idx[ms_i]
                ses_data['ses'] = sess_idx[ms_i]
                sim_data = pd.concat([sim_data,ses_data])

    return c_sim, l_sim, r_sim, sim_data
def q_learning_model_Lrpe(standata,saved_params=None, fit=None, csv=True):
    if saved_params is not None:
        r =  standata['r']
        c =  standata['c']
        l =  standata['l']
        sub_idx =  standata['sub_idx']-1
        sess_idx = standata['sess_idx']-1
        NSxNSESS = standata['NSxNSESS']
        NT_all = standata['NT_all']
    else:
        r =  fit.data['r']
        c =  fit.data['c']
        l =  fit.data['l']
        sub_idx =  standata['sub_idx']-1
        sess_idx = standata['sess_idx']-1
        NSxNSESS = standata['NSxNSESS']
    data = pd.DataFrame()
    for ms_i in np.arange(NSxNSESS):
        if NT_all[ms_i]>0:
            if saved_params is not None:
                if ms_i==0:
                    print('Using saved parameters')
                if csv==True:
                    beta_mouse = float(saved_params.loc[saved_params['name']=='beta_ses['+str(ms_i+1)+']', 'Mean'])
                    stay_mouse = float(saved_params.loc[saved_params['name']=='stay_ses['+str(ms_i+1)+']', 'Mean'])
                    side_mouse = float(saved_params.loc[saved_params['name']=='sides['+str(ms_i+1)+']', 'Mean'])
                    laser_mouse = float(saved_params.loc[saved_params['name']=='laser_ses['+str(ms_i+1)+']', 'Mean'])
                    alpha = float(phi_approx(saved_params.loc[saved_params['name']=='alpha_ses['+str(ms_i+1)+']', 'Mean']/np.sqrt(2)))
                    alphalaser = float(phi_approx(saved_params.loc[saved_params['name']=='alphalaser_ses['+str(ms_i+1)+']', 'Mean']/np.sqrt(2)))
                    alphastay = float(phi_approx(saved_params.loc[saved_params['name']=='alphastay_ses['+str(ms_i+1)+']', 'Mean']/np.sqrt(2)))
                else:
                    beta_mouse =  saved_params['beta_ses['+str(ms_i+1)+']'].mean()
                    stay_mouse = saved_params['stay_ses['+str(ms_i+1)+']'].mean()
                    side_mouse = saved_params['sides['+str(ms_i+1)+']'].mean()
                    laser_mouse = saved_params['laser_ses['+str(ms_i+1)+']'].mean()
                    alpha = phi_approx(saved_params['alpha_ses['+str(ms_i+1)+']'].mean())
                    alphalaser =phi_approx(saved_params['alphalaser_ses['+str(ms_i+1)+']'].mean())
                    alphastay = phi_approx(saved_params['alphastay_ses['+str(ms_i+1)+']'].mean())
            else:
                if ms_i==0:
                    print('No saved parameters')
                beta_mouse = fit.extract()['beta_ses'][:,int(ms_i)].mean()
                stay_mouse = fit.extract()['stay_ses'][:,int(ms_i)].mean()
                side_mouse = fit.extract()['sides'][:,int(ms_i)].mean()
                laser_mouse = fit.extract()['laser_ses'][:,int(ms_i)].mean()
                alpha = phi_approx(fit.extract()['alpha_ses'][:,int(ms_i)].mean())
                alphalaser =phi_approx(fit.extract()['alphalaser_ses'][:,int(ms_i)].mean())
                alphastay = phi_approx(fit.extract()['alphastay_ses'][:,int(ms_i)].mean())
            q = np.zeros(2)
            qstay = np.zeros(2)
            qlaser = np.zeros(2)
            predicted_choices=[]
            QL=[]
            QR=[]
            QLstay=[]
            QRstay=[]
            QLlaser=[]
            QRlaser=[]
            choices=[]
            QLgeneral=[]
            QRgeneral=[]
            for t in np.arange(NT_all[ms_i]):
                t = int(t)
                predicted_choice = inv_logit(side_mouse
                    + beta_mouse  * (q[1] - q[0])
                    + stay_mouse * (qstay[1] - qstay[0])
                    + laser_mouse * (qlaser[1] - qlaser[0]))

                choice = c[sub_idx[ms_i], sess_idx[ms_i], t]
                q[choice] = (1-alpha) * q[choice] + alpha * r[sub_idx[ms_i], sess_idx[ms_i],t]
                qlaser[choice] = qlaser[choice] + alphalaser * l[sub_idx[ms_i], sess_idx[ms_i],t]
                qstay = qstay * (1 - alphastay)
                qstay[choice] = qstay[choice] + alphastay
                # Store variables
                predicted_choices.append(predicted_choice)
                choices.append(choice)
                QL.append(q[0])
                QR.append(q[1])
                QLstay.append(qstay[0])
                QRstay.append(qstay[1])
                QLlaser.append(qlaser[0])
                QRlaser.append(qlaser[1])
                QLgeneral.append((beta_mouse*q[0])+(laser_mouse*qlaser[0])+(stay_mouse*qstay[0]))
                QRgeneral.append((beta_mouse*q[1])+(laser_mouse*qlaser[1])+(stay_mouse*qstay[1]))
            acc = []
            for i in np.arange(len(choices)):
                acc.append(1*(choices[i]==(1*(predicted_choices[i]>0.5))))
            print(np.mean(acc))

            if NT_all[ms_i]!=0:
                ses_data = pd.DataFrame()
                ses_data['predicted_choice'] = predicted_choices
                ses_data['QL'] = QLgeneral
                ses_data['QR'] = QRgeneral
                ses_data['QLreward'] = np.array(QL) * beta_mouse # Note this QL is the reward  - TODO change name
                ses_data['QRreward'] =  np.array(QR) * beta_mouse
                ses_data['QLstay'] =  np.array(QLstay) * stay_mouse
                ses_data['QRstay'] =  np.array(QRstay) * stay_mouse
                ses_data['QLlaser'] =  np.array(QLlaser) * laser_mouse
                ses_data['QRlaser'] =  np.array(QRlaser) * laser_mouse
                ses_data['deltaQ'] = ses_data['QR'] - ses_data['QL']
                ses_data['choices'] = choices
                ses_data['mouse'] = sub_idx[ms_i]
                ses_data['ses'] = sess_idx[ms_i]
                ses_data['acc'] = np.mean(acc)
                data = pd.concat([data,ses_data])
    return make_deltas(data)
def q_learning_model_Lrpeconditional(standata,saved_params=None, fit=None, csv=True):
    if saved_params is not None:
        r =  standata['r']
        c =  standata['c']
        l =  standata['l']
        sub_idx =  standata['sub_idx']-1
        sess_idx = standata['sess_idx']-1
        NSxNSESS = standata['NSxNSESS']
        NT_all = standata['NT_all']
    else:
        r =  fit.data['r']
        c =  fit.data['c']
        l =  fit.data['l']
        sub_idx =  standata['sub_idx']-1
        sess_idx = standata['sess_idx']-1
        NSxNSESS = standata['NSxNSESS']
    data = pd.DataFrame()
    for ms_i in np.arange(NSxNSESS):
        if NT_all[ms_i]>0:
            if saved_params is not None:
                if ms_i==0:
                    print('Using saved parameters')
                if csv==True:
                    beta_mouse = float(saved_params.loc[saved_params['name']=='beta_ses['+str(ms_i+1)+']', 'Mean'])
                    stay_mouse = float(saved_params.loc[saved_params['name']=='stay_ses['+str(ms_i+1)+']', 'Mean'])
                    side_mouse = float(saved_params.loc[saved_params['name']=='sides['+str(ms_i+1)+']', 'Mean'])
                    laser_mouse = float(saved_params.loc[saved_params['name']=='laser_ses['+str(ms_i+1)+']', 'Mean'])
                    alpha = float(phi_approx(saved_params.loc[saved_params['name']=='alpha_ses['+str(ms_i+1)+']', 'Mean']/np.sqrt(2)))
                    alphalaser = float(phi_approx(saved_params.loc[saved_params['name']=='alphalaser_ses['+str(ms_i+1)+']', 'Mean']/np.sqrt(2)))
                    alphastay = float(phi_approx(saved_params.loc[saved_params['name']=='alphastay_ses['+str(ms_i+1)+']', 'Mean']/np.sqrt(2)))
                else:
                    beta_mouse =  saved_params['beta_ses['+str(ms_i+1)+']'].mean()
                    stay_mouse = saved_params['stay_ses['+str(ms_i+1)+']'].mean()
                    side_mouse = saved_params['sides['+str(ms_i+1)+']'].mean()
                    laser_mouse = saved_params['laser_ses['+str(ms_i+1)+']'].mean()
                    alpha = phi_approx(saved_params['alpha_ses['+str(ms_i+1)+']'].mean())
                    alphalaser =phi_approx(saved_params['alphalaser_ses['+str(ms_i+1)+']'].mean())
                    alphastay = phi_approx(saved_params['alphastay_ses['+str(ms_i+1)+']'].mean())
            else:
                if ms_i==0:
                    print('No saved parameters')
                beta_mouse = fit.extract()['beta_ses'][:,int(ms_i)].mean()
                stay_mouse = fit.extract()['stay_ses'][:,int(ms_i)].mean()
                side_mouse = fit.extract()['sides'][:,int(ms_i)].mean()
                laser_mouse = fit.extract()['laser_ses'][:,int(ms_i)].mean()
                alpha = phi_approx(fit.extract()['alpha_ses'][:,int(ms_i)].mean())
                alphalaser =phi_approx(fit.extract()['alphalaser_ses'][:,int(ms_i)].mean())
                alphastay = phi_approx(fit.extract()['alphastay_ses'][:,int(ms_i)].mean())
            q = np.zeros(2)
            qstay = np.zeros(2)
            qlaser = np.zeros(2)
            predicted_choices=[]
            QL=[]
            QR=[]
            QLstay=[]
            QRstay=[]
            QLlaser=[]
            QRlaser=[]
            choices=[]
            QLgeneral=[]
            QRgeneral=[]
            for t in np.arange(NT_all[ms_i]):
                t = int(t)
                predicted_choice = inv_logit(side_mouse
                    + beta_mouse  * (q[1] - q[0])
                    + stay_mouse * (qstay[1] - qstay[0])
                    + laser_mouse * (qlaser[1] - qlaser[0]))

                choice = c[sub_idx[ms_i], sess_idx[ms_i], t]
                q[choice] = (1-alpha) * q[choice] + alpha * r[sub_idx[ms_i], sess_idx[ms_i],t]
                if l[sub_idx[ms_i], sess_idx[ms_i],t]==1:
                    qlaser[choice] = qlaser[choice] + alphalaser * l[sub_idx[ms_i], sess_idx[ms_i],t]
                else:
                    qlaser[choice] = qlaser[choice] + alphalaser * (l[sub_idx[ms_i], sess_idx[ms_i],t] - qlaser[choice])
                qstay = qstay * (1 - alphastay)
                qstay[choice] = qstay[choice] + alphastay
                # Store variables
                predicted_choices.append(predicted_choice)
                choices.append(choice)
                QL.append(q[0])
                QR.append(q[1])
                QLstay.append(qstay[0])
                QRstay.append(qstay[1])
                QLlaser.append(qlaser[0])
                QRlaser.append(qlaser[1])
                QLgeneral.append((beta_mouse*q[0])+(laser_mouse*qlaser[0])+(stay_mouse*qstay[0]))
                QRgeneral.append((beta_mouse*q[1])+(laser_mouse*qlaser[1])+(stay_mouse*qstay[1]))
            acc = []
            for i in np.arange(len(choices)):
                acc.append(1*(choices[i]==(1*(predicted_choices[i]>0.5))))
            print(np.mean(acc))

            if NT_all[ms_i]!=0:
                ses_data = pd.DataFrame()
                ses_data['predicted_choice'] = predicted_choices
                ses_data['QL'] = QLgeneral
                ses_data['QR'] = QRgeneral
                ses_data['QLreward'] = np.array(QL) * beta_mouse # Note this QL is the reward  - TODO change name
                ses_data['QRreward'] =  np.array(QR) * beta_mouse
                ses_data['QLstay'] =  np.array(QLstay) * stay_mouse
                ses_data['QRstay'] =  np.array(QRstay) * stay_mouse
                ses_data['QLlaser'] =  np.array(QLlaser) * laser_mouse
                ses_data['QRlaser'] =  np.array(QRlaser) * laser_mouse
                ses_data['deltaQ'] = ses_data['QR'] - ses_data['QL']
                ses_data['choices'] = choices
                ses_data['mouse'] = sub_idx[ms_i]
                ses_data['ses'] = sess_idx[ms_i]
                ses_data['acc'] = np.mean(acc)
                data = pd.concat([data,ses_data])
    return make_deltas(data)
def q_learning_model_2alphas(standata,saved_params=None, fit=None, csv=True):
    if saved_params is not None:
        r =  standata['r']
        c =  standata['c']
        l =  standata['l']
        b =  standata['b']
        sub_idx =  standata['sub_idx']-1
        sess_idx = standata['sess_idx']-1
        NSxNSESS = standata['NSxNSESS']
        NT_all = standata['NT_all']
    else:
        r =  fit.data['r']
        c =  fit.data['c']
        l =  fit.data['l']
        b =  fit.data['b']
        sub_idx =  fit.data['sub_idx']-1
        sess_idx = fit.data['sess_idx']-1
        NSxNSESS = fit.data['NSxNSESS']
    data = pd.DataFrame()

    for ms_i in np.arange(NSxNSESS):
        if NT_all[ms_i]>0:
            if saved_params is not None:
                if ms_i==0:
                    print('Using saved parameters')
                if csv==True:
                    beta_mouse = float(saved_params.loc[saved_params['name']=='beta_ses['+str(ms_i+1)+']', 'Mean'])
                    stay_mouse = float(saved_params.loc[saved_params['name']=='stay_ses['+str(ms_i+1)+']', 'Mean'])
                    laserstay_mouse = float(saved_params.loc[saved_params['name']=='laserstay_ses['+str(ms_i+1)+']', 'Mean'])
                    side_mouse = float(saved_params.loc[saved_params['name']=='sides['+str(ms_i+1)+']', 'Mean'])
                    laser_mouse = float(saved_params.loc[saved_params['name']=='laser_ses['+str(ms_i+1)+']', 'Mean'])
                    alpha = float(phi_approx(saved_params.loc[saved_params['name']=='alpha_ses['+str(ms_i+1)+']', 'Mean']))
                    alphalaser = float(phi_approx(saved_params.loc[saved_params['name']=='alphalaser_ses['+str(ms_i+1)+']', 'Mean']/np.sqrt(2)))
                    alphastay = float(phi_approx(saved_params.loc[saved_params['name']=='alphastay_ses['+str(ms_i+1)+']', 'Mean']/np.sqrt(2)))
                    alphalaserstay = float(phi_approx(saved_params.loc[saved_params['name']=='alphalaserstay_ses['+str(ms_i+1)+']', 'Mean']/np.sqrt(2)))
                else:
                    beta_mouse =  saved_params['beta_ses['+str(ms_i+1)+']'].mean()
                    stay_mouse = saved_params['stay_ses['+str(ms_i+1)+']'].mean()
                    laserstay_mouse = saved_params['laserstay_ses['+str(ms_i+1)+']'].mean()
                    side_mouse = saved_params['sides['+str(ms_i+1)+']'].mean()
                    laser_mouse = saved_params['laser_ses['+str(ms_i+1)+']'].mean()
                    alpha = phi_approx(saved_params['alpha_ses['+str(ms_i+1)+']'].mean())
                    alphalaser =phi_approx(saved_params['alphalaser_ses['+str(ms_i+1)+']'].mean())
                    alphastay = phi_approx(saved_params['alphastay_ses['+str(ms_i+1)+']'].mean())
                    alphalaserstay = phi_approx(saved_params['alphalaserstay_ses['+str(ms_i+1)+']'].mean())
            else:
                if ms_i==0:
                    print('No saved parameters')
                beta_mouse = fit.extract()['beta_ses'][:,int(ms_i)].mean()
                stay_mouse = fit.extract()['stay_ses'][:,int(ms_i)].mean()
                laserstay_mouse = fit.extract()['laserstay_ses'][:,int(ms_i)].mean()
                side_mouse = fit.extract()['sides'][:,int(ms_i)].mean()
                laser_mouse = fit.extract()['laser_ses'][:,int(ms_i)].mean()
                alpha = phi_approx(fit.extract()['alpha_ses'][:,int(ms_i)].mean())
                alphalaser =phi_approx(fit.extract()['alphalaser_ses'][:,int(ms_i)].mean())
                alphastay = phi_approx(fit.extract()['alphastay_ses'][:,int(ms_i)].mean())
                alphalaserstay = phi_approx(fit.extract()['alphalaserstay_ses'][:,int(ms_i)].mean())
            q = np.zeros(2)
            qstay = np.zeros(2)
            qlaser = np.zeros(2)
            predicted_choices=[]
            QL=[]
            QR=[]
            QLstay=[]
            QRstay=[]
            QLlaser=[]
            QRlaser=[]
            choices=[]
            QLgeneral=[]
            QRgeneral=[]
            for t in np.arange(NT_all[ms_i]):
                t = int(t)
                if b[sub_idx[ms_i], sess_idx[ms_i], t]==1:
                    predicted_choice = inv_logit(side_mouse
                        + beta_mouse  * (q[1] - q[0])
                        + laserstay_mouse * (qstay[1] - qstay[0])
                        + laser_mouse * (qlaser[1] - qlaser[0]))
                else:
                    predicted_choice = inv_logit(side_mouse
                        + beta_mouse  * (q[1] - q[0])
                        + laserstay_mouse * (qstay[1] - qstay[0])
                        + laser_mouse * (qlaser[1] - qlaser[0]))
                choice = c[sub_idx[ms_i], sess_idx[ms_i], t]
                q[choice] = (1-alpha) * q[choice] + alpha * r[sub_idx[ms_i], sess_idx[ms_i],t]
                qlaser[choice] = (1-alphalaser) * qlaser[choice] + alphalaser * l[sub_idx[ms_i], sess_idx[ms_i],t]
                if b[sub_idx[ms_i], sess_idx[ms_i], t]==1:
                    qstay = qstay * (1 - alphalaserstay)
                else:
                    qstay = qstay * (1 - alphastay)
                if b[sub_idx[ms_i], sess_idx[ms_i], t]==1:
                    qstay[choice] = qstay[choice] + alphalaserstay
                else:
                    qstay[choice] = qstay[choice] + alphastay
                # Store variables
                predicted_choices.append(predicted_choice)
                choices.append(choice)
                QL.append(q[0])
                QR.append(q[1])
                QLstay.append(qstay[0])
                QRstay.append(qstay[1])
                QLlaser.append(qlaser[0])
                QRlaser.append(qlaser[1])
                QLgeneral.append((beta_mouse*q[0])+(laser_mouse*qlaser[0])+(stay_mouse*qstay[0]))
                QRgeneral.append((beta_mouse*q[1])+(laser_mouse*qlaser[1])+(stay_mouse*qstay[1]))



            acc = []
            for i in np.arange(len(choices)):
                acc.append(1*(choices[i]==(1*(predicted_choices[i]>0.5))))
            print(np.mean(acc))

            if NT_all[ms_i]!=0:
                ses_data = pd.DataFrame()
                ses_data['predicted_choice'] = predicted_choices
                ses_data['QL'] = QLgeneral
                ses_data['QR'] = QRgeneral
                ses_data['QLreward'] = np.array(QL) * beta_mouse # Note this QL is the reward  - TODO change name
                ses_data['QRreward'] =  np.array(QR) * beta_mouse
                ses_data['QLstay'] =  np.array(QLstay) * stay_mouse
                ses_data['QRstay'] =  np.array(QRstay) * stay_mouse
                ses_data['QLlaser'] =  np.array(QLlaser) * laser_mouse
                ses_data['QRlaser'] =  np.array(QRlaser) * laser_mouse
                ses_data['choices'] = choices
                ses_data['deltaQ'] = ses_data['QR'] - ses_data['QL']
                ses_data['mouse'] = sub_idx[ms_i]
                ses_data['ses'] = sess_idx[ms_i]
                ses_data['acc'] = np.mean(acc)
                data = pd.concat([data,ses_data])
    return make_deltas(data)
def q_learning_model_Lrpeconditional_w_forgetting(standata,saved_params=None, fit=None, csv=True):
    if saved_params is not None:
        r =  standata['r']
        c =  standata['c']
        l =  standata['l']
        sub_idx =  standata['sub_idx']-1
        sess_idx = standata['sess_idx']-1
        NSxNSESS = standata['NSxNSESS']
        NT_all = standata['NT_all']
    else:
        r =  fit.data['r']
        c =  fit.data['c']
        l =  fit.data['l']
        sub_idx =  fit.data['sub_idx']-1
        sess_idx = fit.data['sess_idx']-1
    data = pd.DataFrame()

    for ms_i in np.arange(NSxNSESS):
        if NT_all[ms_i]>0:
            if saved_params is not None:
                if ms_i==0:
                    print('Using saved parameters')
                if csv==True:
                    beta_mouse = float(saved_params.loc[saved_params['name']=='beta_ses['+str(ms_i+1)+']', 'Mean'])
                    stay_mouse = float(saved_params.loc[saved_params['name']=='stay_ses['+str(ms_i+1)+']', 'Mean'])
                    side_mouse = float(saved_params.loc[saved_params['name']=='sides['+str(ms_i+1)+']', 'Mean'])
                    laser_mouse = float(saved_params.loc[saved_params['name']=='laser_ses['+str(ms_i+1)+']', 'Mean'])
                    alpha = float(phi_approx(saved_params.loc[saved_params['name']=='alpha_ses['+str(ms_i+1)+']', 'Mean']/np.sqrt(2)))
                    alphalaser = float(phi_approx(saved_params.loc[saved_params['name']=='alphalaser_ses['+str(ms_i+1)+']', 'Mean']/np.sqrt(2)))
                    alphastay = float(phi_approx(saved_params.loc[saved_params['name']=='alphastay_ses['+str(ms_i+1)+']', 'Mean']/np.sqrt(2)))
                    alphaforgetting= float(phi_approx(saved_params.loc[saved_params['name']=='alphaforgetting_ses['+str(ms_i+1)+']', 'Mean']/np.sqrt(2)))
                    alphalaserforgetting= float(phi_approx(saved_params.loc[saved_params['name']=='alphalaserforgetting_ses['+str(ms_i+1)+']', 'Mean']/np.sqrt(2)))
                else:
                    beta_mouse =  saved_params['beta_ses['+str(ms_i+1)+']'].mean()
                    stay_mouse = saved_params['stay_ses['+str(ms_i+1)+']'].mean()
                    side_mouse = saved_params['sides['+str(ms_i+1)+']'].mean()
                    laser_mouse = saved_params['laser_ses['+str(ms_i+1)+']'].mean()
                    alpha = phi_approx(saved_params['alpha_ses['+str(ms_i+1)+']'].mean())
                    alphalaser =phi_approx(saved_params['alphalaser_ses['+str(ms_i+1)+']'].mean())
                    alphastay = phi_approx(saved_params['alphastay_ses['+str(ms_i+1)+']'].mean())
                    alphaforgetting= phi_approx(saved_params['alphaforgetting_ses['+str(ms_i+1)+']'].mean())
                    alphalaserforgetting= phi_approx(saved_params['alphalaserforgetting_ses['+str(ms_i+1)+']'].mean())
            else:
                if ms_i==0:
                    print('No saved parameters')
                beta_mouse = fit.extract()['beta_ses'][:,int(ms_i)].mean()
                stay_mouse = fit.extract()['stay_ses'][:,int(ms_i)].mean()
                side_mouse = fit.extract()['sides'][:,int(ms_i)].mean()
                laser_mouse = fit.extract()['laser_ses'][:,int(ms_i)].mean()
                alpha = phi_approx(fit.extract()['alpha_ses'][:,int(ms_i)].mean())
                alphalaser =phi_approx(fit.extract()['alphalaser_ses'][:,int(ms_i)].mean())
                alphastay = phi_approx(fit.extract()['alphastay_ses'][:,int(ms_i)].mean())
                alphaforgetting = phi_approx(fit.extract()['alphaforgetting_ses'][:,int(ms_i)].mean())
                alphalaserforgetting = phi_approx(fit.extract()['alphalaserforgetting_ses'][:,int(ms_i)].mean())
            q = np.zeros(2)
            qstay = np.zeros(2)
            qlaser = np.zeros(2)
            predicted_choices=[]
            QL=[]
            QR=[]
            QLstay=[]
            QRstay=[]
            QLlaser=[]
            QRlaser=[]
            choices=[]
            QLgeneral=[]
            QRgeneral=[]
            for t in np.arange(NT_all[ms_i]):
                t = int(t)
                predicted_choice = inv_logit(side_mouse
                    + beta_mouse  * (q[1] - q[0])
                    + stay_mouse * (qstay[1] - qstay[0])
                    + laser_mouse * (qlaser[1] - qlaser[0]))

                choice = c[sub_idx[ms_i], sess_idx[ms_i], t]
                q[choice] = (1-alpha) * q[choice] + alpha * r[sub_idx[ms_i], sess_idx[ms_i],t]
                q[choice^1] = q[choice^1] - (q[choice^1] * alphaforgetting)
                if l[sub_idx[ms_i], sess_idx[ms_i],t]==1:
                    qlaser[choice] = qlaser[choice] + alphalaser * l[sub_idx[ms_i], sess_idx[ms_i],t]
                    qlaser[choice^1] = qlaser[choice^1] - (alphalaserforgetting * qlaser[choice^1])
                else:
                    qlaser[choice] = qlaser[choice] + alphalaser * (l[sub_idx[ms_i], sess_idx[ms_i],t] - qlaser[choice])
                    qlaser[choice^1] = qlaser[choice^1] - (alphalaserforgetting * qlaser[choice^1])
                qstay = qstay * (1 - alphastay)
                qstay[choice] = qstay[choice] + alphastay

                # Store variables
                predicted_choices.append(predicted_choice)
                choices.append(choice)
                QL.append(q[0])
                QR.append(q[1])
                QLstay.append(qstay[0])
                QRstay.append(qstay[1])
                QLlaser.append(qlaser[0])
                QRlaser.append(qlaser[1])
                QLgeneral.append((beta_mouse*q[0])+(laser_mouse*qlaser[0])+(stay_mouse*qstay[0]))
                QRgeneral.append((beta_mouse*q[1])+(laser_mouse*qlaser[1])+(stay_mouse*qstay[1]))
            acc = []
            for i in np.arange(len(choices)):
                acc.append(1*(choices[i]==(1*(predicted_choices[i]>0.5))))
            print(np.mean(acc))

            if NT_all[ms_i]!=0:
                ses_data = pd.DataFrame()
                ses_data['predicted_choice'] = predicted_choices
                ses_data['QL'] = QLgeneral
                ses_data['QR'] = QRgeneral
                ses_data['QLreward'] = np.array(QL) * beta_mouse # Note this QL is the reward  - TODO change name
                ses_data['QRreward'] =  np.array(QR) * beta_mouse
                ses_data['QLstay'] =  np.array(QLstay) * stay_mouse
                ses_data['QRstay'] =  np.array(QRstay) * stay_mouse
                ses_data['QLlaser'] =  np.array(QLlaser) * laser_mouse
                ses_data['QRlaser'] =  np.array(QRlaser) * laser_mouse
                ses_data['deltaQ'] = ses_data['QR'] - ses_data['QL']
                ses_data['choices'] = choices
                ses_data['mouse'] = sub_idx[ms_i]
                ses_data['ses'] = sess_idx[ms_i]
                ses_data['acc'] = np.mean(acc)
                data = pd.concat([data,ses_data])
    return make_deltas(data)
def q_learning_model_lapse(standata,saved_params=None, fit=None, csv=True):
    if saved_params is not None:
        r =  standata['r']
        c =  standata['c']
        l =  standata['l']
        sub_idx =  standata['sub_idx']-1
        sess_idx = standata['sess_idx']-1
        NSxNSESS = standata['NSxNSESS']
        NT_all = standata['NT_all']
    else:
        r =  fit.data['r']
        c =  fit.data['c']
        l =  fit.data['l']
        sub_idx =  fit.data['sub_idx']-1
        sess_idx = fit.data['sess_idx']-1
    data = pd.DataFrame()

    for ms_i in np.arange(NSxNSESS):
        if NT_all[ms_i]>0:
            if saved_params is not None:
                if ms_i==0:
                    print('Using saved parameters')
                if csv==True:
                    beta_mouse = float(saved_params.loc[saved_params['name']=='beta_ses['+str(ms_i+1)+']', 'Mean'])
                    stay_mouse = float(saved_params.loc[saved_params['name']=='stay_ses['+str(ms_i+1)+']', 'Mean'])
                    side_mouse = float(saved_params.loc[saved_params['name']=='sides['+str(ms_i+1)+']', 'Mean'])
                    laser_mouse = float(saved_params.loc[saved_params['name']=='laser_ses['+str(ms_i+1)+']', 'Mean'])
                    alpha = float(saved_params.loc[saved_params['name']=='alpha_ses['+str(ms_i+1)+']', 'Mean'])
                    alphalaser = float(saved_params.loc[saved_params['name']=='alphalaser_ses['+str(ms_i+1)+']', 'Mean'])
                    alphastay = float(saved_params.loc[saved_params['name']=='alphastay_ses['+str(ms_i+1)+']', 'Mean'])
                    lapse = float(saved_params.loc[saved_params['name']=='lapse_ses['+str(ms_i+1)+']', 'Mean'])
                else:
                    Print('Non csv version not avaialble atm')

            q = np.zeros(2)
            qstay = np.zeros(2)
            qlaser = np.zeros(2)
            predicted_choices=[]
            QL=[]
            QR=[]
            QLstay=[]
            QRstay=[]
            QLlaser=[]
            QRlaser=[]
            choices=[]
            QLgeneral=[]
            QRgeneral=[]
            for t in np.arange(NT_all[ms_i]):
                t = int(t)
                predicted_choice = (1-lapse)* inv_logit(side_mouse
                    + beta_mouse  * (q[1] - q[0])
                    + stay_mouse * (qstay[1] - qstay[0])
                    + laser_mouse * (qlaser[1] - qlaser[0])) + (lapse/2)

                choice = c[sub_idx[ms_i], sess_idx[ms_i], t]
                q[choice] = (1-alpha) * q[choice] + alpha * r[sub_idx[ms_i], sess_idx[ms_i],t]
                qlaser[choice] = (1-alphalaser) * qlaser[choice] + alphalaser * l[sub_idx[ms_i], sess_idx[ms_i],t]
                qstay = qstay * (1 - alphastay)
                qstay[choice] = qstay[choice] + alphastay
                # Store variables
                predicted_choices.append(predicted_choice)
                choices.append(choice)
                QL.append(q[0])
                QR.append(q[1])
                QLstay.append(qstay[0])
                QRstay.append(qstay[1])
                QLlaser.append(qlaser[0])
                QRlaser.append(qlaser[1])
                QLgeneral.append((beta_mouse*q[0])+(laser_mouse*qlaser[0])+(stay_mouse*qstay[0]))
                QRgeneral.append((beta_mouse*q[1])+(laser_mouse*qlaser[1])+(stay_mouse*qstay[1]))
            acc = []
            for i in np.arange(len(choices)):
                acc.append(1*(choices[i]==(1*(predicted_choices[i]>0.5))))
            #print(np.mean(acc))

            if NT_all[ms_i]!=0:
                ses_data = pd.DataFrame()
                ses_data['predicted_choice'] = predicted_choices
                ses_data['QL'] = QLgeneral
                ses_data['QR'] = QRgeneral
                ses_data['QLreward'] = np.array(QL) * beta_mouse # Note this QL is the reward  - TODO change name
                ses_data['QRreward'] =  np.array(QR) * beta_mouse
                ses_data['QLstay'] =  np.array(QLstay) * stay_mouse
                ses_data['QRstay'] =  np.array(QRstay) * stay_mouse
                ses_data['QLlaser'] =  np.array(QLlaser) * laser_mouse
                ses_data['QRlaser'] =  np.array(QRlaser) * laser_mouse
                ses_data['deltaQ'] = ses_data['QR'] - ses_data['QL']
                ses_data['choices'] = choices
                ses_data['mouse'] = sub_idx[ms_i]
                ses_data['ses'] = sess_idx[ms_i]
                ses_data['acc'] = np.mean(acc)
                data = pd.concat([data,ses_data])
    return make_deltas(data)   
    
def simulate_q_learning_model(standata_recovery,saved_params=None, fit=None, csv=True):
    b =  standata_recovery['b']
    p =  standata_recovery['p']
    tb = standata_recovery['tb']
    NS =standata_recovery['NS']
    NSESS=standata_recovery['NSESS']
    NT =standata_recovery['NT']
    NT_all =standata_recovery['NT_all']
    r_sim = np.zeros([NS,NSESS,NT])
    c_sim = np.zeros([NS,NSESS,NT])
    l_sim = np.zeros([NS,NSESS,NT])
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
                    beta_mouse = float(saved_params.loc[saved_params['name']=='beta_ses['+str(ms_i+1)+']', 'Mean'])
                    stay_mouse = float(saved_params.loc[saved_params['name']=='stay_ses['+str(ms_i+1)+']', 'Mean'])
                    side_mouse = float(saved_params.loc[saved_params['name']=='sides['+str(ms_i+1)+']', 'Mean'])
                    laser_mouse = float(saved_params.loc[saved_params['name']=='laser_ses['+str(ms_i+1)+']', 'Mean'])
                    alpha = float(saved_params.loc[saved_params['name']=='alpha_ses['+str(ms_i+1)+']', 'Mean'])
                    alphalaser = float(saved_params.loc[saved_params['name']=='alphalaser_ses['+str(ms_i+1)+']', 'Mean'])
                    alphastay = float(saved_params.loc[saved_params['name']=='alphastay_ses['+str(ms_i+1)+']', 'Mean'])
                else:
                    print('Error: Only csv version available at the moment!')
            else:
                if ms_i==0:
                    print('No saved parameters')
                beta_mouse = fit.extract()['beta_ses'][:,int(ms_i)].mean()
                stay_mouse = fit.extract()['stay_ses'][:,int(ms_i)].mean()
                side_mouse = fit.extract()['sides'][:,int(ms_i)].mean()
                laser_mouse = fit.extract()['laser_ses'][:,int(ms_i)].mean()
                alpha = phi_approx(fit.extract()['alpha_ses'][:,int(ms_i)].mean())
                alphalaser =phi_approx(fit.extract()['alphalaser_ses'][:,int(ms_i)].mean())
                alphastay = phi_approx(fit.extract()['alphastay_ses'][:,int(ms_i)].mean())

            q = np.zeros(2)
            qstay = np.zeros(2)
            qlaser = np.zeros(2)
            predicted_choices=[]
            rewards = []
            lasers = []
            outcome=[]

            for t in np.arange(NT_all[ms_i]):
                t = int(t)

                p_choice = inv_logit(side_mouse
                    + beta_mouse  * (q[1] - q[0])
                    + stay_mouse * (qstay[1] - qstay[0])
                    + laser_mouse * (qlaser[1] - qlaser[0]))
                choice =  np.random.choice(2,p= [1-p_choice,p_choice])
                assert np.isin(p[sub_idx[ms_i], sess_idx[ms_i],t], [0.7,0.1])
                p_left= p[sub_idx[ms_i], sess_idx[ms_i],t]
                l = b[sub_idx[ms_i], sess_idx[ms_i],t]
                if p_left==0.7:
                    p_right=0.1
                else:
                    p_right=0.7
                if choice==1:
                    f = np.random.choice([0,1], p=[1-p_right,p_right])
                if choice==0:
                    f = np.random.choice([0,1], p=[1-p_left,p_left])
                q[choice] = (1-alpha) * q[choice] + alpha * (1*(f*(l==0)))
                qlaser[choice] = (1-alphalaser) * qlaser[choice] + alphalaser * f * l
                qstay = qstay * (1 - alphastay)
                qstay[choice] = qstay[choice] + alphastay
                rewards.append((1*(f*(l==0))))
                lasers.append(f*l)
                outcome.append(f)
                # Store variables
                predicted_choices.append(choice)

            if NT_all[ms_i]!=0:
                c_sim[sub_idx[ms_i], sess_idx[ms_i], :int(NT_all[ms_i])]= predicted_choices
                l_sim[sub_idx[ms_i], sess_idx[ms_i], :int(NT_all[ms_i])] = lasers
                r_sim[sub_idx[ms_i], sess_idx[ms_i], :int(NT_all[ms_i])] = rewards
                ses_data = pd.DataFrame()
                ses_data['choices'] = predicted_choices
                ses_data['water'] = rewards
                ses_data['laser'] = lasers
                ses_data['laser_block'] = b[sub_idx[ms_i], sess_idx[ms_i],:int(NT_all[ms_i])]
                ses_data['tb'] = tb[sub_idx[ms_i], sess_idx[ms_i],:int(NT_all[ms_i])]
                ses_data['probabilityLeft'] = p[sub_idx[ms_i], sess_idx[ms_i],:int(NT_all[ms_i])]
                ses_data['outcome'] = outcome
                ses_data['mouse'] = sub_idx[ms_i]
                ses_data['ses'] = sess_idx[ms_i]
                sim_data = pd.concat([sim_data,ses_data])
    return c_sim, l_sim, r_sim, sim_data



def sequential_actor_critic_forward(standata, saved_params):
    params = saved_params
    NS = standata["NS"]
    NSxNSESS = standata["NSxNSESS"]
    NT_all = standata["NT_all"]
    sub_idx = standata["sub_idx"] - 1
    sess_idx = standata["sess_idx"] - 1
    o, c, l = standata["o"], standata["c"], standata["l"]

    def p(name):
        try:
            return params[name]
        except KeyError:
            raise KeyError(f"Parameter '{name}' not found in posterior sample")

    def pa(base, index):
        return p(f"{base}[{index}]")

    # Population means. V_*_init_m_l are on the logit scale.
    alpha_v_cs_m, alpha_v_choice_m = p("alpha_v_cs_m"), p("alpha_v_choice_m")
    alpha_loss_m, alpha_lick_m = p("alpha_loss_m"), p("alpha_lick_m")
    V_cs_init_m_l, V_choice_init_m_l = p("V_cs_init_m_l"), p("V_choice_init_m_l")
    beta_win_m, beta_loss_m = p("beta_win_m"), p("beta_loss_m")
    beta_lick_m, pi_lick_init_m, side_m = p("beta_lick_m"), p("pi_lick_init_m"), p("side_m")

    alpha_v_cs_sd, alpha_v_choice_sd = p("alpha_v_cs_sd"), p("alpha_v_choice_sd")
    alpha_loss_sd, alpha_lick_sd = p("alpha_loss_sd"), p("alpha_lick_sd")
    V_cs_init_sd, V_choice_init_sd = p("V_cs_init_sd"), p("V_choice_init_sd")
    beta_win_sd, beta_loss_sd = p("beta_win_sd"), p("beta_loss_sd")
    beta_lick_sd, pi_lick_init_sd, side_sd = p("beta_lick_sd"), p("pi_lick_init_sd"), p("side_sd")

    # Mouse level
    a_v_cs_mouse = np.zeros(NS); a_v_choice_mouse = np.zeros(NS)
    a_loss_mouse = np.zeros(NS); a_lick_mouse = np.zeros(NS)
    V_cs_init_mouse_l = np.zeros(NS); V_choice_init_mouse_l = np.zeros(NS)
    beta_win_mouse = np.zeros(NS); beta_loss_mouse = np.zeros(NS)
    beta_lick_mouse = np.zeros(NS); pi_lick_init_mouse = np.zeros(NS)
    side_mouse = np.zeros(NS)

    for mouse in range(NS):
        mi = mouse + 1
        a_v_cs_mouse[mouse] = alpha_v_cs_m + alpha_v_cs_sd * pa("alpha_v_cs_mouse_z", mi)
        a_v_choice_mouse[mouse] = alpha_v_choice_m + alpha_v_choice_sd * pa("alpha_v_choice_mouse_z", mi)
        a_loss_mouse[mouse] = alpha_loss_m + alpha_loss_sd * pa("alpha_loss_mouse_z", mi)
        a_lick_mouse[mouse] = alpha_lick_m + alpha_lick_sd * pa("alpha_lick_mouse_z", mi)
        V_cs_init_mouse_l[mouse] = V_cs_init_m_l + V_cs_init_sd * pa("V_cs_init_mouse_z", mi)
        V_choice_init_mouse_l[mouse] = V_choice_init_m_l + V_choice_init_sd * pa("V_choice_init_mouse_z", mi)
        beta_win_mouse[mouse] = beta_win_m + beta_win_sd * pa("beta_win_mouse_z", mi)
        beta_loss_mouse[mouse] = beta_loss_m + beta_loss_sd * pa("beta_loss_mouse_z", mi)
        beta_lick_mouse[mouse] = beta_lick_m + beta_lick_sd * pa("beta_lick_mouse_z", mi)
        pi_lick_init_mouse[mouse] = pi_lick_init_m + pi_lick_init_sd * pa("pi_lick_init_mouse_z", mi)
        side_mouse[mouse] = side_m + side_sd * pa("side_mouse_z", mi)

    # Session level
    alpha_loss_ses_sd = p("alpha_loss_ses_sd")
    V_cs_init_ses_sd, V_choice_init_ses_sd = p("V_cs_init_ses_sd"), p("V_choice_init_ses_sd")
    beta_win_ses_sd, beta_loss_ses_sd = p("beta_win_ses_sd"), p("beta_loss_ses_sd")
    pi_lick_init_ses_sd, side_ses_sd = p("pi_lick_init_ses_sd"), p("side_ses_sd")

    alpha_loss_ses = np.zeros(NSxNSESS); V_cs_init_ses = np.zeros(NSxNSESS)
    V_choice_init_ses = np.zeros(NSxNSESS); beta_win_ses = np.zeros(NSxNSESS)
    beta_loss_ses = np.zeros(NSxNSESS); pi_lick_init_ses = np.zeros(NSxNSESS)
    side_ses = np.zeros(NSxNSESS)

    for ms_i in range(NSxNSESS):
        si, stan_i = sub_idx[ms_i], ms_i + 1
        alpha_loss_ses[ms_i] = a_loss_mouse[si] + alpha_loss_ses_sd * pa("alpha_loss_ses_z", stan_i)
        # Logit-scale add, then squash back to [0,1].
        V_cs_init_ses[ms_i] = inv_logit(
            V_cs_init_mouse_l[si] + V_cs_init_ses_sd * pa("V_cs_init_ses_z", stan_i))
        V_choice_init_ses[ms_i] = inv_logit(
            V_choice_init_mouse_l[si] + V_choice_init_ses_sd * pa("V_choice_init_ses_z", stan_i))
        beta_win_ses[ms_i] = beta_win_mouse[si] + beta_win_ses_sd * pa("beta_win_ses_z", stan_i)
        beta_loss_ses[ms_i] = beta_loss_mouse[si] + beta_loss_ses_sd * pa("beta_loss_ses_z", stan_i)
        pi_lick_init_ses[ms_i] = pi_lick_init_mouse[si] + pi_lick_init_ses_sd * pa("pi_lick_init_ses_z", stan_i)
        side_ses[ms_i] = side_mouse[si] + side_ses_sd * pa("side_ses_z", stan_i)

    all_rows = []
    for ms_i in np.arange(NSxNSESS):
        si, sei, n_trials = sub_idx[ms_i], sess_idx[ms_i], NT_all[ms_i]
        if n_trials <= 0:
            continue

        a_v_cs = inv_logit(a_v_cs_mouse[si])
        a_lick = inv_logit(a_lick_mouse[si])
        a_v_choice = inv_logit(a_v_choice_mouse[si])
        a_loss = inv_logit(alpha_loss_ses[ms_i])

        V_cs, V_choice = V_cs_init_ses[ms_i], V_choice_init_ses[ms_i]
        pi_choice, pi_lick = 0.0, pi_lick_init_ses[ms_i]
        predicted_choices, choices = [], []

        for t in range(int(n_trials)):
            choice, outcome, lick = c[si, sei, t], o[si, sei, t], l[si, sei, t]
            action_sign = 2 * choice - 1

            # Decision time, pre-update: predictions only.
            p_right = inv_logit(side_ses[ms_i] + pi_choice)
            p_chosen = p_right if choice == 1 else 1.0 - p_right

            row = {"mouse": si, "ses": sei, "trial": t, "choices": choice,
                   "outcomes": outcome, "licks": lick, "p_chosen": p_chosen,
                   "predicted_choice": p_right, "predicted_lick": np.nan,
                   "delta_cs": np.nan, "delta_laser": np.nan, "p_lick": np.nan,
                   "side_bias": side_ses[ms_i]}

            if outcome == 1:
                delta_cs = V_cs - V_choice
                V_choice += a_v_choice * delta_cs
                pi_choice = pi_choice + action_sign * beta_win_ses[ms_i] * delta_cs * (1.0 - p_chosen)

                p_lick = inv_logit(pi_lick)          # pre-update, a prediction
                row["predicted_lick"] = p_lick
                row["p_lick"] = p_lick
                row["delta_cs"] = delta_cs

                if lick == 1:
                    delta_laser = 1.0 - V_cs
                    V_cs += a_v_cs * delta_laser
                    pi_lick = ((1.0 - a_lick) * pi_lick
                               + beta_lick_mouse[si] * delta_laser * (1.0 - p_lick))
                else:
                    delta_laser = 0.0 - V_cs
                    V_cs += a_v_cs * delta_laser
                    pi_lick = (1.0 - a_lick) * pi_lick
                row["delta_laser"] = delta_laser
            else:
                delta_cs = 0.0 - V_choice
                V_choice += a_v_choice * delta_cs
                pi_choice = ((1.0 - a_loss) * pi_choice
                             + action_sign * beta_loss_ses[ms_i] * delta_cs * (1.0 - p_chosen))
                row["delta_cs"] = delta_cs

            # Post-update state, matching model_comparison_accu, where QR/QL/V
            # are appended after the update - exact for every trial including
            # the last, since the update has already run.
            row["V_cs"], row["V_choice"] = V_cs, V_choice
            row["pi_choice"], row["pi_lick"] = pi_choice, pi_lick

            predicted_choices.append(p_right)
            choices.append(choice)
            all_rows.append(row)

        acc = np.mean([1 * (ch == (1 * (pc > 0.5)))
                       for ch, pc in zip(choices, predicted_choices)])
        for row in all_rows[-len(choices):]:
            row["acc"] = acc
            row["id"] = si * 100 + sei

    return pd.DataFrame(all_rows)



# Correlation matrix
def correlation_matrix(standata, corr_var, model_standard,model_forgetting,reinforce,LRPE,LRPEc,a2stay,LRPEcf,stdlapse, reinforce_w_stay, reinforce_2stay):
    '''
    Standata: is data used to fit the models in stan format.
    corr_var: is the variable to correlate(e.g. deltaQ) (str)
    the rest are the parameters for every model used
    '''
    if corr_var != 'Qstay':
        model_labels = ['Standard','F-Q', 'Laser=RPE/Reward','Laser=RPE/Reward w/ F',
                        '2 QStay', 'Lapse','REINFORCE','REINFORCE_stay','REINFORCE_2stay']
    else:
        model_labels = ['Standard','F-Q', 'Laser=RPE/Reward','Laser=RPE/Reward w/ F',
                        '2 QStay', 'Lapse','REINFORCE_stay','REINFORCE_2stay']

    standard = q_learning_model(standata,saved_params=model_standard)[corr_var].to_numpy()
    fq = q_learning_model_w_forgetting(standata,saved_params=model_forgetting)[corr_var].to_numpy()
    if corr_var != 'Qstay':
        rnfrc = reinforce_model(standata,saved_params=reinforce)[corr_var].to_numpy()
    l = q_learning_model_Lrpe(standata,saved_params=LRPE)[corr_var].to_numpy()
    lr = q_learning_model_Lrpeconditional(standata,saved_params=LRPEc)[corr_var].to_numpy()
    twoq = q_learning_model_2alphas(standata,saved_params=a2stay)[corr_var].to_numpy()
    lrf = q_learning_model_Lrpeconditional_w_forgetting(standata,saved_params=LRPEcf)[corr_var].to_numpy()
    lapse = q_learning_model_lapse (standata, saved_params=stdlapse)[corr_var].to_numpy()
    rnfrc_stay = reinforce_model_w_stay(standata,saved_params=reinforce_w_stay)[corr_var].to_numpy()
    rnfrc_2stay = reinforce_model_w_2stay(standata,saved_params=reinforce_2stay)[corr_var].to_numpy()

    if corr_var != 'Qstay':
        corr_matrix = np.corrcoef([standard,fq,lr,lrf,twoq,lapse,rnfrc,rnfrc_stay,rnfrc_2stay])
    else:
        corr_matrix = np.corrcoef([standard,fq,lr,lrf,twoq,lapse,rnfrc_stay,rnfrc_2stay])
    return corr_matrix, model_labels
def correlation_matrix_reduced(standata, corr_var, model_standard,reinforce_w_stay):
    '''
    Standata: is data used to fit the models in stan format.
    corr_var: is the variable to correlate(e.g. deltaQ) (str)
    the rest are the parameters for every model used
    '''

    model_labels = ['Standard', 'REINFORCE_stay']

    standard = q_learning_model(standata,saved_params=model_standard)[corr_var].to_numpy()
    rnfrc_stay = reinforce_model_w_stay(standata,saved_params=reinforce_w_stay)[corr_var].to_numpy()

    corr_matrix = np.corrcoef([standard,rnfrc_stay])

    return corr_matrix, model_labels


def correlation_matrix_variables(standata,model_forgetting):
    '''
    Standata: is data used to fit the models in stan format.
    corr_var: is the variable to correlate(e.g. deltaQ) (str)
    the rest are the parameters for every model used
    '''

    labels = ['DQLaser', 'DQstay']

    dlaser = q_learning_model_reduced_stay_forgetting(standata,saved_params=model_forgetting)['QRreward'].to_numpy() - \
         q_learning_model_reduced_stay_forgetting(standata,saved_params=model_forgetting)['QLreward'].to_numpy()
    dstay = q_learning_model_reduced_stay_forgetting(standata,saved_params=model_forgetting)['QRstay'].to_numpy() - \
         q_learning_model_reduced_stay_forgetting(standata,saved_params=model_forgetting)['QLstay'].to_numpy()

    corr_matrix = np.corrcoef([dlaser,dstay])
    plot_corr_matrix(corr_matrix, labels)

    return corr_matrix, labels


def plot_corr_matrix(corr_matrix, model_labels, vmin=None, vmax=None):
    sns.heatmap(
    corr_matrix,
    cmap='viridis',
    square=True, vmin=vmin, vmax=vmax, annot=True)
    plt.xticks(np.arange(len(model_labels))+0.5, model_labels,rotation=45)
    plt.yticks(np.arange(len(model_labels))+0.5, model_labels, rotation=0)
def plot_corr_analysis(standata,model_standard,model_forgetting,reinforce,LRPE,LRPEc,a2stay,LRPEcf, stdlapse):
    '''
    Standata: is data used to fit the models in stan format.
    corr_var: is the variable to correlate(e.g. deltaQ) (str)
    the rest are the parameters for every model used
    Return: Plot correlation heatmaps for DeltaQ and predicted choice
    '''
    corr_delta_q, model_labels = correlation_matrix(standata, 'deltaQ', model_standard,model_forgetting,
                            reinforce,LRPE,LRPEc,a2stay,LRPEcf,stdlapse, reinforce_w_stay, reinforce_2stay)
    corr_water, model_labels_water = correlation_matrix(standata, 'Qwater', model_standard,model_forgetting,
                            reinforce,LRPE,LRPEc,a2stay,LRPEcf,stdlapse, reinforce_w_stay, reinforce_2stay)
    corr_stay, model_labels_stay = correlation_matrix(standata, 'Qstay', model_standard,model_forgetting,
                            reinforce,LRPE,LRPEc,a2stay,LRPEcf,stdlapse, reinforce_w_stay, reinforce_2stay)
    corr_laser, model_labels_laser = correlation_matrix(standata, 'Qlaser', model_standard,model_forgetting,
                            reinforce,LRPE,LRPEc,a2stay,LRPEcf,stdlapse, reinforce_w_stay, reinforce_2stay)

    fig, ax = plt.subplots(2,2)
    plt.sca(ax[0,0])
    plot_corr_matrix(corr_delta_q, model_labels)
    plt.title('ΔQ')
    sns.despine()
    plt.tight_layout()
    plt.sca(ax[0,1])
    plot_corr_matrix(corr_water, model_labels_water)
    plt.title('ΔWater')
    plt.tight_layout()
    sns.despine()
    plt.sca(ax[1,0])
    plot_corr_matrix(corr_laser, model_labels_laser)
    plt.title('ΔLaser')
    plt.tight_layout()
    sns.despine()
    plt.sca(ax[1,1])
    plot_corr_matrix(corr_stay, model_labels_stay)
    plt.title('ΔStay')
    plt.tight_layout()
    sns.despine()
    plt.tight_layout()
def confusion_accu_matrix(standata_recovery):
    standard_recovery = pd.read_csv('/Volumes/witten/Alex/Data/ephys_bandit/stan_fits/recovery_models/standard_recovery/output/summary.csv')
    reinforce_w_stay_recovery = pd.read_csv('/Volumes/witten/Alex/Data/ephys_bandit/stan_fits/recovery_models/REINFORCE_recovery/outputsummary.csv')
    confusion_recover_REINFORCEstay_w_standard = pd.read_csv('/Volumes/witten/Alex/Data/ephys_bandit/stan_fits/recovery_models/confusion_recover_REINFORCEstay_w_standard/output/summary.csv')
    confusion_recover_standard_w_REINFORCEstay = pd.read_csv('/Volumes/witten/Alex/Data/ephys_bandit/stan_fits/recovery_models/confusion_recover_standard_w_REINFORCEstay/output/summary.csv')
    standard_sim_r = np.load('/Volumes/witten/Alex/Data/ephys_bandit/stan_fits/recovery_models/standard_recovery/r_sim.npy')
    standard_sim_l = np.load('/Volumes/witten/Alex/Data/ephys_bandit/stan_fits/recovery_models/standard_recovery/l_sim.npy')
    standard_sim_c = np.load('/Volumes/witten/Alex/Data/ephys_bandit/stan_fits/recovery_models/standard_recovery/c_sim.npy')
    reinforce_sim_r = np.load('/Volumes/witten/Alex/Data/ephys_bandit/stan_fits/recovery_models/REINFORCE_recovery/r_sim.npy')
    reinforce_sim_l = np.load('/Volumes/witten/Alex/Data/ephys_bandit/stan_fits/recovery_models/REINFORCE_recovery/l_sim.npy')
    reinforce_sim_c = np.load('/Volumes/witten/Alex/Data/ephys_bandit/stan_fits/recovery_models/REINFORCE_recovery/c_sim.npy')

    standata_standard = replace_with_sim_data(standata_recovery,standard_sim_r,standard_sim_l,standard_sim_c)
    standata_reinforce_w_stay = replace_with_sim_data(standata_recovery,reinforce_sim_r,reinforce_sim_l,reinforce_sim_c)

    reinforce_og_accu = reinforce_model_w_stay(standata_reinforce_w_stay,saved_params=reinforce_w_stay_recovery)['acc'].unique().mean()
    reinforce_confussion_accu = reinforce_model_w_stay(standata_reinforce_w_stay,saved_params=confusion_recover_standard_w_REINFORCEstay)['acc'].unique().mean()
    standard_og_accu = q_learning_model(standata_standard,saved_params=standard_recovery)['acc'].unique().mean()
    standard_confussion_accu = q_learning_model(standata_standard,saved_params=confusion_recover_REINFORCEstay_w_standard)['acc'].unique().mean()

    accu_matrix = np.array([[standard_og_accu, standard_confussion_accu],[reinforce_confussion_accu,reinforce_og_accu]])
    model_labels = ['Q-Learning','REINFORCE w/ stay']

    plot_corr_matrix(accu_matrix, model_labels, vmin=0.5, vmax=0.9)
    plt.ylabel('Sim w/ Q-Learning')
    plt.xlabel('Sim w/ REINFORCE w/ stay')
    plt.tight_layout()
def replace_with_sim_data(standata_recovery,r,l,c):
    standata_recovery['r'] = r.astype(int)
    standata_recovery['l'] = l.astype(int)
    standata_recovery['c'] = c.astype(int)
    return standata_recovery
def params_reinforce_model_laserdecay_exp(standata, saved_params):
    sub_idx =  standata['sub_idx']-1
    sess_idx = standata['sess_idx']-1
    NSxNSESS = standata['NSxNSESS']
    NT_all = standata['NT_all']
    data = pd.DataFrame()
    for ms_i in np.arange(NSxNSESS):
        if NT_all[ms_i]>0:
                dmwater_ses = float(saved_params.loc[saved_params['name']=='dmwater_ses['+str(ms_i+1)+']', 'Mean'])
                dmlaser_ses = float(saved_params.loc[saved_params['name']=='dmlaser_ses['+str(ms_i+1)+']', 'Mean'])
                beta_ses = float(saved_params.loc[saved_params['name']=='beta_ses['+str(ms_i+1)+']', 'Mean'])
                side_ses = float(saved_params.loc[saved_params['name']=='sides['+str(ms_i+1)+']', 'Mean'])
                laser_ses = float(saved_params.loc[saved_params['name']=='laser_ses['+str(ms_i+1)+']', 'Mean'])
                alpha = float(saved_params.loc[saved_params['name']=='alpha_ses['+str(ms_i+1)+']', 'Mean'])
                alphalaser = float(saved_params.loc[saved_params['name']=='alphalaser_ses['+str(ms_i+1)+']', 'Mean'])
                alphaforgetting= float(saved_params.loc[saved_params['name']=='alphaforgetting_ses['+str(ms_i+1)+']', 'Mean'])
                alphalaserforgetting= float(saved_params.loc[saved_params['name']=='alphalaserforgetting_ses['+str(ms_i+1)+']', 'Mean'])
                laserdecay= float(saved_params.loc[saved_params['name']=='laserdecay_ses['+str(ms_i+1)+']', 'Mean'])
                ses_params = pd.DataFrame()
                ses_params['Parameter'] = ['βWaterLoss','βLaserLoss','βWater','Bias', 
                                           'βLaser_init', 'αWater', 'αLaser','αWaterForget',
                                           'αLaserForget','αLaserDecay']
                ses_params['values'] = [dmwater_ses, dmlaser_ses, beta_ses, side_ses, laser_ses, alpha, alphalaser, alphaforgetting, alphalaserforgetting, laserdecay]
                ses_params['mouse'] = sub_idx[ms_i]
                ses_params['ses'] = sess_idx[ms_i]
                data = pd.concat([data,ses_params])
    return data
def params_reinforce_model_laserdecay_winloss_exp(standata, saved_params):
    sub_idx =  standata['sub_idx']-1
    sess_idx = standata['sess_idx']-1
    NSxNSESS = standata['NSxNSESS']
    NT_all = standata['NT_all']
    data = pd.DataFrame()
    for ms_i in np.arange(NSxNSESS):
        if NT_all[ms_i]>0:
                dmwater_ses = float(saved_params.loc[saved_params['name']=='dmwater_ses['+str(ms_i+1)+']', 'Mean'])
                dmlaser_ses = float(saved_params.loc[saved_params['name']=='dmlaser_ses['+str(ms_i+1)+']', 'Mean'])
                beta_ses = float(saved_params.loc[saved_params['name']=='beta_ses['+str(ms_i+1)+']', 'Mean'])
                side_ses = float(saved_params.loc[saved_params['name']=='sides['+str(ms_i+1)+']', 'Mean'])
                laser_ses = float(saved_params.loc[saved_params['name']=='laser_ses['+str(ms_i+1)+']', 'Mean'])
                alpha = float(saved_params.loc[saved_params['name']=='alpha_ses['+str(ms_i+1)+']', 'Mean'])
                alphalaser = float(saved_params.loc[saved_params['name']=='alphalaser_ses['+str(ms_i+1)+']', 'Mean'])
                alphaforgetting= float(saved_params.loc[saved_params['name']=='alphaforgetting_ses['+str(ms_i+1)+']', 'Mean'])
                alphalaserforgetting= float(saved_params.loc[saved_params['name']=='alphalaserforgetting_ses['+str(ms_i+1)+']', 'Mean'])
                laserdecay= float(saved_params.loc[saved_params['name']=='laserdecay_ses['+str(ms_i+1)+']', 'Mean'])
                laserdecayloss= float(saved_params.loc[saved_params['name']=='laserdecayloss_ses['+str(ms_i+1)+']', 'Mean'])                
                betalaserbase= float(saved_params.loc[saved_params['name']=='betalaserbase_ses['+str(ms_i+1)+']', 'Mean'])
                betalaserlossbase= float(saved_params.loc[saved_params['name']=='betalaserlossbase_ses['+str(ms_i+1)+']', 'Mean'])
                ses_params = pd.DataFrame()
                ses_params['Parameter'] = ['βWaterLoss','βLaserLoss_init','βWater','Bias', 
                                           'βLaser_init', 'αWater', 'αLaser','αWaterLoss',
                                           'αLaserLoss','αLaserWinDecay','αLaserLossDecay','LaserWinBaseline', 
                                           'LaserLossBaseline']
                ses_params['values'] = [dmwater_ses, dmlaser_ses, beta_ses, side_ses, laser_ses, 
                                    alpha, alphalaser, alphaforgetting, alphalaserforgetting, laserdecay,
                                    laserdecayloss,betalaserbase,betalaserlossbase]
                ses_params['mouse'] = sub_idx[ms_i]
                ses_params['ses'] = sess_idx[ms_i]
                data = pd.concat([data,ses_params])
    return data
def recovery_params(params_real, recovered_params, standata, phi_a=False, laserdecay=True, z_score=True):
        params_real['fitting_type'] = 'Normal'
        recovered_params['fitting_type'] = 'Recovery'
        saved_params = pd.concat([params_real,recovered_params])
        if laserdecay==True:
            var_names = {'sides':'Bias', 'alphalaser_ses':'αLaser', 'alpha_ses':'αWater',
                        'alphastay_ses':'αStay', 'laser_ses':'βLaser_init', 'betalaserbase_ses':'βLaser','betalaserlossbase_ses':'βLaserLoss',
                        'stay_ses':'βStay', 'beta_ses': 'βWater', 'alphaforgetting_ses':'αWaterForget',
                        'alphalaserforgetting_ses':'αLaserForget',
                        'dmlaser_ses':'βLaserLoss_init', 'dmwater_ses':'βWaterLoss',
                        'laserstay_ses':'βLaserStay', 'alphalaserstay_ses': 'αStayLaser',
                        'lapse_ses':'ε', 'balphalaser_ses':'αBLaser', 'balpha_ses':'αBWater',
                        'laserdecay_ses':'αlaserdecay',
                        'balphastay_ses':'αBStay'}
        else:
            var_names = {'sides':'Bias', 'alphalaser_ses':'αLaser', 'alpha_ses':'αWater',
                        'alphastay_ses':'αStay', 'laser_ses':'βLaser',
                        'stay_ses':'βStay', 'beta_ses': 'βWater', 'alphaforgetting_ses':'αWaterForget',
                        'alphalaserforgetting_ses':'αLaserForget',
                        'dmlaser_ses':'βLaserLoss', 'dmwater_ses':'βWaterLoss',
                        'laserstay_ses':'βLaserStay', 'alphalaserstay_ses': 'αStayLaser',
                        'lapse_ses':'ε', 'balphalaser_ses':'αBLaser', 'balpha_ses':'αBWater',
                        'balphastay_ses':'αBStay'}           
 
        NSxNSESS = standata['NSxNSESS']
        NT_all = standata['NT_all']
        saved_params_new = pd.DataFrame()
        for ms_i in np.arange(NSxNSESS):
            if NT_all[ms_i]>0:
                saved_params_new = pd.concat([saved_params_new, saved_params.loc[(saved_params['name'].str.endswith('['+str(ms_i+1)+']'))]])
        saved_params = saved_params_new
        params=saved_params.loc[(saved_params['name'].str.endswith(']')) &
            (saved_params['name'].str.contains('s\[')),['name','Mean','fitting_type']]
        params['parameter']=params['name'].str.rstrip('[123456789102030]')
        params['parameter'] = params['parameter'].map(var_names)
        if phi_a==True:
            params.loc[(params['parameter'].str.contains('α')), 'Mean'] = \
                    phi_approx(params.loc[(params['parameter'].str.contains('α')), 'Mean']/np.sqrt(2))
        if z_score==True:
            for par in params.parameter.unique():
                params.loc[params['parameter']==par,'Mean'] = zscore(params.loc[params['parameter']==par,'Mean'], nan_policy='omit')

        sns.catplot(data=params, x='fitting_type', col='parameter' ,y='Mean', hue='name',kind='point',
                    legend=False, color= 'k')
        plt.ylabel('Coefficients')
        plt.legend().remove()
        sns.despine()
        plt.tight_layout()     
def plot_corr_analysis_reduced_confusion (standata,model_standard, reinforce_w_stay):
    '''
    Standata: is data used to fit the models in stan format.
    corr_var: is the variable to correlate(e.g. deltaQ) (str)
    the rest are the parameters for every model used
    Return: Plot correlation heatmaps for DeltaQ and predicted choice
    '''
    corr_delta_q, model_labels = correlation_matrix_reduced_confusion(standata, 'deltaQ', model_standard, reinforce_w_stay)
    corr_water, model_labels_water = correlation_matrix_reduced_confusion(standata, 'Qwater', model_standard, reinforce_w_stay)
    corr_stay, model_labels_stay = correlation_matrix_reduced_confusion(standata, 'Qstay', model_standard, reinforce_w_stay)
    corr_laser, model_labels_laser = correlation_matrix_reduced_confusion(standata, 'Qlaser', model_standard, reinforce_w_stay)

    fig, ax = plt.subplots(2,2)
    plt.sca(ax[0,0])
    plot_corr_matrix(corr_delta_q, model_labels,vmin=0,vmax=1)
    plt.title('ΔQ')
    sns.despine()
    plt.tight_layout()
    plt.sca(ax[0,1])
    plot_corr_matrix(corr_water, model_labels_water,vmin=0,vmax=1)
    plt.title('ΔWater')
    plt.tight_layout()
    sns.despine()
    plt.sca(ax[1,0])
    plot_corr_matrix(corr_laser, model_labels_laser,vmin=0,vmax=1)
    plt.title('ΔLaser')
    plt.tight_layout()
    sns.despine()
    plt.sca(ax[1,1])
    plot_corr_matrix(corr_stay, model_labels_stay,vmin=0,vmax=1)
    plt.title('ΔStay')
    plt.tight_layout()
    sns.despine()
    plt.tight_layout()
def correlation_matrix_reduced_confusion(standata, corr_var, model_standard,reinforce_w_stay, mode='standard'):
    '''
    Standata: is data used to fit the models in stan format.
    corr_var: is the variable to correlate(e.g. deltaQ) (str)
    the rest are the parameters for every model used
    '''
    standard = q_learning_model(standata,saved_params=model_standard)[corr_var].to_numpy()
    rnfrc_stay = reinforce_model_w_stay(standata,saved_params=reinforce_w_stay)[corr_var].to_numpy()
    rnfrc_stay_m = reinforce_model_w_stay(standata,saved_params=model_standard)[corr_var].to_numpy()
    standard_m = q_learning_model(standata,saved_params=reinforce_w_stay)[corr_var].to_numpy()

    standard_waters =model_standard.loc[(model_standard['name'].str.endswith(']')) &
        (model_standard['name'].str.contains('beta_ses\['))]
    standard_lasers=model_standard.loc[(model_standard['name'].str.endswith(']')) &
        (model_standard['name'].str.contains('laser_ses\['))]
    standard_stays=model_standard.loc[(model_standard['name'].str.endswith(']')) &
        (model_standard['name'].str.contains('stay_ses\['))]
    standard_alphas=model_standard.loc[(model_standard['name'].str.endswith(']')) &
        (model_standard['name'].str.contains('s\[')) & (model_standard['name'].str.contains('alpha'))]
    reinforce_w_stay_alphas=reinforce_w_stay.loc[(reinforce_w_stay['name'].str.endswith(']')) &
        (reinforce_w_stay['name'].str.contains('s\[')) & (reinforce_w_stay['name'].str.contains('alpha'))]
    reinforce_w_stay_waters=reinforce_w_stay.loc[(reinforce_w_stay['name'].str.endswith(']')) &
        (reinforce_w_stay['name'].str.contains('beta_ses\['))]
    reinforce_w_stay_lasers=reinforce_w_stay.loc[(reinforce_w_stay['name'].str.endswith(']')) &
        (model_standard['name'].str.contains('laser_ses\['))]
    reinforce_w_stay_stays=reinforce_w_stay.loc[(reinforce_w_stay['name'].str.endswith(']')) &
        (reinforce_w_stay['name'].str.contains('stay_ses\['))]
    standard_wo_waters =model_standard.loc[(~((model_standard['name'].str.endswith(']')) &
        (model_standard['name'].str.contains('beta_ses\['))))]
    standard_wo_lasers=model_standard.loc[~((model_standard['name'].str.endswith(']')) &
        (model_standard['name'].str.contains('laser_ses\[')))]
    standard_wo_stays=model_standard.loc[~((model_standard['name'].str.endswith(']')) &
        (model_standard['name'].str.contains('stay_ses\[')))]
    standard_wo_alphas=model_standard.loc[~((model_standard['name'].str.endswith(']')) &
        (model_standard['name'].str.contains('alpha')))]
    reinforce_w_stay_wo_waters=reinforce_w_stay.loc[~(
        reinforce_w_stay['name'].str.contains('beta_ses\['))]
    reinforce_w_stay_wo_lasers=reinforce_w_stay.loc[~((reinforce_w_stay['name'].str.endswith(']')) &
        (reinforce_w_stay['name'].str.contains('laser_ses\[')))]
    reinforce_w_stay_wo_stays=reinforce_w_stay.loc[~((reinforce_w_stay['name'].str.endswith(']')) &
        (reinforce_w_stay['name'].str.contains('stay_ses\[')))]
    reinforce_w_stay_wo_alphas=reinforce_w_stay.loc[~((reinforce_w_stay['name'].str.endswith(']')) &
        (reinforce_w_stay['name'].str.contains('alpha')))]

    reinforce_alphas_m = reinforce_model_w_stay(standata,saved_params=pd.concat([standard_alphas,reinforce_w_stay_wo_alphas]))[corr_var].to_numpy()
    standard_alphas_m = q_learning_model(standata,saved_params=pd.concat([reinforce_w_stay_alphas,standard_wo_alphas]))[corr_var].to_numpy()
    reinforce_water_m = reinforce_model_w_stay(standata,saved_params=pd.concat([standard_waters,reinforce_w_stay_wo_waters]))[corr_var].to_numpy()
    standard_water_m = q_learning_model(standata,saved_params=pd.concat([reinforce_w_stay_waters,standard_wo_waters]))[corr_var].to_numpy()
    reinforce_laser_m = reinforce_model_w_stay(standata,saved_params=pd.concat([standard_lasers,reinforce_w_stay_wo_lasers]))[corr_var].to_numpy()
    standard_laser_m = q_learning_model(standata,saved_params=pd.concat([reinforce_w_stay_lasers,standard_wo_lasers]))[corr_var].to_numpy()
    reinforce_stay_m = reinforce_model_w_stay(standata,saved_params=pd.concat([standard_stays,reinforce_w_stay_wo_stays]))[corr_var].to_numpy()
    standard_stay_m = q_learning_model(standata,saved_params=pd.concat([reinforce_w_stay_stays,standard_wo_stays]))[corr_var].to_numpy()

    corr_matrix = np.corrcoef([standard,standard_m,standard_alphas_m,standard_water_m,standard_laser_m,standard_stay_m,
                    rnfrc_stay,rnfrc_stay_m,reinforce_alphas_m,reinforce_water_m,reinforce_laser_m,reinforce_stay_m])

    model_labels = ['standard','standard_m','standard_alphas_m','standard_water_m','standard_laser_m','standard_stay_m',
                    'rnfrc_stay','rnfrc_stay_m','reinforce_alphas_m','reinforce_water_m','reinforce_laser_m','reinforce_stay_m']

    return corr_matrix, model_labels
def accuracy_by_section(model_data, n_sections=8):
    model_data = model_data.reset_index()
    model_data['section'] = pd.cut(model_data['index'],bins=n_sections)
    model_data['bin_prediction'] = 1*(model_data['predicted_choice']>0.5)
    model_data['correct_guess'] = model_data['bin_prediction']==model_data['choices']
    breakdown = model_data.groupby(['section']).mean()['correct_guess']
    return breakdown
def time_to_laserbaseline(model,standata,params, variable='betalaser'):
        data = model(standata,saved_params=params)
        data = data.reset_index()
        data['betaqlaser'] = data['Qlaser']*data['betalaser']
        data['id'] = data['mouse']*100+data['ses']
        sns.lineplot(data=data, x='index', y=variable, hue='id', palette='mako')
def plot_params_reinforce_laserdecay_stay(saved_params, stan_data, save=False, output_path=None):
        var_names = {'beta_ses': 'βWater','sides':'Bias', 'laser_ses':'βLaser_init', 'ep_ses':'εStay', 'alpha_ses':'αWater',
                    'alphalaser_ses':'αLaser', 'alphastay_ses':'αStay', 'alphaforgetting_ses':'αWaterForget', 'alphalaserforgetting_ses':'αLaserForget',
                    'laserdecay_ses':'αLaserDecay','betalaserbase_ses':'βLaserBase','dmwater_ses':'βWaterLoss','dmlaser_ses':'βLaserLoss_init',
                    'betalaserlossbase_ses':'βLaserLossBase', 'laserdecayloss_ses':'αLaserDecayLoss'}

        category = {'beta_ses': 'beta_bases','sides':'bias', 'laser_ses':'beta_inits', 'ep_ses':'beta_bases', 'alpha_ses':'alphas',
                    'alphalaser_ses':'alphas', 'alphastay_ses':'alphas', 'alphaforgetting_ses':'alphas', 'alphalaserforgetting_ses':'alphas',
                    'laserdecay_ses':'decays','betalaserbase_ses':'beta_bases','dmwater_ses':'beta_bases','dmlaser_ses':'beta_inits',
                    'betalaserlossbase_ses':'beta_bases', 'laserdecayloss_ses':'decays'}

        pal = {'beta_ses': 'dodgerblue','sides':'gray', 'laser_ses':'orange', 'ep_ses':'gray', 'alpha_ses':'dodgerblue',
                    'alphalaser_ses':'orange', 'alphastay_ses':'gray', 'alphaforgetting_ses':'dodgerblue', 'alphalaserforgetting_ses':'orange',
                    'laserdecay_ses':'orange','betalaserbase_ses':'orange','dmwater_ses':'dodgerblue','dmlaser_ses':'orange',
                    'betalaserlossbase_ses':'orange', 'laserdecayloss_ses':'orange'}    

        NSxNSESS = standata['NSxNSESS']
        NT_all = standata['NT_all']
        saved_params_new = pd.DataFrame()
        for ms_i in np.arange(NSxNSESS):
            if NT_all[ms_i]>0:
                saved_params_new = pd.concat([saved_params_new, saved_params.loc[(saved_params['name'].str.endswith('['+str(ms_i+1)+']'))]])
        saved_params = saved_params_new
        params=saved_params.loc[(saved_params['name'].str.endswith(']')) &
            (saved_params['name'].str.contains('s\[')),['name','Mean']]
        params['parameter']=params['name'].str.rstrip('[123456789102030]')
        params['category'] = params['parameter'].map(category)
        params['pal']= params['parameter'].map(pal)
        params['parameter'] = params['parameter'].map(var_names)
        fig, ax = plt.subplots(2,3, figsize=(10,12))
        plt.sca(ax[0,0])
        sns.swarmplot(data=params.loc[params['category']=='beta_bases'], 
                        x='parameter', y='Mean', color='k')
        sns.barplot(data=params.loc[params['category']=='beta_bases'], x='parameter', y='Mean',
                    palette=params.loc[params['category']=='beta_bases','pal'])
        plt.ylabel('Coefficient')
        plt.xlabel(' ')
        plt.xticks(rotation=45)
        plt.title('Betas')
        plt.sca(ax[0,1])
        sns.swarmplot(data=params.loc[params['category']=='alphas'], 
                        x='parameter', y='Mean', color='k')
        sns.barplot(data=params.loc[params['category']=='alphas'], x='parameter', y='Mean',
                    palette=params.loc[params['category']=='alphas','pal'])
        plt.ylabel('Coefficient')
        plt.xlabel(' ')
        plt.xticks(rotation=45)
        plt.title('Alphas')
        plt.sca(ax[0,2])
        sns.swarmplot(data=params.loc[params['category']=='bias'], 
                        x='parameter', y='Mean', color='k')
        sns.barplot(data=params.loc[params['category']=='bias'], x='parameter', y='Mean',
                    palette=params.loc[params['category']=='bias','pal'])
        plt.ylabel('Coefficient')
        plt.xlabel(' ')
        plt.xticks(rotation=45)
        plt.title('Bias')
        plt.sca(ax[1,0])
        sns.swarmplot(data=params.loc[params['category']=='beta_inits'], 
                        x='parameter', y='Mean', color='k')
        sns.barplot(data=params.loc[params['category']=='beta_inits'], x='parameter', y='Mean',
                    palette=params.loc[params['category']=='beta_inits','pal'])
        plt.ylabel('Coefficient')
        plt.xlabel(' ')
        plt.xticks(rotation=45)
        plt.title('Initializations')
        plt.sca(ax[1,1])
        sns.swarmplot(data=params.loc[params['category']=='decays'], 
                        x='parameter', y='Mean', color='k')
        sns.barplot(data=params.loc[params['category']=='decays'], x='parameter', y='Mean',
                    palette=params.loc[params['category']=='decays','pal'])
        plt.ylabel('Coefficient')
        plt.xlabel(' ')
        plt.title('Decays')
        plt.xticks(rotation=45)
        sns.despine()
        plt.tight_layout()
def plot_params_standard_laserdecay(saved_params, stan_data, save=False, output_path=None):
        var_names = {'beta_ses': 'βWater','sides':'Bias', 'laser_ses':'βLaser_init', 'stay_ses':'βStay', 'alpha_ses':'αWater',
                    'alphalaser_ses':'αLaser', 'alphastay_ses':'αStay', 
                    'laserdecay_ses':'αLaserDecay', 'betalaserbase_ses':'βLaserBaser'}

        category = {'beta_ses': 'beta_bases','sides':'bias', 'laser_ses':'beta_inits', 'stay_ses':'beta_bases', 'alpha_ses':'alphas',
                    'alphalaser_ses':'alphas', 'alphastay_ses':'alphas', 
                    'laserdecay_ses':'decays',  'betalaserbase_ses':'beta_bases'}

        pal = {'beta_ses': 'dodgerblue','sides':'gray', 'laser_ses':'orange', 'stay_ses':'gray', 'alpha_ses':'dodgerblue',
                    'alphalaser_ses':'orange', 'alphastay_ses':'gray',
                    'laserdecay_ses':'orange',  'betalaserbase_ses':'orange'}    

        NSxNSESS = standata['NSxNSESS']
        NT_all = standata['NT_all']
        saved_params_new = pd.DataFrame()
        for ms_i in np.arange(NSxNSESS):
            if NT_all[ms_i]>0:
                saved_params_new = pd.concat([saved_params_new, saved_params.loc[(saved_params['name'].str.endswith('['+str(ms_i+1)+']'))]])
        saved_params = saved_params_new
        params=saved_params.loc[(saved_params['name'].str.endswith(']')) &
            (saved_params['name'].str.contains('s\[')),['name','Mean']]
        params['parameter']=params['name'].str.rstrip('[123456789102030]')
        params['category'] = params['parameter'].map(category)
        params['pal']= params['parameter'].map(pal)
        params['parameter'] = params['parameter'].map(var_names)
        fig, ax = plt.subplots(2,3, figsize=(10,12))
        plt.sca(ax[0,0])
        sns.swarmplot(data=params.loc[params['category']=='beta_bases'], 
                        x='parameter', y='Mean', color='k')
        sns.barplot(data=params.loc[params['category']=='beta_bases'], x='parameter', y='Mean',
                    palette=params.loc[params['category']=='beta_bases','pal'])
        plt.ylabel('Coefficient')
        plt.xlabel(' ')
        plt.xticks(rotation=45)
        plt.title('Betas')
        plt.sca(ax[0,1])
        sns.swarmplot(data=params.loc[params['category']=='alphas'], 
                        x='parameter', y='Mean', color='k')
        sns.barplot(data=params.loc[params['category']=='alphas'], x='parameter', y='Mean',
                    palette=params.loc[params['category']=='alphas','pal'])
        plt.ylabel('Coefficient')
        plt.xlabel(' ')
        plt.xticks(rotation=45)
        plt.title('Alphas')
        plt.sca(ax[0,2])
        sns.swarmplot(data=params.loc[params['category']=='bias'], 
                        x='parameter', y='Mean', color='k')
        sns.barplot(data=params.loc[params['category']=='bias'], x='parameter', y='Mean',
                    palette=params.loc[params['category']=='bias','pal'])
        plt.ylabel('Coefficient')
        plt.xlabel(' ')
        plt.xticks(rotation=45)
        plt.title('Bias')
        plt.sca(ax[1,0])
        sns.swarmplot(data=params.loc[params['category']=='beta_inits'], 
                        x='parameter', y='Mean', color='k')
        sns.barplot(data=params.loc[params['category']=='beta_inits'], x='parameter', y='Mean',
                    palette=params.loc[params['category']=='beta_inits','pal'])
        plt.ylabel('Coefficient')
        plt.xlabel(' ')
        plt.xticks(rotation=45)
        plt.title('Initializations')
        plt.sca(ax[1,1])
        sns.swarmplot(data=params.loc[params['category']=='decays'], 
                        x='parameter', y='Mean', color='k')
        sns.barplot(data=params.loc[params['category']=='decays'], x='parameter', y='Mean',
                    palette=params.loc[params['category']=='decays','pal'])
        plt.ylabel('Coefficient')
        plt.xlabel(' ')
        plt.title('Decays')
        plt.xticks(rotation=45)
        sns.despine()
        plt.tight_layout()
