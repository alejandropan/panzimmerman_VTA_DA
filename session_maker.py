import numpy as np

#####################################################################
######################## Functions ###############################
#####################################################################

# Tools
def texp(factor: float = 0.35, min_: float = 0.2, max_: float = 0.5) -> float:
    x = np.random.exponential(factor)
    if min_ <= x <= max_:
        return x
    else:
        return texp(factor=factor, min_=min_, max_=max_)

# Block parameters
def get_block_len(factor, min_, max_):
    return int(texp(factor=factor, min_=min_, max_=max_))

def update_block_params(tph):
    tph.block_trial_num += 1
    tph.opto_block_trial_num += 1
    last_10 = np.array(tph.response_side_buffer[-10:])
    high_press = (tph.stim_probability_left<0.5)*1 # whether right choices are favored in block,1 yes 0 no
    left_right_total = [len(np.where(last_10==0)[0]),len(np.where(last_10==1)[0])]
    if (tph.block_trial_num > tph.block_len):
        if left_right_total[high_press]<7:
            tph.block_len = tph.block_len + 1
            #print('Debiasing')
        else:
            #print('Block change')
            tph.block_num += 1
            tph.block_trial_num = 1
            tph.block_len = get_block_len(
                factor=tph.block_len_factor, min_=tph.block_len_min, max_=tph.block_len_max
            )
    if (tph.opto_block_trial_num > tph.opto_block_len):
        tph.opto_block = 1 - tph.opto_block
        tph.opto_block_len = get_block_len(
                factor=30, 
                min_=1, 
                max_=60)
        tph.opto_block_trial_num = 1
    return tph


def update_probability_left(tph):
    if tph.block_trial_num != 1:
        return tph.stim_probability_left
    if tph.block_num == 1 and tph.block_init_5050:
        return 0.5
    elif tph.block_num == 1 and not tph.block_init_5050:
        return np.random.choice(tph.block_probability_set)
    elif tph.block_num == 2 and tph.block_init_5050:
        return np.random.choice(tph.block_probability_set)
    else:
        block = np.random.choice(tph.block_probability_set)
        #print(block)
        if block != tph.stim_probability_left:
            return block
        else:
            #print('second', block)
            return update_probability_left(tph)

def draw_reward(stim_probability_left):
    reward_left = np.random.choice([1,0], p=[stim_probability_left, 1 - stim_probability_left])
    if np.isin(stim_probability_left,[0.8,0.2,0.5]):
        stim_probability_right = 1-stim_probability_left
    if np.isin(stim_probability_left,[0.7,0.1]):
        stim_probability_right= 0.1 if stim_probability_left==0.7 else 0.7
    if np.isin(stim_probability_left,[1.0,0.0]):
        stim_probability_right= 0.0 if stim_probability_left==1.0 else 1.0
    reward_right = np.random.choice([1,0], p=[stim_probability_right, 1-stim_probability_right])
    return reward_left, reward_right

#####################################################################
######################## Trial Object ###############################
#####################################################################

class TrialParamHandler(object):
    def __init__(self):
        # Initialize parameters that may change every trial
        self.trial_num = 0
        self.opto_block_prob = 0.5 
        self.opto_block = np.random.choice(2, p=[1-self.opto_block_prob, self.opto_block_prob])
        self.block_num = 0
        self.block_trial_num = 0
        self.opto_block_trial_num = 0
        self.block_len_factor = 30
        self.block_len_min = 20
        self.block_len_max = 40
        self.block_probability_set = [0.1, 0.7]
        self.block_len = get_block_len(self.block_len_factor, self.block_len_min, self.block_len_max)
        self.opto_block_len = get_block_len(
            factor=30, min_=1, max_=60
        )
        self.block_init_5050 = False
        # Rewarded choice
        self.stim_probability_left = np.random.choice(self.block_probability_set, p=[0.5,0.5])
        self.left_reward, self.right_reward = draw_reward(self.stim_probability_left)
        # Choice
        self.response_side_buffer= [] # Init list

    def next_trial(self):
        # First trial exception
        if self.trial_num == 0:
            self.trial_num += 1
            self.block_num += 1
            self.block_trial_num += 1
        # Increment trial number
        self.trial_num += 1
        # Update block
        self = update_block_params(self)
        # Update stim probability left + buffer
        self.stim_probability_left = update_probability_left(self)
        # Update reward
        self.left_reward, self.right_reward = draw_reward(
            self.stim_probability_left
        )
        return self