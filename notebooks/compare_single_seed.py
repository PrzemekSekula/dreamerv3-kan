# %% [markdown]
# # Compare single seed results between Kan and Org netrowks

# %%
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

SEED_NR = 0
KAN_FOLDER = '../data/dmc/kan_enc'
ORG_FOLDER = '../data/dmc/original'


kan_df = pd.read_csv(os.path.join(KAN_FOLDER, f'seed_{SEED_NR}.csv'))
org_df = pd.read_csv(os.path.join(ORG_FOLDER, f'seed_{SEED_NR}.csv'))

org_df.head(3)



# %%
max_step = kan_df.step_nr.max()

kan_df = kan_df[kan_df.step_nr == max_step][['subfolder_name', 'eval_return']]
org_df = org_df[org_df.step_nr == max_step][['subfolder_name', 'eval_return']]

df = pd.merge(kan_df, org_df, on='subfolder_name', suffixes=('_kan', '_org'))
df['diff'] = df['eval_return_kan'] - df['eval_return_org']
df['diff_perc'] = np.abs(df['diff'] / df['eval_return_org']) * 100
df = df.sort_values('diff_perc', ascending=True)
df.head(10)

# %%
def get_dmc_state_size(env_name):
    """
    Calculate the total state size of a given dm_control environment.
    
    This function loads the specified dm_control environment and sums the 
    sizes (products of dimensions) of all observations returned by the 
    environment's observation_spec.
    
    Parameters:
        env_name (str): The name of the environment (e.g., 'dmc_walker_walk' or 'walker_walk').
        
    Returns:
        int: The total size of the state space, or None if the environment 
             cannot be parsed or loaded.
    """
    try:
        from dm_control import suite
        import numpy as np
        
        name = env_name.replace('dmc_', '')
        if '_' not in name:
            return None
            
        domain, task = name.split('_', 1)
        if domain == "cup":
            domain = "ball_in_cup"
            
        env = suite.load(domain, task)
        obs_spec = env.observation_spec()
        
        state_size = 0
        for k, v in obs_spec.items():
            if len(v.shape) == 0:
                state_size += 1
            else:
                state_size += np.prod(v.shape)
        
        return int(state_size)
    except Exception:
        return None

if '/dmc/' in KAN_FOLDER:
    df['state_size'] = df['subfolder_name'].apply(get_dmc_state_size)
    df = df.sort_values(['state_size', 'diff_perc'] )
    df.head(20)
    
# %%
df
# %%
