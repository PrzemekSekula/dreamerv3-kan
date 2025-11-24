# %%
import os
from datetime import datetime
import tensorflow as tf
import pandas as pd
from pathlib import Path

# %%
def get_eval_return_from_log_dir(log_dir):
    """
    Extract eval_return metric values from a TensorBoard log directory.
    
    Parameters:
        log_dir (str): Path to the TensorBoard log directory.
    
    Returns:
        list: List of tuples containing (step_nr, log_time, eval_return) for each log entry.
    """
    eval_return_data = []
    
    # Iterate over all event files in the directory
    for root, _, files in os.walk(log_dir):
        for file in files:
            if file.startswith("events.out.tfevents"):
                file_path = os.path.join(root, file)
                
                # Parse the event file
                try:
                    for event in tf.compat.v1.train.summary_iterator(file_path):
                        if event.wall_time and event.step:
                            # Extract wall time
                            log_time = datetime.fromtimestamp(event.wall_time)
                            
                            # Extract eval_return from event summaries
                            if event.summary:
                                for value in event.summary.value:
                                    # Check for eval_return metric (with or without scalars/ prefix)
                                    metric_name = value.tag
                                    if metric_name in ["eval_return", "scalars/eval_return"]:
                                        eval_return = value.simple_value
                                        eval_return_data.append({
                                            'step_nr': event.step,
                                            'log_time': log_time,
                                            'eval_return': eval_return
                                        })
                except Exception as e:
                    print(f"Warning: Error reading {file_path}: {e}")
                    continue
    
    return eval_return_data


# %%
def extract_eval_return_from_folder(folder_path, verbose=False):
    """
    Extract eval_return metrics from all direct subfolders in the given folder.
    
    Parameters:
        folder_path (str): Path to the parent folder containing subfolders with TensorBoard logs.
    
    Returns:
        pd.DataFrame: DataFrame with columns: subfolder_name, step_nr, log_time, eval_return.
    """
    folder_path = Path(folder_path)
    
    if not folder_path.exists():
        raise ValueError(f"Folder does not exist: {folder_path}")
    
    if not folder_path.is_dir():
        raise ValueError(f"Path is not a directory: {folder_path}")
    
    all_data = []
    
    # Get all direct subfolders (not recursively)
    subfolders = [f for f in folder_path.iterdir() if f.is_dir()]
    
    print(f"Found {len(subfolders)} subfolders")
    for subfolder in subfolders:
        if verbose:
            print(f"  - {subfolder.name}")
    
    # Process each subfolder
    for subfolder in subfolders:
        if verbose:
            print(f"\nProcessing: {subfolder.name}")
        eval_return_data = get_eval_return_from_log_dir(str(subfolder))
        
        if eval_return_data:
            if verbose:
                print(f"  Found {len(eval_return_data)} eval_return entries")
            # Add subfolder name to each entry
            for entry in eval_return_data:
                entry['subfolder_name'] = subfolder.name
                all_data.append(entry)
        else:
            print(f"  No eval_return data found for {subfolder.name} subfolder")
    
    # Create DataFrame
    if all_data:
        df = pd.DataFrame(all_data)
        # Reorder columns to match required format
        df = df[['subfolder_name', 'step_nr', 'log_time', 'eval_return']]
        # Sort by subfolder_name and step_nr for consistency
        df = df.sort_values(['subfolder_name', 'step_nr']).reset_index(drop=True)
        return df
    else:
        # Return empty DataFrame with correct columns
        return pd.DataFrame(columns=['subfolder_name', 'step_nr', 'log_time', 'eval_return'])


# %%
# Example usage
# Set the folder path here
if False:
    LOG_FOLDER = '../log_dmc/dmc_proprio'
    #LOG_FOLDER = '../log_atari100k/original/seed_{}'
    # Extract eval_return metrics
    df = extract_eval_return_from_folder(LOG_FOLDER)

    # Display results
    print("\n" + "="*50)
    print("Results:")
    print("="*50)
    print(f"\nTotal entries: {len(df)}")
    print(f"Unique subfolders: {df['subfolder_name'].nunique() if len(df) > 0 else 0}")
    df.head()
    folder_name = os.path.basename(LOG_FOLDER)
    df.to_csv(os.path.join('../data/dmc', folder_name + '.csv'), index=False)

# %%
# Runniing the data
LOG_FOLDER = '../log_atari100k/original/'
DEST_FOLDER = '../data/atari100k/orginal'
os.makedirs(DEST_FOLDER, exist_ok=True)

df_list = []

for seed_nr in range(4):
    folder = f'{LOG_FOLDER}/seed_{seed_nr}'
    df = extract_eval_return_from_folder(folder)   
    df.to_csv(os.path.join(DEST_FOLDER, 'seed_{}.csv'.format(seed_nr)), index=False)
    pivoted_df = df.pivot(index='step_nr', columns='subfolder_name', values='eval_return')
    df_list.append(pivoted_df)
# %%
def aggregate_pivoted_dataframes(dfs_list):
    """
    Takes a list of DataFrames (with identical indices and columns)
    and calculates the mean and std for each column across the list.
    """
    if not dfs_list:
        return None

    # 1. Concatenate all DataFrames vertically. 
    # Since indices are identical (step_nr), duplicate indices will appear.
    combined_df = pd.concat(dfs_list)

    # 2. Group by the index (step_nr) and calculate mean and std
    # This collapses the duplicate indices back into unique rows
    stats_df = combined_df.groupby(level=0).agg(['mean', 'std'])

    # 3. Flatten the hierarchical column structure
    # The agg function creates a MultiIndex (e.g., 'model_a' -> 'mean', 'std')
    # We map this to 'model_a_mean', 'model_a_std'
    stats_df.columns = [f'{col}_{stat}' for col, stat in stats_df.columns]

    return stats_df

stats_df = aggregate_pivoted_dataframes(df_list)
stats_df.to_csv(os.path.join(DEST_FOLDER, 'summary.csv'), index=False)
stats_df.head()


# %%
import matplotlib.pyplot as plt


def plot_mean_and_std(stats_df):
    """
    Plots mean and std, but scales the Y-axis based ONLY on the mean line.
    """
    base_names = [col.replace('_mean', '') for col in stats_df.columns if col.endswith('_mean')]
    
    for name in base_names:
        mean_col = f"{name}_mean"
        std_col = f"{name}_std"
        
        if std_col not in stats_df.columns:
            continue
            
        plt.figure(figsize=(10, 6))
        
        # 1. Plot Mean and Std
        plt.plot(stats_df.index, stats_df[mean_col], label=f'{name} Mean', linewidth=2)
        plt.fill_between(
            stats_df.index, 
            stats_df[mean_col] - stats_df[std_col], 
            stats_df[mean_col] + stats_df[std_col], 
            alpha=0.3,
            label=f'{name} Std Dev'
        )
        
        # 2. Calculate limits based ONLY on the mean column
        y_min = stats_df[mean_col].min()
        y_max = stats_df[mean_col].max()
        
        # 3. Add 10% padding so the line isn't touching the edges
        padding = (y_max - y_min) * 0.1
        if padding == 0: padding = 0.1  # Safety for flat lines
        
        # 4. Force the Y-axis to ignore the std deviation height
        plt.ylim(y_min - padding, y_max + padding)
        
        plt.title(f'Performance: {name} (Scaled to Mean)')
        plt.xlabel('Step Nr')
        plt.ylabel('Value')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.show()
        
        
plot_mean_and_std(stats_df)
# %%
