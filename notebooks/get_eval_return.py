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
def extract_eval_return_from_folder(folder_path):
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
    
    print(f"Found {len(subfolders)} subfolders:")
    for subfolder in subfolders:
        print(f"  - {subfolder.name}")
    
    # Process each subfolder
    for subfolder in subfolders:
        print(f"\nProcessing: {subfolder.name}")
        eval_return_data = get_eval_return_from_log_dir(str(subfolder))
        
        if eval_return_data:
            print(f"  Found {len(eval_return_data)} eval_return entries")
            # Add subfolder name to each entry
            for entry in eval_return_data:
                entry['subfolder_name'] = subfolder.name
                all_data.append(entry)
        else:
            print(f"  No eval_return data found")
    
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
LOG_FOLDER = '../log_dmc/dmc_proprio'

# Extract eval_return metrics
df = extract_eval_return_from_folder(LOG_FOLDER)

# Display results
print("\n" + "="*50)
print("Results:")
print("="*50)
print(f"\nTotal entries: {len(df)}")
print(f"Unique subfolders: {df['subfolder_name'].nunique() if len(df) > 0 else 0}")
df.head()

# %%
# Get the name of the folder (without path)
folder_name = os.path.basename(LOG_FOLDER)
df.to_csv(os.path.join('../data/dmc', folder_name + '.csv'), index=False)
# %%
# %%
