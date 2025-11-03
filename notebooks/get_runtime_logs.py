# %%
import os
from datetime import datetime
import tensorflow as tf
import pandas as pd

# %%
LOG_FOLDER = '../log_dmc/dmc_kan_enc/'

# Get all subfolders
subfolders = [f.path for f in os.scandir(LOG_FOLDER) if f.is_dir()]
print(f"Found {len(subfolders)} subfolders:")
for subfolder in subfolders:
    print(f"- {subfolder}")
    
# %%


def get_tensorboard_log_times(log_dir):
    """
    Check when logging started and ended in a TensorBoard log directory.

    Parameters:
        log_dir (str): Path to the TensorBoard log directory.

    Returns:
        dict: A dictionary with 'start_time' and 'end_time' as datetime objects.
    """
    start_time = None
    end_time = None

    # Iterate over all event files in the directory
    for root, _, files in os.walk(log_dir):
        for file in files:
            if file.startswith("events.out.tfevents"):
                file_path = os.path.join(root, file)

                # Parse the event file
                for event in tf.compat.v1.train.summary_iterator(file_path):
                    if event.wall_time:  # Extract the wall time
                        event_time = datetime.fromtimestamp(event.wall_time)
                        if start_time is None or event_time < start_time:
                            start_time = event_time
                        if end_time is None or event_time > end_time:
                            end_time = event_time

    return start_time, end_time


df = pd.DataFrame(columns=['env', 'start_time', 'end_time'])

for folder in subfolders:
    df.loc[len(df), :] = [
        os.path.basename(folder),
        *get_tensorboard_log_times(folder)
        ]

df['duration'] = df['end_time'] - df['start_time']  
df.sort_values('duration', ascending=False, inplace=True)
df.head()
# %%