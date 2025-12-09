"""
Prepares summary assuming the data data are already downloaded from
tensorboard with the extract_eval_return_from_folder function.
"""

# %%
import os
import pandas as pd

from result_analysis_functions import (
    aggregate_pivoted_dataframes,
    plot_mean_and_std,
    plot_comparable_results,
    generate_pdf_with_charts,
)

TASK_NAME = 'atari100k'

LOG_FOLDER = f'../data/{TASK_NAME}/'
IMAGE_FOLDER = f'./tmp/{TASK_NAME}/'

SUBFOLDERS = ['original', 'kan_enc'] # subfolder for different models
PDF_NAME = os.path.join(LOG_FOLDER, f'{TASK_NAME}_summary.pdf')


# %%
# If true, it aggregates the data from the individual files into a single summary file.
# Useful when the individual data is downloaded from other servers
if False: 

    for subfolder in SUBFOLDERS:

        log_folder = os.path.join(LOG_FOLDER, subfolder)

        df_list = []

        files = [
            entry for entry in os.scandir(log_folder)
            if entry.is_file() and entry.name.endswith('csv') and entry.name != 'summary.csv'
        ]

        for entry in files:

            df = pd.read_csv(entry.path)

            pivoted_df = df.pivot(
                index="step_nr", columns="subfolder_name", values="eval_return"
            )
            df_list.append(pivoted_df)

        stats_df = aggregate_pivoted_dataframes(df_list)

        mean_cols = [col for col in stats_df.columns if col.endswith('_mean')]
        stats_df = stats_df.dropna(subset=mean_cols)
        stats_df.to_csv(os.path.join(log_folder, 'summary.csv'), index=False)
        #stats_df.head(20)


# %%
#plot_mean_and_std(stats_df)

# %%
# Prepare summary for both kan and original dataframes


df_kan = pd.read_csv('../data/dmc/kan_enc/summary.csv')
df_org = pd.read_csv('../data/dmc/original/summary.csv')

plot_comparable_results(df_org, df_kan, save_path = IMAGE_FOLDER)
# %%
# Generate PDF with charts

generate_pdf_with_charts(PDF_NAME, IMAGE_FOLDER, nr_cols=3)
# %%