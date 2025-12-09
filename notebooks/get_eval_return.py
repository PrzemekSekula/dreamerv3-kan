# %%
import os

from result_analysis_functions import (
    aggregate_pivoted_dataframes,
    extract_eval_return_from_folder,
    plot_mean_and_std,
)


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
    folder = f"{LOG_FOLDER}/seed_{seed_nr}"
    df = extract_eval_return_from_folder(folder)
    df.to_csv(os.path.join(DEST_FOLDER, "seed_{}.csv".format(seed_nr)), index=False)
    pivoted_df = df.pivot(
        index="step_nr", columns="subfolder_name", values="eval_return"
    )
    df_list.append(pivoted_df)

stats_df = aggregate_pivoted_dataframes(df_list)
stats_df.to_csv(os.path.join(DEST_FOLDER, 'summary.csv'), index=False)
stats_df.head()


# %%
plot_mean_and_std(stats_df)
# %%
