# %%
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

# %%
def load_csv_data(csv_path):
    """
    Load CSV file containing eval_return data.
    
    Parameters:
        csv_path (str): Path to the CSV file.
    
    Returns:
        pd.DataFrame: DataFrame with the loaded data.
    """
    csv_path = Path(csv_path)
    
    if not csv_path.exists():
        raise ValueError(f"CSV file does not exist: {csv_path}")
    
    df = pd.read_csv(csv_path)
    
    # Validate required columns
    required_columns = ['subfolder_name', 'step_nr', 'eval_return']
    missing_columns = [col for col in required_columns if col not in df.columns]
    if missing_columns:
        raise ValueError(f"Missing required columns: {missing_columns}")
    
    return df


# %%
def plot_environment_comparison(df1, df2, environment, label1="Dataset 1", label2="Dataset 2"):
    """
    Plot comparison of eval_return for a single environment from two datasets.
    
    Parameters:
        df1 (pd.DataFrame): First dataset.
        df2 (pd.DataFrame): Second dataset.
        environment (str): Name of the environment (subfolder_name).
        label1 (str): Label for the first dataset (default: "Dataset 1").
        label2 (str): Label for the second dataset (default: "Dataset 2").
    
    Returns:
        matplotlib.figure.Figure: The created figure.
    """
    # Filter data for this environment
    env_data1 = df1[df1['subfolder_name'] == environment].copy()
    env_data2 = df2[df2['subfolder_name'] == environment].copy()
    
    # Sort by step_nr
    env_data1 = env_data1.sort_values('step_nr')
    env_data2 = env_data2.sort_values('step_nr')
    
    # Create the plot
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Plot both datasets
    if len(env_data1) > 0:
        ax.plot(env_data1['step_nr'], env_data1['eval_return'], 
                label=label1, marker='o', markersize=3, linewidth=1.5, alpha=0.7)
    
    if len(env_data2) > 0:
        ax.plot(env_data2['step_nr'], env_data2['eval_return'], 
                label=label2, marker='s', markersize=3, linewidth=1.5, alpha=0.7)
    
    # Set labels and title
    ax.set_xlabel('Step Number', fontsize=12)
    ax.set_ylabel('Eval Return', fontsize=12)
    ax.set_title(environment, fontsize=14, fontweight='bold')
    ax.legend(loc='best')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    return fig


# %%
def compare_results(csv_path1, csv_path2, label1="Dataset 1", label2="Dataset 2", 
                    save_plots=False, output_dir=None):
    """
    Compare eval_return results from two CSV files, creating plots for each environment.
    
    Parameters:
        csv_path1 (str): Path to the first CSV file.
        csv_path2 (str): Path to the second CSV file.
        label1 (str): Label for the first dataset (default: "Dataset 1").
        label2 (str): Label for the second dataset (default: "Dataset 2").
        save_plots (bool): Whether to save plots to files (default: False).
        output_dir (str): Directory to save plots if save_plots is True (default: None, uses current directory).
    
    Returns:
        dict: Dictionary mapping environment names to their matplotlib figures.
    """
    # Load both datasets
    print(f"Loading first dataset: {csv_path1}")
    df1 = load_csv_data(csv_path1)
    print(f"  Loaded {len(df1)} entries from {df1['subfolder_name'].nunique()} environments")
    
    print(f"\nLoading second dataset: {csv_path2}")
    df2 = load_csv_data(csv_path2)
    print(f"  Loaded {len(df2)} entries from {df2['subfolder_name'].nunique()} environments")
    
    # Get all unique environments from both datasets
    all_environments = set(df1['subfolder_name'].unique()) | set(df2['subfolder_name'].unique())
    all_environments = sorted(list(all_environments))
    
    print(f"\nFound {len(all_environments)} unique environments:")
    for env in all_environments:
        count1 = len(df1[df1['subfolder_name'] == env])
        count2 = len(df2[df2['subfolder_name'] == env])
        print(f"  - {env}: {count1} entries (dataset 1), {count2} entries (dataset 2)")
    
    # Create plots for each environment
    figures = {}
    print(f"\nCreating plots for {len(all_environments)} environments...")
    
    for environment in all_environments:
        print(f"  Plotting: {environment}")
        fig = plot_environment_comparison(df1, df2, environment, label1, label2)
        figures[environment] = fig
        
        # Save plot if requested
        if save_plots:
            if output_dir is None:
                output_dir = Path.cwd()
            else:
                output_dir = Path(output_dir)
                output_dir.mkdir(parents=True, exist_ok=True)
            
            # Sanitize environment name for filename
            safe_env_name = environment.replace('/', '_').replace('\\', '_')
            output_path = output_dir / f"{safe_env_name}_comparison.png"
            fig.savefig(output_path, dpi=150, bbox_inches='tight')
            print(f"    Saved to: {output_path}")
    
    print("\nComparison complete!")
    return figures


# %%
def display_all_plots(figures):
    """
    Display all plots. Useful for showing all figures at once.
    
    Parameters:
        figures (dict): Dictionary of environment names to matplotlib figures.
    """
    for environment, fig in figures.items():
        plt.figure(fig.number)
        plt.show()


# %%
# Example usage
# Set paths to CSV files
CSV_PATH_1 = '../data/dmc/dmc_kan_enc.csv'
CSV_PATH_2 = '../data/dmc/dmc_proprio.csv'

# Optional: Set labels for the datasets
LABEL_1 = 'KAN Encoder'
LABEL_2 = 'Proprio'

# Compare results and create plots
figures = compare_results(
    csv_path1=CSV_PATH_1,
    csv_path2=CSV_PATH_2,
    label1=LABEL_1,
    label2=LABEL_2,
    save_plots=False  # Set to True to save plots to files
)

# %%
# Display all plots
# Uncomment to show all plots at once
# display_all_plots(figures)

# %%
# Access individual plots if needed
# For example, to show a specific environment:
# plt.figure(figures['acrobot_swingup'].number)
# plt.show()

