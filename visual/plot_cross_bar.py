import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Set academic style
sns.set_theme(style="whitegrid", context="paper", font_scale=1.8)
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Times New Roman', 'DejaVu Serif', 'serif']
plt.rcParams['axes.labelsize'] = 22
plt.rcParams['axes.titlesize'] = 24
plt.rcParams['xtick.labelsize'] = 18
plt.rcParams['ytick.labelsize'] = 18
plt.rcParams['legend.fontsize'] = 20
plt.rcParams['axes.edgecolor'] = 'black'
plt.rcParams['axes.linewidth'] = 1.5

def plot_cross_bar():
    df = pd.read_csv('/root/autodl-tmp/rl4co-urban/visual/cross_clean.csv')
    
    # Create a composite label for X-axis
    # e.g., "Nairobi->Berlin\n(N=20)"
    # Shorten names for readability
    def shorten_transfer(t):
        t = t.replace("Nairobi", "N").replace("Berlin", "B")
        t = t.replace("Prodhon", "P").replace("Barreto", "B")
        return t
    
    df['Transfer_Short'] = df['Transfer'].apply(shorten_transfer)
    df['Scenario'] = df['Transfer_Short'] + "\n" + df['Size']
    
    # Melt for plotting
    df_long = pd.melt(
        df, 
        id_vars=['Task', 'Scenario', 'Gain'], 
        value_vars=['STS (SFT)', 'STS (Cross)'],
        var_name='Model', 
        value_name='Value'
    )
    
    # Define colors
    # SFT (Baseline) -> Grey/Blue
    # Cross (Ours) -> Red (Highlighted)
    palette = {
        'STS (SFT)': '#90A4AE',  # Blue Grey (Muted)
        'STS (Cross)': '#E31A1C' # Bright Red
    }
    
    tasks = ['TDTSP-TW', 'TDVRP-TW', 'CLRP']
    
    fig, axes = plt.subplots(1, 3, figsize=(24, 8), constrained_layout=True)
    
    for i, task in enumerate(tasks):
        ax = axes[i]
        subset = df_long[df_long['Task'] == task]
        subset_gain = df[df['Task'] == task] # Original df for gain values
        
        # Plot
        sns.barplot(
            data=subset,
            x='Scenario',
            y='Value',
            hue='Model',
            palette=palette,
            ax=ax,
            edgecolor='black',
            linewidth=1.5,
            width=0.6
        )
        
        # Add Gain Annotations
        # We want to place the gain text above the bars for each scenario
        # Since there are 2 bars per scenario, we place it centered above the pair or above the Cross bar
        
        # Get x-coordinates of bars
        # Seaborn/Matplotlib barplot places bars at integer indices +/- width/2
        # But accessing patches is easier.
        
        # Iterate over scenarios (0, 1, 2, 3)
        scenarios = subset['Scenario'].unique()
        for idx, scenario in enumerate(scenarios):
            row = subset_gain[subset_gain['Scenario'] == scenario].iloc[0]
            gain_txt = row['Gain']
            val_sft = row['STS (SFT)']
            val_cross = row['STS (Cross)']
            
            # Max height for placement
            max_h = max(val_sft, val_cross)
            
            # Place text
            ax.text(
                idx, 
                max_h * 1.05, 
                f"Gain\n{gain_txt}", 
                ha='center', 
                va='bottom', 
                fontsize=16, 
                fontweight='bold',
                color='#E31A1C'
            )
            
        # Increase Y-limit to fit annotations
        y_min, y_max = ax.get_ylim()
        ax.set_ylim(0, y_max * 1.2)
        
        # Formatting
        ax.set_title(task, pad=20, fontweight='bold')
        ax.set_xlabel("")
        
        if i == 0:
            ax.set_ylabel("Objective Value")
        else:
            ax.set_ylabel("")
            
        # Log scale might be needed if values vary wildly?
        # TDTSP is small (4-10), others are large (300-2700).
        # But each subplot has independent scale, so linear is fine.
        
        # Spines
        for spine in ax.spines.values():
            spine.set_edgecolor('black')
            spine.set_linewidth(1.5)
            
        # Legend only on first plot
        if i == 0:
            ax.legend(loc='upper right', frameon=False)
        else:
            if ax.get_legend():
                ax.legend_.remove()

    output_path = '/root/autodl-tmp/rl4co-urban/visual/cross_bar_plot.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Plot saved to {output_path}")

if __name__ == "__main__":
    plot_cross_bar()
