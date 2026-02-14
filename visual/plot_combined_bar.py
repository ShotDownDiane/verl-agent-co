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

def plot_combined_bar():
    # Setup Figure
    # Single row: 2 subplots for Cross-Task Transfer Analysis (TDTSP-TW, TDVRP-TW)
    
    fig, axes = plt.subplots(1, 2, figsize=(16, 6), constrained_layout=True)
    
    # ==========================================
    # Part (a): Cross-Task Transfer Analysis
    # ==========================================
    df_cross = pd.read_csv('/root/autodl-tmp/rl4co-urban/visual/cross_clean.csv')
    
    # Process Data
    def shorten_transfer(t):
        t = t.replace("Nairobi", "N").replace("Berlin", "B")
        t = t.replace("Prodhon", "P").replace("Barreto", "B")
        return t
    
    df_cross['Transfer_Short'] = df_cross['Transfer'].apply(shorten_transfer)
    df_cross['Scenario'] = df_cross['Transfer_Short'] + "\n" + df_cross['Size']
    
    df_long = pd.melt(
        df_cross, 
        id_vars=['Task', 'Scenario', 'Gain'], 
        value_vars=['STS (SFT)', 'STS (Cross)'],
        var_name='Model', 
        value_name='Value'
    )
    
    palette_cross = {
        'STS (SFT)': '#90A4AE',  # Blue Grey
        'STS (Cross)': '#E31A1C' # Bright Red
    }
    
    tasks = ['TDTSP-TW', 'TDVRP-TW']
    
    for i, task in enumerate(tasks):
        ax = axes[i]
        
        subset = df_long[df_long['Task'] == task]
        subset_gain = df_cross[df_cross['Task'] == task]
        
        # Plot
        sns.barplot(
            data=subset,
            x='Scenario',
            y='Value',
            hue='Model',
            palette=palette_cross,
            ax=ax,
            edgecolor='black',
            linewidth=1.5,
            width=0.6
        )
        
        # Annotations
        scenarios = subset['Scenario'].unique()
        for idx, scenario in enumerate(scenarios):
            row = subset_gain[subset_gain['Scenario'] == scenario].iloc[0]
            gain_txt = row['Gain']
            val_sft = row['STS (SFT)']
            val_cross = row['STS (Cross)']
            max_h = max(val_sft, val_cross)
            
            ax.text(
                idx, 
                max_h * 1.05, 
                f"Gain\n{gain_txt}", 
                ha='center', 
                va='bottom', 
                fontsize=20, 
                fontweight='bold',
                color='#E31A1C'
            )
            
        # Limits & Labels
        y_min, y_max = ax.get_ylim()
        ax.set_ylim(0, y_max * 1.25)
        ax.set_title(task, pad=20, fontweight='bold')
        ax.set_xlabel("")
        if i == 0:
            ax.set_ylabel("Objective Value")
        else:
            ax.set_ylabel("")
            
        # Legend (Only on first subplot)
        if i == 0:
            ax.legend(loc='upper right', frameon=False)
        else:
            if ax.get_legend():
                ax.legend_.remove()
                
        # Spines
        for spine in ax.spines.values():
            spine.set_edgecolor('black')
            spine.set_linewidth(1.5)

    # Save
    output_path = '/root/autodl-tmp/rl4co-urban/visual/combined_bar_plot.pdf'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Plot saved to {output_path}")

if __name__ == "__main__":
    plot_combined_bar()
