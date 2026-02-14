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
plt.rcParams['xtick.labelsize'] = 20
plt.rcParams['ytick.labelsize'] = 20
plt.rcParams['legend.fontsize'] = 20
plt.rcParams['axes.edgecolor'] = 'black'
plt.rcParams['axes.linewidth'] = 1.5

def plot_ablation_drop():
    df = pd.read_csv('/root/autodl-tmp/rl4co-urban/visual/ablation_clean.csv')
    
    # Filter out STS (Baseline, 0% drop) if desired, or keep it to show 0?
    # Usually "Drop" implies performance drop compared to STS.
    # The user said "drop bar only chat".
    # Let's keep all except STS if it's 0, or show it as reference?
    # If STS drop is *, it's 0. Showing it might be redundant but good for reference.
    # Let's remove STS from the plot as "Drop" is relative to it.
    
    df_plot = df[df['Variant'] != 'STS'].copy()
    
    # Define colors
    # We have N=20 and N=50. Grouped bar chart.
    palette = sns.color_palette("Blues_d", n_colors=2)
    
    fig, ax = plt.subplots(figsize=(14, 8), constrained_layout=True)
    
    # Bar Plot
    sns.barplot(
        data=df_plot,
        x='Variant',
        y='Drop_Value',
        hue='Size',
        palette='viridis', # Distinct colors for sizes
        ax=ax,
        edgecolor='black',
        linewidth=1.5,
        width=0.7
    )
    
    # Add Value Annotations
    for container in ax.containers:
        ax.bar_label(container, fmt='%.0f%%', fontsize=18, padding=3, fontweight='bold')
        
    # Formatting
    ax.set_ylabel("Performance Drop (%)", fontweight='bold')
    ax.set_xlabel("") # Variant names are self-explanatory
    ax.set_title("Ablation Study: Performance Drop", pad=20, fontweight='bold')
    
    # Spines
    for spine in ax.spines.values():
        spine.set_edgecolor('black')
        spine.set_linewidth(1.5)
        
    # Legend
    ax.legend(title='Problem Size', loc='upper left', frameon=False)
    
    output_path = '/root/autodl-tmp/rl4co-urban/visual/ablation_drop_plot.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Plot saved to {output_path}")

if __name__ == "__main__":
    plot_ablation_drop()
