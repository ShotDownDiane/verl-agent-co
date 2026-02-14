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

def plot_ablation_bar():
    df = pd.read_csv('/root/autodl-tmp/rl4co-urban/visual/ablation_clean.csv')
    
    # Filter out STS (Baseline, 0% drop) if desired, or keep it to show 0 reference.
    # Usually ablation plots exclude the baseline or show it as 0.
    # Let's keep it but maybe it will just be empty bars.
    # Actually, let's exclude STS to focus on the impact of removing components.
    df = df[df['Variant'] != 'STS']
    
    # Create the plot
    plt.figure(figsize=(14, 8))
    
    # Define colors for N=20 and N=50
    # Maybe Blue for N=20, Red for N=50? Or a nice palette.
    palette = sns.color_palette("muted")
    
    # Grouped bar plot
    ax = sns.barplot(
        data=df,
        x='Variant',
        y='Drop_Value',
        hue='Size',
        palette="Blues_d", # Different shades of blue? Or maybe just 'hue' default.
        edgecolor='black',
        linewidth=1.5
    )
    
    # Add annotations
    for p in ax.patches:
        height = p.get_height()
        if height > 0:
            ax.annotate(f'{height:.0f}%',
                        (p.get_x() + p.get_width() / 2., height),
                        ha='center', va='bottom',
                        xytext=(0, 5),
                        textcoords='offset points',
                        fontsize=14,
                        fontweight='bold')

    # Formatting
    plt.title("Ablation Study: Performance Drop", fontweight='bold', pad=20)
    plt.xlabel("") # Variant names are self-explanatory
    plt.ylabel("Performance Drop (%)")
    plt.ylim(0, df['Drop_Value'].max() * 1.15) # Add space for annotations
    
    # Legend
    plt.legend(title="", loc='upper right', frameon=False)
    
    # Spines
    for spine in plt.gca().spines.values():
        spine.set_edgecolor('black')
        spine.set_linewidth(1.5)
        
    output_path = '/root/autodl-tmp/rl4co-urban/visual/ablation_bar_plot.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Plot saved to {output_path}")

if __name__ == "__main__":
    plot_ablation_bar()
