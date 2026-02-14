
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import re
from matplotlib.ticker import ScalarFormatter, FuncFormatter

# Set academic style
sns.set_theme(style="whitegrid", context="paper", font_scale=1.5)
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Times New Roman', 'DejaVu Serif', 'serif']
plt.rcParams['axes.labelsize'] = 16
plt.rcParams['axes.titlesize'] = 18
plt.rcParams['xtick.labelsize'] = 14
plt.rcParams['ytick.labelsize'] = 14
plt.rcParams['legend.fontsize'] = 14
plt.rcParams['lines.linewidth'] = 2.5
plt.rcParams['lines.markersize'] = 10

def parse_and_plot_refined_lines():
    # Read raw data
    df_raw = pd.read_csv('/root/autodl-tmp/rl4co-urban/visual/scale.csv', header=None)
    
    models = df_raw.iloc[2:, 0].values
    structured_data = []
    current_city_size = None
    
    # Parse data
    for col_idx in range(1, df_raw.shape[1]):
        header1 = df_raw.iloc[0, col_idx]
        header2 = df_raw.iloc[1, col_idx]
        
        if pd.notna(header1) and str(header1).strip() != "":
            current_city_size = header1
            
        if not current_city_size:
            continue
            
        match = re.match(r"(.+)\s+\(N=(\d+)\)", current_city_size)
        if match:
            city = match.group(1).strip()
            size = int(match.group(2))
            metric = header2.strip()
            values = df_raw.iloc[2:, col_idx].values
            
            for i, model in enumerate(models):
                try:
                    val = float(values[i])
                except:
                    val = np.nan
                
                structured_data.append({
                    "Model": model,
                    "City": city,
                    "Size": size,
                    "Metric": metric,
                    "Value": val
                })
                
    df_long = pd.DataFrame(structured_data)
    
    # Plotting
    cities = df_long['City'].unique() # ['Berlin', 'Nairobi']
    metrics = ['Obj', 'Time']
    
    # Create figure with 2 rows, 2 columns
    fig, axes = plt.subplots(2, 2, figsize=(16, 12), constrained_layout=True)
    
    # Define markers and palette
    unique_models = df_long['Model'].unique()
    palette = sns.color_palette("bright", n_colors=len(unique_models))
    markers = ['o', 's', '^', 'D', 'v', 'P', 'X', '*', '<', '>']
    if len(unique_models) > len(markers):
        markers = markers * (len(unique_models) // len(markers) + 1)
    markers = markers[:len(unique_models)]
    
    model_style = {model: {'color': palette[i], 'marker': markers[i]} for i, model in enumerate(unique_models)}
    
    # Common X-axis ticks
    x_ticks = [100, 200, 500, 1000, 2000]
    
    for row_idx, city in enumerate(cities):
        for col_idx, metric in enumerate(metrics):
            ax = axes[row_idx, col_idx]
            
            subset = df_long[(df_long['City'] == city) & (df_long['Metric'] == metric)]
            subset = subset.sort_values(by="Size")
            
            # Plot each model
            for model in unique_models:
                model_data = subset[subset['Model'] == model]
                if model_data.empty:
                    continue
                    
                ax.plot(
                    model_data['Size'], 
                    model_data['Value'], 
                    label=model,
                    color=model_style[model]['color'],
                    marker=model_style[model]['marker'],
                    linestyle='-',
                    alpha=0.8
                )
            
            # Formatting
            ax.set_title(f"{city} - {metric}", fontweight='bold')
            
            # X-axis: Use Log scale for Size to space out 100...2000 nicely
            ax.set_xscale('log')
            ax.set_xticks(x_ticks)
            ax.get_xaxis().set_major_formatter(ScalarFormatter()) # No scientific notation for axis labels
            ax.set_xlabel("Problem Size (N)", fontweight='bold')
            
            # Y-axis
            # Time usually needs Log scale. Obj might be linear or log.
            # Let's check range ratio
            v_min = subset['Value'].min()
            v_max = subset['Value'].max()
            
            if metric == 'Time':
                ax.set_ylabel("Time (s)", fontweight='bold')
                ax.set_yscale('log')
            else:
                ax.set_ylabel("Objective Value", fontweight='bold')
                # If range is huge, use log, otherwise linear
                if v_max / v_min > 50:
                    ax.set_yscale('log')
                else:
                    # Use scientific notation for large Obj values
                    formatter = ScalarFormatter(useMathText=True)
                    formatter.set_powerlimits((-2, 3))
                    ax.yaxis.set_major_formatter(formatter)

            ax.grid(True, which='major', linestyle='-', alpha=0.5)
            ax.grid(True, which='minor', linestyle=':', alpha=0.3)
            
    # Extract handles and labels for a global legend
    handles, labels = axes[0,0].get_legend_handles_labels()
    
    # Place legend at the bottom of the figure
    fig.legend(
        handles, labels, 
        loc='lower center', 
        bbox_to_anchor=(0.5, -0.05),
        ncol=len(unique_models)//2 + 1, # Multi-column legend
        frameon=False,
        fontsize=16
    )
    
    plt.suptitle("Scalability Analysis: Problem Size vs Performance", fontsize=22, fontweight='bold')
    
    output_path = '/root/autodl-tmp/rl4co-urban/visual/scale_plot_refined.png'
    # Use bbox_inches='tight' to include external legend
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Refined line plot saved to {output_path}")

if __name__ == "__main__":
    parse_and_plot_refined_lines()
