
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import re
from matplotlib.ticker import ScalarFormatter

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

def parse_and_plot_filtered():
    # Read raw data
    df_raw = pd.read_csv('/root/autodl-tmp/rl4co-urban/visual/scale.csv', header=None)
    
    models = df_raw.iloc[2:, 0].values
    structured_data = []
    current_city_size = None
    
    # Models to EXCLUDE
    excluded_models = ['ACO', 'GRASP', 'SAH', 'ALNS']
    
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
                if model in excluded_models:
                    continue
                    
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
    cities = df_long['City'].unique()
    metrics = ['Obj', 'Time']
    
    # Create figure
    fig, axes = plt.subplots(2, 2, figsize=(16, 12), constrained_layout=True)
    
    # Define Custom Palette for remaining models
    # Remaining: OR-Tools, SAH*, ALNS*, STS (50)
    unique_models = sorted(df_long['Model'].unique())
    
    # Custom colors mapping
    # OR-Tools: Grey (Baseline)
    # SAH*: Green
    # ALNS*: Orange
    # STS (50): Red/Purple (Ours?)
    
    color_map = {
        'OR-Tools': '#7f7f7f',   # Grey
        'SAH*': '#2ca02c',       # Green
        'ALNS*': '#ff7f0e',      # Orange
        'STS (50)': '#d62728'    # Red
    }
    
    marker_map = {
        'OR-Tools': 'o',
        'SAH*': 's',
        'ALNS*': '^',
        'STS (50)': 'D'
    }
    
    # Fallback if other models appear
    default_palette = sns.color_palette("bright", n_colors=10)
    default_markers = ['P', 'X', '*', '<']
    
    for i, model in enumerate(unique_models):
        if model not in color_map:
            color_map[model] = default_palette[i % len(default_palette)]
        if model not in marker_map:
            marker_map[model] = default_markers[i % len(default_markers)]
            
    x_ticks = [100, 200, 500, 1000, 2000]
    
    for row_idx, city in enumerate(cities):
        for col_idx, metric in enumerate(metrics):
            ax = axes[row_idx, col_idx]
            
            subset = df_long[(df_long['City'] == city) & (df_long['Metric'] == metric)]
            subset = subset.sort_values(by="Size")
            
            for model in unique_models:
                model_data = subset[subset['Model'] == model]
                if model_data.empty:
                    continue
                    
                ax.plot(
                    model_data['Size'], 
                    model_data['Value'], 
                    label=model,
                    color=color_map[model],
                    marker=marker_map[model],
                    linestyle='-',
                    alpha=0.8
                )
            
            ax.set_title(f"{city} - {metric}", fontweight='bold')
            ax.set_xscale('log')
            ax.set_xticks(x_ticks)
            ax.get_xaxis().set_major_formatter(ScalarFormatter())
            ax.set_xlabel("Problem Size (N)", fontweight='bold')
            
            v_min = subset['Value'].min()
            v_max = subset['Value'].max()
            
            if metric == 'Time':
                ax.set_ylabel("Time (s)", fontweight='bold')
                ax.set_yscale('log')
            else:
                ax.set_ylabel("Objective Value", fontweight='bold')
                if v_max / v_min > 50:
                    ax.set_yscale('log')
                else:
                    formatter = ScalarFormatter(useMathText=True)
                    formatter.set_powerlimits((-2, 3))
                    ax.yaxis.set_major_formatter(formatter)

            ax.grid(True, which='major', linestyle='-', alpha=0.5)
            ax.grid(True, which='minor', linestyle=':', alpha=0.3)
            
    handles = []
    for model in unique_models:
        h = plt.Line2D(
            [], [], 
            color=color_map[model], 
            marker=marker_map[model], 
            linestyle='-', 
            markersize=10, 
            label=model
        )
        handles.append(h)
    
    fig.legend(
        handles=handles,
        loc='lower center', 
        bbox_to_anchor=(0.5, -0.05),
        ncol=len(unique_models),
        frameon=False,
        fontsize=16
    )
    
    plt.suptitle("Scalability Analysis (Selected Models)", fontsize=22, fontweight='bold')
    
    output_path = '/root/autodl-tmp/rl4co-urban/visual/scale_plot_filtered.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Filtered line plot saved to {output_path}")

if __name__ == "__main__":
    parse_and_plot_filtered()
