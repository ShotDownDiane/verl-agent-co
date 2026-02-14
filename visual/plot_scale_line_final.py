
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import re
from matplotlib.ticker import ScalarFormatter

# Set academic style with TIMES NEW ROMAN and LARGER FONTS
sns.set_theme(style="whitegrid", context="paper", font_scale=1.5)
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Times New Roman', 'DejaVu Serif', 'serif']
plt.rcParams['axes.labelsize'] = 22
plt.rcParams['axes.titlesize'] = 24
plt.rcParams['xtick.labelsize'] = 18
plt.rcParams['ytick.labelsize'] = 18
plt.rcParams['legend.fontsize'] = 18
plt.rcParams['lines.linewidth'] = 3.0
plt.rcParams['lines.markersize'] = 14
plt.rcParams['axes.edgecolor'] = 'black'
plt.rcParams['axes.linewidth'] = 1.5

def parse_and_plot_final_line():
    # Read raw data
    df_raw = pd.read_csv('/root/autodl-tmp/rl4co-urban/visual/scale.csv', header=None)
    
    models = df_raw.iloc[2:, 0].values
    structured_data = []
    current_city_size = None
    
    # Models to KEEP
    keep_models = ['SAH*', 'ALNS*', 'STS']
    
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
                if model not in keep_models:
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
    fig, axes = plt.subplots(2, 2, figsize=(10, 6), constrained_layout=True)
    
    unique_models = sorted(df_long['Model'].unique())
    
    # Custom colors mapping
    # STS: Bright Red
    # OR-Tools, SAH*, ALNS*: Distinct Muted
    # Others (ACO, GRASP, SAH, ALNS): Unified Cool Greys (Low variation, low prominence)
    color_map = {
        'OR-T': '#B0B0B0',   # Muted Grey (Reference)
        'SAH*': '#8FBC8F',       # Muted Green
        'ALNS*': '#F4A460',      # Muted Orange
        'STS': '#E31A1C',   # Bright Red
        
        # Unified Cool Grey Palette for background models
        'ACO': '#546E7A',        # Blue Grey 600
        'GRA': '#78909C',      # Blue Grey 400
        'SAH': '#90A4AE',        # Blue Grey 300
        'ALNS': '#B0BEC5'        # Blue Grey 200
    }
    
    marker_map = {
        'OR-T': 'o',
        'SAH*': 's',
        'ALNS*': '^',
        'STS': 'D',
        'ACO': 'v',
        'GRA': '<',
        'SAH': '>',
        'ALNS': 'p'
    }
    
    # Fallback
    default_palette = sns.color_palette("muted", n_colors=10)
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
            
            # Identify reference models for scaling and limits
            reference_models = ['SAH*', 'ALNS*', 'STS']
            subset_ref = subset[subset['Model'].isin(reference_models)]
            
            # Determine Scale based on Reference Models ONLY
            v_min_ref = subset_ref['Value'].min()
            v_max_ref = subset_ref['Value'].max()
            
            # Custom Layout:
            # - Remove individual subplot titles
            # - Top Row: Column Titles (Metric Name)
            # - Left Column: Row Labels (City Name) on Y-axis
            
            # Set Column Titles (only for top row)
            # if row_idx == 0:
                # col_title = "Time (s)" if metric == 'Time' else "Objective Value"
                # ax.set_title(col_title, pad=20)
            
            # Set Y-Axis Labels
            if col_idx == 0:
                # First column: Show City Name as Y-label
                ax.set_ylabel(city+f" Obj")
            else:
                # Other columns: No Y-label (units implied by column title or redundant)
                ax.set_ylabel(city+f" Time (s)")

            # Apply conditional log scale logic (using reference stats)
            if v_min_ref > 0 and v_max_ref / v_min_ref > 50:
                ax.set_yscale('log')
            else:
                formatter = ScalarFormatter(useMathText=True)
                formatter.set_powerlimits((-2, 3))
                ax.yaxis.set_major_formatter(formatter)

            # 1. Plot Reference Models First to establish limits
            for model in unique_models:
                if model not in reference_models:
                    continue
                    
                model_data = subset[subset['Model'] == model]
                if model_data.empty:
                    continue
                
                # Highlight STS
                is_sts = (model == 'STS')
                lw = 4.5 if is_sts else 3.0
                alpha = 0.9 if is_sts else 0.6
                ms = 16 if is_sts else 14
                zorder = 10 if is_sts else 5
                
                ax.plot(
                    model_data['Size'], 
                    model_data['Value'], 
                    label=model,
                    color=color_map[model],
                    marker=marker_map[model],
                    linestyle='-',
                    linewidth=lw,
                    markersize=ms,
                    alpha=alpha,
                    zorder=zorder
                )
            
            # Capture the Y-limits established by reference models
            y_limits = ax.get_ylim()
            
            # 2. Plot Other Models (ACO, GRASP, etc.)
            for model in unique_models:
                if model in reference_models:
                    continue
                    
                model_data = subset[subset['Model'] == model]
                if model_data.empty:
                    continue
                
                # Low prominence styling for others
                ax.plot(
                    model_data['Size'], 
                    model_data['Value'], 
                    label=model,
                    color=color_map[model],
                    marker=marker_map[model],
                    linestyle='--', # Dashed for differentiation
                    linewidth=2.0,  # Standard width
                    markersize=12,  # Standard size
                    alpha=0.6,      # Reduced alpha for lower prominence
                    zorder=4
                )
            
            # Restore the limits so "exceeding parts are not displayed"
            ax.set_ylim(y_limits)
            
            # ax.set_title(f"{city} - {metric}", pad=15)
            
            # X-Axis settings
            ax.set_xscale('log')
            ax.set_xticks(x_ticks)
            # ax.get_xaxis().set_major_formatter(ScalarFormatter())
            
            # Only show X-axis labels on the bottom row
            if row_idx == len(cities) - 1:
                ax.set_xticklabels(['100', '200', '500', '1k', '2k'])
                ax.set_xlabel("Problem Size (N)")
            else:
                ax.set_xticklabels([])
                ax.set_xlabel("")
                
            ax.set_xlim(100, 2000) # Tight X-axis as requested ("directly on coordinate axis")
            
            # Grid settings
            ax.grid(True, which='major', linestyle='-', alpha=0.4, color='gray')
            ax.grid(True, which='minor', linestyle=':', alpha=0.2, color='gray')
            
            # Ensure all spines are black and visible
            for spine in ax.spines.values():
                spine.set_edgecolor('black')
                spine.set_linewidth(1.5)
            
    # Create Custom Legend Order
    # Desired order: ORTools, GRASP, SAH, SAH*, ALNS, ALNS*, ACO, STS
    custom_order = [
        'SAH*',
        'ALNS*',
        'STS'
    ]
    
    handles = []
    # Use custom_order to iterate, ensuring the legend follows this exact sequence
    for model in custom_order:
        # Check if model exists in data to avoid errors if missing
        if model in unique_models:
            h = plt.Line2D(
                [], [], 
                color=color_map[model], 
                marker=marker_map[model], 
                linestyle='-', 
                markersize=16, 
                label=model,
                linewidth=3
            )
            handles.append(h)
    
    # Calculate number of columns for legend to force multiline
    # Aim for roughly 2 rows if many models
    n_models = len(unique_models)
    ncol = (n_models + 1) // 2 if n_models > 4 else n_models

    fig.legend(
        handles=handles,
        loc='lower center', 
        bbox_to_anchor=(0.5, 1.0), # Top
        ncol=4, # 4 columns
        frameon=False,
        fontsize=16,
        title_fontsize=16
    )

    # plt.suptitle("Scalability Analysis: Problem Size vs Performance", fontsize=28, fontweight='bold', y=1.02)
    
    output_path = '/root/autodl-tmp/rl4co-urban/visual/scale_plot_final.pdf'
    # Use bbox_inches='tight' to include external legend
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Final line plot saved to {output_path}")

if __name__ == "__main__":
    parse_and_plot_final_line()
