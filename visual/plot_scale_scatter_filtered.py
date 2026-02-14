
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import re
import matplotlib.patches as mpatches

# Set academic style
sns.set_theme(style="whitegrid", context="paper", font_scale=1.5)
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Times New Roman', 'DejaVu Serif', 'serif']
plt.rcParams['axes.labelsize'] = 16
plt.rcParams['axes.titlesize'] = 18
plt.rcParams['xtick.labelsize'] = 14
plt.rcParams['ytick.labelsize'] = 14
plt.rcParams['legend.fontsize'] = 13
plt.rcParams['lines.linewidth'] = 2.5
plt.rcParams['lines.markersize'] = 12

def parse_and_plot_filtered_scatter():
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
    
    # Pivot
    df_pivot = df_long.pivot_table(
        index=['Model', 'City', 'Size'], 
        columns='Metric', 
        values='Value'
    ).reset_index()
    
    cities = df_pivot['City'].unique()
    
    # Create figure
    fig, axes = plt.subplots(1, 2, figsize=(20, 9), constrained_layout=True)
    
    # Define Custom Palette for remaining models
    unique_models = sorted(df_pivot['Model'].unique())
    
    # Custom colors mapping
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
    
    # Fallback
    default_palette = sns.color_palette("bright", n_colors=10)
    default_markers = ['P', 'X', '*', '<']
    
    for i, model in enumerate(unique_models):
        if model not in color_map:
            color_map[model] = default_palette[i % len(default_palette)]
        if model not in marker_map:
            marker_map[model] = default_markers[i % len(default_markers)]
            
    # Short label map
    def format_size(s):
        if s >= 1000:
            return f"{s//1000}k"
        return str(s)

    for i, city in enumerate(cities):
        ax = axes[i]
        subset = df_pivot[df_pivot['City'] == city]
        
        # Draw connecting lines
        for model in unique_models:
            model_data = subset[subset['Model'] == model].sort_values(by="Size")
            if not model_data.empty:
                ax.plot(
                    model_data['Time'], 
                    model_data['Obj'], 
                    color=color_map[model],
                    linewidth=2,
                    alpha=0.4,
                    linestyle='-',
                    zorder=1
                )
        
        # Scatter plot
        sns.scatterplot(
            data=subset,
            x='Time',
            y='Obj',
            hue='Model',
            style='Model',
            size='Size',
            sizes=(100, 600), 
            palette=color_map,
            markers=marker_map,
            ax=ax,
            alpha=0.9,
            edgecolor='black',
            linewidth=1,
            zorder=2,
            legend=False 
        )
        
        # Add TEXT ANNOTATIONS for Size
        for idx, row in subset.iterrows():
            if pd.isna(row['Time']) or pd.isna(row['Obj']):
                continue
                
            label = format_size(row['Size'])
            
            # Use offset
            xytext = (5, 5)
            
            txt = ax.annotate(
                label, 
                xy=(row['Time'], row['Obj']),
                xytext=xytext, 
                textcoords='offset points',
                fontsize=10, 
                fontweight='bold',
                color='black',
                ha='left',
                va='bottom'
            )
            # Add white halo/outline
            txt.set_bbox(dict(facecolor='white', alpha=0.5, edgecolor='none', boxstyle='round,pad=0.1'))

        # Scales
        ax.set_xscale('log')
        ax.set_yscale('log')
        
        # Titles and Labels
        ax.set_title(f"{city}", fontweight='bold', pad=15)
        ax.set_xlabel("Time (s) [Log Scale]", fontweight='bold')
        
        if i == 0:
            ax.set_ylabel("Objective Value [Log Scale]", fontweight='bold')
        else:
            ax.set_ylabel("") 
            
        # Grid
        ax.grid(True, which="major", ls="-", alpha=0.4, color='gray')
        ax.grid(True, which="minor", ls=":", alpha=0.2, color='gray')
        
        # Arrow annotation
        ax.annotate(
            "Better", 
            xy=(0.05, 0.05), xycoords='axes fraction',
            xytext=(0.15, 0.15), textcoords='axes fraction',
            arrowprops=dict(facecolor='black', shrink=0.05, alpha=0.7),
            fontsize=14, fontweight='bold', color='black', alpha=0.7
        )

    # Legends
    # Model Legend
    model_handles = []
    for model in unique_models:
        h = plt.Line2D(
            [], [], 
            color=color_map[model], 
            marker=marker_map[model], 
            linestyle='', 
            markersize=12, 
            label=model,
            markeredgecolor='black'
        )
        model_handles.append(h)
        
    unique_sizes = sorted(df_pivot['Size'].unique())
    size_handles = []
    min_size, max_size = min(unique_sizes), max(unique_sizes)
    
    for s in unique_sizes:
        ms = np.sqrt(100 + (s - min_size) / (max_size - min_size) * 500)
        # Convert matplotlib markersize (points) to scatter size (area) roughly
        # This is purely for legend display
        
        h = plt.Line2D(
            [], [], 
            color='white', 
            marker='o', 
            markerfacecolor='gray',
            linestyle='', 
            markersize=np.sqrt(ms)*1.5 if ms < 50 else np.sqrt(ms), # heuristic adjustment for visual match
            label=format_size(s),
            markeredgecolor='black'
        )
        # Actually standard markersize in Line2D is diameter in points
        # In scatter, s is area in points^2
        # ms above is area. so sqrt(ms) is diameter.
        
        h = plt.Line2D(
             [], [],
             color='white',
             marker='o',
             markerfacecolor='gray',
             linestyle='',
             markersize=np.sqrt(100 + (s - min_size) / (max_size - min_size) * 500), # Correct scale matching scatter
             label=format_size(s),
             markeredgecolor='black'
        )
        size_handles.append(h)
        
    leg1 = fig.legend(
        handles=model_handles, 
        title="Models", 
        bbox_to_anchor=(1.08, 0.85), 
        loc='upper right',
        frameon=False,
        title_fontsize=16
    )
    
    fig.legend(
        handles=size_handles, 
        title="Size (N)", 
        bbox_to_anchor=(1.08, 0.35), 
        loc='upper right',
        frameon=False,
        title_fontsize=16,
        labelspacing=1.5
    )
    
    fig.add_artist(leg1)
    
    plt.suptitle("Scalability Analysis: Objective vs Time (Selected Models)", fontsize=22, fontweight='bold', y=1.05)
    
    output_path = '/root/autodl-tmp/rl4co-urban/visual/scale_scatter_filtered.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Filtered scatter plot saved to {output_path}")

if __name__ == "__main__":
    parse_and_plot_filtered_scatter()
