
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

def parse_and_plot_labels():
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
    
    # Pivot
    df_pivot = df_long.pivot_table(
        index=['Model', 'City', 'Size'], 
        columns='Metric', 
        values='Value'
    ).reset_index()
    
    cities = df_pivot['City'].unique()
    
    # Create figure
    fig, axes = plt.subplots(1, 2, figsize=(20, 9), constrained_layout=True)
    
    # Define palette and markers
    unique_models = df_pivot['Model'].unique()
    palette = sns.color_palette("bright", n_colors=len(unique_models))
    markers = ['o', 's', '^', 'D', 'v', 'P', 'X', '*', '<', '>']
    if len(unique_models) > len(markers):
        markers = markers * (len(unique_models) // len(markers) + 1)
    markers = markers[:len(unique_models)]
    
    model_markers = {model: markers[i] for i, model in enumerate(unique_models)}
    model_colors = {model: palette[i] for i, model in enumerate(unique_models)}
    
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
                    color=model_colors[model],
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
            palette=model_colors,
            markers=model_markers,
            ax=ax,
            alpha=0.9,
            edgecolor='black',
            linewidth=1,
            zorder=2,
            legend=False 
        )
        
        # Add TEXT ANNOTATIONS for Size
        # Iterate over points to label them
        for idx, row in subset.iterrows():
            if pd.isna(row['Time']) or pd.isna(row['Obj']):
                continue
                
            label = format_size(row['Size'])
            
            # Offset logic: 
            # Default offset to the right
            xytext = (5, 5)
            ha = 'left'
            va = 'bottom'
            
            # Adjust based on position if needed, but standard offset is usually fine 
            # unless crowded. 
            # Let's use a small white outline for text to make it readable over lines
            
            txt = ax.annotate(
                label, 
                xy=(row['Time'], row['Obj']),
                xytext=xytext, 
                textcoords='offset points',
                fontsize=10, 
                fontweight='bold',
                color='black',
                ha=ha,
                va=va
            )
            # Add white halo/outline to text
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
            color=model_colors[model], 
            marker=model_markers[model], 
            linestyle='', 
            markersize=12, 
            label=model,
            markeredgecolor='black'
        )
        model_handles.append(h)
        
    # Size Legend - Optional now since we have text, but good to keep for reference
    # Or maybe remove it to declutter since we have explicit labels?
    # User said "distinguish size is difficult", so explicit labels solve it. 
    # But let's keep the legend to explain the bubble size meaning if they look at it globally.
    
    unique_sizes = sorted(df_pivot['Size'].unique())
    size_handles = []
    min_size, max_size = min(unique_sizes), max(unique_sizes)
    
    for s in unique_sizes:
        ms = np.sqrt(100 + (s - min_size) / (max_size - min_size) * 500)
        h = plt.Line2D(
            [], [], 
            color='white', 
            marker='o', 
            markerfacecolor='gray',
            linestyle='', 
            markersize=ms, 
            label=format_size(s), # Match text labels
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
    
    plt.suptitle("Scalability Analysis: Objective vs Time", fontsize=22, fontweight='bold', y=1.05)
    
    output_path = '/root/autodl-tmp/rl4co-urban/visual/scale_scatter_labeled.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Labeled scatter plot saved to {output_path}")

if __name__ == "__main__":
    parse_and_plot_labels()
