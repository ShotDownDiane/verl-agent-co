
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

def parse_and_plot_refined():
    # Read raw data
    df_raw = pd.read_csv('/root/autodl-tmp/rl4co-urban/visual/scale.csv', header=None)
    
    models = df_raw.iloc[2:, 0].values
    structured_data = []
    current_city_size = None
    
    # Parse data into long format
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
    
    # Pivot to get Obj and Time in same row
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
    # Use a high contrast palette
    palette = sns.color_palette("bright", n_colors=len(unique_models))
    # Make sure markers are distinct
    markers = ['o', 's', '^', 'D', 'v', 'P', 'X', '*', '<', '>']
    # If more models than markers, cycle
    if len(unique_models) > len(markers):
        markers = markers * (len(unique_models) // len(markers) + 1)
    markers = markers[:len(unique_models)]
    
    model_markers = {model: markers[i] for i, model in enumerate(unique_models)}
    model_colors = {model: palette[i] for i, model in enumerate(unique_models)}
    
    for i, city in enumerate(cities):
        ax = axes[i]
        subset = df_pivot[df_pivot['City'] == city]
        
        # Draw connecting lines first (so they are behind markers)
        for model in unique_models:
            model_data = subset[subset['Model'] == model].sort_values(by="Size")
            if not model_data.empty:
                ax.plot(
                    model_data['Time'], 
                    model_data['Obj'], 
                    color=model_colors[model],
                    linewidth=2,
                    alpha=0.6,
                    linestyle='-',
                    zorder=1
                )
        
        # Scatter plot
        # We handle legend manually for better control
        sns.scatterplot(
            data=subset,
            x='Time',
            y='Obj',
            hue='Model',
            style='Model',
            size='Size',
            sizes=(100, 600), # Larger bubbles
            palette=model_colors,
            markers=model_markers,
            ax=ax,
            alpha=0.9,
            edgecolor='black',
            linewidth=1,
            zorder=2,
            legend=False # Disable auto legend
        )
        
        # Scales
        ax.set_xscale('log')
        ax.set_yscale('log')
        
        # Titles and Labels
        ax.set_title(f"{city}", fontweight='bold', pad=15)
        ax.set_xlabel("Time (s) [Log Scale]", fontweight='bold')
        
        if i == 0:
            ax.set_ylabel("Objective Value [Log Scale]", fontweight='bold')
        else:
            ax.set_ylabel("") # Hide y label for second plot to save space
            
        # Grid
        ax.grid(True, which="major", ls="-", alpha=0.4, color='gray')
        ax.grid(True, which="minor", ls=":", alpha=0.2, color='gray')
        
        # Add "Better" arrow to bottom-left
        # Coordinates in axes fraction
        ax.annotate(
            "Better", 
            xy=(0.05, 0.05), xycoords='axes fraction',
            xytext=(0.15, 0.15), textcoords='axes fraction',
            arrowprops=dict(facecolor='black', shrink=0.05, alpha=0.7),
            fontsize=14, fontweight='bold', color='black', alpha=0.7
        )

    # Global Legend Construction
    # We want two legend sections: "Model" and "Size"
    
    # Model Legend Handles
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
        
    # Size Legend Handles
    # Use dummy points for size
    unique_sizes = sorted(df_pivot['Size'].unique())
    size_handles = []
    # Normalize sizes for display to match scatter sizes (approx mapping)
    # sizes=(100, 600)
    min_size, max_size = min(unique_sizes), max(unique_sizes)
    
    def get_marker_size(s):
        # Linear interpolation for display size matching scatterplot sizes
        return 10 + (s - min_size) / (max_size - min_size) * 15
        
    for s in unique_sizes:
        # For legend, we just use gray circles
        # Note: markersize in Line2D is points, scatter sizes is area (points^2)
        # Sqrt(100) = 10, Sqrt(600) ~ 24.5
        ms = np.sqrt(100 + (s - min_size) / (max_size - min_size) * 500)
        h = plt.Line2D(
            [], [], 
            color='white', 
            marker='o', 
            markerfacecolor='gray',
            linestyle='', 
            markersize=ms, 
            label=f"N={s}",
            markeredgecolor='black'
        )
        size_handles.append(h)
        
    # Add legends to the right of the figure
    # Create a new axis for legend or place on figure
    # We will place it on the right side of the second plot
    
    # Legend for Models
    leg1 = fig.legend(
        handles=model_handles, 
        title="Models", 
        bbox_to_anchor=(1.08, 0.85), 
        loc='upper right',
        frameon=False,
        title_fontsize=16
    )
    
    # Legend for Sizes
    fig.legend(
        handles=size_handles, 
        title="Problem Size", 
        bbox_to_anchor=(1.08, 0.35), 
        loc='upper right',
        frameon=False,
        title_fontsize=16,
        labelspacing=1.5
    )
    
    # Re-add leg1 because adding the second legend removes the first unless we add it back
    fig.add_artist(leg1)
    
    # Adjust layout to make room for legend
    # constrained_layout usually handles this if legend is part of a subplot, 
    # but here it is figure-level. 
    # We set bbox_to_anchor > 1, so we need to adjust right margin.
    # fig.subplots_adjust(right=0.85) # constrained_layout might conflict with subplots_adjust
    # Actually, constrained_layout=True in subplots() works well for internal spacing.
    # We might just rely on bbox_inches='tight' in savefig to include the legend.
    
    plt.suptitle("Scalability Analysis: Objective vs Time", fontsize=22, fontweight='bold', y=1.05)
    
    output_path = '/root/autodl-tmp/rl4co-urban/visual/scale_scatter_refined.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Refined scatter plot saved to {output_path}")

if __name__ == "__main__":
    parse_and_plot_refined()
