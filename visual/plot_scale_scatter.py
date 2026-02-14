
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import re

# Set academic style
sns.set_theme(style="whitegrid", context="paper", font_scale=1.4)
plt.rcParams['font.family'] = 'serif'
plt.rcParams['axes.labelsize'] = 14
plt.rcParams['axes.titlesize'] = 16
plt.rcParams['xtick.labelsize'] = 12
plt.rcParams['ytick.labelsize'] = 12
plt.rcParams['legend.fontsize'] = 12
plt.rcParams['lines.linewidth'] = 2
plt.rcParams['lines.markersize'] = 10

def parse_and_plot_scatter():
    # Read raw data
    df_raw = pd.read_csv('/root/autodl-tmp/rl4co-urban/visual/scale.csv', header=None)
    
    models = df_raw.iloc[2:, 0].values
    structured_data = []
    current_city_size = None
    
    # Parse data into long format first
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
    
    # df_pivot columns: Model, City, Size, Obj, Time
    
    # Plotting
    cities = df_pivot['City'].unique()
    fig, axes = plt.subplots(1, 2, figsize=(18, 8))
    
    # Define palette and markers
    unique_models = df_pivot['Model'].unique()
    palette = sns.color_palette("tab10", n_colors=len(unique_models))
    markers = ['o', 's', '^', 'v', 'D', 'p', '*', 'X', '<', '>']
    if len(unique_models) > len(markers):
        markers = markers * (len(unique_models) // len(markers) + 1)
    markers = markers[:len(unique_models)]
    
    # Map model to marker
    model_markers = {model: markers[i] for i, model in enumerate(unique_models)}
    
    for i, city in enumerate(cities):
        ax = axes[i]
        subset = df_pivot[df_pivot['City'] == city]
        
        # Scatter plot with seaborn
        # We use Time as X, Obj as Y
        # Hue = Model
        # Size = Size (to show N=100 vs N=2000)
        sns.scatterplot(
            data=subset,
            x='Time',
            y='Obj',
            hue='Model',
            style='Model', # Use different markers for models
            size='Size',   # Bubble size for problem size
            sizes=(50, 400), # Range of bubble sizes
            palette=palette,
            markers=model_markers,
            ax=ax,
            alpha=0.8,
            edgecolor='w',
            linewidth=0.5
        )
        
        # Optional: Connect points of the same model with lines to show trajectory
        for model in unique_models:
            model_data = subset[subset['Model'] == model].sort_values(by="Size")
            if not model_data.empty:
                ax.plot(
                    model_data['Time'], 
                    model_data['Obj'], 
                    color=palette[list(unique_models).index(model)],
                    linewidth=1.5,
                    alpha=0.5,
                    linestyle='--'
                )
        
        # Log scales are crucial here because N=2000 is huge compared to N=100
        ax.set_xscale('log')
        ax.set_yscale('log')
        
        ax.set_title(f"{city}: Objective vs Time (Log-Log Scale)", fontweight='bold')
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Objective Value")
        
        ax.grid(True, which="both", ls="--", alpha=0.3)
        
        # Legend handling
        # Remove individual legends to create a single global one or keep per plot?
        # User asked for "只在一张图上标注图例" (Label legend on only one chart) previously.
        # But for scatter with sizes, the legend is complex (Model + Size).
        # Let's keep legend on the second plot or first, but make it outside.
        if i == 0:
            # Get handles and labels
            handles, labels = ax.get_legend_handles_labels()
            ax.legend_.remove()
        else:
             # Put legend on the right of the second plot
             ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)

    # If we removed legend from first, we need to ensure the second one has everything.
    # Seaborn automatically adds it to both if not controlled.
    # The code above removes it from the first (i=0) and keeps it on the second (i=1) moved outside.
    
    plt.tight_layout()
    plt.savefig('/root/autodl-tmp/rl4co-urban/visual/scale_scatter.png', dpi=300, bbox_inches='tight')
    print("Scatter plot saved to /root/autodl-tmp/rl4co-urban/visual/scale_scatter.png")

if __name__ == "__main__":
    parse_and_plot_scatter()
