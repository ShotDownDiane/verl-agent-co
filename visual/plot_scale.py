
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import re

# Set academic style
sns.set_theme(style="whitegrid", context="paper", font_scale=1.2)
plt.rcParams['font.family'] = 'serif'
plt.rcParams['axes.labelsize'] = 12
plt.rcParams['axes.titlesize'] = 14
plt.rcParams['xtick.labelsize'] = 10
plt.rcParams['ytick.labelsize'] = 10
plt.rcParams['legend.fontsize'] = 10
plt.rcParams['lines.linewidth'] = 2
plt.rcParams['lines.markersize'] = 8

def parse_and_plot():
    # Read CSV with multi-level header
    # The first column is index (Model names)
    df = pd.read_csv('/root/autodl-tmp/rl4co-urban/visual/scale.csv', header=[0, 1], index_col=0)
    
    # Process columns to extract City, Size, Metric
    # Current columns are like: ('Berlin (N=100)', 'Obj'), ('Berlin (N=100)', 'Time'), ...
    # Note: Pandas might load empty cells in header 0 as Unnamed if they were empty in CSV
    # But based on my fix, I filled the CSV carefully? 
    # Actually, looking at the previous cat output:
    # ,Berlin (N=100),,Berlin (N=200),...
    # ,Obj,Time,Obj,Time,...
    # Pandas read_csv with header=[0,1] should handle the "empty" top cells by forward filling if using a specific option, 
    # but standard CSV doesn't imply merge.
    # However, let's inspect the columns after read.
    
    data = []
    
    # Iterate over columns
    # The dataframe columns will be a MultiIndex
    for (col_l1, col_l2) in df.columns:
        # col_l1 expected: "Berlin (N=100)" or "Unnamed: ..."
        # col_l2 expected: "Obj" or "Time"
        
        # If col_l1 is "Unnamed...", it might belong to the previous valid column if the CSV was merged cells.
        # But in standard CSV, it's just a separate column.
        # My previous fix wrote: ,Berlin (N=100),,Berlin (N=200)...
        # So column 1 is "Berlin (N=100)", column 2 is empty/unnamed.
        # Wait, the CSV I generated has:
        # [empty], Berlin(N=100), [empty], Berlin(N=200)...
        # [empty], Obj, Time, Obj, Time...
        # So index_col=0 takes the first column.
        # Then column 0 (originally 1) is "Berlin (N=100)" / "Obj"
        # Column 1 (originally 2) is "Unnamed..." / "Time" (because header 0 was empty for that col)
        pass

    # Re-reading to debug structure first might be safer, but let's implement robust logic.
    # We will iterate and keep track of the current "City + Size" context.
    
    # Re-read without header parsing to be safe and manually construct
    df_raw = pd.read_csv('/root/autodl-tmp/rl4co-urban/visual/scale.csv', header=None)
    
    # Row 0: City info
    # Row 1: Metric info
    # Row 2+: Data
    
    models = df_raw.iloc[2:, 0].values
    
    structured_data = []
    
    current_city_size = None
    
    for col_idx in range(1, df_raw.shape[1]):
        header1 = df_raw.iloc[0, col_idx]
        header2 = df_raw.iloc[1, col_idx]
        
        if pd.notna(header1) and str(header1).strip() != "":
            current_city_size = header1
            
        if not current_city_size:
            continue
            
        # Parse City and Size from "Berlin (N=100)"
        # Regex to capture City and Number
        match = re.match(r"(.+)\s+\(N=(\d+)\)", current_city_size)
        if match:
            city = match.group(1).strip()
            size = int(match.group(2))
            metric = header2.strip()
            
            # Extract values for all models
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
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # Define markers and palette
    unique_models = df_long['Model'].unique()
    palette = sns.color_palette("tab10", n_colors=len(unique_models))
    markers = ['o', 's', '^', 'v', 'D', 'p', '*', 'X', '<', '>']
    if len(unique_models) > len(markers):
        markers = markers * (len(unique_models) // len(markers) + 1)
    markers = markers[:len(unique_models)]
    
    model_style = {model: {'color': palette[i], 'marker': markers[i]} for i, model in enumerate(unique_models)}
    
    # We want Legend ONLY on one chart. Let's put it on the last one or collect handles.
    handles, labels = [], []
    
    # Order: Row 1 = Berlin, Row 2 = Nairobi
    # Col 1 = Obj, Col 2 = Time
    
    for row_idx, city in enumerate(cities):
        for col_idx, metric in enumerate(metrics):
            ax = axes[row_idx, col_idx]
            
            subset = df_long[(df_long['City'] == city) & (df_long['Metric'] == metric)]
            
            # Sort by Size
            subset = subset.sort_values(by="Size")
            
            # Plot each model
            for model in unique_models:
                model_data = subset[subset['Model'] == model]
                if model_data.empty:
                    continue
                    
                line, = ax.plot(
                    model_data['Size'], 
                    model_data['Value'], 
                    label=model,
                    color=model_style[model]['color'],
                    marker=model_style[model]['marker'],
                    markersize=8,
                    linewidth=2,
                    alpha=0.9
                )
                
                # Collect handles for legend (only once)
                if row_idx == 0 and col_idx == 0:
                    handles.append(line)
                    labels.append(model)
            
            # Formatting
            ax.set_title(f"{city} - {metric}", fontweight='bold')
            ax.set_xlabel("Problem Size (N)")
            ax.set_ylabel(metric)
            ax.set_xticks([100, 200, 500, 1000, 2000])
            ax.grid(True, which='both', linestyle='--', alpha=0.7)
            
            # Log scale for Time if values vary significantly?
            # Obj values: 1k to 70k -> Linear is ok, or Log.
            # Time values: 50s to 7000s -> Log scale is usually better for Time.
            # Let's verify range.
            if metric == 'Time':
                # Use log scale for Time to see differences at small N and large N
                # But user didn't strictly ask. Let's check min/max ratio.
                v_min = subset['Value'].min()
                v_max = subset['Value'].max()
                if v_max / v_min > 50:
                    ax.set_yscale('log')
                    ax.set_ylabel("Time (s) - Log Scale")
            
            # Obj scale might also benefit from log if it grows linearly with N (TSP length grows as sqrt(N)*N? No, sqrt(N)).
            # Actually TSP length grows as sqrt(N) * N_nodes? No.
            # For random uniform, Length ~ sqrt(N).
            # 2000 nodes vs 100 nodes => sqrt(20) ~ 4.5x.
            # Values are 2000 -> 28000 (approx 10x).
            # Linear scale is fine for Obj.
            
    # Add legend to the figure (global legend)
    # Or put it in the first subplot. User said "只在一张图上标注图例" (Label legend on only one chart).
    # I will put it on the first chart (Berlin - Obj) or outside.
    # Putting it inside the first chart might obscure data.
    # Putting it outside is "on one chart" broadly interpreted?
    # I will put it in the top-left chart, best location.
    
    axes[0, 0].legend(loc='upper left', frameon=True, framealpha=0.9, fontsize=9)
    
    plt.tight_layout()
    plt.savefig('/root/autodl-tmp/rl4co-urban/visual/scale_plot.png', dpi=300, bbox_inches='tight')
    print("Plot saved to /root/autodl-tmp/rl4co-urban/visual/scale_plot.png")

if __name__ == "__main__":
    parse_and_plot()
