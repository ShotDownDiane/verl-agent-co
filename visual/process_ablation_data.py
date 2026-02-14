import pandas as pd
import numpy as np

def clean_ablation_csv():
    input_path = '/root/autodl-tmp/rl4co-urban/visual/ablation.csv'
    output_path = '/root/autodl-tmp/rl4co-urban/visual/ablation_clean.csv'
    
    # Read the file
    # Assuming tab separated
    with open(input_path, 'r') as f:
        lines = f.readlines()
    
    # Parse lines
    # Row 0: Header (Variants, Berlin (N=20), Berlin (N=50))
    # Row 1: Subheader (Obj, Drop, Obj, Drop)
    # Rows 2-7: Data
    
    # Manual extraction
    data = []
    
    # Skip headers
    for line in lines[2:]:
        parts = line.strip().split('\t')
        if len(parts) < 5: continue
        
        variant = parts[0]
        # N=20
        obj_20 = float(parts[1])
        drop_20 = parts[2]
        # N=50
        obj_50 = float(parts[3])
        drop_50 = parts[4]
        
        data.append({
            "Variant": variant,
            "Size": "Berlin (N=20)",
            "Obj": obj_20,
            "Drop": drop_20
        })
        data.append({
            "Variant": variant,
            "Size": "Berlin (N=50)",
            "Obj": obj_50,
            "Drop": drop_50
        })
        
    df = pd.DataFrame(data)
    
    # Clean Drop values
    # Remove '%', convert to float. '*' becomes 0.
    def clean_drop(val):
        if val == '*': return 0.0
        val = val.replace('%', '').strip()
        try:
            return float(val)
        except:
            return 0.0
            
    df['Drop_Value'] = df['Drop'].apply(clean_drop)
    
    print("Cleaned Data:")
    print(df)
    
    df.to_csv(output_path, index=False)
    print(f"Saved to {output_path}")

if __name__ == "__main__":
    clean_ablation_csv()
