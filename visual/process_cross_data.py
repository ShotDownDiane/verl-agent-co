import pandas as pd
import numpy as np

def clean_cross_csv():
    input_path = '/root/autodl-tmp/rl4co-urban/visual/cross.csv'
    output_path = '/root/autodl-tmp/rl4co-urban/visual/cross_clean.csv'
    
    # Read the file
    # Assuming tab separated based on visual inspection
    with open(input_path, 'r') as f:
        lines = f.readlines()
    
    # Parse header rows manually to handle merged cell logic
    # Row 0: Task (TDTSP-TW, TDVRP-TW, CLRP)
    # Row 1: Transfer (Nairobi->Berlin, etc)
    # Row 2: Size (berlin-20, etc)
    
    # Use rstrip() to avoid removing leading tabs which are crucial for alignment
    row0 = lines[0].rstrip('\n').split('\t')
    row1 = lines[1].rstrip('\n').split('\t')
    row2 = lines[2].rstrip('\n').split('\t')
    row3 = lines[3].rstrip('\n').split('\t') # STS (Cross)
    row4 = lines[4].rstrip('\n').split('\t') # STS (SFT)
    row5 = lines[5].rstrip('\n').split('\t') # gain
    
    # Clean empty strings from split
    # Logic: The header structure implies merged cells.
    # TDTSP-TW is likely applicable to first 4 data columns (indices 1-4 if index 0 is label)
    # Let's align them.
    
    # Function to fill forward empty values in list (merged cells behavior)
    def fill_forward(lst, expected_length):
        new_lst = []
        current = ""
        # The input list might be shorter or contain empty strings for merged cells
        # We need to map them to the data columns.
        # Data columns start at index 1 (index 0 is row label)
        # Let's look at the raw line split.
        
        # Raw split analysis:
        # Line 0: ['', 'TDTSP-TW', '', '', '', 'TDVRP-TW', '', '', '', 'CLRP', '', '', ''] 
        # (Assuming tab after label)
        
        return lst

    # Parse strictly based on column indices of data
    # There are 12 data columns.
    # Row 3 (Cross) has label + 12 values = 13 items.
    
    # Function to get 12 data items, skipping first (label)
    def get_data_items(row_split):
        # Filter out empty strings that might be at the end due to trailing tabs
        # But be careful about the first empty string if it exists (label placeholder)
        
        # If the row starts with a label (not empty), index 0 is label.
        # If the row starts with empty string (tab), index 0 is empty label.
        
        # Take elements 1 to 13 (indices)
        # Check length
        if len(row_split) < 13:
            # Pad with None or handle error
            print(f"Warning: Row has {len(row_split)} items, expected at least 13")
            return row_split[1:] + [None]*(12 - len(row_split) + 1)
        return row_split[1:13]

    data_values_cross = get_data_items(row3)
    data_values_sft = get_data_items(row4)
    data_values_gain = get_data_items(row5)
    
    # Construct headers for the 12 columns
    tasks = []
    transfers = []
    sizes = []
    
    # Hardcoded mapping based on visual layout if parsing is tricky
    # Indices 0-3: TDTSP-TW
    # Indices 4-7: TDVRP-TW
    # Indices 8-11: CLRP
    
    for i in range(4): tasks.append("TDTSP-TW")
    for i in range(4): tasks.append("TDVRP-TW")
    for i in range(4): tasks.append("CLRP")
    
    # Transfers
    # 0-1: Nairobi->Berlin
    # 2-3: Berlin-Nairobi
    # 4-5: Nairobi->Berlin
    # 6-7: Berlin-Nairobi
    # 8-9: Prodhon->Barreto
    # 10-11: Barreto->Prodhon
    transfers.extend(["Nairobi->Berlin"] * 2)
    transfers.extend(["Berlin->Nairobi"] * 2)
    transfers.extend(["Nairobi->Berlin"] * 2)
    transfers.extend(["Berlin->Nairobi"] * 2)
    transfers.extend(["Prodhon->Barreto"] * 2)
    transfers.extend(["Barreto->Prodhon"] * 2)
    
    # Sizes - take directly from Row 2 (indices 1 to 12)
    # But clean them up (remove whitespace)
    raw_sizes = get_data_items(row2)
    
    final_sizes = []
    for s in raw_sizes:
        if s and "20" in s: final_sizes.append("N=20")
        elif s and "50" in s: final_sizes.append("N=50")
        elif s and "100" in s: final_sizes.append("N=100")
        else: final_sizes.append(str(s).strip())
        
    print(f"Lengths: Tasks={len(tasks)}, Transfers={len(transfers)}, Sizes={len(final_sizes)}")
    print(f"Data Lengths: Cross={len(data_values_cross)}, SFT={len(data_values_sft)}, Gain={len(data_values_gain)}")
    
    # Build DataFrame
    df = pd.DataFrame({
        "Task": tasks,
        "Transfer": transfers,
        "Size": final_sizes,
        "STS (Cross)": [float(x) for x in data_values_cross],
        "STS (SFT)": [float(x) for x in data_values_sft],
        "Gain": [x.strip() for x in data_values_gain]
    })
    
    print("Cleaned Data:")
    print(df)
    
    df.to_csv(output_path, index=False)
    print(f"Saved to {output_path}")

if __name__ == "__main__":
    clean_cross_csv()
