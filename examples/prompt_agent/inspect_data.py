
import os
import pickle
import torch
import numpy as np

def inspect():
    base_path = "/root/autodl-tmp/rl4co-urban"
    file_path = os.path.join(base_path, "routing_results.pkl")
    
    with open(file_path, "rb") as f:
        routing_data = pickle.load(f)
    
    if 'cvrp' in routing_data:
        cvrp_data = routing_data['cvrp']
        if len(cvrp_data) > 0:
            item = cvrp_data[0]
            print("\nCVRP Item Keys:", item.keys())
            if 'objs' in item:
                 print("CVRP Objs Sample:", item['objs'][:1])

if __name__ == "__main__":
    inspect()
