
import sys
import os
import torch
import numpy as np
from adapter import RL4COEnvAdapter

adapter = RL4COEnvAdapter()
state = adapter.reset()
nodes = state['raw_state']['nodes']
node_1 = nodes[1]
print(f"Node 1: {node_1}")
print(f"Time Window: {node_1['time_window']}")

# Step 1
next_node = 1
state = adapter.step(next_node)
print(f"Step 1 Cost (Time): {state['raw_state']['current_cost']}")
