from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response
from pydantic import BaseModel
from typing import List, Optional, Dict, Any
import random
import numpy as np
import math
import cv2
import base64

app = FastAPI()

# Allow CORS for frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Data Models ---
class Node(BaseModel):
    id: int
    type: str  # "depot" or "customer"
    lat: float
    lon: float
    x: float  # Normalized x (0-1) or grid (0-224)
    y: float  # Normalized y (0-1) or grid (0-224)
    demand: Optional[float] = 0.0

class State(BaseModel):
    nodes: List[Node]
    current_path: List[int]
    current_cost: float
    capacity: float
    remaining_capacity: float
    mode: str = "real"  # "real" or "virtual"
    logs: List[str] = []

class Action(BaseModel):
    node_id: int

class ModelRequest(BaseModel):
    model_name: str

class ModelResponse(BaseModel):
    observation: str
    thought: str
    decision: str
    node_id: int

# --- Global State (In-memory for demo simplicity) ---
# In a real app, use a database or session management
global_state: Dict[str, Any] = {
    "nodes": [],
    "current_path": [],
    "current_cost": 0.0,
    "capacity": 100.0,
    "remaining_capacity": 100.0,
    "logs": []
}

# --- Helper Functions ---
def calculate_distance(node1: Dict, node2: Dict) -> float:
    return math.sqrt((node1["x"] - node2["x"])**2 + (node1["y"] - node2["y"])**2)

def generate_random_nodes(num_customers: int = 10) -> List[Node]:
    nodes = []
    # Center around San Francisco for Real Map
    center_lat = 37.7749
    center_lon = -122.4194
    
    # Depot
    nodes.append(Node(
        id=0,
        type="depot",
        lat=center_lat,
        lon=center_lon,
        x=112, # Center of 224x224 grid
        y=112,
        demand=0
    ))
    
    for i in range(1, num_customers + 1):
        # Random offset for lat/lon
        lat_offset = (random.random() - 0.5) * 0.1
        lon_offset = (random.random() - 0.5) * 0.1
        
        # Map to 224x224 grid roughly
        # This is a simple linear mapping for demo purposes
        x = 112 + (lon_offset * 2000) # Scale factor
        y = 112 + (lat_offset * 2000)
        
        # Clamp to 0-224
        x = max(0, min(224, x))
        y = max(0, min(224, y))
        
        nodes.append(Node(
            id=i,
            type="customer",
            lat=center_lat + lat_offset,
            lon=center_lon + lon_offset,
            x=x,
            y=y,
            demand=random.randint(1, 10)
        ))
    return nodes

def draw_virtual_map(nodes: List[Dict], current_path: List[int]) -> np.ndarray:
    # Create white canvas 224x224
    img = np.ones((224, 224, 3), dtype=np.uint8) * 255
    
    # Draw Grid (optional, faint gray)
    step = 28
    for i in range(0, 224, step):
        cv2.line(img, (i, 0), (i, 224), (240, 240, 240), 1)
        cv2.line(img, (0, i), (224, i), (240, 240, 240), 1)

    # Draw Path
    path_nodes = [n for id in current_path for n in nodes if n["id"] == id]
    for i in range(len(path_nodes) - 1):
        pt1 = (int(path_nodes[i]["x"]), int(path_nodes[i]["y"]))
        pt2 = (int(path_nodes[i+1]["x"]), int(path_nodes[i+1]["y"]))
        cv2.line(img, pt1, pt2, (255, 0, 0), 2) # Blue path

    # Draw Nodes
    for node in nodes:
        pt = (int(node["x"]), int(node["y"]))
        color = (0, 0, 255) if node["type"] == "depot" else (0, 200, 0) # Red depot, Green customer
        radius = 5 if node["type"] == "depot" else 3
        
        # If visited, gray out
        if node["id"] in current_path:
             if node["type"] != "depot" or (node["type"] == "depot" and len(current_path) > 1 and current_path[-1] == 0):
                color = (150, 150, 150)

        cv2.circle(img, pt, radius, color, -1)
        # Optional: Demand text
        # cv2.putText(img, str(node["demand"]), (pt[0]+5, pt[1]-5), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0,0,0), 1)

    return img

# --- API Endpoints ---

@app.post("/api/reset")
def reset_environment():
    global global_state
    nodes = generate_random_nodes(10)
    global_state["nodes"] = [node.dict() for node in nodes]
    global_state["current_path"] = [0] # Start at depot
    global_state["current_cost"] = 0.0
    global_state["capacity"] = 20.0
    global_state["remaining_capacity"] = 20.0
    global_state["logs"] = ["Environment initialized. Started at Depot (Node 0)."]
    return global_state

@app.get("/api/state")
def get_state():
    return global_state

@app.post("/api/step")
def step(action: Action):
    global global_state
    node_id = action.node_id
    
    nodes = global_state["nodes"]
    target_node = next((n for n in nodes if n["id"] == node_id), None)
    
    if not target_node:
        raise HTTPException(status_code=404, detail="Node not found")
    
    last_node_id = global_state["current_path"][-1]
    last_node = next((n for n in nodes if n["id"] == last_node_id), None)
    
    dist = calculate_distance(target_node, last_node)
    
    global_state["current_path"].append(node_id)
    global_state["current_cost"] += dist
    global_state["remaining_capacity"] -= target_node["demand"]
    
    log_msg = f"User selected Node {node_id}. Cost: +{dist:.2f}, Remaining Cap: {global_state['remaining_capacity']}"
    global_state["logs"].append(log_msg)
    
    return global_state

@app.get("/api/virtual-map")
def get_virtual_map():
    global global_state
    if not global_state["nodes"]:
        reset_environment()
        
    img = draw_virtual_map(global_state["nodes"], global_state["current_path"])
    _, buffer = cv2.imencode('.png', img)
    return Response(content=buffer.tobytes(), media_type="image/png")

@app.post("/api/predict")
def predict(req: ModelRequest):
    global global_state
    model = req.model_name
    
    # Mock Logic for different models
    # In a real app, this would call the LLM API with the image from get_virtual_map and state text
    
    nodes = global_state["nodes"]
    current_path = global_state["current_path"]
    visited = set(current_path)
    
    # Simple Greedy Strategy for "Mock" Model
    unvisited = [n for n in nodes if n["id"] not in visited]
    
    decision_node = -1
    thought = ""
    
    if not unvisited:
        # Return to depot if all visited
        if current_path[-1] != 0:
            decision_node = 0
            thought = "All customers visited. Returning to depot."
        else:
             thought = "All done."
    else:
        # Find closest unvisited
        last_node = next(n for n in nodes if n["id"] == current_path[-1])
        closest = min(unvisited, key=lambda n: calculate_distance(n, last_node))
        decision_node = closest["id"]
        thought = f"Calculated distances to remaining {len(unvisited)} nodes. Node {closest['id']} is closest at distance {calculate_distance(closest, last_node):.2f}."

    # Simulate Model Output
    observation = f"I see {len(nodes)} nodes. Current position is Node {current_path[-1]}. {len(unvisited)} unvisited customers remaining."
    
    # Format for Chat
    log_msg = f"Model ({model}) chose Node {decision_node}."
    global_state["logs"].append(log_msg)
    
    return {
        "observation": observation,
        "thought": thought,
        "decision": f"\\boxed{{Option A [Node {decision_node}]}}", # Simulate LaTeX output
        "node_id": decision_node
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
