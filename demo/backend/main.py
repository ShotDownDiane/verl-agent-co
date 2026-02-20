from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response
from pydantic import BaseModel
from typing import List, Optional, Dict, Any
import base64
import sys
import os

# Ensure adapter can be imported
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from adapter import RL4COEnvAdapter

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
    time_window: Optional[List[float]] = None

class State(BaseModel):
    nodes: List[Node]
    current_path: List[int]
    current_cost: float
    capacity: float
    remaining_capacity: float
    mode: str = "real"  # "real" or "virtual"
    logs: List[str] = []
    text_prompt: str = ""

class Action(BaseModel):
    node_id: int

class ModelRequest(BaseModel):
    model_name: str

class ModelResponse(BaseModel):
    observation: str
    thought: str
    decision: str
    node_id: int

# --- Global State ---
# We store the adapter instance and the latest response
adapter = RL4COEnvAdapter()
latest_response: Dict[str, Any] = {}
global_logs: List[str] = []

def format_time(seconds: float) -> str:
    start_hour = 15
    total_seconds = int(seconds) + start_hour * 3600
    h = (total_seconds // 3600) % 24
    m = (total_seconds % 3600) // 60
    s = total_seconds % 60
    return f"{h:02}:{m:02}:{s:02}"

# --- API Endpoints ---

@app.post("/api/reset")
def reset_environment():
    global latest_response, global_logs
    try:
        latest_response = adapter.reset()
        global_logs = ["Environment initialized via RL4CO Adapter."]
        
        # Append Initial Prompt
        # prompt_log = f"[Context Update]\n{latest_response.get('text_prompt', '')}"
        # global_logs.append(prompt_log)
        
        # Construct State object
        raw = latest_response["raw_state"]
        return {
            "nodes": raw["nodes"],
            "current_path": raw["current_path"],
            "current_cost": raw["current_cost"],
            "capacity": raw["capacity"],
            "remaining_capacity": raw["remaining_capacity"],
            "mode": "real",
            "logs": global_logs,
            "text_prompt": latest_response.get("text_prompt", "")
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/state")
def get_state():
    global latest_response, global_logs
    if not latest_response:
        # If no state, try reset
        return reset_environment()
        
    raw = latest_response["raw_state"]
    return {
        "nodes": raw["nodes"],
        "current_path": raw["current_path"],
        "current_cost": raw["current_cost"],
        "capacity": raw["capacity"],
        "remaining_capacity": raw["remaining_capacity"],
        "mode": "real",
        "logs": global_logs,
        "text_prompt": latest_response.get("text_prompt", "")
    }

@app.post("/api/step")
def step(action: Action):
    global latest_response, global_logs
    node_id = action.node_id
    
    try:
        # Call Adapter
        latest_response = adapter.step(node_id)
        
        # Log
        raw = latest_response["raw_state"]
        nodes = raw["nodes"]
        
        # Find node TW
        target_node = next((n for n in nodes if n["id"] == node_id), None)
        tw_str = ""
        if target_node and "time_window" in target_node:
            tw = target_node["time_window"]
            tw_str = f" [TW: {format_time(tw[0])} - {format_time(tw[1])}]"
            
        current_time_str = format_time(raw['current_cost'])
        
        log_msg = f"Moved to Node {node_id}{tw_str}. Time: {current_time_str}"
        global_logs.append(log_msg)
        
        # Append Prompt as a System/Context Log
        # prompt_log = f"[Context Update]\n{latest_response.get('text_prompt', '')}"
        # global_logs.append(prompt_log)
        
        return {
            "nodes": raw["nodes"],
            "current_path": raw["current_path"],
            "current_cost": raw["current_cost"],
            "capacity": raw["capacity"],
            "remaining_capacity": raw["remaining_capacity"],
            "mode": "real",
            "logs": global_logs,
            "text_prompt": latest_response.get("text_prompt", "")
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/virtual-map")
def get_virtual_map():
    global latest_response
    if not latest_response:
        reset_environment()
        
    b64_str = latest_response.get("virtual_map", "")
    if not b64_str:
        # Return empty transparent pixel or placeholder
        return Response(content=b"", media_type="image/png")
        
    # Decode base64
    try:
        img_data = base64.b64decode(b64_str)
        return Response(content=img_data, media_type="image/png")
    except Exception as e:
        print(f"Error decoding image: {e}")
        return Response(content=b"", media_type="image/png")

@app.post("/api/predict")
def predict(req: ModelRequest):
    global latest_response, global_logs
    model = req.model_name
    
    if not latest_response:
        reset_environment()
    
    # 1. Get Context from Adapter
    text_prompt = latest_response.get("text_prompt", "")
    # Note: We also have the image in latest_response["virtual_map"]
    
    # 2. Mock Model Logic (Greedy based on text prompt parsing or raw state)
    # For now, we still use a simple heuristic on raw state, 
    # but in future this is where we call the LLM API with `text_prompt` + image.
    
    raw = latest_response["raw_state"]
    current_path = raw["current_path"]
    nodes = raw["nodes"]
    
    # Find unvisited nodes
    visited_ids = set(current_path)
    unvisited = [n for n in nodes if n["id"] not in visited_ids]
    
    decision_node = -1
    thought = "Thinking..."
    
    if not unvisited:
        if current_path[-1] != 0:
            decision_node = 0
            thought = "All customers visited. Returning to depot."
        else:
            thought = "Mission Complete."
    else:
        # Greedy heuristic: closest unvisited
        # We can calculate distance using lat/lon or x/y
        last_node_id = current_path[-1]
        last_node = next(n for n in nodes if n["id"] == last_node_id)
        
        def dist(n1, n2):
            return math.sqrt((n1["x"] - n2["x"])**2 + (n1["y"] - n2["y"])**2)
            
        closest = min(unvisited, key=lambda n: dist(n, last_node))
        decision_node = closest["id"]
        thought = f"Based on spatial analysis, Node {decision_node} is the optimal next stop to minimize travel time."

    # Log the model's decision
    target_node = next((n for n in nodes if n["id"] == decision_node), None)
    tw_str = ""
    if target_node and "time_window" in target_node:
        tw = target_node["time_window"]
        tw_str = f" [TW: {format_time(tw[0])} - {format_time(tw[1])}]"
    
    log_msg = f"Model ({model}) selected Node {decision_node}{tw_str}."
    global_logs.append(log_msg)
    
    # We don't append context here because predict doesn't change state/context, only step does.
    # However, if we wanted to show what the model saw, we could. 
    # But user asked for "environment prompt" which usually updates on step.
    # Step updates the state, so the prompt changes. Predict uses current state.
    
    return {
        "observation": text_prompt, # Send the actual text prompt the model "saw"
        "thought": thought,
        "decision": f"Action: {decision_node}",
        "node_id": decision_node
    }

if __name__ == "__main__":
    import uvicorn
    import math
    uvicorn.run(app, host="0.0.0.0", port=8000)
