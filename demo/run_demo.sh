#!/bin/bash

# Get the directory where the script is located
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
REPO_ROOT="$(dirname "$SCRIPT_DIR")"
BACKEND_DIR="$SCRIPT_DIR/backend"
FRONTEND_DIR="$SCRIPT_DIR/frontend"

echo "=== Stopping existing services... ==="

# Kill backend
# We try to match "main.py" which is the entry point
if pkill -f "python3.*main.py"; then
    echo "Killed old backend process."
else
    echo "No backend process found (or pkill failed)."
fi

# Kill frontend (vite)
if pkill -f "vite"; then
    echo "Killed old frontend process."
else
    echo "No frontend process found."
fi

# Wait a moment to ensure ports are freed
sleep 2

echo "=== Starting Backend... ==="
cd "$BACKEND_DIR" || { echo "Backend directory not found"; exit 1; }
# Run in background with nohup
nohup python3 main.py > ../backend.log 2>&1 &
BACKEND_PID=$!
echo "Backend started with PID $BACKEND_PID"
echo "Backend Logs: $SCRIPT_DIR/backend.log"

echo "=== Starting Frontend... ==="
cd "$FRONTEND_DIR" || { echo "Frontend directory not found"; exit 1; }
# Run in background with nohup
nohup npm run dev -- --host 0.0.0.0 > ../frontend.log 2>&1 &
FRONTEND_PID=$!
echo "Frontend started with PID $FRONTEND_PID"
echo "Frontend Logs: $SCRIPT_DIR/frontend.log"

echo "=== Demo is running! ==="
echo "Backend API: http://localhost:8000"
echo "Frontend UI: http://localhost:5173"
echo "To stop them later, you can run this script again (it kills old processes first) or use 'pkill -f ...'"
