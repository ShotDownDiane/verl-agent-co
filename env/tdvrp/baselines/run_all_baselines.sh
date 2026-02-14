#!/bin/bash

# Default mode is "fast"
MODE=${1:-slow}

# List of methods to test
METHODS=("sa" "ortools" "alns")

echo "Starting TDVRP parallel benchmark run in '$MODE' mode..."

pids=()

for method in "${METHODS[@]}"; do
    echo "Launching $method..."
    python run_baseline.py --method "$method" --mode "$MODE" > "log_${method}_${MODE}.txt" 2>&1 &
    pids+=($!)
done

echo "All solvers launched. Waiting for completion..."

# Wait for all background processes
for pid in "${pids[@]}"; do
    wait "$pid"
done

echo "All benchmark runs completed."
echo "Check log_*.txt files for details."
