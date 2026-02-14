#!/bin/bash

# Default mode is "fast"
MODE=${1:-slow}

# List of methods to test
# Note: "bks" is also available but typically runs differently (evaluating best known solutions)
METHODS=("ortools")

echo "Starting TDTSP parallel benchmark run in '$MODE' mode..."

pids=()

for method in "${METHODS[@]}"; do
    echo "Launching $method..."
    python run_baseline_vrpbench_tw.py --method "$method" --mode "$MODE"
done

echo "All solvers launched. Waiting for completion..."
echo "All benchmark runs completed."
