# JSON Format Specification for CoT Dataset Generation

This document defines the JSON data structure required by `generate_cot_dataset.py` and implemented in the single worker scripts (`flp`, `tsp`, `cvrp`, `mclp`, `stp`).

## 1. Root Structure
The output JSON file is a **List of Objects**, where each object represents a single problem instance's solution trajectory.

```json
[
  {
    "node_coords": { ... },
    "trajectory": [ ... ],
    "obs_list": [ ... ],
    "image_list": [ ... ],
    "candidates": [ ... ],
    "solution_tour": [ ... ]
  },
  ...
]
```

## 2. Field Definitions

### `node_coords`
- **Type**: `Dict[str, List[float]]`
- **Description**: Global coordinates of all nodes in the graph. Keys are node indices (as strings in JSON), values are `[x, y]` coordinates.
- **Example**:
  ```json
  "node_coords": {
    "0": [0.5, 0.5],
    "1": [0.2, 0.8],
    "2": [0.9, 0.1]
  }
  ```

### `trajectory`
- **Type**: `List[str]`
- **Description**: The sequence of actions (decisions) made by the agent.
- **Format**: LaTeX boxed format for labels (e.g., `\boxed{A}`) or raw indices/values.
- **Example**:
  ```json
  "trajectory": [
    "\\boxed{A}",
    "\\boxed{B}",
    "\\boxed{0}"
  ]
  ```

### `obs_list`
- **Type**: `List[str]`
- **Description**: The textual observation prompt received by the agent at each step.
- **Alignment**: `obs_list[i]` corresponds to the state *before* `trajectory[i]` action is taken.

### `image_list`
- **Type**: `List[Union[str, null]]`
- **Description**: The visual observation (Base64 encoded image string) received at each step.
- **Value**: `null` if no image is used or available for that step.

### `candidates`
- **Type**: `List[List[int]]`
- **Description**: The list of available candidate node indices presented to the agent at each step.
- **Alignment**: `candidates[i]` lists the nodes corresponding to options (A, B, C...) at step `i`.
- **Example**:
  ```json
  "candidates": [
    [0, 5, 2],  # Step 0: A->0, B->5, C->2
    [1, 3, 4]   # Step 1: A->1, B->3, C->4
  ]
  ```

### `solution_tour`
- **Type**: `List[int]`
- **Description**: The ground truth or reference solution sequence of node indices.
- **Example**: `[0, 5, 2, 1, 3, 4, 0]`

## 3. Data Compatibility
- **Parallel Lists**: `trajectory`, `obs_list`, `image_list`, and `candidates` must have the exact same length (number of steps).
- **Coordinate Mapping**: All indices in `solution_tour` and `candidates` must exist as keys in `node_coords`.
- **Serialization**: All numeric values are standard JSON numbers (floats/ints), not strings (except keys).

## 4. Usage in `generate_cot_dataset.py`
The ingestion script expects to iterate through these fields simultaneously to reconstruct the Chain-of-Thought (CoT) samples:
- It uses `node_coords` to inject geometric information.
- It pairs `obs_list[i]` with `trajectory[i]` to form input-output pairs.
- It uses `candidates` to validate choices.
