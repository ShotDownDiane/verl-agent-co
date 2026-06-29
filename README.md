# OSM-v2 Workspace Agent

This repository is a compact artifact for the OSM-v2 workspace-agent experiments. It is based on ROLL, which is included as a submodule pinned to the source workspace version:

```text
ROLL commit: 4bb7d74af11ce569f53eda0f18a77a5c8bdf3600
ROLL remote: https://github.com/alibaba/ROLL.git
```

The OSM-v2 environment and data were an overlay in the source workspace, so the overlay files are kept in this repository while the base ROLL framework is referenced through `third_party/ROLL`.

## Contents

- `examples/qwen3-8B-osm_v2_atomic/data/`: expected location for the OSM-v2 production instances, configs, prompt data, plan bank, fixed eval subsets, and workspace scaffolds. The data files are not included in this anonymous repository.
- `examples/qwen3-8B-osm_v2_atomic/run_osm_v2_grpo.sh`: GRPO training launcher.
- `examples/qwen3-8B-osm_v2_atomic/run_osm_v2_rollout_only.sh`: rollout-only launcher for pipeline debugging.
- `examples/qwen3-8B-osm_v2_atomic/eval_full_test.py`: OpenAI-compatible model evaluation entrypoint.
- `examples/qwen3-8B-osm_v2_atomic/eval_heuristics_test.py`: offline heuristic evaluation entrypoint.
- `examples/config/traj_envs_osm_v2.yaml`: ROLL environment registration config used by the training launchers.
- `roll/pipeline/agentic/env/osm_v2/`: OSM-v2 runtime overlay for ROLL.
- `third_party/ROLL/`: pinned ROLL submodule.

Internal handoff notes, plotting scripts, data-generation scripts, benchmark result files, and the large expanded data directory are intentionally omitted from git history. Generated outputs should be written under `output/`, which is not part of the submitted artifact.

## Data

The expanded OSM-v2 data is about 1.2GB and is not included in this anonymous repository. We will release the data after the review process permits public artifact distribution.

The code expects the released data to be placed under `examples/qwen3-8B-osm_v2_atomic/data/`.

## Setup

```bash
git submodule update --init --recursive
pip install -r requirements.txt
export PYTHONPATH="$PWD:$PWD/third_party/ROLL:$PWD/roll/pipeline/agentic/env/osm_v2/deps:$PYTHONPATH"
```

## Offline Smoke Test

After the data is available, run a small heuristic check without an LLM server:

```bash
python examples/qwen3-8B-osm_v2_atomic/eval_heuristics_test.py \
  --tasks urban_planning \
  --limit-per-task 2 \
  --workers 2 \
  --sa-iters 10 \
  --alns-iters 10 \
  --metaheuristic-init random \
  --output-dir output/smoke_urban \
  --no-resume
```

## Training

After the data is available, set the model path in `examples/qwen3-8B-osm_v2_atomic/osm_v2_grpo.yaml` or override it through ROLL/Hydra command-line arguments, then run:

```bash
bash examples/qwen3-8B-osm_v2_atomic/run_osm_v2_grpo.sh
```

For a rollout-only pipeline check:

```bash
bash examples/qwen3-8B-osm_v2_atomic/run_osm_v2_rollout_only.sh
```

## Model Evaluation

After the data is available, start any OpenAI-compatible chat-completions server separately, then run:

```bash
python examples/qwen3-8B-osm_v2_atomic/eval_full_test.py \
  --url http://localhost:8000 \
  --model /path/to/model \
  --instances-file examples/qwen3-8B-osm_v2_atomic/data/eval_subsets/osm_v2_test_20_per_task_balanced_20260614.json \
  --workers 4 \
  --max-turns 10 \
  --patience 3 \
  --output output/eval_full_test.json \
  --rollout-dir output/rollouts \
  --skip-healthcheck
```

Use `--n` instead of `--instances-file` to sample the first `n` test instances per task.
