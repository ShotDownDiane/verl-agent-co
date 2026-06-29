#!/usr/bin/env bash
set -euo pipefail

# OSM-v2 Atomic City GRPO training on ROLL with Qwen3-8B
# Requires:
#   - /data/models/qwen3-8b (HF checkpoint)
#   - Prompt data JSONL (large_4000_500 or custom)

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
ARTIFACT_DIR="${ARTIFACT_DIR:-$(cd "${SCRIPT_DIR}/../.." && pwd)}"
ROLL_BASE_DIR="${ROLL_BASE_DIR:-${ARTIFACT_DIR}/third_party/ROLL}"

export PYTHONPATH="${ARTIFACT_DIR}:${ROLL_BASE_DIR}:${ARTIFACT_DIR}/roll/pipeline/agentic/env/osm_v2/deps:${PYTHONPATH:-}"
export ROLL_DIR="${ARTIFACT_DIR}"

# Env variables for the config (paths relative to ROLL_DIR)
export OSM_V2_PROMPT_DATA="${OSM_V2_PROMPT_DATA:-examples/qwen3-8B-osm_v2_atomic/data/large_4000_500/prompt/train.jsonl}"
export OSM_V2_SCAFFOLD_CONFIG="${OSM_V2_SCAFFOLD_CONFIG:-examples/qwen3-8B-osm_v2_atomic/data/scaffolds/osm_v2_sandbox.yaml}"
export OSM_V2_RUN_ROOT="${OSM_V2_RUN_ROOT:-/tmp/roll_osm_v2_atomic_grpo}"

mkdir -p "${OSM_V2_RUN_ROOT}"

cd "${ARTIFACT_DIR}"

python3 "${ROLL_BASE_DIR}/examples/start_agentic_pipeline.py" \
  --config_path "${ARTIFACT_DIR}/examples/qwen3-8B-osm_v2_atomic" \
  --config_name osm_v2_grpo \
  "$@"
