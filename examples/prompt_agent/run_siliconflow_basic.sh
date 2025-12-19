#!/bin/bash
set -e

echo "开始执行Python脚本..."

python examples/prompt_agent/siliconflow_poi_placement.py meta-llama/Llama-3.3-70B-Instruct
# python examples/prompt_agent/siliconflow_road_planning.py Pro/Qwen/Qwen2.5-7B-Instruct
# python examples/prompt_agent/siliconflow_route_planning.py meta-llama/Llama-3.3-70B-Instruct
# python examples/prompt_agent/siliconflow_urban_planning.py meta-llama/Llama-3.3-70B-Instruct

echo "所有脚本执行完毕！"