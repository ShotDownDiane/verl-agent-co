#!/bin/bash
set -e

echo "开始执行Python脚本..."

python examples/prompt_agent/reflection_poi_placement.py Qwen/Qwen2.5-7B-Instruct
# python examples/prompt_agent/reflection_road_planning.py Qwen/Qwen2.5-7B-Instruct
# python examples/prompt_agent/reflection_route_planning.py Qwen/Qwen2.5-7B-Instruct
# python examples/prompt_agent/reflection_urban_planning.py Qwen/Qwen2.5-7B-Instruct
# python examples/prompt_agent/reflection_poi_placement.py deepseek-ai/DeepSeek-V3

echo "所有脚本执行完毕！"