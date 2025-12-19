#!/bin/bash
set -e

echo "开始执行Python脚本..."

python examples/prompt_agent/torch_road_planning.py Qwen/Qwen2.5-1.5B-Instruct
# python examples/prompt_agent/torch_route_planning.py Qwen/Qwen2.5-1.5B-Instruct
# python examples/prompt_agent/torch_traffic_signal_control.py Qwen/Qwen2.5-1.5B-Instruct
# python examples/prompt_agent/torch_urban_planning.py Qwen/Qwen2.5-1.5B-Instruct

echo "所有脚本执行完毕！"