#!/bin/bash
# 一键启动本地模型服务并运行测试的脚本

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# 颜色输出
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

echo -e "${GREEN}========================================${NC}"
echo -e "${GREEN}Two-Level Agent 本地测试启动脚本${NC}"
echo -e "${GREEN}========================================${NC}"

# 检查模型路径
VLM_PATH="/root/autodl-tmp/Qwen3VL-4B-Instruct"
LLM_PATH="/root/autodl-tmp/Qwen3-4B"

if [ ! -d "$VLM_PATH" ]; then
    echo -e "${RED}错误: VLM 模型路径不存在: $VLM_PATH${NC}"
    exit 1
fi

if [ ! -d "$LLM_PATH" ]; then
    echo -e "${RED}错误: LLM 模型路径不存在: $LLM_PATH${NC}"
    exit 1
fi

# 检查 vllm 是否安装
if ! python -c "import vllm" 2>/dev/null; then
    echo -e "${YELLOW}警告: vLLM 未安装，正在尝试安装...${NC}"
    pip install vllm || {
        echo -e "${RED}错误: 无法安装 vLLM，请手动安装: pip install vllm${NC}"
        exit 1
    }
fi

# 检查端口是否被占用
check_port() {
    if lsof -Pi :$1 -sTCP:LISTEN -t >/dev/null 2>&1 ; then
        echo -e "${YELLOW}警告: 端口 $1 已被占用${NC}"
        read -p "是否继续？(y/n) " -n 1 -r
        echo
        if [[ ! $REPLY =~ ^[Yy]$ ]]; then
            exit 1
        fi
    fi
}

check_port 8000
check_port 8001

# 启动服务
echo -e "${GREEN}启动本地模型服务...${NC}"
echo -e "${YELLOW}注意: 服务启动可能需要几分钟时间${NC}"

# 在后台启动服务
python start_local_models.py > /tmp/vllm_services.log 2>&1 &
SERVICES_PID=$!

echo -e "${GREEN}服务启动中 (PID: $SERVICES_PID)${NC}"
echo -e "${YELLOW}日志文件: /tmp/vllm_services.log${NC}"

# 等待服务启动
echo -e "${YELLOW}等待服务启动...${NC}"
for i in {1..60}; do
    if curl -s http://localhost:8000/v1/models > /dev/null 2>&1 && \
       curl -s http://localhost:8001/v1/models > /dev/null 2>&1; then
        echo -e "${GREEN}服务已启动！${NC}"
        break
    fi
    if [ $i -eq 60 ]; then
        echo -e "${RED}错误: 服务启动超时${NC}"
        echo -e "${YELLOW}查看日志: tail -f /tmp/vllm_services.log${NC}"
        kill $SERVICES_PID 2>/dev/null || true
        exit 1
    fi
    sleep 2
    echo -n "."
done
echo

# 运行测试
ENV_NAME="${1:-ml4co-kit/tsp}"
echo -e "${GREEN}运行测试: $ENV_NAME${NC}"
echo -e "${GREEN}========================================${NC}"

# 设置清理函数
cleanup() {
    echo -e "\n${YELLOW}清理中...${NC}"
    kill $SERVICES_PID 2>/dev/null || true
    echo -e "${GREEN}完成${NC}"
}

trap cleanup EXIT INT TERM

# 运行测试
python siliconflow_ml4cokit_two_level_local.py "$ENV_NAME"

echo -e "${GREEN}========================================${NC}"
echo -e "${GREEN}测试完成！${NC}"

