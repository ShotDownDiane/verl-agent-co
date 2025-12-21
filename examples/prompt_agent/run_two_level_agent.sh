#!/bin/bash
# Two-Level Agent Pipeline 运行脚本

# 颜色定义
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

echo -e "${GREEN}========================================${NC}"
echo -e "${GREEN}Two-Level Agent Pipeline${NC}"
echo -e "${GREEN}========================================${NC}"

# 检查 vLLM 服务是否运行
echo -e "${YELLOW}Checking vLLM services...${NC}"

# 检查 VLM 服务 (端口 8000)
if ! curl -s http://localhost:8000/health > /dev/null 2>&1; then
    echo -e "${RED}Warning: VLM service (port 8000) is not running${NC}"
    echo -e "${YELLOW}Please start it with: bash start_vlm_with_vllm.sh${NC}"
else
    echo -e "${GREEN}✓ VLM service is running on port 8000${NC}"
fi

# 检查 LLM 服务 (端口 8001)
if ! curl -s http://localhost:8001/health > /dev/null 2>&1; then
    echo -e "${RED}Warning: LLM service (port 8001) is not running${NC}"
    echo -e "${YELLOW}Please start it with: bash start_llm_with_vllm.sh${NC}"
else
    echo -e "${GREEN}✓ LLM service is running on port 8001${NC}"
fi

echo ""

# 设置环境变量（可选）
export VLM_API_URL="${VLM_API_URL:-http://localhost:8000/v1}"
export LLM_API_URL="${LLM_API_URL:-http://localhost:8001/v1}"
export API_KEY="${API_KEY:-token-abc123456}"
export VLM_MODEL_NAME="${VLM_MODEL_NAME:-vlm}"
export LLM_MODEL_NAME="${LLM_MODEL_NAME:-llm}"
export ENV_NUM="${ENV_NUM:-3}"
export TEST_TIMES="${TEST_TIMES:-1}"

# 获取命令行参数
ENV_NAME="${1:-ml4co-kit/tsp}"
SUB_ENV="${2:-tsp}"

echo -e "${GREEN}Configuration:${NC}"
echo "  Environment: $ENV_NAME"
echo "  Sub-env: $SUB_ENV"
echo "  VLM API: $VLM_API_URL"
echo "  LLM API: $LLM_API_URL"
echo "  Env num: $ENV_NUM"
echo "  Test times: $TEST_TIMES"
echo ""

# 运行 pipeline
echo -e "${GREEN}Starting Two-Level Agent Pipeline...${NC}"
python3 two_level_agent_pipeline.py "$ENV_NAME" "$SUB_ENV"

echo -e "${GREEN}========================================${NC}"
echo -e "${GREEN}Pipeline completed!${NC}"
echo -e "${GREEN}========================================${NC}"

