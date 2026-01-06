# python two_level_agent_pipeline.py \
#     --env_name ml4co-kit \
#     --sub_env tsp \
#     --vlm_api_url https://api.siliconflow.cn/v1 \
#     --llm_api_url https://api.siliconflow.cn/v1 \
#     --api_key sk-saxqqtlyqrpconxlgcslqhrgvhwnfmuhnimiyzfvpcxqgmkh \
#     --mode train \
#     --env_num 15 \
#     --test_times 1 \
#     --vlm_model_name Qwen/Qwen3-VL-235B-A22B-Instruct\
#     --llm_model_name deepseek-ai/DeepSeek-V3.2 \

# python llm_only_pipeline.py \
#     --env_name rl4co \
#     --sub_env tsp \
#     --llm_api_url https://api.siliconflow.cn/v1 \
#     --api_key sk-saxqqtlyqrpconxlgcslqhrgvhwnfmuhnimiyzfvpcxqgmkh \
#     --env_num 1 \
#     --test_times 1 \
#     --llm_model_name deepseek-ai/DeepSeek-V3.2 \

python vlm_only_pipeline.py \
    --env_name rl4co \
    --sub_env tsp \
    --vlm_api_url http://localhost:8000 \
    --api_key token-abc123456 \
    --env_num 1 \
    --test_times 1 \
    --vlm_model_name vlm \