# vllm serve /root/autodl-tmp/Qwen3-4B \
#     --served-model-name llm \
#     --gpu-memory-utilization 0.4 \
#     --tensor-parallel-size 1 \
#     --host 0.0.0.0 \
#     --port 8001 \
#     --api-key token-abc123456 &

vllm serve /root/autodl-tmp/Qwen3VL-4B-Instruct \
    --served-model-name vlm \
    --gpu-memory-utilization 0.6 \
    --tensor-parallel-size 1 \
    --host 0.0.0.0 \
    --port 8000 \
    --api-key token-abc123456 