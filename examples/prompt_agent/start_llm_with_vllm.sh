vllm serve /root/autodl-tmp/Qwen3-4B \
    --served-model-name llm \
    --gpu-memory-utilization 0.8 \
    --tensor-parallel-size 1 \
    --host 0.0.0.0 \
    --port 8001 \
    --api-key token-abc123456 