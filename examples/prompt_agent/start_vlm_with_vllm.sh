vllm serve /root/autodl-tmp/Qwen3VL-4B-Instruct \
    --served-model-name vlm \
    --gpu-memory-utilization 0.8 \
    --tensor-parallel-size 1 \
    --host 0.0.0.0 \
    --port 8000 \
    --api-key token-abc123456