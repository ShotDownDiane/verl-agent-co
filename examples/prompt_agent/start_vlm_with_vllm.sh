
vllm serve /root/autodl-tmp/Qwen3-VL-32B-Thinking \
    --served-model-name vlm \
    --gpu-memory-utilization 0.8 \
    --max-model-len 4096 \
    --tensor-parallel-size 2 \
    --host 0.0.0.0 \
    --port 8000 \
    --api-key token-abc123456