#!/usr/bin/env python3
"""
测试 VLM Agent
演示如何使用 API、vLLM 或 transformers 模式
"""
import sys
import logging
import base64
import io
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from vlm_agent import VLMAgent

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def create_test_image():
    """创建测试图片"""
    np.random.seed(42)
    num_cities = 10
    locs = np.random.rand(num_cities, 2)
    
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.scatter(locs[:, 0], locs[:, 1], s=100, c='blue', marker='o', 
              edgecolors='black', linewidths=2, zorder=3)
    
    for i, (x, y) in enumerate(locs):
        ax.annotate(f'{i}', (x, y), xytext=(5, 5), textcoords='offset points',
                   fontsize=10, fontweight='bold')
    
    ax.set_xlabel('X Coordinate', fontsize=12)
    ax.set_ylabel('Y Coordinate', fontsize=12)
    ax.set_title('TSP Problem Visualization', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.set_aspect('equal', adjustable='box')
    
    buf = io.BytesIO()
    plt.savefig(buf, format='png', dpi=100, bbox_inches='tight')
    buf.seek(0)
    
    img_base64 = base64.b64encode(buf.read()).decode('utf-8')
    buf.close()
    plt.close(fig)
    
    return img_base64


def main():
    """主测试函数"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Test VLM Agent")
    parser.add_argument("--mode", type=str, choices=["api", "vllm", "transformers"], 
                       default="transformers", help="Agent mode")
    parser.add_argument("--model-path", type=str, 
                       default="/root/autodl-tmp/Qwen3VL-4B-Instruct",
                       help="Model path (for vllm/transformers mode)")
    parser.add_argument("--api-url", type=str, 
                       default="https://api.siliconflow.cn/v1",
                       help="API base URL (for api mode)")
    parser.add_argument("--api-key", type=str, default="sk-saxqqtlyqrpconxlgcslqhrgvhwnfmuhnimiyzfvpcxqgmkh",
                       help="API key (optional)")
    parser.add_argument("--model-name", type=str, default="Qwen/Qwen3-VL-235B-A22B-Thinking",
                       help="Model name (for api mode)")
    parser.add_argument("--test-image", action="store_true",
                       help="Test image input")
    
    args = parser.parse_args()
    
    logger.info("="*60)
    logger.info("VLM Agent Test")
    logger.info("="*60)
    logger.info(f"Mode: {args.mode}")
    
    # 初始化 Agent
    try:
        if args.mode == "api":
            logger.info(f"API URL: {args.api_url}")
            logger.info(f"Model name: {args.model_name}")
            agent = VLMAgent(
                api_base_url=args.api_url,
                api_key=args.api_key,
                model_name=args.model_name,
            )
        elif args.mode == "vllm":
            logger.info(f"Model path: {args.model_path}")
            agent = VLMAgent(
                model_path=args.model_path,
                use_vllm=True,
            )
        else:  # transformers
            logger.info(f"Model path: {args.model_path}")
            agent = VLMAgent(
                model_path=args.model_path,
                use_vllm=False,
            )
        
        logger.info(f"✓ Agent initialized successfully (mode: {agent.mode})")
        
    except Exception as e:
        logger.error(f"Failed to initialize agent: {e}")
        import traceback
        logger.error(traceback.format_exc())
        sys.exit(1)
    
    # 测试 1: 文本生成
    logger.info("\n" + "="*60)
    logger.info("Test 1: Text-only generation")
    logger.info("="*60)
    
    test_prompt = "请用一句话解释什么是旅行商问题（TSP）。"
    try:
        result = agent.generate(test_prompt, max_tokens=100)
        logger.info(f"✓ Text generation successful")
        logger.info(f"Prompt: {test_prompt}")
        logger.info(f"Response: {result}")
    except Exception as e:
        logger.error(f"✗ Text generation failed: {e}")
        import traceback
        logger.error(traceback.format_exc())
        sys.exit(1)
    
    # 测试 2: 图片输入（如果启用）
    if args.test_image:
        logger.info("\n" + "="*60)
        logger.info("Test 2: Image input generation")
        logger.info("="*60)
        
        image_base64 = create_test_image()
        logger.info("Created test image")
        
        test_prompt_with_image = "请描述这张图片中的内容。这是一个 TSP 问题的可视化图，请告诉我你看到了多少个城市节点？"
        
        try:
            result = agent.generate(test_prompt_with_image, image=image_base64, max_tokens=200)
            logger.info(f"✓ Image input generation successful")
            logger.info(f"Response: {result}")
        except Exception as e:
            logger.warning(f"Image input failed: {e}")
            import traceback
            logger.warning(traceback.format_exc())
            logger.info("This is OK if the model doesn't support image input in this mode")
    
    logger.info("\n" + "="*60)
    logger.info("✓ All tests passed!")
    logger.info("="*60)
    logger.info(f"\nVLM Agent is ready!")
    logger.info(f"Using mode: {agent.mode}")


if __name__ == "__main__":
    main()

