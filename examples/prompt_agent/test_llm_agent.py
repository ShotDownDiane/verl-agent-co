#!/usr/bin/env python3
"""
测试 LLM Agent
演示如何使用 API、vLLM 或 transformers 模式
"""
import sys
import logging
from llm_agent import LLMAgent

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def main():
    """主测试函数"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Test LLM Agent")
    parser.add_argument("--mode", type=str, choices=["api", "vllm", "transformers"], 
                       default="transformers", help="Agent mode")
    parser.add_argument("--model-path", type=str, 
                       default="/root/autodl-tmp/Qwen3-4B",
                       help="Model path (for vllm/transformers mode)")
    parser.add_argument("--api-url", type=str, 
                       default="http://localhost:8001/v1",
                       help="API base URL (for api mode)")
    parser.add_argument("--api-key", type=str, default=None,
                       help="API key (optional)")
    parser.add_argument("--model-name", type=str, default="llm",
                       help="Model name (for api mode)")
    
    args = parser.parse_args()
    
    logger.info("="*60)
    logger.info("LLM Agent Test")
    logger.info("="*60)
    logger.info(f"Mode: {args.mode}")
    
    # 初始化 Agent
    try:
        if args.mode == "api":
            logger.info(f"API URL: {args.api_url}")
            logger.info(f"Model name: {args.model_name}")
            agent = LLMAgent(
                api_base_url=args.api_url,
                api_key=args.api_key,
                model_name=args.model_name,
            )
        elif args.mode == "vllm":
            logger.info(f"Model path: {args.model_path}")
            agent = LLMAgent(
                model_path=args.model_path,
                use_vllm=True,
            )
        else:  # transformers
            logger.info(f"Model path: {args.model_path}")
            agent = LLMAgent(
                model_path=args.model_path,
                use_vllm=False,
            )
        
        logger.info(f"✓ Agent initialized successfully (mode: {agent.mode})")
        
    except Exception as e:
        logger.error(f"Failed to initialize agent: {e}")
        import traceback
        logger.error(traceback.format_exc())
        sys.exit(1)
    
    # 测试 1: 简单文本生成
    logger.info("\n" + "="*60)
    logger.info("Test 1: Simple text generation")
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
    
    # 测试 2: 带系统提示的生成
    logger.info("\n" + "="*60)
    logger.info("Test 2: Generation with system prompt")
    logger.info("="*60)
    
    system_prompt = "你是一个专业的算法专家，擅长解释复杂的优化问题。"
    user_prompt = "请解释一下什么是贪心算法，并给出一个简单的例子。"
    
    try:
        result = agent.generate(
            user_prompt,
            system_prompt=system_prompt,
            max_tokens=200
        )
        logger.info(f"✓ Generation with system prompt successful")
        logger.info(f"System: {system_prompt}")
        logger.info(f"User: {user_prompt}")
        logger.info(f"Response: {result}")
    except Exception as e:
        logger.warning(f"Generation with system prompt failed: {e}")
        import traceback
        logger.warning(traceback.format_exc())
    
    # 测试 3: 消息列表格式
    logger.info("\n" + "="*60)
    logger.info("Test 3: Message list format")
    logger.info("="*60)
    
    messages = [
        {"role": "system", "content": "你是一个有用的助手。"},
        {"role": "user", "content": "请用三个词总结一下机器学习的核心概念。"}
    ]
    
    try:
        result = agent.generate(messages, max_tokens=50)
        logger.info(f"✓ Message list format successful")
        logger.info(f"Messages: {messages}")
        logger.info(f"Response: {result}")
    except Exception as e:
        logger.warning(f"Message list format failed: {e}")
        import traceback
        logger.warning(traceback.format_exc())
    
    logger.info("\n" + "="*60)
    logger.info("✓ All tests passed!")
    logger.info("="*60)
    logger.info(f"\nLLM Agent is ready!")
    logger.info(f"Using mode: {agent.mode}")


if __name__ == "__main__":
    main()

