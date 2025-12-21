"""
简单的 LLM Agent 类
支持通过 API、vLLM 或 transformers 加载，接受文本输入
"""
import os
import logging
import time
from typing import Optional, List, Dict, Union
import re

logger = logging.getLogger(__name__)


class LLMAgent:
    """简单的 LLM Agent，支持多种加载方式"""
    
    def __init__(
        self,
        model_path: Optional[str] = None,
        api_base_url: Optional[str] = None,
        api_key: Optional[str] = None,
        model_name: Optional[str] = None,
        use_vllm: bool = False,
        vllm_config: Optional[dict] = None,
        api_client: Optional[object] = None,
    ):
        """
        初始化 LLM Agent
        
        Args:
            model_path: 模型路径（用于 vLLM 或 transformers）
            api_base_url: API 服务地址（如 "http://localhost:8000/v1"）
            api_key: API 密钥（可选）
            model_name: API 模型名称（用于 API 模式）
            use_vllm: 是否使用 vLLM（如果 False，使用 transformers）
            vllm_config: vLLM 配置字典
        """
        self.model_path = model_path
        self.api_base_url = api_base_url
        self.api_key = api_key
        self.model_name = model_name
        self.use_vllm = use_vllm
        self.api_client = api_client
        self.mode = None  # 'api', 'vllm', 'transformers'
        
        # 初始化组件
        self.llm = None
        self.tokenizer = None
        self.model = None
        self.client = None
        
        # 确定使用哪种模式
        if api_base_url or api_client:
            self._init_with_api()
        elif use_vllm and model_path:
            self._init_with_vllm(vllm_config)
        elif model_path:
            self._init_with_transformers()
        else:
            raise ValueError("必须提供 api_base_url 或 model_path")
    
    def _init_with_api(self):
        """使用 API 初始化"""
        try:
            # If a custom api_client is provided, use it directly
            if self.api_client is not None:
                self.client = self.api_client
                self.model_name = self.model_name or "llm"
                self.mode = 'api'
                logger.info("✓ LLM Agent initialized with provided api_client")
                return

            from openai import OpenAI

            self.client = OpenAI(
                api_key=self.api_key or "token-abc123456",
                base_url=self.api_base_url,
                max_retries=3,
                timeout=60.0,
            )
            self.model_name = self.model_name or "llm"
            self.mode = 'api'
            logger.info(f"✓ LLM Agent initialized with API: {self.api_base_url}")
            
        except Exception as e:
            logger.error(f"Failed to initialize with API: {e}")
            raise
    
    def _call_chat_completions(self, model, messages, temperature, max_tokens):
        """统一调用不同 API 客户端的 chat completions"""
        client = self.client
        # openai-like client
        try:
            if hasattr(client, "chat") and hasattr(client.chat, "completions"):
                return client.chat.completions.create(
                    model=model, messages=messages, temperature=temperature, max_tokens=max_tokens
                )
        except Exception:
            pass

        # callable client (user provided function)
        if callable(client):
            return client(model=model, messages=messages, temperature=temperature, max_tokens=max_tokens)

        # requests-like client (Session) or raw HTTP fallback
        try:
            import requests
            session = client if hasattr(client, "post") else requests
            url = self.api_base_url.rstrip("/") + "/chat/completions"
            headers = {}
            if self.api_key:
                headers["Authorization"] = f"Bearer {self.api_key}"
            payload = {"model": model, "messages": messages, "temperature": temperature, "max_tokens": max_tokens}
            resp = session.post(url, json=payload, headers=headers, timeout=60.0)
            resp.raise_for_status()
            return resp.json()
        except Exception as e:
            raise RuntimeError(f"Failed to call chat completions: {e}")
    
    def _init_with_vllm(self, config: Optional[dict] = None):
        """使用 vLLM 初始化"""
        try:
            from vllm import LLM
            
            default_config = {
                "trust_remote_code": True,
                "max_model_len": 2048,
                "tensor_parallel_size": 1,
            }
            
            if config:
                default_config.update(config)
            
            logger.info(f"Loading LLM with vLLM from: {self.model_path}")
            logger.info(f"Config: {default_config}")
            
            self.llm = LLM(model=self.model_path, **default_config)
            self.mode = 'vllm'
            logger.info("✓ LLM loaded successfully with vLLM")
            
        except Exception as e:
            logger.error(f"Failed to load with vLLM: {e}")
            logger.info("Falling back to transformers...")
            self._init_with_transformers()
    
    def _init_with_transformers(self):
        """使用 transformers 初始化"""
        try:
            from transformers import AutoModelForCausalLM, AutoTokenizer
            import torch
            
            logger.info(f"Loading LLM with transformers from: {self.model_path}")
            
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_path,
                trust_remote_code=True
            )
            
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_path,
                trust_remote_code=True,
                dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
                device_map="auto",
            )
            
            self.mode = 'transformers'
            logger.info("✓ LLM loaded successfully with transformers")
            
        except Exception as e:
            logger.error(f"Failed to load with transformers: {e}")
            import traceback
            logger.error(traceback.format_exc())
            raise
    
    def generate(
        self,
        text: Union[str, List[Dict[str, str]]],
        max_tokens: int = 512,
        temperature: float = 0.7,
        system_prompt: Optional[str] = None,
    ) -> str:
        """
        生成文本
        
        Args:
            text: 输入文本提示，可以是字符串或消息列表（格式：[{"role": "user", "content": "..."}]）
            max_tokens: 最大生成 token 数
            temperature: 温度参数
            system_prompt: 系统提示（可选）
        
        Returns:
            生成的文本
        """
        start_time = time.time()
        
        if self.mode == 'api':
            result = self._generate_with_api(text, max_tokens, temperature, system_prompt)
        elif self.mode == 'vllm':
            result = self._generate_with_vllm(text, max_tokens, temperature, system_prompt)
        elif self.mode == 'transformers':
            result = self._generate_with_transformers(text, max_tokens, temperature, system_prompt)
        else:
            raise RuntimeError("Model not initialized")
        
        elapsed_time = time.time() - start_time
        logger.info(f"Generation completed in {elapsed_time:.2f}s")
        
        return result
    
    def _generate_with_api(
        self,
        text: Union[str, List[Dict[str, str]]],
        max_tokens: int = 512,
        temperature: float = 0.7,
        system_prompt: Optional[str] = None,
    ) -> str:
        """使用 API 生成"""
        # 构建消息列表
        messages = []
        
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        
        if isinstance(text, str):
            messages.append({"role": "user", "content": text})
        elif isinstance(text, list):
            # 已经是消息列表格式
            messages.extend(text)
        else:
            raise ValueError(f"Unsupported text type: {type(text)}")
        
        # 调用 API
        resp = self._call_chat_completions(self.model_name, messages, temperature, max_tokens)

        if isinstance(resp, dict):
            try:
                return resp["choices"][0]["message"]["content"].strip()
            except Exception:
                return resp["choices"][0].get("text", "").strip()
        else:
            try:
                return resp.choices[0].message.content.strip()
            except Exception:
                return str(resp).strip()
    
    def _generate_with_vllm(
        self,
        text: Union[str, List[Dict[str, str]]],
        max_tokens: int = 512,
        temperature: float = 0.7,
        system_prompt: Optional[str] = None,
    ) -> str:
        """使用 vLLM 生成"""
        from vllm import SamplingParams
        
        # 将输入转换为字符串格式
        if isinstance(text, list):
            # 如果有 tokenizer，使用它来格式化消息
            if self.tokenizer and hasattr(self.tokenizer, 'apply_chat_template'):
                prompt = self.tokenizer.apply_chat_template(
                    text,
                    tokenize=False,
                    add_generation_prompt=True
                )
            else:
                # 简单拼接
                prompt = "\n".join([f"{msg['role']}: {msg['content']}" for msg in text])
        else:
            prompt = text
        
        if system_prompt:
            prompt = f"System: {system_prompt}\n\n{prompt}"
        
        sampling_params = SamplingParams(
            temperature=temperature,
            max_tokens=max_tokens,
            stop=None,
        )
        
        outputs = self.llm.generate([prompt], sampling_params)
        result = outputs[0].outputs[0].text.strip()
        
        return result
    
    def _generate_with_transformers(
        self,
        text: Union[str, List[Dict[str, str]]],
        max_tokens: int = 512,
        temperature: float = 0.7,
        system_prompt: Optional[str] = None,
    ) -> str:
        """使用 transformers 生成"""
        import torch
        
        # 构建消息
        messages = []
        
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        
        if isinstance(text, str):
            messages.append({"role": "user", "content": text})
        elif isinstance(text, list):
            messages.extend(text)
        else:
            raise ValueError(f"Unsupported text type: {type(text)}")
        
        # 使用 tokenizer 的 chat template
        if hasattr(self.tokenizer, 'apply_chat_template'):
            prompt = self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True
            )
        else:
            # 回退到简单格式
            prompt = "\n".join([f"{msg['role']}: {msg['content']}" for msg in messages])
        
        # Tokenize
        inputs = self.tokenizer(prompt, return_tensors="pt")
        
        # 移动到 GPU
        if torch.cuda.is_available():
            inputs = {k: v.cuda() for k, v in inputs.items()}
        
        # 生成
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_tokens,
                temperature=temperature,
                do_sample=temperature > 0,
                eos_token_id=self.tokenizer.eos_token_id if hasattr(self.tokenizer, 'eos_token_id') else None,
            )
        
        # 解码输出
        # 找到响应开始的位置（跳过输入部分）
        response_start_index = inputs.input_ids.shape[1]
        result = self.tokenizer.decode(
            outputs[0][response_start_index:],
            skip_special_tokens=True
        ).strip()
        
        # 清理输出：移除常见的模板标记
        result = re.sub(r'^assistant\s*:\s*', '', result, flags=re.IGNORECASE)
        result = re.sub(r'^user\s*:\s*', '', result, flags=re.IGNORECASE)
        result = result.strip()
        
        return result

