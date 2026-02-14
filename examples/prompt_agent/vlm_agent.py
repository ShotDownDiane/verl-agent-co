"""
简单的 VLM Agent 类
支持通过 API、vLLM 或 transformers 加载，接受文本和图片输入
"""
import os
import logging
import time
import base64
import numpy as np
from typing import Optional, Union, List
from io import BytesIO
from PIL import Image

logger = logging.getLogger(__name__)


class VLMAgent:
    """简单的 VLM Agent，支持多种加载方式"""
    
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
        初始化 VLM Agent
        
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
        self.processor = None
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
                self.model_name = self.model_name or "vlm"
                self.mode = 'api'
                logger.info("✓ VLM Agent initialized with provided api_client")
                return

            from openai import OpenAI

            self.client = OpenAI(
                api_key=self.api_key or "token-abc123456",
                base_url=self.api_base_url,
                max_retries=3,
                timeout=240.0, # 如果是vllm的话应该更长
            )
            self.model_name = self.model_name or "vlm"
            self.mode = 'api'
            logger.info(f"✓ VLM Agent initialized with API: {self.api_base_url}")
            
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
        except Exception as e:
            # Only ignore if it's an AttributeError (meaning client doesn't support this)
            # If it's a API error (e.g. AuthenticationError), we should probably raise it or log it
            logger.warning(f"OpenAI-like call failed: {e}")
            # If it was an OpenAI client, falling back to 'post' is risky if the error was unrelated to method existence
            if "got an unexpected keyword argument" in str(e) or "object has no attribute" in str(e):
                 pass # Fallback to next method
            else:
                 raise e # Re-raise real API errors

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
            
            logger.info(f"Loading VLM with vLLM from: {self.model_path}")
            logger.info(f"Config: {default_config}")
            
            self.llm = LLM(model=self.model_path, **default_config)
            self.mode = 'vllm'
            logger.info("✓ VLM loaded successfully with vLLM")
            
        except Exception as e:
            logger.error(f"Failed to load with vLLM: {e}")
            logger.info("Falling back to transformers...")
            self._init_with_transformers()
    
    def _init_with_transformers(self):
        """使用 transformers 初始化"""
        try:
            from transformers import AutoProcessor, AutoConfig
            import torch
            
            logger.info(f"Loading VLM with transformers from: {self.model_path}")
            
            # 检查模型类型
            config = AutoConfig.from_pretrained(
                self.model_path,
                trust_remote_code=True
            )
            
            logger.info(f"Model type: {config.model_type}, Architectures: {getattr(config, 'architectures', 'N/A')}")
            
            # 根据模型类型选择合适的模型类
            if config.model_type == "qwen3_vl":
                from transformers import Qwen3VLForConditionalGeneration
                model_class = Qwen3VLForConditionalGeneration
            else:
                # 回退到 AutoModel
                from transformers import AutoModel
                model_class = AutoModel
            
            self.processor = AutoProcessor.from_pretrained(
                self.model_path,
                trust_remote_code=True
            )
            
            self.model = model_class.from_pretrained(
                self.model_path,
                trust_remote_code=True,
                dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
                device_map="auto",
            )
            
            self.mode = 'transformers'
            logger.info("✓ VLM loaded successfully with transformers")
            
        except Exception as e:
            logger.error(f"Failed to load with transformers: {e}")
            import traceback
            logger.error(traceback.format_exc())
            raise
    
    def _image_to_base64(self, image: Union[str, Image.Image, bytes]) -> str:
        """将图片转换为 base64 字符串"""
        if isinstance(image, str):
            # 如果是文件路径
            if os.path.exists(image):
                with open(image, 'rb') as f:
                    return base64.b64encode(f.read()).decode('utf-8')
            # 如果已经是 base64 字符串
            return image
        elif isinstance(image, Image.Image):
            # PIL Image
            buf = BytesIO()
            image.save(buf, format='PNG')
            return base64.b64encode(buf.getvalue()).decode('utf-8')
        elif isinstance(image, bytes):
            # 字节数据
            return base64.b64encode(image).decode('utf-8')
        elif isinstance(image, np.ndarray):
            # NumPy 数组 (通常是 RGB)
            buf = BytesIO()
            pil_img = Image.fromarray(image.astype('uint8'))
            pil_img.save(buf, format='PNG')
            return base64.b64encode(buf.getvalue()).decode('utf-8')
        else:
            raise ValueError(f"Unsupported image type: {type(image)}")
    
    def generate(
        self,
        system_prompt: str,
        text: str,
        image: Optional[Union[str, Image.Image, bytes]] = None,
        max_tokens: int = 512,
        temperature: float = 0.7,
    ) -> str:
        """
        生成文本（支持文本和图片输入）
        
        Args:
            system_prompt: 系统提示
            text: 输入文本提示
            image: 输入图片（可以是文件路径、PIL Image、bytes 或 base64 字符串）
            max_tokens: 最大生成 token 数
            temperature: 温度参数
        
        Returns:
            生成的文本
        """
        start_time = time.time()
        
        if self.mode == 'api':
            result = self._generate_with_api(system_prompt, text, image, max_tokens, temperature)
        elif self.mode == 'vllm':
            result = self._generate_with_vllm(system_prompt, text, image, max_tokens, temperature)
        elif self.mode == 'transformers':
            result = self._generate_with_transformers(system_prompt, text, image, max_tokens, temperature)
        else:
            raise RuntimeError("Model not initialized")
        
        elapsed_time = time.time() - start_time
        logger.info(f"Generation completed in {elapsed_time:.2f}s")
        
        return result
        
    def batch_generate(
        self,
        system_prompts: Union[str, List[str]],
        texts: List[str],
        images: Optional[List[Union[str, Image.Image, bytes]]] = None,
        max_tokens: int = 512,
        temperature: float = 0.7,
        max_workers: int = 16
    ) -> List[str]:
        """
        批量生成文本
        """
        start_time = time.time()
        batch_size = len(texts)
        
        # Normalize system_prompts
        if isinstance(system_prompts, str):
            system_prompts = [system_prompts] * batch_size
            
        # Normalize images
        if images is None:
            images = [None] * batch_size
        
        if len(system_prompts) != batch_size or len(images) != batch_size:
            raise ValueError("Length of system_prompts, texts, and images must match")

        results = []
        
        if self.mode == 'api':
            # Use ThreadPoolExecutor for parallel API calls
            from concurrent.futures import ThreadPoolExecutor
            
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                futures = []
                for i in range(batch_size):
                    futures.append(
                        executor.submit(
                            self._generate_with_api,
                            system_prompts[i],
                            texts[i],
                            images[i],
                            max_tokens,
                            temperature
                        )
                    )
                
                for future in futures:
                    try:
                        results.append(future.result())
                    except Exception as e:
                        logger.error(f"Batch generation failed for one item: {e}")
                        results.append("") # Return empty string on failure
                        
        elif self.mode == 'vllm':
            # Use native batch generation for vLLM
            # Note: _generate_with_vllm currently ignores images
            from vllm import SamplingParams
            
            if any(img is not None for img in images):
                logger.warning("vLLM mode: image input not fully supported in batch, using text only")
                
            sampling_params = SamplingParams(
                temperature=temperature,
                max_tokens=max_tokens,
            )
            
            # TODO: Add system prompt handling for vLLM if needed (currently just passing text)
            outputs = self.llm.generate(texts, sampling_params)
            results = [output.outputs[0].text.strip() for output in outputs]
            
        elif self.mode == 'transformers':
             # Sequential loop for transformers (safest for now)
             for i in range(batch_size):
                 try:
                     res = self._generate_with_transformers(
                         system_prompts[i],
                         texts[i],
                         images[i],
                         max_tokens,
                         temperature
                     )
                     results.append(res)
                 except Exception as e:
                     logger.error(f"Batch generation failed for item {i}: {e}")
                     results.append("")
        
        elapsed_time = time.time() - start_time
        logger.info(f"Batch generation of {batch_size} items completed in {elapsed_time:.2f}s")
        
        return results
    
    def _generate_with_api(
        self,
        system_prompt: str,
        text: str,
        image: Optional[Union[str, Image.Image, bytes]] = None,
        max_tokens: int = None,
        temperature: float = 0.7,
    ) -> str:
        """使用 API 生成"""
        messages = []
        
        # 构建消息内容
        if image is not None:
            # 有图片的情况
            image_base64 = self._image_to_base64(image)
            content = [
                {"type": "text", "text": text},
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/png;base64,{image_base64}",
                        "detail":"low"
                    }
                }
            ]
        else:
            # 只有文本
            content = text
        
        # 添加系统提示
        messages.append({"role": "system", "content": system_prompt})   
        messages.append({"role": "user", "content": content})
        
        # 调用 API（统一入口，兼容多种客户端）
        
        resp = self._call_chat_completions(self.model_name, messages, temperature, max_tokens)

        # 兼容不同返回类型
        if isinstance(resp, dict):
            # JSON response
            try:
                return resp["choices"][0]["message"]["content"].strip()
            except Exception:
                # Older style
                return resp["choices"][0]["text"].strip()
        else:
            # object-like response (openai.OpenAI)
            try:
                message = resp.choices[0].message
                content = message.content
                
                # Handle case where content is None (e.g. reasoning models or truncation)
                if content is None:
                    if hasattr(message, 'reasoning_content') and message.reasoning_content:
                        # If we have reasoning content, return it (better than nothing)
                        content = message.reasoning_content
                    else:
                        content = ""
                
                return content.strip()
            except Exception:
                # fallback: string
                return str(resp).strip()
    
    def _generate_with_vllm(
        self,
        system_prompt: str,
        text: str,
        image: Optional[Union[str, Image.Image, bytes]] = None,
        max_tokens: int = 512,
        temperature: float = 0.7,
    ) -> str:
        """使用 vLLM 生成"""
        from vllm import SamplingParams
        
        # vLLM 目前对多模态的支持可能有限，这里先处理文本
        if image is not None:
            logger.warning("vLLM mode: image input may not be fully supported, using text only")
        
        sampling_params = SamplingParams(
            temperature=temperature,
            max_tokens=max_tokens,
            stop=None,
        )
        
        outputs = self.llm.generate([text], sampling_params)
        result = outputs[0].outputs[0].text.strip()
        
        return result
    
    def _generate_with_transformers(
        self,
        system_prompt: str,
        text: str,
        image: Optional[Union[str, Image.Image, bytes]] = None,
        max_tokens: int = None,
        temperature: float = 0.7,
    ) -> str:
        """使用 transformers 生成"""
        import torch
        
        # 处理图片输入
        if image is not None:
            # 如果是字符串，尝试加载为图片
            if isinstance(image, str) and os.path.exists(image):
                image = Image.open(image)
            elif isinstance(image, str):
                # 假设是 base64
                try:
                    image_data = base64.b64decode(image)
                    image = Image.open(BytesIO(image_data))
                except:
                    raise ValueError(f"Cannot decode image from string: {image}")
            elif isinstance(image, bytes):
                image = Image.open(BytesIO(image))
            elif isinstance(image, np.ndarray):
                image = Image.fromarray(image.astype('uint8'))
            
            # 处理多模态输入
            inputs = self.processor(
                text=text,
                images=image,
                return_tensors="pt"
            )
        else:
            # 只有文本
            try:
                # 尝试使用 messages 格式（Qwen3VL 推荐的方式）
                if hasattr(self.processor, 'apply_chat_template'):
                    messages = [{"role": "user", "content": text}]
                    text_formatted = self.processor.apply_chat_template(
                        messages, tokenize=False, add_generation_prompt=True
                    )
                    inputs = self.processor(text=[text_formatted], return_tensors="pt", padding=True)
                else:
                    inputs = self.processor(text=text, return_tensors="pt")
            except Exception:
                # 回退到简单方式
                inputs = self.processor(text=text, return_tensors="pt")
        
        # 移动到 GPU
        if torch.cuda.is_available():
            inputs = {
                k: v.cuda() if isinstance(v, torch.Tensor) else v
                for k, v in inputs.items()
            }
        
        # 生成
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_tokens,
                temperature=temperature,
                do_sample=temperature > 0,
            )
        
        # 解码输出
        if hasattr(self.processor, 'decode'):
            result = self.processor.decode(outputs[0], skip_special_tokens=True)
        elif hasattr(self.processor, 'tokenizer'):
            result = self.processor.tokenizer.decode(outputs[0], skip_special_tokens=True)
        else:
            raise RuntimeError("Cannot decode outputs: processor has no decode method")
        
        # 清理输出：移除原始 prompt 和模板文本
        if text in result:
            result = result.replace(text, "").strip()
        
        # 移除常见的模板标记
        import re
        result = re.sub(r'^user\s*\n\s*assistant\s*\n', '', result, flags=re.IGNORECASE | re.MULTILINE)
        result = re.sub(r'^assistant\s*\n', '', result, flags=re.IGNORECASE | re.MULTILINE)
        result = result.strip()
        
        return result

