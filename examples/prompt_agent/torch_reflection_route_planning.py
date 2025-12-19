import os
import numpy as np
import time
import logging
from datetime import datetime
from agent_system.environments.env_manager import *
from openai import OpenAI
from typing import List, Tuple
import sys
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from reflection_prompt import (
    SIMPLE_ACTION_INSTRUCTION,
    REFLEXION_ACTION_INSTRUCTION,
    SELF_REFLECTION_ACTION_INSTRUCTION,
    EVALUATION_ACTION_INSTRUCTION,
    EVALUATION_FEW_SHOT,
    SELF_REFLECTION_FEW_SHOT,
    REFLEXION_FEW_SHOT,
)

# TODO: 模板的修改


# ======================= Build Envs =======================
def build_env(env_name, env_num=1):
    group_n = 1
    if env_name == "route_planning":
        from agent_system.environments.env_package.route_planning import route_projection
        from agent_system.environments.env_package.route_planning import build_route_envs

        envs = build_route_envs(
            seed=1,
            env_num=env_num,
            group_n=group_n,
            is_train=False,
        )
        env_manager = UrbanEnvironmentManager(envs, route_projection, "route_planning")
    else:
        raise ValueError(f"Unsupported environment name: {env_name}")
    return env_manager


# ======================= Reflexion Agent =======================
class ReflexionAgent:
    def __init__(self, model_name="gpt-4o", max_iters=2, verbose=True):
        self.model_name = model_name
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"正在使用设备: {self.device}")
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            torch_dtype="auto",  
            device_map="auto"
        )
        self.max_iters = max_iters
        self.verbose = verbose

        self.simple_action_instruction = SIMPLE_ACTION_INSTRUCTION
        self.reflexion_action_instruction = REFLEXION_ACTION_INSTRUCTION
        self.self_reflection_action_instruction = SELF_REFLECTION_ACTION_INSTRUCTION
        self.evaluation_action_instruction = EVALUATION_ACTION_INSTRUCTION
        self.evaluation_few_shot = EVALUATION_FEW_SHOT
        self.self_reflection_few_shot = SELF_REFLECTION_FEW_SHOT
        self.reflexion_few_shot = REFLEXION_FEW_SHOT

    def call_llm(self, messages: List[dict]) -> str:
        """封装的LLM调用方法"""
        prompt = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
        outputs = self.model.generate(
            **inputs,
            max_new_tokens=500,  
            eos_token_id=self.tokenizer.eos_token_id 
        )
        response_start_index = inputs.input_ids.shape[1]
        action = self.tokenizer.decode(
            outputs[0][response_start_index:],
            skip_special_tokens=True
        ).strip()
        
        return action

    def get_action(self, query: str) -> str:
        """
        使用Reflexion流程获取最终行动方案
        这个方法取代了原来的 get_action_from_gpt
        """
        mem = []  
        
        # 第一次尝试 (Simple generation)
        if self.verbose:
            logging.info("  (Agent) -> 初步尝试...")
        current_action = self._generate_action(query, "simple")
        
        # 第一次评估
        is_passing, feedback = self._evaluate_action(current_action, query)
        
        if self.verbose:
            logging.info(f"  (Agent) -> 初步评估结果: {'通过' if is_passing else '失败'}")
            logging.info(f"  (Agent) -> 反馈: {feedback}")

        # 如果第一次就通过了，直接返回
        if is_passing:
            return current_action
            
        # 如果未通过，则进入反思迭代循环
        for i in range(self.max_iters):
            if self.verbose:
                logging.info(f"  (Agent) -> --- 开始第 {i + 1}/{self.max_iters} 轮反思迭代 ---")
            
            # 1. 生成自我反思
            reflection = self._generate_self_reflection(current_action, feedback, query)
            mem.append(reflection)
            if self.verbose:
                logging.info(f"  (Agent) -> 生成反思: {reflection[:100]}...")

            # 2. 应用反思生成新行动
            current_action = self._generate_action(
                query,
                "reflexion",
                prev_action=current_action,
                feedback=feedback,
                reflection=reflection,
                reflections=mem
            )
            if self.verbose:
                logging.info(f"  (Agent) -> 生成新行动: {current_action}")
            
            # 3. 评估新行动
            is_passing, feedback = self._evaluate_action(current_action, query)
            if self.verbose:
                logging.info(f"  (Agent) -> 第 {i + 1} 轮评估结果: {'通过' if is_passing else '失败'}")
                logging.info(f"  (Agent) -> 反馈: {feedback}")

            # 如果通过，提前退出循环
            if is_passing:
                if self.verbose:
                    logging.info("  (Agent) -> 行动方案通过评估，结束迭代。")
                break
        
        return current_action

    def _generate_action(
        self,
        query: str,
        strategy: str,
        prev_action: str = None,
        feedback: str = None,
        reflection: str = None,
        reflections: List[str] = None
    ) -> str:
        """生成行动方案（分为简单和反思两种策略）"""
        if strategy == "simple":
            messages = [
                {"role": "system", "content": self.simple_action_instruction},
                {"role": "user", "content": query}
            ]
        elif strategy == "reflexion":
            reflection_context = "\n".join([f"- {r}" for r in reflections])
            # prompt = (
            #     f"你之前的行动方案 '{prev_action}' 收到了以下反馈: '{feedback}'.\n"
            #     f"这是你过去的思考总结:\n{reflection_context}\n\n"
            #     f"现在，请根据原始观察信息，提出一个更好的行动方案。\n"
            #     f"原始观察信息:\n{query}"
            # )
            # messages = [
            #     {"role": "system", "content": self.reflexion_action_instruction},
            #     {"role": "user", "content": prompt}
            # ]
            messages = [
                {"role": "system", "content": self.reflexion_action_instruction},
                {"role": "user", "content": self.reflexion_few_shot},
                {"role": "assistant", "content": prev_action},
                {"role": "user", "content": f"[evaluation feedback]:\n{feedback}\n\n[self-reflection]:"},
                {"role": "assistant", "content": reflection},
                {"role": "user", "content": f"[past reflections]:\n{reflection_context}\n\n[improved answer]:\n{query}"}
            ]
        else:
            raise ValueError(f"不支持的策略: {strategy}")
        
        return self.call_llm(messages)

    def _evaluate_action(self, action: str, query: str) -> Tuple[bool, str]:
        """使用LLM评估行动方案"""
        user_prompt = f"{self.evaluation_few_shot}\n\n[QUERY]:\n{query}\n\n[ACTION]:\n{action}\n\n[EVALUATION]:"
        messages = [
            {"role": "system", "content": self.evaluation_action_instruction},
            {"role": "user", "content": user_prompt}
        ]
        response = self.call_llm(messages)
        
        # 解析评估结果
        is_passing = "PASS" in response.upper()
        feedback = response.strip()
        return is_passing, feedback

    def _generate_self_reflection(self, action: str, feedback: str, query: str) -> str:
        """生成自我反思"""
        user_prompt = f"{self.self_reflection_few_shot}\n\n[QUERY]:\n{query}\n\n[ANSWER]:\n{action}\n\n[EVALUATION FEEDBACK]:\n{feedback}\n\n[SELF-REFLECTION]:"
        messages = [
            {"role": "system", "content": self.self_reflection_action_instruction},
            {"role": "user", "content": user_prompt}
        ]
        return self.call_llm(messages)


if __name__ == "__main__":
    # -------- logging (保持不变) ----------
    os.makedirs(f"logs/reflection/{sys.argv[1]}/route_planning", exist_ok=True)
    log_fp = os.path.join(
        f"logs/reflection/{sys.argv[1]}/route_planning", f"run_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    )
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(message)s",
        handlers=[
            logging.FileHandler(log_fp, encoding="utf-8"),
            logging.StreamHandler(),
        ],
    )

    # -------- Parameters ----------
    max_steps = 100
    env_num = 3
    test_times = 1
    env_name = "route_planning"

    # -------- Environment and agent setup ----------
    env_manager = build_env(env_name, env_num)
    agent = ReflexionAgent(model_name=sys.argv[1], max_iters=2)

    # ======================= Main Loop =======================
    for test_idx in range(test_times):
        logging.info(f"\n========== Start test {test_idx} ==========")
        start_time = time.time()

        obs, infos = env_manager.reset()
        env_dones = [False] * env_num
        env_end_infos = [{} for _ in range(env_num)]

        for step_idx in range(max_steps):
            logging.info(
                f"Step {step_idx + 1} / {max_steps}"
            )

            actions = []
            for i in range(env_num):
                if env_dones[i]:
                    actions.append("None")
                else:
                    env_obs = obs['text'][i] 
                    logging.info(f"--- Env {i} Observation --- \n{env_obs}")
                    response = agent.get_action(env_obs)
                    logging.info(f"--- Env {i} Final Action --- \n{response}")
                    actions.append(response)

            obs, rewards, dones, infos = env_manager.step(actions)

            for i in range(env_num):
                if env_dones[i]:
                    continue

                if dones[i]:
                    env_dones[i] = True
                    env_end_infos[i] = infos[i]

                if step_idx == max_steps - 1:
                    env_end_infos[i] = infos[i]

            if all(env_dones):
                logging.info("All environments finished early!")
                break

        # -------- Single round results --------
        won_values = []
        rate_average_travel_time = []
        rate_arrived_vehicles = []
        for i in range(env_num):
            won_value = int(env_end_infos[i]['average_travel_time'])
            won_value1 = int(env_end_infos[i]['arrived_vehicles'])
            logging.info(f'Env {i} success_rate: {won_value}')  
            logging.info(f'Env {i} success_rate_average_travel_time: {won_value}')
            logging.info(f'Env {i} success_rate_arrived_vehicles: {won_value1}')
            won_values.append(won_value)
            rate_average_travel_time.append(won_value)
            rate_arrived_vehicles.append(won_value1)

        logging.info(f"Test {test_idx} time elapsed: {time.time() - start_time:.2f}s\n")

        logging.info("=============== Single Test Summary ===============")
        logging.info(f"Overall success rate avg ± std: {np.mean(won_values):.4f} ± {np.std(won_values):.4f}")
        logging.info(f"Overall rate_average_travel_time avg ± std: {np.mean(rate_average_travel_time):.4f} ± {np.std(rate_average_travel_time):.4f}")
        logging.info(f"Overall rate_arrived_vehicles avg ± std: {np.mean(rate_arrived_vehicles):.4f} ± {np.std(rate_arrived_vehicles):.4f}")
            