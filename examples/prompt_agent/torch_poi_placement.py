import os
import numpy as np
import time
import logging
from datetime import datetime
from agent_system.environments.env_manager import *
from openai import OpenAI
import sys
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def build_env(env_name, env_num=1):
    group_n = 1
    if env_name == "poi_placement":
        from agent_system.environments.env_package.poi_placement import poi_projection
        from agent_system.environments.env_package.poi_placement import build_poi_envs

        envs = build_poi_envs(
            seed=1,
            env_num=env_num,
            group_n=group_n,
            is_train=False,
            env_kwargs={"location": "Qiaonan"},
        )
        env_manager = UrbanEnvironmentManager(envs, poi_projection, "poi_placement")
    else:
        raise ValueError(f"Unsupported environment name: {env_name}")
    return env_manager


class Agent:
    def __init__(self, model_name="gpt-4o"):
        # self.model_name = model_name
        # self.client = OpenAI(
        #     api_key=os.environ.get("SILICONFLOW_API_KEY"),
        #     base_url="https://api.siliconflow.cn/v1",
        # )
        self.model_name = model_name
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"正在使用设备: {self.device}")
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            torch_dtype="auto",  
            device_map="auto"
        )

    def get_action_from_gpt(self, obs):
        # response = self.client.chat.completions.create(
        #     model=self.model_name,
        #     messages=[{"role": "user", "content": obs}],
        #     temperature=0.4,
        #     n=1,
        #     stop=None,
        # )
        # action = response.choices[0].message.content.strip()
        # return action

        # 步骤 1: 生成思考计划
        messages = [{"role": "user", "content": obs}]
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
        plan = self.tokenizer.decode(
            outputs[0][response_start_index:],
            skip_special_tokens=True
        ).strip()
        logging.info(f"生成的计划: {plan}")

        # 步骤 2: 组合观察和计划
        combined_input = f"观察: {obs}\n计划: {plan}\n根据上述观察和计划，确定下一步行动是什么？"

        # 步骤 3: 获取最终行动
        messages = [{"role": "user", "content": combined_input}]
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
        logging.info(f"最终行动: {action}")

        return action


if __name__ == "__main__":
    # -------- logging ----------
    os.makedirs(f"logs/{sys.argv[1]}/poi_placement", exist_ok=True)
    log_fp = os.path.join(
        f"logs/{sys.argv[1]}/poi_placement", f"run_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
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
    env_name = "poi_placement"

    # -------- Environment and agent setup ----------
    env_manager = build_env(env_name, env_num)
    agent = Agent(model_name=sys.argv[1])

    # ======================= Main Loop =======================
    for test_idx in range(test_times):
        logging.info(f"\n========== Start test {test_idx} ==========")
        start_time = time.time()

        obs, infos = env_manager.reset()
        env_dones = [False] * env_num
        env_end_infos = [{} for _ in range(env_num)]

        for step_idx in range(max_steps):
            logging.info(
                f"Step {step_idx + 1} / {max_steps} | "
            )

            # --- Assemble actions ---
            actions = []

            for i in range(env_num):
                if env_dones[i]:
                    actions.append("None")
                else:
                    env_obs = obs['text'][i] 
                    logging.info(f"Env {i} Observation: {env_obs}")
                    response = agent.get_action_from_gpt(env_obs)
                    logging.info(f"Env {i} Response: {response}")
                    actions.append(response)

            # --- Environment stepping ---
            obs, rewards, dones, infos = env_manager.step(actions)

            # --- Determine endings and successes ---
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
        won_values1 = []
        won_values2 = []
        won_values3 = []
        won_values4 = []
        won_values5 = []
        for i in range(env_num):
            won_value = env_end_infos[i]['benefit_info']['score']
            won_value1 = env_end_infos[i]['benefit_info']['cov']
            won_value2 = env_end_infos[i]['benefit_info']['cos']
            won_value3 = env_end_infos[i]['benefit_info']['wat']
            won_value4 = env_end_infos[i]['benefit_info']['tra']
            won_value5 = env_end_infos[i]['benefit_info']['chg']
            logging.info(f'Env {i} success_rate: {won_value}')
            logging.info(f'Env {i} success_rate_coverage: {won_value1}')
            logging.info(f'Env {i} success_rate_cost: {won_value2}')
            logging.info(f'Env {i} success_rate_waiting_time: {won_value3}')
            logging.info(f'Env {i} success_rate_travel_time: {won_value4}')
            logging.info(f'Env {i} success_rate_charging_time: {won_value5}')
            won_values.append(won_value)
            won_values1.append(won_value1)
            won_values2.append(won_value2)
            won_values3.append(won_value3)
            won_values4.append(won_value4)
            won_values5.append(won_value5)

        logging.info(f"Test {test_idx} time elapsed: {time.time() - start_time:.2f}s\n")

        logging.info("=============== Single Test Summary ===============")
        logging.info(f"Overall success rate avg ± std: {np.mean(won_values):.4f} ± {np.std(won_values):.4f}")
        logging.info(f"Overall rate_coverage avg ± std: {np.mean(won_values1):.4f} ± {np.std(won_values1):.4f}")
        logging.info(f"Overall rate_cost avg ± std: {np.mean(won_values2):.4f} ± {np.std(won_values2):.4f}")
        logging.info(f"Overall rate_waiting_time avg ± std: {np.mean(won_values3):.4f} ± {np.std(won_values3):.4f}")
        logging.info(f"Overall rate_travel_time avg ± std: {np.mean(won_values4):.4f} ± {np.std(won_values4):.4f}")
        logging.info(f"Overall rate_charging_time avg ± std: {np.mean(won_values5):.4f} ± {np.std(won_values5):.4f}")

    # ======================= Final Summary =======================
    # logging.info("=============== Final Summary ===============")
    # logging.info(
    #     f"Total tests: {test_times} | Envs / test: {env_num} | Total envs: {env_num * test_times}"
    # )
    # logging.info(
    #     f"Overall success avg ± std: "
    #     f"{np.mean(overall_success_rates):.4f} ± {np.std(overall_success_rates):.4f}"
    # )
