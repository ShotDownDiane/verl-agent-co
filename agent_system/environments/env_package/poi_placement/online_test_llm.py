from online_env_tensor import StationPlacement  # 根据你项目路径修改
from online_env_llm import LLMWrapperEnv  # 根据你项目路径修改
import pprint
import random

# 环境初始化
location = "Qiaonan"
graph_file = f"Data/Graph/{location}/{location}.graphml"
node_file = f"Data/Graph/{location}/nodes_extended_{location}.txt"
plan_file = f"Data/Graph/{location}/existingplan_{location}.pkl"

base_env = StationPlacement(graph_file, node_file, plan_file)
env = LLMWrapperEnv(base_env)
env.reset()

history = []
done = False
step_count = 0

print("=== 手动充电站安装测试 ===")
print("说明：输入一个合法的 node_id（来自 node_list），和充电桩配置（如 0 0 3 表示安装3个fast充电桩）")

while not done:

    try:
        answer = random.choice(["A","B","C","D","E"])
        llm_action = {
            "summary": "Let's create a new station at a location with low coverage.",
            "answer": answer  # 对应建站选项
        }
        obs, reward, terminated, truncated, info = env.step(llm_action)
        print(f"✅ 安装完成，费用: {reward}，剩余预算: {env.base_env.budget:.2f}")
        print(f"当前 observation（部分）：{obs}")
        if terminated or truncated:
            done = True
            print("游戏结束！")
            print(f"最终得分:{env.base_env.best_score}")

        step_count += 1

    except Exception as e:
        print(f"❌ 动作执行失败: {e}")
        exit(0)

print(f"\n=== 测试结束，共执行 {step_count} 步 ===")
