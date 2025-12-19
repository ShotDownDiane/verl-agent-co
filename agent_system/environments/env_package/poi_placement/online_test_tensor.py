from online_env_tensor import StationPlacement  # 根据你项目路径修改
import pprint
import random

# 环境初始化
location = "Qiaonan"
graph_file = f"Data/Graph/{location}/{location}.graphml"
node_file = f"Data/Graph/{location}/nodes_extended_{location}.txt"
plan_file = f"Data/Graph/{location}/existingplan_{location}.pkl"

env = StationPlacement(graph_file, node_file, plan_file)
env.reset(seed=42)  # 设置随机种子以确保可重复性

history = []
done = False
step_count = 0

print("=== 手动充电站安装测试 ===")
print("说明：输入一个合法的 node_id（来自 node_list），和充电桩配置（如 0 0 3 表示安装3个fast充电桩）")

while not done:
    try:
        answer = random.choice([0,1,2,3,4])
        # answer = 4
        user_input = str(answer)  # 模拟用户输入，实际使用时可以替换为 input() 函数获取用户输入
        if user_input.lower() in ["q", "quit", "exit"]:
            break
        parts = user_input.strip().split()

        action = int(parts[0])
        obs, reward, terminated, truncated, info = env.step(action)
        print(f"✅ 安装完成，费用: {reward}，剩余预算: {env.budget:.2f}")
        print(f"当前 observation（部分）：{obs}")
        if terminated or truncated:
            done = True
            print("游戏结束！")
            print(f"最终得分:{env.best_score}")

        step_count += 1

    except Exception as e:
        print(f"❌ 动作执行失败: {e}")
        exit(0)

print(f"\n=== 测试结束，共执行 {step_count} 步 ===")
