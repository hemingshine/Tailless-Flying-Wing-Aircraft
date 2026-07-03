import os
import math
import numpy as np
import matplotlib.pyplot as plt
from stable_baselines3 import PPO

# 导入你的环境构建函数
from train_innernew import make_env

def evaluate_stage2_model():
    print("=====================================================")
    print("  加载 Stage 2 (横向通道) 预训练模型进行闭环测试")
    print("=====================================================")
    
    # 检查所需的两个模型文件是否存在
    # dir_model_path = "ppo_dir_stage1.zip"
    # lat_model_path = "ppo_lat_stage2.zip"
    
    # if not os.path.exists(dir_model_path):
    #     print(f"❌ 找不到 Stage 1 模型: {dir_model_path}")
    #     return
    # if not os.path.exists(lat_model_path):
    #     print(f"❌ 找不到 Stage 2 模型: {lat_model_path}")
    #     return

    # 实例化 Stage 2 环境，必须注入 Stage 1 的模型路径
    env_creator = make_env(stage=2, model_paths={'dir': 'ppo_dir_stage1'}, seed=44234)
    env = env_creator()
    
    print(f"✅ 成功加载环境与前置模型")
    model = PPO.load("./logs/best_model_stage2/best_model.zip", device='cpu')
    print(f"✅ 成功加载 Stage 2 测试模型")
    
    obs, _ = env.reset()
    
    # 提取当前 episode 的随机目标指令
    target_beta = env.target_beta
    target_phi = env.target_phi
    target_theta = env.target_theta
    print(f"🎯 本次测试目标 -> 侧滑(Beta): {target_beta:.2f}°, 滚转(Roll): {target_phi:.2f}°, 俯仰(Pitch): {target_theta:.2f}°\n")
    
    # 数据记录器
    history = {
        't': [], 'phi': [], 'beta': [], 'theta': [], 
        'act_a': [], 'r': [], 'reward': []
    }
    
    total_reward = 0.0
    step_count = 0
    
    # 开始闭环仿真 (20秒)
    for step in range(2000):
        # 👑 deterministic=True，关闭探索噪声，测试绝对性能
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, _ = env.step(action)
        
        # 从底层读取真实物理状态
        state = env.sim.state
        V = max(math.sqrt(state[3]**2 + state[4]**2 + state[5]**2), 1.0)
        beta = math.degrees(math.asin(np.clip(state[4] / V, -1.0, 1.0)))
        phi = math.degrees(state[6])
        theta = math.degrees(state[7])
        r = math.degrees(state[11]) # 偏航率
        
        # 记录数据
        t = step * env.dt
        history['t'].append(t)
        history['phi'].append(phi)
        history['beta'].append(beta)
        history['theta'].append(theta)
        history['act_a'].append(action[0]) # 副翼网络输出
        history['r'].append(r)
        history['reward'].append(reward)
        
        total_reward += reward
        step_count += 1
        
        if terminated or truncated:
            print(f"⚠️ 仿真在第 {step} 步结束 (时长 {t:.2f}秒). Terminated: {terminated}, Truncated: {truncated}")
            break

    print(f"🏁 测试完成！存活步数: {step_count}/2000, 累计总奖励: {total_reward:.2f}")
    
    # ==========================================
    # 👑 绘制飞行核心数据图 (针对 Stage 2 定制)
    # ==========================================
    plt.figure(figsize=(15, 10))
    plt.style.use('bmh')
    
    # 1. 滚转角跟踪 (Stage 2 核心任务)
    plt.subplot(3, 2, 1)
    plt.plot(history['t'], history['phi'], 'm-', label='Actual Roll (Phi)')
    plt.axhline(y=target_phi, color='r', linestyle='--', label=f'Target ({target_phi:.1f}°)')
    plt.title('Roll Angle Tracking (Primary Objective)')
    plt.ylabel('Degrees')
    plt.legend()
    
    # 2. 副翼动作输出
    plt.subplot(3, 2, 2)
    plt.plot(history['t'], history['act_a'], 'g-', label='Network Output (Aileron)')
    plt.axhline(y=0, color='gray', linestyle='--')
    plt.title('Agent Action (Aileron Command)')
    plt.ylabel('Normalized [-1, 1]')
    plt.legend()
    
    # 3. 侧滑角 (Stage 1 辅助抑制效果)
    plt.subplot(3, 2, 3)
    plt.plot(history['t'], history['beta'], 'b-', label='Actual Beta')
    plt.axhline(y=target_beta, color='r', linestyle='--', label='Target Beta (0°)')
    plt.title('Sideslip Angle (Stage 1 Suppression Check)')
    plt.ylabel('Degrees')
    plt.legend()
    
    # 4. 俯仰角 (PID 兜底效果)
    plt.subplot(3, 2, 4)
    plt.plot(history['t'], history['theta'], 'c-', label='Actual Pitch')
    plt.axhline(y=target_theta, color='r', linestyle='--', label=f'Target ({target_theta:.1f}°)')
    plt.title('Pitch Angle (PID Baseline Check)')
    plt.ylabel('Degrees')
    plt.legend()
    
    # 5. 偏航率 (交叉耦合监控)
    plt.subplot(3, 2, 5)
    plt.plot(history['t'], history['r'], 'orange', label='Yaw Rate (r)')
    plt.axhline(y=2.0, color='r', linestyle=':', label='Penalty Deadzone (+2)')
    plt.axhline(y=-2.0, color='r', linestyle=':', label='Penalty Deadzone (-2)')
    plt.title('Yaw Rate (Adverse Yaw Monitor)')
    plt.xlabel('Time (s)')
    plt.ylabel('Deg/s')
    plt.legend()
    
    # 6. 实时奖励值
    plt.subplot(3, 2, 6)
    plt.plot(history['t'], history['reward'], 'k-', label='Step Reward')
    plt.title('Reward Signal over Time')
    plt.xlabel('Time (s)')
    plt.legend()
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    evaluate_stage2_model()