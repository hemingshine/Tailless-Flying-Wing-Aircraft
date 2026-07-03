import os
import math
import numpy as np
import matplotlib.pyplot as plt
from stable_baselines3 import PPO

# 导入你写好的环境构建函数
from train_innernew import make_env

def evaluate_stage1_model():
    print("=====================================================")
    print("  加载 Stage 1 (航向通道) 预训练模型进行闭环测试")
    print("=====================================================")
    
    # 初始化环境 (换一个非 42 的种子，测试它的泛化能力)
    env_creator = make_env(stage=1, seed=56)
    env = env_creator()
    
    model_path = "ppo_dir_stage1.zip"
    if not os.path.exists(model_path):
        print(f"❌ 找不到模型文件: {model_path}，请确认文件名和路径！")
        return
        
    print(f"✅ 成功加载模型: {model_path}")
    model = PPO.load("./logs/best_model_stage1/best_model.zip", device='cpu')
    
    obs, _ = env.reset()
    
    # 提取当前 episode 的随机目标指令
    target_beta = env.target_beta
    target_phi = env.target_phi
    target_theta = env.target_theta
    print(f"🎯 本次测试目标 -> 侧滑(Beta): {target_beta:.2f}°, 滚转(Roll): {target_phi:.2f}°, 俯仰(Pitch): {target_theta:.2f}°\n")
    
    # 数据记录器
    history = {
        't': [], 'beta': [], 'phi': [], 'theta': [], 
        'act': [], 'p': [], 'reward': []
    }
    
    total_reward = 0.0
    step_count = 0
    
    # 开始闭环仿真
    for step in range(20000):
        # 👑 deterministic=True 非常重要，测试时我们要看它确定的策略，而不是探索时的随机动作
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, _ = env.step(action)
        
        # 从底层读取真实物理状态
        state = env.sim.state
        V = max(math.sqrt(state[3]**2 + state[4]**2 + state[5]**2), 1.0)
        beta = math.degrees(math.asin(np.clip(state[4] / V, -1.0, 1.0)))
        phi = math.degrees(state[6])
        theta = math.degrees(state[7])
        p = math.degrees(state[9]) # 滚转率
        
        # 记录数据
        t = step * env.dt
        history['t'].append(t)
        history['beta'].append(beta)
        history['phi'].append(phi)
        history['theta'].append(theta)
        history['act'].append(action[0])
        history['p'].append(p)
        history['reward'].append(reward)
        
        total_reward += reward
        step_count += 1
        
        if terminated or truncated:
            print(f"⚠️ 仿真在第 {step} 步结束 (时长 {t:.2f}秒). Terminated: {terminated}, Truncated: {truncated}")
            break

    print(f"🏁 测试完成！存活步数: {step_count}/2000, 累计总奖励: {total_reward:.2f}")
    
    # ==========================================
    # 👑 绘制飞行核心数据图
    # ==========================================
    plt.figure(figsize=(15, 10))
    plt.style.use('bmh') # 使用一种比较好看的绘图风格
    
    # 1. 侧滑角跟踪 (最核心任务)
    plt.subplot(3, 2, 1)
    plt.plot(history['t'], history['beta'], 'b-', label='Actual Beta')
    plt.axhline(y=target_beta, color='r', linestyle='--', label='Target Beta (0°)')
    plt.title('Sideslip Angle (Beta) Tracking')
    plt.ylabel('Degrees')
    plt.legend()
    
    # 2. 阻力舵动作输出
    plt.subplot(3, 2, 2)
    plt.plot(history['t'], history['act'], 'g-', label='Network Output (Action)')
    plt.axhline(y=0, color='gray', linestyle='--')
    plt.title('Agent Action (Drag Rudder Command)')
    plt.ylabel('Normalized [-1, 1]')
    plt.legend()
    
    # 3. 滚转角 (受扰动通道，看 PD 兜底效果)
    plt.subplot(3, 2, 3)
    plt.plot(history['t'], history['phi'], 'm-', label='Actual Roll')
    plt.axhline(y=target_phi, color='r', linestyle='--', label='Target Roll')
    plt.title('Roll Angle (Phi) Disturbance Rejection')
    plt.ylabel('Degrees')
    plt.legend()
    
    # 4. 俯仰角 (未受训通道)
    plt.subplot(3, 2, 4)
    plt.plot(history['t'], history['theta'], 'c-', label='Actual Pitch')
    plt.axhline(y=target_theta, color='r', linestyle='--', label='Target Pitch')
    plt.title('Pitch Angle (Theta)')
    plt.ylabel('Degrees')
    plt.legend()
    
    # 5. 滚转率 (交叉耦合监控)
    plt.subplot(3, 2, 5)
    plt.plot(history['t'], history['p'], 'orange', label='Roll Rate (p)')
    plt.axhline(y=5.0, color='r', linestyle=':', label='Penalty Deadzone (+5)')
    plt.axhline(y=-5.0, color='r', linestyle=':', label='Penalty Deadzone (-5)')
    plt.title('Roll Rate (Cross-Coupling Monitor)')
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
    evaluate_stage1_model()