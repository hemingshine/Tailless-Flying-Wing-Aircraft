from flight_simulate import FlightSimulator6DOF, HybridAeroDatabase, EngineDatabase
from PPO_env import FlightEnvRL,PPOAgent
import torch
import matplotlib.pyplot as plt
import numpy as np

def plot_reward_curve(rewards, window_size=10, title="飞行模拟PPO训练奖励曲线"):
    # 1. 计算滑动平均（平滑曲线，避免抖动）
    rewards_np = np.array(rewards)
    if len(rewards_np) < window_size:
        smoothed = rewards_np
    else:
        # 滑动平均计算
        smoothed = np.convolve(rewards_np, np.ones(window_size)/window_size, mode='valid')
    
    # 2. 绘制原始曲线 + 平滑曲线
    plt.figure(figsize=(10, 6))
    # 原始奖励（浅灰色，透明）
    plt.plot(rewards_np, color='lightgray', alpha=0.5, label='原始奖励')
    # 平滑奖励（深蓝色，加粗）
    plt.plot(range(window_size-1, len(rewards_np)), smoothed, color='darkblue', linewidth=2, label=f'滑动平均（窗口={window_size}）')
    
    # 3. 美化图表（适配飞行模拟场景）
    plt.xlabel("训练回合数", fontsize=12)
    plt.ylabel("每回合总奖励", fontsize=12)
    plt.title(title, fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=10)
    plt.tight_layout()  # 自动调整布局
    plt.show()


if __name__ == '__main__':
    # 1. 初始化气动与发动机数据库（以你自己的为准）
    # aircraft_params = {
    #     'S': 3.857, 'b': 4.2, 'c_bar': 1.380462, 'mass': 14000,
    #     'Ixx': 313220.6, 'Iyy': 273435.3, 'Izz': 392903.9, 'Ixz': 0.0
    # }
    aircraft_params = {
        'S': 88.58, 'b': 18.9, 'c_bar': 4.6, 'mass': 14000,
        'Ixx': 313220.6, 'Iyy': 273435.3, 'Izz': 392903.9, 'Ixz': 0.0
    }
    flight_db = HybridAeroDatabase()
    engine_db = EngineDatabase()
    flight_db._load_from_pickle('X47B.pkl')
    engine_db.load1('engine.pkl')
    model_codes = []    
    # 定义所有可供 AI 切换的模型代号
    for i in range(1,47):
        if i<10:
            model_codes.append(f'state0{i}')
        else:
            model_codes.append(f'state{i}')
    
    # 2. 实例化环境和 PPO 智能体
    env = FlightEnvRL(flight_db, engine_db, aircraft_params, model_codes, max_steps=2000, dt=0.02)
    
    # 状态维度 10， 动作维度取决于模型代号的数量
    agent = PPOAgent(state_dim=env.state_dim, action_dim=env.action_dim, lr=3e-4)
    
    # 3. 训练参数
    max_episodes = 5000       # 训练总回合数
    update_timestep = 800    # 每收集 4000 步数据，进行一次 PPO 网络更新 (On-policy)
    
    time_step = 0
    rewards=[]
    # 开始训练！
    for ep in range(1, max_episodes + 1):
        state = env.reset()
        current_ep_reward = 0
        
        while True:
            # AI 做出动作选择
            action = agent.select_action(state)
            
            # 环境推演一步
            state, reward, done = env.step(action)
            
            # 将环境反馈存入 Buffer
            agent.buffer.rewards.append(reward)
            agent.buffer.is_terminals.append(done)
            
            time_step += 1
            current_ep_reward += reward
            
            # 收集足够多的数据后，触发 PPO 学习更新
            if time_step % update_timestep == 0:
                print(f"[{time_step} steps] 触发 PPO 网络更新...")
                agent.update()
                
            if done:
                break
            
        rewards.append(current_ep_reward)
        # 打印训练日志 (可以配合 TensorboardX 观察损失和奖励的收敛)
        if ep % 10 == 0:
            print(f"Episode: {ep} \t Reward: {current_ep_reward:.2f}")
    plot_reward_curve(rewards)
    # 保存模型
    torch.save(agent.policy.state_dict(), 'PPO_Flight_Policy.pth')
    print("模型训练完毕并已保存！")