import torch
import numpy as np
import math
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# 导入你之前的环境、网络和数据库类
from PPO_env import FlightEnvRL, ActorCritic
from flight_simulate import HybridAeroDatabase, EngineDatabase

def test_trained_model(model_path, env, model_codes):
    """
    加载训练好的模型并进行一次完整的试飞测试
    """
    print(f"正在加载模型: {model_path} ...")
    
    # 1. 初始化相同的网络结构
    state_dim = env.state_dim
    action_dim = env.action_dim
    policy = ActorCritic(state_dim, action_dim)
    
    # 2. 加载权重并设置为评估模式 (eval)
    policy.load_state_dict(torch.load(model_path))
    policy.eval() # 关闭 Dropout, BatchNorm 等训练专用机制
    
    # 3. 初始化测试环境
    state = env.reset()
    done = False
    total_reward = 0.0
    
    # --- 用于画图的“黑匣子”数据 ---
    history_time = []
    history_alt = []
    history_vel = []
    history_pitch = []
    history_alpha = []
    history_pn = []
    history_pe = []
    history_actions = [] # 记录 AI 每步选择的模型代号索引
    
    print("开始自动驾驶飞行测试...")
    
    while not done:
        # 4. 关键：确定性动作选择 (Deterministic Action)
        # 在测试时，我们不使用 Categorical.sample() 抽样，而是直接选概率最大的动作
        state_tensor = torch.from_numpy(state).float().unsqueeze(0)
        with torch.no_grad():
            action_probs = policy.actor(state_tensor)
            # 取出概率最高的动作索引
            action = torch.argmax(action_probs, dim=-1).item() 
            
        # 5. 在环境中执行动作
        next_state, reward, done = env.step(action)
        
        # 6. 提取并记录物理参数 (从环境的 sim 对象中拿未归一化的真实数据)
        sim_state = env.sim.state
        pn, pe, pd, u, v, w, phi, theta, psi, p, q, r = sim_state
        
        h = -pd
        V = math.sqrt(u**2 + v**2 + w**2)
        alpha = math.degrees(math.atan2(w, u)) if u != 0 else 0.0
        pitch = math.degrees(theta)
        
        history_time.append(env.current_step * env.dt)
        history_alt.append(h)
        history_vel.append(V)
        history_alpha.append(alpha)
        history_pitch.append(pitch)
        history_pn.append(pn)
        history_pe.append(pe)
        history_actions.append(action)
        
        # 步进更新
        state = next_state
        total_reward += reward
        
    print(f"飞行测试结束！")
    print(f"总存活步数: {env.current_step}")
    print(f"总航程 (水平直线距离): {math.sqrt((history_pn[-1]-history_pn[0])**2 + (history_pe[-1]-history_pe[0])**2):.2f} 米")
    print(f"累积奖励: {total_reward:.2f}")
    
    # 7. 绘制测试结果图表
    plot_test_results(history_time, history_alt, history_vel, history_pitch, history_alpha, history_pn, history_pe, history_actions, model_codes)


def plot_test_results(t, alt, vel, pitch, alpha, pn, pe, actions, model_codes):
    """绘制飞行的轨迹与状态，特别是动作切换图"""
    plt.style.use('dark_background')
    plt.rcParams['font.sans-serif'] = ['SimHei'] 
    plt.rcParams['axes.unicode_minus'] = False   
    
    fig = plt.figure(figsize=(16, 12))
    fig.suptitle('强化学习控制策略飞行遥测数据', fontsize=20, color='cyan', fontweight='bold')

    # 1. 3D 轨迹图 (左上)
    ax1 = fig.add_subplot(2, 2, 1, projection='3d')
    ax1.plot(pe, pn, alt, color='cyan', linewidth=2)
    ax1.scatter(pe[0], pn[0], alt[0], color='lime', s=100, label='起点')
    ax1.scatter(pe[-1], pn[-1], alt[-1], color='red', s=100, label='终点')
    ax1.set_title('3D 空间航迹 (NED)')
    ax1.set_xlabel('East (m)')
    ax1.set_ylabel('North (m)')
    ax1.set_zlabel('Altitude (m)')
    ax1.legend()

    # 2. 高度与速度剖面 (右上)
    ax2 = fig.add_subplot(2, 2, 2)
    ax2.plot(t, alt, color='springgreen', label='高度 (m)', linewidth=2)
    ax2_twin = ax2.twinx()
    ax2_twin.plot(t, vel, color='gold', label='速度 (m/s)', linewidth=2, linestyle='--')
    ax2.set_title('高度与速度趋势')
    ax2.set_xlabel('时间 (s)')
    ax2.set_ylabel('高度 (m)', color='springgreen')
    ax2_twin.set_ylabel('速度 (m/s)', color='gold')
    ax2.grid(True, alpha=0.3)

    # 3. 姿态角变化 (左下)
    ax3 = fig.add_subplot(2, 2, 3)
    ax3.plot(t, pitch, color='hotpink', label='俯仰角 (Pitch)', linewidth=2)
    ax3.plot(t, alpha, color='dodgerblue', label='迎角 (Alpha)', linewidth=2)
    ax3.set_title('纵向气动姿态')
    ax3.set_xlabel('时间 (s)')
    ax3.set_ylabel('角度 (°)')
    ax3.legend()
    ax3.grid(True, alpha=0.3)

    # 4. 【核心】模型代号动作切换图 (右下)
    ax4 = fig.add_subplot(2, 2, 4)
    # 用散点图直观展示在哪个时间点用了哪个外形
    ax4.scatter(t, actions, color='orange', s=10, alpha=0.5)
    ax4.set_title('模型代号(动作)切换策略')
    ax4.set_xlabel('时间 (s)')
    ax4.set_ylabel('模型代号')
    # 将 Y 轴的数字刻度替换为你的模型代号字符串
    ax4.set_yticks(range(len(model_codes)))
    ax4.set_yticklabels(model_codes)
    ax4.grid(True, axis='y', alpha=0.5, linestyle='--')

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()

# ================= 运行测试脚本 =================
if __name__ == '__main__':
    # 初始化你的参数
    aircraft_params = {
        'S': 3.857, 'b': 4.2, 'c_bar': 1.380462, 'mass': 14000,
        'Ixx': 313220.6, 'Iyy': 273435.3, 'Izz': 392903.9, 'Ixz': 0.0
    }
    
    # 1. 实例化并严格加载你的真实数据库
    flight_db = HybridAeroDatabase() 
    flight_db._load_from_pickle(pickle_path='X47B.pkl') # 必须先 load 数据
    
    engine_db = EngineDatabase()     
    engine_db.load1('engine.pkl')       # 必须先 load 数据
    
    # 2. 【核心修复】：动态获取所有的模型代号，保证与训练时绝对一致！
    model_codes = list(flight_db.models_db.keys())
    print(f"当前加载了 {len(model_codes)} 个飞行模型代号用于测试。") 
    # 这里打印出来的数字必须是 46，才能完美匹配你的 .pth 模型
    
    # 3. 实例化测试环境
    test_env = FlightEnvRL(flight_db, engine_db, aircraft_params, model_codes, max_steps=5000, dt=0.02)
    
    # 替换为你实际保存的模型路径
    saved_model_path = 'PPO_Flight_Policy.pth' 
    
    # 运行测试
    test_trained_model(saved_model_path, test_env, model_codes)