import os
import math
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from stable_baselines3 import PPO

# 导入你刚刚重构完备的外环环境
from train_outer import make_outer_env

def run_turn_evaluation():
    print("=====================================================")
    print("  ✈️  X-47B 终极试飞: 90度大角度转弯性能评估")
    print("=====================================================")
    
    model_path = "./logs/best_model_outer1/best_model.zip"
    if not os.path.exists(model_path):
        print(f"❌ 找不到模型文件: {model_path}，请确认路径是否正确。")
        return

    # 初始化环境
    env_creator = make_outer_env(seed=888)
    env = env_creator()
    
    # 加载模型 (纯推理模式)
    print("🔄 正在加载机长大脑...")
    model = PPO.load(model_path[:-4], device='cpu')
    
    obs, _ = env.reset()
    
    # ==========================================
    # 🎯 设定极为严苛的测试工况
    # ==========================================
    env.target_yaw = 90.0   # 挑战 90 度右转！
    env.target_alt = 3200.0 # 目标高度 3000 米
    
    # 强制物理引擎进入平飞初始态
    env.inner_env.sim.set_initial_state(3000.0, 200.0, theta_deg=2.0)
    env.inner_env.sim.state[6] = 0.0  # 初始坡度 0
    env.inner_env.sim.state[8] = 0.0  # 初始航向 0
    for _ in range(5): 
        env.inner_env._update_history()
    
    # 遥测数据存储桶
    history = {
        't': [], 'yaw': [], 'target_yaw': [],
        'cmd_roll': [], 'actual_roll': [], 
        'cmd_pitch': [], 'actual_pitch': [], 
        'delta_e': [], 'delta_a': [], 'delta_r': [], # ★ 替换为实际控制舵面偏角
        'beta': [], 'alpha': [], 'velocity': [], 'altitude': []
    }
    
    print(f"🛫 试飞开始: 初始航向 0° -> 目标航向 {env.target_yaw}°")
    
    max_test_steps = 6000 # 测试 60 秒 (外环 dt=0.1s)
    
    for step in range(max_test_steps):
        t = step * env.outer_dt
        
        # 👑 绝对确定性模式：剥夺任何随机噪声，只看它的真实实力
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, _ = env.step(action)
        
        state = env.inner_env.sim.state
        u, v, w = state[3], state[4], state[5]
        V = max(math.sqrt(u**2 + v**2 + w**2), 1.0)
        
        # 记录遥测数据
        history['t'].append(t)
        history['yaw'].append(math.degrees(state[8]))
        history['target_yaw'].append(env.target_yaw)
        
        # 记录外环解算出的物理指令 vs 实际物理姿态
        history['cmd_roll'].append(env.cmd_phi)
        history['actual_roll'].append(math.degrees(state[6]))
        history['cmd_pitch'].append(env.cmd_theta)
        history['actual_pitch'].append(math.degrees(state[7]))
        
        # ★ 记录内环真正输出给物理引擎的物理舵面偏角 (度)
        # 通过访问 inner_env 内部缓存的 prev_actions 获取
        history['delta_e'].append(env.inner_env.prev_actions['e'])
        history['delta_a'].append(env.inner_env.prev_actions['a'])
        history['delta_r'].append(env.inner_env.prev_actions['r'])
        
        history['alpha'].append(math.degrees(math.atan2(w, u)))
        history['beta'].append(math.degrees(math.asin(np.clip(v / V, -1.0, 1.0))))
        history['velocity'].append(V)
        history['altitude'].append(-state[2])
        
        if terminated or truncated:
            print(f"⚠️ 试飞于 {t:.1f} 秒提前终止 (物理截断)。")
            break

    print(f"🛬 试飞结束 (存活时间: {t:.1f} 秒)。正在生成遥测报告...")
    
    # ==========================================
    # 📊 绘制专业遥测图表
    # ==========================================
    plt.style.use('bmh')
    fig = plt.figure(figsize=(18, 12))
    gs = GridSpec(3, 3, figure=fig)
    fig.suptitle('X-47B Autonomous Navigation: 90° Turn Step Response', fontsize=16, fontweight='bold')
    
    # --- 1. 终极目标：航向追踪 ---
    ax1 = fig.add_subplot(gs[0, 0:2])
    ax1.plot(history['t'], history['yaw'], 'b-', linewidth=2.5, label='Actual Yaw')
    ax1.plot(history['t'], history['target_yaw'], 'r--', linewidth=2, label='Target Yaw (90°)')
    ax1.set_title('Navigation: Yaw Tracking')
    ax1.set_ylabel('Heading (deg)'); ax1.legend()
    
    # --- 2. 控制指令分析：滚转 (Roll) ---
    ax2 = fig.add_subplot(gs[1, 0])
    ax2.plot(history['t'], history['actual_roll'], 'm-', linewidth=2, label='Actual Roll')
    ax2.plot(history['t'], history['cmd_roll'], 'r--', linewidth=2, label='Cmd Roll')
    ax2.set_title('Lateral Control: Bank Angle')
    ax2.set_ylabel('Degrees'); ax2.legend()
    
    # --- 3. 控制指令分析：俯仰 (Pitch) ---
    ax3 = fig.add_subplot(gs[1, 1])
    ax3.plot(history['t'], history['actual_pitch'], 'c-', linewidth=2, label='Actual Pitch')
    ax3.plot(history['t'], history['cmd_pitch'], 'r--', linewidth=2, label='Cmd Pitch')
    ax3.set_title('Longitudinal Control: Pitch Angle')
    ax3.set_ylabel('Degrees'); ax3.legend()
    
    # --- 4. 能量管理：高度与速度 ---
    ax4 = fig.add_subplot(gs[0, 2])
    ax4.plot(history['t'], history['altitude'], 'b-', linewidth=2, label='Altitude')
    ax4.plot(history['t'], [3000]*len(history['t']), 'r--', alpha=0.5)
    ax4.set_title('Energy: Altitude (Target 3000m)')
    ax4.set_ylabel('Meters'); ax4.legend()
    
    ax5 = fig.add_subplot(gs[1, 2])
    ax5.plot(history['t'], history['velocity'], 'k-', linewidth=2, label='Airspeed')
    ax5.set_title('Energy: Airspeed')
    ax5.set_ylabel('m/s'); ax5.legend()
    
    # --- 5. 安全包线监控：Alpha & Beta ---
    ax6 = fig.add_subplot(gs[2, 0])
    ax6.plot(history['t'], history['beta'], 'g-', linewidth=2, label='Sideslip (Beta)')
    ax6.axhline(2.0, color='orange', linestyle=':', label='Penalty Line')
    ax6.axhline(-2.0, color='orange', linestyle=':')
    ax6.set_title('Safety: Sideslip')
    ax6.set_ylabel('Degrees'); ax6.legend()
    
    ax7 = fig.add_subplot(gs[2, 1])
    ax7.plot(history['t'], history['alpha'], 'orange', linewidth=2, label='Angle of Attack')
    ax7.axhline(8.0, color='r', linestyle=':', label='Penalty Line')
    ax7.set_title('Safety: Angle of Attack')
    ax7.set_ylabel('Degrees'); ax7.legend()
    
    # --- 6. 物理舵效监控：实际舵面偏转 (替代原先的网络 Raw 输出) ---
    ax8 = fig.add_subplot(gs[2, 2])
    ax8.plot(history['t'], history['delta_e'], 'c-', linewidth=1.5, label='Elevon (Pitch)')
    ax8.plot(history['t'], history['delta_a'], 'm-', linewidth=1.5, label='Aileron (Roll)')
    ax8.plot(history['t'], history['delta_r'], 'g-', linewidth=1.5, label='Spoiler (Yaw)')
    ax8.set_title('Actuators: Physical Surface Deflections')
    ax8.set_ylabel('Degrees')
    ax8.set_ylim(-30.0, 30.0) # 设置为飞翼典型的舵面满行程 (-30° 到 30°)
    ax8.set_xlabel('Time (s)'); ax8.legend()

    plt.tight_layout()
    plt.savefig('./logs/eval_turn_report.png', dpi=150)
    plt.show()

if __name__ == "__main__":
    run_turn_evaluation()