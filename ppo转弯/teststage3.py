import os
import math
import numpy as np
import matplotlib.pyplot as plt
from stable_baselines3 import PPO

# 导入你的环境构建函数
from train_inner import make_env

def evaluate_stage3_model():
    print("=====================================================")
    print("  加载 Stage 3 (纵向通道) 预训练模型进行全轴闭环测试")
    print("=====================================================")
    
    # 定义模型路径 (请确保这些文件或目录存在)
    # 注意：Stage 3 我们默认去读 logs 里的 best_model
    # paths_s3 = {'dir': 'ppo_dir_stage1', 'lat': 'ppo_lat_stage2'}
    paths_s3 = {'dir': './logs/best_model_stage1/best_model', 'lat': './logs/best_model_stage2/best_model'}
    model3_path = "./logs/best_model_stage3/best_model.zip"
    
    for k, path in paths_s3.items():
        if not os.path.exists(path + ".zip"):
            print(f"❌ 找不到后台护航模型: {path}.zip")
            return
            
    if not os.path.exists(model3_path):
        print(f"❌ 找不到 Stage 3 主模型: {model3_path} (如果保存在了根目录，请修改路径)")
        return

    # 实例化 Stage 3 环境，注入双脑
    env_creator = make_env(stage=3, model_paths=paths_s3, seed=432)
    env = env_creator()
    
    print(f"✅ 成功加载环境与后台双脑 (Stage 1 & Stage 2)")
    model3 = PPO.load(model3_path[:-4], device='cpu') # 去掉 .zip 后缀加载
    print(f"✅ 成功加载 Stage 3 测试模型")
    
    obs, _ = env.reset()
    
    # 提取当前 episode 的随机目标指令
    target_beta = env.target_beta
    target_phi = env.target_phi
    target_theta = env.target_theta
    print(f"🎯 本次测试目标 -> 俯仰(Pitch): {target_theta:.2f}°, 滚转(Roll): {target_phi:.2f}°, 侧滑(Beta): {target_beta:.2f}°\n")
    
    # 数据记录器
    history = {
        't': [], 'theta': [], 'phi': [], 'beta': [], 'alpha': [],
        'act_e': [], 'd_flap': [], 'd_ail': [], 'd_spoil': [], 'reward': []
    }
    
    total_reward = 0.0
    step_count = 0
    
    # 开始全轴闭环仿真 (20秒)
    for step in range(2000):
        # 👑 deterministic=True，关闭探索噪声，展现绝对实力
        action, _ = model3.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, _ = env.step(action)
        
        # 从底层读取真实物理状态
        state = env.sim.state
        u, v, w = state[3], state[4], state[5]
        V = max(math.sqrt(u**2 + v**2 + w**2), 1.0)
        
        alpha = math.degrees(math.atan2(w, u))
        beta = math.degrees(math.asin(np.clip(v / V, -1.0, 1.0)))
        phi = math.degrees(state[6])
        theta = math.degrees(state[7])
        
        # 记录数据
        t = step * env.dt
        history['t'].append(t)
        history['theta'].append(theta)
        history['phi'].append(phi)
        history['beta'].append(beta)
        history['alpha'].append(alpha)
        
        history['act_e'].append(action[0])              # Stage 3 网络输出
        history['d_flap'].append(env.prev_actions['e']) # 实际升降舵偏角
        history['d_ail'].append(env.prev_actions['a'])  # 实际副翼偏角
        history['d_spoil'].append(env.prev_actions['r'])# 实际阻力舵偏角
        history['reward'].append(reward)
        
        total_reward += reward
        step_count += 1
        
        if terminated or truncated:
            print(f"⚠️ 仿真在第 {step} 步结束 (时长 {t:.2f}秒). Terminated: {terminated}, Truncated: {truncated}")
            break

    print(f"🏁 测试完成！存活步数: {step_count}/2000, 累计总奖励: {total_reward:.2f}")
    
    # ==========================================
    # 👑 绘制全轴飞行核心数据图 (针对 Stage 3 定制)
    # ==========================================
    plt.figure(figsize=(16, 10))
    plt.style.use('bmh')
    
    # 1. 俯仰角跟踪 (Stage 3 核心任务)
    plt.subplot(3, 2, 1)
    plt.plot(history['t'], history['theta'], 'c-', linewidth=2, label='Actual Pitch (Theta)')
    plt.axhline(y=target_theta, color='r', linestyle='--', label=f'Target ({target_theta:.1f}°)')
    plt.title('Pitch Angle Tracking (Stage 3 Primary)')
    plt.ylabel('Degrees')
    plt.legend()
    
    # 2. 迎角安全监控 (生死线)
    plt.subplot(3, 2, 2)
    plt.plot(history['t'], history['alpha'], 'orange', linewidth=2, label='Angle of Attack (Alpha)')
    plt.axhline(y=12.0, color='r', linestyle=':', label='Stall Upper Limit (+12°)')
    plt.axhline(y=-3.0, color='r', linestyle=':', label='Stall Lower Limit (-3°)')
    plt.title('Alpha Safety Envelope Monitor')
    plt.ylabel('Degrees')
    plt.legend()
    
    # 3. 滚转角 (Stage 2 护航效果)
    plt.subplot(3, 2, 3)
    plt.plot(history['t'], history['phi'], 'm-', linewidth=2, label='Actual Roll (Phi)')
    plt.axhline(y=target_phi, color='r', linestyle='--', label=f'Target ({target_phi:.1f}°)')
    plt.title('Roll Angle (Stage 2 Disturbance Rejection)')
    plt.ylabel('Degrees')
    plt.legend()
    
    # 4. 侧滑角 (Stage 1 护航效果)
    plt.subplot(3, 2, 4)
    plt.plot(history['t'], history['beta'], 'b-', linewidth=2, label='Actual Beta')
    plt.axhline(y=target_beta, color='r', linestyle='--', label='Target Beta (0°)')
    plt.title('Sideslip Angle (Stage 1 Suppression Check)')
    plt.ylabel('Degrees')
    plt.legend()
    
    # 5. Stage 3 动作输出 (网络平滑度检查)
    plt.subplot(3, 2, 5)
    plt.plot(history['t'], history['act_e'], 'g-', linewidth=1.5, label='Network Output (Elevon)')
    plt.axhline(y=0, color='gray', linestyle='--')
    plt.title('Agent Action (Elevon Command)')
    plt.xlabel('Time (s)')
    plt.ylabel('Normalized [-1, 1]')
    plt.legend()
    
    # 6. 三轴舵面物理偏转 (查看是否饱和冲突)
    plt.subplot(3, 2, 6)
    plt.plot(history['t'], history['d_flap'], 'c-', label='Flap (Stage 3)')
    plt.plot(history['t'], history['d_ail'], 'm-', alpha=0.7, label='Aileron (Stage 2)')
    plt.plot(history['t'], history['d_spoil'], 'b-', alpha=0.7, label='Drag Rudder (Stage 1)')
    plt.title('Physical Actuator Deflections (Saturation Check)')
    plt.xlabel('Time (s)')
    plt.ylabel('Degrees')
    plt.legend()
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    evaluate_stage3_model()