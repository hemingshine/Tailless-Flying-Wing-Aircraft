#coding=utf-8
import numpy as np
import math
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.gridspec import GridSpec
from stable_baselines3 import PPO

# 导入最新的MIMO环境（仅替换导入，其他完全不变）
from train_pponew import MIMOX47BEnv

# =========================================================
# ⚙️ 在这里自由设置你要测试的极端飞行条件！
# =========================================================
TEST_INIT_ALT = 2800.0    # 初始高度 (m)
TEST_INIT_VEL = 250.0     # 初始速度 (m/s)
TEST_TARGET_ALT = 3300.0  # 目标高度 (m)
TEST_TARGET_VEL = 260.0   # 目标速度 (m/s) 【新增：MIMO双通道目标】
# =========================================================

def run_custom_test():
    print(f"===========================================")
    print(f"  正在测试MIMO任务：从 {TEST_INIT_ALT}m/{TEST_INIT_VEL}m/s")
    print(f"  飞至 {TEST_TARGET_ALT}m/{TEST_TARGET_VEL}m/s")
    print(f"===========================================")

    env = MIMOX47BEnv()
    try:
        # 适配最新的模型保存路径
        model = PPO.load("rl_models/mimo_dy/best_model.zip")
    except Exception as e:
        print(f"模型加载失败：{e}，请先运行 train_pponew.py")
        return

    # 魔改环境的初始状态（完全保留你原来的逻辑）
    obs, _ = env.reset()
    env.target_alt = TEST_TARGET_ALT
    env.target_vel = TEST_TARGET_VEL  # 【新增：设置目标速度】
    env.sim.set_initial_state(
        h_m=TEST_INIT_ALT, 
        V_mps=TEST_INIT_VEL, 
        theta_deg=0.0, 
        alpha_deg=0.0
    )
    
    # ✅ 重置MIMO环境的所有积分器（必须加，否则观测值偏移）
    env.integral_h = 0.0
    env.integral_v = 0.0
    env.integral_alpha = 0.0
    env.last_action = np.array([0.0, 0.0])
    
    # 强制重新获取一次观测值
    obs = env._get_obs()

    # 历史数据存储（适配MIMO双通道）
    history_time = []
    history_alt = []
    history_vel = []       # 【新增：实际速度】
    history_alpha = []
    history_pitch = []
    history_target_alpha = []  # 【替换：目标迎角】
    history_target_throttle = [] # 【新增：目标油门】

    for step in range(env.max_steps):
        action, _states = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action)
        
        u, v, w = env.sim.state[3], env.sim.state[4], env.sim.state[5]
        h = -env.sim.state[2]
        V = math.sqrt(u**2 + v**2 + w**2)
        alpha = math.degrees(math.atan2(w, u))
        pitch = math.degrees(env.sim.state[7])
        
        # ✅ 适配MIMO动作解算：2维动作→目标迎角+目标油门
        target_alpha = ((action[0] + 1.0) / 2.0) * 8.0 - 2.0
        target_throttle = ((action[1] + 1.0) / 2.0) * 0.9 + 0.1
        
        history_time.append(step * env.action_repeat * env.dt)
        history_alt.append(h)
        history_vel.append(V)
        history_alpha.append(alpha)
        history_pitch.append(pitch)
        history_target_alpha.append(target_alpha)
        history_target_throttle.append(target_throttle)
        
        if terminated or truncated:
            break

    print("✅ 试飞完成！开始渲染动画...")
    render_hud(
        history_time, history_alt, history_vel,
        history_alpha, history_pitch,
        history_target_alpha, history_target_throttle
    )

def render_hud(
    history_time, history_alt, history_vel,
    history_alpha, history_pitch,
    history_target_alpha, history_target_throttle
):
    # 完全保留你原来的绘图样式和布局
    plt.style.use('dark_background')
    plt.rcParams['font.sans-serif'] = ['SimHei']
    plt.rcParams['axes.unicode_minus'] = False
    
    fig = plt.figure(figsize=(12, 8))
    fig.suptitle(
        f'X-47B MIMO双通道AI机长试飞\n目标: {TEST_TARGET_ALT}m / {TEST_TARGET_VEL}m/s', 
        fontsize=18, fontweight='bold', color='cyan'
    )
    gs = GridSpec(2, 2, figure=fig)

    # 姿态仪（完全不变）
    ax_plane = fig.add_subplot(gs[:, 0])
    ax_plane.set_xlim(-15, 15); ax_plane.set_ylim(-15, 15)
    ax_plane.set_aspect('equal')
    ax_plane.grid(True, linestyle='--', alpha=0.3)
    ax_plane.set_title("姿态仪 (Attitude)")
    plane_line, = ax_plane.plot([], [], color='springgreen', linewidth=6, solid_capstyle='round', label='机身')
    velocity_line, = ax_plane.plot([], [], color='dodgerblue', linewidth=2, linestyle='--', label='速度矢量')
    ax_plane.legend(loc='lower right')

    # 右上：高度+速度双轴追踪（布局位置不变，新增速度曲线）
    ax_alt = fig.add_subplot(gs[0, 1])
    ax_vel = ax_alt.twinx()  # 双Y轴，不改变原有布局
    
    ax_alt.set_xlim(0, max(history_time))
    min_h, max_h = min(TEST_INIT_ALT, TEST_TARGET_ALT), max(TEST_INIT_ALT, TEST_TARGET_ALT)
    ax_alt.set_ylim(min_h - 200, max_h + 200)
    ax_vel.set_ylim(200, 300)  # 速度范围匹配训练时的220-280
    
    # 高度曲线（原逻辑不变）
    ax_alt.axhline(TEST_TARGET_ALT, color='white', linestyle='--', alpha=0.5, label='目标高度')
    alt_line, = ax_alt.plot([], [], color='springgreen', linewidth=2, label='实际高度')
    ax_alt.set_ylabel("高度 (m)", color='springgreen')
    ax_alt.tick_params(axis='y', labelcolor='springgreen')
    
    # 新增速度曲线（同位置双轴）
    ax_vel.axhline(TEST_TARGET_VEL, color='orange', linestyle='--', alpha=0.5, label='目标速度')
    vel_line, = ax_vel.plot([], [], color='orange', linewidth=2, label='实际速度')
    ax_vel.set_ylabel("速度 (m/s)", color='orange')
    ax_vel.tick_params(axis='y', labelcolor='orange')
    
    ax_alt.set_title("高度+速度追踪")
    ax_alt.legend(loc='upper left')
    ax_vel.legend(loc='upper right')

    # HUD显示（更新为MIMO控制参数）
    ax_hud = fig.add_subplot(gs[1, 1]); ax_hud.axis('off')
    hud_text = ax_hud.text(0.1, 0.5, '', fontsize=16, family='monospace', color='gold', verticalalignment='center')

    def update(frame):
        t, pitch, alpha = history_time[frame], history_pitch[frame], history_alpha[frame]
        alt, vel = history_alt[frame], history_vel[frame]
        cmd_alpha, cmd_throttle = history_target_alpha[frame], history_target_throttle[frame]
        
        # 姿态仪更新（完全不变）
        pitch_rad = math.radians(pitch)
        x_body = [-5.0 * math.cos(pitch_rad), 5.0 * math.cos(pitch_rad)]
        y_body = [-5.0 * math.sin(pitch_rad), 5.0 * math.sin(pitch_rad)]
        plane_line.set_data(x_body, y_body)
        
        fpa_rad = math.radians(pitch - alpha)
        velocity_line.set_data([0, 7.5 * math.cos(fpa_rad)], [0, 7.5 * math.sin(fpa_rad)])
        
        # 曲线更新
        alt_line.set_data(history_time[:frame], history_alt[:frame])
        vel_line.set_data(history_time[:frame], history_vel[:frame])
        
        # ✅ HUD文本更新为MIMO双通道参数
        hud_text.set_text(
            f"时间: {t:>5.1f} s\n\n"
            f"高度: {alt:>6.1f} m\n"
            f"速度: {vel:>6.1f} m/s\n\n"
            f"AI指令(迎角): {cmd_alpha:>5.1f} °\n"
            f"AI指令(油门): {cmd_throttle:>5.2f}\n"
            f"实际俯仰   : {pitch:>5.1f} °\n"
            f"当前迎角   : {alpha:>5.1f} °\n"
        )
        
        return plane_line, velocity_line, alt_line, vel_line, hud_text

    print("\n🎬 正在播放动画...")
    ani = animation.FuncAnimation(
        fig, update, 
        frames=len(history_time), 
        interval=50, 
        blit=False
    )
    plt.tight_layout()
    plt.show()
    
    # 完全保留你原来的GIF保存逻辑
    ani.save(f"mimo_ppo_flight_{TEST_INIT_ALT}_{TEST_TARGET_ALT}_{TEST_INIT_VEL}_{TEST_TARGET_VEL}.gif")

if __name__ == "__main__":
    run_custom_test()