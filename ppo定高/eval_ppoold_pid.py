#coding=utf-8
import numpy as np
import math
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.gridspec import GridSpec
from stable_baselines3 import PPO

# 导入通用环境
from train_ppo import GeneralX47BEnv

# =========================================================
# ⚙️ 在这里自由设置你要测试的极端飞行条件！
# =========================================================
TEST_INIT_ALT = 3000.0    # 初始高度 (m)
TEST_INIT_VEL = 250.0     # 初始速度 (m/s)
TEST_TARGET_ALT = 3300.0  # 目标高度 (m) - 比如测试一次“大坡度下降”任务！
# =========================================================

def run_custom_test():
    print(f"===========================================")
    print(f"  正在测试任务：从 {TEST_INIT_ALT}m 飞至 {TEST_TARGET_ALT}m")
    print(f"  初始速度：{TEST_INIT_VEL}m/s")
    print(f"===========================================")

    env = GeneralX47BEnv()
    try:
        model = PPO.load("rl_models/best_model.zip")
    except Exception as e:
        print("模型加载失败，请先运行 train_general_ppo.py")
        return

    # 魔改环境的初始状态
    obs, _ = env.reset()
    env.target_alt = TEST_TARGET_ALT
    env.sim.set_initial_state(h_m=TEST_INIT_ALT, V_mps=TEST_INIT_VEL, theta_deg=0.0, alpha_deg=0.0)
    
    # 强制重新获取一次观测值 (因为改了内部变量)
    obs = env._get_obs()

    history_time, history_alt, history_vel = [], [], []
    history_alpha, history_pitch, history_target_pitch = [], [], []

    for step in range(env.max_steps):
        action, _states = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action)
        
        u, v, w = env.sim.state[3], env.sim.state[4], env.sim.state[5]
        h = -env.sim.state[2]
        V = math.sqrt(u**2 + v**2 + w**2)
        alpha = math.degrees(math.atan2(w, u))
        pitch = math.degrees(env.sim.state[7])
        target_pitch = ((action[0] + 1.0) / 2.0) * 7.0 - 2.0 
        
        history_time.append(step * env.action_repeat * env.dt_physics)
        history_alt.append(h)
        history_vel.append(V)
        history_alpha.append(alpha)
        history_pitch.append(pitch)
        history_target_pitch.append(target_pitch)
        
        if terminated or truncated:
            break

    print("✅ 试飞完成！开始渲染动画...")
    render_hud(history_time, history_alt, history_vel, history_alpha, history_pitch, history_target_pitch)

def render_hud(history_time, history_alt, history_vel, history_alpha, history_pitch, history_target_pitch):
    plt.style.use('dark_background')
    plt.rcParams['font.sans-serif'] = ['SimHei']
    plt.rcParams['axes.unicode_minus'] = False
    
    fig = plt.figure(figsize=(12, 8))
    fig.suptitle(f'X-47B AI机长试飞 (目标高度: {TEST_TARGET_ALT}m)', fontsize=18, fontweight='bold', color='cyan')
    gs = GridSpec(2, 2, figure=fig)

    ax_plane = fig.add_subplot(gs[:, 0])
    ax_plane.set_xlim(-15, 15); ax_plane.set_ylim(-15, 15)
    ax_plane.set_aspect('equal')
    ax_plane.grid(True, linestyle='--', alpha=0.3)
    ax_plane.set_title("姿态仪 (Attitude)")
    plane_line, = ax_plane.plot([], [], color='springgreen', linewidth=6, solid_capstyle='round', label='机身')
    velocity_line, = ax_plane.plot([], [], color='dodgerblue', linewidth=2, linestyle='--', label='速度矢量')
    ax_plane.legend(loc='lower right')

    ax_alt = fig.add_subplot(gs[0, 1])
    ax_alt.set_xlim(0, max(history_time))
    # 动态调整 Y 轴范围，以包容你设置的自定义高度
    min_h, max_h = min(TEST_INIT_ALT, TEST_TARGET_ALT), max(TEST_INIT_ALT, TEST_TARGET_ALT)
    ax_alt.set_ylim(min_h - 200, max_h + 200)
    ax_alt.axhline(TEST_TARGET_ALT, color='white', linestyle='--', alpha=0.5, label='Target')
    alt_line, = ax_alt.plot([], [], color='springgreen', linewidth=2)
    ax_alt.set_title("高度追踪"); ax_alt.legend(loc='lower right')

    ax_hud = fig.add_subplot(gs[1, 1]); ax_hud.axis('off')
    hud_text = ax_hud.text(0.1, 0.5, '', fontsize=16, family='monospace', color='gold', verticalalignment='center')

    def update(frame):
        t, pitch, alpha = history_time[frame], history_pitch[frame], history_alpha[frame]
        alt, vel, cmd_pitch = history_alt[frame], history_vel[frame], history_target_pitch[frame]
        
        pitch_rad = math.radians(pitch)
        x_body = [-5.0 * math.cos(pitch_rad), 5.0 * math.cos(pitch_rad)]
        y_body = [-5.0 * math.sin(pitch_rad), 5.0 * math.sin(pitch_rad)]
        plane_line.set_data(x_body, y_body)
        
        fpa_rad = math.radians(pitch - alpha)
        velocity_line.set_data([0, 7.5 * math.cos(fpa_rad)], [0, 7.5 * math.sin(fpa_rad)])
        
        alt_line.set_data(history_time[:frame], history_alt[:frame])
        
        hud_text.set_text(
            f"时间: {t:>5.1f} s\n\n"
            f"高度: {alt:>6.1f} m\n"
            f"速度: {vel:>6.1f} m/s\n\n"
            f"AI指令(Cmd): {cmd_pitch:>5.1f} °\n"
            f"实际俯仰   : {pitch:>5.1f} °\n"
            f"当前迎角   : {alpha:>5.1f} °\n"
        )
        return plane_line, velocity_line, alt_line, hud_text

    print("\n🎬 正在播放动画...")
    ani = animation.FuncAnimation(fig, update, frames=len(history_time), interval=50, blit=False)
    plt.show()
    ani.save(f"ppo_flight{TEST_INIT_ALT}_{TEST_TARGET_ALT}.gif")

if __name__ == "__main__":
    run_custom_test()