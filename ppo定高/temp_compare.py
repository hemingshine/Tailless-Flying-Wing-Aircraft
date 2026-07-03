#coding=utf-8
"""
eval_compare_4methods.py —— 内环容错控制四方对比
在你原有 RBF-NSMC / IDHP 基础上, 增加:
  · NDI  : 经典非线性动态逆 (模型求逆, 非容错基线; PID/NDI 类)
  · IBSMC: Liu 2025 增量反步滑模 (SOTA 模型基容错; 增量式, 天然抗舵效丧失)
任务/故障/外环 PPO 与原脚本完全一致 (复用 run_nsmc / run_idhp), 唯一变量是内环控制律。
"""
import os
import math
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from stable_baselines3 import PPO
import warnings

from fly import NeuralAeroDatabase, EngineDatabase, FlightSimulator6DOF
# 复用原脚本里已实现并验证过的两套方法 + 物理探针
from eval_changeheight_compare import run_nsmc, run_idhp, get_current_derivatives

warnings.filterwarnings('ignore')

# ===== 故障与任务常量 (与原脚本一致) =====
FAULT_T0, FAULT_T1, FAULT_EFF = 40.0, 80.0, 0.45


# =================================================================
# 共用外环+内环骨架 (与 run_nsmc 完全一致, 仅 inner_law 不同)
# =================================================================
def _run_with_inner(ppo_model, flight_db, engine_db, aircraft_params, inner_law, hist_keys):
    """inner_law(ctx)->(u_total, extra_dict): 给定内环上下文算升降舵指令; extra_dict 记录额外通道。"""
    sim = FlightSimulator6DOF(flight_db, engine_db, aircraft_params)
    sim.set_initial_state(h_m=3000.0, V_mps=240.0, theta_deg=2.0, alpha_deg=2.0)

    dt = 0.02
    action_repeat = 10
    total_steps = int(200 / (dt * action_repeat))

    hist = {k: [] for k in (['t', 'alt', 'target_alt', 'pitch', 'target_pitch'] + hist_keys)}

    integral_h, integral_v = 0.0, 0.0
    sim_time = 0.0
    smoothed_action = np.array([0.0, 0.0], dtype=np.float32)
    pitch_c = 2.0; pitch_c_dot = 0.0; omega_n = 2.0; zeta = 0.9
    last_V = 240.0
    sine_freq_h, sine_amp_h, sine_bias_h = 5 / 200.0, 30.0, 3000.0

    # 内环控制器持久状态 (供 inner_law 跨步使用)
    ctx_state = {'integral_e_theta': 0.0, 'last_delta_e': 0.0,
                 'q_prev': 0.0, 'qdd_f': 0.0, 'u_rob': 0.0, 'inited': False}

    for step in range(total_steps):
        u, v, w = sim.state[3], sim.state[4], sim.state[5]
        phi, theta = sim.state[6], sim.state[7]
        current_h = -sim.state[2]
        V = math.sqrt(u**2 + v**2 + w**2)
        if np.isnan(V) or V < 50.0:
            break

        current_vz = u*math.sin(theta) - v*math.sin(phi)*math.cos(theta) - w*math.cos(phi)*math.cos(theta)
        current_ax = (V - last_V) / (dt * action_repeat)
        last_V = V
        alpha = math.degrees(math.atan2(w, u))
        gamma = math.degrees(theta) - alpha

        target_alt_real = sine_bias_h + sine_amp_h * math.sin(2 * math.pi * sine_freq_h * sim_time)
        t_lead = 1.8
        target_alt_future = sine_bias_h + sine_amp_h * math.sin(2 * math.pi * sine_freq_h * (sim_time + t_lead))
        err_h = target_alt_future - current_h
        err_v = 230.0 - V

        integral_h = np.clip(integral_h + (target_alt_real - current_h) * (dt * action_repeat), -1000.0, 1000.0)
        integral_v = np.clip(integral_v + err_v * (dt * action_repeat), -100.0, 100.0)

        obs = np.array([err_h/500., current_vz/10., integral_h/1000., err_v/50., current_ax/5.,
                        integral_v/100., gamma/10., alpha/10., math.degrees(sim.state[10])/10.], dtype=np.float32)
        obs = np.nan_to_num(obs, nan=0.0, posinf=1.0, neginf=-1.0)
        action, _ = ppo_model.predict(obs, deterministic=True)
        smoothed_action = 0.5 * smoothed_action + 0.5 * action

        target_pitch_ppo = ((smoothed_action[0] + 1.0) / 2.0) * 10.0 - 2.0
        target_throttle = ((smoothed_action[1] + 1.0) / 2.0) * 0.9 + 0.1

        for _ in range(action_repeat):
            phi_inner, theta_inner = sim.state[6], sim.state[7]
            current_pitch = math.degrees(theta_inner)
            current_q = math.degrees(sim.state[10])
            current_p = math.degrees(sim.state[9])
            current_d_ail = np.clip(1.0 * (0.0 - math.degrees(phi_inner)) - 0.5 * current_p, -10.0, 10.0)

            # 二阶指令参考滤波 (得到 theta_c / theta_c_dot / theta_c_ddot)
            pitch_c_ddot = omega_n**2 * (target_pitch_ppo - pitch_c) - 2 * zeta * omega_n * pitch_c_dot
            pitch_c += pitch_c_dot * dt
            pitch_c_dot += pitch_c_ddot * dt

            # 名义模型探针 (零偏 / 单位偏) -> f2_nom, ce0_nom
            controls_0 = {'d_flap_L': 0.0, 'd_flap_R': 0.0, 'd_ail_L': current_d_ail, 'd_ail_R': -current_d_ail, 'throttle': target_throttle}
            controls_1 = {'d_flap_L': 1.0, 'd_flap_R': 1.0, 'd_ail_L': current_d_ail, 'd_ail_R': -current_d_ail, 'throttle': target_throttle}
            f2_nom = get_current_derivatives(sim, controls_0)
            ce0_nom = get_current_derivatives(sim, controls_1) - f2_nom
            if abs(ce0_nom) < 1e-2:
                ce0_nom = -1e-2 if ce0_nom <= 0 else 1e-2

            ctx = dict(current_pitch=current_pitch, current_q=current_q,
                       pitch_c=pitch_c, pitch_c_dot=pitch_c_dot, pitch_c_ddot=pitch_c_ddot,
                       f2_nom=f2_nom, ce0_nom=ce0_nom, dt=dt, st=ctx_state)
            u_total, extra = inner_law(ctx)

            # 故障注入: 40-80s 升降舵效剩 45%
            d_flap_physical = u_total * FAULT_EFF if FAULT_T0 <= sim_time <= FAULT_T1 else u_total
            sim.step(dt, {'d_flap_L': d_flap_physical, 'd_flap_R': d_flap_physical,
                          'd_ail_L': current_d_ail, 'd_ail_R': -current_d_ail, 'throttle': target_throttle})
            sim_time += dt

            hist['t'].append(sim_time)
            hist['alt'].append(-sim.state[2])
            hist['target_alt'].append(target_alt_real)
            hist['pitch'].append(current_pitch)
            hist['target_pitch'].append(pitch_c)
            for kk in hist_keys:
                hist[kk].append(extra.get(kk, 0.0))

    return hist


# ---------------- NDI: 经典动态逆基线 (非容错) ----------------
def run_ndi(ppo_model, flight_db, engine_db, aircraft_params):
    Kp_th, Kp_q, Ki_th = 2.5, 6.0, 0.5

    def law(c):
        st = c['st']
        e_theta = c['current_pitch'] - c['pitch_c']
        st['integral_e_theta'] = float(np.clip(st['integral_e_theta'] + e_theta * c['dt'], -10.0, 10.0))
        q_c = -Kp_th * e_theta + c['pitch_c_dot']
        q_dot_des = -Kp_q * (c['current_q'] - q_c) + c['pitch_c_ddot'] - Ki_th * st['integral_e_theta']
        delta_e = float(np.clip((q_dot_des - c['f2_nom']) / c['ce0_nom'], -20.0, 20.0))  # 名义求逆
        return delta_e, {'u_total': delta_e}

    return _run_with_inner(ppo_model, flight_db, engine_db, aircraft_params, law, ['u_total'])


# ---------------- IBSMC: Liu 2025 增量反步滑模 (SOTA) ----------------
def run_ibsmc(ppo_model, flight_db, engine_db, aircraft_params):
    c1, K, eta, phi = 4.0, 6.0, 4.0, 0.5
    a_f = 0.30           # 角加速度低通系数 (模拟陀螺微分滤波)
    inc_clip = 3.0       # 单步增量限幅(deg)

    def law(c):
        st = c['st']
        q = c['current_q']
        if not st['inited']:
            st['q_prev'] = q; st['inited'] = True
        # 实测角加速度 (有限差分 + 低通) —— 增量法不依赖模型 f2, 故对舵效丧失天然鲁棒
        qdd_raw = (q - st['q_prev']) / c['dt']
        st['qdd_f'] = (1 - a_f) * st['qdd_f'] + a_f * qdd_raw
        st['q_prev'] = q

        e_theta = c['current_pitch'] - c['pitch_c']
        q_c = -c1 * e_theta + c['pitch_c_dot']
        q_c_dot = -c1 * (q - c['pitch_c_dot']) + c['pitch_c_ddot']
        s = q - q_c
        nu = q_c_dot - K * s - eta * math.tanh(s / phi)         # 期望角加速度(反步+滑模)
        delta_inc = float(np.clip((nu - st['qdd_f']) / c['ce0_nom'], -inc_clip, inc_clip))
        delta_e = float(np.clip(st['last_delta_e'] + delta_inc, -20.0, 20.0))  # ★ 增量式
        st['last_delta_e'] = delta_e
        # 记录鲁棒/增量补偿(可视化用): 滑模项折算的舵偏增量累积
        robust_inc = (-K * s - eta * math.tanh(s / phi)) / c['ce0_nom']
        st['u_rob'] = float(np.clip(st['u_rob'] + robust_inc, -25.0, 25.0))
        return delta_e, {'u_total': delta_e, 'u_rob': st['u_rob']}

    return _run_with_inner(ppo_model, flight_db, engine_db, aircraft_params, law, ['u_total', 'u_rob'])


# =================================================================
# 指标
# =================================================================
def compute_metrics(hist):
    t = np.array(hist['t'])
    fz = np.where((t >= FAULT_T0) & (t <= FAULT_T1))[0]
    ae = np.array(hist['alt'])[fz] - np.array(hist['target_alt'])[fz]
    pe = np.array(hist['pitch'])[fz] - np.array(hist['target_pitch'])[fz]
    tz = np.where((t >= FAULT_T0) & (t <= FAULT_T0 + 5.0))[0]
    mta = float(np.max(np.abs(np.array(hist['alt'])[tz] - np.array(hist['target_alt'])[tz])))
    mtp = float(np.max(np.abs(np.array(hist['pitch'])[tz] - np.array(hist['target_pitch'])[tz])))
    return (float(np.sqrt(np.mean(ae**2))), float(np.sqrt(np.mean(pe**2))), mta, mtp)


# =================================================================
# 主流程
# =================================================================
def main():
    print("=" * 60)
    print("  内环容错四方对比: NDI / RBF-NSMC / IDHP / IBSMC(Liu2025)")
    print("=" * 60)

    aircraft_params = {'S': 88.58, 'b': 18.9, 'c_bar': 4.6, 'mass': 14000,
                       'Ixx': 313220.6, 'Iyy': 273435.3, 'Izz': 392903.9, 'Ixz': 0.0}
    flight_db = NeuralAeroDatabase(); flight_db._load_from_pickle('aero_surrogate.pth')
    engine_db = EngineDatabase(); engine_db.load1('engine.pkl')
    try:
        ppo_model = PPO.load("rl_models/mimo_ultimate/best_model.zip")
        print("已加载外环 PPO 轨迹跟踪模型")
    except Exception as e:
        print(f"PPO 加载失败: {e}"); return

    print("运行 NDI (经典动态逆基线) ...")
    hist_ndi = run_ndi(ppo_model, flight_db, engine_db, aircraft_params)
    print("运行 RBF-NSMC ...")
    hist_nsmc = run_nsmc(ppo_model, flight_db, engine_db, aircraft_params)
    print("运行 IDHP ...")
    hist_idhp = run_idhp(ppo_model, flight_db, engine_db, aircraft_params)
    print("运行 IBSMC (Liu 2025 增量反步滑模) ...")
    hist_ibsmc = run_ibsmc(ppo_model, flight_db, engine_db, aircraft_params)

    METHODS = [
        ('NDI (baseline)',  hist_ndi,   '#9467bd', [('u_total', 'Total Control')]),
        ('RBF-NSMC',        hist_nsmc,  '#1f77b4', [('u_total', 'Total Control'), ('f_nn', 'NN Comp')]),
        ('IDHP',            hist_idhp,  '#2ca02c', [('u_0', 'Base PID'), ('u_d', 'IDHP Comp')]),
        ('IBSMC (Liu 2025)', hist_ibsmc, '#d62728', [('u_total', 'Total Control'), ('u_rob', 'Incr. SMC Comp')]),
    ]

    # ---------- 量化表 ----------
    print("\n" + "=" * 78)
    print(" 故障期间(40-80s, 舵效剩 45%) 量化对比")
    print("=" * 78)
    print(f" {'Method':<18}{'Alt RMSE[m]':>13}{'Pitch RMSE[deg]':>17}{'Max|Δh|[m]':>13}{'Max|Δθ|[deg]':>15}")
    print("-" * 78)
    table_rows = []
    for name, hist, _, _ in METHODS:
        m = compute_metrics(hist)
        table_rows.append((name, *m))
        print(f" {name:<18}{m[0]:>13.2f}{m[1]:>17.2f}{m[2]:>13.2f}{m[3]:>15.2f}")
    print("=" * 78 + "\n")

    # =================================================================
    # 👑 绘图大字号学术版 (完全适配双格式导出)
    # =================================================================
    plt.style.use('bmh')
    
    # 设置为纯英文衬线字体
    plt.rcParams['font.family'] = 'serif'
    plt.rcParams['font.serif'] = ['Times New Roman', 'DejaVu Serif']
    plt.rcParams['axes.unicode_minus'] = False
    
    # 全局字号极度放大
    plt.rcParams['axes.labelsize'] = 25    # X/Y轴标签
    plt.rcParams['xtick.labelsize'] = 25   # X轴刻度
    plt.rcParams['ytick.labelsize'] = 25   # Y轴刻度
    plt.rcParams['legend.fontsize'] = 18   # 图例

    # ---------- 明细图 (4 行 x 3 列: Altitude / Pitch / Control) ----------
    fig = plt.figure(figsize=(20, 18))  # 增大画板容纳大字体
    gs = GridSpec(4, 3, figure=fig)
    for r, (name, hist, color, ctrl) in enumerate(METHODS):
        ax_a = fig.add_subplot(gs[r, 0])
        ax_a.plot(hist['t'], hist['target_alt'], 'k--', lw=2.0, label='Target Alt')
        ax_a.plot(hist['t'], hist['alt'], color=color, lw=2.5, alpha=0.9, label=f'Alt ({name})')
        ax_a.axvspan(FAULT_T0, FAULT_T1, color='red', alpha=0.10)
        ax_a.set_ylabel('Altitude [m]', fontweight='bold')
        ax_a.legend(loc='lower right', framealpha=0.85)
        if r == 3: ax_a.set_xlabel('Time [s]', fontweight='bold')

        ax_p = fig.add_subplot(gs[r, 1])
        ax_p.plot(hist['t'], hist['target_pitch'], 'k--', lw=2.0, label='Cmd Pitch')
        ax_p.plot(hist['t'], hist['pitch'], color=color, lw=2.5, alpha=0.9, label=f'Pitch ({name})')
        ax_p.axvspan(FAULT_T0, FAULT_T1, color='red', alpha=0.10)
        ax_p.set_ylabel('Pitch [deg]', fontweight='bold')
        ax_p.legend(loc='lower right', framealpha=0.85)
        if r == 3: ax_p.set_xlabel('Time [s]', fontweight='bold')

        ax_c = fig.add_subplot(gs[r, 2])
        cstyle = [color, '#ff7f0e']
        for i, (key, lab) in enumerate(ctrl):
            ax_c.plot(hist['t'], hist[key], color=cstyle[i % 2], lw=2.5, alpha=0.85, label=lab)
        ax_c.axvspan(FAULT_T0, FAULT_T1, color='red', alpha=0.10)
        ax_c.set_ylabel('Control [deg]', fontweight='bold')
        ax_c.legend(loc='upper right', framealpha=0.85)
        if r == 3: ax_c.set_xlabel('Time [s]', fontweight='bold')
        
    plt.tight_layout()
    fig.savefig('./compare4_detail.png', dpi=300, bbox_inches='tight')
    fig.savefig('./compare4_detail.pdf', format='pdf', bbox_inches='tight')

    # ---------- 叠加对比图 (1 x 3: 高度 / |高度误差| / 俯仰误差, 四方同图) ----------
    fig2 = plt.figure(figsize=(24, 7)) # 加宽画板防止横向图例遮挡
    gs2 = GridSpec(1, 3, figure=fig2)
    
    ax1 = fig2.add_subplot(gs2[0, 0])
    ax1.plot(hist_ndi['t'], hist_ndi['target_alt'], 'k--', lw=2.0, label='Target')
    for name, hist, color, _ in METHODS:
        ax1.plot(hist['t'], hist['alt'], color=color, lw=2.5, alpha=0.9, label=name)
    ax1.axvspan(FAULT_T0, FAULT_T1, color='red', alpha=0.10)
    ax1.set_xlabel('Time [s]', fontweight='bold'); ax1.set_ylabel('Altitude [m]', fontweight='bold')
    ax1.legend(loc='lower right', framealpha=0.85)

    ax2 = fig2.add_subplot(gs2[0, 1])
    for name, hist, color, _ in METHODS:
        e = np.abs(np.array(hist['alt']) - np.array(hist['target_alt']))
        ax2.plot(hist['t'], e, color=color, lw=2.5, alpha=0.9, label=name)
    ax2.axvspan(FAULT_T0, FAULT_T1, color='red', alpha=0.10)
    ax2.set_xlabel('Time [s]', fontweight='bold'); ax2.set_ylabel('|Altitude Error| [m]', fontweight='bold')
    ax2.legend(loc='upper left', framealpha=0.85)

    ax3 = fig2.add_subplot(gs2[0, 2])
    for name, hist, color, _ in METHODS:
        e = np.abs(np.array(hist['pitch']) - np.array(hist['target_pitch']))
        ax3.plot(hist['t'], e, color=color, lw=2.5, alpha=0.9, label=name)
    ax3.axvspan(FAULT_T0, FAULT_T1, color='red', alpha=0.10)
    ax3.set_xlabel('Time [s]', fontweight='bold'); ax3.set_ylabel('|Pitch Error| [deg]', fontweight='bold')
    ax3.legend(loc='upper left', framealpha=0.85)
    
    plt.tight_layout()
    fig2.savefig('./compare4_overlay.png', dpi=300, bbox_inches='tight')
    fig2.savefig('./compare4_overlay.pdf', format='pdf', bbox_inches='tight')

    print("\n✅ 图表已保存:\n - 明细图: ./compare4_detail.png / .pdf\n - 叠加图: ./compare4_overlay.png / .pdf")
    plt.show()


if __name__ == "__main__":
    main()