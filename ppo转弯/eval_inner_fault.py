#coding=utf-8
"""
FTC 增益验证 A/B 测试 (严谨版)
关键设计:
  · 自由选择故障通道 (pitch/roll/yaw) 与 注入时刻
  · 自动匹配正弦激励指令，稳态平衡力矩≈0，精准逼出动态跟踪下的舵效损失
  · 输出定量指标：故障前/后 RMS 跟踪误差 + FTC 改善百分比
"""
import os
import math
import warnings
import argparse
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from stable_baselines3 import PPO

from fly import NeuralAeroDatabase, EngineDatabase, FlightSimulator6DOF
from train_inner_fault import X47BInnerEnv

warnings.filterwarnings('ignore')

# ================= 👑 核心可调配置区 =================
# 你可以直接在这里修改，或者通过命令行传递参数，例如：
# python eval_inner_fault.py --axis roll --time 15.0
DEFAULT_FAULT_AXIS = 'pitch'  # 可选: 'pitch' (升降副翼), 'roll' (副翼), 'yaw' (阻力舵)
DEFAULT_FAULT_TIME = 20.0     # 故障注入时刻 (秒)
# =====================================================

AIRCRAFT = {'S': 88.58, 'b': 18.9, 'c_bar': 4.6, 'mass': 14000,
            'Ixx': 313220.6, 'Iyy': 273435.3, 'Izz': 392903.9, 'Ixz': 0.0}

PATHS = {
    'lon': './logs/best_model_stage3fault/best_model.zip',
    'dir': './logs/best_model_stage1fault/best_model.zip',
    'lat': './logs/best_model_stage2fault/best_model.zip',
}

SIM_TIME     = 50.0                    # 总时长，跑过故障点很久
FAULT_SCALES = [0.50, 0.35, 0.25]      # 舵效残留(损失 50% / 65% / 75%)
SETTLE       = 5.0                     # 故障后跳过的暂态(s)，再开始统计稳态指标

# 正弦激励指令: bias ± amp, 频率 freq(Hz)
SINE_BIAS, SINE_AMP, SINE_FREQ = 2.0, 4.0, 0.1
SEED = 2025


class DummyModel:
    """横滚/偏航模型缺失时垫后，避免报错"""
    def predict(self, obs, deterministic=True):
        return np.zeros(1, dtype=np.float32), None


def load_models():
    if not os.path.exists(PATHS['lon']):
        raise FileNotFoundError(f"缺少纵向主控模型: {PATHS['lon']}")
    m = {'lon': PPO.load(PATHS['lon'][:-4], device='cpu')}
    m['dir'] = PPO.load(PATHS['dir'][:-4], device='cpu') if os.path.exists(PATHS['dir']) else DummyModel()
    m['lat'] = PPO.load(PATHS['lat'][:-4], device='cpu') if os.path.exists(PATHS['lat']) else DummyModel()
    return m


def get_targets(t, fault_axis):
    """根据故障通道，动态将正弦波注入对应的姿态角指令"""
    sine_wave = SINE_AMP * math.sin(2 * math.pi * SINE_FREQ * t)
    if fault_axis == 'pitch':
        return SINE_BIAS + sine_wave, 0.0, 0.0
    elif fault_axis == 'roll':
        return SINE_BIAS, sine_wave, 0.0
    elif fault_axis == 'yaw':
        return SINE_BIAS, 0.0, sine_wave
    return SINE_BIAS, 0.0, 0.0


def run_case(aero_db, engine_db, models, use_ftc, fault_scale, fault_axis, fault_time):
    """跑一次仿真。A(use_ftc=False) 与 B(use_ftc=True) 除 FTC 外完全一致。"""
    sim = FlightSimulator6DOF(aero_db, engine_db, AIRCRAFT)
    env = X47BInnerEnv(sim, stage=3)
    env.trained_models = models
    try:
        env.curr = 1.0                 # 课程拉满，对应部署期增益
    except Exception:
        pass
    env.max_steps = int(SIM_TIME / env.dt) + 50

    obs, _ = env.reset(seed=SEED)

    # ---- 强制健康基线 + 统一初始状态(A/B 起点绝对一致) ----
    env.eff = {'pitch': 1.0, 'roll': 1.0, 'yaw': 1.0}   
    env._fault_t = fault_time
    env._fault_axis = fault_axis
    env._fault_scale = fault_scale
    env.ftc_enabled = use_ftc
    if hasattr(env, 'ftc'):
        env.ftc.reset()

    env.sim.set_initial_state(h_m=3000.0, V_mps=200.0, theta_deg=2.0)
    env.sim.state[6] = 0.0     # phi
    env.sim.state[8] = 0.0     # yaw
    env.sim.state[9:12] = 0.0  # p,q,r
    for _ in range(5):
        env._update_history()
    obs = env._get_obs(3)

    hist = {
        't': [], 'theta': [], 'phi': [], 'beta': [],
        'target_theta': [], 'target_phi': [], 'target_beta': [],
        'de': [], 'da': [], 'dr': [],
        'alpha': [], 'ftc_I_pitch': [], 'ftc_I_roll': [], 'ftc_I_yaw': []
    }
    
    crashed_at = None
    n_steps = int(SIM_TIME / env.dt)
    for step in range(n_steps):
        t = step * env.dt
        
        # 智能匹配指令
        tgt_theta, tgt_phi, tgt_beta = get_targets(t, fault_axis)
        env.target_theta = tgt_theta
        env.target_phi = tgt_phi
        env.target_beta = tgt_beta

        action, _ = models['lon'].predict(obs, deterministic=True)
        obs, _, term, trunc, _ = env.step(action)

        # 记录全轴状态与指令
        hist['t'].append(t)
        hist['theta'].append(math.degrees(env.sim.state[7]))
        hist['phi'].append(math.degrees(env.sim.state[6]))
        V = max(math.sqrt(env.sim.state[3]**2 + env.sim.state[4]**2 + env.sim.state[5]**2), 1.0)
        hist['beta'].append(math.degrees(math.asin(np.clip(env.sim.state[4]/V, -1.0, 1.0))))
        
        hist['target_theta'].append(tgt_theta)
        hist['target_phi'].append(tgt_phi)
        hist['target_beta'].append(tgt_beta)
        
        hist['de'].append(env.prev_actions['e'])
        hist['da'].append(env.prev_actions['a'])
        hist['dr'].append(env.prev_actions['r'])
        hist['alpha'].append(math.degrees(math.atan2(env.sim.state[5], env.sim.state[3])))
        
        # FTC 状态记录
        if use_ftc and hasattr(env, 'ftc'):
            hist['ftc_I_pitch'].append(env.ftc.pitch.I if hasattr(env.ftc, 'pitch') and env.ftc.pitch else 0.0)
            hist['ftc_I_roll'].append(env.ftc.roll.I if hasattr(env.ftc, 'roll') and env.ftc.roll else 0.0)
            hist['ftc_I_yaw'].append(env.ftc.yaw.I if hasattr(env.ftc, 'yaw') and env.ftc.yaw else 0.0)
        else:
            hist['ftc_I_pitch'].append(0.0)
            hist['ftc_I_roll'].append(0.0)
            hist['ftc_I_yaw'].append(0.0)

        if term:                 
            crashed_at = t
            break
        if trunc:                
            break
    return hist, crashed_at


def rms_error(hist, t0, t1, state_key, target_key):
    e = [abs(th - tg) for tt, th, tg in zip(hist['t'], hist[state_key], hist[target_key]) if t0 <= tt < t1]
    return float(np.sqrt(np.mean(np.square(e)))) if e else float('nan')


def main():
    parser = argparse.ArgumentParser(description="FTC 增益验证 A/B 测试")
    parser.add_argument('--axis', type=str, default=DEFAULT_FAULT_AXIS, choices=['pitch', 'roll', 'yaw'], help="选择故障通道")
    parser.add_argument('--time', type=float, default=DEFAULT_FAULT_TIME, help="设置故障注入起始时间 (秒)")
    args = parser.parse_args()

    FAULT_AXIS = args.axis
    FAULT_TIME = args.time

    cmd_name = {'pitch': '俯仰', 'roll': '滚转', 'yaw': '偏航'}[FAULT_AXIS]
    print("=" * 64)
    print(f" FTC 增益验证 A/B 测试  (动态正弦{cmd_name}指令激发)")
    print(f" 故障通道={FAULT_AXIS}  注入时刻={FAULT_TIME}s  总时长={SIM_TIME}s")
    print(f" 动态波形: {SINE_BIAS}° ± {SINE_AMP}° @ {SINE_FREQ}Hz")
    print("=" * 64)

    aero_db = NeuralAeroDatabase(); aero_db._load_from_pickle('X47B_coeffs.pkl')
    engine_db = EngineDatabase(); engine_db.load1('engine.pkl')
    models = load_models()

    summary = []
    plot_data = []  

    # ==========================================
    # 动态映射作图和计算所需的键值对
    # ==========================================
    if FAULT_AXIS == 'pitch':
        state_k, target_k, act_k, ftc_k = 'theta', 'target_theta', 'de', 'ftc_I_pitch'
        state_lbl, act_lbl = 'Pitch (θ) [deg]', 'Elevon (δe) [deg]'
    elif FAULT_AXIS == 'roll':
        state_k, target_k, act_k, ftc_k = 'phi', 'target_phi', 'da', 'ftc_I_roll'
        state_lbl, act_lbl = 'Roll (φ) [deg]', 'Aileron (δa) [deg]'
    else:
        state_k, target_k, act_k, ftc_k = 'beta', 'target_beta', 'dr', 'ftc_I_yaw'
        state_lbl, act_lbl = 'Sideslip (β) [deg]', 'Spoiler (δr) [deg]'

    for scale in FAULT_SCALES:
        loss_pct = int(round((1 - scale) * 100))
        print(f"\n──── 故障档: 舵效残留 {scale:.2f} (损失 {loss_pct}%) ────")
        base, base_crash = run_case(aero_db, engine_db, models, use_ftc=False, fault_scale=scale, fault_axis=FAULT_AXIS, fault_time=FAULT_TIME)
        ftc,  ftc_crash  = run_case(aero_db, engine_db, models, use_ftc=True,  fault_scale=scale, fault_axis=FAULT_AXIS, fault_time=FAULT_TIME)

        pre0, pre1 = FAULT_TIME - 10.0, FAULT_TIME
        post0, post1 = FAULT_TIME + SETTLE, SIM_TIME
        
        a_pre = rms_error(base, pre0, pre1, state_k, target_k)
        a_post = rms_error(base, post0, post1, state_k, target_k)
        b_pre = rms_error(ftc, pre0, pre1, state_k, target_k)
        b_post = rms_error(ftc, post0, post1, state_k, target_k)
        improve = (a_post - b_post) / a_post * 100 if (a_post and not math.isnan(a_post)) else float('nan')

        print(f"  RMS跟踪误差(°)  故障前 | 故障后")
        print(f"    仅RL      : {a_pre:6.3f} | {a_post:6.3f}" + (f"  ⚠{base_crash:.1f}s坠毁" if base_crash else ""))
        print(f"    RL+FTC    : {b_pre:6.3f} | {b_post:6.3f}" + (f"  ⚠{ftc_crash:.1f}s坠毁" if ftc_crash else ""))
        print(f"    FTC改善   : 故障后误差降低 {improve:5.1f}%")
        
        summary.append((scale, loss_pct, a_post, b_post, improve, base_crash, ftc_crash))
        plot_data.append({'loss_pct': loss_pct, 'base': base, 'ftc': ftc})

    # ---- 总表 + 判读 ----
    print("\n" + "=" * 64)
    print(f" 汇总 ({FAULT_AXIS.capitalize()} 通道故障后 RMS 跟踪误差，越小越好)")
    print(" 舵效残留 | 损失 | 仅RL  | RL+FTC | FTC改善 | 备注")
    for scale, loss, a, b, imp, bc, fc in summary:
        note = []
        if bc: note.append("RL坠")
        if fc: note.append("FTC坠")
        print(f"  {scale:.2f}   | {loss:3d}% | {a:5.2f} | {b:5.2f}  | {imp:5.1f}% | {' '.join(note)}")
    print("=" * 64)

    # ====================================================
    # 👑 绘制论文使用的 3x3 纯英文对比图 (学术大字体矢量版)
    # ====================================================
    plt.style.use('bmh')
    
    # 纯英文学术字体
    plt.rcParams['font.family'] = 'serif'
    plt.rcParams['font.serif'] = ['Times New Roman', 'DejaVu Serif']
    plt.rcParams['axes.unicode_minus'] = False
    
    # 全局超大字体配置
    plt.rcParams['axes.labelsize'] = 25    # X/Y 轴标签
    plt.rcParams['xtick.labelsize'] = 25   # X 轴刻度
    plt.rcParams['ytick.labelsize'] = 25   # Y 轴刻度
    plt.rcParams['legend.fontsize'] = 25   # 图例字体
    
    c_rl = '#1f77b4'   # Blue
    c_ftc = '#2ca02c'  # Green
    c_cmd = 'black'    # Black line
    c_ftc_i = '#ff7f0e'# Orange
    
    # 加大画板尺寸以容纳 3x3 的粗线和大字体
    fig, axes = plt.subplots(3, 3, figsize=(22, 12))
    
    for row_idx, p_data in enumerate(plot_data):
        loss_pct = p_data['loss_pct']
        base = p_data['base']
        ftc = p_data['ftc']
        
        # --- 列 1: 姿态角跟踪 ---
        axes[row_idx, 0].plot(base['t'], base[target_k], color=c_cmd, ls='--', lw=2.5, label='Command')
        axes[row_idx, 0].plot(base['t'], base[state_k], color=c_rl, lw=2.5, alpha=0.9, label='RL Only')
        axes[row_idx, 0].plot(ftc['t'], ftc[state_k], color=c_ftc, lw=2.5, alpha=0.9, label='RL + FTC')
        axes[row_idx, 0].axvspan(FAULT_TIME, SIM_TIME, color='red', alpha=0.1, label='Fault Zone')
        axes[row_idx, 0].set_ylabel(f'{loss_pct}% Loss\n{state_lbl}', fontweight='bold')
        if row_idx == 0: 
            axes[row_idx, 0].legend(loc='best', framealpha=0.85)
        
        # --- 列 2: 实际舵偏 ---
        axes[row_idx, 1].plot(base['t'], base[act_k], color=c_rl, lw=2.5, alpha=0.8, label='RL Only')
        axes[row_idx, 1].plot(ftc['t'], ftc[act_k], color=c_ftc, lw=2.5, alpha=0.8, label='RL + FTC')
        axes[row_idx, 1].axvspan(FAULT_TIME, SIM_TIME, color='red', alpha=0.1)
        axes[row_idx, 1].set_ylabel(act_lbl, fontweight='bold')
        if row_idx == 0: 
            axes[row_idx, 1].legend(loc='best', framealpha=0.85)
        
        # --- 列 3: FTC 积分状态 ---
        axes[row_idx, 2].plot(ftc['t'], ftc[ftc_k], color=c_ftc_i, lw=2.5, label='FTC Integral')
        axes[row_idx, 2].axvspan(FAULT_TIME, SIM_TIME, color='red', alpha=0.1)
        axes[row_idx, 2].set_ylabel('FTC Integral', fontweight='bold')
        if row_idx == 0: 
            axes[row_idx, 2].legend(loc='best', framealpha=0.85)
        
        # 仅在最后一行显示 X 轴标签
        if row_idx == 2:
            axes[row_idx, 0].set_xlabel('Time [s]', fontweight='bold')
            axes[row_idx, 1].set_xlabel('Time [s]', fontweight='bold')
            axes[row_idx, 2].set_xlabel('Time [s]', fontweight='bold')

    plt.tight_layout()
    
    # 自动保存为双格式
    os.makedirs('./logs/', exist_ok=True)
    save_path = f'./logs/eval_inner_fault_{FAULT_AXIS}_3x3'
    plt.savefig(f'{save_path}.pdf', format='pdf', bbox_inches='tight')
    plt.savefig(f'{save_path}.png', format='png', bbox_inches='tight', dpi=300)
    print(f"\n✅ 图表已保存:\n - 矢量图: {save_path}.pdf\n - 位图: {save_path}.png")
    
    plt.show()

if __name__ == "__main__":
    main()