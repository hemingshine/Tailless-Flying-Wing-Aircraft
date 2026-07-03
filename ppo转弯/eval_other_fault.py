#coding=utf-8
"""
eval_fault_nonloe.py —— 内环容错增强: 非 LoE 故障 (卡死 stuck / 偏置 bias)
动机: 纯乘性 LoE 仅缩放舵效、保号, 对自适应/滑模律"太友好"; 卡死与偏置是
      加性/非乘性故障, 更接近真实作动器失效, 对容错律更具挑战。
做法: 在升降舵物理舵偏上注入故障 (sim.step monkeypatch, 与 step_tecs 同款),
      对比 FTC 开/关 在 协同转弯+定高 下的姿态/航向保持。

故障类型 (均注于升降舵 d_flap, 故障窗 [FT0,FT1]):
  · loe   : delta_e *= scale            (乘性, 基线对照)
  · bias  : delta_e += bias_deg         (恒偏置 -> 积分类容错可消除)
  · stuck : delta_e := stuck_deg        (卡死在固定角, 指令增广无法移动物理面)

物理提醒: 升降舵是该飞翼唯一俯仰舵面, 完全卡死时俯仰不可控 —— FTC 只能"限幅/不发散",
不可能完全恢复; bias 则可被 FTC 积分项基本消除。这一对照正好量化容错边界 (呼应 future work)。
"""
import os, math, warnings
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from stable_baselines3 import PPO

from fly_robust import NeuralAeroDatabase, EngineDatabase, FlightSimulator6DOF
from train_inner_fault import X47BInnerEnv
from train_outerfault import X47BOuterEnv, FastPredictor

warnings.filterwarnings('ignore')

SIM_TIME = 120.0
TARGET_ALT, V0 = 3000.0, 200.0
YAW_FINAL = 90.0
RAMP_T0, RAMP_T1 = 5.0, 65.0
FT0, FT1 = 70.0, 110.0        # 故障窗 (落在转弯后保持段, 看容错维持)
SEED = 20260619
TURB_SIGMA, TURB_L = 2.0, 530.0     # 中等湍流, 突出故障影响
AP = {'S': 88.58, 'b': 18.9, 'c_bar': 4.6, 'mass': 14000,
      'Ixx': 313220.6, 'Iyy': 273435.3, 'Izz': 392903.9, 'Ixz': 0.0}
INNER_FULL = {'dir': './logs/best_model_stage1fault/best_model.zip',
              'lat': './logs/best_model_stage2fault/best_model.zip',
              'lon': './logs/best_model_stage3fault/best_model.zip'}
OUTER = './logs/best_model_outer/best_model.zip'


def ref_yaw(t):
    if t <= RAMP_T0: return 0.0
    if t >= RAMP_T1: return YAW_FINAL
    x = (t - RAMP_T0) / (RAMP_T1 - RAMP_T0); return YAW_FINAL * x * x * (3 - 2 * x)


def make_turb(n, dt, sigma, L, V, seed):
    rng = np.random.default_rng(seed); wg = np.zeros(3); seq = np.zeros((n, 3))
    b = dt * V / L; c = sigma * math.sqrt(max(2 * b, 1e-6))
    for k in range(n): wg = (1 - b) * wg + c * rng.standard_normal(3); seq[k] = wg
    return seq


def run(fault_type, ftc_on, fault_arg, wind):
    """fault_type in {'none','loe','bias','stuck'}; fault_arg: loe=scale, bias=deg, stuck=deg"""
    inner = {k: FastPredictor(PPO.load(p[:-4], device='cpu')) for k, p in INNER_FULL.items()}
    outer = PPO.load(OUTER[:-4], device='cpu')
    aero = NeuralAeroDatabase(); aero._load_from_pickle('X47B_coeffs.pkl')
    eng = EngineDatabase(); eng.load1('engine.pkl')
    sim = FlightSimulator6DOF(aero, eng, dict(AP))
    ie = X47BInnerEnv(sim, stage=3); ie.trained_models = inner
    ie.max_steps = int(SIM_TIME / ie.dt) + 50
    oe = X47BOuterEnv(ie, inner); oe.max_steps = int(SIM_TIME / oe.outer_dt) + 5
    oe.reset(seed=SEED)
    ie.domain_rand = False; ie.eff = {'pitch': 1.0, 'roll': 1.0, 'yaw': 1.0}; ie._fault_t = 1e9
    ie.ftc_enabled = bool(ftc_on)
    sim.set_initial_state(TARGET_ALT, V0, theta_deg=2.0); sim.state[6] = 0.0; sim.state[8] = 0.0
    for _ in range(5): ie._update_history()

    # —— 故障注入: 改物理舵偏 d_flap_L/R; 同时记录指令(故障前)与物理(故障后)舵偏 ——
    clk = {'t': 0.0}
    delog = {'t': [], 'cmd': [], 'phys': []}
    orig = sim.step
    def step_f(dt, controls):
        de_cmd = float(controls['d_flap_L'])          # 控制器指令(含 FTC 增广)
        if FT0 <= clk['t'] <= FT1 and fault_type != 'none':
            de = de_cmd
            if fault_type == 'loe':
                de = de * fault_arg
            elif fault_type == 'bias':
                de = de + fault_arg
            elif fault_type == 'stuck':
                de = fault_arg
            controls['d_flap_L'] = float(np.clip(de, -25, 25))
            controls['d_flap_R'] = float(np.clip(de, -25, 25))
        delog['t'].append(clk['t']); delog['cmd'].append(de_cmd)
        delog['phys'].append(float(controls['d_flap_L']))
        clk['t'] += dt
        return orig(dt, controls)
    sim.step = step_f

    dt_o = oe.outer_dt; n = int(SIM_TIME / dt_o)
    H = {'t': [], 'yaw': [], 'ref_yaw': [], 'pitch': [], 'cmd_pitch': [], 'alt': []}
    for k in range(n):
        t = k * dt_o
        sim.wind = wind[k]
        oe.target_yaw = ref_yaw(t); oe.target_alt = TARGET_ALT
        a, _ = outer.predict(oe._get_obs(), deterministic=True)
        oe.step(a)
        s = sim.state
        H['t'].append(t); H['yaw'].append(math.degrees(s[8])); H['ref_yaw'].append(ref_yaw(t))
        H['pitch'].append(math.degrees(s[7])); H['cmd_pitch'].append(oe.cmd_theta)
        H['alt'].append(-s[2])
    out = {k: np.asarray(v) for k, v in H.items()}
    out['de_t'] = np.asarray(delog['t'])
    out['de_cmd'] = np.asarray(delog['cmd'])
    out['de_phys'] = np.asarray(delog['phys'])
    return out


def metrics(H):
    t = H['t']; m = (t >= FT0) & (t <= FT1)
    ey = H['yaw'][m] - H['ref_yaw'][m]
    ep = H['pitch'][m] - H['cmd_pitch'][m]
    return {'yaw_rmse': float(np.sqrt(np.mean(ey**2))),
            'pitch_rmse': float(np.sqrt(np.mean(ep**2))),
            'yaw_max': float(np.max(np.abs(ey))),
            'diverged': bool(np.max(np.abs(ey)) > 60 or not np.all(np.isfinite(H['yaw'])))}


def main():
    dt_o = 0.1; n = int(SIM_TIME / dt_o)
    wind = make_turb(n + 10, dt_o, TURB_SIGMA, TURB_L, V0, SEED)

    # 场景: bias(+8°) 与 stuck(-6°), 各 FTC 开/关; 另加 none 作参考
    SCN = [
        ('Nominal (no fault)', 'none',  0.0, True,  '#000000'),
        ('Bias +8°, FTC off', 'bias',  8.0, False, '#ff7f0e'),
        ('Bias +8°, FTC on',  'bias',  8.0, True,  '#1f77b4'),
        ('Stuck -6°, FTC off','stuck', -6.0, False, '#d62728'),
        ('Stuck -6°, FTC on', 'stuck', -6.0, True,  '#2ca02c'),
    ]
    res = []
    for label, ft, arg, ftc, col in SCN:
        print(f"运行 {label} ...")
        H = run(ft, ftc, arg, wind)
        res.append((label, H, col, metrics(H)))

    print("\n" + "=" * 78)
    print(f" 故障窗 [{FT0:.0f},{FT1:.0f}]s 内 容错对比")
    print("=" * 78)
    print(f" {'Scenario':<22}{'Yaw RMSE':>10}{'Pitch RMSE':>12}{'Yaw max':>10}{'diverged':>10}")
    for label, H, col, m in res:
        print(f" {label:<22}{m['yaw_rmse']:>10.2f}{m['pitch_rmse']:>12.2f}{m['yaw_max']:>10.2f}{str(m['diverged']):>10}")

    # ---- 图: 航向 / 俯仰 / 升降舵 三行 ----
    plt.rcParams.update({'font.family': 'serif', 'font.size': 10, 'axes.grid': True, 'grid.alpha': 0.3})
    fig = plt.figure(figsize=(9.0, 8.2)); gs = GridSpec(3, 1, figure=fig, hspace=0.28)
    ax1 = fig.add_subplot(gs[0])
    ax1.plot(res[0][1]['t'], res[0][1]['ref_yaw'], 'k--', lw=1.1, label='Reference')
    for label, H, col, m in res:
        ax1.plot(H['t'], H['yaw'], color=col, lw=1.5, alpha=0.9, label=label)
    ax1.axvspan(FT0, FT1, color='red', alpha=0.08); ax1.set_ylabel('Heading (deg)')
    ax1.legend(loc='lower right', fontsize=7.5, ncol=2); ax1.set_title('(a) Heading tracking under non-LoE actuator faults')

    ax2 = fig.add_subplot(gs[1])
    for label, H, col, m in res:
        ax2.plot(H['t'], H['pitch'] - H['cmd_pitch'], color=col, lw=1.4, alpha=0.9, label=label)
    ax2.axvspan(FT0, FT1, color='red', alpha=0.08); ax2.set_ylabel('Pitch error (deg)')
    ax2.set_title('(b) Pitch tracking error'); ax2.legend(loc='upper right', fontsize=7.5, ncol=2)

    ax3 = fig.add_subplot(gs[2])
    for label, H, col, m in res:
        ax3.plot(H['de_t'], H['de_cmd'], color=col, lw=1.2, alpha=0.9, label=label)
    # 物理舵偏(故障后)用灰虚线叠一条代表性的(取最后一个 stuck/bias 场景), 显示故障形态
    for label, H, col, m in res:
        if 'Stuck' in label and 'on' in label:
            ax3.plot(H['de_t'], H['de_phys'], color='0.45', lw=1.0, ls='--', alpha=0.8,
                     label='physical (stuck)')
            break
    ax3.axvspan(FT0, FT1, color='red', alpha=0.08); ax3.set_ylabel('Elevator (deg)')
    ax3.set_xlabel('Time (s)')
    ax3.set_title('(c) Commanded elevator (solid; FTC augmentation) vs physical (dashed)')
    ax3.legend(loc='upper right', fontsize=7.5, ncol=2)
    fig.savefig('./fault_nonloe.png', dpi=300, bbox_inches='tight')
    fig.savefig('./fault_nonloe.pdf', bbox_inches='tight')
    print("\n图已存: fault_nonloe.png / .pdf")


if __name__ == '__main__':
    main()