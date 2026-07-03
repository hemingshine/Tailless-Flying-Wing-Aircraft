#coding=utf-8
"""
scan_lookahead.py —— 超前预见 (look-ahead) 扫描 + 相位滞后量化
扫描 t_p ∈ {0, 0.5, 1, 2, 3, 4} s:
  (a) 航向 RMSE vs t_p   (b) 高度 RMSE vs t_p   (c) 相位滞后 vs t_p
并在【故障窗内】对比 开/关预见 的相位滞后 (cross-correlation 时延 + 50% 穿越时延)。

相位滞后定义:
  · 机动段 (航向爬升) 用 "50% 穿越时延": 输出到达 45° 的时刻 − 参考到达 45° 的时刻。
  · 通用/故障窗 用 互相关时延: 使 corr(y(t), ref(t−τ)) 最大的 τ (>0 表示输出滞后参考)。
预期: t_p 增大相位滞后单调下降, 但过大 (over-preview) 会引入超调/失稳 -> RMSE 呈 U 形,
存在最优 t_p。故障期开预见可显著削减滞后。
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
TARGET_ALT, V0, TARGET_V = 3000.0, 200.0, 200.0
YAW_FINAL = 90.0
RAMP_T0, RAMP_T1 = 5.0, 65.0
T_LEADS = [0,2,4,6,8,10,12]
N_SEEDS = 3
SEED = 20260619
TURB_SIGMA, TURB_L = 3.0, 530.0
# 故障窗 (与机动重叠, 使"故障期相位滞后"可测); 升降舵 45% LoE
FT0, FT1, FAULT_EFF = 30.0, 60.0, 1
AP = {'S': 88.58, 'b': 18.9, 'c_bar': 4.6, 'mass': 14000,
      'Ixx': 313220.6, 'Iyy': 273435.3, 'Izz': 392903.9, 'Ixz': 0.0}
INNER_FULL = {'dir': './logs_paper/best_model_stage1/best_model.zip',
              'lat': './logs_paper/best_model_stage2/best_model.zip',
              'lon': './logs_paper/best_model_stage3/best_model.zip'}
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


def run_tp(t_lead, wind, seed):
    inner = {k: FastPredictor(PPO.load(p[:-4], device='cpu')) for k, p in INNER_FULL.items()}
    outer = PPO.load(OUTER[:-4], device='cpu')
    aero = NeuralAeroDatabase(); aero._load_from_pickle('X47B_coeffs.pkl')
    eng = EngineDatabase(); eng.load1('engine.pkl')
    sim = FlightSimulator6DOF(aero, eng, dict(AP))
    ie = X47BInnerEnv(sim, stage=3); ie.trained_models = inner
    ie.max_steps = int(SIM_TIME / ie.dt) + 50
    oe = X47BOuterEnv(ie, inner); oe.max_steps = int(SIM_TIME / oe.outer_dt) + 5
    oe.reset(seed=seed)
    ie.domain_rand = False; ie.eff = {'pitch': 1.0, 'roll': 1.0, 'yaw': 1.0}; ie._fault_t = 1e9
    ie.ftc_enabled = True
    sim.set_initial_state(TARGET_ALT, V0, theta_deg=2.0); sim.state[6] = 0.0; sim.state[8] = 0.0
    for _ in range(5): ie._update_history()

    # 故障: 故障窗内升降舵 45% LoE (注于物理舵偏)
    clk = {'t': 0.0}; orig = sim.step
    def step_f(dt, controls):
        if FT0 <= clk['t'] <= FT1:
            controls['d_flap_L'] = float(np.clip(controls['d_flap_L'] * FAULT_EFF, -25, 25))
            controls['d_flap_R'] = float(np.clip(controls['d_flap_R'] * FAULT_EFF, -25, 25))
        clk['t'] += dt; return orig(dt, controls)
    sim.step = step_f

    dt_o = oe.outer_dt; n = int(SIM_TIME / dt_o)
    H = {'t': [], 'yaw': [], 'ref_yaw': [], 'alt': [], 'ref_alt': []}
    for k in range(n):
        t = k * dt_o
        sim.wind = wind[k]
        oe.target_yaw = ref_yaw(t + t_lead)        # ④ 预见: 喂 t+t_p 的参考
        oe.target_alt = TARGET_ALT
        a, _ = outer.predict(oe._get_obs(), deterministic=True)
        oe.step(a)
        s = sim.state
        H['t'].append(t); H['yaw'].append(math.degrees(s[8])); H['ref_yaw'].append(ref_yaw(t))
        H['alt'].append(-s[2]); H['ref_alt'].append(TARGET_ALT)
    return {k: np.asarray(v) for k, v in H.items()}


def rmse(y, r, t, win):
    m = (t >= win[0]) & (t <= win[1]); return float(np.sqrt(np.mean((y[m] - r[m])**2)))


def phase_lag_ramp(t, y, ref, win, n_levels=15):
    """相位滞后 = 故障窗内"水平时移"中位数: 对参考跨越的若干电平, 取
    输出到达该电平时刻 − 参考到达该电平时刻, 多电平取中位数。
    对单调变化的参考 (机动爬升段) 稳健; >0 表示输出滞后参考。"""
    m = (t >= win[0]) & (t <= win[1])
    tw, yw, rw = t[m], y[m], ref[m]
    lo, hi = float(rw.min()), float(rw.max())
    if hi - lo < 3.0:                          # 参考在窗内几乎不变, 相位无意义
        return float('nan')
    levels = np.linspace(lo + 0.1 * (hi - lo), lo + 0.9 * (hi - lo), n_levels)
    lags = []
    for lv in levels:
        ir = np.where(rw >= lv)[0]; iy = np.where(yw >= lv)[0]
        if len(ir) and len(iy):
            lags.append(tw[iy[0]] - tw[ir[0]])
    return float(np.median(lags)) if lags else float('nan')


def crossing_delay(t, y, ref, level=45.0):
    """50% 穿越时延: 输出首次到 level 的时刻 − 参考首次到 level 的时刻 (机动段)。"""
    def first_cross(sig):
        idx = np.where(sig >= level)[0]
        return t[idx[0]] if len(idx) else float('nan')
    return first_cross(y) - first_cross(ref)


def main():
    dt_o = 0.1; n = int(SIM_TIME / dt_o)
    seeds = [SEED + i * 101 for i in range(N_SEEDS)]
    hold_win = (max(RAMP_T1 + 10, FT1 + 5), SIM_TIME)   # 稳态保持段算 RMSE

    agg = {tp: {'yaw_rmse': [], 'alt_rmse': [], 'lag_fault': [], 'cross': []} for tp in T_LEADS}
    curves = {}   # tp -> 一条代表轨迹 (第一个 seed) 供画图
    for tp in T_LEADS:
        for sd in seeds:
            wind = make_turb(n + 10, dt_o, TURB_SIGMA, TURB_L, V0, sd)
            print(f"运行 t_p={tp:.1f}s  (seed={sd}) ...")
            H = run_tp(tp, wind, sd)
            agg[tp]['yaw_rmse'].append(rmse(H['yaw'], H['ref_yaw'], H['t'], hold_win))
            agg[tp]['alt_rmse'].append(rmse(H['alt'], H['ref_alt'], H['t'], hold_win))
            agg[tp]['lag_fault'].append(phase_lag_ramp(H['t'], H['yaw'], H['ref_yaw'], (FT0, FT1)))
            agg[tp]['cross'].append(crossing_delay(H['t'], H['yaw'], H['ref_yaw']))
            if sd == seeds[0]: curves[tp] = H

    def ms(tp, key): a = np.array(agg[tp][key], float); return float(np.nanmean(a)), float(np.nanstd(a))

    print("\n" + "=" * 84)
    print(f" look-ahead 扫描 (n_seed={N_SEEDS}); 故障窗 [{FT0:.0f},{FT1:.0f}]s 升降舵 {int(FAULT_EFF*100)}% LoE")
    print("=" * 84)
    print(f" {'t_p[s]':>7}{'Yaw RMSE':>12}{'Alt RMSE':>12}{'PhaseLag_fault[s]':>20}{'CrossDelay[s]':>16}")
    for tp in T_LEADS:
        yr, _ = ms(tp, 'yaw_rmse'); ar, _ = ms(tp, 'alt_rmse')
        lf, _ = ms(tp, 'lag_fault'); cd, _ = ms(tp, 'cross')
        print(f" {tp:>7.1f}{yr:>12.3f}{ar:>12.3f}{lf:>20.3f}{cd:>16.3f}")

    # ---- 图 ----
    plt.rcParams.update({'font.family': 'serif', 'font.size': 10, 'axes.grid': True, 'grid.alpha': 0.3})
    fig = plt.figure(figsize=(11.5, 7.2)); gs = GridSpec(2, 3, figure=fig, hspace=0.32, wspace=0.30)
    tps = np.array(T_LEADS)

    def errbar(ax, key, ylab, ttl, color):
        mean = np.array([ms(tp, key)[0] for tp in T_LEADS])
        std = np.array([ms(tp, key)[1] for tp in T_LEADS])
        ax.errorbar(tps, mean, yerr=std, marker='o', color=color, capsize=3, lw=1.6)
        imin = int(np.nanargmin(mean))
        ax.scatter([tps[imin]], [mean[imin]], s=90, facecolors='none', edgecolors='k', zorder=5,
                   label=f'min @ {tps[imin]:.1f}s')
        ax.set_xlabel('Look-ahead $t_p$ (s)'); ax.set_ylabel(ylab); ax.set_title(ttl)
        ax.legend(fontsize=8)

    errbar(fig.add_subplot(gs[0, 0]), 'yaw_rmse', 'Heading RMSE (deg)', '(a) Heading RMSE vs $t_p$', '#1f77b4')
    errbar(fig.add_subplot(gs[0, 1]), 'alt_rmse', 'Altitude RMSE (m)', '(b) Altitude RMSE vs $t_p$', '#2ca02c')
    axc = fig.add_subplot(gs[0, 2])
    lagm = np.array([ms(tp, 'lag_fault')[0] for tp in T_LEADS])
    lags = np.array([ms(tp, 'lag_fault')[1] for tp in T_LEADS])
    axc.errorbar(tps, lagm, yerr=lags, marker='s', color='#d62728', capsize=3, lw=1.6)
    axc.axhline(lagm[0], ls=':', color='0.5', lw=1)
    axc.annotate(f'off (t_p=0): {lagm[0]:.2f}s', (tps[0], lagm[0]), textcoords='offset points',
                 xytext=(8, 8), fontsize=8)
    axc.set_xlabel('Look-ahead $t_p$ (s)'); axc.set_ylabel('Phase lag in fault window (s)')
    axc.set_title('(c) Fault-window phase lag vs $t_p$')

    # (d) 故障窗内航向轨迹: 预见 off vs 适中 vs 大, 直观看相位
    axd = fig.add_subplot(gs[1, :])
    show_tps = [0.0, 2.0, 4.0]
    axd.plot(curves[0.0]['t'], curves[0.0]['ref_yaw'], 'k--', lw=1.2, label='Reference')
    cols = {0.0: '#9467bd', 2.0: '#1f77b4', 4.0: '#ff7f0e'}
    for tp in show_tps:
        if tp in curves:
            axd.plot(curves[tp]['t'], curves[tp]['yaw'], color=cols[tp], lw=1.6, label=f'$t_p$={tp:.0f}s')
    axd.axvspan(FT0, FT1, color='red', alpha=0.08)
    axd.set_xlim(RAMP_T0, RAMP_T1 + 10)
    axd.set_xlabel('Time (s)'); axd.set_ylabel('Heading (deg)')
    axd.set_title('(d) Heading trace (fault window shaded): preview reduces phase lag')
    axd.legend(loc='lower right', fontsize=8.5)

    fig.savefig('./scan_lookahead.png', dpi=300, bbox_inches='tight')
    fig.savefig('./scan_lookahead.pdf', bbox_inches='tight')
    print("\n图已存: scan_lookahead.png / .pdf")


if __name__ == '__main__':
    main()