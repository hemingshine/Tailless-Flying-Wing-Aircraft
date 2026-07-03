#coding=utf-8
"""
eval_robustness_mc.py —— 参数化蒙特卡洛鲁棒性评估 (冻结控制器，不重训)

按给定散布表，对【已训练好的内外环控制器】在一群被扰动的飞机上跑
"定高 + 协调转弯"任务，统计成功率与性能包络，全程可视化。

散布表(相对名义):
  基础气动系数 Cx,Cy,Cz,Cl,Cm,Cn(基线/静稳定部分) : 90%–110%  (各自独立抽)
  动导数 Clp,Cmq,Cnr                              : 80%–120%
  舵效(对 Cl/Cm/Cn 的操纵贡献)                    : 80%–120%
  发动机推力 T                                    : 90%–110%
  质量 m                                          : 85%–100%
  质心 CG(弦向/展向)                              : 95%–105% (±5%偏移)
  转动惯量 Ixx,Iyy,Izz                            : 80%–120%
(Clr,Cnp 本模型未建模=0，缩放无效，从略)

可选：叠加一阶 Dryden 体轴湍流(ENABLE_TURBULENCE=True)。
关键：控制器全程冻结，eff=1、FTC关 —— 测的是对"没见过的参数不确定性"的容忍度。
"""
import os
import math
import warnings
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from stable_baselines3 import PPO

from fly_robust import NeuralAeroDatabase, EngineDatabase, FlightSimulator6DOF
from train_inner_fault import X47BInnerEnv
from train_outerfault import X47BOuterEnv, FastPredictor

warnings.filterwarnings('ignore')

# ===================== 配置 =====================
N_SAMPLES   = 300          # 蒙特卡洛样本数(想更密就调大；运行时间近似线性)
SIM_TIME    = 100.0         # 每个样本飞行时长(s)
TARGET_YAW  = 90.0         # 目标航向(°)
TARGET_ALT  = 3200.0       # 目标高度(m)
SEED        = 20260612

BASE_AIRCRAFT = {'S': 88.58, 'b': 18.9, 'c_bar': 4.6, 'mass': 14000,
                 'Ixx': 313220.6, 'Iyy': 273435.3, 'Izz': 392903.9, 'Ixz': 0.0}

# 模型路径(优先容错版，缺失回退非容错版)
INNER = {'dir': './logs/best_model_stage1fault/best_model.zip',
         'lat': './logs/best_model_stage2fault/best_model.zip',
         'lon': './logs/best_model_stage3fault/best_model.zip'}
# OUTER_CANDIDATES = ['./logs/best_model_outerfault/best_model.zip',
#                     './logs/best_model_outer/best_model.zip']
OUTER_CANDIDATES = [
                    './logs/best_model_outer1/best_model.zip']

# 散布范围 (low, high)
R_BASIC = (0.90, 1.10)     # 基础气动系数
R_DYN   = (0.80, 1.20)     # 动导数
R_CTRL  = (0.80, 1.20)     # 舵效
R_THR   = (0.90, 1.10)     # 推力
R_MASS  = (0.85, 1.00)     # 质量
R_CG    = (-0.05, 0.05)    # 质心偏移(占 c_bar / b 的比例，±5%)
R_INERT = (0.80, 1.20)     # 转动惯量

# 可选真·湍流(一阶 Dryden 近似，体轴风)
ENABLE_TURBULENCE = True
TURB_SIGMA = 3.0           # 湍流强度(m/s)，severe 量级
TURB_L     = 530.0         # 湍流尺度长度(m)


# ===================== 扰动包装 =====================
class PerturbedAero:
    """包装解析气动模型：基础系数 / 舵效 / 质心 三类扰动。控制器看到的是被扰动的飞机。"""
    def __init__(self, base, basic_scale, k_ctrl_cl, k_ctrl_cm, k_ctrl_cn, cg_dx, cg_dy):
        self.base = base
        self.bs = basic_scale          # array(6) for Cx,Cy,Cz,Cl,Cm,Cn
        self.kcl, self.kcm, self.kcn = k_ctrl_cl, k_ctrl_cm, k_ctrl_cn
        self.cg_dx, self.cg_dy = cg_dx, cg_dy   # 弦向/展向 CG 偏移(占 c_bar/b)

    def get_body_axis_coeffs(self, mach, d_flap_L, d_flap_R, d_ail_L, d_ail_R,
                             d_spoil_L, d_spoil_R, alpha, beta):
        full = self.base.get_body_axis_coeffs(mach, d_flap_L, d_flap_R, d_ail_L, d_ail_R,
                                              d_spoil_L, d_spoil_R, alpha, beta)
        noc = self.base.get_body_axis_coeffs(mach, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, alpha, beta)
        out = [0.0] * 6
        for j in range(6):
            basic = noc[j] * self.bs[j]          # 基础(基线+静稳定)部分缩放
            ctrl = full[j] - noc[j]              # 纯操纵面贡献
            if j == 3:   ctrl *= self.kcl        # 舵效 -> Cl (副翼)
            elif j == 4: ctrl *= self.kcm        # 舵效 -> Cm (升降)
            elif j == 5: ctrl *= self.kcn        # 舵效 -> Cn (扰流板)
            out[j] = basic + ctrl
        cx, cy, cz, cl, cm, cn = out
        # 质心偏移 = 力臂改变(对力矩的增量)
        cm += cz * self.cg_dx                    # 弦向CG -> 俯仰(静稳定裕度)
        cl += -cz * self.cg_dy                   # 展向CG -> 滚转
        cn += cy * self.cg_dy                    # 展向CG -> 偏航
        return cx, cy, cz, cl, cm, cn


class PerturbedEngine:
    def __init__(self, base, k_thrust):
        self.base = base; self.k = k_thrust
    def get_thrust_newtons(self, alt, mach):
        return self.base.get_thrust_newtons(alt, mach) * self.k


def sample_params(rng):
    """抽一组散布因子"""
    return {
        'basic': rng.uniform(*R_BASIC, size=6),     # Cx..Cn 各自独立
        'kcl': rng.uniform(*R_CTRL), 'kcm': rng.uniform(*R_CTRL), 'kcn': rng.uniform(*R_CTRL),
        'kclp': rng.uniform(*R_DYN), 'kcmq': rng.uniform(*R_DYN), 'kcnr': rng.uniform(*R_DYN),
        'kthr': rng.uniform(*R_THR),
        'kmass': rng.uniform(*R_MASS),
        'cg_dx': rng.uniform(*R_CG), 'cg_dy': rng.uniform(*R_CG),
        'kixx': rng.uniform(*R_INERT), 'kiyy': rng.uniform(*R_INERT), 'kizz': rng.uniform(*R_INERT),
    }


NOMINAL = {  # 名义(无扰动)用于对照
    'basic': np.ones(6), 'kcl': 1, 'kcm': 1, 'kcn': 1, 'kclp': 1, 'kcmq': 1, 'kcnr': 1,
    'kthr': 1, 'kmass': 1, 'cg_dx': 0.0, 'cg_dy': 0.0, 'kixx': 1, 'kiyy': 1, 'kizz': 1,
}


# ===================== 环境构建与单次飞行 =====================
def load_models():
    for k, p in INNER.items():
        if not os.path.exists(p):
            raise FileNotFoundError(f"缺少内环模型 {p}")
    inner = {kk: FastPredictor(PPO.load(p[:-4], device='cpu')) for kk, p in INNER.items()}
    outer_path = next((p for p in OUTER_CANDIDATES if os.path.exists(p)), None)
    if outer_path is None:
        raise FileNotFoundError(f"缺少外环模型，尝试过 {OUTER_CANDIDATES}")
    print(f"  外环模型: {outer_path}")
    outer = PPO.load(outer_path[:-4], device='cpu')
    return inner, outer


def build_env(prm, base_aero, base_engine, inner_models):
    params = dict(BASE_AIRCRAFT)
    params['mass'] *= prm['kmass']
    params['Ixx'] *= prm['kixx']; params['Iyy'] *= prm['kiyy']; params['Izz'] *= prm['kizz']

    aero = PerturbedAero(base_aero, prm['basic'], prm['kcl'], prm['kcm'], prm['kcn'],
                         prm['cg_dx'], prm['cg_dy'])
    engine = PerturbedEngine(base_engine, prm['kthr'])
    sim = FlightSimulator6DOF(aero, engine, params)
    sim.k_clp = prm['kclp']; sim.k_cmq = prm['kcmq']; sim.k_cnr = prm['kcnr']

    inner_env = X47BInnerEnv(sim, stage=3)
    inner_env.max_steps = int(SIM_TIME / inner_env.dt) + 50
    inner_env.trained_models = inner_models
    outer_env = X47BOuterEnv(inner_env, inner_models)
    outer_env.max_steps = int(SIM_TIME / outer_env.outer_dt) + 5
    return outer_env, sim


def run_episode(outer_env, sim, outer_model, turbulence=False, rng=None):
    outer_env.reset(seed=SEED)
    # 冻结控制器：关闭内环自带的随机舵效/突发故障/FTC，确保唯一变量是"参数散布"
    ie = outer_env.inner_env
    ie.eff = {'pitch': 1.0, 'roll': 1.0, 'yaw': 1.0}
    ie._fault_t = 1e9
    ie.ftc_enabled = False
    outer_env.target_yaw = TARGET_YAW
    outer_env.target_alt = TARGET_ALT
    sim.set_initial_state(TARGET_ALT, 200.0, theta_deg=2.0)
    sim.state[6] = 0.0; sim.state[8] = 0.0
    for _ in range(5):
        ie._update_history()
    obs = outer_env._get_obs()

    # Dryden 一阶湍流状态
    wg = np.zeros(3)
    dt_o = outer_env.outer_dt

    hist = {'t': [], 'yaw': [], 'alt': [], 'beta': [], 'alpha': [], 'V': []}
    crashed = False; t = 0.0
    n = int(SIM_TIME / dt_o)
    for k in range(n):
        t = k * dt_o
        if turbulence and rng is not None:
            V = max(np.linalg.norm(sim.state[3:6]), 1.0)
            beta_t = dt_o * V / TURB_L
            wg = (1 - beta_t) * wg + TURB_SIGMA * math.sqrt(max(2 * beta_t, 1e-6)) * rng.standard_normal(3)
            sim.wind = wg
        action, _ = outer_model.predict(obs, deterministic=True)
        obs, _, term, trunc, _ = outer_env.step(action)
        s = sim.state
        V = max(math.sqrt(s[3]**2 + s[4]**2 + s[5]**2), 1.0)
        hist['t'].append(t)
        hist['yaw'].append(math.degrees(s[8]))
        hist['alt'].append(-s[2])
        hist['alpha'].append(math.degrees(math.atan2(s[5], s[3])))
        hist['beta'].append(math.degrees(math.asin(np.clip(s[4] / V, -1, 1))))
        hist['V'].append(V)
        if term:
            crashed = True; break
        if trunc:
            break

    yaw_err = abs(((TARGET_YAW - hist['yaw'][-1] + 180) % 360) - 180) if hist['yaw'] else 999
    m = {
        'survived': (not crashed),
        't_end': t,
        'yaw_err': yaw_err,
        'max_beta': max(abs(b) for b in hist['beta']) if hist['beta'] else 999,
        'max_alpha': max(hist['alpha']) if hist['alpha'] else 999,
        'max_alt_dev': max(abs(a - TARGET_ALT) for a in hist['alt']) if hist['alt'] else 999,
    }
    return hist, m


# ===================== 主流程 =====================
def main():
    print("=" * 66)
    print(" 参数化蒙特卡洛鲁棒性评估 (冻结控制器)")
    print(f" 样本={N_SAMPLES}  时长={SIM_TIME}s  目标航向={TARGET_YAW}°  湍流={ENABLE_TURBULENCE}")
    print("=" * 66)
    rng = np.random.default_rng(SEED)

    base_aero = NeuralAeroDatabase(); base_aero._load_from_pickle('X47B_coeffs.pkl')
    base_engine = EngineDatabase(); base_engine.load1('engine.pkl')
    inner_models, outer_model = load_models()

    # 名义对照
    env0, sim0 = build_env(NOMINAL, base_aero, base_engine, inner_models)
    nom_hist, nom_m = run_episode(env0, sim0, outer_model, turbulence=False)
    print(f"\n[名义] 存活={nom_m['survived']} 航向误差={nom_m['yaw_err']:.1f}° "
          f"max|β|={nom_m['max_beta']:.2f}° maxα={nom_m['max_alpha']:.2f}° "
          f"max|Δh|={nom_m['max_alt_dev']:.0f}m")

    # 蒙特卡洛
    hists, metrics, params = [], [], []
    for i in range(N_SAMPLES):
        prm = sample_params(rng)
        env, sim = build_env(prm, base_aero, base_engine, inner_models)
        h, m = run_episode(env, sim, outer_model, turbulence=ENABLE_TURBULENCE, rng=rng)
        hists.append(h); metrics.append(m); params.append(prm)
        if (i + 1) % 20 == 0:
            sr = 100.0 * np.mean([mm['survived'] for mm in metrics])
            print(f"  已跑 {i+1}/{N_SAMPLES}  当前存活率 {sr:.1f}%")

    surv = np.array([m['survived'] for m in metrics])
    sr = 100.0 * surv.mean()
    ok = [m for m in metrics if m['survived']]
    def stat(key):
        v = np.array([m[key] for m in ok]) if ok else np.array([np.nan])
        return v.mean(), np.percentile(v, 95) if len(v) else np.nan, v.max()

    print("\n" + "=" * 66)
    print(f" 成功率(全程存活): {sr:.1f}%  ({int(surv.sum())}/{N_SAMPLES})")
    print(" 存活样本的性能 (均值 / 95分位 / 最坏):")
    for key, name, unit in [('yaw_err', '航向误差', '°'), ('max_beta', 'max|侧滑|', '°'),
                            ('max_alpha', 'max迎角', '°'), ('max_alt_dev', 'max|高度偏差|', 'm')]:
        mean, p95, mx = stat(key)
        print(f"   {name:10s}: {mean:7.2f} / {p95:7.2f} / {mx:7.2f}  {unit}")
    print("=" * 66)

    # ===================== 可视化 (学术大字体矢量版) =====================
    plt.style.use('bmh')
    
    # 设置为纯英文衬线字体
    plt.rcParams['font.family'] = 'serif'
    plt.rcParams['font.serif'] = ['Times New Roman', 'DejaVu Serif']
    plt.rcParams['axes.unicode_minus'] = False
    
    # 全局字号极度放大
    plt.rcParams['axes.labelsize'] = 18    # X/Y轴标签
    plt.rcParams['axes.titlesize'] = 18    # 子图标题
    plt.rcParams['xtick.labelsize'] = 16   # X轴刻度
    plt.rcParams['ytick.labelsize'] = 16   # Y轴刻度
    plt.rcParams['legend.fontsize'] = 15   # 图例

    # 去除大标题，扩大画布容纳大字体
    fig = plt.figure(figsize=(24, 14))
    gs = GridSpec(3, 4, figure=fig, hspace=0.35, wspace=0.3)

    # 包络1：航向 (Heading)
    ax = fig.add_subplot(gs[0, 0:2])
    for h, m in zip(hists, metrics):
        ax.plot(h['t'], h['yaw'], color='steelblue' if m['survived'] else 'red',
                alpha=0.25, lw=1.2) # 加粗背景蒙特卡洛线条
    ax.plot(nom_hist['t'], nom_hist['yaw'], 'k-', lw=3.0, label='Nominal')
    ax.axhline(TARGET_YAW, color='green', ls='--', lw=2.5, label=f'Target {TARGET_YAW:.0f}°')
    ax.set_title('Heading Tracking Envelope (Red=Unstable)', fontweight='bold')
    ax.set_xlabel('Time [s]', fontweight='bold')
    ax.set_ylabel('Yaw [deg]', fontweight='bold')
    ax.legend(loc='lower right', framealpha=0.85)

    # 包络2：高度 (Altitude)
    ax = fig.add_subplot(gs[0, 2:4])
    for h, m in zip(hists, metrics):
        ax.plot(h['t'], h['alt'], color='seagreen' if m['survived'] else 'red', 
                alpha=0.25, lw=1.2)
    ax.plot(nom_hist['t'], nom_hist['alt'], 'k-', lw=3.0, label='Nominal')
    ax.axhline(TARGET_ALT, color='green', ls='--', lw=2.5, label=f'Target {TARGET_ALT:.0f}m')
    ax.set_title('Altitude Hold Envelope', fontweight='bold')
    ax.set_xlabel('Time [s]', fontweight='bold')
    ax.set_ylabel('Altitude [m]', fontweight='bold')
    ax.legend(loc='lower right', framealpha=0.85)

    # 指标分布直方图 (Histograms)
    def hist_panel(pos, key, title, unit, line=None):
        ax = fig.add_subplot(pos)
        vals = [m[key] for m in metrics if m['survived']]
        ax.hist(vals, bins=20, color='slateblue', alpha=0.8, edgecolor='white', linewidth=1.2)
        if line is not None:
            ax.axvline(line, color='red', ls='--', lw=2.5, label=f'Limit {line}')
            ax.legend(loc='best', framealpha=0.85)
        ax.set_title(title, fontweight='bold')
        ax.set_xlabel(f'{title} [{unit}]', fontweight='bold')
        ax.set_ylabel('Frequency', fontweight='bold')
        
    hist_panel(gs[1, 0], 'yaw_err', 'Final Yaw Error', 'deg')
    hist_panel(gs[1, 1], 'max_beta', 'Max |Sideslip|', 'deg', line=2.0)
    hist_panel(gs[1, 2], 'max_alpha', 'Max Angle of Attack', 'deg', line=8.0)
    hist_panel(gs[1, 3], 'max_alt_dev', 'Max |Alt Error|', 'm')

    # 敏感性散点 (Scatter Plots)
    def scatter_panel(pos, pkey, label, getter):
        ax = fig.add_subplot(pos)
        xs = [getter(p) for p in params]
        ys = [m['max_beta'] for m in metrics]
        cs = ['steelblue' if m['survived'] else 'red' for m in metrics]
        # 散点点放大至 s=60，更加显眼
        ax.scatter(xs, ys, c=cs, s=60, alpha=0.7, edgecolors='white', linewidths=0.5)
        if len(xs) > 2 and np.std(xs) > 1e-9:
            cc = np.corrcoef(xs, ys)[0, 1]
            ax.set_title(f'{label} (r={cc:+.2f})', fontweight='bold')
        else:
            ax.set_title(label, fontweight='bold')
        ax.set_ylabel('Max |Sideslip| [deg]', fontweight='bold')
        ax.set_xlabel(f'{label} Shift/Multiplier', fontweight='bold')
        
    scatter_panel(gs[2, 0], 'cg_dx', 'CG Shift (Chordwise)', lambda p: p['cg_dx'])
    scatter_panel(gs[2, 1], 'kcm', 'Pitch Ctrl Eff (kcm)', lambda p: p['kcm'])
    scatter_panel(gs[2, 2], 'kcn', 'Yaw Ctrl Eff (kcn)', lambda p: p['kcn'])
    scatter_panel(gs[2, 3], 'kiyy', 'Pitch Inertia (Iyy)', lambda p: p['kiyy'])

    plt.tight_layout()
    os.makedirs('./logs/', exist_ok=True)
    
    # 自动保存为双格式 (位图与矢量图)
    out_pdf = './logs/robustness_mc.pdf'
    out_png = './logs/robustness_mc.png'
    plt.savefig(out_pdf, format='pdf', bbox_inches='tight')
    plt.savefig(out_png, format='png', dpi=300, bbox_inches='tight')
    print(f"\n✅ 图表已保存:\n - 矢量图: {out_pdf}\n - 位图: {out_png}")
    
    print("\n判读：")
    print(" · 成功率高(>95%)、各指标分布都贴近名义 → 控制器对该散布鲁棒，达标。")
    print(" · 散点里某参数 corr 大、且红点集中在其一端 → 该参数是薄弱方向。")
    
    plt.show()

if __name__ == "__main__":
    main()