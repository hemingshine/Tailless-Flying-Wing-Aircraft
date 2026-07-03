#coding=utf-8
"""
compare_turb.py —— 两套控制器(内环+外环参数均不同)在严重大气湍流下的外环跟踪对比

设计:
  · 两个模型回放【完全相同】的一段 Dryden 体轴湍流(同一阵风序列)，差异只来自控制器本身
  · 任务: 定高 + 90° 协调转弯; 全程注入 severe 量级体轴湍流
  · 输出一张论文级对比图(英文标注/无标题/紧凑排版)，含机体三轴湍流子图
依赖与现有管线一致: fly_robust(带 self.wind 钩子) + train_inner_fault + train_outerfault
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

# ============================ 配置 ============================
SIM_TIME   = 150.0          # 飞行时长(s)
TARGET_YAW = 90.0           # 目标航向(°)
TARGET_ALT = 3000.0         # 目标高度(m)
V0         = 200.0          # 初始/名义空速(m/s)
SEED       = 20260615

# --- 严重大气湍流 (一阶 Dryden 近似, 体轴风) ---
TURB_SIGMA = 3.0            # 湍流强度(m/s, severe 量级)
TURB_L     = 530.0          # 湍流尺度长度(m)

BASE_AIRCRAFT = {'S': 88.58, 'b': 18.9, 'c_bar': 4.6, 'mass': 14000,
                 'Ixx': 313220.6, 'Iyy': 273435.3, 'Izz': 392903.9, 'Ixz': 0.0}

# --- 两个待对比的控制器 (内环三阶段 + 外环, 路径按你的实际填) ---
MODELS = [
    {
        'label': 'Controller A',
        'color': '#1f5fb4', 'ls': '-',
        'inner': {'dir': './logs/best_model_stage1/best_model.zip',
                  'lat': './logs/best_model_stage2/best_model.zip',
                  'lon': './logs/best_model_stage3/best_model.zip'},
        'outer': './logs/best_model_outer/best_model.zip',
        'ftc': False,
    },
    {
        'label': 'Controller B',
        'color': '#2ca02c', 'ls': '--',
        'inner': {'dir': './logs_paper/best_model_stage1/best_model.zip',
                  'lat': './logs_paper/best_model_stage2/best_model.zip',
                  'lon': './logs_paper/best_model_stage3/best_model.zip'},
        'outer': './logs/best_model_outer1/best_model.zip',
        'ftc': False,
    },
]


# ============================ 湍流预生成 ============================
def make_turbulence(n, dt, sigma, L, V, seed):
    """一阶 Dryden 近似的体轴三轴阵风序列, 形状 (n,3)。
    用固定名义空速 V 生成, 保证两模型回放同一段风。"""
    rng = np.random.default_rng(seed)
    wg = np.zeros(3)
    seq = np.zeros((n, 3))
    beta_t = dt * V / L
    coef = sigma * math.sqrt(max(2.0 * beta_t, 1e-6))
    for k in range(n):
        wg = (1.0 - beta_t) * wg + coef * rng.standard_normal(3)
        seq[k] = wg
    return seq


# ============================ 环境与单次飞行 ============================
def load_one(cfg):
    for kk, p in cfg['inner'].items():
        if not os.path.exists(p):
            raise FileNotFoundError(f"[{cfg['label']}] 缺少内环模型 {p}")
    if not os.path.exists(cfg['outer']):
        raise FileNotFoundError(f"[{cfg['label']}] 缺少外环模型 {cfg['outer']}")
    inner = {kk: FastPredictor(PPO.load(p[:-4], device='cpu')) for kk, p in cfg['inner'].items()}
    outer = PPO.load(cfg['outer'][:-4], device='cpu')
    return inner, outer


def build_env(inner_models):
    aero = NeuralAeroDatabase(); aero._load_from_pickle('X47B_coeffs.pkl')
    engine = EngineDatabase(); engine.load1('engine.pkl')
    sim = FlightSimulator6DOF(aero, engine, dict(BASE_AIRCRAFT))
    inner_env = X47BInnerEnv(sim, stage=3)
    inner_env.max_steps = int(SIM_TIME / inner_env.dt) + 50
    inner_env.trained_models = inner_models
    outer_env = X47BOuterEnv(inner_env, inner_models)
    outer_env.max_steps = int(SIM_TIME / outer_env.outer_dt) + 5
    return outer_env, sim


def run_episode(cfg, wind_seq):
    inner_models, outer_model = load_one(cfg)
    outer_env, sim = build_env(inner_models)
    outer_env.reset(seed=SEED)

    ie = outer_env.inner_env
    ie.domain_rand = False
    ie.eff = {'pitch': 1.0, 'roll': 1.0, 'yaw': 1.0}
    ie._fault_t = 1e9
    ie.ftc_enabled = bool(cfg.get('ftc', False))

    outer_env.target_yaw = TARGET_YAW
    outer_env.target_alt = TARGET_ALT
    sim.set_initial_state(TARGET_ALT, V0, theta_deg=2.0)
    sim.state[6] = 0.0; sim.state[8] = 0.0
    for _ in range(5):
        ie._update_history()
    outer_env.prev_yaw_error = ((TARGET_YAW - math.degrees(sim.state[8]) + 180) % 360) - 180
    outer_env.prev_alt_error = TARGET_ALT - (-sim.state[2])
    obs = outer_env._get_obs()

    dt_o = outer_env.outer_dt
    n = int(SIM_TIME / dt_o)
    H = {'t': [], 'yaw': [], 'alt': [], 'phi': [], 'beta': [], 'alpha': [], 'V': []}
    for k in range(n):
        sim.wind = wind_seq[k]                      # ★ 注入同一段体轴湍流
        action, _ = outer_model.predict(obs, deterministic=True)
        obs, _, term, trunc, _ = outer_env.step(action)
        s = sim.state
        V = max(math.sqrt(s[3]**2 + s[4]**2 + s[5]**2), 1.0)
        H['t'].append(k * dt_o)
        H['yaw'].append(math.degrees(s[8]))
        H['alt'].append(-s[2])
        H['phi'].append(math.degrees(s[6]))
        H['alpha'].append(math.degrees(math.atan2(s[5], s[3])))
        H['beta'].append(math.degrees(math.asin(np.clip(s[4] / V, -1.0, 1.0))))
        H['V'].append(V)
        if term or trunc:
            break
    return {k: np.asarray(v) for k, v in H.items()}


def steady_rms(t, y, ref, t0):
    m = t >= t0
    if not np.any(m):
        return float('nan')
    return float(np.sqrt(np.mean((y[m] - ref) ** 2)))


# ============================ 主流程 ============================
def main():
    dt_o = 0.1
    n = int(SIM_TIME / dt_o)
    # 湍流序列(两个模型共用)
    wind_seq = make_turbulence(n + 10, dt_o, TURB_SIGMA, TURB_L, V0, SEED)
    t_wind = np.arange(n) * dt_o

    runs = []
    for cfg in MODELS:
        print(f"运行 {cfg['label']} ...")
        H = run_episode(cfg, wind_seq)
        t_steady = 0.6 * SIM_TIME
        H['rms_yaw'] = steady_rms(H['t'], H['yaw'], TARGET_YAW, t_steady)
        H['rms_alt'] = steady_rms(H['t'], H['alt'], TARGET_ALT, t_steady)
        H['max_beta'] = float(np.max(np.abs(H['beta'])))
        runs.append((cfg, H))
        print(f"  稳态航向RMSE={H['rms_yaw']:.2f}°  高度RMSE={H['rms_alt']:.2f}m  "
              f"max|β|={H['max_beta']:.2f}°")

    # ===================== 论文级可视化 (大图+大字体+防重叠) =====================
    plt.style.use('bmh')
    
    # 全局字体和样式设置 (纯英文论文标准样式)
    plt.rcParams.update({
        'font.family': 'serif', 
        'font.serif': ['Times New Roman', 'DejaVu Serif'],
        'axes.unicode_minus': False,
        'axes.labelsize': 25,       # 轴标签大字体
        'axes.titlesize': 25,       # 子图标题大字体
        'xtick.labelsize': 25,      # 刻度大字体
        'ytick.labelsize': 25,
        'legend.fontsize': 25,      # 图例大字体
        'axes.linewidth': 1.5,
        'lines.linewidth': 2.5,     # 线条加粗
        'axes.grid': True, 
        'grid.alpha': 0.4, 
        'grid.linewidth': 0.8,
    })
    
    # ★ 彻底解放画板空间，扩大到 24x12，解决所有的重叠问题
    fig = plt.figure(figsize=(24, 12))
    # 移除写死的 hspace 和 wspace，交给 tight_layout 自动处理
    gs = GridSpec(2, 3, figure=fig)

    ax_yaw = fig.add_subplot(gs[0, 0])
    ax_alt = fig.add_subplot(gs[0, 1])
    ax_phi = fig.add_subplot(gs[0, 2])
    ax_bet = fig.add_subplot(gs[1, 0])
    ax_alp = fig.add_subplot(gs[1, 1])
    ax_turb = fig.add_subplot(gs[1, 2])

    # --- (a) 航向跟踪 ---
    for cfg, H in runs:
        ax_yaw.plot(H['t'], H['yaw'], color=cfg['color'], ls=cfg['ls'],
                    label=f"{cfg['label']} (RMSE={H['rms_yaw']:.2f}\u00b0)")
    ax_yaw.axhline(TARGET_YAW, color='k', ls=':', lw=2.0, label=f'Reference {TARGET_YAW:.0f}\u00b0')
    ax_yaw.set_xlabel('Time [s]', fontweight='bold')
    ax_yaw.set_ylabel('Heading [deg]', fontweight='bold')
    ax_yaw.set_title('(a) Heading Tracking', fontweight='bold')
    ax_yaw.legend(loc='best', framealpha=0.85)

    # --- (b) 高度保持 ---
    for cfg, H in runs:
        ax_alt.plot(H['t'], H['alt'], color=cfg['color'], ls=cfg['ls'],
                    label=f"{cfg['label']} (RMSE={H['rms_alt']:.1f} m)")
    ax_alt.axhline(TARGET_ALT, color='k', ls=':', lw=2.0, label=f'Reference {TARGET_ALT:.0f} m')
    ax_alt.set_xlabel('Time [s]', fontweight='bold')
    ax_alt.set_ylabel('Altitude [m]', fontweight='bold')
    ax_alt.set_title('(b) Altitude Hold', fontweight='bold')
    ax_alt.legend(loc='best', framealpha=0.85)

    # --- (c) 滚转角 ---
    for cfg, H in runs:
        ax_phi.plot(H['t'], H['phi'], color=cfg['color'], ls=cfg['ls'], label=cfg['label'])
    ax_phi.set_xlabel('Time [s]', fontweight='bold')
    ax_phi.set_ylabel('Bank Angle [deg]', fontweight='bold')
    ax_phi.set_title('(c) Bank Angle', fontweight='bold')
    ax_phi.legend(loc='best', framealpha=0.85)

    # --- (d) 侧滑角 ---
    for cfg, H in runs:
        ax_bet.plot(H['t'], H['beta'], color=cfg['color'], ls=cfg['ls'], label=cfg['label'])
    ax_bet.axhline(2.0, color='gray', ls=':', lw=1.5)
    ax_bet.axhline(-2.0, color='gray', ls=':', lw=1.5)
    ax_bet.set_xlabel('Time [s]', fontweight='bold')
    ax_bet.set_ylabel('Sideslip [deg]', fontweight='bold')
    ax_bet.set_title('(d) Sideslip Angle', fontweight='bold')
    ax_bet.legend(loc='best', framealpha=0.85)

    # --- (e) 迎角 ---
    for cfg, H in runs:
        ax_alp.plot(H['t'], H['alpha'], color=cfg['color'], ls=cfg['ls'], label=cfg['label'])
    ax_alp.set_xlabel('Time [s]', fontweight='bold')
    ax_alp.set_ylabel('Angle of Attack [deg]', fontweight='bold')
    ax_alp.set_title('(e) Angle of Attack', fontweight='bold')
    ax_alp.legend(loc='best', framealpha=0.85)

    # --- (f) 机体三轴大气湍流 ---
    # 湍流线稍微细一点，防变纯色块
    ax_turb.plot(t_wind, wind_seq[:n, 0], color='#1f77b4', lw=1.8, label=r'$u_g$ (long.)')
    ax_turb.plot(t_wind, wind_seq[:n, 1], color='#2ca02c', lw=1.8, label=r'$v_g$ (lat.)')
    ax_turb.plot(t_wind, wind_seq[:n, 2], color='#ff7f0e', lw=1.8, label=r'$w_g$ (vert.)')
    ax_turb.set_xlabel('Time [s]', fontweight='bold')
    ax_turb.set_ylabel('Gust Velocity [m/s]', fontweight='bold')
    ax_turb.set_title(rf'(f) Body-Axis Turbulence ($\sigma$={TURB_SIGMA:.0f} m/s)', fontweight='bold')
    ax_turb.legend(loc='best', ncol=1, framealpha=0.85) # 恢复单列或最佳排布防遮挡

    # ★ 使用 tight_layout 自动重算间距，这是防止标题和刻度重叠的关键
    plt.tight_layout()
    
    os.makedirs('./logs/', exist_ok=True)
    fig.savefig('./logs/compare_turb.pdf', format='pdf', bbox_inches='tight')   # 矢量, 适合 LaTeX
    fig.savefig('./logs/compare_turb.png', format='png', dpi=300, bbox_inches='tight')
    print('\n✅ 图表已成功保存:\n - 矢量图: ./logs/compare_turb.pdf\n - 位图: ./logs/compare_turb.png')
    
    plt.show()

if __name__ == '__main__':
    main()