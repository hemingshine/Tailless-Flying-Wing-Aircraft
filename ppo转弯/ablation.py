#coding=utf-8
"""
run_ablation_turn.py —— 6-DOF 协同转弯模型 的四组件消融 runner

任务: 协同转弯(航向 0->90° 平滑爬升保持) + 定高 3000m + severe 体轴湍流(压力源)
变体(5): full / 去①时序 / 去②三阶段课程 / 去③TECS / 去④超前预见
落地:
  ①时序 ②课程  -> 训练期: 加载不同的【内环模型路径】(用内环开关重训的变体)
  ③TECS         -> 测试期: 外环挂能量/油门协调器(开/关)
  ④超前预见      -> 测试期: 参考轨迹前瞻 t_lead(开/关)
产出: 调用 ablation_viz.plot_ablation 出"跟踪叠加+误差叠加+组件贡献柱状+指标表"。
依赖与现有管线一致: fly_robust(sim.wind) + train_inner_fault + train_outerfault + ablation_viz
"""
import os
import math
import warnings
import numpy as np
from stable_baselines3 import PPO

from fly_robust import NeuralAeroDatabase, EngineDatabase, FlightSimulator6DOF
from train_inner_fault import X47BInnerEnv
from train_outerfault import X47BOuterEnv, FastPredictor
import ablationvis as AV

warnings.filterwarnings('ignore')

# ============================ 任务配置 ============================
SIM_TIME   = 120.0
TARGET_ALT = 3000.0
V0         = 200.0
TARGET_V   = 200.0          # TECS 保持的空速
YAW_FINAL  = 90.0
RAMP_T0, RAMP_T1 = 5.0, 65.0   # 航向指令从 0 平滑爬升到 90° 的时段(让"前瞻"有意义)
T_LEAD     = 2.0           # 超前预见前瞻时间(s)

ENABLE_TURB = True
TURB_SIGMA, TURB_L = 3.0, 530.0
SEED = 20260619
N_SEEDS = 5                 # 多 seed 误差棒: 每个变体跑 N_SEEDS 段不同湍流, 指标取 mean±std

BASE_AIRCRAFT = {'S': 88.58, 'b': 18.9, 'c_bar': 4.6, 'mass': 14000,
                 'Ixx': 313220.6, 'Iyy': 273435.3, 'Izz': 392903.9, 'Ixz': 0.0}

# ============================ 5 个变体(路径按你的实际填) ============================
# 第一个必须是 full。①②换内环路径; ③④是测试开关(use_tecs/use_preview)。
INNER_FULL = {'dir': './logs_paper/best_model_stage1/best_model.zip',
              'lat': './logs_paper/best_model_stage2/best_model.zip',
              'lon': './logs_paper/best_model_stage3/best_model.zip'}
OUTER_FULL = './logs/best_model_outer/best_model.zip'

VARIANTS = [
    {'key': 'full',          'label': 'Full (ours)',
     'inner': INNER_FULL, 'outer': OUTER_FULL, 'use_preview': True,  'use_tecs': True},
    {'key': 'no_temporal',   'label': 'w/o Temporal feat.',
     'inner': {'dir': './logs/best_model_stage1_notemp/best_model.zip',
               'lat': './logs/best_model_stage2_notemp/best_model.zip',
               'lon': './logs/best_model_stage3_notemp/best_model.zip'},
     'outer': OUTER_FULL, 'use_preview': True,  'use_tecs': True,'inner_temporal': False},
    {'key': 'no_curriculum', 'label': 'w/o 3-stage curriculum',
     'inner': {'dir': './logs/best_model_stage1fault/best_model.zip',
               'lat': './logs/best_model_stage2fault/best_model.zip',
               'lon': './logs/best_model_stage3_nocurr/best_model.zip'},
     'outer': OUTER_FULL, 'use_preview': True,  'use_tecs': True},
    {'key': 'no_tecs',       'label': 'w/o energy mgmt.',
     'inner': INNER_FULL, 'outer': OUTER_FULL, 'use_preview': True,  'use_tecs': False},
    {'key': 'no_preview',    'label': 'w/o Look-ahead',
     'inner': INNER_FULL, 'outer': OUTER_FULL, 'use_preview': False, 'use_tecs': True},
]


# ============================ 参考轨迹 ============================
def ref_yaw(t):
    """航向指令: [0,RAMP_T0] 保持0, (RAMP_T0,RAMP_T1) 平滑爬升到 YAW_FINAL, 之后保持。"""
    if t <= RAMP_T0:
        return 0.0
    if t >= RAMP_T1:
        return YAW_FINAL
    x = (t - RAMP_T0) / (RAMP_T1 - RAMP_T0)
    s = x * x * (3 - 2 * x)               # smoothstep
    return YAW_FINAL * s

def ref_alt(t):
    return TARGET_ALT


# ============================ 湍流 ============================
def make_turbulence(n, dt, sigma, L, V, seed):
    rng = np.random.default_rng(seed)
    wg = np.zeros(3); seq = np.zeros((n, 3))
    b = dt * V / L; coef = sigma * math.sqrt(max(2.0 * b, 1e-6))
    for k in range(n):
        wg = (1 - b) * wg + coef * rng.standard_normal(3); seq[k] = wg
    return seq


# ============================ 构建与运行 ============================
def _exists_all(cfg):
    for p in list(cfg['inner'].values()) + [cfg['outer']]:
        if not os.path.exists(p):
            return False, p
    return True, None


def run_turn_variant(cfg, wind_seq, seed=SEED):
    inner_models = {k: FastPredictor(PPO.load(p[:-4], device='cpu')) for k, p in cfg['inner'].items()}
    outer_model = PPO.load(cfg['outer'][:-4], device='cpu')

    aero = NeuralAeroDatabase(); aero._load_from_pickle('X47B_coeffs.pkl')
    engine = EngineDatabase(); engine.load1('engine.pkl')
    sim = FlightSimulator6DOF(aero, engine, dict(BASE_AIRCRAFT))
    inner_env = X47BInnerEnv(sim, stage=3, use_temporal=cfg.get('inner_temporal', True))
    inner_env.max_steps = int(SIM_TIME / inner_env.dt) + 50
    inner_env.trained_models = inner_models
    outer_env = X47BOuterEnv(inner_env, inner_models)
    outer_env.max_steps = int(SIM_TIME / outer_env.outer_dt) + 5
    outer_env.reset(seed=seed)

    ie = outer_env.inner_env
    ie.domain_rand = False
    ie.eff = {'pitch': 1.0, 'roll': 1.0, 'yaw': 1.0}
    ie._fault_t = 1e9
    ie.ftc_enabled = False

    sim.set_initial_state(TARGET_ALT, V0, theta_deg=2.0)
    sim.state[6] = 0.0; sim.state[8] = 0.0
    for _ in range(5):
        ie._update_history()

    # ③ TECS: 测试期油门能量协调器(开则按空速误差调油门, 关则固定 0.65)
    tecs = {'iv': 0.0}
    orig_step = sim.step
    def step_tecs(dt_val, controls):
        if cfg['use_tecs']:
            uu, vv, ww = sim.state[3], sim.state[4], sim.state[5]
            V = max(math.sqrt(uu * uu + vv * vv + ww * ww), 1.0)
            ev = TARGET_V - V
            tecs['iv'] = float(np.clip(tecs['iv'] + ev * dt_val, -60.0, 60.0))
            controls['throttle'] = float(np.clip(0.65 + 0.010 * ev + 0.0015 * tecs['iv'], 0.10, 0.95))
        return orig_step(dt_val, controls)
    sim.step = step_tecs

    dt_o = outer_env.outer_dt
    n = int(SIM_TIME / dt_o)
    t_lead = T_LEAD if cfg['use_preview'] else 0.0
    H = {'t': [], 'yaw': [], 'ref_yaw': [], 'alt': [], 'ref_alt': [], 'vel': [], 'ref_vel': []}
    for k in range(n):
        t = k * dt_o
        if ENABLE_TURB:
            sim.wind = wind_seq[k]
        # ④ 超前预见: 喂给外环的目标取 t+t_lead 处(关则取当前)
        outer_env.target_yaw = ref_yaw(t + t_lead)
        outer_env.target_alt = ref_alt(t + t_lead)
        obs = outer_env._get_obs()
        action, _ = outer_model.predict(obs, deterministic=True)
        outer_env.step(action)
        s = sim.state
        V = math.sqrt(s[3] * s[3] + s[4] * s[4] + s[5] * s[5])
        H['t'].append(t)
        H['yaw'].append(math.degrees(s[8]))
        H['ref_yaw'].append(ref_yaw(t))        # 误差对真实参考(非前瞻)算
        H['alt'].append(-s[2])
        H['ref_alt'].append(ref_alt(t))
        H['vel'].append(V)
        H['ref_vel'].append(TARGET_V)
    return {kk: np.asarray(vv) for kk, vv in H.items()}


# ============================ 主流程 ============================
def main():
    dt_o = 0.1
    n = int(SIM_TIME / dt_o)
    seeds = [SEED + i * 101 for i in range(N_SEEDS)]

    # 三个信号各自收集: key -> {'label','t','ref','ys':[...]}
    res_head, res_alt, res_vel = {}, {}, {}
    order = []
    for cfg in VARIANTS:
        ok, miss = _exists_all(cfg)
        if not ok:
            print(f"跳过 {cfg['label']}: 缺模型 {miss}")
            continue
        ys_y, ys_a, ys_v = [], [], []
        t_arr = ref_y = ref_a = ref_v = None
        for sd in seeds:
            wind_seq = make_turbulence(n + 10, dt_o, TURB_SIGMA, TURB_L, V0, sd)
            print(f"运行 {cfg['label']}  (seed={sd}) ...")
            Hh = run_turn_variant(cfg, wind_seq, seed=sd)
            ys_y.append(Hh['yaw']); ys_a.append(Hh['alt']); ys_v.append(Hh['vel'])
            t_arr = Hh['t']; ref_y = Hh['ref_yaw']; ref_a = Hh['ref_alt']; ref_v = Hh['ref_vel']
        key = cfg['key']; order.append(key)
        res_head[key] = {'label': cfg['label'], 't': t_arr, 'ref': ref_y, 'ys': ys_y}
        res_alt[key]  = {'label': cfg['label'], 't': t_arr, 'ref': ref_a, 'ys': ys_a}
        res_vel[key]  = {'label': cfg['label'], 't': t_arr, 'ref': ref_v, 'ys': ys_v}

    if 'full' not in res_head:
        print("缺 full 变体, 无法对比。请先填好 full 的模型路径。"); return

    def ordered(d):
        o = {'full': d['full']}
        for k in order:
            if k != 'full':
                o[k] = d[k]
        return o

    AV.SHOW = False   # 连出三套图, 不阻塞; 看保存的 png 即可

    # ① 航向: 稳态保持段(看湍流下航向保持 —— ①时序/②课程 的主场)
    AV.SIGNAL_LABEL = 'Heading'; AV.SIGNAL_UNIT = 'deg'
    AV.FAULT_WINDOW = (75.0, SIM_TIME)
    AV.TRANSIENT = (RAMP_T0, RAMP_T1 + 5.0)
    AV.SAVE_PREFIX = './ablation_turn_heading'
    AV.plot_ablation(ordered(res_head))

    # ② 高度: 改成只看转弯后的保持段(避开"转弯掉高-爬回"的公共大坑)
    AV.SIGNAL_LABEL = 'Altitude'; AV.SIGNAL_UNIT = 'm'
    AV.FAULT_WINDOW = (75.0, SIM_TIME)        # ← 原 (RAMP_T0, SIM_TIME) 改成保持段
    AV.TRANSIENT = (RAMP_T0, RAMP_T1 + 5.0)
    AV.SAVE_PREFIX = './ablation_turn_alt'
    AV.plot_ablation(ordered(res_alt))

    # ③ 空速: 机动+保持全段(转弯掉速 —— ③能量管理 的直接主场)
    AV.SIGNAL_LABEL = 'Airspeed'; AV.SIGNAL_UNIT = 'm/s'
    AV.FAULT_WINDOW = (75, SIM_TIME)
    AV.TRANSIENT = (RAMP_T0, RAMP_T1 + 5.0)
    AV.SAVE_PREFIX = './ablation_turn_vel'
    AV.plot_ablation(ordered(res_vel))

    print("\n全部完成。三套图: ablation_turn_{heading,alt,vel}_curves.png / _table.png / _metrics.csv")


if __name__ == '__main__':
    main()