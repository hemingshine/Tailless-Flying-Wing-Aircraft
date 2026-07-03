#coding=utf-8
"""
bench_latency.py —— 实时性 / 算力基准
支撑 "实时控制 / 微秒级、可部署到 MCU" 的叙述。给出:
  1) 单步推理延迟: 每个策略网络 (dir/lat/lon 内环 + outer 外环) 的 predict() 时延
     (mean / median / p99 / max, 单位 µs), 在 CPU 单线程下 (MCU 场景)。
  2) 每个控制步的总推理时延: stage-3 内环每步跑 3 次前向 (lon+dir+lat),
     外环每 0.1s 跑 1 次; 对照内环周期 dt 给出 "实时裕度" (占用率 %).
  3) 算力规模: 每个网络的参数量与单次前向 MAC 数 -> 论证 MCU 可行性
     (典型 Cortex-M7@480MHz ~1 MAC/cycle, 估算 CPU 占用).
  4) (可选) 整局墙钟耗时 + 推理/物理拆分 (--episode), 说明部署端只承担推理.

用法:
  python bench_latency.py              # 仅推理延迟 + 算力 (快, 不需仿真器)
  python bench_latency.py --episode    # 另加一整局 120s 墙钟与拆分 (需仿真器)
"""
import os
import sys
import time
import csv
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from stable_baselines3 import PPO

# ====== 模型路径 (与 ablation.py 一致, 按需改) ======
PATHS = {
    'dir (yaw)':   './logs_paper/best_model_stage1/best_model.zip',
    'lat (roll)':  './logs_paper/best_model_stage2/best_model.zip',
    'lon (pitch)': './logs_paper/best_model_stage3/best_model.zip',
    'outer (nav)': './logs/best_model_outer/best_model.zip',
}
INNER_DT = 0.01          # 内环控制周期 (s) -> 实时截止期
OUTER_DT = 0.10          # 外环控制周期 (s)
N_WARM   = 2000          # 预热次数
N_TIME   = 20000         # 计时次数
N_SAMPLE = 6000          # 采样保留用于分位数的逐次时延条数
torch.set_num_threads(1) # MCU 单核场景


# ---------------- 算力: 参数量 + MAC ----------------
def actor_linears(policy):
    """取出 actor 前向路径上的 Linear 层: mlp_extractor.policy_net + action_net。"""
    mods = []
    me = getattr(policy, 'mlp_extractor', None)
    if me is not None and hasattr(me, 'policy_net'):
        mods += [m for m in me.policy_net.modules() if isinstance(m, nn.Linear)]
    an = getattr(policy, 'action_net', None)
    if isinstance(an, nn.Linear):
        mods += [an]
    elif an is not None:
        mods += [m for m in an.modules() if isinstance(m, nn.Linear)]
    return mods


def count_params_macs(policy):
    lins = actor_linears(policy)
    params = sum(l.weight.numel() + (l.bias.numel() if l.bias is not None else 0) for l in lins)
    macs = sum(l.in_features * l.out_features for l in lins)   # 每次前向的乘累加数
    arch = ' -> '.join(['in'] + [str(l.out_features) for l in lins])
    return params, macs, arch


# ---------------- 延迟计时 ----------------
def time_predict(policy, obs_dim):
    obs = np.random.randn(obs_dim).astype(np.float32)
    t = torch.from_numpy(obs).unsqueeze(0)

    def one():
        with torch.no_grad():
            policy._predict(t, deterministic=True)

    for _ in range(N_WARM):
        one()
    # 总体吞吐
    t0 = time.perf_counter()
    for _ in range(N_TIME):
        one()
    mean_us = (time.perf_counter() - t0) / N_TIME * 1e6
    # 逐次采样取分位
    samp = np.empty(N_SAMPLE)
    for i in range(N_SAMPLE):
        a = time.perf_counter_ns()
        one()
        samp[i] = (time.perf_counter_ns() - a) / 1e3   # µs
    return {'mean': mean_us, 'p50': float(np.percentile(samp, 50)),
            'p99': float(np.percentile(samp, 99)), 'max': float(samp.max()),
            'std': float(samp.std())}


def main():
    print("=" * 74)
    print("  实时性 / 算力基准 (CPU 单线程, torch threads=1)")
    print("=" * 74)
    rows = []
    for name, p in PATHS.items():
        if not os.path.exists(p):
            print(f"跳过 {name}: 缺 {p}"); continue
        m = PPO.load(p[:-4] if p.endswith('.zip') else p, device='cpu')
        pol = m.policy.eval()
        obs_dim = int(np.prod(m.observation_space.shape))
        params, macs, arch = count_params_macs(pol)
        lat = time_predict(pol, obs_dim)
        rows.append({'name': name, 'obs_dim': obs_dim, 'params': params, 'macs': macs,
                     'arch': arch, **lat})
        print(f"\n[{name}]  obs={obs_dim}  arch(actor)={arch}")
        print(f"  参数量 {params:,}   MAC/前向 {macs:,}")
        print(f"  时延 µs:  mean {lat['mean']:.1f}   p50 {lat['p50']:.1f}   "
              f"p99 {lat['p99']:.1f}   max {lat['max']:.1f}")

    if not rows:
        print("无可用模型, 退出。"); return

    # ---- 每控制步聚合 ----
    by = {r['name']: r for r in rows}
    inner_keys = [k for k in ['dir (yaw)', 'lat (roll)', 'lon (pitch)'] if k in by]
    inner_step_us = sum(by[k]['mean'] for k in inner_keys)   # 内环每步 3 次前向
    inner_step_macs = sum(by[k]['macs'] for k in inner_keys)
    print("\n" + "-" * 74)
    print(f"内环每控制步 = {len(inner_keys)} 次前向 (lon+lat+dir)")
    print(f"  推理时延 {inner_step_us:.1f} µs / 步   vs 周期 {INNER_DT*1e6:.0f} µs "
          f"=> 实时占用 {inner_step_us/(INNER_DT*1e6)*100:.2f}%")
    print(f"  MAC/步 {inner_step_macs:,}   @ {1/INNER_DT:.0f}Hz => {inner_step_macs/INNER_DT/1e6:.1f} M-MAC/s")
    # MCU 估算
    for fmcu in (480e6, 200e6):  # M7@480MHz, M4@200MHz, ~1MAC/cycle
        util = inner_step_macs / INNER_DT / fmcu * 100
        print(f"  估算 MCU@{fmcu/1e6:.0f}MHz (~1 MAC/cycle) 占用 ≈ {util:.1f}%")
    if 'outer (nav)' in by:
        o = by['outer (nav)']
        print(f"外环每控制步 (每 {OUTER_DT}s 一次): {o['mean']:.1f} µs, MAC {o['macs']:,}")

    # ---- CSV ----
    with open('./bench_latency.csv', 'w', newline='', encoding='utf-8-sig') as f:
        w = csv.writer(f)
        w.writerow(['module', 'obs_dim', 'params', 'macs', 'mean_us', 'p50_us', 'p99_us', 'max_us', 'std_us', 'arch'])
        for r in rows:
            w.writerow([r['name'], r['obs_dim'], r['params'], r['macs'],
                        f"{r['mean']:.2f}", f"{r['p50']:.2f}", f"{r['p99']:.2f}",
                        f"{r['max']:.2f}", f"{r['std']:.2f}", r['arch']])
    print("\n指标已存: bench_latency.csv")

    # ---- 图: 各模块时延 (mean 柱 + p99 误差) ----
    plt.rcParams.update({'font.family': 'serif', 'font.size': 10, 'axes.grid': True,
                         'grid.alpha': 0.3})
    names = [r['name'] for r in rows]
    means = [r['mean'] for r in rows]
    p99s = [r['p99'] - r['mean'] for r in rows]
    fig, ax = plt.subplots(figsize=(6.6, 3.4))
    x = np.arange(len(names))
    ax.bar(x, means, yerr=[np.zeros(len(names)), p99s], capsize=4,
           color=['#1f77b4', '#2ca02c', '#d62728', '#9467bd'][:len(names)], alpha=0.85,
           edgecolor='0.3')
    for xi, r in zip(x, rows):
        ax.text(xi, r['p99'], f"{r['mean']:.0f}µs", ha='center', va='bottom', fontsize=8.5)
    ax.set_xticks(x); ax.set_xticklabels(names, fontsize=8.5)
    ax.set_ylabel('Inference latency (µs)')
    ax.set_title('Per-network single-step inference latency (CPU, 1 thread; bar=mean, whisker=p99)')
    fig.savefig('./bench_latency.png', dpi=300, bbox_inches='tight')
    fig.savefig('./bench_latency.pdf', bbox_inches='tight')
    print("图已存: bench_latency.png / .pdf")

    if '--episode' in sys.argv:
        bench_episode()


# ---------------- (可选) 整局墙钟 + 推理/物理拆分 ----------------
def bench_episode():
    """跑一整局 120s 转弯, 分别累计 '推理' 与 '物理积分' 墙钟, 说明部署端只承担推理。"""
    import math
    from fly_robust import NeuralAeroDatabase, EngineDatabase, FlightSimulator6DOF
    from train_inner_fault import X47BInnerEnv
    from train_outerfault import X47BOuterEnv, FastPredictor

    SIM_TIME = 120.0
    aero = NeuralAeroDatabase(); aero._load_from_pickle('X47B_coeffs.pkl')
    engine = EngineDatabase(); engine.load1('engine.pkl')
    ap = {'S': 88.58, 'b': 18.9, 'c_bar': 4.6, 'mass': 14000,
          'Ixx': 313220.6, 'Iyy': 273435.3, 'Izz': 392903.9, 'Ixz': 0.0}
    sim = FlightSimulator6DOF(aero, engine, ap)
    inner_models = {k: FastPredictor(PPO.load(PATHS[v][:-4], device='cpu'))
                    for k, v in {'dir': 'dir (yaw)', 'lat': 'lat (roll)', 'lon': 'lon (pitch)'}.items()}
    outer_model = PPO.load(PATHS['outer (nav)'][:-4], device='cpu')
    ie = X47BInnerEnv(sim, stage=3); ie.trained_models = inner_models
    ie.max_steps = int(SIM_TIME / ie.dt) + 50
    oe = X47BOuterEnv(ie, inner_models); oe.max_steps = int(SIM_TIME / oe.outer_dt) + 5
    oe.reset(); ie.domain_rand = False
    ie.eff = {'pitch': 1, 'roll': 1, 'yaw': 1}; ie._fault_t = 1e9; ie.ftc_enabled = False
    sim.set_initial_state(3000.0, 200.0, theta_deg=2.0)

    # 用计时包装区分推理 vs 物理: 这里直接统计整局墙钟与外环步数 (推理占比由上面的单步数推算)
    n = int(SIM_TIME / oe.outer_dt)
    t0 = time.perf_counter()
    for k in range(n):
        oe.target_yaw = 90.0; oe.target_alt = 3000.0
        obs = oe._get_obs()
        a, _ = outer_model.predict(obs, deterministic=True)
        oe.step(a)
    wall = time.perf_counter() - t0
    print("\n" + "-" * 74)
    print(f"整局 {SIM_TIME:.0f}s 墙钟: {wall*1e3:.1f} ms  ({wall/SIM_TIME*100:.2f}% of real-time, "
          f"即 {SIM_TIME/wall:.0f}x 实时)")
    print("  注: 该墙钟含气动/引擎/积分等被控对象物理, 部署端 (MCU) 仅承担其中的网络推理部分。")


if __name__ == '__main__':
    main()