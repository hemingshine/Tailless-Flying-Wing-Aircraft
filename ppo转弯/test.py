#coding=utf-8
"""
外环诊断：一次性查清两件事
  1) 飞机干不干净：连续 reset 看内环舵效 eff 是否在随机变(变=域随机化没关掉=训练在流沙上)
  2) 策略真实行为：在固定目标(+30/-30/+90/-90)上看它捕获/反向/只会小角度
"""
import os, math
import numpy as np
from stable_baselines3 import PPO
from train_outerfault import make_outer_env

env = make_outer_env(seed=123)()

print("=" * 60)
print("① 飞机是否名义一致")
print("=" * 60)
print("inner_env.domain_rand =", getattr(env.inner_env, 'domain_rand', '【属性不存在 → train_inner_fault.py 没换!】'))
print("\n连续 5 次 reset 后的内环舵效(应每次都 1.0；若在变 → 飞机仍随机):")
for i in range(5):
    env.reset()
    eff = env.inner_env.eff
    ft = env.inner_env._fault_t
    clean = all(abs(v - 1.0) < 1e-6 for v in eff.values()) and ft > 1e8
    print(f"  reset {i}: pitch={eff['pitch']:.3f} roll={eff['roll']:.3f} yaw={eff['yaw']:.3f} "
          f"fault_t={ft:.0f}  {'✓干净' if clean else '✗仍随机'}")

print("\n" + "=" * 60)
print("② 策略在固定目标上的真实表现(干净飞机)")
print("=" * 60)
mp = "./logs/best_model_outer/best_model.zip"
if not os.path.exists(mp):
    print(f"找不到 {mp}"); raise SystemExit
model = PPO.load(mp[:-4], device='cpu')

def run_fixed(target, n=1500):
    env.reset()
    env.inner_env.domain_rand = False
    env.inner_env.eff = {'pitch': 1.0, 'roll': 1.0, 'yaw': 1.0}
    env.inner_env._fault_t = 1e9
    env.inner_env.ftc_enabled = False
    env.target_yaw = float(target); env.target_alt = 3000.0
    env.inner_env.sim.set_initial_state(3000.0, 200.0, theta_deg=2.0)
    env.inner_env.sim.state[6] = 0.0; env.inner_env.sim.state[8] = 0.0
    for _ in range(5): env.inner_env._update_history()
    env.prev_yaw_error = ((target + 180) % 360) - 180
    env.prev_alt_error = 0.0
    obs = env._get_obs()
    yaw = roll = 0.0; rolls = []
    for k in range(n):
        a, _ = model.predict(obs, deterministic=True)
        obs, _, term, trunc, _ = env.step(a)
        yaw = math.degrees(env.inner_env.sim.state[8]); rolls.append(env.cmd_phi)
        if term or trunc: break
    err = ((target - yaw + 180) % 360) - 180
    return yaw, err, float(np.mean(rolls[100:])) if len(rolls) > 100 else rolls[-1]

print(" 目标 | 最终航向 | 残差 | 平均坡度指令 | 判定")
for tgt in [30, -30, 90, -90]:
    yaw, err, mroll = run_fixed(tgt)
    # 方向对不对：目标>0应右坡度(+)，目标<0应左坡度(-)
    dir_ok = (tgt > 0 and mroll > 2) or (tgt < 0 and mroll < -2)
    cap = abs(err) < 10
    verdict = "✓捕获" if cap else ("方向对但没到位" if dir_ok else "✗方向错/没动")
    print(f"  {tgt:+4.0f} | {yaw:+7.1f} | {err:+6.1f} | {mroll:+7.1f} | {verdict}")

print("\n判读：")
print(" · 若①里舵效在变 → train_inner_fault.py 没换成带 domain_rand 的版本，先换它(或本次外环加了reset兜底，重训即可)。")
print(" · 若①干净、但②四个目标都没捕获/坐标错 → 才是策略/奖励问题，把这张表发我。")
print(" · 若②里 +30/-30 能捕获、+90/-90 不行 → 是探索/课程问题(大角度没探到)，我加目标课程。")