import os
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

import torch
import torch.nn as nn
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv, VecMonitor, DummyVecEnv
import gymnasium as gym
from gymnasium import spaces
from stable_baselines3.common.callbacks import EvalCallback, BaseCallback, CallbackList

from fly import AeroSurrogate, NeuralAeroDatabase, EngineDatabase, FlightSimulator6DOF
from train_inner_fault import X47BInnerEnv

plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False
torch.set_num_threads(1)


class OuterMetricsCallback(BaseCallback):
    def __init__(self, verbose=0):
        super().__init__(verbose)
        self.episode_rewards = []; self.episode_lengths = []
        self.timesteps = []; self.episode_count = 0

    def _on_step(self) -> bool:
        if hasattr(self.model, 'ep_info_buffer') and len(self.model.ep_info_buffer) > 0:
            for ep_info in self.model.ep_info_buffer:
                self.episode_count += 1
                self.episode_rewards.append(ep_info['r'])
                self.episode_lengths.append(ep_info['l'])
                self.timesteps.append(self.num_timesteps)
                if self.episode_count % 10 == 0:
                    print(f"\n【外环 Ep {self.episode_count} | 步数 {self.num_timesteps:,}】"
                          f" 奖励(近10) {np.mean(self.episode_rewards[-10:]):.2f} | "
                          f"存活(近10) {np.mean(self.episode_lengths[-10:]):.1f}")
            self.model.ep_info_buffer.clear()
        return True

    def _on_training_end(self) -> None:
        if len(self.episode_rewards) == 0: return
        fig, axes = plt.subplots(1, 2, figsize=(15, 5))
        fig.suptitle('X-47B 外环导航制导训练监控', fontsize=16, fontweight='bold')
        rs = pd.Series(self.episode_rewards)
        axes[0].plot(self.timesteps, self.episode_rewards, color='#1f77b4', alpha=0.3)
        axes[0].plot(self.timesteps, rs.rolling(20, min_periods=1).mean(), color='#1f77b4', lw=2)
        axes[0].set_title('平均奖励'); axes[0].grid(alpha=0.3)
        ls = pd.Series(self.episode_lengths)
        axes[1].plot(self.timesteps, self.episode_lengths, color='#ff7f0e', alpha=0.3)
        axes[1].plot(self.timesteps, ls.rolling(20, min_periods=1).mean(), color='#ff7f0e', lw=2)
        axes[1].set_title('存活时长'); axes[1].grid(alpha=0.3)
        save_dir = './logs/metrics_plots/'; os.makedirs(save_dir, exist_ok=True)
        plt.tight_layout(); plt.savefig(os.path.join(save_dir, 'outer_training_metrics_final.png'), dpi=150)


class OuterTargetCurriculum(BaseCallback):
    """目标航向范围从小到大：前 warm 比例训练里从 ±lo° 线性长到 ±hi°。
    先在小角度学会'误差符号→坡度符号'映射，再外推到大角度，跳出'只会小左坡度'的局部最优。"""
    def __init__(self, total_timesteps, warm=0.5, lo=15.0, hi=90.0, verbose=0):
        super().__init__(verbose)
        self.total = total_timesteps; self.warm = warm; self.lo = lo; self.hi = hi
        self._last = None

    def _on_step(self) -> bool:
        frac = min(1.0, self.num_timesteps / (self.total * self.warm))
        rng = self.lo + (self.hi - self.lo) * frac
        if self._last is None or abs(rng - self._last) > 1.0:
            self.training_env.set_attr('target_range', rng)
            self._last = rng
        return True


class FastPredictor:
    def __init__(self, ppo_model):
        self.policy = ppo_model.policy.eval()
        for p in self.policy.parameters():
            p.requires_grad = False
    def predict(self, obs, deterministic=True):
        with torch.no_grad():
            t = torch.from_numpy(obs).float().unsqueeze(0)
            return self.policy._predict(t, deterministic=deterministic).numpy()[0], None


class X47BOuterEnv(gym.Env):
    def __init__(self, inner_env, inner_models):
        super().__init__()
        self.inner_env = inner_env
        self.models = inner_models
        self.outer_dt = 0.1
        self.inner_steps_per_outer = int(self.outer_dt / self.inner_env.dt)
        self.max_steps = 2000
        self.action_space = spaces.Box(-1.0, 1.0, shape=(2,), dtype=np.float32)
        # ★ 观测 11 -> 12：新增爬升率(给定高阻尼)
        self.observation_space = spaces.Box(-np.inf, np.inf, shape=(12,), dtype=np.float32)
        self.target_yaw = 0.0; self.target_alt = 3000.0; self.step_count = 0
        self.target_range = 90.0   # ★ 目标航向采样范围(课程从小到大；评估恒为90)
        self.cmd_phi = 0.0; self.cmd_theta = 2.0
        self.prev_action = np.zeros(2, dtype=np.float32)
        self.prev_yaw_error = 0.0
        self.prev_alt_error = 0.0    # ★ 高度进度项用

    def _climb_rate(self):
        s = self.inner_env.sim.state
        u, v, w = s[3], s[4], s[5]; phi, theta = s[6], s[7]
        return u * math.sin(theta) - v * math.sin(phi) * math.cos(theta) - w * math.cos(phi) * math.cos(theta)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.step_count = 0
        self.inner_env.reset(seed=seed)
        # ★ 兜底：调完内环reset后强制名义一致(不依赖 train_inner_fault.py 是否加了 domain_rand)
        self.inner_env.eff = {'pitch': 1.0, 'roll': 1.0, 'yaw': 1.0}
        self.inner_env._fault_t = 1e9
        self.inner_env.sim.set_initial_state(3000.0, 200.0, theta_deg=2.0)
        self.inner_env.sim.state[6] = 0.0
        self.inner_env.sim.state[8] = 0.0
        self.target_yaw = self.np_random.uniform(-self.target_range, self.target_range)
        self.target_alt = 3000.0
        yaw = np.degrees(self.inner_env.sim.state[8])
        self.prev_yaw_error = ((self.target_yaw - yaw + 180) % 360) - 180
        self.prev_alt_error = self.target_alt - (-self.inner_env.sim.state[2])
        self.cmd_phi = 0.0; self.cmd_theta = 2.0
        self.prev_action = np.zeros(2, dtype=np.float32)
        for _ in range(5):
            self.inner_env._update_history()
        return self._get_obs(), {}

    def _get_obs(self):
        s = self.inner_env.sim.state
        yaw, phi, theta = np.degrees(s[8]), np.degrees(s[6]), np.degrees(s[7])
        alt = -s[2]
        V = max(np.linalg.norm(s[3:6]), 1.0)
        alpha = np.degrees(np.arctan2(s[5], s[3]))
        beta = np.degrees(np.arcsin(np.clip(s[4] / V, -1.0, 1.0)))
        yaw_error = ((self.target_yaw - yaw + 180) % 360) - 180
        alt_error = self.target_alt - alt
        return np.array([
            yaw_error / 180.0,
            phi / 60.0,
            (theta - 2.0) / 15.0,
            alt_error / 100.0,
            (V - 200.0) / 50.0,
            alpha / 15.0,
            beta / 10.0,
            self.cmd_phi / 60.0,
            (self.cmd_theta - 2.0) / 15.0,
            self.prev_action[0],
            self.prev_action[1],
            self._climb_rate() / 20.0,        # ★ 新增：爬升率(定高阻尼)
        ], dtype=np.float32)

    def step(self, action):
        target_phi_intent = float(np.clip(action[0], -1.0, 1.0)) * 35.0
        # ★ 俯仰指令重新居中到配平 2°(原来是 *5+3，中心 3° 一直在爬)
        target_theta_intent = float(np.clip(action[1], -1.0, 1.0)) * 4.0 + 2.0

        reward = 0.0; terminated = False
        V_pre = max(np.linalg.norm(self.inner_env.sim.state[3:6]), 1.0)
        dynamic_scale = np.clip(V_pre / 200.0, 0.5, 1.2)
        max_phi_rate = 30.0 * self.inner_env.dt * dynamic_scale
        max_theta_rate = 10.0 * self.inner_env.dt * dynamic_scale

        for _ in range(self.inner_steps_per_outer):
            self.cmd_phi += np.clip(target_phi_intent - self.cmd_phi, -max_phi_rate, max_phi_rate)
            self.cmd_theta += np.clip(target_theta_intent - self.cmd_theta, -max_theta_rate, max_theta_rate)
            self.inner_env.target_phi = self.cmd_phi
            self.inner_env.target_theta = self.cmd_theta
            self.inner_env.target_beta = 0.0
            obs3 = self.inner_env._get_obs(3)
            act3, _ = self.models['lon'].predict(obs3, deterministic=True)
            _, _, inner_term, _, _ = self.inner_env.step(np.array([act3[0]]))
            if inner_term:
                terminated = True; reward -= 10.0; break

        s = self.inner_env.sim.state
        yaw = np.degrees(s[8])
        yaw_error = ((self.target_yaw - yaw + 180) % 360) - 180
        V = max(np.linalg.norm(s[3:6]), 1.0)
        alpha = np.degrees(np.arctan2(s[5], s[3]))
        beta = np.degrees(np.arcsin(np.clip(s[4] / V, -1.0, 1.0)))
        alt_error = self.target_alt - (-s[2])

        if not terminated:
            # ===== 彻底去掉"躺平地板"：大奖只能靠真正转到目标才拿得到 =====
            # 上一版病因：r_head=2*(1-|err|/180) 常开，±90随机目标下平均|err|≈45°，
            #   不转也白拿 ~1.5/步 + 高度 ≈2/步×2000≈4000 → 策略又躺平在新地板。
            # 现在：进度(密集主驱动) + 捕获奖励(仅接近目标才给) 主导；朝向压到0.3仅作辅助梯度。
            abs_ye = abs(yaw_error)
            # 进度：朝目标转严格涨分、反向严格扣分(密集主驱动，治"不转/反向")
            r_progress = np.clip(abs(self.prev_yaw_error) - abs_ye, -1.0, 1.0) * 4.0
            # 捕获大奖：只有真正接近目标才拿得到(躺平=0，杜绝舒适地板)
            if abs_ye < 5.0:
                r_capture = 4.0
            elif abs_ye < 20.0:
                r_capture = 4.0 * (1.0 - (abs_ye - 5.0) / 15.0)
            else:
                r_capture = 0.0
            # 朝向：很小，仅提供辅助梯度，不足以构成地板
            r_head = 0.3 * (1.0 - abs_ye / 180.0)
            # 高度(次要)
            r_alt_hold = 0.3 * np.exp(-(alt_error / 80.0)**2)
            r_alt_prog = np.clip(abs(self.prev_alt_error) - abs(alt_error), -2.0, 2.0) * 0.3
            # 安全 / 平滑 / 极小生存
            r_beta = np.clip(-0.5 * abs(beta), -2.0, 0.0) if abs(beta) > 2.0 else 0.0
            r_alpha = np.clip(-0.5 * abs(alpha - 8.0), -2.0, 0.0) if alpha > 8.0 else 0.0
            r_energy = np.clip(-0.2 * (180.0 - V), -2.0, 0.0) if V < 180.0 else 0.0
            r_smooth = np.clip(-0.2 * np.sum(np.abs(action - self.prev_action)), -1.0, 0.0)
            reward = (r_progress + r_capture + r_head + r_alt_hold + r_alt_prog
                      + r_beta + r_alpha + r_energy + r_smooth + 0.02)

        self.prev_action = action
        self.prev_yaw_error = yaw_error
        self.prev_alt_error = alt_error
        self.step_count += 1
        truncated = self.step_count >= self.max_steps
        return self._get_obs(), float(reward), terminated, truncated, {}


def make_outer_env(seed=42, use_fault_inner=False):
    """use_fault_inner=False(默认): 用平滑的非故障内环模型当基线，故障容错交给 FTC(部署推荐)。
       =True: 用故障训练内环模型(名义下会高频抖振，一般不推荐)。"""
    def _init():
        os.environ["OMP_NUM_THREADS"] = "1"; os.environ["MKL_NUM_THREADS"] = "1"
        torch.set_num_threads(1)
        aircraft_params = {'S': 88.58, 'b': 18.9, 'c_bar': 4.6, 'mass': 14000,
                           'Ixx': 313220.6, 'Iyy': 273435.3, 'Izz': 392903.9, 'Ixz': 0.0}
        aero_db, engine_db = NeuralAeroDatabase(), EngineDatabase()
        aero_db._load_from_pickle('X47B_coeffs.pkl')
        engine_db.load1("engine.pkl")
        sim = FlightSimulator6DOF(aero_db, engine_db, aircraft_params)
        sfx = 'fault' if use_fault_inner else ''   # ★ 切换内环权重来源
        inner_models = {
            'dir': FastPredictor(PPO.load(f'./logs/best_model_stage1{sfx}/best_model.zip', device='cpu')),
            'lat': FastPredictor(PPO.load(f'./logs/best_model_stage2{sfx}/best_model.zip', device='cpu')),
            'lon': FastPredictor(PPO.load(f'./logs/best_model_stage3{sfx}/best_model.zip', device='cpu'))
        }
        inner_env = X47BInnerEnv(sim, stage=3)
        inner_env.max_steps = 25000          # ≥ 整局内环步数(2000外环×10)，避免中途截断
        inner_env.trained_models = inner_models
        inner_env.domain_rand = False        # ★ 关键：外环在"名义一致"内环上学导航(故障由内环层自己扛)
        inner_env.ftc_enabled = False        # 训练时关FTC(近休眠，部署期再开)
        outer_env = X47BOuterEnv(inner_env, inner_models)
        outer_env.action_space.seed(seed)
        
        return outer_env
    return _init


if __name__ == '__main__':
    import multiprocessing
    SEED = 888
    np.random.seed(SEED); torch.manual_seed(SEED)
    n_envs = min(8, max(1, multiprocessing.cpu_count() // 2))
    print(f"启动外环训练，进程数 {n_envs}")
    vec_env = VecMonitor(SubprocVecEnv([make_outer_env(seed=SEED + i) for i in range(n_envs)]))
    eval_env = VecMonitor(DummyVecEnv([make_outer_env(seed=SEED + 100)]))
    eval_callback = EvalCallback(eval_env, best_model_save_path='./logs/best_model_outerfault/',
                                 log_path='./logs/results_outer/',
                                 eval_freq=max(100_000 // n_envs, 1), deterministic=True, render=False)
    callback_list = CallbackList([eval_callback, OuterMetricsCallback(verbose=1),
                                  OuterTargetCurriculum(total_timesteps=1_000_000, warm=0.5, lo=15.0, hi=90.0)])
    ppo_kwargs = dict(learning_rate=3e-4, n_steps=2048, batch_size=512, n_epochs=10,
                      gamma=0.99, gae_lambda=0.95, clip_range=0.2, ent_coef=0.005,
                      verbose=1, seed=SEED, target_kl=0.05)
    policy_kwargs = dict(activation_fn=nn.Tanh, net_arch=dict(pi=[128, 128], vf=[256, 256]),
                         log_std_init=-0.5)
    model = PPO("MlpPolicy", vec_env, policy_kwargs=policy_kwargs, **ppo_kwargs)
    model.learn(total_timesteps=1_000_000, callback=callback_list)
    vec_env.close(); eval_env.close()
    print("外环训练完成")