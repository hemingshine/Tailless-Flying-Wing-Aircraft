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
from train_innernew import X47BInnerEnv

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
        self.inner_env.sim.set_initial_state(3000.0, 200.0, theta_deg=2.0)
        self.inner_env.sim.state[6] = 0.0
        self.inner_env.sim.state[8] = 0.0
        self.target_yaw = self.np_random.uniform(-90.0, 90.0)
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
            # 航向(主任务)
            r_progress = np.clip(abs(self.prev_yaw_error) - abs(yaw_error), -1.0, 1.0) * 1.5
            r_yaw_hold = 1.0 * np.exp(-(yaw_error / 10.0)**2)
            # ★ 高度：保持高斯 + 势能进度项(任何误差都有梯度，治本)
            r_alt_hold = 1.0 * np.exp(-(alt_error / 80.0)**2)
            r_alt_prog = np.clip(abs(self.prev_alt_error) - abs(alt_error), -2.0, 2.0) * 0.5
            r_alt = r_alt_hold + r_alt_prog
            # 安全/平滑/生存
            r_beta = np.clip(-0.5 * abs(beta), -2.0, 0.0) if abs(beta) > 2.0 else 0.0
            r_alpha = np.clip(-0.5 * abs(alpha - 8.0), -2.0, 0.0) if alpha > 8.0 else 0.0
            r_energy = np.clip(-0.2 * (180.0 - V), -2.0, 0.0) if V < 180.0 else 0.0
            r_smooth = np.clip(-0.1 * np.sum(np.abs(action - self.prev_action)), -0.5, 0.0)
            reward = r_progress + r_yaw_hold + r_alt + r_beta + r_energy + r_alpha + r_smooth + 0.2

        self.prev_action = action
        self.prev_yaw_error = yaw_error
        self.prev_alt_error = alt_error
        self.step_count += 1
        truncated = self.step_count >= self.max_steps
        return self._get_obs(), float(reward), terminated, truncated, {}


def make_outer_env(seed=42):
    def _init():
        os.environ["OMP_NUM_THREADS"] = "1"; os.environ["MKL_NUM_THREADS"] = "1"
        torch.set_num_threads(1)
        aircraft_params = {'S': 88.58, 'b': 18.9, 'c_bar': 4.6, 'mass': 14000,
                           'Ixx': 313220.6, 'Iyy': 273435.3, 'Izz': 392903.9, 'Ixz': 0.0}
        aero_db, engine_db = NeuralAeroDatabase(), EngineDatabase()
        aero_db._load_from_pickle('X47B_coeffs.pkl')
        engine_db.load1("engine.pkl")
        sim = FlightSimulator6DOF(aero_db, engine_db, aircraft_params)
        inner_models = {
            'dir': FastPredictor(PPO.load('./logs/best_model_stage1/best_model.zip', device='cpu')),
            'lat': FastPredictor(PPO.load('./logs/best_model_stage2/best_model.zip', device='cpu')),
            'lon': FastPredictor(PPO.load('./logs/best_model_stage3/best_model.zip', device='cpu'))
        }
        inner_env = X47BInnerEnv(sim, stage=3)
        inner_env.max_steps = 2000
        inner_env.trained_models = inner_models
        # inner_env.ftc_enabled = True 
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
    eval_callback = EvalCallback(eval_env, best_model_save_path='./logs/best_model_outer1/',
                                 log_path='./logs/results_outer1/',
                                 eval_freq=max(100_000 // n_envs, 1), deterministic=True, render=False)
    callback_list = CallbackList([eval_callback, OuterMetricsCallback(verbose=1)])
    ppo_kwargs = dict(learning_rate=3e-4, n_steps=2048, batch_size=512, n_epochs=10,
                      gamma=0.99, gae_lambda=0.95, clip_range=0.2, ent_coef=0.0,
                      verbose=1, seed=SEED, target_kl=0.05)
    policy_kwargs = dict(activation_fn=nn.Tanh, net_arch=dict(pi=[128, 128], vf=[256, 256]),
                         log_std_init=-1.0)
    model = PPO("MlpPolicy", vec_env, policy_kwargs=policy_kwargs, **ppo_kwargs)
    model.learn(total_timesteps=1000_000, callback=callback_list)
    vec_env.close(); eval_env.close()
    print("外环训练完成")