import os
import math
import numpy as np
from collections import deque
import torch
import torch.nn as nn
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv, VecMonitor, DummyVecEnv
import gymnasium as gym
from gymnasium import spaces
from stable_baselines3.common.callbacks import EvalCallback, BaseCallback, CallbackList
import matplotlib.pyplot as plt
import pandas as pd
import warnings
warnings.filterwarnings('ignore')
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

# 底层物理引擎
from fly import AeroSurrogate, NeuralAeroDatabase, EngineDatabase, FlightSimulator6DOF

GRAV = 9.80665

# =================================================================
# 全局可调超参 (集中在此，方便扫参)
# =================================================================
# 课程学习：command/扰动范围在 curriculum level c∈[0,1] 上线性插值
# c=0 是“原始小范围”，c=1 是“外环需要的全包络”
ENV_CFG = dict(
    # 进度奖励(势能塑形)系数：无论误差多大都能给出梯度，专治大初始误差下高斯梯度消失
    K_PROG=2.0,
    # 终止软边界（比原来宽很多，给大坡度暂态留出恢复空间）
    PHI_LIM=80.0, THETA_LIM=40.0, BETA_LIM=25.0,
    ALPHA_STALL=18.0, ALPHA_MIN=-12.0, V_MIN=120.0,  # TEST1: α>~8° Cm 反号上仰，硬终止留到 18°
    TERM_PENALTY=10.0,       # 终止惩罚(机会成本量级，而非 -1000 雪崩)
    SURVIVE=0.1,             # 微弱生存低保，避免探索期“自杀式”提前终止
    # 俯仰载荷因子前馈：test_fly2 实测该机型 200m/s 下转弯不掉高(反而微爬)，
    # 前馈会把俯仰目标抬过头(30°/FF=10 直接顶翻)，故关闭(=0)。高度交给外环 cmd_theta 管。
    LOAD_FF_DEG=0.0,
)


# =================================================================
# 1. 强鲁棒 PID 兜底控制器
#    新增：① 按动压(速度)增益调度 ② 载荷因子前馈接口 ③ 抗积分饱和
# =================================================================
PID_CFG = {
    'pitch': {'kp': 2.0, 'ki': 0.5, 'kd': 1.5, 'limit': 25.0},
    'roll':  {'kp': 4.5, 'ki': 0.2, 'kd': 2.5, 'limit': 20.0},
}


class RobustPID:
    def __init__(self, kp, ki, kd, limit):
        self.kp, self.ki, self.kd, self.limit = kp, ki, kd, limit
        self.prev_error = 0.0
        self.prev_derivative = 0.0
        self.integral = 0.0
        self.alpha = 0.2  # 导数低通滤波系数

    def reset(self):
        self.prev_error = 0.0
        self.prev_derivative = 0.0
        self.integral = 0.0

    def compute(self, error, dt, gain_scale=1.0):
        # 增益调度：舵效 ∝ 动压 ∝ V^2，低速时放大增益维持带宽稳定
        kp, ki, kd = self.kp * gain_scale, self.ki * gain_scale, self.kd * gain_scale
        self.integral = np.clip(self.integral + error * dt, -2.0, 2.0)
        raw_derivative = (error - self.prev_error) / dt
        derivative = self.alpha * raw_derivative + (1 - self.alpha) * self.prev_derivative
        out = kp * error + ki * self.integral + kd * derivative
        self.prev_error = error
        self.prev_derivative = derivative
        return float(np.clip(out, -self.limit, self.limit))


# =================================================================
# 2. 课程学习回调：随训练进度把 command 范围从“易”推到“全包络”
# =================================================================
class CurriculumCallback(BaseCallback):
    def __init__(self, total_timesteps, warmup_frac=0.6, verbose=0):
        super().__init__(verbose)
        self.total = total_timesteps
        self.warmup = warmup_frac      # 在前 warmup_frac 的训练里把难度从 0 拉到 1
        self.last_level = -1.0

    def _on_step(self) -> bool:
        frac = self.num_timesteps / max(self.total * self.warmup, 1)
        level = float(np.clip(frac, 0.0, 1.0))
        # 减少进程间通信：每变化 0.05 才广播一次
        if abs(level - self.last_level) >= 0.05 or level >= 1.0 and self.last_level < 1.0:
            try:
                self.training_env.env_method('set_curriculum', level)
            except Exception:
                pass
            self.last_level = level
            if self.verbose:
                print(f"\n[课程] 总步数 {self.num_timesteps:,} -> 难度 level={level:.2f}")
        return True


# =================================================================
# 3. 训练指标可视化回调 (沿用你的版本，标题通用化)
# =================================================================
class TrainingMetricsCallback(BaseCallback):
    def __init__(self, tag='stage', verbose=0):
        super().__init__(verbose)
        self.tag = tag
        self.episode_rewards = []
        self.episode_lengths = []
        self.kl_divergences = []
        self.value_losses = []
        self.action_stds = []
        self.timesteps = []
        self.episode_count = 0

    def _on_training_start(self) -> None:
        self.episode_rewards, self.episode_lengths = [], []
        self.kl_divergences, self.value_losses, self.action_stds = [], [], []
        self.timesteps, self.episode_count = [], 0
        print(f"\n===== [{self.tag}] 开始收集训练指标 =====")

    def _on_step(self) -> bool:
        if hasattr(self.model, 'ep_info_buffer') and len(self.model.ep_info_buffer) > 0:
            for ep_info in self.model.ep_info_buffer:
                self.episode_count += 1
                self.episode_rewards.append(ep_info['r'])
                self.episode_lengths.append(ep_info['l'])
                self.timesteps.append(self.num_timesteps)
                if self.episode_count % 10 == 0:
                    avg_r = np.mean(self.episode_rewards[-10:])
                    avg_len = np.mean(self.episode_lengths[-10:])
                    print(f"\n【{self.tag} Ep {self.episode_count} | 步数 {self.num_timesteps:,}】"
                          f" 奖励(近10) {avg_r:.2f} | 存活(近10) {avg_len:.1f}")
            self.model.ep_info_buffer.clear()
        return True

    def _on_rollout_end(self) -> None:
        logger = self.model.logger
        kl = logger.name_to_value.get('train/approx_kl', None)
        if kl is not None:
            self.kl_divergences.append(kl)
        v_loss = logger.name_to_value.get('train/value_loss', None)
        if v_loss is not None:
            self.value_losses.append(v_loss)
        log_std = self.model.policy.log_std
        if log_std is not None:
            self.action_stds.append(torch.exp(log_std).detach().cpu().numpy()[0])
        if len(self.kl_divergences) > 0:
            print(f"\n【{self.tag} Rollout {len(self.kl_divergences)} | 步数 {self.num_timesteps:,}】"
                  f" KL {self.kl_divergences[-1]:.4f} | VLoss {self.value_losses[-1] if self.value_losses else 0:.3f}"
                  f" | std {self.action_stds[-1] if self.action_stds else 0:.3f}")

    def _on_training_end(self) -> None:
        print(f"\n===== [{self.tag}] 训练结束，生成图表 =====")
        if len(self.episode_rewards) == 0:
            return
        fig, axes = plt.subplots(2, 2, figsize=(14, 9))
        fig.suptitle(f'X-47B 内环 {self.tag} 训练监控', fontsize=15, fontweight='bold')

        rs = pd.Series(self.episode_rewards)
        axes[0, 0].plot(self.timesteps, self.episode_rewards, color='#1f77b4', alpha=0.3)
        axes[0, 0].plot(self.timesteps, rs.rolling(20, min_periods=1).mean(), color='#1f77b4', lw=2)
        axes[0, 0].set_title('平均奖励'); axes[0, 0].grid(alpha=0.3)

        ls = pd.Series(self.episode_lengths)
        axes[0, 1].plot(self.timesteps, self.episode_lengths, color='#ff7f0e', alpha=0.3)
        axes[0, 1].plot(self.timesteps, ls.rolling(20, min_periods=1).mean(), color='#ff7f0e', lw=2)
        axes[0, 1].set_title('存活时长'); axes[0, 1].grid(alpha=0.3)

        if self.value_losses:
            steps = [i * self.model.n_steps * self.model.n_envs for i in range(len(self.value_losses))]
            axes[1, 0].plot(steps, self.value_losses, color='#d62728', alpha=0.4)
            axes[1, 0].plot(steps, pd.Series(self.value_losses).rolling(5, min_periods=1).mean(),
                            color='#d62728', lw=2)
        axes[1, 0].set_title('Value Loss'); axes[1, 0].grid(alpha=0.3)

        if self.action_stds:
            steps = [i * self.model.n_steps * self.model.n_envs for i in range(len(self.action_stds))]
            axes[1, 1].plot(steps, self.action_stds, color='#9467bd', lw=2)
        axes[1, 1].set_title('动作标准差'); axes[1, 1].grid(alpha=0.3)

        save_dir = './logs/metrics_plots/'
        os.makedirs(save_dir, exist_ok=True)
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, f'training_metrics_{self.tag}.png'), dpi=150)
        print(f"✅ 图表已保存: {save_dir}training_metrics_{self.tag}.png")


# =================================================================
# 4. 内环强化学习核心环境 (全包络 + 协调转弯感知奖励)
# =================================================================
class X47BInnerEnv(gym.Env):
    def __init__(self, simulator, stage=1, model_paths=None,use_temporal=True):
        super(X47BInnerEnv, self).__init__()

        from fault_ftc import InnerFTC
        self.ftc = InnerFTC(pitch=True, roll=True, yaw=False)  # 偏航舵弱+强耦合，先不开
        self.ftc_enabled = False   # 训练时关，部署时开

        self.use_temporal = use_temporal        # 放在 dim 那行之前
        self.sim = simulator
        self.stage = stage
        self.dt = 0.01
        self.max_steps = 2000
        self.alpha_lpf = 0.1
        self.smoothed_act_dir = 0.0
        self.curr = 1.0  # 课程难度，默认全包络(评估/外环不调用 set_curriculum 时即满难度)

        self.trained_models = {}
        if model_paths:
            for k, path in model_paths.items():
                if os.path.exists(path + ".zip"):
                    self.trained_models[k] = PPO.load(path, device='cpu')
                else:
                    raise FileNotFoundError(f"缺失阶段模型: {path}.zip")

        self.pid_pitch = RobustPID(**PID_CFG['pitch'])
        self.pid_roll = RobustPID(**PID_CFG['roll'])

        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(1,), dtype=np.float32)
        # dim = 16 if self.stage == 3 else 15

        if self.use_temporal:
            dim = 16 if self.stage == 3 else 15
        else:
            dim = 8 if self.stage == 3 else 7  

        self.observation_space = spaces.Box(-np.inf, np.inf, shape=(dim,), dtype=np.float32)

        self.hist_e_theta = deque([0.0] * 5, maxlen=5); self.hist_q = deque([0.0] * 5, maxlen=5)
        self.hist_e_phi = deque([0.0] * 5, maxlen=5);   self.hist_p = deque([0.0] * 5, maxlen=5)
        self.hist_e_beta = deque([0.0] * 5, maxlen=5);  self.hist_r = deque([0.0] * 5, maxlen=5)

        self.filtered_p_dot, self.filtered_q_dot, self.filtered_r_dot = 0.0, 0.0, 0.0
        self.target_theta, self.target_phi, self.target_beta = 0.0, 0.0, 0.0
        self.prev_actions = {'e': 0.0, 'a': 0.0, 'r': 0.0, 'net_out': 0.0}
        # ★ 一阶作动器动力学(低通)：模拟真实舵机有限带宽，压平 RL 高频控制毛刺
        #   tau≈0.04s → 截止≈25rad/s(~4Hz)，远高于姿态环带宽(~1Hz)，不影响跟踪；设0关闭
        self.act_lpf_tau = 0.04
        self.act_lpf = {'e': 0.0, 'a': 0.0, 'r': 0.0}
        self.prev_abs_err = {'beta': 0.0, 'phi': 0.0, 'theta': 0.0}
        self.step_count = 0

    # 课程接口：被 CurriculumCallback 通过 env_method 调用
    def set_curriculum(self, level):
        self.curr = float(np.clip(level, 0.0, 1.0))

    @staticmethod
    def _lerp(easy, full, c):
        return easy + (full - easy) * c

    def _gain_scale(self):
        # 由当前速度推动压增益调度
        s = self.sim.state
        V = max(math.sqrt(s[3]**2 + s[4]**2 + s[5]**2), 1.0)
        return float(np.clip((200.0 / V) ** 2, 0.7, 1.6))

    def _pitch_feedforward_deg(self):
        # 载荷因子前馈：把俯仰目标按坡度抬高，banked 时帮飞机抗掉高度损失
        phi = self.sim.state[6]
        return ENV_CFG['LOAD_FF_DEG'] * (1.0 / max(math.cos(phi), 0.5) - 1.0)

    def _update_history(self):
        state = self.sim.state
        V = max(math.sqrt(state[3]**2 + state[4]**2 + state[5]**2), 1.0)
        beta = math.degrees(math.asin(np.clip(state[4] / V, -1.0, 1.0)))
        theta, phi = math.degrees(state[7]), math.degrees(state[6])
        p, q, r = math.degrees(state[9]), math.degrees(state[10]), math.degrees(state[11])
        self.hist_e_theta.appendleft(self.target_theta - theta)
        self.hist_e_phi.appendleft(self.target_phi - phi)
        self.hist_e_beta.appendleft(self.target_beta - beta)
        self.hist_q.appendleft(q); self.hist_p.appendleft(p); self.hist_r.appendleft(r)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.ftc.reset()
        self.pid_pitch.reset(); self.pid_roll.reset()
        self.smoothed_act_dir = 0.0
        self.sim.state = np.zeros(12)
        self.filtered_p_dot, self.filtered_q_dot, self.filtered_r_dot = 0.0, 0.0, 0.0
        self.step_count = 0
        self.prev_actions = {'e': 0.0, 'a': 0.0, 'r': 0.0, 'net_out': 0.0}
        self.act_lpf = {'e': 0.0, 'a': 0.0, 'r': 0.0}

        c = self.curr
        # ---- 初始状态随机化(随课程展开) ----
        h_init = self.np_random.uniform(2800.0, 3200.0)
        V_lo = self._lerp(195.0, 175.0, c); V_hi = self._lerp(205.0, 215.0, c)
        V_init = self.np_random.uniform(V_lo, V_hi)

        init_theta_amp = self._lerp(1.5, 4.0, c)
        init_phi_amp = self._lerp(10.0, 35.0, c)
        initial_theta = 2.0 + self.np_random.uniform(-init_theta_amp, init_theta_amp)
        initial_phi = self.np_random.uniform(-init_phi_amp, init_phi_amp)

        self.sim.set_initial_state(h_init, V_init, theta_deg=initial_theta)
        self.sim.state[6] = math.radians(initial_phi)
        # 满课程注入初始角速率扰动，训练抗扰
        rate_amp = self._lerp(0.0, 5.0, c)
        self.sim.state[9] = math.radians(self.np_random.uniform(-rate_amp, rate_amp))
        self.sim.state[10] = math.radians(self.np_random.uniform(-rate_amp, rate_amp))
        self.sim.state[11] = math.radians(self.np_random.uniform(-rate_amp, rate_amp))

        # ---- command 随机化(全包络) ----
        self.target_beta = 0.0
        if self.stage == 1:
            # 偏航(β)轴：在“易→±40°坡度”的全范围内学习把侧滑压到 0(协调转弯核心)
            self.target_phi = self.np_random.uniform(-self._lerp(10.0, 40.0, c),
                                                      self._lerp(10.0, 40.0, c))
            self.target_theta = 2.0
        elif self.stage == 2:
            self.target_phi = self.np_random.uniform(-self._lerp(15.0, 40.0, c),
                                                      self._lerp(15.0, 40.0, c))
            self.target_theta = 2.0
        elif self.stage == 3:
            self.target_theta = self.np_random.uniform(-2.0, self._lerp(8.0, 12.0, c))
            big_bank = self._lerp(15.0, 40.0, c)
            self.target_phi = self.np_random.uniform(-big_bank, big_bank) \
                if self.np_random.random() < 0.6 else 0.0

        self.prev_abs_err = {
            'beta': abs(self.hist_e_beta[0]),
            'phi': abs(self.hist_e_phi[0]),
            'theta': abs(self.hist_e_theta[0]),
        }

        # ---- 5 步 PID 预热，建立平滑误差导数 ----
        gscale = self._gain_scale()
        for _ in range(5):
            state = self.sim.state
            theta, phi = math.degrees(state[7]), math.degrees(state[6])
            theta_cmd = self.target_theta + self._pitch_feedforward_deg()
            delta_e = -self.pid_pitch.compute(theta_cmd - theta, self.dt, gscale)
            delta_a = 0.0
            if self.stage == 1:
                delta_a = self.pid_roll.compute(self.target_phi - phi, self.dt, gscale)
            controls = {
                'd_flap_L': np.clip(delta_e, -25.0, 25.0), 'd_flap_R': np.clip(delta_e, -25.0, 25.0),
                'd_ail_L': np.clip(delta_a, -20.0, 20.0), 'd_ail_R': np.clip(-delta_a, -20.0, 20.0),
                'd_spoil_L': 0.0, 'd_spoil_R': 0.0, 'throttle': 0.65,
            }
            self.sim.step(self.dt, controls)
            self._update_history()
            gscale = self._gain_scale()

        self.prev_abs_err = {
            'beta': abs(self.hist_e_beta[0]),
            'phi': abs(self.hist_e_phi[0]),
            'theta': abs(self.hist_e_theta[0]),
        }
        # 舵效域随机化(容错训练):随课程允许更大舵效损失。
        # domain_rand=False 时(外环训练/纯净评估)强制名义飞机，保证一致、可学。
        if getattr(self, 'domain_rand', True):
            c = self.curr
            lo_pr = self._lerp(1.0, 0.55, c)   # 俯仰/滚转最低掉到 0.55
            lo_y  = self._lerp(1.0, 0.70, c)   # 偏航舵本就弱，少削一点
            self.eff = {'pitch': self.np_random.uniform(lo_pr, 1.0),
                        'roll':  self.np_random.uniform(lo_pr, 1.0),
                        'yaw':   self.np_random.uniform(lo_y, 1.0)}
            # 30% 概率注入突发失效(某轴某刻舵效骤降)
            self._fault_t = self.np_random.uniform(3.0, 12.0) if self.np_random.random() < 0.3 else 1e9
            self._fault_axis = self.np_random.choice(['pitch', 'roll'])   # 去掉 'yaw'
            self._fault_scale = self.np_random.uniform(0.3, 0.7)
        else:
            self.eff = {'pitch': 1.0, 'roll': 1.0, 'yaw': 1.0}
            self._fault_t = 1e9
            self._fault_axis = 'pitch'
            self._fault_scale = 1.0
        return self._get_obs(self.stage), {}

    def _get_obs(self, stage_req):
        # ★ 保持与原版完全一致的观测格式，确保 train_outer 接口不变
        state = self.sim.state
        

        V = max(math.sqrt(state[3]**2 + state[4]**2 + state[5]**2), 1.0)
        alpha = math.degrees(math.atan2(state[5], state[3]))
        beta = math.degrees(math.asin(np.clip(state[4] / V, -1.0, 1.0)))
        phi = math.degrees(state[6])
        p, r = math.degrees(state[9]), math.degrees(state[11])

        raw_q_dot = (self.hist_q[0] - self.hist_q[1]) / self.dt
        raw_p_dot = (self.hist_p[0] - self.hist_p[1]) / self.dt
        raw_r_dot = (self.hist_r[0] - self.hist_r[1]) / self.dt
        self.filtered_q_dot = (1 - self.alpha_lpf) * self.filtered_q_dot + self.alpha_lpf * raw_q_dot
        self.filtered_p_dot = (1 - self.alpha_lpf) * self.filtered_p_dot + self.alpha_lpf * raw_p_dot
        self.filtered_r_dot = (1 - self.alpha_lpf) * self.filtered_r_dot + self.alpha_lpf * raw_r_dot
        q_dot, p_dot, r_dot = self.filtered_q_dot, self.filtered_p_dot, self.filtered_r_dot
        def _get_hist(err_q, rate_q):
            return [err_q[i] for i in range(5)] + [rate_q[i] for i in range(5)]

        # if stage_req == 1:
        #     base = _get_hist(self.hist_e_beta, self.hist_r)
        #     return np.array([base[0], base[5], self.prev_actions['r'], r_dot] +
        #                     base[1:5] + base[6:10] + [phi, p, self.prev_actions['a']], dtype=np.float32)
        # elif stage_req == 2:
        #     base = _get_hist(self.hist_e_phi, self.hist_p)
        #     return np.array([base[0], base[5], self.prev_actions['a'], p_dot] +
        #                     base[1:5] + base[6:10] + [beta, r, self.prev_actions['r']], dtype=np.float32)
        # elif stage_req == 3:
        #     base = _get_hist(self.hist_e_theta, self.hist_q)
        #     return np.array([base[0], base[5], alpha, self.prev_actions['e'], q_dot] +
        #                     base[1:5] + base[6:10] + [phi, p, r], dtype=np.float32)

        if stage_req == 1:
            base = _get_hist(self.hist_e_beta, self.hist_r)
            H = (base[1:5] + base[6:10]) if self.use_temporal else []
            return np.array([base[0], base[5], self.prev_actions['r'], r_dot] +
                            H + [phi, p, self.prev_actions['a']], dtype=np.float32)
        elif stage_req == 2:
            base = _get_hist(self.hist_e_phi, self.hist_p)
            H = (base[1:5] + base[6:10]) if self.use_temporal else []
            return np.array([base[0], base[5], self.prev_actions['a'], p_dot] +
                            H + [beta, r, self.prev_actions['r']], dtype=np.float32)
        elif stage_req == 3:
            base = _get_hist(self.hist_e_theta, self.hist_q)
            H = (base[1:5] + base[6:10]) if self.use_temporal else []
            return np.array([base[0], base[5], alpha, self.prev_actions['e'], q_dot] +
                            H + [phi, p, r], dtype=np.float32)



    def _coordinated_yaw_rate_deg(self):
        # 协调转弯的运动学偏航角速率(机体近似)：r_coord ≈ (g/V) sinφ
        # 来自文献“maneuvering turn angular rate estimation and subtraction”思想：
        # 偏航阻尼只惩罚“残差”角速率，避免阻尼器与协调转弯本身对抗
        s = self.sim.state
        V = max(math.sqrt(s[3]**2 + s[4]**2 + s[5]**2), 1.0)
        return math.degrees((GRAV / V) * math.sin(s[6]))

    def step(self, action):
        raw_act = float(np.clip(action[0], -1.0, 1.0))
        gscale = self._gain_scale()

        # 读取步前姿态(给 PID 用)
        state = self.sim.state
        theta_pre, phi_pre = math.degrees(state[7]), math.degrees(state[6])
        theta_cmd = self.target_theta + self._pitch_feedforward_deg()

        delta_e, delta_a, delta_r = 0.0, 0.0, 0.0

        # ======= 渐进式三轴独立接管，组装舵面指令 =======
        if self.stage == 1:
            self.smoothed_act_dir = 0.8 * self.smoothed_act_dir + 0.2 * raw_act
            delta_r = self.smoothed_act_dir * 25.0
            delta_e = -self.pid_pitch.compute(theta_cmd - theta_pre, self.dt, gscale)
            delta_a = self.pid_roll.compute(self.target_phi - phi_pre, self.dt, gscale)
        elif self.stage == 2:
            delta_a = raw_act * 20.0
            delta_e = -self.pid_pitch.compute(theta_cmd - theta_pre, self.dt, gscale)
            obs_dir = self._get_obs(1)
            with torch.no_grad():
                act_r, _ = self.trained_models['dir'].predict(obs_dir, deterministic=True)
            self.smoothed_act_dir = 0.8 * self.smoothed_act_dir + 0.2 * float(act_r[0])
            delta_r = self.smoothed_act_dir * 25.0
        elif self.stage == 3:
            delta_e = raw_act * 25.0
            obs_dir, obs_lat = self._get_obs(1), self._get_obs(2)
            with torch.no_grad():
                act_r, _ = self.trained_models['dir'].predict(obs_dir, deterministic=True)
                act_a, _ = self.trained_models['lat'].predict(obs_lat, deterministic=True)
            self.smoothed_act_dir = 0.8 * self.smoothed_act_dir + 0.2 * float(act_r[0])
            delta_r = self.smoothed_act_dir * 25.0
            delta_a = float(act_a[0]) * 20.0

        if self.ftc_enabled:
            s = self.sim.state
            q = math.degrees(s[10]); p = math.degrees(s[9]); r = math.degrees(s[11])
            e_theta = self.target_theta - theta_pre
            e_phi   = self.target_phi   - phi_pre
            e_beta  = self.target_beta  - math.degrees(math.asin(np.clip(s[4]/max(math.sqrt(s[3]**2+s[4]**2+s[5]**2),1.0),-1,1)))
            delta_e, delta_a, delta_r = self.ftc.augment(delta_e, delta_a, delta_r,
                                                        e_theta, q, e_phi, p, e_beta, r, self.dt)

        if self.step_count * self.dt > self._fault_t:
            self.eff[self._fault_axis] = self._fault_scale
        ef_p, ef_r, ef_y = self.eff['pitch'], self.eff['roll'], self.eff['yaw']

        # ★ 一阶作动器低通：作用在"指令舵偏"上(舵机带宽)，再经舵效(气动)，符合物理因果
        if self.act_lpf_tau > 1e6:
            a_act = self.dt / (self.act_lpf_tau + self.dt)
            self.act_lpf['e'] += (delta_e - self.act_lpf['e']) * a_act
            self.act_lpf['a'] += (delta_a - self.act_lpf['a']) * a_act
            self.act_lpf['r'] += (delta_r - self.act_lpf['r']) * a_act
            delta_e, delta_a, delta_r = self.act_lpf['e'], self.act_lpf['a'], self.act_lpf['r']

        controls = {
            'd_flap_L': np.clip(delta_e * ef_p, -25, 25), 
            'd_flap_R': np.clip(delta_e * ef_p, -25, 25),
            'd_ail_L':  np.clip(delta_a * ef_r, -20, 20),
            'd_ail_R': np.clip(-delta_a * ef_r, -20, 20),
            'd_spoil_L': np.clip(delta_r * ef_y, -25, 25),
            'd_spoil_R': 0.0,
            'throttle': 0.65,
        }

        # ======= 推进物理 =======
        self.sim.step(self.dt, controls)
        self._update_history()
        self.step_count += 1

        # ======= 步后状态(奖励基于动作产生的结果，更符合 MDP) =======
        s = self.sim.state
        V = max(math.sqrt(s[3]**2 + s[4]**2 + s[5]**2), 1.0)
        alpha = math.degrees(math.atan2(s[5], s[3]))
        beta = math.degrees(math.asin(np.clip(s[4] / V, -1.0, 1.0)))
        theta, phi = math.degrees(s[7]), math.degrees(s[6])
        p, q, r = math.degrees(s[9]), math.degrees(s[10]), math.degrees(s[11])

        K = ENV_CFG['K_PROG']
        d_net = abs(raw_act - self.prev_actions['net_out'])
        # 软抗饱和栅栏：只在接近 ±1 时轻罚
        r_barrier = -0.3 * max(0.0, abs(raw_act) - 0.9) / 0.1
        reward = 0.0

        if self.stage == 1:
            err = self.hist_e_beta[0]
            r_track = math.exp(-(err / 5.0)**2) + 0.5 * math.exp(-(err / 0.3)**2)
            r_prog = K * float(np.clip(self.prev_abs_err['beta'] - abs(err), -1.0, 1.0))
            # ★ 转弯角速率减除后的偏航阻尼(只压残差，允许维持转弯固有偏航率)
            resid_r = r - self._coordinated_yaw_rate_deg()
            r_damp = -0.8 * (1.0 - math.exp(-(resid_r / 8.0)**2))
            r_smooth = -5.0 * d_net
            r_effort = -0.05 * raw_act**2   # 大幅减小(原 -1.0*raw^2)，允许稳定维持配平舵
            r_cross = -0.05 * max(0.0, abs(p) - 5.0)
            reward = (r_track + r_prog + r_damp + r_smooth + r_effort
                      + r_barrier + r_cross + ENV_CFG['SURVIVE'])
            self.prev_actions.update({'r': delta_r, 'a': delta_a, 'e': delta_e, 'net_out': raw_act})
            self.prev_abs_err['beta'] = abs(err)

        elif self.stage == 2:
            err = self.hist_e_phi[0]
            r_track = math.exp(-(err / 20.0)**2) + 0.5 * math.exp(-(err / 1.0)**2)
            r_prog = K * float(np.clip(self.prev_abs_err['phi'] - abs(err), -1.0, 1.0))
            r_smooth = -2.0 * d_net
            r_damp = -0.4 * (1.0 - math.exp(-(p / 40.0)**2))  # 放宽阻尼，允许大机动滚转率
            r_cross_beta = -0.1 * abs(self.hist_e_beta[0])
            r_cross_theta = -0.05 * abs(self.hist_e_theta[0])
            reward = (r_track + r_prog + r_smooth + r_damp + r_barrier
                      + r_cross_beta + r_cross_theta + ENV_CFG['SURVIVE'])
            self.prev_actions.update({'a': delta_a, 'r': delta_r, 'e': delta_e, 'net_out': raw_act})
            self.prev_abs_err['phi'] = abs(err)

        elif self.stage == 3:
            err = self.hist_e_theta[0]
            r_track = math.exp(-(err / 15.0)**2) + 0.5 * math.exp(-(err / 0.5)**2)
            r_prog = K * float(np.clip(self.prev_abs_err['theta'] - abs(err), -1.0, 1.0))
            r_smooth = -5.0 * d_net
            # 迎角安全红线：TEST1 显示 α>~8° 后 Cm 反号(俯仰上仰失稳)，红线压到 8°
            r_alpha = 0.0
            if alpha > 8.0:
                r_alpha = -1.0 * (alpha - 8.0)
            elif alpha < -3.0:
                r_alpha = -1.0 * (-3.0 - alpha)
            r_damp = -0.4 * (1.0 - math.exp(-(q / 20.0)**2))
            r_cross = -0.05 * (abs(p) + abs(r))
            reward = (r_track + r_prog + r_smooth + r_alpha + r_damp + r_barrier
                      + r_cross + ENV_CFG['SURVIVE'])
            self.prev_actions.update({'e': delta_e, 'r': delta_r, 'a': delta_a, 'net_out': raw_act})
            self.prev_abs_err['theta'] = abs(err)

        # ======= 终止判定(放宽边界 + 增加真·失速/能量耗尽护栏) =======
        truncated = self.step_count >= self.max_steps
        terminated = False
        if (abs(phi) > ENV_CFG['PHI_LIM'] or abs(theta) > ENV_CFG['THETA_LIM']
                or abs(beta) > ENV_CFG['BETA_LIM'] or alpha > ENV_CFG['ALPHA_STALL']
                or alpha < ENV_CFG['ALPHA_MIN'] or V < ENV_CFG['V_MIN']):
            reward -= ENV_CFG['TERM_PENALTY']
            terminated = True

        return self._get_obs(self.stage), float(reward), terminated, truncated, {}


def make_env(stage, model_paths=None, seed=42):
    def _init():
        os.environ["OMP_NUM_THREADS"] = "1"
        os.environ["MKL_NUM_THREADS"] = "1"
        torch.set_num_threads(1)
        aircraft_params = {'S': 88.58, 'b': 18.9, 'c_bar': 4.6, 'mass': 14000,
                           'Ixx': 313220.6, 'Iyy': 273435.3, 'Izz': 392903.9, 'Ixz': 0.0}
        aero_db, engine_db = NeuralAeroDatabase(), EngineDatabase()
        aero_db._load_from_pickle('X47B_coeffs.pkl')
        engine_db.load1("engine.pkl")
        sim = FlightSimulator6DOF(aero_db, engine_db, aircraft_params)
        env = X47BInnerEnv(sim, stage=stage, model_paths=model_paths,use_temporal=False)
        env.action_space.seed(seed)
        return env
    return _init


def linear_schedule(initial, final=0.0):
    def f(progress_remaining):  # SB3: 1.0 -> 0.0
        return final + (initial - final) * progress_remaining
    return f


def train_stage(stage, total_timesteps, n_envs, seed,
                model_paths, best_dir, save_name, results_dir):
    vec_env = VecMonitor(SubprocVecEnv(
        [make_env(stage=stage, model_paths=model_paths, seed=seed + i) for i in range(n_envs)]))
    eval_env = VecMonitor(DummyVecEnv(
        [make_env(stage=stage, model_paths=model_paths, seed=seed + 500)]))

    eval_cb = EvalCallback(eval_env, best_model_save_path=best_dir, log_path=results_dir,
                           eval_freq=max(40000 // n_envs, 1), deterministic=True, render=False)
    metrics_cb = TrainingMetricsCallback(tag=f'stage{stage}', verbose=1)
    curr_cb = CurriculumCallback(total_timesteps=total_timesteps, warmup_frac=0.6, verbose=1)
    callbacks = CallbackList([eval_cb, metrics_cb, curr_cb])
    # callbacks = CallbackList([eval_cb, metrics_cb])

    ppo_kwargs = dict(
        learning_rate=linear_schedule(3e-4, 5e-5),  # 线性退火，前期快收敛后期稳精修
        n_steps=1024, batch_size=256, n_epochs=10,
        gamma=0.99, gae_lambda=0.95, clip_range=0.2,
        ent_coef=0.005, verbose=1, seed=seed,
        target_kl=0.03,  # 防 KL/Value Loss 爆炸
    )
    policy_kwargs = dict(
        activation_fn=nn.Tanh,
        net_arch=dict(pi=[256, 256], vf=[256, 256]),
        log_std_init=-1.0,
    )
    model = PPO("MlpPolicy", vec_env, policy_kwargs=policy_kwargs, **ppo_kwargs)
    model.learn(total_timesteps=total_timesteps, callback=callbacks)
    model.save(save_name)
    vec_env.close(); eval_env.close()
    print(f"🎉 Stage{stage} 训练完成，保存 -> {save_name}.zip / best:{best_dir}")


if __name__ == '__main__':
    import multiprocessing
    SEED = 42
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.set_num_threads(1)
    n_envs = max(1, multiprocessing.cpu_count() - 2)

    print("=====================================================")
    print("  X-47B 飞翼内环 全包络重构训练 (课程学习+协调转弯感知)")
    print(f"  进程数: {n_envs}")
    print("=====================================================")

    # ★ 因为扩大了包络，三个 stage 都需要重新训练(不能复用旧小范围模型)
    # ---- Stage 1: 偏航/侧滑(差动扰流板) ----
    train_stage(stage=1, total_timesteps=1_200_000, n_envs=n_envs, seed=SEED,
                model_paths=None, best_dir='./logs/best_model_stage1_notemp/',
                save_name='ppo_dir_stage1_notemp', results_dir='./logs/results_stage1_notemp/')

    # ---- Stage 2: 横滚(副翼) ----
    train_stage(stage=2, total_timesteps=1_200_000, n_envs=n_envs, seed=SEED,
                model_paths={'dir': 'ppo_dir_stage1_notemp'}, best_dir='./logs/best_model_stage2_notemp/',
                save_name='ppo_lat_stage2_notemp', results_dir='./logs/results_stage2_notemp/')

    # # ---- Stage 3: 俯仰(升降副翼) ----
    train_stage(stage=3, total_timesteps=1_400_000, n_envs=n_envs, seed=SEED,
                model_paths={'dir': 'ppo_dir_stage1_notemp', 'lat': 'ppo_lat_stage2_notemp'},
                best_dir='./logs/best_model_stage3_notemp/',
                save_name='ppo_lon_stage3_notemp', results_dir='./logs/results_stage3_notemp/')

    # print("✅ 三阶段全包络内环训练全部完成。可直接运行 train_outer.py。")