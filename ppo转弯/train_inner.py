import os
import math
import numpy as np
from collections import deque
import torch
import torch.nn as nn
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv, VecMonitor
import gymnasium as gym
from gymnasium import spaces
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.vec_env import DummyVecEnv
# 新增绘图/数据处理库
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')
from stable_baselines3.common.callbacks import EvalCallback, BaseCallback, CallbackList  # 新增CallbackList和BaseCallback
from stable_baselines3.common.vec_env import DummyVecEnv
# 设置matplotlib中文字体（解决中文乱码）
plt.rcParams['font.sans-serif'] = ['SimHei']  # 黑体
plt.rcParams['axes.unicode_minus'] = False    # 负号正常显示

# 导入底层物理引擎
from ppo转弯.flyoo import AeroSurrogate, NeuralAeroDatabase, EngineDatabase, FlightSimulator6DOF

# =================================================================
# 1. 强鲁棒性 PD 兜底控制器 (专为高动压、无尾翼布局设计)
# 取消纯积分项，防止在高频气动耦合中产生积分饱和导致发散
# =================================================================
# =================================================================
# 1. 强鲁棒性 PID 兜底控制器 (带低通滤波与抗积分饱和)
# =================================================================
PID_CFG = {
    'pitch': {'kp': 2.0, 'ki': 0.5, 'kd': 1.5, 'limit': 25.0},  
    'roll':  {'kp': 4.5, 'ki': 0.2, 'kd': 2.5, 'limit': 20.0}
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

    def compute(self, error, dt):
        # 微弱的积分项，限幅非常严（±2.0），专门用来“啃”掉最后零点几度的稳态误差
        self.integral = np.clip(self.integral + error * dt, -2.0, 2.0)
        
        raw_derivative = (error - self.prev_error) / dt
        derivative = self.alpha * raw_derivative + (1 - self.alpha) * self.prev_derivative
        
        out = self.kp * error + self.ki * self.integral + self.kd * derivative
        
        self.prev_error = error
        self.prev_derivative = derivative
        return np.clip(out, -self.limit, self.limit)


class TrainingMetricsCallback(BaseCallback):
    def __init__(self, verbose=0):
        super().__init__(verbose)
        # 初始化指标存储列表
        self.episode_rewards = []    # 每个episode的奖励
        self.episode_lengths = []    # 每个episode的存活时长（步数）
        self.kl_divergences = []     # KL散度
        self.value_losses = []       # Value Loss
        self.action_stds = []        # 动作标准差
        self.timesteps = []          # 对应的累计时间步
        self.episode_count = 0       # 累计episode数量

    def _on_training_start(self) -> None:
        """训练开始时初始化指标列表"""
        self.episode_rewards = []
        self.episode_lengths = []
        self.kl_divergences = []
        self.value_losses = []
        self.action_stds = []
        self.timesteps = []
        self.episode_count = 0
        print("\n===== 开始收集训练指标 =====")

    def _on_step(self) -> bool:
        """必须实现的抽象方法，同时在这里收集多进程环境的episode信息"""
        # ====================== 核心修改1：从VecMonitor缓冲区获取episode信息 ======================
        # 这是多进程环境下唯一可靠的episode信息获取方式
        if hasattr(self.model, 'ep_info_buffer') and len(self.model.ep_info_buffer) > 0:
            # 遍历所有新完成的episode
            for ep_info in self.model.ep_info_buffer:
                self.episode_count += 1
                self.episode_rewards.append(ep_info['r'])
                self.episode_lengths.append(ep_info['l'])
                self.timesteps.append(self.num_timesteps)  # 使用SB3自带的准确总步数

                # 每10个Episode打印一次实时指标
                if self.episode_count % 10 == 0:
                    recent_rewards = self.episode_rewards[-10:]
                    recent_lengths = self.episode_lengths[-10:]
                    avg_r = np.mean(recent_rewards)
                    avg_len = np.mean(recent_lengths)
                    print(f"\n【Episode {self.episode_count} | 总步数: {self.num_timesteps:,}】")
                    print(f"平均奖励(近10): {avg_r:.2f} | 平均存活时长(近10): {avg_len:.2f}")
            
            # 清空缓冲区，避免重复处理
            self.model.ep_info_buffer.clear()
        
        return True

    def _on_rollout_end(self) -> None:
        """每次rollout结束后：收集训练指标"""
        # 从SB3 Logger中提取最新的训练指标
        logger = self.model.logger
        
        # 提取KL散度（SB3 v2.0+正确键名）
        kl = logger.name_to_value.get('train/approx_kl', None)
        if kl is not None:
            self.kl_divergences.append(kl)
        
        # 提取Value Loss
        v_loss = logger.name_to_value.get('train/value_loss', None)
        if v_loss is not None:
            self.value_losses.append(v_loss)
        
        # 计算动作标准差（策略网络log_std → exp(log_std)）
        log_std = self.model.policy.log_std
        if log_std is not None:
            action_std = torch.exp(log_std).detach().cpu().numpy()[0]
            self.action_stds.append(action_std)

        # 每次rollout结束打印一次训练指标
        if len(self.kl_divergences) > 0:
            last_kl = self.kl_divergences[-1]
            last_vloss = self.value_losses[-1] if self.value_losses else 0
            last_std = self.action_stds[-1] if self.action_stds else 0
            print(f"\n【Rollout {len(self.kl_divergences)} | 总步数: {self.num_timesteps:,}】")
            print(f"KL散度: {last_kl:.4f} | Value Loss: {last_vloss:.4f} | 动作标准差: {last_std:.4f}")

    def _on_training_end(self) -> None:
        """训练结束时：绘制并保存指标图表"""
        print("\n===== 训练结束，生成指标图表 =====")
        # 打印最终统计信息
        if len(self.episode_rewards) > 0:
            print(f"📊 总Episode数: {self.episode_count}")
            print(f"🏆 最高单Episode奖励: {max(self.episode_rewards):.2f}")
            print(f"📈 最终平均奖励(最后10个): {np.mean(self.episode_rewards[-10:]):.2f}")
            print(f"⏱️  最终平均存活时长(最后10个): {np.mean(self.episode_lengths[-10:]):.2f}")
        
        self.plot_metrics()

    def plot_metrics(self):
        """绘制所有指标的可视化图表（2行3列布局）"""
        # 创建图表容器
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle('X-47B Stage2 训练指标监控', fontsize=16, fontweight='bold')

        # ========== 子图1：平均奖励（滑动平均+原始值） ==========
        ax1 = axes[0, 0]
        if len(self.episode_rewards) > 0:
            reward_series = pd.Series(self.episode_rewards)
            avg_reward = reward_series.rolling(window=10, min_periods=1).mean()  # 10步滑动平均
            ax1.plot(self.timesteps, avg_reward, color='#1f77b4', linewidth=2, label='滑动平均奖励(窗口10)')
            ax1.plot(self.timesteps, self.episode_rewards, color='#1f77b4', alpha=0.3, label='单Episode奖励')
        ax1.set_title('平均奖励', fontsize=14)
        ax1.set_xlabel('累计时间步')
        ax1.set_ylabel('奖励值')
        ax1.legend()
        ax1.grid(alpha=0.3)

        # ========== 子图2：平均存活时长 ==========
        ax2 = axes[0, 1]
        if len(self.episode_lengths) > 0:
            len_series = pd.Series(self.episode_lengths)
            avg_len = len_series.rolling(window=10, min_periods=1).mean()
            ax2.plot(self.timesteps, avg_len, color='#ff7f0e', linewidth=2, label='滑动平均时长(窗口10)')
            ax2.plot(self.timesteps, self.episode_lengths, color='#ff7f0e', alpha=0.3, label='单Episode时长')
        ax2.set_title('平均存活时长', fontsize=14)
        ax2.set_xlabel('累计时间步')
        ax2.set_ylabel('步数')
        ax2.legend()
        ax2.grid(alpha=0.3)

        # ========== 子图3：KL散度（对比Target KL） ==========
        ax3 = axes[0, 2]
        if len(self.kl_divergences) > 0:
            # 生成对应的时间步（每个rollout对应n_envs*n_steps步）
            rollout_steps = [i * self.model.n_steps * self.model.n_envs for i in range(len(self.kl_divergences))]
            ax3.plot(rollout_steps, self.kl_divergences, color='#2ca02c', linewidth=2)
            if self.model.target_kl is not None:
                target_kl = self.model.target_kl if self.model.target_kl is not None else 0.05
                ax3.axhline(y=target_kl, color='red', linestyle='--', label=f'Target KL={target_kl}')
        ax3.set_title('KL散度', fontsize=14)
        ax3.set_xlabel('累计时间步')
        ax3.set_ylabel('KL值')
        ax3.legend()
        ax3.grid(alpha=0.3)

        # ========== 子图4：Value Loss（滑动平均） ==========
        ax4 = axes[1, 0]
        if len(self.value_losses) > 0:
            rollout_steps = [i * self.model.n_steps * self.model.n_envs for i in range(len(self.value_losses))]
            loss_series = pd.Series(self.value_losses)
            smooth_loss = loss_series.rolling(window=5, min_periods=1).mean()
            ax4.plot(rollout_steps, self.value_losses, color='#d62728', alpha=0.3, label='原始Value Loss')
            ax4.plot(rollout_steps, smooth_loss, color='#d62728', linewidth=2, label='滑动平均Loss(窗口5)')
        ax4.set_title('Value Loss', fontsize=14)
        ax4.set_xlabel('累计时间步')
        ax4.set_ylabel('损失值')
        ax4.legend()
        ax4.grid(alpha=0.3)

        # ========== 子图5：动作标准差 ==========
        ax5 = axes[1, 1]
        if len(self.action_stds) > 0:
            rollout_steps = [i * self.model.n_steps * self.model.n_envs for i in range(len(self.action_stds))]
            std_series = pd.Series(self.action_stds)
            smooth_std = std_series.rolling(window=5, min_periods=1).mean()
            ax5.plot(rollout_steps, self.action_stds, color='#9467bd', alpha=0.3, label='原始动作标准差')
            ax5.plot(rollout_steps, smooth_std, color='#9467bd', linewidth=2, label='滑动平均标准差(窗口5)')
        ax5.set_title('动作标准差', fontsize=14)
        ax5.set_xlabel('累计时间步')
        ax5.set_ylabel('标准差')
        ax5.legend()
        ax5.grid(alpha=0.3)

        # 隐藏最后一个备用子图
        axes[1, 2].axis('off')

        # 保存图表（自动创建目录）
        save_dir = './logs/metrics_plots/'
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, 'training_metrics_stage2.png'), dpi=300, bbox_inches='tight')
        plt.show()

        # ====================== 核心修改2：修正CSV保存逻辑 ======================
        # 保存原始指标数据为CSV（确保所有列长度一致）
        max_len = max(
            len(self.timesteps), 
            len(self.kl_divergences), 
            len(self.value_losses), 
            len(self.action_stds)
        )
        
        # 对齐不同长度的数组
        def pad_array(arr, target_len):
            return arr + [None] * (target_len - len(arr))
        
        metrics_data = {
            'timestep': pad_array(self.timesteps, max_len),
            'episode_reward': pad_array(self.episode_rewards, max_len),
            'episode_length': pad_array(self.episode_lengths, max_len),
            'kl_divergence': pad_array(self.kl_divergences, max_len),
            'value_loss': pad_array(self.value_losses, max_len),
            'action_std': pad_array(self.action_stds, max_len)
        }
        
        pd.DataFrame(metrics_data).to_csv(os.path.join(save_dir, 'training_metrics_stage2.csv'), index=False)
        print(f"\n✅ 指标图表已保存至: {save_dir}training_metrics_stage2.png")
        print(f"✅ 指标数据已保存至: {save_dir}training_metrics_stage2.csv")

# =================================================================
# 2. 内环强化学习核心环境 (解耦与交叉惩罚)
# =================================================================
class X47BInnerEnv(gym.Env):
    def __init__(self, simulator, stage=1, model_paths=None):
        super(X47BInnerEnv, self).__init__()
        self.sim = simulator
        self.stage = stage
        self.dt = 0.01  
        self.max_steps = 2000 
        self.alpha_lpf = 0.1  
        self.smoothed_act_dir = 0.0
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
        dim = 16 if self.stage == 3 else 15
        self.observation_space = spaces.Box(-np.inf, np.inf, shape=(dim,), dtype=np.float32)
            
        self.hist_e_theta = deque([0.0]*5, maxlen=5); self.hist_q = deque([0.0]*5, maxlen=5)
        self.hist_e_phi = deque([0.0]*5, maxlen=5);   self.hist_p = deque([0.0]*5, maxlen=5)
        self.hist_e_beta = deque([0.0]*5, maxlen=5);  self.hist_r = deque([0.0]*5, maxlen=5)
        
        self.filtered_p_dot, self.filtered_q_dot, self.filtered_r_dot = 0.0, 0.0, 0.0
        self.target_theta, self.target_phi, self.target_beta = 0.0, 0.0, 0.0
        
        # 缓存网络原始输出以计算速率惩罚
        self.prev_actions = {'e': 0.0, 'a': 0.0, 'r': 0.0, 'net_out': 0.0}
        self.step_count = 0

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
        self.pid_pitch.reset()
        self.pid_roll.reset()
        
        self.smoothed_act_dir = 0.0 
        
        self.sim.state = np.zeros(12) 
        self.filtered_p_dot, self.filtered_q_dot, self.filtered_r_dot = 0.0, 0.0, 0.0
        self.step_count = 0
        self.prev_actions = {'e': 0.0, 'a': 0.0, 'r': 0.0, 'net_out': 0.0}
        
        h_init = self.np_random.uniform(2800.0, 3200.0) 
        V_init = self.np_random.uniform(190.0, 210.0)
        
        # 👑 全包络初始姿态随机化
        initial_theta = 2.0 + self.np_random.uniform(-1.5, 1.5) 
        initial_phi = self.np_random.uniform(-15.0, 15.0) 
            
        self.sim.set_initial_state(h_init, V_init, theta_deg=initial_theta)
        self.sim.state[6] = math.radians(initial_phi) 
        
        self.target_beta = 0.0  
        if self.stage == 1:
            self.target_phi = self.np_random.uniform(-10.0, 10.0) 
            self.target_theta = 2.0
        elif self.stage == 2:
            self.target_phi = self.np_random.uniform(-20.0, 20.0)
            self.target_theta = 2.0
        elif self.stage == 3:
            self.target_theta = self.np_random.uniform(-2.0, 8.0) 
            self.target_phi = self.np_random.uniform(-15.0, 15.0) if self.np_random.random() < 0.5 else 0.0
            
        # 👑 核心修复：在 5 步预热期内激活 PID，让控制序列平滑过渡！
        for _ in range(5):
            state = self.sim.state
            theta, phi = math.degrees(state[7]), math.degrees(state[6])
            
            # 让 PID 提前工作，建立平滑的误差导数
            delta_e = -self.pid_pitch.compute(self.target_theta - theta, self.dt)
            delta_a = 0.0
            if self.stage == 1:
                delta_a = self.pid_roll.compute(self.target_phi - phi, self.dt)
                
            controls = {
                'd_flap_L': np.clip(delta_e, -25.0, 25.0),
                'd_flap_R': np.clip(delta_e, -25.0, 25.0),
                'd_ail_L':  np.clip(delta_a, -20.0, 20.0),
                'd_ail_R':  np.clip(-delta_a, -20.0, 20.0),
                'd_spoil_L': 0.0,
                'd_spoil_R': 0.0,
                'throttle': 0.65
            }
            self.sim.step(self.dt, controls)
            self._update_history()
            
        return self._get_obs(self.stage), {}

    def _get_obs(self, stage_req):
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
            
        if stage_req == 1:
            base = _get_hist(self.hist_e_beta, self.hist_r)
            return np.array([base[0], base[5], self.prev_actions['r'], r_dot] + base[1:5] + base[6:10] + [phi, p, self.prev_actions['a']], dtype=np.float32)
        elif stage_req == 2:
            base = _get_hist(self.hist_e_phi, self.hist_p)
            return np.array([base[0], base[5], self.prev_actions['a'], p_dot] + base[1:5] + base[6:10] + [beta, r, self.prev_actions['r']], dtype=np.float32)
        elif stage_req == 3:
            base = _get_hist(self.hist_e_theta, self.hist_q)
            return np.array([base[0], base[5], alpha, self.prev_actions['e'], q_dot] + base[1:5] + base[6:10] + [phi, p, r], dtype=np.float32)

    def _log_cosh_reward(self, error, scale=1.0):
        """
        👑 核心：连续平滑无突变损失函数。
        小误差时近似二次函数，大误差时近似绝对值函数，彻底避免梯度爆炸。
        """
        # 引入缩放因子，让函数对特定的误差范围敏感
        x = error * scale 
        # 为了稳定，将返回一个经过平移和翻转的正向/负向平滑值
        loss = math.log(math.cosh(x))
        return 1.0 - loss  # 最大奖励为 1，误差越大迅速下降为负数

    def step(self, action):
        state = self.sim.state
        theta, phi = math.degrees(state[7]), math.degrees(state[6])
        q, p, r = math.degrees(state[10]), math.degrees(state[9]), math.degrees(state[11])
        
        # 👑 1. 动作低通滤波 (Action Smoothing)
        # 强制将当前输出与上一步平滑融合 (80% 继承惯性，20% 听从新指令)，极大抑制高频震荡！
        raw_act = float(np.clip(action[0], -1.0, 1.0))

        delta_e, delta_a, delta_r, reward = 0.0, 0.0, 0.0, 0.0

        # ======= 渐进式三轴独立接管 =======
        if self.stage == 1:
            self.smoothed_act_dir = 0.8 * self.smoothed_act_dir + 0.2 * raw_act
            delta_r = self.smoothed_act_dir * 25.0 
            
            delta_e = -self.pid_pitch.compute(self.target_theta - theta, self.dt)
            delta_a = self.pid_roll.compute(self.target_phi - phi, self.dt)
            
            err_beta = self.hist_e_beta[0]
            # 1. 核心跟踪：收紧高斯尖峰的“容忍度”
            # 把尖峰宽度从 0.5 缩小到 0.1！侧滑必须小于 0.1° 才能拿满这 0.5 分
            r_track = math.exp(- (err_beta / 5.0)**2) + 0.5 * math.exp(- (err_beta / 0.1)**2)
            
            # 2. 动作平滑：防高频毛刺
            delta_raw_act = abs(raw_act - self.prev_actions['net_out'])
            r_smooth = -5.0 * delta_raw_act 
            
            # ==========================================================
            # 👑 3. 核心修复：控制能量惩罚 (打破方波陷阱的最强武器！)
            # 采用平方惩罚：打满舵(1.0)扣 1.0 分；打 10% 微调(0.1)只扣 0.01 分！
            # 这种非线性的压制会逼迫 AI 在稳态时把输出乖乖降到 0 附近。
            # ==========================================================
            r_effort = -1.0 * (raw_act ** 2)
            
            # 4. 偏航阻尼：加大惩罚力度，严禁机头左右乱甩
            r_damp = -1.0 * (1.0 - math.exp(-(r / 5.0)**2))
            
            r_cross = 0.0
            if abs(p) > 5.0: 
                r_cross = -0.05 * (abs(p) - 5.0) 
            
            # 奖励汇总
            reward = r_track + r_smooth + r_effort + r_damp + r_cross
            
            self.prev_actions.update({'r': delta_r, 'a': delta_a, 'e': delta_e, 'net_out': raw_act})     
        elif self.stage == 2:
            # Stage 2 训练时：副翼(当前网络)使用绝对位置，依靠强惩罚来平滑
            delta_a = raw_act * 20.0
            delta_e = -self.pid_pitch.compute(self.target_theta - theta, self.dt)
            
            # 调取 Stage 1 模型推理
            obs_dir = self._get_obs(1)
            with torch.no_grad():
                act_r, _ = self.trained_models['dir'].predict(obs_dir, deterministic=True)
                
            # 核心修复：为 Stage 1 的推理输出重新套上它赖以生存的低通滤波器！
            self.smoothed_act_dir = 0.8 * self.smoothed_act_dir + 0.2 * float(act_r[0])
            delta_r = self.smoothed_act_dir * 25.0
            
            # ==========================================================
            # 👑 适配域随机化的高级奖励函数
            # ==========================================================
            err_phi = self.hist_e_phi[0]
            
            # 1. 核心跟踪：扩大基础高斯的感受野！
            # 将分母从 10 扩大到 20，防止在 35° 极端初始误差下发生梯度消失
            r_track = math.exp(- (err_phi / 20.0)**2) + 0.5 * math.exp(- (err_phi / 1.0)**2)
                
            # 2. 动作平滑：重罚当前网络(副翼)的锯齿输出
            delta_raw_act = abs(raw_act - self.prev_actions['net_out'])
            r_smooth = -2.0 * delta_raw_act
            
            # 3. 滚转阻尼：限制超速滚转，给后台的 PID 和 Stage 1 留出反应时间
            r_damp = -0.5 * (1.0 - math.exp(-(p / 30.0)**2))
            
            # 4. 交叉惩罚 A (侧滑)：相信 Stage 1，但也不能让副翼动作太嚣张
            r_cross_beta = -0.1 * abs(self.hist_e_beta[0])
            
            # 👑 5. 交叉惩罚 B (俯仰)：核心新增！
            # 逼迫副翼网络具备“全局大局观”，在机头掉落严重时主动减缓滚转动作
            r_cross_theta = -0.05 * abs(self.hist_e_theta[0])
            
            reward = r_track + r_smooth + r_damp + r_cross_beta + r_cross_theta
            
            self.prev_actions.update({'a': delta_a, 'r': delta_r, 'e': delta_e, 'net_out': raw_act})
        elif self.stage == 3:
            raw_act = float(np.clip(action[0], -1.0, 1.0))
            delta_e = raw_act * 25.0 
            
            # 后台双脑护航
            obs_dir, obs_lat = self._get_obs(1), self._get_obs(2)
            with torch.no_grad():
                act_r, _ = self.trained_models['dir'].predict(obs_dir, deterministic=True)
                act_a, _ = self.trained_models['lat'].predict(obs_lat, deterministic=True)
            
            # 严格遵守物理接口
            self.smoothed_act_dir = 0.8 * self.smoothed_act_dir + 0.2 * float(act_r[0])
            delta_r = self.smoothed_act_dir * 25.0
            delta_a = float(act_a[0]) * 20.0
            
            # 1. 核心追踪
            err_theta = self.hist_e_theta[0]
            r_track = math.exp(- (err_theta / 15.0)**2) + 0.5 * math.exp(- (err_theta / 0.5)**2)
            
            # 2. 防抖动惩罚
            delta_raw_act = abs(raw_act - self.prev_actions['net_out'])
            r_smooth = -5.0 * delta_raw_act  
            # 🚨 注意：去掉了 r_effort，允许网络维持恒定力量进行配平！
            
            # 3. 迎角安全红线 (核心生命线)
            u_b, v_b, w_b = state[3], state[4], state[5]
            alpha_deg = math.degrees(math.atan2(w_b, u_b))
            r_alpha = 0.0
            if alpha_deg > 12.0:
                r_alpha = -1.0 * (alpha_deg - 12.0)
            elif alpha_deg < -3.0:
                r_alpha = -1.0 * (-3.0 - alpha_deg)
                
            # 4. 俯仰阻尼
            r_damp = -0.5 * (1.0 - math.exp(-(q / 15.0)**2))
            
            # 5. 跨轴干扰惩罚 (极其关键，不许破坏 Stage 1/2 的心血)
            r_cross = -0.05 * (abs(p) + abs(r))
            
            reward = r_track + r_smooth + r_alpha + r_damp + r_cross
            
            self.prev_actions.update({'e': delta_e, 'r': delta_r, 'a': delta_a, 'net_out': raw_act})
        # 端到端物理映射
        controls = {
            'd_flap_L': np.clip(delta_e, -25.0, 25.0),
            'd_flap_R': np.clip(delta_e, -25.0, 25.0),
            'd_ail_L':  np.clip(delta_a, -20.0, 20.0),
            'd_ail_R':  np.clip(-delta_a, -20.0, 20.0),
            'd_spoil_L': np.clip(max(0, delta_r), 0.0, 25.0), 
            'd_spoil_R': np.clip(max(0, -delta_r), 0.0, 25.0),
            'throttle': 0.65  
        }
        
        self.sim.step(self.dt, controls)
        self._update_history() 
        self.step_count += 1
        
        truncated = self.step_count >= self.max_steps
        terminated = False
        
        # 更加严苛的软截断（提前结束失败的探索，节省计算资源）
        if abs(phi) > 60 or abs(theta) > 30 or abs(self.hist_e_beta[0]) > 15:
            reward -= 5.0  # 象征性扣 5 分即可，终止回合
            terminated = True
            
        return self._get_obs(self.stage), float(reward), terminated, truncated, {}

def make_env(stage, model_paths=None, seed=42):
    def _init():
        aircraft_params = {'S': 88.58, 'b': 18.9, 'c_bar': 4.6, 'mass': 14000,
                           'Ixx': 313220.6, 'Iyy': 273435.3, 'Izz': 392903.9, 'Ixz': 0.0}
        aero_db, engine_db = NeuralAeroDatabase(), EngineDatabase()
        aero_db._load_from_pickle('aero_surrogate.pth')
        engine_db.load1("engine.pkl")
        sim = FlightSimulator6DOF(aero_db, engine_db, aircraft_params)
        env = X47BInnerEnv(sim, stage=stage, model_paths=model_paths)
        return env
    return _init

if __name__ == '__main__':
    import multiprocessing
    SEED = 42
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    
    n_envs = max(1, multiprocessing.cpu_count() - 2)
    
    ppo_kwargs = dict(
        learning_rate=1e-4,   
        n_steps=512,         
        batch_size=128,       
        n_epochs=10, # 稍微增加 epoch 以榨取经验       
        gamma=0.99,          
        gae_lambda=0.95,      
        clip_range=0.2, 
        ent_coef=0.005,       
        verbose=1, 
        seed=SEED,
        # target_kl=0.05
    )
    
    def get_policy_kwargs(arch):
        return dict(
            activation_fn=nn.Tanh, # Tanh 对连续控制的平滑性通常比 ReLU 更好
            net_arch=dict(pi=arch, vf=arch),
            log_std_init=-1.0 
        )

    print(f"=====================================================")
    print(f"  启动 X-47B 飞翼布局专用 DRL 训练脚本 (交叉惩罚机制)")
    print(f"=====================================================")

    # vec_env1 = VecMonitor(SubprocVecEnv([make_env(stage=1, seed=SEED+i) for i in range(n_envs)]))
    # eval_env1 = VecMonitor(DummyVecEnv([make_env(stage=1, seed=SEED+100)]))
    
    # eval_callback1 = EvalCallback(
    #     eval_env1,
    #     best_model_save_path='./logs/best_model_stage1/', 
    #     log_path='./logs/results_stage1/',
    #     eval_freq=max(40000 // n_envs, 1), 
    #     deterministic=True, 
    #     render=False
    # )
    # model1 = PPO("MlpPolicy", vec_env1, policy_kwargs=get_policy_kwargs([1024, 1024]), **ppo_kwargs)
    # model1.learn(total_timesteps=800_000,callback=eval_callback1)
    # model1.save("ppo_dir_stage1")
    # vec_env1.close()
    # eval_env1.close()



    # vec_env2 = VecMonitor(SubprocVecEnv([make_env(stage=2, model_paths={'dir': 'ppo_dir_stage1'}, seed=SEED+i) for i in range(n_envs)]))
    # eval_env2 = VecMonitor(DummyVecEnv([make_env(stage=2, model_paths={'dir': 'ppo_dir_stage1'}, seed=SEED+100)]))
    # # 2. 定义评估回调函数 (EvalCallback)
    # eval_callback2 = EvalCallback(
    #     eval_env2,
    #     best_model_save_path='./logs/best_model_stage2/', # 历史最高分模型将被存在这里
    #     log_path='./logs/results_stage2/',
    #     eval_freq=max(40000 // n_envs, 1), # 大约每 2000 步进行一次全方位闭环测试
    #     deterministic=True, # 测试时关闭探索噪声，展现绝对实力
    #     render=False
    # )
    # # 2. 新增指标回调
    # metrics_callback = TrainingMetricsCallback(verbose=1)
    
    # # 3. 组合回调（同时生效）
    # callback_list = CallbackList([eval_callback2, metrics_callback])

    # model2 = PPO("MlpPolicy", vec_env2, policy_kwargs=get_policy_kwargs([1024, 1024]), **ppo_kwargs)
    # model2.learn(total_timesteps=900_000, callback=callback_list)
    # model2.save("ppo_lat_stage2")
    # vec_env2.close()
    # eval_env2.close()

    paths_s3 = {'dir': 'ppo_dir_stage1', 'lat': 'ppo_lat_stage2'}
    vec_env3 = VecMonitor(SubprocVecEnv([make_env(stage=3, model_paths=paths_s3, seed=SEED+i) for i in range(n_envs)]))
    
    # 1. 创建stage3独立评估环境（与训练环境隔离，使用不同seed）
    eval_env3 = VecMonitor(DummyVecEnv([make_env(stage=3, model_paths=paths_s3, seed=SEED+200)]))
    
    # 2. 定义stage3评估回调
    eval_callback3 = EvalCallback(
        eval_env3,
        best_model_save_path='./logs/best_model_stage3/',
        log_path='./logs/results_stage3/',
        eval_freq=max(40000 // n_envs, 1),
        deterministic=True,
        render=False
    )
    
    # 3. 定义stage3指标回调
    metrics_callback3 = TrainingMetricsCallback(verbose=1)
    
    # 4. 组合回调
    callback_list3 = CallbackList([eval_callback3, metrics_callback3])
    
    model3 = PPO("MlpPolicy", vec_env3, policy_kwargs=get_policy_kwargs([1024, 1024]), **ppo_kwargs)
    model3.learn(total_timesteps=1000_000, callback=callback_list3)
    model3.save("ppo_lon_stage3")
    
    # 5. 关闭所有环境释放资源
    vec_env3.close()
    eval_env3.close()