#coding=utf-8
import numpy as np
import math
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import torch
import torch.nn as nn
import torch.optim as optim
import gymnasium as gym
from gymnasium import spaces
from stable_baselines3 import PPO
import warnings

# 导入神经引擎与模拟器
from fly import NeuralAeroDatabase, EngineDatabase, FlightSimulator6DOF

warnings.filterwarnings('ignore')

# =================================================================
# 1. RLS 在线增量模型辨识器 (包含3大工业级修正)
# =================================================================
class RLS_Identifier:
    def __init__(self, state_dim=2, input_dim=1):
        self.state_dim = state_dim
        self.input_dim = input_dim
        self.gamma_rls = 0.95 
        self.P = np.eye(state_dim + input_dim) * 1e6 
        self.Theta = np.zeros((state_dim + input_dim, state_dim))
        self.Theta[:state_dim, :state_dim] = np.eye(state_dim) 
        
        # 【修正1】物理先验：正偏角导致低头
        self.Theta[2, 1] = -0.1

    def update(self, delta_x_prev, delta_u_prev, delta_x_curr):
        Xi = np.zeros((self.state_dim + self.input_dim, 1))
        Xi[:self.state_dim, 0] = delta_x_prev
        Xi[self.state_dim:, 0] = delta_u_prev
        
        # 【修正2】死区：动作太小不更新，防止拟合噪声
        if np.linalg.norm(Xi) < 1e-4:
            return self.Theta[:self.state_dim, :].T, self.Theta[self.state_dim:, :].T
            
        delta_x_pred = self.Theta.T @ Xi
        error = delta_x_curr.reshape(-1, 1) - delta_x_pred
        
        K_num = self.P @ Xi
        K_den = self.gamma_rls + Xi.T @ self.P @ Xi
        K = K_num / K_den[0, 0]
        
        self.Theta = self.Theta + K @ error.T
        self.P = (1.0 / self.gamma_rls) * (self.P - K @ Xi.T @ self.P)
        
        # 【修正3】钳制梯度：绝不允许 Theta[2,1] 反向
        if self.Theta[2, 1] > -0.001:
            self.Theta[2, 1] = -0.001 
            
        return self.Theta[:self.state_dim, :].T, self.Theta[self.state_dim:, :].T

# =================================================================
# 2. IDHP 容错内环智能体
# =================================================================
class IDHP_Agent:
    def __init__(self):
        self.critic = nn.Sequential(nn.Linear(3, 32), nn.Mish(), nn.Linear(32, 2))
        self.actor = nn.Sequential(nn.Linear(3, 32), nn.Mish(), nn.Linear(32, 1), nn.Tanh())
        self.opt_c = optim.Adam(self.critic.parameters(), lr=0.01)
        self.opt_a = optim.Adam(self.actor.parameters(), lr=0.005)
        self.gamma = 0.8 
        self.Q_mat = torch.tensor([[90.0, 0.0], [0.0, 5.0]], dtype=torch.float32)

    def get_assistance_action(self, e_theta, e_q, target_theta):
        x_in = torch.tensor([e_theta, e_q, target_theta], dtype=torch.float32)
        with torch.no_grad(): u_d_norm = self.actor(x_in).item()
        return u_d_norm * 3.0 # 限制 IDHP 最大辅助能力为 3度

    def online_train_step(self, e_prev, e_curr, target_theta, F_hat, G_hat):
        e_p = torch.tensor(e_prev, dtype=torch.float32)
        e_c = torch.tensor(e_curr, dtype=torch.float32)
        target = torch.tensor([target_theta], dtype=torch.float32)
        
        in_p = torch.cat([e_p, target])
        in_c = torch.cat([e_c, target])
        G_e = torch.tensor(-G_hat, dtype=torch.float32)
        
        self.opt_c.zero_grad()
        lambda_prev = self.critic(in_p)
        with torch.no_grad():
            lambda_curr = self.critic(in_c)
            dc_de = 2.0 * torch.matmul(self.Q_mat, e_p)
            F_e = torch.tensor(F_hat, dtype=torch.float32)
            target_lambda = dc_de + self.gamma * torch.matmul(F_e.T, lambda_curr)
        loss_c = nn.MSELoss()(lambda_prev, target_lambda)
        loss_c.backward()
        self.opt_c.step()
        
        self.opt_a.zero_grad()
        u_d_norm = self.actor(in_p)
        with torch.no_grad(): lambda_curr_new = self.critic(in_c)
        actor_grad_direction = torch.matmul(G_e.T, lambda_curr_new)
        loss_a = torch.sum(u_d_norm * actor_grad_direction)
        loss_a.backward()
        self.opt_a.step()

# =================================================================
# 3. 强化学习微调环境 (封装 PPO+IDHP 联合运行)
# =================================================================
class CoLearningEnv(gym.Env):
    def __init__(self, target_alt=2500.0):
        super(CoLearningEnv, self).__init__()
        self.target_alt = target_alt
        
        aircraft_params = {'S': 88.58, 'b': 18.9, 'c_bar': 4.6, 'mass': 14000,
                           'Ixx': 313220.6, 'Iyy': 273435.3, 'Izz': 392903.9, 'Ixz': 0.0}
        self.flight_db = NeuralAeroDatabase()
        self.flight_db._load_from_pickle('aero_surrogate.pth')
        self.engine_db = EngineDatabase()
        self.engine_db.load1('engine.pkl')

        self.sim = FlightSimulator6DOF(self.flight_db, self.engine_db, aircraft_params)
        
        # 6 维观测空间
        high = np.array([5.0, 2.0, 5.0, 2.0, 2.0, 2.0], dtype=np.float32)
        self.observation_space = spaces.Box(-high, high, dtype=np.float32)
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(1,), dtype=np.float32)
        
        self.dt = 0.02
        self.action_repeat = 10
        self.max_steps = 600
        
        # ★ 故障参数
        self.fault_start = 40.0
        self.fault_end = 80.0

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.current_step = 0
        self.sim_time = 0.0
        
        # 内环状态缓存重置
        self.rls = RLS_Identifier(state_dim=2, input_dim=1)
        self.idhp = IDHP_Agent()
        self.integral_theta = 0.0
        self.last_x = np.array([math.radians(0.7), 0.0])
        self.last_u, self.last_delta_u = 0.0, 0.0
        self.last_delta_x, self.last_e = np.zeros(2), np.zeros(2)
        self.last_action = 0.0
        
        # 带有小扰动的初始状态，防止灾难性遗忘
        h_noise = np.random.uniform(-10.0, 10.0)
        v_noise = np.random.uniform(-2.0, 2.0)
        self.sim.set_initial_state(h_m=2000.0 + h_noise, V_mps=250.0 + v_noise, theta_deg=0.7, alpha_deg=0.7)
        
        return self._get_obs(), {}

    def _get_obs(self):
        u, v, w = self.sim.state[3], self.sim.state[4], self.sim.state[5]
        phi, theta = self.sim.state[6], self.sim.state[7]
        current_h = -self.sim.state[2]
        V = max(math.sqrt(u**2 + v**2 + w**2), 1.0)
        vz = u*math.sin(theta) - v*math.sin(phi)*math.cos(theta) - w*math.cos(phi)*math.cos(theta)
        
        return np.array([
            (self.target_alt - current_h) / 1000.0,
            (V - 250.0) / 50.0,
            vz / 10.0,
            math.degrees(theta) / 10.0,
            math.degrees(self.sim.state[10]) / 10.0,
            math.degrees(math.atan2(w, u)) / 10.0
        ], dtype=np.float32)

    def step(self, action):
        self.current_step += 1
        target_pitch = ((action[0] + 1.0) / 2.0) * 7.0 - 2.0 

        # ==============================================================
        # 高频内环步进 (融合 PID 和 IDHP)
        # ==============================================================
        for _ in range(self.action_repeat):
            current_pitch = math.degrees(self.sim.state[7])
            current_q = math.degrees(self.sim.state[10])
            current_p = math.degrees(self.sim.state[9])
            
            err_theta = target_pitch - current_pitch
            err_q = 0.0 - current_q
            current_e = np.array([err_theta, err_q])
            current_x = np.array([current_pitch, current_q])
            
            # [A] IDHP 在线学习
            if self.sim_time > 0.05:
                delta_x_curr = current_x - self.last_x
                F_hat, G_hat = self.rls.update(self.last_delta_x, self.last_delta_u, delta_x_curr)
                self.idhp.online_train_step(self.last_e, current_e, target_pitch, F_hat, G_hat)
                self.last_delta_x = delta_x_curr
            
            # [B] 基础强力 PID
            self.integral_theta = np.clip(self.integral_theta + err_theta * self.dt, -10.0, 10.0)
            u_0 = -2.5 * err_theta + 0.8 * current_q - 0.5 * self.integral_theta
            
            # [C] 提取 IDHP 辅助与 PE 激励
            u_d = self.idhp.get_assistance_action(err_theta, err_q, target_pitch)
            u_PE = 0.05 * math.sin(2 * math.pi * 0.5 * self.sim_time) + 0.02 * math.sin(2 * math.pi * 1.5 * self.sim_time)
            
            d_flap = np.clip(u_0 + u_d + u_PE, -15.0, 15.0) 
            
            # ★ 【核心修改：物理故障注入】★
            actual_d_flap = d_flap
            if self.fault_start <= self.sim_time <= self.fault_end:
                # 物理执行机构坏了，只响应控制器指令的 50%
                actual_d_flap = d_flap * 0.5
            
            # 更新缓存：注意，飞控程序自己并不知道机构坏了，它发出的指令依然是 d_flap
            # 辨识器也是用"自以为发出的指令"和"飞机实际产生的运动"作对比，这样才能发现故障
            self.last_delta_u = d_flap - self.last_u
            self.last_u = d_flap
            self.last_x = current_x.copy()
            self.last_e = current_e.copy()
            
            d_ail = np.clip(1.0 * (0.0 - math.degrees(self.sim.state[6])) - 0.5 * current_p, -10.0, 10.0)
            
            # 传入引擎的是损坏后的 actual_d_flap
            self.sim.step(self.dt, {'d_flap_L': actual_d_flap, 'd_flap_R': actual_d_flap, 'd_ail_L': d_ail, 'd_ail_R': -d_ail})
            self.sim_time += self.dt

        # ==============================================================
        # 奖励结算
        # ==============================================================
        obs = self._get_obs()
        err_h = self.target_alt - (-self.sim.state[2])
        vz = obs[2] * 10.0
        
        base_reward = 2.0 - (abs(err_h) / 1000.0)
        base_reward += math.exp(-((err_h / 5.0)**2)) * 2.0
        base_reward += math.exp(-((err_h / 1.0)**2)) * 3.0 
        
        action_diff = abs(action[0] - self.last_action)
        self.last_action = action[0]
        
        damping_weight = math.exp(-((err_h / 10.0)**2)) * 0.1
        penalty = action_diff * 0.02 + abs(vz) * damping_weight
        
        reward = base_reward - penalty
        terminated = -self.sim.state[2] < 500.0 or -self.sim.state[2] > 5000.0
        truncated = self.current_step >= self.max_steps
            
        return obs, float(reward), terminated, truncated, {}


# =================================================================
# 4. 主程序：在线协同微调与最终验收试飞
# =================================================================
if __name__ == "__main__":
    
    print("=====================================================")
    print(" 🚀 启动 PPO(外环) + IDHP(内环) 容错试飞测试")
    print("=====================================================")
    
    env = CoLearningEnv(target_alt=2500.0)
    
    try:
        model = PPO.load("rl_models/best_model.zip", env=env)
        print("✅ 成功加载 PPO 模型。")
    except Exception as e:
        print(f"❌ 无法加载 PPO 模型: {e}")
        exit()

    # ============================================================
    # 5. 验收容错试飞
    # ============================================================
    print(f"\n🎬 开始最终验收试飞 (故障区间设为 {env.fault_start}s - {env.fault_end}s，舵效减半)...")
    obs, _ = env.reset()
    
    history_time, history_alt, history_pitch, history_cmd = [], [], [], []
    history_u0, history_ud = [], []
    
    for step in range(env.max_steps):
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action)
        
        h = -env.sim.state[2]
        pitch = math.degrees(env.sim.state[7])
        target_pitch = ((action[0] + 1.0) / 2.0) * 7.0 - 2.0
        
        # 记录内部控制量用于画图 (从环境对象中提取)
        u_0 = -2.5 * (target_pitch - pitch) + 0.8 * math.degrees(env.sim.state[10]) - 0.5 * env.integral_theta
        u_d = env.idhp.get_assistance_action(target_pitch - pitch, 0.0 - math.degrees(env.sim.state[10]), target_pitch)
        
        history_time.append(step * env.action_repeat * env.dt)
        history_alt.append(h)
        history_pitch.append(pitch)
        history_cmd.append(target_pitch)
        history_u0.append(u_0)
        history_ud.append(u_d)
        
        if step % 50 == 0:
            print(f"Time: {history_time[-1]:4.1f}s | 高度: {h:6.1f}m | 指令: {target_pitch:4.1f}° | IDHP辅助: {u_d:5.2f}°")
            
        if terminated or truncated: break

    # ============================================================
    # 6. 绘制容错试飞报告 (包含故障区间高亮)
    # ============================================================
    plt.style.use('dark_background') 
    plt.rcParams['font.sans-serif'] = ['SimHei'] 
    plt.rcParams['axes.unicode_minus'] = False   
    fig = plt.figure(figsize=(15, 8))
    fig.suptitle('PPO外环 + IDHP内环：控制面故障在线容错测试', fontsize=18, color='cyan', fontweight='bold')

    gs = GridSpec(2, 2, figure=fig)
    
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.plot(history_time, history_alt, color='springgreen', linewidth=2.5)
    ax1.axhline(2500.0, color='white', linestyle='--', alpha=0.5)
    # 绘制故障高亮区域
    ax1.axvspan(env.fault_start, env.fault_end, color='red', alpha=0.25, label='舵效减半故障区间')
    ax1.set_title('高度轨迹 (Altitude)'); ax1.legend(loc='lower left'); ax1.grid(True, alpha=0.2)

    ax2 = fig.add_subplot(gs[0, 1])
    ax2.plot(history_time, history_pitch, color='hotpink', linewidth=2, label='实际俯仰')
    ax2.plot(history_time, history_cmd, color='white', linestyle='--', label='PPO 指令')
    ax2.axvspan(env.fault_start, env.fault_end, color='red', alpha=0.25)
    ax2.set_title('PPO外环指令追踪'); ax2.legend(loc='lower left'); ax2.grid(True, alpha=0.2)

    ax3 = fig.add_subplot(gs[1, :])
    ax3.plot(history_time, history_u0, color='dodgerblue', linewidth=2, label='PID 基础舵角 (u_0)')
    ax3.plot(history_time, history_ud, color='gold', linewidth=2, label='IDHP 在线微调舵角 (u_d)')
    ax3.axvspan(env.fault_start, env.fault_end, color='red', alpha=0.25, label='舵效减半故障区间')
    ax3.set_title('内环控制分配细节 (观察在故障区间的 IDHP 神经补偿输出)', fontsize=14)
    ax3.legend(loc='lower left'); ax3.grid(True, alpha=0.2)

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()