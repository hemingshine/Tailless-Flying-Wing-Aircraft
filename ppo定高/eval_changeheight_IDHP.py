#coding=utf-8
import numpy as np
import math
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import torch
import torch.nn as nn
import torch.optim as optim
from stable_baselines3 import PPO
import warnings

# 导入物理引擎
from fly import NeuralAeroDatabase, EngineDatabase, FlightSimulator6DOF

warnings.filterwarnings('ignore')

# =================================================================
# 1. RLS 在线辨识器 (修复交叉耦合拟合)
# =================================================================
class RLS_Identifier:
    def __init__(self, state_dim=2, input_dim=1):
        self.state_dim = state_dim
        self.input_dim = input_dim
        self.gamma_rls = 0.95 
        self.P = np.eye(state_dim + input_dim) * 1e6 
        self.Theta = np.zeros((state_dim + input_dim, state_dim))
        self.Theta[:state_dim, :state_dim] = np.eye(state_dim) 
        
        # 物理先验：
        self.Theta[2, 0] = 0.0   # 舵面不能瞬间改变角度
        self.Theta[2, 1] = -0.1  # 正舵面导致低头 (角速度减小)

    def update(self, delta_x_prev, delta_u_prev, delta_x_curr):
        Xi = np.zeros((self.state_dim + self.input_dim, 1))
        Xi[:self.state_dim, 0] = delta_x_prev
        Xi[self.state_dim:, 0] = delta_u_prev
        
        if np.linalg.norm(Xi) < 1e-4:
            return self.Theta[:self.state_dim, :].T, self.Theta[self.state_dim:, :].T
            
        delta_x_pred = self.Theta.T @ Xi
        error = delta_x_curr.reshape(-1, 1) - delta_x_pred
        
        K_num = self.P @ Xi
        K_den = self.gamma_rls + Xi.T @ self.P @ Xi
        K = K_num / K_den[0, 0]
        
        self.Theta = self.Theta + K @ error.T
        self.P = (1.0 / self.gamma_rls) * (self.P - K @ Xi.T @ self.P)
        
        # 【物理钳制】：切断幽灵耦合与防反向
        self.Theta[2, 0] = 0.0 
        if self.Theta[2, 1] > -0.01:
            self.Theta[2, 1] = -0.01 
            
        return self.Theta[:self.state_dim, :].T, self.Theta[self.state_dim:, :].T

# =================================================================
# 2. IDHP 容错内环智能体 (加入核心控制惩罚 R)
# =================================================================
class IDHP_Agent:
    def __init__(self):
        self.critic = nn.Sequential(nn.Linear(3, 32), nn.Mish(), nn.Linear(32, 2))
        self.actor = nn.Sequential(nn.Linear(3, 32), nn.Mish(), nn.Linear(32, 1), nn.Tanh())
        self.opt_c = optim.Adam(self.critic.parameters(), lr=0.01)
        self.opt_a = optim.Adam(self.actor.parameters(), lr=0.005)
        self.gamma = 0.8 
        
        # 状态误差惩罚矩阵 Q
        self.Q_mat = torch.tensor([[90.0, 0.0], [0.0, 5.0]], dtype=torch.float32)
        # 控制动作惩罚矩阵 R
        self.R_mat = torch.tensor([20.0], dtype=torch.float32) 

    def get_assistance_action(self, e_theta, e_q, target_theta):
        x_in = torch.tensor([e_theta, e_q, target_theta], dtype=torch.float32)
        with torch.no_grad(): u_d_norm = self.actor(x_in).item()
        return u_d_norm * 3.0 

    def online_train_step(self, e_prev, e_curr, target_theta, F_hat, G_hat, u_d_last_norm):
        e_p = torch.tensor(e_prev, dtype=torch.float32)
        e_c = torch.tensor(e_curr, dtype=torch.float32)
        target = torch.tensor([target_theta], dtype=torch.float32)
        u_d_tensor = torch.tensor([u_d_last_norm], dtype=torch.float32)
        
        in_p = torch.cat([e_p, target])
        in_c = torch.cat([e_c, target])
        G_e = torch.tensor(-G_hat, dtype=torch.float32)
        
        # 1. 更新 Critic
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
        
        # 2. 更新 Actor
        self.opt_a.zero_grad()
        u_d_norm = self.actor(in_p)
        with torch.no_grad(): lambda_curr_new = self.critic(in_c)
        
        actor_grad_direction = torch.matmul(G_e.T, lambda_curr_new) + 2.0 * self.R_mat * u_d_tensor
        
        loss_a = torch.sum(u_d_norm * actor_grad_direction)
        loss_a.backward()
        self.opt_a.step()


# =================================================================
# 3. 终极抗灾正弦追踪联调程序 (PPO预见未来 + IDHP容错内环)
# =================================================================
if __name__ == "__main__":
    print(f"===========================================")
    print(f" 🚀 终极地狱考核：动态正弦追踪 + 55% 断崖式故障 (IDHP容错版)")
    print(f"===========================================")

    aircraft_params = {'S': 88.58, 'b': 18.9, 'c_bar': 4.6, 'mass': 14000,
                       'Ixx': 313220.6, 'Iyy': 273435.3, 'Izz': 392903.9, 'Ixz': 0.0}
    flight_db = NeuralAeroDatabase()
    flight_db._load_from_pickle('aero_surrogate.pth')
    engine_db = EngineDatabase()
    engine_db.load1('engine.pkl')

    sim = FlightSimulator6DOF(flight_db, engine_db, aircraft_params)
    sim.set_initial_state(h_m=3000.0, V_mps=240.0, theta_deg=2.0, alpha_deg=2.0)

    try:
        # 注意修改为你的最新模型路径
        ppo_model = PPO.load("rl_models/mimo_ultimate/best_model.zip")
    except Exception as e:
        print(f"❌ PPO 模型加载失败: {e}")
        exit()

    # 替换原本的 NSMC 控制器为 IDHP 和 RLS
    rls = RLS_Identifier(state_dim=2, input_dim=1)
    idhp = IDHP_Agent()
    
    dt = 0.02
    action_repeat = 10 
    total_steps = int(200 / (dt * action_repeat)) 
    
    history_time, history_alt, history_target_alt, history_vel = [], [], [], []
    history_alpha, history_target_pitch_ppo = [], []
    history_pitch = []
    history_throttle, history_u0, history_ud = [], [], []
    
    integral_h, integral_v, integral_theta = 0.0, 0.0, 0.0
    sim_time = 0.0
    
    # IDHP 缓存状态
    last_x = np.array([math.radians(2.0), 0.0])
    last_u, last_delta_u = 0.0, 0.0
    last_delta_x, last_e = np.zeros(2), np.zeros(2)
    last_u_d_norm = 0.0 
    
    smoothed_action = np.array([0.0, 0.0], dtype=np.float32)
    last_V = 230.0
    
    sine_freq_h = 5 / 200.0  
    sine_amp_h = 30.0
    sine_bias_h = 3000.0

    for step in range(total_steps):
        u, v, w = sim.state[3], sim.state[4], sim.state[5]
        phi, theta = sim.state[6], sim.state[7]
        current_h = -sim.state[2]
        V = math.sqrt(u**2 + v**2 + w**2)
        
        if np.isnan(V) or V < 50.0 or V > 500.0 or current_h < 0 or current_h > 15000.0:
            print(f"❌ 警告：飞机进入物理学死亡螺旋，飞行模拟被强制终止！")
            break
            
        current_vz = u*math.sin(theta) - v*math.sin(phi)*math.cos(theta) - w*math.cos(phi)*math.cos(theta)
        current_ax = (V - last_V) / (dt * action_repeat)
        last_V = V
        
        alpha = math.degrees(math.atan2(w, u))
        gamma = math.degrees(theta) - alpha
        
        # =================================================================
        # 👑 预见未来神技：提前 1.8 秒锁定目标，完美抹平相位滞后！
        # =================================================================
        target_alt_real = sine_bias_h + sine_amp_h * math.sin(2 * math.pi * sine_freq_h * sim_time)
        
        t_lead = 1.8  # 1.8秒提前量
        target_alt_future = sine_bias_h + sine_amp_h * math.sin(2 * math.pi * sine_freq_h * (sim_time + t_lead))
        
        # PPO 追踪的是未来的高度，从而抵消反馈延迟
        err_h = target_alt_future - current_h
        err_v = 230.0 - V  
        
        # 积分器必须使用真实的当前误差，防止积累虚假偏差
        integral_h = np.clip(integral_h + (target_alt_real - current_h) * (dt * action_repeat), -1000.0, 1000.0)
        integral_v = np.clip(integral_v + err_v * (dt * action_repeat), -100.0, 100.0)
        
        obs = np.array([
            err_h / 500.0,       
            current_vz / 10.0,         
            integral_h / 1000.0,  
            err_v / 50.0,              
            current_ax / 5.0,          
            integral_v / 100.0,   
            gamma / 10.0,               
            alpha / 10.0,               
            math.degrees(sim.state[10]) / 10.0
        ], dtype=np.float32)
        
        obs = np.nan_to_num(obs, nan=0.0, posinf=1.0, neginf=-1.0)
        action, _ = ppo_model.predict(obs, deterministic=True)
        
        smoothed_action = 0.5 * smoothed_action + 0.5 * action
        
        target_pitch_ppo = ((smoothed_action[0] + 1.0) / 2.0) * 10.0 - 2.0
        target_throttle = ((smoothed_action[1] + 1.0) / 2.0) * 0.9 + 0.1

        for _ in range(action_repeat):
            u_inner, v_inner, w_inner = sim.state[3], sim.state[4], sim.state[5]
            phi_inner, theta_inner = sim.state[6], sim.state[7]
            
            current_alpha_inner = math.degrees(math.atan2(w_inner, u_inner))
            current_pitch = math.degrees(theta_inner)
            current_q = math.degrees(sim.state[10])
            current_p = math.degrees(sim.state[9])
            current_d_ail = np.clip(1.0 * (0.0 - math.degrees(phi_inner)) - 0.5 * current_p, -10.0, 10.0)
            
            # --- IDHP 内环控制逻辑 ---
            err_theta = target_pitch_ppo - current_pitch
            err_q = 0.0 - current_q
            current_e = np.array([err_theta, err_q])
            current_x = np.array([current_pitch, current_q])
            
            if sim_time > 0.05:
                delta_x_curr = current_x - last_x
                F_hat, G_hat = rls.update(last_delta_x, last_delta_u, delta_x_curr)
                idhp.online_train_step(last_e, current_e, target_pitch_ppo, F_hat, G_hat, last_u_d_norm)
                last_delta_x = delta_x_curr
            
            integral_theta = np.clip(integral_theta + err_theta * dt, -15.0, 15.0)
            u_0 = -3.0 * err_theta + 0.8 * current_q - 0.8 * integral_theta
            
            u_d = idhp.get_assistance_action(err_theta, err_q, target_pitch_ppo)
            last_u_d_norm = u_d / 3.0
            
            u_PE = 0.05 * math.sin(2 * math.pi * 0.5 * sim_time) + 0.02 * math.sin(2 * math.pi * 1.5 * sim_time)
            
            d_flap_cmd = np.clip(u_0 + u_d + u_PE, -20.0, 20.0) 
            
            # =================================================================
            # 🔥 地狱级故障注入：40秒~80秒，舵效骤降 55%（只剩 0.45 动力）！
            # =================================================================
            if 40.0 <= sim_time <= 80.0:
                d_flap_physical = d_flap_cmd * 0.45 
            else:
                d_flap_physical = d_flap_cmd
            
            last_delta_u = d_flap_cmd - last_u
            last_u = d_flap_cmd
            last_x = current_x.copy()
            last_e = current_e.copy()

            sim.step(dt, {'d_flap_L': d_flap_physical, 'd_flap_R': d_flap_physical, 
                          'd_ail_L': current_d_ail, 'd_ail_R': -current_d_ail, 'throttle': target_throttle})
            
            sim_time += dt

            history_time.append(sim_time)
            history_alt.append(-sim.state[2])
            history_target_alt.append(target_alt_real)
            history_vel.append(math.sqrt(sim.state[3]**2 + sim.state[4]**2 + sim.state[5]**2))
            history_alpha.append(current_alpha_inner)
            history_target_pitch_ppo.append(target_pitch_ppo)
            history_pitch.append(current_pitch)
            history_throttle.append(target_throttle)
            history_u0.append(u_0)
            history_ud.append(u_d)

        if step % 20 == 0: 
            print(f"Time: {sim_time:4.1f}s | 目标高度: {target_alt_real:6.1f}m | 实际高度: {-sim.state[2]:6.1f}m | PPO目标Pitch: {target_pitch_ppo:5.2f}° | 实际Pitch: {current_pitch:5.2f}° | IDHP舵角(u_d): {u_d:5.1f}°")

    print("\n✅ 正弦高度追踪与 55% 极端故障 IDHP 容错测试结束！")
    
    # === 绘图 ===
    plt.style.use('dark_background'); plt.rcParams['font.sans-serif'] = ['SimHei']; plt.rcParams['axes.unicode_minus'] = False   
    fig = plt.figure(figsize=(16, 12)); fig.suptitle('飞控地狱模式：PPO 正弦航迹零延迟追踪 + IDHP 55% 失效容错', fontsize=18, color='cyan')
    gs = GridSpec(3, 2, figure=fig)
    
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.plot(history_time, history_alt, color='springgreen', linewidth=3, label='实际高度轨迹')
    ax1.plot(history_time, history_target_alt, color='white', linestyle='--', linewidth=2, label='真实正弦目标高度')
    ax1.axvspan(40, 80, color='red', alpha=0.3, label='突发故障区 (舵效骤降 55%)')
    ax1.set_title('PPO 宏观高度追踪 (预见魔法消除延迟)'); ax1.grid(True, alpha=0.2); ax1.legend()

    ax2 = fig.add_subplot(gs[0, 1])
    ax2.plot(history_time, history_vel, color='cyan', linewidth=2.5)
    ax2.axhline(230.0, color='white', linestyle='--', alpha=0.5, label='目标速度')
    ax2.set_title('PPO 宏观速度管理'); ax2.grid(True, alpha=0.2); ax2.legend()

    ax3 = fig.add_subplot(gs[1, 0])
    ax3.plot(history_time, history_pitch, color='hotpink', linewidth=3, label='实际俯仰角 (Pitch)')
    ax3.plot(history_time, history_target_pitch_ppo, color='white', linestyle='-.', linewidth=2, label='PPO 指令俯仰角')
    ax3.axvspan(40, 80, color='red', alpha=0.3)
    ax3.set_title('IDHP 底层核心追踪'); ax3.legend(); ax3.grid(True, alpha=0.2)

    ax4 = fig.add_subplot(gs[1, 1])
    ax4.plot(history_time, history_throttle, color='gold', linewidth=2, label='PPO 油门输出')
    ax4.set_title('PPO 油门动态决策'); ax4.legend(); ax4.grid(True, alpha=0.2)

    ax5 = fig.add_subplot(gs[2, :])
    ax5.plot(history_time, history_u0, color='dodgerblue', linewidth=2, label='PID 基础舵角 (u_0)')
    ax5.plot(history_time, history_ud, color='gold', linewidth=2, label='IDHP 智能微调舵角 (u_d)')
    ax5.axvspan(40, 80, color='red', alpha=0.3, label='突发故障区 (舵效骤降 55%)')
    ax5.set_title('神级底层：IDHP 瞬间爆发补偿抵消 55% 舵效损失！', fontsize=14)
    ax5.legend(); ax5.grid(True, alpha=0.2)

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()