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
        # 【致胜修复】：控制动作惩罚矩阵 R！这就是防止它暴走顶死 3.0 的弹簧！
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
        
        # 【致胜修复】：Actor 的梯度 = G_e^T * \lambda + 2 * R * u_d
        # 加上 2 * R * u_d 会逼迫 Actor 尽量输出 0！
        actor_grad_direction = torch.matmul(G_e.T, lambda_curr_new) + 2.0 * self.R_mat * u_d_tensor
        
        loss_a = torch.sum(u_d_norm * actor_grad_direction)
        loss_a.backward()
        self.opt_a.step()

# =================================================================
# 3. 主程序：原汁原味 PPO + 数学完备 IDHP
# =================================================================
if __name__ == "__main__":
    TEST_INIT_ALT = 2000.0
    TEST_TARGET_ALT = 2500.0
    TEST_INIT_VEL = 220.0
    TEST_TARGET_VEL = 260.0

    print(f"===========================================")
    print(f" 🚀 拨乱反正：基于数学完备惩罚的最终架构")
    print(f"===========================================")

    aircraft_params = {'S': 88.58, 'b': 18.9, 'c_bar': 4.6, 'mass': 14000,
                       'Ixx': 313220.6, 'Iyy': 273435.3, 'Izz': 392903.9, 'Ixz': 0.0}
    flight_db = NeuralAeroDatabase()
    flight_db._load_from_pickle('aero_surrogate.pth')
    engine_db = EngineDatabase()
    engine_db.load1('engine.pkl')

    sim = FlightSimulator6DOF(flight_db, engine_db, aircraft_params)
    sim.set_initial_state(h_m=TEST_INIT_ALT, V_mps=TEST_INIT_VEL, theta_deg=0.7, alpha_deg=0.7)

    try:
        ppo_model = PPO.load("rl_models/mimo/best_model.zip")
    except Exception as e:
        print(f"❌ PPO 模型加载失败: {e}")
        exit()

    rls = RLS_Identifier(state_dim=2, input_dim=1)
    idhp = IDHP_Agent()
    
    dt = 0.02
    action_repeat = 10 
    total_steps = int(200 / (dt * action_repeat)) 
    
    history_time, history_alt, history_vel = [], [], []
    history_alpha, history_target_alpha, history_throttle = [], [], []
    history_u0, history_ud = [], []
    
    integral_h, integral_v, integral_theta = 0.0, 0.0, 0.0
    sim_time = 0.0
    
    last_x = np.array([math.radians(0.7), 0.0])
    last_u, last_delta_u = 0.0, 0.0
    last_delta_x, last_e = np.zeros(2), np.zeros(2)
    last_u_d_norm = 0.0 # 记录用于 R 惩罚

    for step in range(total_steps):
        u, v, w = sim.state[3], sim.state[4], sim.state[5]
        phi, theta = sim.state[6], sim.state[7]
        current_h = -sim.state[2]
        V = math.sqrt(u**2 + v**2 + w**2)
        
        alpha = math.degrees(math.atan2(w, u))
        gamma = math.degrees(theta) - alpha
        
        err_h = TEST_TARGET_ALT - current_h
        err_v = TEST_TARGET_VEL - V
        
        integral_h = np.clip(integral_h + err_h * (dt * action_repeat), -1000.0, 1000.0)
        integral_v = np.clip(integral_v + err_v * (dt * action_repeat), -100.0, 100.0)
        
        obs = np.array([
            err_h / 500.0,       
            integral_h / 1000.0,  
            err_v / 50.0,
            integral_v / 100.0,
            gamma / 10.0,               
            alpha / 10.0,               
            math.degrees(sim.state[10]) / 10.0
        ], dtype=np.float32)
        
        # PPO 下达原汁原味的指令
        action, _ = ppo_model.predict(obs, deterministic=True)
        target_alpha_ppo = ((action[0] + 1.0) / 2.0) * 8.0 - 2.0
        target_throttle = ((action[1] + 1.0) / 2.0) * 0.9 + 0.1

        # 转换为安全的俯仰角指令
        target_pitch = target_alpha_ppo + gamma

        for _ in range(action_repeat):
            u_inner, v_inner, w_inner = sim.state[3], sim.state[4], sim.state[5]
            phi_inner, theta_inner = sim.state[6], sim.state[7]
            
            current_pitch = math.degrees(theta_inner)
            current_q = math.degrees(sim.state[10])
            current_p = math.degrees(sim.state[9])
            
            err_theta = target_pitch - current_pitch
            err_q = 0.0 - current_q
            current_e = np.array([err_theta, err_q])
            current_x = np.array([current_pitch, current_q])
            
            if sim_time > 0.05:
                delta_x_curr = current_x - last_x
                F_hat, G_hat = rls.update(last_delta_x, last_delta_u, delta_x_curr)
                idhp.online_train_step(last_e, current_e, target_pitch, F_hat, G_hat, last_u_d_norm)
                last_delta_x = delta_x_curr
            
            integral_theta = np.clip(integral_theta + err_theta * dt, -15.0, 15.0)
            u_0 = -3 * err_theta + 0.8 * current_q - 0.8 * integral_theta
            
            u_d = idhp.get_assistance_action(err_theta, err_q, target_pitch)
            last_u_d_norm = u_d / 3.0 # 保存归一化值，供下次更新惩罚
            
            u_PE = 0.05 * math.sin(2 * math.pi * 0.5 * sim_time) + 0.02 * math.sin(2 * math.pi * 1.5 * sim_time)
            d_flap_cmd = np.clip(u_0 + u_d + u_PE, -15.0, 15.0) 
            
            # 40-60s 故障注入
            d_flap_physical = d_flap_cmd
            if 80.0 < sim_time < 100.0:
                d_flap_physical = d_flap_cmd * 0.5 
            
            d_ail = np.clip(1.0 * (0.0 - math.degrees(phi_inner)) - 0.5 * current_p, -10.0, 10.0)
            
            last_delta_u = d_flap_cmd - last_u
            last_u = d_flap_cmd
            last_x = current_x.copy()
            last_e = current_e.copy()
            
            sim.step(dt, {'d_flap_L': d_flap_physical, 'd_flap_R': d_flap_physical, 
                          'd_ail_L': d_ail, 'd_ail_R': -d_ail, 'throttle': target_throttle})
            sim_time += dt

            history_time.append(sim_time)
            history_alt.append(-sim.state[2])
            history_vel.append(math.sqrt(sim.state[3]**2 + sim.state[4]**2 + sim.state[5]**2))
            history_alpha.append(math.degrees(math.atan2(w_inner, u_inner)))
            history_target_alpha.append(target_alpha_ppo)
            history_throttle.append(target_throttle)
            history_u0.append(u_0)
            history_ud.append(u_d)

        if step % 50 == 0: 
            print(f"Time: {sim_time:4.1f}s | 高度: {-sim.state[2]:6.1f}m | PPO油门: {target_throttle:.2f} | PID: {u_0:5.1f}° | IDHP: {u_d:5.2f}°|ppo决策迎角：{target_alpha_ppo:5.2f} |实际迎角：{math.degrees(math.atan2(w_inner, u_inner)):5.2f}")

    print("\n✅ 回归与救赎完成！")
    
    # ============================================================
    # 6. 精简版可视化：高度、迎角、内环控制 + 故障区间展示
    # ============================================================
    plt.style.use('dark_background') 
    plt.rcParams['font.sans-serif'] = ['SimHei'] 
    plt.rcParams['axes.unicode_minus'] = False   
    fig = plt.figure(figsize=(15, 10))
    fig.suptitle('原生 MIMO PPO + 数学完备版 IDHP', fontsize=18, color='cyan')

    # 改为3行1列的布局，使三个关键子图更加宽阔
    gs = GridSpec(3, 1, figure=fig)
    
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.plot(history_time, history_alt, color='springgreen', linewidth=2.5)
    ax1.axhline(TEST_TARGET_ALT, color='white', linestyle='--', alpha=0.5)
    ax1.axvspan(80, 100, color='red', alpha=0.2, label='突发故障区') # 加入故障高亮
    ax1.set_title('高度轨迹'); ax1.legend(loc='upper right'); ax1.grid(True, alpha=0.2)

    ax2 = fig.add_subplot(gs[1, 0])
    ax2.plot(history_time, history_alpha, color='hotpink', linewidth=2, label='实际迎角')
    ax2.plot(history_time, history_target_alpha, color='white', linestyle='--', label='PPO 指令迎角')
    ax2.axvspan(80, 100, color='red', alpha=0.2, label='突发故障区') # 加入故障高亮
    ax2.set_title('PPO 迎角决策与追踪'); ax2.legend(loc='upper right'); ax2.grid(True, alpha=0.2)

    ax3 = fig.add_subplot(gs[2, 0])
    ax3.plot(history_time, history_u0, color='dodgerblue', linewidth=2, label='PID 基础舵角 (u_0)')
    ax3.plot(history_time, history_ud, color='gold', linewidth=2, label='IDHP 智能微调 (u_d)')
    ax3.axvspan(80, 100, color='red', alpha=0.2, label='突发故障区') # 加入故障高亮
    ax3.set_title('内环控制：告别暴走，精准容错', fontsize=14)
    ax3.legend(loc='upper right'); ax3.grid(True, alpha=0.2)

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()