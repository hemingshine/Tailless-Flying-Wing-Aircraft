#coding=utf-8
import os
import math
import numpy as np
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
# 1. RLS 在线辨识器 & IDHP 智能体 (Method 2)
# =================================================================
class RLS_Identifier:
    def __init__(self, state_dim=2, input_dim=1):
        self.state_dim = state_dim
        self.input_dim = input_dim
        self.gamma_rls = 0.95 
        self.P = np.eye(state_dim + input_dim) * 1e6 
        self.Theta = np.zeros((state_dim + input_dim, state_dim))
        self.Theta[:state_dim, :state_dim] = np.eye(state_dim) 
        
        self.Theta[2, 0] = 0.0   
        self.Theta[2, 1] = -0.1  

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
        
        self.Theta[2, 0] = 0.0 
        if self.Theta[2, 1] > -0.01:
            self.Theta[2, 1] = -0.01 
            
        return self.Theta[:self.state_dim, :].T, self.Theta[self.state_dim:, :].T

class IDHP_Agent:
    def __init__(self):
        self.critic = nn.Sequential(nn.Linear(3, 32), nn.Mish(), nn.Linear(32, 2))
        self.actor = nn.Sequential(nn.Linear(3, 32), nn.Mish(), nn.Linear(32, 1), nn.Tanh())
        self.opt_c = optim.Adam(self.critic.parameters(), lr=0.01)
        self.opt_a = optim.Adam(self.actor.parameters(), lr=0.005)
        self.gamma = 0.8 
        self.Q_mat = torch.tensor([[90.0, 0.0], [0.0, 5.0]], dtype=torch.float32)
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
        actor_grad_direction = torch.matmul(G_e.T, lambda_curr_new) + 2.0 * self.R_mat * u_d_tensor
        loss_a = torch.sum(u_d_norm * actor_grad_direction)
        loss_a.backward()
        self.opt_a.step()

# =================================================================
# 2. 物理探针 & RBF 积分神经滑模控制器 (NSMC) (Method 1)
# =================================================================
def get_current_derivatives(sim, controls):
    dots = sim.get_derivatives(sim.state, controls)
    return math.degrees(dots[10]) 

class RBF_Integral_NSMC:
    def __init__(self):
        self.c1 = 4.0      
        self.ki1 = 2.0     
        self.K = 10.0      
        self.eta = 5.0     
        self.phi = 0.5     
        
        e1_c = np.linspace(-1.0, 1.0, 5) 
        e2_c = np.linspace(-1.0, 1.0, 5)
        self.centers = np.array(np.meshgrid(e1_c, e2_c)).T.reshape(-1, 2)
        self.width = 1.0
        self.W = np.zeros(self.centers.shape[0])
        
        self.Gamma = 50.0  
        self.kappa = 0.05   

    def compute_control(self, e_theta, int_e_theta, q, theta_c_dot, theta_c_ddot, f2_nom, ce0, dt):
        q_c = -self.c1 * e_theta - self.ki1 * int_e_theta + theta_c_dot
        q_c_dot = -self.c1 * (q - theta_c_dot) - self.ki1 * e_theta + theta_c_ddot
        s = q - q_c  
        
        x_nn = np.array([e_theta / 10.0, s / 20.0])
        dist_sq = np.sum((self.centers - x_nn)**2, axis=1)
        h = np.exp(-dist_sq / (2 * self.width**2))
        
        self.W += (self.Gamma * (s / 20.0) * h - self.kappa * self.W) * dt
        f_nn = np.dot(self.W, h) 
        
        u_eq = (-f2_nom + q_c_dot - f_nn * 10.0) / ce0 
        u_sw = (-self.K * s - self.eta * math.tanh(s / self.phi)) / ce0
        
        u_total = np.clip(u_eq + u_sw, -20.0, 20.0)
        return u_total, f_nn * 10.0, s


# =================================================================
# 3. 核心仿真封装函数
# =================================================================
def run_nsmc(ppo_model, flight_db, engine_db, aircraft_params):
    sim = FlightSimulator6DOF(flight_db, engine_db, aircraft_params)
    sim.set_initial_state(h_m=3000.0, V_mps=240.0, theta_deg=2.0, alpha_deg=2.0)
    controller = RBF_Integral_NSMC()
    
    dt = 0.02
    action_repeat = 10 
    total_steps = int(200 / (dt * action_repeat)) 
    
    hist = {'t': [], 'alt': [], 'target_alt': [], 'pitch': [], 'target_pitch': [], 'u_total': [], 'f_nn': []}
    
    integral_h, integral_v, integral_e_theta = 0.0, 0.0, 0.0
    sim_time = 0.0
    smoothed_action = np.array([0.0, 0.0], dtype=np.float32)
    pitch_c = 2.0; pitch_c_dot = 0.0; omega_n = 2.0; zeta = 0.9          
    last_V = 240.0
    sine_freq_h, sine_amp_h, sine_bias_h = 5 / 200.0, 30.0, 3000.0

    for step in range(total_steps):
        u, v, w = sim.state[3], sim.state[4], sim.state[5]
        phi, theta = sim.state[6], sim.state[7]
        current_h = -sim.state[2]
        V = math.sqrt(u**2 + v**2 + w**2)
        if np.isnan(V) or V < 50.0: break
            
        current_vz = u*math.sin(theta) - v*math.sin(phi)*math.cos(theta) - w*math.cos(phi)*math.cos(theta)
        current_ax = (V - last_V) / (dt * action_repeat)
        last_V = V
        alpha = math.degrees(math.atan2(w, u))
        gamma = math.degrees(theta) - alpha
        
        target_alt_real = sine_bias_h + sine_amp_h * math.sin(2 * math.pi * sine_freq_h * sim_time)
        t_lead = 1.8  
        target_alt_future = sine_bias_h + sine_amp_h * math.sin(2 * math.pi * sine_freq_h * (sim_time + t_lead))
        err_h = target_alt_future - current_h
        err_v = 230.0 - V  
        
        integral_h = np.clip(integral_h + (target_alt_real - current_h) * (dt * action_repeat), -1000.0, 1000.0)
        integral_v = np.clip(integral_v + err_v * (dt * action_repeat), -100.0, 100.0)
        
        obs = np.array([err_h/500., current_vz/10., integral_h/1000., err_v/50., current_ax/5., integral_v/100., gamma/10., alpha/10., math.degrees(sim.state[10])/10.], dtype=np.float32)
        obs = np.nan_to_num(obs, nan=0.0, posinf=1.0, neginf=-1.0)
        action, _ = ppo_model.predict(obs, deterministic=True)
        smoothed_action = 0.5 * smoothed_action + 0.5 * action
        
        target_pitch_ppo = ((smoothed_action[0] + 1.0) / 2.0) * 10.0 - 2.0
        target_throttle = ((smoothed_action[1] + 1.0) / 2.0) * 0.9 + 0.1

        for _ in range(action_repeat):
            u_inner, w_inner = sim.state[3], sim.state[5]
            phi_inner, theta_inner = sim.state[6], sim.state[7]
            current_pitch = math.degrees(theta_inner)
            current_q = math.degrees(sim.state[10])
            current_p = math.degrees(sim.state[9])
            current_d_ail = np.clip(1.0 * (0.0 - math.degrees(phi_inner)) - 0.5 * current_p, -10.0, 10.0)
            
            pitch_c_ddot = omega_n**2 * (target_pitch_ppo - pitch_c) - 2 * zeta * omega_n * pitch_c_dot
            pitch_c += pitch_c_dot * dt
            pitch_c_dot += pitch_c_ddot * dt
            
            controls_0 = {'d_flap_L': 0.0, 'd_flap_R': 0.0, 'd_ail_L': current_d_ail, 'd_ail_R': -current_d_ail, 'throttle': target_throttle}
            controls_1 = {'d_flap_L': 1.0, 'd_flap_R': 1.0, 'd_ail_L': current_d_ail, 'd_ail_R': -current_d_ail, 'throttle': target_throttle}
            
            f2_nom = get_current_derivatives(sim, controls_0) 
            q_dot_1 = get_current_derivatives(sim, controls_1)
            ce0_nom = q_dot_1 - f2_nom 
            if abs(ce0_nom) < 1e-2: ce0_nom = -1e-2 if ce0_nom <= 0 else 1e-2
            
            e_theta = current_pitch - pitch_c
            integral_e_theta = np.clip(integral_e_theta + e_theta * dt, -10.0, 10.0)
            
            u_total, f_nn, s_val = controller.compute_control(e_theta, integral_e_theta, current_q, pitch_c_dot, pitch_c_ddot, f2_nom, ce0_nom, dt)
            
            # 故障注入：40s-80s 舵效剩 45%
            d_flap_physical = u_total * 0.45 if 40.0 <= sim_time <= 80.0 else u_total
            sim.step(dt, {'d_flap_L': d_flap_physical, 'd_flap_R': d_flap_physical, 'd_ail_L': current_d_ail, 'd_ail_R': -current_d_ail, 'throttle': target_throttle})
            sim_time += dt

            hist['t'].append(sim_time)
            hist['alt'].append(-sim.state[2])
            hist['target_alt'].append(target_alt_real)
            hist['pitch'].append(current_pitch)
            hist['target_pitch'].append(pitch_c)
            hist['u_total'].append(u_total)
            hist['f_nn'].append(f_nn)

    return hist

def run_idhp(ppo_model, flight_db, engine_db, aircraft_params):
    sim = FlightSimulator6DOF(flight_db, engine_db, aircraft_params)
    sim.set_initial_state(h_m=3000.0, V_mps=240.0, theta_deg=2.0, alpha_deg=2.0)
    
    rls = RLS_Identifier(state_dim=2, input_dim=1)
    idhp = IDHP_Agent()
    
    dt = 0.02
    action_repeat = 10 
    total_steps = int(200 / (dt * action_repeat)) 
    
    hist = {'t': [], 'alt': [], 'target_alt': [], 'pitch': [], 'target_pitch': [], 'u_0': [], 'u_d': []}
    
    integral_h, integral_v, integral_theta = 0.0, 0.0, 0.0
    sim_time = 0.0
    
    last_x = np.array([math.radians(2.0), 0.0])
    last_u, last_delta_u, last_u_d_norm = 0.0, 0.0, 0.0
    last_delta_x, last_e = np.zeros(2), np.zeros(2)
    smoothed_action = np.array([0.0, 0.0], dtype=np.float32)
    last_V = 240.0
    sine_freq_h, sine_amp_h, sine_bias_h = 5 / 200.0, 30.0, 3000.0

    for step in range(total_steps):
        u, v, w = sim.state[3], sim.state[4], sim.state[5]
        phi, theta = sim.state[6], sim.state[7]
        current_h = -sim.state[2]
        V = math.sqrt(u**2 + v**2 + w**2)
        if np.isnan(V) or V < 50.0: break
            
        current_vz = u*math.sin(theta) - v*math.sin(phi)*math.cos(theta) - w*math.cos(phi)*math.cos(theta)
        current_ax = (V - last_V) / (dt * action_repeat)
        last_V = V
        alpha = math.degrees(math.atan2(w, u))
        gamma = math.degrees(theta) - alpha
        
        target_alt_real = sine_bias_h + sine_amp_h * math.sin(2 * math.pi * sine_freq_h * sim_time)
        t_lead = 1.8 
        target_alt_future = sine_bias_h + sine_amp_h * math.sin(2 * math.pi * sine_freq_h * (sim_time + t_lead))
        err_h = target_alt_future - current_h
        err_v = 230.0 - V  
        
        integral_h = np.clip(integral_h + (target_alt_real - current_h) * (dt * action_repeat), -1000.0, 1000.0)
        integral_v = np.clip(integral_v + err_v * (dt * action_repeat), -100.0, 100.0)
        
        obs = np.array([err_h/500., current_vz/10., integral_h/1000., err_v/50., current_ax/5., integral_v/100., gamma/10., alpha/10., math.degrees(sim.state[10])/10.], dtype=np.float32)
        obs = np.nan_to_num(obs, nan=0.0, posinf=1.0, neginf=-1.0)
        action, _ = ppo_model.predict(obs, deterministic=True)
        smoothed_action = 0.5 * smoothed_action + 0.5 * action
        
        target_pitch_ppo = ((smoothed_action[0] + 1.0) / 2.0) * 10.0 - 2.0
        target_throttle = ((smoothed_action[1] + 1.0) / 2.0) * 0.9 + 0.1

        for _ in range(action_repeat):
            u_inner, w_inner = sim.state[3], sim.state[5]
            phi_inner, theta_inner = sim.state[6], sim.state[7]
            current_pitch = math.degrees(theta_inner)
            current_q = math.degrees(sim.state[10])
            current_p = math.degrees(sim.state[9])
            current_d_ail = np.clip(1.0 * (0.0 - math.degrees(phi_inner)) - 0.5 * current_p, -10.0, 10.0)
            
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
            
            d_flap_physical = d_flap_cmd * 0.45 if 40.0 <= sim_time <= 80.0 else d_flap_cmd
            
            last_delta_u = d_flap_cmd - last_u
            last_u = d_flap_cmd
            last_x = current_x.copy()
            last_e = current_e.copy()

            sim.step(dt, {'d_flap_L': d_flap_physical, 'd_flap_R': d_flap_physical, 'd_ail_L': current_d_ail, 'd_ail_R': -current_d_ail, 'throttle': target_throttle})
            sim_time += dt

            hist['t'].append(sim_time)
            hist['alt'].append(-sim.state[2])
            hist['target_alt'].append(target_alt_real)
            hist['pitch'].append(current_pitch)
            hist['target_pitch'].append(target_pitch_ppo)
            hist['u_0'].append(u_0)
            hist['u_d'].append(u_d)

    return hist

def main():
    print("=====================================================")
    print("  🚀 对比仿真启动: RBF-NSMC (方法1) vs IDHP (方法2)")
    print("=====================================================")

    # 环境与模型初始化
    aircraft_params = {'S': 88.58, 'b': 18.9, 'c_bar': 4.6, 'mass': 14000,
                       'Ixx': 313220.6, 'Iyy': 273435.3, 'Izz': 392903.9, 'Ixz': 0.0}
    flight_db = NeuralAeroDatabase()
    flight_db._load_from_pickle('aero_surrogate.pth')
    engine_db = EngineDatabase()
    engine_db.load1('engine.pkl')

    try:
        model_path = "rl_models/mimo_ultimate/best_model.zip"
        ppo_model = PPO.load(model_path)
        print("✅ 成功加载高层宏观 PPO 轨迹跟踪大脑")
    except Exception as e:
        print(f"❌ PPO 模型加载失败: {e}")
        return

    print("🛫 正在运行 NSMC (神经积分滑模) 容错试飞...")
    hist_nsmc = run_nsmc(ppo_model, flight_db, engine_db, aircraft_params)
    
    print("🛫 正在运行 IDHP (启发式动态规划) 容错试飞...")
    hist_idhp = run_idhp(ppo_model, flight_db, engine_db, aircraft_params)

    # ==========================================
    # 📊 提取并打印量化评估指标 (聚焦 40s - 80s 故障区)
    # ==========================================
    def compute_metrics(hist):
        t_arr = np.array(hist['t'])
        idx_fault = np.where((t_arr >= 40.0) & (t_arr <= 80.0))[0]
        
        alt_err = np.array(hist['alt'])[idx_fault] - np.array(hist['target_alt'])[idx_fault]
        pitch_err = np.array(hist['pitch'])[idx_fault] - np.array(hist['target_pitch'])[idx_fault]
        
        rmse_alt = float(np.sqrt(np.mean(alt_err**2)))
        rmse_pitch = float(np.sqrt(np.mean(pitch_err**2)))
        
        # 提取瞬态最大偏差 (故障发生后5秒内的偏差)
        idx_transient = np.where((t_arr >= 40.0) & (t_arr <= 45.0))[0]
        max_transient_alt = float(np.max(np.abs(np.array(hist['alt'])[idx_transient] - np.array(hist['target_alt'])[idx_transient])))
        max_transient_pitch = float(np.max(np.abs(np.array(hist['pitch'])[idx_transient] - np.array(hist['target_pitch'])[idx_transient])))
        
        return rmse_alt, rmse_pitch, max_transient_alt, max_transient_pitch

    m_nsmc = compute_metrics(hist_nsmc)
    m_idhp = compute_metrics(hist_idhp)

    print("\n" + "="*80)
    print(" 📊 底层容错算法量化对比矩阵 (故障期间: 40s - 80s, 舵效剩余 45%)")
    print("="*80)
    print(f" {'控制算法':<15} | {'高度 RMSE':<12} | {'俯仰角 RMSE':<12} | {'高度最大瞬态偏差':<16} | {'俯仰最大瞬态偏差'}")
    print("-" * 80)
    print(f" {'RBF-NSMC':<15} | {m_nsmc[0]:8.2f} m  | {m_nsmc[1]:8.2f}°    | {m_nsmc[2]:12.2f} m  | {m_nsmc[3]:10.2f}°")
    print(f" {'IDHP':<15} | {m_idhp[0]:8.2f} m  | {m_idhp[1]:8.2f}°    | {m_idhp[2]:12.2f} m  | {m_idhp[3]:10.2f}°")
    print("="*80 + "\n")

    print("🛬 试飞完毕！正在生成全英文 2行3列 学术排版图表...")

    # ==========================================
    # 📊 绘制专业学术图表 (2行3列，无标题，纯英文)
    # ==========================================
    plt.style.use('bmh')
    plt.rcParams['font.sans-serif'] = ['SimHei']
    plt.rcParams['axes.unicode_minus'] = False
    
    fig = plt.figure(figsize=(18, 10))
    gs = GridSpec(2, 3, figure=fig)
    
    fault_t0, fault_t1 = 40.0, 80.0

    # ------------------ Row 1: RBF-NSMC ------------------
    # NSMC: Altitude
    ax_n1 = fig.add_subplot(gs[0, 0])
    ax_n1.plot(hist_nsmc['t'], hist_nsmc['target_alt'], 'k--', lw=1.5, label='Target Alt')
    ax_n1.plot(hist_nsmc['t'], hist_nsmc['alt'], '#1f77b4', lw=2.0, alpha=0.9, label='Actual Alt (NSMC)')
    ax_n1.axvspan(fault_t0, fault_t1, color='red', alpha=0.1)
    ax_n1.set_ylabel('Altitude [m]', fontweight='bold')
    ax_n1.legend(loc='lower right', fontsize=9)
    
    # NSMC: Pitch
    ax_n2 = fig.add_subplot(gs[0, 1])
    ax_n2.plot(hist_nsmc['t'], hist_nsmc['target_pitch'], 'k--', lw=1.5, label='Cmd Pitch')
    ax_n2.plot(hist_nsmc['t'], hist_nsmc['pitch'], '#1f77b4', lw=2.0, alpha=0.9, label='Actual Pitch (NSMC)')
    ax_n2.axvspan(fault_t0, fault_t1, color='red', alpha=0.1)
    ax_n2.set_ylabel('Pitch Angle [deg]', fontweight='bold')
    ax_n2.legend(loc='lower right', fontsize=9)
    
    # NSMC: Compensation
    ax_n3 = fig.add_subplot(gs[0, 2])
    ax_n3.plot(hist_nsmc['t'], hist_nsmc['u_total'], '#1f77b4', lw=2.0, alpha=0.8, label='Total Control (NSMC)')
    ax_n3.plot(hist_nsmc['t'], hist_nsmc['f_nn'], '#ff7f0e', lw=2.0, alpha=0.9, label='NN Comp (f_nn)')
    ax_n3.axvspan(fault_t0, fault_t1, color='red', alpha=0.1)
    ax_n3.set_ylabel('Control & Comp [deg]', fontweight='bold')
    ax_n3.legend(loc='upper right', fontsize=9)

    # ------------------ Row 2: IDHP ------------------
    # IDHP: Altitude
    ax_i1 = fig.add_subplot(gs[1, 0])
    ax_i1.plot(hist_idhp['t'], hist_idhp['target_alt'], 'k--', lw=1.5, label='Target Alt')
    ax_i1.plot(hist_idhp['t'], hist_idhp['alt'], '#2ca02c', lw=2.0, alpha=0.9, label='Actual Alt (IDHP)')
    ax_i1.axvspan(fault_t0, fault_t1, color='red', alpha=0.1)
    ax_i1.set_xlabel('Time [s]', fontweight='bold')
    ax_i1.set_ylabel('Altitude [m]', fontweight='bold')
    ax_i1.legend(loc='lower right', fontsize=9)
    
    # IDHP: Pitch
    ax_i2 = fig.add_subplot(gs[1, 1])
    ax_i2.plot(hist_idhp['t'], hist_idhp['target_pitch'], 'k--', lw=1.5, label='Cmd Pitch')
    ax_i2.plot(hist_idhp['t'], hist_idhp['pitch'], '#2ca02c', lw=2.0, alpha=0.9, label='Actual Pitch (IDHP)')
    ax_i2.axvspan(fault_t0, fault_t1, color='red', alpha=0.1)
    ax_i2.set_xlabel('Time [s]', fontweight='bold')
    ax_i2.set_ylabel('Pitch Angle [deg]', fontweight='bold')
    ax_i2.legend(loc='lower right', fontsize=9)
    
    # IDHP: Compensation
    ax_i3 = fig.add_subplot(gs[1, 2])
    ax_i3.plot(hist_idhp['t'], hist_idhp['u_0'], '#1f77b4', lw=2.0, alpha=0.8, label='Base PID (u_0)')
    ax_i3.plot(hist_idhp['t'], hist_idhp['u_d'], '#d62728', lw=2.0, alpha=0.9, label='IDHP Comp (u_d)')
    ax_i3.axvspan(fault_t0, fault_t1, color='red', alpha=0.1)
    ax_i3.set_xlabel('Time [s]', fontweight='bold')
    ax_i3.set_ylabel('PID & IDHP Comp [deg]', fontweight='bold')
    ax_i3.legend(loc='upper right', fontsize=9)

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()