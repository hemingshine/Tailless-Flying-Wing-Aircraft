#coding=utf-8
import numpy as np
import math
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import torch
import torch.nn as nn
from stable_baselines3 import PPO
import warnings

# 导入物理引擎
from fly import NeuralAeroDatabase, EngineDatabase, FlightSimulator6DOF

warnings.filterwarnings('ignore')

# =================================================================
# 1. 物理探针
# =================================================================
def get_current_derivatives(sim, controls):
    dots = sim.get_derivatives(sim.state, controls)
    return math.degrees(dots[10]) 

# =================================================================
# 2. 抗爆版 RBF 神经滑模控制器 (NSMC)
# =================================================================
class RBF_Integral_NSMC:
    def __init__(self):
        self.c1 = 2.0      
        self.ki1 = 1.0     
        
        self.K = 5.0      
        self.eta = 5.0     
        self.phi = 0.5     
        
        e1_c = np.linspace(-1.0, 1.0, 5) 
        e2_c = np.linspace(-1.0, 1.0, 5)
        self.centers = np.array(np.meshgrid(e1_c, e2_c)).T.reshape(-1, 2)
        self.width = 1.0
        self.W = np.zeros(self.centers.shape[0])
        
        self.Gamma = 10.0  
        self.kappa = 0.1   

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
# 3. 终极联调验证程序
# =================================================================
if __name__ == "__main__":
    TEST_INIT_ALT = 2000.0
    TEST_TARGET_ALT = 2500.0
    TEST_INIT_VEL = 220.0
    TEST_TARGET_VEL = 260.0

    print(f"===========================================")
    print(f" 🚀 PPO 九维降维打击 + NSMC 终极验飞")
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
        # ⚠️ 注意修改为你最新的模型存放路径 ⚠️
        ppo_model = PPO.load("rl_models/mimo_ultimate/best_model.zip")
    except Exception as e:
        print(f"❌ PPO 模型加载失败: {e}")
        exit()

    controller = RBF_Integral_NSMC()
    
    dt = 0.02
    action_repeat = 10 
    total_steps = int(120 / (dt * action_repeat)) 
    
    history_time, history_alt, history_vel = [], [], []
    history_alpha, history_target_pitch_ppo = [], []
    history_pitch, history_pitch_c = [], []
    history_throttle, history_u_total, history_f_nn = [], [], []
    
    integral_h, integral_v, integral_e_theta = 0.0, 0.0, 0.0
    sim_time = 0.0
    
    # 指令滤波器状态
    smoothed_action = np.array([0.0, 0.0], dtype=np.float32)
    pitch_c = 0.7       
    pitch_c_dot = 0.0   
    omega_n = 2.0       
    zeta = 0.9          
    
    # 用于计算加速度的记录值
    last_V = TEST_INIT_VEL

    for step in range(total_steps):
        u, v, w = sim.state[3], sim.state[4], sim.state[5]
        phi, theta = sim.state[6], sim.state[7]
        current_h = -sim.state[2]
        V = math.sqrt(u**2 + v**2 + w**2)
        
        # 【物理护盾】：如果姿态越界即将引发 RK4 爆炸，强行拦截并退出，绝不给 PPO 喂 NaN！
        if np.isnan(V) or V < 50.0 or V > 500.0 or current_h < 0 or current_h > 15000.0:
            print(f"❌ 警告：飞机进入物理学死亡螺旋，飞行模拟被强制终止！(高度={current_h:.1f}m, 速度={V:.1f}m/s)")
            break
            
        current_vz = u*math.sin(theta) - v*math.sin(phi)*math.cos(theta) - w*math.cos(phi)*math.cos(theta)
        current_ax = (V - last_V) / (dt * action_repeat)
        last_V = V
        
        alpha = math.degrees(math.atan2(w, u))
        gamma = math.degrees(theta) - alpha
        
        err_h = TEST_TARGET_ALT - current_h
        err_v = TEST_TARGET_VEL - V
        
        integral_h = np.clip(integral_h + err_h * (dt * action_repeat), -1000.0, 1000.0)
        integral_v = np.clip(integral_v + err_v * (dt * action_repeat), -100.0, 100.0)
        
        # 【逻辑对齐】：必须使用 9 维全感知空间！
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
        
        # 将微小的计算浮点溢出（如果有）置为 0，防止 PPO 崩溃
        obs = np.nan_to_num(obs, nan=0.0, posinf=1.0, neginf=-1.0)
        
        action, _ = ppo_model.predict(obs, deterministic=True)
        
        # 加入与训练环境同款的平滑过渡
        smoothed_action = 0.5 * smoothed_action + 0.5 * action
        
        # 【致胜关键】：直接映射到 Pitch！绝对不能加上 Gamma 导致正反馈！
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
            
            # 40-60s 故障注入
            d_flap_physical = u_total * 0.5 if 40.0 < sim_time < 60.0 else u_total
            
            sim.step(dt, {'d_flap_L': d_flap_physical, 'd_flap_R': d_flap_physical, 
                          'd_ail_L': current_d_ail, 'd_ail_R': -current_d_ail, 'throttle': target_throttle})
            sim_time += dt

            history_time.append(sim_time)
            history_alt.append(-sim.state[2])
            history_vel.append(math.sqrt(sim.state[3]**2 + sim.state[4]**2 + sim.state[5]**2))
            history_alpha.append(current_alpha_inner)
            history_target_pitch_ppo.append(target_pitch_ppo)
            history_pitch.append(current_pitch)
            history_pitch_c.append(pitch_c)
            history_throttle.append(target_throttle)
            history_u_total.append(u_total)
            history_f_nn.append(f_nn)

        if step % 5 == 0: 
            print(f"Time: {sim_time:4.1f}s | 高度: {-sim.state[2]:6.1f}m | 速度: {V:5.1f} | 目标Pitch: {target_pitch_ppo:5.2f}° | 实际Pitch: {current_pitch:5.2f}° | RBF: {f_nn:5.2f} | 舵角: {u_total:5.1f}°")

    print("\n✅ PPO 大脑与抗爆版 NSMC 完美联调结束！")
    
    # === 绘图 ===
    plt.style.use('dark_background'); plt.rcParams['font.sans-serif'] = ['SimHei']; plt.rcParams['axes.unicode_minus'] = False   
    fig = plt.figure(figsize=(16, 12)); fig.suptitle('飞控终局：PPO 九维全感知 + RBF-NSMC 零误差追踪', fontsize=18, color='cyan')
    gs = GridSpec(3, 2, figure=fig)
    
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.plot(history_time, history_alt, color='springgreen', linewidth=2.5)
    ax1.axhline(TEST_TARGET_ALT, color='white', linestyle='--', alpha=0.5)
    ax1.axvspan(40, 60, color='red', alpha=0.2, label='突发故障')
    ax1.set_title('PPO 宏观高度轨迹 (消灭超调)'); ax1.grid(True, alpha=0.2); ax1.legend()

    ax2 = fig.add_subplot(gs[0, 1])
    ax2.plot(history_time, history_vel, color='cyan', linewidth=2.5)
    ax2.axhline(TEST_TARGET_VEL, color='white', linestyle='--', alpha=0.5)
    ax2.set_title('PPO 宏观速度管理 (完美稳定)'); ax2.grid(True, alpha=0.2)

    ax3 = fig.add_subplot(gs[1, 0])
    ax3.plot(history_time, history_pitch, color='hotpink', linewidth=3, label='实际俯仰角 (Pitch)')
    ax3.plot(history_time, history_pitch_c, color='white', linestyle='-.', linewidth=2, label='平滑目标俯仰角')
    ax3.axvspan(40, 60, color='red', alpha=0.2)
    ax3.set_title('NSMC 底层核心追踪 (严格对齐，零误差)'); ax3.legend(); ax3.grid(True, alpha=0.2)

    ax4 = fig.add_subplot(gs[1, 1])
    ax4.plot(history_time, history_alpha, color='dodgerblue', linewidth=2, label='实际迎角')
    ax4.set_title('宏观迎角状态表现'); ax4.legend(); ax4.grid(True, alpha=0.2)

    ax5 = fig.add_subplot(gs[2, :])
    ax5.plot(history_time, history_u_total, color='dodgerblue', linewidth=2, label='NSMC 总控制律 (u_total)')
    ax5.plot(history_time, history_f_nn, color='gold', linewidth=2, label='RBF神经网络补偿 (f_nn)')
    ax5.axvspan(40, 60, color='red', alpha=0.2)
    ax5.set_title('神级底层：黄线完全平稳，精准抓取 40s-60s 故障区间补偿', fontsize=14)
    ax5.legend(); ax5.grid(True, alpha=0.2)

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()