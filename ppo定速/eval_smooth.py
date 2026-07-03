#coding=utf-8
import numpy as np
import math
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import torch
from stable_baselines3 import PPO
import warnings

from fly import NeuralAeroDatabase, EngineDatabase, FlightSimulator6DOF

warnings.filterwarnings('ignore')

def get_current_derivatives(sim, controls):
    return math.degrees(sim.get_derivatives(sim.state, controls)[10]) 

# =================================================================
# 1. 统一个体：RBF-NSMC 保持完全不变
# =================================================================
class RBF_Integral_NSMC:
    def __init__(self):
        self.c1 = 4.0; self.ki1 = 2.0; self.K = 10.0; self.eta = 5.0; self.phi = 0.5
        e1_c = np.linspace(-1.0, 1.0, 5); e2_c = np.linspace(-1.0, 1.0, 5)
        self.centers = np.array(np.meshgrid(e1_c, e2_c)).T.reshape(-1, 2)
        self.width = 1.0; self.W = np.zeros(self.centers.shape[0])
        self.Gamma = 50.0; self.kappa = 0.05      
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
        return np.clip(u_eq + u_sw, -20.0, 20.0), f_nn * 10.0, s

if __name__ == "__main__":
    TEST_TARGET_ALT = 4000.0  
    TEST_INIT_VEL = 200.0     

    print(f"===========================================")
    print(f" 🚀 终极地狱考核：TECS 阶梯变速 + 55% 突发故障")
    print(f"===========================================")

    aircraft_params = {'S': 88.58, 'b': 18.9, 'c_bar': 4.6, 'mass': 14000,
                       'Ixx': 313220.6, 'Iyy': 273435.3, 'Izz': 392903.9, 'Ixz': 0.0}
    flight_db = NeuralAeroDatabase(); flight_db._load_from_pickle('aero_surrogate.pth')
    engine_db = EngineDatabase(); engine_db.load1('engine.pkl')

    sim = FlightSimulator6DOF(flight_db, engine_db, aircraft_params)
    sim.set_initial_state(h_m=TEST_TARGET_ALT, V_mps=TEST_INIT_VEL, theta_deg=3.0, alpha_deg=3.0)

    try:
        # 注意修改为你的 TECS 模型路径
        ppo_model = PPO.load("rl_models/mimo_tecs/best_model.zip")
    except Exception as e:
        print(f"❌ PPO 模型加载失败: {e}")
        exit()

    controller = RBF_Integral_NSMC()
    dt = 0.02; action_repeat = 10; 
    # 延长仿真时间至 160 秒以容纳复杂的阶梯变化
    total_steps = int(400 / (dt * action_repeat)) 
    g = 9.80665
    
    history_time, history_alt, history_vel, history_target_vel = [], [], [], []
    history_pitch, history_pitch_c = [], []
    history_throttle, history_u_total, history_f_nn = [], [], []
    sim_time = 0.0; last_V = TEST_INIT_VEL
    
    integral_e_theta = 0.0
    smoothed_action = np.array([0.0, 0.0], dtype=np.float32)
    pitch_c = 3.0; pitch_c_dot = 0.0; omega_n = 2.0; zeta = 0.9          
    actual_throttle = 0.5

    for step in range(total_steps):
        # =================================================================
        # 👑 阶梯速度调度逻辑
        # =================================================================
        if sim_time < 80.0:
            current_target_vel = 180.0  # 初始巡航
        elif sim_time < 160.0:
            current_target_vel = 230.0  # 阶跃加速1
        elif sim_time < 240.0:
            current_target_vel = 220.0  # 阶跃加速2 (极限高速)
        else:
            current_target_vel = 240.0  # 断崖式减速 (极限低速)

        u, v, w = sim.state[3], sim.state[4], sim.state[5]
        phi, theta = sim.state[6], sim.state[7]
        current_h = -sim.state[2]
        V = math.sqrt(u**2 + v**2 + w**2)
        
        current_vz = u*math.sin(theta) - v*math.sin(phi)*math.cos(theta) - w*math.cos(phi)*math.cos(theta)
        current_ax = (V - last_V) / (dt * action_repeat)
        last_V = V
        
        alpha = math.degrees(math.atan2(w, u))
        gamma = math.degrees(theta) - alpha
        
        err_h = TEST_TARGET_ALT - current_h
        err_v = current_target_vel - V
        
        # 👑 TECS 能量解耦观测 (严格使用当前的动态目标速度)
        E_t_target = TEST_TARGET_ALT + (current_target_vel**2) / (2 * g)
        E_t_current = current_h + (V**2) / (2 * g)
        err_Et = E_t_target - E_t_current
        
        E_d_target = TEST_TARGET_ALT - (current_target_vel**2) / (2 * g)
        E_d_current = current_h - (V**2) / (2 * g)
        err_Ed = E_d_target - E_d_current
        
        obs = np.array([
            err_Et / 100.0, err_Ed / 100.0, err_h / 500.0,  
            current_vz / 10.0, err_v / 50.0, current_ax / 5.0,   
            gamma / 10.0, alpha / 10.0, math.degrees(sim.state[10]) / 10.0
        ], dtype=np.float32)
        
        obs = np.nan_to_num(obs, nan=0.0, posinf=1.0, neginf=-1.0)
        action, _ = ppo_model.predict(obs, deterministic=True)
        smoothed_action = 0.5 * smoothed_action + 0.5 * action
        
        target_pitch_ppo = ((smoothed_action[0] + 1.0) / 2.0) * 10.0 - 2.0
        target_throttle_ppo = ((smoothed_action[1] + 1.0) / 2.0) * 0.9 + 0.1

        for _ in range(action_repeat):
            tau_engine = 0.5 
            actual_throttle += (target_throttle_ppo - actual_throttle) * (dt / tau_engine)
            
            pitch_c_ddot = omega_n**2 * (target_pitch_ppo - pitch_c) - 2 * zeta * omega_n * pitch_c_dot
            pitch_c += pitch_c_dot * dt
            pitch_c_dot += pitch_c_ddot * dt
            
            u_inner, v_inner, w_inner = sim.state[3], sim.state[4], sim.state[5]
            phi_inner, theta_inner = sim.state[6], sim.state[7]
            current_pitch = math.degrees(theta_inner)
            current_q = math.degrees(sim.state[10])
            current_p = math.degrees(sim.state[9])
            current_d_ail = np.clip(1.0 * (0.0 - math.degrees(phi_inner)) - 0.5 * current_p, -10.0, 10.0)
            
            controls_0 = {'d_flap_L': 0.0, 'd_flap_R': 0.0, 'd_ail_L': current_d_ail, 'd_ail_R': -current_d_ail, 'throttle': actual_throttle}
            controls_1 = {'d_flap_L': 1.0, 'd_flap_R': 1.0, 'd_ail_L': current_d_ail, 'd_ail_R': -current_d_ail, 'throttle': actual_throttle}
            
            f2_nom = get_current_derivatives(sim, controls_0) 
            q_dot_1 = get_current_derivatives(sim, controls_1)
            ce0_nom = q_dot_1 - f2_nom 
            if abs(ce0_nom) < 1e-2: ce0_nom = -1e-2 if ce0_nom <= 0 else 1e-2
            
            e_theta = current_pitch - pitch_c
            integral_e_theta = np.clip(integral_e_theta + e_theta * dt, -10.0, 10.0)
            
            u_total, f_nn, s_val = controller.compute_control(e_theta, integral_e_theta, current_q, pitch_c_dot, pitch_c_ddot, f2_nom, ce0_nom, dt)
            
            # =================================================================
            # 🔥 致命故障注入：70秒 ~ 130秒，舵面效率跌至 45%
            # =================================================================
            if 190.0 <= sim_time <= 210.0:
                d_flap_physical = u_total * 0.45
            else:
                d_flap_physical = u_total

            sim.step(dt, {'d_flap_L': d_flap_physical, 'd_flap_R': d_flap_physical, 'd_ail_L': current_d_ail, 'd_ail_R': -current_d_ail, 'throttle': actual_throttle})
            sim_time += dt

            history_time.append(sim_time); history_alt.append(-sim.state[2]); history_vel.append(math.sqrt(sim.state[3]**2 + sim.state[4]**2 + sim.state[5]**2))
            history_pitch.append(current_pitch); history_pitch_c.append(pitch_c)
            history_throttle.append(actual_throttle); history_u_total.append(u_total); history_f_nn.append(f_nn)
            history_target_vel.append(current_target_vel)

        if step % 5 == 0: 
            print(f"Time: {sim_time:4.1f}s | 高度: {-sim.state[2]:6.1f}m | 目标速度: {current_target_vel:5.1f}m/s | 实际速度: {V:5.1f}m/s | Pitch: {current_pitch:5.2f}° | 物理油门: {actual_throttle:4.2f}")

    # === 绘图 ===
    plt.style.use('dark_background'); plt.rcParams['font.sans-serif'] = ['SimHei']; plt.rcParams['axes.unicode_minus'] = False   
    fig = plt.figure(figsize=(16, 12)); fig.suptitle('TECS 终局：多段阶梯变速与 55% 严重故障联合测试', fontsize=18, color='cyan')
    gs = GridSpec(3, 2, figure=fig)
    
    # 速度追踪图
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.plot(history_time, history_vel, color='cyan', linewidth=3, label='实际速度')
    ax1.plot(history_time, history_target_vel, color='white', linestyle='--', linewidth=2, label='动态目标速度')
    ax1.axvspan(70, 130, color='red', alpha=0.3, label='突发故障 (舵效仅剩 45%)')
    ax1.set_title('阶梯速度追踪 (考验多工况能量跨度)'); ax1.grid(True, alpha=0.2); ax1.legend()

    # 高度保持图
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.plot(history_time, history_alt, color='springgreen', linewidth=3, label='实际高度')
    ax2.axhline(TEST_TARGET_ALT, color='white', linestyle='--', linewidth=2, label=f'目标高度 ({TEST_TARGET_ALT}m)')
    ax2.axvspan(70, 130, color='red', alpha=0.3)
    ax2.set_title('定高表现 (无论怎么加减速，高度雷打不动)'); ax2.grid(True, alpha=0.2); ax2.legend()

    # 油门管理图
    ax3 = fig.add_subplot(gs[1, 0])
    ax3.plot(history_time, history_throttle, color='gold', linewidth=3, label='实际物理油门 (受涡轮迟滞约束)')
    ax3.axvspan(70, 130, color='red', alpha=0.3)
    ax3.set_title('能量调度 (加速推满，减速收干)'); ax3.grid(True, alpha=0.2); ax3.legend()

    # 姿态解耦图
    ax4 = fig.add_subplot(gs[1, 1])
    ax4.plot(history_time, history_pitch, color='hotpink', linewidth=3, label='实际俯仰角')
    ax4.plot(history_time, history_pitch_c, color='white', linestyle='-.', linewidth=2, label='平滑目标俯仰角')
    ax4.axvspan(70, 130, color='red', alpha=0.3)
    ax4.set_title('姿态分配 (加速压头防升，减速抬头刹车)'); ax4.grid(True, alpha=0.2); ax4.legend()

    # 内环 NSMC 与 故障补偿图
    ax5 = fig.add_subplot(gs[2, :])
    ax5.plot(history_time, history_u_total, color='dodgerblue', linewidth=2, label='NSMC 总舵面指令 (u_total)')
    ax5.plot(history_time, history_f_nn, color='gold', linewidth=2, label='RBF神经网络动态补偿 (f_nn)')
    ax5.axvspan(70, 130, color='red', alpha=0.3)
    ax5.set_title('神级底层：故障瞬间 RBF 黄线暴起护主，完美抵消 55% 舵效流失', fontsize=14)
    ax5.legend(); ax5.grid(True, alpha=0.2)

    plt.tight_layout(rect=[0, 0.03, 1, 0.95]); plt.show()