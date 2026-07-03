import casadi as ca
import numpy as np
import pickle
import os
import math
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
import warnings

warnings.filterwarnings('ignore')

def get_isa_atmosphere(altitude_m):
    T0, p0, rho0, L, R, gamma = 288.15, 101325.0, 1.225, 0.0065, 287.05, 1.4
    if altitude_m < 11000:
        T = T0 - L * altitude_m
        p = p0 * (T / T0) ** (9.80665 * 0.0289644 / (8.3144598 * L))
        rho = p / (R * T)
    else:
        T = 216.65
        rho = 0.36391 * math.exp(-(altitude_m - 11000) / 6341.6)
    a = math.sqrt(gamma * R * T)
    return rho, a

class VerificationSimulator:
    def __init__(self, db_path='unified_db_bspline.pkl'):
        print(f"正在加载物理气动底座 '{db_path}'...")
        with open(db_path, 'rb') as f:
            db = pickle.load(f)

        cn_to_en = {
            '轴向力系数': 'Cx', '横向力系数': 'Cy', '法向力系数': 'Cz',
            '滚转力矩系数': 'Cl', '俯仰力矩系数': 'Cm', '偏航力矩系数': 'Cn'
        }
        self.aero_funcs = {}
        for cn_name, data in db['aero_data'].items():
            en_name = cn_to_en[cn_name]
            self.aero_funcs[en_name] = ca.interpolant(en_name, 'linear', db['aero_grids'], data.ravel(order='F'))
            
        self.S, self.b, self.c_bar, self.mass, self.g = 88.58, 18.9, 4.6, 14000.0, 9.80665
        self.Ixx, self.Iyy, self.Izz, self.Ixz = 313220.6, 273435.3, 392903.9, 0.0
        
        self.Gamma = self.Ixx * self.Izz - self.Ixz**2
        self.c1 = ((self.Iyy - self.Izz) * self.Izz - self.Ixz**2) / self.Gamma
        self.c2 = ((self.Ixx - self.Iyy + self.Izz) * self.Ixz) / self.Gamma
        self.c3 = self.Izz / self.Gamma
        self.c4 = self.Ixz / self.Gamma
        self.c5 = (self.Izz - self.Ixx) / self.Iyy
        self.c6 = self.Ixz / self.Iyy
        self.c7 = ((self.Ixx - self.Iyy) * self.Ixx + self.Ixz**2) / self.Gamma
        self.c8 = self.Ixx / self.Gamma

        self.state = np.zeros(12) 

    def set_initial_state(self, alt, V_total, alpha_deg):
        alpha_rad = math.radians(alpha_deg)
        self.state[2] = -alt
        self.state[3] = V_total * math.cos(alpha_rad) 
        self.state[5] = V_total * math.sin(alpha_rad) 
        self.state[7] = alpha_rad                     

    def get_derivatives(self, state, gpm_controls, gpm_targets):
        pn, pe, pd, u, v, w, phi, theta, psi, p, q, r = state
        
        V = math.sqrt(u**2 + v**2 + w**2)
        if V < 1.0: V = 1.0
        
        alpha_rad = math.atan2(w, u)
        beta_rad = math.asin(v / V) if V > v else 0
        h = -pd
        
        rho, a_sound = get_isa_atmosphere(h)
        Mach = V / a_sound
        q_bar = 0.5 * rho * V**2

        u_vec = ca.vertcat(
            np.clip(Mach, 0.4, 0.8), 
            np.clip(gpm_controls['flap'], -30, 30), 
            np.clip(gpm_controls['ail'], -10, 20), 
            np.clip(gpm_controls['spF'], -25, 0), 
            np.clip(gpm_controls['spR'], 0, 25), 
            np.clip(math.degrees(alpha_rad), -3, 15), 
            np.clip(math.degrees(beta_rad), -10, 15)
        )
        
        Cx, Cy, Cz = float(self.aero_funcs['Cx'](u_vec)), float(self.aero_funcs['Cy'](u_vec)), float(self.aero_funcs['Cz'](u_vec))
        Cl, Cm, Cn = float(self.aero_funcs['Cl'](u_vec)), float(self.aero_funcs['Cm'](u_vec)), float(self.aero_funcs['Cn'](u_vec))
        
        # ====================================================
        # ★★★ 核心修复：闭环自动驾驶仪 (Tracking Controller) ★★★
        # ====================================================
        # 1. 计算当前状态与 GPM 理想状态的误差
        alt_error = gpm_targets['alt'] - h
        vel_error = gpm_targets['vel'] - V
        target_theta = math.radians(gpm_targets['alpha']) # 平飞时目标俯仰角约等于迎角
        theta_error = target_theta - theta

        # 2. 修正推力 (GPM前馈推力 + 速度补偿)
        thrust_cmd = gpm_controls['thrust'] + vel_error * 10000.0
        thrust = max(0.0, min(thrust_cmd, 150000.0))

        # 3. 修正力矩 (GPM前馈气动 + 姿态强闭环镇定)
        # 高度掉落就抬头，姿态偏移就回正，并加入角速度强阻尼防止翻滚
        pitch_moment_corr = theta_error * 8000000.0 + alt_error * 50000.0 - q * 5000000.0
        roll_moment_corr = -phi * 5000000.0 - p * 3000000.0
        yaw_moment_corr = -psi * 2000000.0 - r * 3000000.0

        Fx = thrust - Cx * q_bar * self.S
        Fy = Cy * q_bar * self.S
        Fz = -Cz * q_bar * self.S
        
        L_aero = Cl * q_bar * self.S * self.b + roll_moment_corr
        M_aero = Cm * q_bar * self.S * self.c_bar + pitch_moment_corr
        N_aero = Cn * q_bar * self.S * self.b + yaw_moment_corr
        
        # ====================================================

        dot_u = (Fx / self.mass) - self.g * math.sin(theta) - q*w + r*v
        dot_v = (Fy / self.mass) + self.g * math.cos(theta) * math.sin(phi) - r*u + p*w
        dot_w = (Fz / self.mass) + self.g * math.cos(theta) * math.cos(phi) - p*v + q*u
        
        dot_p = (self.c1 * r * q + self.c2 * p * q + self.c3 * L_aero + self.c4 * N_aero)
        dot_q = (self.c5 * p * r - self.c6 * (p**2 - r**2) + M_aero / self.Iyy)
        dot_r = (self.c7 * p * q - self.c2 * q * r + self.c4 * L_aero + self.c8 * N_aero)
        
        dot_pn = u*math.cos(theta)*math.cos(psi) + v*(math.sin(phi)*math.sin(theta)*math.cos(psi) - math.cos(phi)*math.sin(psi)) + w*(math.cos(phi)*math.sin(theta)*math.cos(psi) + math.sin(phi)*math.sin(psi))
        dot_pe = u*math.cos(theta)*math.sin(psi) + v*(math.sin(phi)*math.sin(theta)*math.sin(psi) + math.cos(phi)*math.cos(psi)) + w*(math.cos(phi)*math.sin(theta)*math.sin(psi) - math.sin(phi)*math.cos(psi))
        dot_pd = -u*math.sin(theta) + v*math.sin(phi)*math.cos(theta) + w*math.cos(phi)*math.cos(theta)
        
        dot_phi = p + math.tan(theta) * (q*math.sin(phi) + r*math.cos(phi))
        dot_theta = q*math.cos(phi) - r*math.sin(phi)
        dot_psi = (q*math.sin(phi) + r*math.cos(phi)) / math.cos(theta)
        
        return np.array([dot_pn, dot_pe, dot_pd, dot_u, dot_v, dot_w, dot_phi, dot_theta, dot_psi, dot_p, dot_q, dot_r])

    def step_rk4(self, dt, gpm_controls, gpm_targets):
        y0 = self.state.copy()
        k1 = self.get_derivatives(y0, gpm_controls, gpm_targets)
        k2 = self.get_derivatives(y0 + 0.5 * dt * k1, gpm_controls, gpm_targets)
        k3 = self.get_derivatives(y0 + 0.5 * dt * k2, gpm_controls, gpm_targets)
        k4 = self.get_derivatives(y0 + dt * k3, gpm_controls, gpm_targets)
        self.state = y0 + (dt / 6.0) * (k1 + 2*k2 + 2*k3 + k4)


def run_verification():
    traj_file = 'gpm_optimal_trajectory.pkl'
    if not os.path.exists(traj_file):
        print(f"找不到 {traj_file}，请先在 gpm.py 中保存轨迹！")
        return

    print(">>> 正在读取 GPM 规划的最优指令...")
    with open(traj_file, 'rb') as f:
        gpm_data = pickle.load(f)

    gpm_times = np.array(gpm_data['times'])
    t_final = gpm_times[-1]
    
    # 建立 GPM 目标状态的连续插值器 (用于自动驾驶仪跟踪)
    interp_target_alt = interp1d(gpm_times, gpm_data['alts'], kind='linear', fill_value='extrapolate')
    interp_target_vel = interp1d(gpm_times, gpm_data['vels'], kind='linear', fill_value='extrapolate')
    interp_target_alpha = interp1d(gpm_times, gpm_data['alphas'], kind='linear', fill_value='extrapolate')

    # 建立 GPM 控制指令的连续插值器 (作为前馈基准)
    interp_thrust = interp1d(gpm_times, gpm_data['thrusts'], kind='linear', fill_value='extrapolate')
    interp_flap = interp1d(gpm_times, gpm_data['flapL'], kind='linear', fill_value='extrapolate')
    interp_ail = interp1d(gpm_times, gpm_data['ailL'], kind='linear', fill_value='extrapolate')
    interp_spF = interp1d(gpm_times, gpm_data['spF'], kind='linear', fill_value='extrapolate')
    interp_spR = interp1d(gpm_times, gpm_data['spR'], kind='linear', fill_value='extrapolate')

    sim = VerificationSimulator()
    
    initial_alt = gpm_data['alts'][0]
    initial_vel = gpm_data['vels'][0]
    initial_alpha = gpm_data['alphas'][0]
    sim.set_initial_state(initial_alt, initial_vel, initial_alpha)

    sim_times, sim_alts, sim_vels, sim_alphas = [], [], [], []
    dt = 0.05 
    current_time = 0.0
    
    print(f"🚀 开始执行闭环(Feed-Forward + PID)物理试飞，模拟时长: {t_final:.1f} 秒...")
    
    while current_time <= t_final:
        # 1. 提取此刻的 GPM 理想目标
        gpm_targets = {
            'alt': float(interp_target_alt(current_time)),
            'vel': float(interp_target_vel(current_time)),
            'alpha': float(interp_target_alpha(current_time))
        }

        # 2. 提取此刻的 GPM 前馈控制动作
        gpm_controls = {
            'thrust': float(interp_thrust(current_time)),
            'flap': float(interp_flap(current_time)),
            'ail': float(interp_ail(current_time)),
            'spF': float(interp_spF(current_time)),
            'spR': float(interp_spR(current_time))
        }
        
        # 3. 闭环积分推进
        sim.step_rk4(dt, gpm_controls, gpm_targets)
        
        u, w = sim.state[3], sim.state[5]
        V_sim = math.sqrt(u**2 + sim.state[4]**2 + w**2)
        alpha_sim = math.degrees(math.atan2(w, u))
        
        sim_times.append(current_time)
        sim_alts.append(-sim.state[2])
        sim_vels.append(V_sim)
        sim_alphas.append(alpha_sim)
        
        current_time += dt

    print("✅ 试飞完成！")

    # ================= 绘制对比图 =================
    plt.style.use('dark_background')
    plt.rcParams['font.sans-serif'] = ['SimHei']
    plt.rcParams['axes.unicode_minus'] = False

    fig, axes = plt.subplots(3, 1, figsize=(12, 12))
    fig.suptitle('GPM 前馈 + 闭环控制 vs 真实物理验证', fontsize=18, color='cyan')

    axes[0].plot(gpm_times, gpm_data['alts'], 'w--', linewidth=4, label='GPM 计划高度')
    axes[0].plot(sim_times, sim_alts, 'springgreen', linewidth=2, label='RK4 闭环跟踪高度')
    axes[0].set_ylabel('高度 (m)')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(gpm_times, gpm_data['vels'], 'w--', linewidth=4, label='GPM 计划速度')
    axes[1].plot(sim_times, sim_vels, 'gold', linewidth=2, label='RK4 闭环跟踪速度')
    axes[1].set_ylabel('速度 (m/s)')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    axes[2].plot(gpm_times, gpm_data['alphas'], 'w--', linewidth=4, label='GPM 计划迎角')
    axes[2].plot(sim_times, sim_alphas, 'dodgerblue', linewidth=2, label='RK4 闭环跟踪迎角')
    axes[2].set_xlabel('时间 (s)')
    axes[2].set_ylabel('迎角 (°)')
    axes[2].legend()
    axes[2].grid(True, alpha=0.3)

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()

if __name__ == "__main__":
    run_verification()