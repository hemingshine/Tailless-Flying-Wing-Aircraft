#coding=utf-8
import os
import pickle
import numpy as np
import math
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import warnings

warnings.filterwarnings('ignore')

# =================================================================
# 1. 气动与发动机数据库 (原汁原味)
# =================================================================
class HybridAeroDatabase:
    def __init__(self):
        self.raw_data = {} 
        self.models_db = {}
        
        self.control_cols = [
            '左襟翼偏角（°）', '右襟翼偏角（°）', 
            '左副翼偏角（°）', '右副翼偏角（°）', 
            '前扰流板偏角（°）', '后扰流板偏角（°）'
        ]
        self.state_cols = ['迎角（°）', '侧滑角（°）']
        self.output_cols = [
            '轴向力系数', '横向力系数', '法向力系数', 
            '滚转力矩系数', '俯仰力矩系数', '偏航力矩系数'
        ]

    def _load_from_pickle(self, pickle_path):
        with open(pickle_path, 'rb') as f:
            self.models_db = pickle.load(f)

    def get_body_axis_coeffs(self, mach, d_flap_L, d_flap_R, d_ail_L, d_ail_R, d_spoil_F, d_spoil_R, alpha, beta):
        if not self.models_db:
            raise ValueError("气动数据库为空，请先加载！")
            
        available_machs = list(self.models_db.keys())
        closest_mach = min(available_machs, key=lambda x: abs(x - mach))
        model_info = self.models_db[closest_mach]
        
        query_point = np.array([d_flap_L, d_flap_R, d_ail_L, d_ail_R, d_spoil_F, d_spoil_R, alpha, beta])
        
        if model_info['type'] == 'ND':
            q = query_point[model_info['active_dims']]
            res = model_info['interp'](q)
            res = res[0] if res.ndim > 1 else res
            if np.isnan(res).any():
                res = np.nan_to_num(res, nan=0.0)
        elif model_info['type'] == '1D':
            res = model_info['interp'](query_point[model_info['active_dims'][0]])
        else:
            res = model_info['val']
            
        return dict(zip(self.output_cols, res))

class EngineDatabase:
    def __init__(self):
        self.thrust_interpolator = None
        
    def load1(self, pickle_path="engine.pkl"):
        try:
            with open(pickle_path, 'rb') as f:
                self.thrust_interpolator = pickle.load(f)
        except:
            pass

    def get_thrust_newtons(self, alt, mach):
        if self.thrust_interpolator is None:
            return 7000.0 * 10.0 
        query_point = np.array([[alt, mach]])
        thrust_dan = self.thrust_interpolator(query_point)[0]
        return 0.0 if np.isnan(thrust_dan) else thrust_dan * 10.0

# =================================================================
# 2. 六自由度飞行模拟器 (带可控护盾的 RK4)
# =================================================================
class FlightSimulator6DOF:
    def __init__(self, aero_db, engine_db, params):
        self.aero_db = aero_db
        self.engine_db = engine_db
        self.S, self.b, self.c_bar, self.mass = params['S'], params['b'], params['c_bar'], params['mass']
        self.g = 9.80665
        self.Ixx, self.Iyy, self.Izz, self.Ixz = params['Ixx'], params['Iyy'], params['Izz'], params['Ixz']
        
        Gamma = self.Ixx * self.Izz - self.Ixz**2
        self.c1 = ((self.Iyy - self.Izz) * self.Izz - self.Ixz**2) / Gamma
        self.c2 = ((self.Ixx - self.Iyy + self.Izz) * self.Ixz) / Gamma
        self.c3 = self.Izz / Gamma
        self.c4 = self.Ixz / Gamma
        self.c5 = (self.Izz - self.Ixx) / self.Iyy
        self.c6 = self.Ixz / self.Iyy
        self.c7 = ((self.Ixx - self.Iyy) * self.Ixx + self.Ixz**2) / Gamma
        self.c8 = self.Ixx / Gamma
        
        self.state = np.zeros(12)
        self.SAFE_ALPHA_MIN, self.SAFE_ALPHA_MAX, self.SAFE_BETA_MAX = -10.0, 30.0, 10.0 

    def set_initial_state(self, h_m, V_mps, theta_deg, alpha_deg=0.0):
        self.state[2] = -h_m              
        alpha_rad = math.radians(alpha_deg)
        self.state[3] = V_mps * math.cos(alpha_rad) 
        self.state[5] = V_mps * math.sin(alpha_rad) 
        self.state[7] = math.radians(theta_deg)     
        
    def get_atmosphere(self, h):
        if h < 11000:
            T = 288.15 - 0.0065 * h
            p = 101325.0 * (T / 288.15) ** 5.2561
        else:
            T = 216.65
            p = 22632.1 * math.exp(-9.80665 * (h - 11000) / (287.05 * 216.65))
        return p / (287.05 * T), math.sqrt(1.4 * 287.05 * T)

    def get_derivatives(self, state, controls):
        pn, pe, pd, u, v, w, phi, theta, psi, p, q, r = state
        h, V = -pd, max(math.sqrt(u**2 + v**2 + w**2), 1.0)
        
        alpha_deg = math.degrees(math.atan2(w, u))
        beta_deg = math.degrees(math.asin(v / V))
        
        query_alpha = np.clip(alpha_deg, self.SAFE_ALPHA_MIN, self.SAFE_ALPHA_MAX)
        query_beta = np.clip(beta_deg, -self.SAFE_BETA_MAX, self.SAFE_BETA_MAX)
        
        rho, a = self.get_atmosphere(h)
        Mach, q_dyn = V / a, 0.5 * rho * V**2
        
        try:
            coeffs = self.aero_db.get_body_axis_coeffs(
                mach=Mach,
                d_flap_L=controls.get('d_flap_L', 0.0), d_flap_R=controls.get('d_flap_R', 0.0),
                d_ail_L=controls.get('d_ail_L', 0.0), d_ail_R=controls.get('d_ail_R', 0.0),
                d_spoil_F=0.0, d_spoil_R=0.0, 
                alpha=query_alpha, beta=query_beta
            )
            if abs(coeffs.get('法向力系数', 0.0)) < 1e-5 and abs(coeffs.get('轴向力系数', 0.0)) < 1e-5:
                raise ValueError("真空陷阱")
                
            Fx = -coeffs['轴向力系数'] * q_dyn * self.S
            Fy = coeffs['横向力系数'] * q_dyn * self.S
            Fz = -coeffs['法向力系数'] * q_dyn * self.S
            L_aero = coeffs['滚转力矩系数'] * q_dyn * self.S * self.b
            M_aero = coeffs['俯仰力矩系数'] * q_dyn * self.S * self.c_bar
            N_aero = coeffs['偏航力矩系数'] * q_dyn * self.S * self.b
            
        except Exception:
            # 【重大修复】：接通操纵杆的合成气动模型！
            d_sym = (controls.get('d_flap_L', 0.0) + controls.get('d_flap_R', 0.0)) / 2.0
            d_diff = (controls.get('d_ail_L', 0.0) - controls.get('d_ail_R', 0.0)) / 2.0
            
            Cz_syn = 0.05 + 0.06 * query_alpha + 0.005 * d_sym
            Cx_syn = 0.01 + 0.002 * query_alpha**2 + 0.0005 * abs(d_sym)
            Fx, Fy, Fz = -Cx_syn * q_dyn * self.S, 0.0, -Cz_syn * q_dyn * self.S
            
            L_aero = (-0.1 * p + 0.005 * d_diff) * q_dyn * self.S * self.b
            M_aero = (-0.05 * (query_alpha - 0.7) - 0.01 * d_sym) * q_dyn * self.S * self.c_bar
            N_aero = -0.1 * r * q_dyn * self.S * self.b

        Fx += self.engine_db.get_thrust_newtons(h, Mach) * 5.0 
        
        gx, gy, gz = -self.g * math.sin(theta), self.g * math.sin(phi) * math.cos(theta), self.g * math.cos(phi) * math.cos(theta)
        
        dot_u, dot_v, dot_w = (Fx/self.mass)+gx-q*w+r*v, (Fy/self.mass)+gy-r*u+p*w, (Fz/self.mass)+gz-p*v+q*u
        
        dot_p = self.c1 * r * q + self.c2 * p * q + self.c3 * L_aero + self.c4 * N_aero
        dot_q = self.c5 * p * r - self.c6 * (p**2 - r**2) + M_aero / self.Iyy
        dot_r = self.c7 * p * q - self.c2 * q * r + self.c4 * L_aero + self.c8 * N_aero
        
        dot_phi = p + math.tan(theta) * (q*math.sin(phi) + r*math.cos(phi))
        dot_theta = q*math.cos(phi) - r*math.sin(phi)
        dot_psi = (q*math.sin(phi) + r*math.cos(phi)) / math.cos(theta)
        
        dot_pn = u*math.cos(theta)*math.cos(psi) + v*(math.sin(phi)*math.sin(theta)*math.cos(psi) - math.cos(phi)*math.sin(psi)) + w*(math.cos(phi)*math.sin(theta)*math.cos(psi) + math.sin(phi)*math.sin(psi))
        dot_pe = u*math.cos(theta)*math.sin(psi) + v*(math.sin(phi)*math.sin(theta)*math.sin(psi) + math.cos(phi)*math.cos(psi)) + w*(math.cos(phi)*math.sin(theta)*math.sin(psi) - math.sin(phi)*math.cos(psi))
        dot_pd = -u*math.sin(theta) + v*math.sin(phi)*math.cos(theta) + w*math.cos(phi)*math.cos(theta)
        
        return np.array([dot_pn, dot_pe, dot_pd, dot_u, dot_v, dot_w, dot_phi, dot_theta, dot_psi, dot_p, dot_q, dot_r])

    def step(self, dt, controls):
        y0 = self.state.copy()
        k1 = self.get_derivatives(y0, controls)
        k2 = self.get_derivatives(y0 + 0.5 * dt * k1, controls)
        k3 = self.get_derivatives(y0 + 0.5 * dt * k2, controls)
        k4 = self.get_derivatives(y0 + dt * k3, controls)
        self.state = y0 + (dt / 6.0) * (k1 + 2*k2 + 2*k3 + k4)
        
        u, v, w = self.state[3], self.state[4], self.state[5]
        return {
            "Altitude": -self.state[2], "Velocity": math.sqrt(u**2 + v**2 + w**2),
            "Alpha": math.degrees(math.atan2(w, u)), "Pitch": math.degrees(self.state[7]),
            "Roll": math.degrees(self.state[6])
        }

# =================================================================
# 3. 绘图函数 (原汁原味)
# =================================================================
def plot1(time_arr, alt_arr, vel_arr, alpha_arr, pitch_arr, pn_arr, pe_arr):
    plt.style.use('dark_background') 
    plt.rcParams['font.sans-serif'] = ['SimHei'] 
    plt.rcParams['axes.unicode_minus'] = False   
    fig = plt.figure(figsize=(15, 10))
    fig.suptitle('6-DOF 飞行轨迹与姿态遥测数据', fontsize=20, fontweight='bold', color='cyan')

    ax1 = fig.add_subplot(1, 2, 1, projection='3d')
    ax1.plot(pe_arr, pn_arr, alt_arr, color='cyan', linewidth=2.5)
    ax1.scatter(pe_arr[0], pn_arr[0], alt_arr[0], color='lime', s=100, label='起点')
    ax1.scatter(pe_arr[-1], pn_arr[-1], alt_arr[-1], color='red', s=100, label='终点')
    ax1.set_title('3D 空间航迹 (NED 坐标)'); ax1.legend()

    ax2 = fig.add_subplot(3, 2, 2)
    ax2.plot(time_arr, alt_arr, color='springgreen', linewidth=2)
    ax2.set_title('高度 (Altitude)'); ax2.grid(True, alpha=0.3)

    ax3 = fig.add_subplot(3, 2, 4)
    ax3.plot(time_arr, vel_arr, color='gold', linewidth=2)
    ax3.set_title('速度 (Velocity)'); ax3.grid(True, alpha=0.3)

    ax4 = fig.add_subplot(3, 2, 6)
    ax4.plot(time_arr, pitch_arr, color='hotpink', linewidth=2, label='俯仰角')
    ax4_twin = ax4.twinx()
    ax4_twin.plot(time_arr, alpha_arr, color='dodgerblue', linewidth=2, linestyle='-.', label='迎角')
    ax4.set_title('姿态 (Attitude)'); ax4.grid(True, alpha=0.3)

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()

# =================================================================
# 4. 完全版 PID (消除 Bug 版)
# =================================================================
# if __name__ == "__main__":
#     aircraft_params = {
#         'S': 88.58, 'b': 18.9, 'c_bar': 4.6, 'mass': 14000,
#         'Ixx': 313220.6, 'Iyy': 273435.3, 'Izz': 392903.9, 'Ixz': 0.0
#     }

#     flight_db = HybridAeroDatabase()
#     try: flight_db._load_from_pickle('X47B.pkl')
#     except: pass
#     engine_db = EngineDatabase()
#     engine_db.load1('engine.pkl')

#     sim = FlightSimulator6DOF(flight_db, engine_db, aircraft_params)
#     sim.set_initial_state(h_m=2000.0, V_mps=250.0, theta_deg=0.7, alpha_deg=0.7)
    
#     dt = 0.02 
#     target_altitude = 2500.0
    
#     history_time, history_alt, history_vel = [], [], []
#     history_alpha, history_pitch, history_pn, history_pe = [], [], [], []
#     sim_time = 0.0
    
#     integral_h, integral_theta = 0.0, 0.0
    
#     print(f"开始【真正的终极 PID】定高测试：目标高度 {target_altitude}m ...")
    
#     for step in range(int(120 / dt)):
#         u, v, w = sim.state[3], sim.state[4], sim.state[5]
#         phi, theta = sim.state[6], sim.state[7]
        
#         current_h = -sim.state[2]
#         current_vz = u*math.sin(theta) - v*math.sin(phi)*math.cos(theta) - w*math.cos(phi)*math.cos(theta)
#         current_pitch, current_roll = math.degrees(theta), math.degrees(phi)
#         current_q, current_p = math.degrees(sim.state[10]), math.degrees(sim.state[9])
        
#         # 1. 高度外环
#         err_h = target_altitude - current_h
#         integral_h = np.clip(integral_h + err_h * dt, -1000.0, 1000.0) 
        
#         target_pitch = 0.7 + (0.01 * err_h) - (0.05 * current_vz) + (0.001 * integral_h)
#         target_pitch = np.clip(target_pitch, -1.0, 4.0) 
        
#         # 2. 俯仰内环
#         err_theta = target_pitch - current_pitch
#         integral_theta = np.clip(integral_theta + err_theta * dt, -20.0, 20.0)
        
#         # 【重大修复】：D项必须是加号(+)，完美阻尼！
#         d_flap = -1.5 * err_theta + 0.8 * current_q - 0.2 * integral_theta
#         d_flap = np.clip(d_flap, -10.0, 10.0)
        
#         # 3. 横滚增稳环
#         d_ail = 1.0 * (0.0 - current_roll) - 0.5 * current_p
#         d_ail = np.clip(d_ail, -10.0, 10.0)
        
#         controls = {'d_flap_L': d_flap, 'd_flap_R': d_flap, 'd_ail_L': d_ail, 'd_ail_R': -d_ail}
#         result = sim.step(dt, controls)
#         sim_time += dt

#         history_time.append(sim_time)
#         history_alt.append(result['Altitude'])
#         history_vel.append(result['Velocity'])
#         history_alpha.append(result['Alpha'])
#         history_pitch.append(result['Pitch'])
#         history_pn.append(sim.state[0])
#         history_pe.append(sim.state[1])
        
#         if step % 500 == 0: 
#             print(f"Time: {sim_time:.1f}s | 高度: {result['Altitude']:.1f}m | 速度: {result['Velocity']:.1f}m/s | 目标俯仰: {target_pitch:.1f}° | 实际俯仰: {result['Pitch']:.1f}° | 舵面: {d_flap:.1f}°")

#     print("\n✅ 测试完成！正在生成图表...")
#     plot1(history_time, history_alt, history_vel, history_alpha, history_pitch, history_pn, history_pe)


# =================================================================
# 4. 迎角 (Alpha) 阶跃跟踪测试任务
# =================================================================
if __name__ == "__main__":
    aircraft_params = {
        'S': 88.58, 'b': 18.9, 'c_bar': 4.6, 'mass': 14000,
        'Ixx': 313220.6, 'Iyy': 273435.3, 'Izz': 392903.9, 'Ixz': 0.0
    }

    flight_db = HybridAeroDatabase()
    try: flight_db._load_from_pickle('X47B.pkl')
    except: pass
    engine_db = EngineDatabase()
    engine_db.load1('engine.pkl')

    sim = FlightSimulator6DOF(flight_db, engine_db, aircraft_params)
    sim.set_initial_state(h_m=2000.0, V_mps=250.0, theta_deg=0.7, alpha_deg=0.7)
    
    dt = 0.02 
    
    history_time, history_alt, history_vel = [], [], []
    history_alpha, history_pitch, history_pn, history_pe = [], [], [], []
    history_target_alpha = [] # 记录目标迎角用于画图对比
    
    sim_time = 0.0
    integral_alpha = 0.0 # 迎角环积分器
    
    print("开始【迎角 (Alpha) 跟踪】测试任务 ...")
    
    for step in range(int(100 / dt)): # 模拟 100 秒
        u, v, w = sim.state[3], sim.state[4], sim.state[5]
        phi, theta = sim.state[6], sim.state[7]
        
        current_alpha = math.degrees(math.atan2(w, u))
        current_roll = math.degrees(phi)
        current_q = math.degrees(sim.state[10]) 
        current_p = math.degrees(sim.state[9])  
        
        # ================= 设定阶跃目标迎角 =================
        if sim_time < 20.0:
            target_alpha = 0.7
        elif sim_time < 60.0:
            target_alpha = 4.0
        else:
            target_alpha = 2.0
            
        history_target_alpha.append(target_alpha)
        
        # ================= 迎角跟踪 PID 核心 =================
        err_alpha = target_alpha - current_alpha
        
        # 积分器抗饱和
        integral_alpha = np.clip(integral_alpha + err_alpha * dt, -30.0, 30.0)
        
        # 迎角 PID 控制律
        # P(比例)拉近误差，I(积分)消除静差，D(使用俯仰角速度 q)提供阻尼防止震荡
        Kp_alpha = 2.0
        Ki_alpha = 0.5
        Kd_q = 0.8
        
        # 飞翼后缘舵面负偏角 -> 抬头 -> Alpha增加
        # 所以 err_alpha > 0 时，需要输出负舵面
        d_flap = -Kp_alpha * err_alpha - Ki_alpha * integral_alpha + Kd_q * current_q
        d_flap = np.clip(d_flap, -20.0, 20.0)
        
        # ================= 横滚增稳环 =================
        d_ail = 1.0 * (0.0 - current_roll) - 0.5 * current_p
        d_ail = np.clip(d_ail, -10.0, 10.0)
        
        controls = {'d_flap_L': d_flap, 'd_flap_R': d_flap, 'd_ail_L': d_ail, 'd_ail_R': -d_ail}
        result = sim.step(dt, controls)
        sim_time += dt

        history_time.append(sim_time)
        history_alt.append(result['Altitude'])
        history_vel.append(result['Velocity'])
        history_alpha.append(result['Alpha'])
        history_pitch.append(result['Pitch'])
        history_pn.append(sim.state[0])
        history_pe.append(sim.state[1])
        
        if step % 500 == 0: 
            print(f"Time: {sim_time:.1f}s | 高度: {result['Altitude']:.1f}m | 速度: {result['Velocity']:.1f}m/s | 目标迎角: {target_alpha:.1f}° | 实际迎角: {result['Alpha']:.2f}° | 舵面: {d_flap:.1f}°")

    print("\n✅ 测试完成！请查看生成的图表，重点关注迎角跟踪效果。")
    
    # 覆盖原画图函数中的 Alpha 曲线，把目标迎角也画上去对比
    plt.style.use('dark_background') 
    plt.rcParams['font.sans-serif'] = ['SimHei'] 
    plt.rcParams['axes.unicode_minus'] = False   
    fig = plt.figure(figsize=(15, 10))
    fig.suptitle('迎角跟踪任务遥测数据', fontsize=20, fontweight='bold', color='cyan')

    ax1 = fig.add_subplot(2, 2, 1)
    ax1.plot(history_time, history_alt, color='springgreen', linewidth=2)
    ax1.set_title('高度 (Altitude)'); ax1.grid(True, alpha=0.3)

    ax2 = fig.add_subplot(2, 2, 2)
    ax2.plot(history_time, history_vel, color='gold', linewidth=2)
    ax2.set_title('速度 (Velocity)'); ax2.grid(True, alpha=0.3)

    ax3 = fig.add_subplot(2, 2, 3)
    ax3.plot(history_time, history_pitch, color='hotpink', linewidth=2)
    ax3.set_title('俯仰角 (Pitch)'); ax3.grid(True, alpha=0.3)

    ax4 = fig.add_subplot(2, 2, 4)
    ax4.plot(history_time, history_alpha, color='dodgerblue', linewidth=2, label='实际迎角 (Alpha)')
    ax4.plot(history_time, history_target_alpha, color='red', linewidth=2, linestyle='--', label='目标迎角 (Target)')
    ax4.set_title('迎角跟踪性能 (Alpha Tracking)'); ax4.legend(); ax4.grid(True, alpha=0.3)

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()