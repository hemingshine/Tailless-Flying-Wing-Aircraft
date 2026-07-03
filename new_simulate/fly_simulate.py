#coding=utf-8
import os
import pickle
import numpy as np
import math
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import warnings
from scipy.interpolate import LinearNDInterpolator, interp1d

warnings.filterwarnings('ignore')

# =================================================================
# 1. 气动与发动机数据库 (已升级为高维多舵面插值版)
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
        self.input_cols = self.control_cols + self.state_cols
        
        self.output_cols = [
            '轴向力系数', '横向力系数', '法向力系数', 
            '滚转力矩系数', '俯仰力矩系数', '偏航力矩系数'
        ]

    def _load_from_pickle(self, pickle_path):
        with open(pickle_path, 'rb') as f:
            self.models_db = pickle.load(f)
        print(f"气动数据库 '{pickle_path}' 加载成功！\n" + "-"*40)

    def get_body_axis_coeffs(self, mach, d_flap_L, d_flap_R, d_ail_L, d_ail_R, d_spoil_F, d_spoil_R, alpha, beta):
        """核心查询函数：输入 8 个飞行状态参数，返回 6 个气动系数"""
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
            q = query_point[model_info['active_dims'][0]]
            res = model_info['interp'](q)
            
        else:
            res = model_info['val']
            
        return dict(zip(self.output_cols, res))


class EngineDatabase:
    def __init__(self):
        self.thrust_interpolator = None
        
    def load1(self, pickle_path="engine_cache.pkl"):
        print(f"检测到发动机缓存文件 '{pickle_path}'，正在秒速加载...")
        with open(pickle_path, 'rb') as f:
            self.thrust_interpolator = pickle.load(f)
        print("发动机数据库加载成功！\n" + "-"*40)

    def get_thrust_newtons(self, alt, mach):
        if self.thrust_interpolator is None:
            raise RuntimeError("请先加载发动机数据库！")
            
        query_point = np.array([[alt, mach]])
        thrust_dan = self.thrust_interpolator(query_point)[0]
        
        if np.isnan(thrust_dan):
            thrust_dan = 0.0 
            
        return thrust_dan * 10.0


# =================================================================
# 2. 六自由度飞行模拟器 (接入多维舵面控制)
# =================================================================
class FlightSimulator6DOF:
    def __init__(self, aero_db, engine_db, global_params):
        self.aero_db = aero_db
        self.engine_db = engine_db
        
        self.S = global_params['S']
        self.b = global_params['b']
        self.c_bar = global_params['c_bar']
        self.mass = global_params['mass']
        
        self.Ixx = global_params['Ixx']
        self.Iyy = global_params['Iyy']
        self.Izz = global_params['Izz']
        self.Ixz = global_params['Ixz']
        
        self.Gamma = self.Ixx * self.Izz - self.Ixz**2
        self.c1 = ((self.Iyy - self.Izz) * self.Izz - self.Ixz**2) / self.Gamma
        self.c2 = ((self.Ixx - self.Iyy + self.Izz) * self.Ixz) / self.Gamma
        self.c3 = self.Izz / self.Gamma
        self.c4 = self.Ixz / self.Gamma
        self.c5 = (self.Izz - self.Ixx) / self.Iyy
        self.c6 = self.Ixz / self.Iyy
        self.c7 = ((self.Ixx - self.Iyy) * self.Ixx + self.Ixz**2) / self.Gamma
        self.c8 = self.Ixx / self.Gamma
        
        self.g = 9.80665
        self.state = np.zeros(12)

    def set_initial_state(self, h_m, V_mps, theta_deg):
        self.state[2] = -h_m              
        self.state[3] = V_mps             
        self.state[7] = math.radians(theta_deg) 
        
    def isa_atmosphere(self, altitude_m):
        T0, p0, rho0, L, R, gamma = 288.15, 101325.0, 1.225, 0.0065, 287.05, 1.4
        if altitude_m < 11000:
            T = T0 - L * altitude_m
            p = p0 * (T / T0) ** (self.g * 0.0289644 / (8.3144598 * L))
            rho = p / (R * T)
        else:
            T = 216.65
            rho = 0.36391 * math.exp(-(altitude_m - 11000) / 6341.6)
        a = math.sqrt(gamma * R * T)
        return rho, a

    def get_derivatives(self, state, controls):
        """核心物理引擎：使用 controls 字典传入各舵面角度"""
        pn, pe, pd, u, v, w, phi, theta, psi, p, q, r = state
        
        V = math.sqrt(u**2 + v**2 + w**2)
        if V == 0: V = 0.001 
        
        alpha_rad = math.atan2(w, u)
        beta_rad = math.asin(v / V)
        alpha_deg = math.degrees(alpha_rad)
        beta_deg = math.degrees(beta_rad)
        
        h = -pd
        rho, a = self.isa_atmosphere(h)
        Mach = V / a
        q_dyn = 0.5 * rho * V**2
        
        # 调用 8 维气动数据库接口 (利用controls字典安全取值，缺省为0)
        coeffs = self.aero_db.get_body_axis_coeffs(
            mach=Mach,
            d_flap_L=controls.get('d_flap_L', 0.0),
            d_flap_R=controls.get('d_flap_R', 0.0),
            d_ail_L=controls.get('d_ail_L', 0.0),
            d_ail_R=controls.get('d_ail_R', 0.0),
            d_spoil_F=controls.get('d_spoil_F', 0.0),
            d_spoil_R=controls.get('d_spoil_R', 0.0),
            alpha=alpha_deg,
            beta=beta_deg
        )
        
        thrust = self.engine_db.get_thrust_newtons(h, Mach)
        
        Fx = thrust - coeffs['轴向力系数'] * q_dyn * self.S
        Fy = coeffs['横向力系数'] * q_dyn * self.S
        Fz = - coeffs['法向力系数'] * q_dyn * self.S
        
        L_aero = coeffs['滚转力矩系数'] * q_dyn * self.S * self.b
        M_aero = coeffs['俯仰力矩系数'] * q_dyn * self.S * self.c_bar
        N_aero = coeffs['偏航力矩系数'] * q_dyn * self.S * self.b
        M_aero += -q * 200000.0 # 简易阻尼
        
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

    def step_rk4(self, dt, controls):
        """四阶龙格-库塔 (RK4) 积分步进器 (接收controls)"""
        y0 = self.state.copy()
        
        k1 = self.get_derivatives(y0, controls)
        k2 = self.get_derivatives(y0 + 0.5 * dt * k1, controls)
        k3 = self.get_derivatives(y0 + 0.5 * dt * k2, controls)
        k4 = self.get_derivatives(y0 + dt * k3, controls)
        
        self.state = y0 + (dt / 6.0) * (k1 + 2*k2 + 2*k3 + k4)
        
        u, v, w = self.state[3], self.state[4], self.state[5]
        V = math.sqrt(u**2 + v**2 + w**2)
        alpha = math.degrees(math.atan2(w, u))
        h = -self.state[2]
        
        return {"Time": 0, "Altitude": h, "Velocity": V, "Alpha": alpha, "Pitch": math.degrees(self.state[7])}


# =================================================================
# 3. 绘图函数 (参数化解耦)
# =================================================================
def plot1(time_arr, alt_arr, vel_arr, alpha_arr, pitch_arr, pn_arr, pe_arr):
    plt.style.use('dark_background') 
    plt.rcParams['font.sans-serif'] = ['SimHei'] 
    plt.rcParams['axes.unicode_minus'] = False   
    
    fig = plt.figure(figsize=(15, 10))
    fig.suptitle('6-DOF 飞行轨迹与姿态遥测数据', fontsize=20, fontweight='bold', color='cyan')

    ax1 = fig.add_subplot(1, 2, 1, projection='3d')
    ax1.plot(pe_arr, pn_arr, alt_arr, color='cyan', linewidth=2.5, label='飞行轨迹')
    ax1.scatter(pe_arr[0], pn_arr[0], alt_arr[0], color='lime', s=100, label='起点 (Takeoff)', zorder=5)
    ax1.scatter(pe_arr[-1], pn_arr[-1], alt_arr[-1], color='red', s=100, label='终点 (Current)', zorder=5)
    ax1.set_title('3D 空间航迹 (NED 坐标)', fontsize=14)
    ax1.set_xlabel('东向位置 East (m)')
    ax1.set_ylabel('北向位置 North (m)')
    ax1.set_zlabel('飞行高度 Altitude (m)')
    ax1.legend()
    ax1.view_init(elev=25, azim=-45) 

    ax2 = fig.add_subplot(3, 2, 2)
    ax2.plot(time_arr, alt_arr, color='springgreen', linewidth=2)
    ax2.set_title('高度剖面 (Altitude Profile)', fontsize=12)
    ax2.set_ylabel('高度 (m)')
    ax2.grid(True, linestyle='--', alpha=0.3)

    ax3 = fig.add_subplot(3, 2, 4)
    ax3.plot(time_arr, vel_arr, color='gold', linewidth=2)
    ax3.set_title('速度剖面 (Velocity Profile)', fontsize=12)
    ax3.set_ylabel('速度 (m/s)')
    ax3.grid(True, linestyle='--', alpha=0.3)

    ax4 = fig.add_subplot(3, 2, 6)
    line1 = ax4.plot(time_arr, pitch_arr, color='hotpink', linewidth=2, label='俯仰角 (Pitch)')
    ax4.set_xlabel('时间 (s)')
    ax4.set_ylabel('俯仰角 (°)', color='hotpink')
    ax4.tick_params(axis='y', labelcolor='hotpink')
    
    ax4_twin = ax4.twinx()
    line2 = ax4_twin.plot(time_arr, alpha_arr, color='dodgerblue', linewidth=2, linestyle='-.', label='迎角 (Alpha)')
    ax4_twin.set_ylabel('迎角 (°)', color='dodgerblue')
    ax4_twin.tick_params(axis='y', labelcolor='dodgerblue')
    
    ax4.set_title('纵向气动姿态 (Longitudinal Attitude)', fontsize=12)
    ax4.grid(True, linestyle='--', alpha=0.3)

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()


# ================= 测试运行用例 =================
# if __name__ == "__main__":
    
    # aircraft_params = {
    #     'S': 88.58, 'b': 18.9, 'c_bar': 4.6, 'mass': 14000,
    #     'Ixx': 313220.6, 'Iyy': 273435.3, 'Izz': 392903.9, 'Ixz': 0.0
    # }

    # flight_db = HybridAeroDatabase()
    # flight_db._load_from_pickle(pickle_path='X47B.pkl')
    
    # engine_db = EngineDatabase()
    # # 你的原代码里写的是 engine.pkl，请确保此文件名跟你上一步生成的名字一致（上一步默认是 engine_cache.pkl）
    # engine_db.load1('engine.pkl')

    # sim = FlightSimulator6DOF(flight_db, engine_db, aircraft_params)
    
    # sim.set_initial_state(h_m=2000.0, V_mps=250.0, theta_deg=0)
    
    # history_time, history_alt, history_vel = [], [], []
    # history_alpha, history_pitch, history_pn, history_pe = [], [], [], []

    # dt = 0.02 
    # sim_time = 0.0
    
    # print("开始飞行模拟...")
    # for step in range(1000): 
        
    #     # =================================================================
    #     # 【核心控制区】：根据你给出的约束表设置各舵面偏角
    #     # 注意保持在允许范围内，如果 Right 舵面没有出现在独立约束表里，
    #     # 说明它与 Left 舵面是线性耦合的，这里为了严谨我们赋相同的值。
    #     # =================================================================
    #     current_controls = {
    #         'd_flap_L': -5.0,   # 允许范围: [-30, 30]
    #         'd_flap_R': -5.0,   # 跟随左侧
    #         'd_ail_L': 0.0,     # 允许范围: [-10, 20]
    #         'd_ail_R': 0.0,     # 跟随左侧
    #         'd_spoil_F': -5.0,  # 允许范围: [-25, 0]
    #         'd_spoil_R': 5.0    # 允许范围: [0, 25]
    #     }
        
    #     # RK4 接收你的动态多维控制
    #     result = sim.step_rk4(dt, current_controls)
    #     sim_time += dt

    #     history_time.append(sim_time)
    #     history_alt.append(result['Altitude'])
    #     history_vel.append(result['Velocity'])
    #     history_alpha.append(result['Alpha'])
    #     history_pitch.append(result['Pitch'])
    #     history_pn.append(sim.state[0]) 
    #     history_pe.append(sim.state[1]) 
        
    #     if step % 50 == 0:
    #         print(f"Time: {sim_time:.2f}s | 高度: {result['Altitude']:.1f}m | 速度: {result['Velocity']:.1f}m/s | 迎角: {result['Alpha']:.2f}° | 俯仰角: {result['Pitch']:.2f}°")
            
    # # 将列表传参给图表函数
    # plot1(history_time, history_alt, history_vel, history_alpha, history_pitch, history_pn, history_pe)

    # =================================================================

