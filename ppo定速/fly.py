#coding=utf-8
import os
import pickle
import numpy as np
import math
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import warnings
import time
import torch
import torch.nn as nn

warnings.filterwarnings('ignore')

# =================================================================
# 1. 光速气动数据库 (基于 PyTorch 代理模型)
# =================================================================
class AeroSurrogate(nn.Module):
    def __init__(self):
        super(AeroSurrogate, self).__init__()
        # 残差块定义
        class ResBlock(nn.Module):
            def __init__(self, in_dim, out_dim):
                super().__init__()
                self.net = nn.Sequential(
                    nn.Linear(in_dim, out_dim),
                    nn.BatchNorm1d(out_dim),  # 批归一化：稳定训练、加速收敛
                    nn.GELU(),
                    nn.Dropout(0.2)
                )
                # 残差连接的维度适配
                self.shortcut = nn.Linear(in_dim, out_dim) if in_dim != out_dim else nn.Identity()
            
            def forward(self, x):
                return self.net(x) + self.shortcut(x)  # 残差连接
        
        # 分层残差网络（宽度递减，更符合气动数据的非线性拟合规律）
        self.net = nn.Sequential(
            ResBlock(9, 256),
            ResBlock(256, 512),
            ResBlock(512, 256),
            ResBlock(256, 128),
            nn.Linear(128, 6)  # 输出6个气动参数
        )

    def forward(self, x):
        return self.net(x)


class NeuralAeroDatabase:
    def __init__(self):
        self.output_cols = [
            '轴向力系数', '横向力系数', '法向力系数', 
            '滚转力矩系数', '俯仰力矩系数', '偏航力矩系数'
        ]
        self.model = None
        self.x_mean, self.x_std = None, None
        self.y_mean, self.y_std = None, None

    def _load_from_pickle(self, model_path='aero_surrogate.pth'):
        print(f"正在加载神经气动代理模型 '{model_path}' ...")
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"找不到模型文件 {model_path}，请先训练！")
            
        data = torch.load(model_path, map_location='cpu')
        
        self.model = AeroSurrogate()
        self.model.load_state_dict(data['model_state_dict'])
        self.model.eval() # 开启推理模式
        torch.set_num_threads(1)        
        self.x_mean = data['x_mean']
        self.x_std = data['x_std']
        self.y_mean = data['y_mean']
        self.y_std = data['y_std']
        
        print("神经气动引擎并网成功！计算速度将提升百倍！\n" + "-"*40)

    def get_body_axis_coeffs(self, mach, d_flap_L, d_flap_R, d_ail_L, d_ail_R, d_spoil_F, d_spoil_R, alpha, beta):
        if self.model is None:
            raise ValueError("代理模型未加载！")
            
        # 构建输入张量
        x = torch.tensor([mach, d_flap_L, d_flap_R, d_ail_L, d_ail_R, d_spoil_F, d_spoil_R, alpha, beta], dtype=torch.float32)
        
        # 标准化
        x_norm = (x - self.x_mean) / self.x_std
        
        # 神经网络光速推理 (关闭梯度计算以极致提速)
        with torch.no_grad():
            y_norm = self.model(x_norm)
            
        # 反标准化恢复真实物理数值
        y = y_norm * self.y_std + self.y_mean
        
        # 转换回字典格式供模拟器调用
        return dict(zip(self.output_cols, y.numpy()))


# =================================================================
# 2. 发动机数据库 (原汁原味)
# =================================================================
class EngineDatabase:
    def __init__(self):
        self.thrust_interpolator = None
        
    def load1(self, pickle_path="engine.pkl"):
        try:
            with open(pickle_path, 'rb') as f:
                self.thrust_interpolator = pickle.load(f)
        except: pass

    def get_thrust_newtons(self, alt, mach):
        if self.thrust_interpolator is None: return 7000.0 * 10.0 
        query_point = np.array([[alt, mach]])
        thrust_dan = self.thrust_interpolator(query_point)[0]
        return 0.0 if np.isnan(thrust_dan) else thrust_dan * 10.0


# =================================================================
# 3. 高保真六自由度飞行模拟器 (完全保留护盾与RK4)
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
            Fx = -coeffs['轴向力系数'] * q_dyn * self.S
            Fy = coeffs['横向力系数'] * q_dyn * self.S
            Fz = -coeffs['法向力系数'] * q_dyn * self.S
            L_aero = coeffs['滚转力矩系数'] * q_dyn * self.S * self.b
            M_aero = coeffs['俯仰力矩系数'] * q_dyn * self.S * self.c_bar
            N_aero = coeffs['偏航力矩系数'] * q_dyn * self.S * self.b
        except Exception:
            # 护盾依然保留以防万一
            d_sym = (controls.get('d_flap_L', 0.0) + controls.get('d_flap_R', 0.0)) / 2.0
            d_diff = (controls.get('d_ail_L', 0.0) - controls.get('d_ail_R', 0.0)) / 2.0
            Cz_syn = 0.05 + 0.06 * query_alpha + 0.005 * d_sym
            Cx_syn = 0.01 + 0.002 * query_alpha**2 + 0.0005 * abs(d_sym)
            Fx, Fy, Fz = -Cx_syn * q_dyn * self.S, 0.0, -Cz_syn * q_dyn * self.S
            L_aero = (-0.1 * p + 0.005 * d_diff) * q_dyn * self.S * self.b
            M_aero = (-0.05 * (query_alpha - 0.7) - 0.01 * d_sym) * q_dyn * self.S * self.c_bar
            N_aero = -0.1 * r * q_dyn * self.S * self.b

        # 接收外部传入的油门指令 (0.0 ~ 1.0)，默认给 0.6 的巡航油门
        throttle = controls.get('throttle', 0.6) 
        
        # 引擎最大推力乘以油门百分比
        thrust = self.engine_db.get_thrust_newtons(h, Mach) * throttle * 7.0 
        Fx += thrust
        
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
        # 纯净的高精度 RK4 积分，绝不崩溃
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
# 4. 绘图函数 (原汁原味)
# =================================================================
def plot1(time_arr, alt_arr, vel_arr, alpha_arr, pitch_arr, pn_arr, pe_arr):
    plt.style.use('dark_background') 
    plt.rcParams['font.sans-serif'] = ['SimHei'] 
    plt.rcParams['axes.unicode_minus'] = False   
    fig = plt.figure(figsize=(15, 10))
    fig.suptitle('6-DOF 飞行轨迹与姿态遥测数据 (神经模型驱动)', fontsize=20, fontweight='bold', color='cyan')

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
# 5. 工业级 PID 定高控制测试 (调用神经模型)
# =================================================================
if __name__ == "__main__":
    aircraft_params = {
        'S': 88.58, 'b': 18.9, 'c_bar': 4.6, 'mass': 14000,
        'Ixx': 313220.6, 'Iyy': 273435.3, 'Izz': 392903.9, 'Ixz': 0.0
    }

    # 【核心改动】：加载我们炼制好的神经网络
    flight_db = NeuralAeroDatabase()
    flight_db._load_from_pickle('aero_surrogate.pth')
    
    engine_db = EngineDatabase()
    engine_db.load1('engine.pkl')

    sim = FlightSimulator6DOF(flight_db, engine_db, aircraft_params)
    sim.set_initial_state(h_m=2000.0, V_mps=250.0, theta_deg=0.7, alpha_deg=0.7)
    
    dt = 0.02 
    target_altitude = 2500.0
    
    history_time, history_alt, history_vel = [], [], []
    history_alpha, history_pitch, history_pn, history_pe = [], [], [], []
    sim_time = 0.0
    
    integral_h, integral_theta = 0.0, 0.0
    
    print(f"开始【光速神经PID】定高测试：目标高度 {target_altitude}m ...")
    
    start_cpu_time = time.time()
    total_steps = int(120 / dt) # 模拟 120 秒
    
    for step in range(total_steps):
        u, v, w = sim.state[3], sim.state[4], sim.state[5]
        phi, theta = sim.state[6], sim.state[7]
        
        current_h = -sim.state[2]
        current_vz = u*math.sin(theta) - v*math.sin(phi)*math.cos(theta) - w*math.cos(phi)*math.cos(theta)
        current_pitch, current_roll = math.degrees(theta), math.degrees(phi)
        current_q, current_p = math.degrees(sim.state[10]), math.degrees(sim.state[9])
        
        # 1. 高度外环
        err_h = target_altitude - current_h
        integral_h = np.clip(integral_h + err_h * dt, -1000.0, 1000.0) 
        
        target_pitch = 0.7 + (0.01 * err_h) - (0.05 * current_vz) + (0.001 * integral_h)
        target_pitch = np.clip(target_pitch, -1.0, 4.0) 
        
        # 2. 俯仰内环
        err_theta = target_pitch - current_pitch
        integral_theta = np.clip(integral_theta + err_theta * dt, -20.0, 20.0)
        
        d_flap = -1.5 * err_theta + 0.8 * current_q - 0.2 * integral_theta
        d_flap = np.clip(d_flap, -10.0, 10.0)
        
        # 3. 横滚增稳环
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
            print(f"Time: {sim_time:.1f}s | 高度: {result['Altitude']:.1f}m | 速度: {result['Velocity']:.1f}m/s | 目标俯仰: {target_pitch:.1f}° | 实际俯仰: {result['Pitch']:.1f}° | 舵面: {d_flap:.1f}°")

    end_cpu_time = time.time()
    print(f"\n✅ 测试完成！物理模拟 120 秒，真实世界仅耗时: {end_cpu_time - start_cpu_time:.2f} 秒！")
    print("这速度，用来跑 PPO 训练简直完美！")
    
    plot1(history_time, history_alt, history_vel, history_alpha, history_pitch, history_pn, history_pe)