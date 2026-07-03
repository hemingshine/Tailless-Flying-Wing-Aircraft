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
# 1. 气动与发动机数据库 (保持原样，极其稳定)
# =================================================================
class AeroSurrogate(nn.Module):
    def __init__(self):
        super(AeroSurrogate, self).__init__()
        class ResBlock(nn.Module):
            def __init__(self, in_dim, out_dim):
                super().__init__()
                self.net = nn.Sequential(
                    nn.Linear(in_dim, out_dim), nn.BatchNorm1d(out_dim), nn.GELU(), nn.Dropout(0.2)
                )
                self.shortcut = nn.Linear(in_dim, out_dim) if in_dim != out_dim else nn.Identity()
            def forward(self, x): return self.net(x) + self.shortcut(x)
        self.net = nn.Sequential(
            ResBlock(9, 256), ResBlock(256, 512), ResBlock(512, 256), ResBlock(256, 128), nn.Linear(128, 6)
        )
    def forward(self, x): return self.net(x)

class NeuralAeroDatabase:
    def __init__(self):
        self.output_cols = ['轴向力系数', '横向力系数', '法向力系数', '滚转力矩系数', '俯仰力矩系数', '偏航力矩系数']
        self.model = None

    def _load_from_pickle(self, model_path='aero_surrogate.pth'):
        if not os.path.exists(model_path): raise FileNotFoundError(f"找不到模型文件 {model_path}")
        data = torch.load(model_path, map_location='cpu')
        self.model = AeroSurrogate()
        self.model.load_state_dict(data['model_state_dict'])
        self.model.eval() 
        torch.set_num_threads(1)        
        self.x_mean, self.x_std = data['x_mean'], data['x_std']
        self.y_mean, self.y_std = data['y_mean'], data['y_std']

    def get_body_axis_coeffs(self, mach, d_flap_L, d_flap_R, d_ail_L, d_ail_R, d_spoil_L, d_spoil_R, alpha, beta):
        # 注意：使用用户原数据集中定义的舵面映射
        x = torch.tensor([[mach, d_flap_L, d_flap_R, d_ail_L, d_ail_R, d_spoil_L, d_spoil_R, alpha, beta]], dtype=torch.float32)
        x_norm = (x - self.x_mean) / self.x_std
        with torch.no_grad(): 
            y_norm = self.model(x_norm)[0] # 取出第 0 个 batch 的结果
        y = y_norm * self.y_std + self.y_mean
        return dict(zip(self.output_cols, y.numpy()))   

class EngineDatabase:
    def __init__(self): self.thrust_interpolator = None
    def load1(self, pickle_path="engine.pkl"):
        try:
            with open(pickle_path, 'rb') as f: self.thrust_interpolator = pickle.load(f)
        except: pass
    def get_thrust_newtons(self, alt, mach):
        if self.thrust_interpolator is None: return 7000.0 * 10.0 
        thrust_dan = self.thrust_interpolator(np.array([[alt, mach]]))[0]
        return 0.0 if np.isnan(thrust_dan) else thrust_dan * 10.0

# =================================================================
# 2. 核心飞行模拟器 (升级 6 舵面全开接口)
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
        self.SAFE_ALPHA_MIN, self.SAFE_ALPHA_MAX, self.SAFE_BETA_MAX = -10.0, 30.0, 15.0 

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
        
        # 👑 引入全部 6 个操纵面
        try:
            coeffs = self.aero_db.get_body_axis_coeffs(
                mach=Mach,
                d_flap_L=controls.get('d_flap_L', 0.0), d_flap_R=controls.get('d_flap_R', 0.0),
                d_ail_L=controls.get('d_ail_L', 0.0),   d_ail_R=controls.get('d_ail_R', 0.0),
                d_spoil_L=controls.get('d_spoil_L', 0.0), d_spoil_R=controls.get('d_spoil_R', 0.0), 
                alpha=query_alpha, beta=query_beta
            )
            Fx = -coeffs['轴向力系数'] * q_dyn * self.S
            Fy = coeffs['横向力系数'] * q_dyn * self.S
            Fz = -coeffs['法向力系数'] * q_dyn * self.S
            L_aero = coeffs['滚转力矩系数'] * q_dyn * self.S * self.b
            M_aero = coeffs['俯仰力矩系数'] * q_dyn * self.S * self.c_bar
            N_aero = coeffs['偏航力矩系数'] * q_dyn * self.S * self.b
        except Exception:
            Fx, Fy, Fz, L_aero, M_aero, N_aero = 0,0,0,0,0,0

        throttle = controls.get('throttle', 0.6) 
        thrust = self.engine_db.get_thrust_newtons(h, Mach) * throttle * 5.0 
        Fx += thrust
        
        gx, gy, gz = -self.g * math.sin(theta), self.g * math.sin(phi) * math.cos(theta), self.g * math.cos(phi) * math.cos(theta)
        
        dot_u, dot_v, dot_w = (Fx/self.mass)+gx-q*w+r*v, (Fy/self.mass)+gy-r*u+p*w, (Fz/self.mass)+gz-p*v+q*u
        
        # 角加速度 (p_dot, q_dot, r_dot)
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
            "Alpha": math.degrees(math.atan2(w, u)), "Beta": math.degrees(math.asin(v / max(math.sqrt(u**2 + v**2 + w**2), 1.0))),
            "Pitch": math.degrees(self.state[7]), "Roll": math.degrees(self.state[6]), "Yaw": math.degrees(self.state[8])
        }

    # 👑 新增：实时控制效能矩阵 B 探测器
    def get_control_effectiveness_matrix(self, current_controls):
        """
        实时解算 B 矩阵 (3x6)。
        计算 6 个操纵面偏转对 p_dot, q_dot, r_dot 的偏导数。
        """
        B = np.zeros((3, 6))
        delta = 1.0 # 探测步长 1.0 度
        
        base_dots = self.get_derivatives(self.state, current_controls)
        base_pqr_dot = np.array([base_dots[9], base_dots[10], base_dots[11]])
        
        control_keys = ['d_flap_L', 'd_flap_R', 'd_ail_L', 'd_ail_R', 'd_spoil_L', 'd_spoil_R']
        
        # 对每一个舵面进行前向差分探测
        for i, key in enumerate(control_keys):
            test_controls = current_controls.copy()
            test_controls[key] += delta
            test_dots = self.get_derivatives(self.state, test_controls)
            test_pqr_dot = np.array([test_dots[9], test_dots[10], test_dots[11]])
            
            # 转为角度单位制下的角加速度导数 (deg/s^2 / deg)
            B[:, i] = math.degrees(1.0) * (test_pqr_dot - base_pqr_dot) / delta 
            
        return B


# =================================================================
# 3. 👑 全新：冗余操纵面控制分配器 (带再分配机制)
# =================================================================
class ControlAllocator:
    def __init__(self):
        # 权重矩阵 W：决定优先级。数字越大，越不愿意被调用。
        # [flapL, flapR, ailL, ailR, spoilL, spoilR]
        # 襟翼/副翼做主操纵面(权重低)，扰流板会剧增阻力，主要负责偏航，设为极高权重。
        self.W = np.diag([1.0, 1.0, 1.0, 1.0, 50.0, 50.0])
        self.W_inv = np.linalg.inv(self.W)
        
        # 舵面物理限幅
        self.limit_min = np.array([-30.0, -30.0, -20.0, -20.0, 0.0, 0.0])  # 扰流板只能向上翻转(>0)
        self.limit_max = np.array([30.0,  30.0,  20.0,  20.0,  25.0, 25.0])

    def allocate(self, v_req, B):
        """
        v_req: 三轴虚拟控制需求 [p_dot_req, q_dot_req, r_dot_req] (deg/s^2)
        B: 3x6 效能矩阵
        """
        # --- 第一阶段：基础加权伪逆 (WPI) ---
        temp = B @ self.W_inv @ B.T
        # 防止小动压/失速下的矩阵奇异
        if np.linalg.cond(temp) > 1e8: 
            temp += np.eye(3) * 1e-4
            
        P_inv = self.W_inv @ B.T @ np.linalg.inv(temp)
        u_opt = P_inv @ v_req
        
        # 限幅
        u_clipped = np.clip(u_opt, self.limit_min, self.limit_max)
        
        # --- 第二阶段：饱和再分配 (Redistribution) ---
        saturated = (u_opt <= self.limit_min) | (u_opt >= self.limit_max)
        
        if np.any(saturated) and not np.all(saturated):
            # 1. 计算因为饱和而损失的虚拟力矩
            v_achieved = B @ u_clipped
            v_error = v_req - v_achieved
            
            # 2. 剥离已饱和的舵面，寻找剩余健康舵面的自由度
            B_free = B.copy()
            B_free[:, saturated] = 0.0 
            
            temp_free = B_free @ self.W_inv @ B_free.T
            # 如果剩余自由度仍能配平力矩
            if np.linalg.cond(temp_free) < 1e8:
                P_inv_free = self.W_inv @ B_free.T @ np.linalg.inv(temp_free + np.eye(3)*1e-6)
                delta_u = P_inv_free @ v_error
                
                # 叠加补偿并再次限幅
                u_final = np.clip(u_clipped + delta_u, self.limit_min, self.limit_max)
            else:
                u_final = u_clipped
        else:
            u_final = u_clipped
            
        # 格式化输出为字典
        return {
            'd_flap_L': u_final[0], 'd_flap_R': u_final[1],
            'd_ail_L':  u_final[2], 'd_ail_R':  u_final[3],
            'd_spoil_L': u_final[4], 'd_spoil_R': u_final[5]
        }