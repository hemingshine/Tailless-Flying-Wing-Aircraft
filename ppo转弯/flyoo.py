#coding=utf-8
import os
import pickle
import numpy as np
import math
import warnings
import torch
import torch.nn as nn
from numba import njit  # 👑 引入 Numba

warnings.filterwarnings('ignore')

# =================================================================
# 👑 1. Numba 纯净计算引擎 (C++ 级极速执行)
# =================================================================

@njit(cache=True) # 开启缓存，下次运行免编译
def get_atmosphere_numba(h):
    if h < 11000.0:
        T = 288.15 - 0.0065 * h
        p = 101325.0 * (T / 288.15) ** 5.2561
    else:
        T = 216.65
        p = 22632.1 * math.exp(-9.80665 * (h - 11000.0) / (287.05 * 216.65))
    rho = p / (287.05 * T)
    a = math.sqrt(1.4 * 287.05 * T)
    return rho, a

@njit(cache=True)
def calc_derivatives_numba(state, Fx, Fy, Fz, L_aero, M_aero, N_aero, 
                           mass, g, c1, c2, c3, c4, c5, c6, c7, c8, Iyy):
    # 解包状态
    pn, pe, pd, u, v, w, phi, theta, psi, p, q, r = state
    
    # 缓存三角函数
    sin_phi, cos_phi = math.sin(phi), math.cos(phi)
    sin_theta, cos_theta = math.sin(theta), math.cos(theta)
    sin_psi, cos_psi = math.sin(psi), math.cos(psi)

    gx = -g * sin_theta
    gy = g * sin_phi * cos_theta
    gz = g * cos_phi * cos_theta
    
    # 线加速度
    dot_u = (Fx/mass) + gx - q*w + r*v
    dot_v = (Fy/mass) + gy - r*u + p*w
    dot_w = (Fz/mass) + gz - p*v + q*u
    
    # 角加速度
    dot_p = c1 * r * q + c2 * p * q + c3 * L_aero + c4 * N_aero
    dot_q = c5 * p * r - c6 * (p**2 - r**2) + M_aero / Iyy
    dot_r = c7 * p * q - c2 * q * r + c4 * L_aero + c8 * N_aero
    
    # 姿态角速率
    tan_theta = math.tan(theta)
    dot_phi = p + tan_theta * (q * sin_phi + r * cos_phi)
    dot_theta = q * cos_phi - r * sin_phi
    dot_psi = (q * sin_phi + r * cos_phi) / cos_theta
    
    # 导航速率
    dot_pn = u * cos_theta * cos_psi + v * (sin_phi * sin_theta * cos_psi - cos_phi * sin_psi) + w * (cos_phi * sin_theta * cos_psi + sin_phi * sin_psi)
    dot_pe = u * cos_theta * sin_psi + v * (sin_phi * sin_theta * sin_psi + cos_phi * cos_psi) + w * (cos_phi * sin_theta * sin_psi - sin_phi * cos_psi)
    dot_pd = -u * sin_theta + v * sin_phi * cos_theta + w * cos_phi * cos_theta
    
    return np.array([dot_pn, dot_pe, dot_pd, dot_u, dot_v, dot_w, dot_phi, dot_theta, dot_psi, dot_p, dot_q, dot_r])


# =================================================================
# 2. 气动与发动机数据库 (JIT 编译，保持现状)
# =================================================================
class AeroSurrogate(nn.Module):
    def __init__(self):
        super(AeroSurrogate, self).__init__()
        class ResBlock(nn.Module):
            def __init__(self, in_dim, out_dim):
                super().__init__()
                self.net = nn.Sequential(nn.Linear(in_dim, out_dim), nn.BatchNorm1d(out_dim), nn.GELU(), nn.Dropout(0.2))
                self.shortcut = nn.Linear(in_dim, out_dim) if in_dim != out_dim else nn.Identity()
            def forward(self, x): return self.net(x) + self.shortcut(x)
        self.net = nn.Sequential(
            ResBlock(9, 256), ResBlock(256, 512), ResBlock(512, 256), ResBlock(256, 128), nn.Linear(128, 6)
        )
    def forward(self, x): return self.net(x)

class NeuralAeroDatabase:
    def __init__(self):
        self.model = None
        self._input_tensor = torch.zeros((1, 9), dtype=torch.float32)

    def _load_from_pickle(self, model_path='aero_surrogate.pth'):
        if not os.path.exists(model_path): raise FileNotFoundError(f"找不到模型文件 {model_path}")
        data = torch.load(model_path, map_location='cpu')
        raw_model = AeroSurrogate()
        raw_model.load_state_dict(data['model_state_dict'])
        raw_model.eval() 
        example_input = torch.zeros((1, 9), dtype=torch.float32)
        with torch.no_grad():
            self.model = torch.jit.trace(raw_model, example_input)
            
        torch.set_num_threads(1)        
        self.x_mean = torch.from_numpy(data['x_mean']).float() if isinstance(data['x_mean'], np.ndarray) else data['x_mean']
        self.x_std = torch.from_numpy(data['x_std']).float() if isinstance(data['x_std'], np.ndarray) else data['x_std']
        self.y_mean = torch.from_numpy(data['y_mean']).float() if isinstance(data['y_mean'], np.ndarray) else data['y_mean']
        self.y_std = torch.from_numpy(data['y_std']).float() if isinstance(data['y_std'], np.ndarray) else data['y_std']

    def get_body_axis_coeffs(self, mach, d_flap_L, d_flap_R, d_ail_L, d_ail_R, d_spoil_L, d_spoil_R, alpha, beta):
        with torch.no_grad(): 
            self._input_tensor[0, 0] = mach
            self._input_tensor[0, 1] = d_flap_L
            self._input_tensor[0, 2] = d_flap_R
            self._input_tensor[0, 3] = d_ail_L
            self._input_tensor[0, 4] = d_ail_R
            self._input_tensor[0, 5] = d_spoil_L
            self._input_tensor[0, 6] = d_spoil_R
            self._input_tensor[0, 7] = alpha
            self._input_tensor[0, 8] = beta

            x_norm = (self._input_tensor - self.x_mean) / self.x_std
            y_norm = self.model(x_norm)[0]
            y = y_norm * self.y_std + self.y_mean
        
        return y[0].item(), y[1].item(), y[2].item(), y[3].item(), y[4].item(), y[5].item()

class EngineDatabase:
    def __init__(self): self.thrust_interpolator = None
    def load1(self, pickle_path="engine.pkl"):
        try:
            with open(pickle_path, 'rb') as f: self.thrust_interpolator = pickle.load(f)
        except: pass
    def get_thrust_newtons(self, alt, mach):
        if self.thrust_interpolator is None: return 70000.0 
        thrust_dan = self.thrust_interpolator(np.array([[alt, mach]]))[0]
        return 0.0 if np.isnan(thrust_dan) else thrust_dan * 10.0


# =================================================================
# 3. 核心飞行模拟器 (融合 Numba)
# =================================================================
# =================================================================
# 3. 核心飞行模拟器 (绝对保真的 RK4 + Numba 融合)
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
        
        self.state = np.zeros(12, dtype=np.float64)
        self.SAFE_ALPHA_MIN, self.SAFE_ALPHA_MAX, self.SAFE_BETA_MAX = -10.0, 30.0, 15.0 

    def set_initial_state(self, h_m, V_mps, theta_deg, alpha_deg=0.0):
        self.state[2] = -h_m              
        alpha_rad = math.radians(alpha_deg)
        self.state[3] = V_mps * math.cos(alpha_rad) 
        self.state[5] = V_mps * math.sin(alpha_rad) 
        self.state[7] = math.radians(theta_deg)     
        
    def _get_k(self, current_state, controls):
        """
        内部辅助函数：计算特定状态下的导数 (用于 RK4 的 k1, k2, k3, k4)
        完美解耦了 Python(PyTorch) 和 Numba 的运算边界
        """
        h = -current_state[2]
        u, v, w = current_state[3], current_state[4], current_state[5]
        V = max(math.sqrt(u**2 + v**2 + w**2), 1.0)
        
        alpha_deg = math.degrees(math.atan2(w, u))
        beta_deg = math.degrees(math.asin(max(min(v / V, 1.0), -1.0)))
        query_alpha = max(min(alpha_deg, self.SAFE_ALPHA_MAX), self.SAFE_ALPHA_MIN)
        query_beta = max(min(beta_deg, self.SAFE_BETA_MAX), -self.SAFE_BETA_MAX)
        
        # 1. Numba 加速调用：大气参数
        rho, a = get_atmosphere_numba(h)
        Mach, q_dyn = V / a, 0.5 * rho * V**2
        
        # 2. PyTorch JIT 查表
        try:
            cx, cy, cz, cl, cm, cn = self.aero_db.get_body_axis_coeffs(
                mach=Mach,
                d_flap_L=controls.get('d_flap_L', 0.0), d_flap_R=controls.get('d_flap_R', 0.0),
                d_ail_L=controls.get('d_ail_L', 0.0),   d_ail_R=controls.get('d_ail_R', 0.0),
                d_spoil_L=controls.get('d_spoil_L', 0.0), d_spoil_R=controls.get('d_spoil_R', 0.0), 
                alpha=query_alpha, beta=query_beta
            )
            Fx = -cx * q_dyn * self.S
            Fy = cy * q_dyn * self.S
            Fz = -cz * q_dyn * self.S
            L_aero = cl * q_dyn * self.S * self.b
            M_aero = cm * q_dyn * self.S * self.c_bar
            N_aero = cn * q_dyn * self.S * self.b
        except Exception:
            Fx, Fy, Fz, L_aero, M_aero, N_aero = 0.0, 0.0, 0.0, 0.0, 0.0, 0.0

        throttle = controls.get('throttle', 0.6) 
        thrust = self.engine_db.get_thrust_newtons(h, Mach) * throttle * 5.0 
        Fx += thrust
        
        # 3. Numba 加速调用：解算动力学导数
        return calc_derivatives_numba(
            current_state, Fx, Fy, Fz, L_aero, M_aero, N_aero, 
            self.mass, self.g, self.c1, self.c2, self.c3, self.c4, self.c5, self.c6, self.c7, self.c8, self.Iyy
        )

    def step(self, dt, controls):
        y0 = self.state.copy()
        
        # 👑 绝对纯正的 RK4！
        # 每次都会提取最新状态的气动力并结合 Numba 计算，精度 100% 保真
        k1 = self._get_k(y0, controls)
        k2 = self._get_k(y0 + 0.5 * dt * k1, controls)
        k3 = self._get_k(y0 + 0.5 * dt * k2, controls)
        k4 = self._get_k(y0 + dt * k3, controls)
        
        self.state = y0 + (dt / 6.0) * (k1 + 2*k2 + 2*k3 + k4)
        
        u_n, v_n, w_n = self.state[3], self.state[4], self.state[5]
        V_mag = max(math.sqrt(u_n**2 + v_n**2 + w_n**2), 1.0)
        
        return {
            "Altitude": -self.state[2], "Velocity": V_mag,
            "Alpha": math.degrees(math.atan2(w_n, u_n)), 
            "Beta": math.degrees(math.asin(max(min(v_n / V_mag, 1.0), -1.0))),
            "Pitch": math.degrees(self.state[7]), "Roll": math.degrees(self.state[6]), "Yaw": math.degrees(self.state[8])
        }

    # 为了保证代码完整性，如果你的控制分配器 (get_control_effectiveness_matrix) 还在用，
    # 也可以改写为基于 self._get_k()
    def get_control_effectiveness_matrix(self, current_controls):
        B = np.zeros((3, 6))
        delta = 1.0 
        base_dots = self._get_k(self.state, current_controls)
        base_pqr_dot = np.array([base_dots[9], base_dots[10], base_dots[11]])
        
        control_keys = ['d_flap_L', 'd_flap_R', 'd_ail_L', 'd_ail_R', 'd_spoil_L', 'd_spoil_R']
        for i, key in enumerate(control_keys):
            test_controls = current_controls.copy()
            test_controls[key] += delta
            test_dots = self._get_k(self.state, test_controls)
            test_pqr_dot = np.array([test_dots[9], test_dots[10], test_dots[11]])
            
            B[:, i] = math.degrees(1.0) * (test_pqr_dot - base_pqr_dot) / delta 
        return B