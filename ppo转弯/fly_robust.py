#coding=utf-8
import os
import pickle
import numpy as np
import math
import warnings
from numba import njit

warnings.filterwarnings('ignore')

# =================================================================
# 1. Numba 纯净计算引擎 (保留原加速机制)
# =================================================================
@njit(cache=True)
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
def eval_coeffs_numba(B, alpha, sf, df, sa, da, spF, spR, beta):
    # 特征顺序必须与 build_dat.py 的 FEATURES 完全一致；用 float 数组避免异构元组
    f = np.empty(13)
    f[0] = 1.0; f[1] = alpha; f[2] = alpha * alpha; f[3] = alpha * alpha * alpha
    f[4] = sf; f[5] = df; f[6] = sa; f[7] = da
    f[8] = spF; f[9] = spR; f[10] = beta
    f[11] = alpha * sf; f[12] = alpha * da
    out = np.zeros(6)
    for j in range(6):
        s = 0.0
        for i in range(13):
            s += f[i] * B[i, j]
        out[j] = s
    return out


@njit(cache=True)
def calc_derivatives_numba(state, Fx, Fy, Fz, L_aero, M_aero, N_aero,
                           mass, g, c1, c2, c3, c4, c5, c6, c7, c8, Iyy):
    pn, pe, pd, u, v, w, phi, theta, psi, p, q, r = state
    sin_phi, cos_phi = math.sin(phi), math.cos(phi)
    sin_theta, cos_theta = math.sin(theta), math.cos(theta)
    sin_psi, cos_psi = math.sin(psi), math.cos(psi)
    gx = -g * sin_theta; gy = g * sin_phi * cos_theta; gz = g * cos_phi * cos_theta
    dot_u = (Fx / mass) + gx - q * w + r * v
    dot_v = (Fy / mass) + gy - r * u + p * w
    dot_w = (Fz / mass) + gz - p * v + q * u
    dot_p = c1 * r * q + c2 * p * q + c3 * L_aero + c4 * N_aero
    dot_q = c5 * p * r - c6 * (p**2 - r**2) + M_aero / Iyy
    dot_r = c7 * p * q - c2 * q * r + c4 * L_aero + c8 * N_aero
    tan_theta = math.tan(theta)
    dot_phi = p + tan_theta * (q * sin_phi + r * cos_phi)
    dot_theta = q * cos_phi - r * sin_phi
    dot_psi = (q * sin_phi + r * cos_phi) / cos_theta
    dot_pn = u * cos_theta * cos_psi + v * (sin_phi * sin_theta * cos_psi - cos_phi * sin_psi) + w * (cos_phi * sin_theta * cos_psi + sin_phi * sin_psi)
    dot_pe = u * cos_theta * sin_psi + v * (sin_phi * sin_theta * sin_psi + cos_phi * cos_psi) + w * (cos_phi * sin_theta * sin_psi - sin_phi * cos_psi)
    dot_pd = -u * sin_theta + v * sin_phi * cos_theta + w * cos_phi * cos_theta
    return np.array([dot_pn, dot_pe, dot_pd, dot_u, dot_v, dot_w,
                     dot_phi, dot_theta, dot_psi, dot_p, dot_q, dot_r])


# =================================================================
# 2. 解析气动模型 (替代神经代理；保留类名以兼容现有 import)
# =================================================================
class AeroSurrogate:   # 占位，保持 `from fly import AeroSurrogate` 不报错
    pass


# 数据范围内才可信，超出做温和裁剪防多项式发散
ALPHA_CLIP = (-5.0, 26.0)
BETA_CLIP = (-12.0, 16.0)
CTRL_CLIP = 30.0


class NeuralAeroDatabase:
    """名字沿用，内部已是解析模型：poly3(alpha)+线性操纵/侧滑导数。"""
    def __init__(self):
        self.machs = None
        self.Bs = None

    def _load_from_pickle(self, model_path='X47B_coeffs.pkl'):
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"找不到系数文件 {model_path}（请先运行 build_dat.py）")
        with open(model_path, 'rb') as f:
            data = pickle.load(f)
        self.machs = np.array(sorted([k for k in data.keys() if isinstance(k, float)]))
        self.Bs = [np.ascontiguousarray(data[float(m)], dtype=np.float64) for m in self.machs]

    def get_body_axis_coeffs(self, mach, d_flap_L, d_flap_R, d_ail_L, d_ail_R,
                             d_spoil_L, d_spoil_R, alpha, beta):
        a = float(min(max(alpha, ALPHA_CLIP[0]), ALPHA_CLIP[1]))
        b = float(min(max(beta, BETA_CLIP[0]), BETA_CLIP[1]))
        c = CTRL_CLIP
        fL = float(min(max(d_flap_L, -c), c)); fR = float(min(max(d_flap_R, -c), c))
        aL = float(min(max(d_ail_L, -c), c)); aR = float(min(max(d_ail_R, -c), c))
        spF = float(min(max(d_spoil_L, -c), c))   # d_spoil_L 对应数据"前扰流板"
        spR = float(min(max(d_spoil_R, -c), c))   # d_spoil_R 对应数据"后扰流板"
        sf = (fL + fR) / 2.0; df = (fL - fR) / 2.0
        sa = (aL + aR) / 2.0; da = (aL - aR) / 2.0
        idx = int(np.argmin(np.abs(self.machs - mach)))   # 就近取表
        y = eval_coeffs_numba(self.Bs[idx], a, sf, df, sa, da, spF, spR, b)
        return y[0], y[1], y[2], y[3], y[4], y[5]


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
# 3. 核心飞行模拟器 (RK4 + Numba + 解析气动 + 转动阻尼)
# =================================================================
# 转动阻尼导数 (数据缺失，用 DATCOM 式飞翼典型量级估算，可调)
CL_P = -0.40   # 滚转阻尼
CM_Q = -4.0    # 俯仰阻尼
CN_R = -0.05   # 偏航阻尼(无尾翼，弱)


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
        # === 鲁棒性评估钩子(默认1.0/0，对训练无任何影响) ===
        self.k_clp = 1.0; self.k_cmq = 1.0; self.k_cnr = 1.0   # 动导数(Clp/Cmq/Cnr)散布缩放
        self.wind = np.zeros(3, dtype=np.float64)              # 体轴湍流风速[u,v,w](m/s)，气动用相对风

    def set_initial_state(self, h_m, V_mps, theta_deg, alpha_deg=0.0):
        self.state[2] = -h_m
        alpha_rad = math.radians(alpha_deg)
        self.state[3] = V_mps * math.cos(alpha_rad)
        self.state[5] = V_mps * math.sin(alpha_rad)
        self.state[7] = math.radians(theta_deg)

    def _get_k(self, current_state, controls):
        h = -current_state[2]
        # 气动相对风 = 体轴速度 - 体轴风(湍流)；平动动力学仍用惯性速度(在 calc_derivatives_numba 内)
        u = current_state[3] - self.wind[0]
        v = current_state[4] - self.wind[1]
        w = current_state[5] - self.wind[2]
        p, q, r = current_state[9], current_state[10], current_state[11]
        V = max(math.sqrt(u**2 + v**2 + w**2), 1.0)
        alpha_deg = math.degrees(math.atan2(w, u))
        beta_deg = math.degrees(math.asin(max(min(v / V, 1.0), -1.0)))
        query_alpha = max(min(alpha_deg, self.SAFE_ALPHA_MAX), self.SAFE_ALPHA_MIN)
        query_beta = max(min(beta_deg, self.SAFE_BETA_MAX), -self.SAFE_BETA_MAX)
        rho, a = get_atmosphere_numba(h)
        Mach, q_dyn = V / a, 0.5 * rho * V**2

        try:
            cx, cy, cz, cl, cm, cn = self.aero_db.get_body_axis_coeffs(
                mach=Mach,
                d_flap_L=controls.get('d_flap_L', 0.0), d_flap_R=controls.get('d_flap_R', 0.0),
                d_ail_L=controls.get('d_ail_L', 0.0), d_ail_R=controls.get('d_ail_R', 0.0),
                d_spoil_L=controls.get('d_spoil_L', 0.0), d_spoil_R=controls.get('d_spoil_R', 0.0),
                alpha=query_alpha, beta=query_beta)
            # 转动阻尼(数据缺失，解析补：用无量纲角速率)
            phat = p * self.b / (2.0 * V)
            qhat = q * self.c_bar / (2.0 * V)
            rhat = r * self.b / (2.0 * V)
            cl += CL_P * self.k_clp * phat
            cm += CM_Q * self.k_cmq * qhat
            cn += CN_R * self.k_cnr * rhat

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
        return calc_derivatives_numba(current_state, Fx, Fy, Fz, L_aero, M_aero, N_aero,
                                      self.mass, self.g, self.c1, self.c2, self.c3, self.c4,
                                      self.c5, self.c6, self.c7, self.c8, self.Iyy)

    def step(self, dt, controls):
        y0 = self.state.copy()
        k1 = self._get_k(y0, controls)
        k2 = self._get_k(y0 + 0.5 * dt * k1, controls)
        k3 = self._get_k(y0 + 0.5 * dt * k2, controls)
        k4 = self._get_k(y0 + dt * k3, controls)
        self.state = y0 + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)
        u_n, v_n, w_n = self.state[3], self.state[4], self.state[5]
        V_mag = max(math.sqrt(u_n**2 + v_n**2 + w_n**2), 1.0)
        return {"Altitude": -self.state[2], "Velocity": V_mag,
                "Alpha": math.degrees(math.atan2(w_n, u_n)),
                "Beta": math.degrees(math.asin(max(min(v_n / V_mag, 1.0), -1.0))),
                "Pitch": math.degrees(self.state[7]), "Roll": math.degrees(self.state[6]),
                "Yaw": math.degrees(self.state[8])}

    def get_derivatives(self, state, controls):
        # 兼容旧接口别名(eval_nsmc_alph.py 等用 get_derivatives)
        return self._get_k(state, controls)

    def get_control_effectiveness_matrix(self, current_controls):
        B = np.zeros((3, 6))
        delta = 1.0
        base = self._get_k(self.state, current_controls)
        base_pqr = np.array([base[9], base[10], base[11]])
        keys = ['d_flap_L', 'd_flap_R', 'd_ail_L', 'd_ail_R', 'd_spoil_L', 'd_spoil_R']
        for i, key in enumerate(keys):
            tc = current_controls.copy(); tc[key] = tc.get(key, 0.0) + delta
            td = self._get_k(self.state, tc)
            B[:, i] = math.degrees(1.0) * (np.array([td[9], td[10], td[11]]) - base_pqr) / delta
        return B