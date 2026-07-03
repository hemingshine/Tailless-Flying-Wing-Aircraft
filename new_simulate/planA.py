#coding=utf-8
import os
import pickle
import math
import random
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.gridspec as gridspec
import warnings

warnings.filterwarnings('ignore')

# ==========================================
# 0. 全局任务设置
# ==========================================
TARGET_ALT = 3000.0   
TARGET_VEL = 250.0    

# ==========================================
# 1. 数据库模块 (边界绝对锁死)
# ==========================================
class HybridAeroDatabase:
    def __init__(self):
        self.raw_data = {} 
        self.models_db = {}
        self.output_cols = [
            '轴向力系数', '横向力系数', '法向力系数', 
            '滚转力矩系数', '俯仰力矩系数', '偏航力矩系数'
        ]
        self._cache = {}
        self.available_machs = []

    def _load_from_pickle(self, pickle_path):
        if not os.path.exists(pickle_path):
            return
        with open(pickle_path, 'rb') as f:
            self.models_db = pickle.load(f)
        self.available_machs = list(self.models_db.keys())

    def get_body_axis_coeffs(self, mach, d_flap_L, d_flap_R, d_ail_L, d_ail_R, d_spoil_F, d_spoil_R, alpha, beta):
        if not self.models_db:
            return dict(zip(self.output_cols, [0.01, 0, 0.1, 0, 0.05, 0]))

        alpha = max(-3.0, min(alpha, 15.0))
        beta = max(-10.0, min(beta, 15.0))
        d_flap_L = max(-30.0, min(d_flap_L, 30.0))
        d_flap_R = max(-30.0, min(d_flap_R, 30.0))
        d_spoil_F = max(-25.0, min(d_spoil_F, 0.0))
        d_spoil_R = max(0.0, min(d_spoil_R, 25.0))

        r_alpha = round(alpha * 2) / 2.0  
        r_beta = round(beta * 2) / 2.0
        r_mach = round(mach, 2)
        r_flap_L = round(d_flap_L * 2) / 2.0
        r_spoil_F = round(d_spoil_F * 2) / 2.0
        r_spoil_R = round(d_spoil_R * 2) / 2.0
        
        cache_key = (r_mach, r_flap_L, r_flap_L, d_ail_L, d_ail_R, r_spoil_F, r_spoil_R, r_alpha, r_beta)
        if cache_key in self._cache:
            return self._cache[cache_key]

        closest_mach = min(self.available_machs, key=lambda x: abs(x - r_mach))
        model_info = self.models_db[closest_mach]
        query_point = np.array([r_flap_L, r_flap_L, d_ail_L, d_ail_R, r_spoil_F, r_spoil_R, r_alpha, r_beta])
        
        if model_info['type'] == 'ND':
            q = query_point[model_info['active_dims']]
            res = model_info['interp'](q)
            res = res[0] if res.ndim > 1 else res
            if np.isnan(res).any(): res = np.nan_to_num(res, nan=0.0)
        elif model_info['type'] == '1D':
            q = query_point[model_info['active_dims'][0]]
            res = model_info['interp'](q)
        else:
            res = model_info['val']
            
        result_dict = dict(zip(self.output_cols, res))
        self._cache[cache_key] = result_dict
        return result_dict

class EngineDatabase:
    def __init__(self):
        self.thrust_interpolator = None
    
    def load1(self, pickle_path="engine.pkl"):
        if not os.path.exists(pickle_path):
            pickle_path = "engine_cache.pkl"
        if not os.path.exists(pickle_path):
            return
        with open(pickle_path, 'rb') as f:
            self.thrust_interpolator = pickle.load(f)

    def get_thrust_newtons(self, alt, mach):
        if self.thrust_interpolator is None: return 50000.0 
        query_point = np.array([[alt, mach]])
        thrust_dan = self.thrust_interpolator(query_point)[0]
        if np.isnan(thrust_dan): thrust_dan = 0.0 
        return thrust_dan * 10.0

# ==========================================
# 2. 6-DOF 动力学引擎模块 (保留极弱增稳护航)
# ==========================================
class FlightSimulator6DOF:
    def __init__(self, aero_db, engine_db, global_params):
        self.aero_db = aero_db
        self.engine_db = engine_db
        self.S, self.b, self.c_bar, self.mass = global_params['S'], global_params['b'], global_params['c_bar'], global_params['mass']
        self.Ixx, self.Iyy, self.Izz, self.Ixz = global_params['Ixx'], global_params['Iyy'], global_params['Izz'], global_params['Ixz']
        
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
        self.last_thrust = 50000.0  

    def set_initial_state(self, h_m, V_mps, theta_deg):
        self.state = np.zeros(12)
        self.state[2] = -h_m
        self.state[3] = V_mps
        self.state[7] = math.radians(theta_deg)
        self.last_thrust = self.engine_db.get_thrust_newtons(h_m, V_mps / 340.0)
        
    def isa_atmosphere(self, altitude_m):
        T0, p0, rho0, L, R, gamma = 288.15, 101325.0, 1.225, 0.0065, 287.05, 1.4
        if altitude_m < 11000:
            T = T0 - L * altitude_m
            p = p0 * (T / T0) ** (self.g * 0.0289644 / (8.3144598 * L))
            rho = p / (R * T)
        else:
            T = 216.65
            rho = 0.36391 * math.exp(-(altitude_m - 11000) / 6341.6)
        return rho, math.sqrt(gamma * R * T)

    def get_derivatives(self, state, controls):
        pn, pe, pd, u, v, w, phi, theta, psi, p, q, r = state
        V = math.sqrt(u**2 + v**2 + w**2)
        if V == 0: V = 0.001
        
        alpha_rad = math.atan2(w, u)
        beta_rad = math.asin(v / V) if V > v else 0
        h = -pd
        rho, a = self.isa_atmosphere(h)
        Mach = V / a
        q_dyn = 0.5 * rho * V**2
        
        coeffs = self.aero_db.get_body_axis_coeffs(
            mach=Mach,
            d_flap_L=controls.get('d_flap_L', 0.0), d_flap_R=controls.get('d_flap_R', 0.0),
            d_ail_L=controls.get('d_ail_L', 0.0), d_ail_R=controls.get('d_ail_R', 0.0),
            d_spoil_F=controls.get('d_spoil_F', 0.0), d_spoil_R=controls.get('d_spoil_R', 0.0),
            alpha=math.degrees(alpha_rad), beta=math.degrees(beta_rad)
        )
        
        # 推力平滑
        base_thrust = self.engine_db.get_thrust_newtons(h, Mach)
        target_thrust = base_thrust + (TARGET_VEL - V) * 4000.0  
        target_thrust = max(0.0, min(target_thrust, 150000.0))
        thrust = 0.90 * self.last_thrust + 0.10 * target_thrust
        self.last_thrust = thrust
        
        Fx = thrust - coeffs['轴向力系数'] * q_dyn * self.S
        Fy = coeffs['横向力系数'] * q_dyn * self.S
        Fz = - coeffs['法向力系数'] * q_dyn * self.S
        
        v_down = -u * math.sin(theta) + w * math.cos(theta)
        alt_error = TARGET_ALT - h
        
        target_theta_deg = max(min(2 + alt_error * 0.05 + v_down * 0.1, 5.0), -1.5) 
        target_theta = math.radians(target_theta_deg)
        
        theta_err = theta - target_theta
        theta_err_deg = math.degrees(abs(theta_err))
        
        # 维持 15% 以内的安全阻尼，主力还是靠舵面
        k_pitch = min(0.05 + 0.02 * theta_err_deg, 0.15)
        
        base_M = q_dyn * self.S * self.c_bar
        base_L = q_dyn * self.S * self.b
        
        raw_pitch_M = -theta_err * base_M * k_pitch - q * base_M * 0.15
        
        M_fbw = np.clip(raw_pitch_M, -0.15 * base_M, 0.15 * base_M)
        L_fbw = np.clip(-(phi * 5.0 + p * 3.0) * base_L, -0.10 * base_L, 0.10 * base_L)
        N_fbw = np.clip(-(r * 3.0) * base_L, -0.10 * base_L, 0.10 * base_L)
        
        L_aero = coeffs['滚转力矩系数'] * base_L + L_fbw
        M_aero = coeffs['俯仰力矩系数'] * base_M + M_fbw
        N_aero = coeffs['偏航力矩系数'] * base_L + N_fbw
        
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
        y0 = self.state.copy()
        k1 = self.get_derivatives(y0, controls)
        k2 = self.get_derivatives(y0 + 0.5 * dt * k1, controls)
        k3 = self.get_derivatives(y0 + 0.5 * dt * k2, controls)
        k4 = self.get_derivatives(y0 + dt * k3, controls)
        self.state = y0 + (dt / 6.0) * (k1 + 2*k2 + 2*k3 + k4)

# ==========================================
# 3. Kinodynamic RRT (全舵面解封！)
# ==========================================
class KinodynamicRRT:
    def __init__(self, simulator, t_max=10.0, rrt_dt=1.0, sim_dt=0.05):
        self.sim = simulator
        self.t_max = t_max
        self.rrt_dt = rrt_dt 
        self.sim_dt = sim_dt
        self.tree = []
        self.weights = np.array([0.1, 0.1, 1.0, 0.5, 0.5, 0.5, 5.0, 5.0, 5.0, 1.0, 1.0, 1.0])
        
        self.neutral_control = {
            'd_flap_L': 0.0, 'd_flap_R': 0.0, 'd_ail_L': 0.0, 'd_ail_R': 0.0, 
            'd_spoil_F': 0.0, 'd_spoil_R': 0.0
        }

    def calc_distance(self, state1, state2):
        diff = state1 - state2
        diff[6:9] = (diff[6:9] + np.pi) % (2 * np.pi) - np.pi
        return np.linalg.norm(diff * self.weights)

    def sample_random_state(self, current_best_pn):
        rand_state = np.zeros(12)
        rand_state[0] = random.uniform(current_best_pn, current_best_pn + 3000) 
        rand_state[1] = random.uniform(-300, 300)               
        rand_state[2] = -random.uniform(TARGET_ALT - 100, TARGET_ALT + 100)              
        rand_state[3] = random.uniform(TARGET_VEL - 5, TARGET_VEL + 5) 
        rand_state[7] = math.radians(random.uniform(0.0, 4.0))
        return rand_state

    def sample_random_controls(self, prev_action, current_state):
        if prev_action is None: prev_action = self.neutral_control.copy()
            
        h = -current_state[2]
        alt_error = TARGET_ALT - h
        
        is_stable = abs(alt_error) < 3.0
        
        if is_stable and random.random() < 0.70: 
            return prev_action.copy()
            
        flap = prev_action['d_flap_L']
        spoil_F = prev_action['d_spoil_F']
        spoil_R = prev_action['d_spoil_R']
        
        # 🚀 彻底解放，不对打舵方向做人工限制！让 RRT 自己试出气动特性
        flap += random.choice([-2.0, -1.0, -0.5, 0.0, 0.5, 1.0, 2.0])
        spoil_F += random.choice([-2.0, -1.0, 0.0, 1.0, 2.0])
        spoil_R += random.choice([-2.0, -1.0, 0.0, 1.0, 2.0])
            
        flap = max(-25.0, min(flap, 25.0)) 
        spoil_F = max(-25.0, min(spoil_F, 0.0))  # 前扰流板限幅
        spoil_R = max(0.0, min(spoil_R, 25.0))   # 后扰流板限幅
            
        return {
            'd_flap_L': flap, 'd_flap_R': flap,
            'd_ail_L': 0.0, 'd_ail_R': 0.0,
            'd_spoil_F': spoil_F, 'd_spoil_R': spoil_R
        }

    def simulate_forward(self, start_state, prev_action, target_action, duration):
        self.sim.state = start_state.copy()
        steps = int(duration / self.sim_dt)
        if prev_action is None: prev_action = self.neutral_control.copy()
            
        for i in range(steps):
            progress = (i + 1) / steps
            current_controls = {}
            for k in target_action:
                current_controls[k] = prev_action[k] + (target_action[k] - prev_action[k]) * progress
                
            self.sim.step_rk4(self.sim_dt, current_controls)
            
        return self.sim.state.copy()

    def generate_expert_trajectories(self, max_iter=1200, num_trajectories=1, initial_action=None):
        if initial_action is None:
            initial_action = self.neutral_control.copy()
            
        root_node = {
            'id': 0, 'state': self.sim.state.copy(), 'time': 0.0, 
            'parent_id': -1, 'action_from_parent': initial_action.copy()
        }
        self.tree = [root_node]
        successful_leaves = []
        current_best_pn = self.sim.state[0]

        for i in range(max_iter):
            x_rand = self.sample_random_state(current_best_pn)
            valid_nodes = [n for n in self.tree if n['time'] < self.t_max]
            if not valid_nodes: break
                
            valid_nodes.sort(key=lambda n: self.calc_distance(n['state'], x_rand))
            n_near = random.choice(valid_nodes[:min(5, len(valid_nodes))])
            
            prev_action = n_near['action_from_parent']
            best_new_state, best_action, min_dist = None, None, float('inf')
            
            sampled_actions = [prev_action.copy()]
            for _ in range(5): 
                sampled_actions.append(self.sample_random_controls(prev_action, n_near['state']))
            
            for action_dict in sampled_actions:
                new_state = self.simulate_forward(n_near['state'], prev_action, action_dict, self.rrt_dt)
                if new_state is not None:
                    dist = self.calc_distance(new_state, x_rand)
                    if dist < min_dist:
                        min_dist, best_new_state, best_action = dist, new_state, action_dict

            if best_new_state is not None:
                new_time = n_near['time'] + self.rrt_dt
                new_node = {
                    'id': len(self.tree), 'state': best_new_state,
                    'time': new_time, 'parent_id': n_near['id'], 'action_from_parent': best_action
                }
                self.tree.append(new_node)
                if best_new_state[0] > current_best_pn: current_best_pn = best_new_state[0]
                
                if new_time >= self.t_max:
                    successful_leaves.append(new_node)
                    if len(successful_leaves) >= 3: break 

        if not successful_leaves: return []

        def evaluate_leaf(n):
            state = n['state']
            pn, h = state[0], -state[2]
            u, v, w = state[3], state[4], state[5]
            V = math.sqrt(u**2 + v**2 + w**2)
            phi, theta = state[6], state[7]
            p, q, r = state[9], state[10], state[11]
            
            # 只有极度离谱才枪毙，保证代码绝对跑通
            if h < 500.0 or h > 10000.0: return -float('inf')
            if abs(math.degrees(theta)) > 40.0: return -float('inf')
            
            # 高度偏差即是天理！
            alt_penalty = abs(h - TARGET_ALT) * 10000.0 
            attitude_penalty = abs(phi)*5000.0 + abs(q)*2000.0 
            speed_penalty = abs(V - TARGET_VEL) * 50.0
            
            # 惩罚无意义的阻力增加（防止扰流板一直开着）
            action = n['action_from_parent']
            drag_penalty = (abs(action['d_spoil_F']) + abs(action['d_spoil_R'])) * 50.0
            
            return pn - attitude_penalty - speed_penalty - alt_penalty - drag_penalty

        valid_leaves = []
        for leaf in successful_leaves:
            score = evaluate_leaf(leaf)
            if score != -float('inf'):
                leaf['score'] = score
                valid_leaves.append(leaf)

        if not valid_leaves: return []
        valid_leaves.sort(key=lambda n: n['score'], reverse=True)
        
        expert_trajectories = []
        for leaf in valid_leaves[:num_trajectories]:
            path, curr_id = [], leaf['id']
            while curr_id != -1:
                node = self.tree[curr_id]
                path.append({'state': node['state'], 'time': node['time'], 'action': node['action_from_parent']})
                curr_id = node['parent_id']
            path.reverse()
            expert_trajectories.append(path)
            
        return expert_trajectories

# ==========================================
# 4. 可视化模块 (三舵面联动完整展现)
# ==========================================
def plot_trajectory(trajectory):
    times, alts, vels, alphas, pitches, pns, pes = [], [], [], [], [], [], []
    flap_L, spoil_F, spoil_R = [], [], []
    
    for pt in trajectory:
        state = pt['state']
        action = pt.get('action', {'d_flap_L':0, 'd_spoil_F':0, 'd_spoil_R':0})
        
        u, v, w = state[3], state[4], state[5]
        V = math.sqrt(u**2 + v**2 + w**2)
        
        times.append(pt['time'])
        alts.append(-state[2])
        vels.append(V)
        alphas.append(math.degrees(math.atan2(w, u)) if u != 0 else 0)
        pitches.append(math.degrees(state[7]))
        pns.append(state[0])
        pes.append(state[1])
        
        flap_L.append(action['d_flap_L'])
        spoil_F.append(action['d_spoil_F'])
        spoil_R.append(action['d_spoil_R'])

    plt.style.use('dark_background')
    plt.rcParams['font.sans-serif'] = ['SimHei'] 
    plt.rcParams['axes.unicode_minus'] = False   
    
    fig = plt.figure(figsize=(16, 12))
    fig.suptitle(f'长航时 RRT (三舵面联动：完全气动自由版)', fontsize=20, fontweight='bold', color='cyan')

    gs = gridspec.GridSpec(4, 2, figure=fig)

    ax1 = fig.add_subplot(gs[:, 0], projection='3d')
    ax1.plot(pes, pns, alts, color='cyan', linewidth=2.5, label='飞行轨迹')
    ax1.scatter(pes[0], pns[0], alts[0], color='lime', s=100, label='起点', zorder=5)
    ax1.scatter(pes[-1], pns[-1], alts[-1], color='red', s=100, label='终点', zorder=5)
    ax1.set_title('3D 空间航迹', fontsize=14)
    ax1.set_xlabel('东向位置 (m)')
    ax1.set_ylabel('北向位置 (m)')
    ax1.set_zlabel('高度 (m)')
    ax1.legend()
    ax1.view_init(elev=25, azim=-45) 

    ax2 = fig.add_subplot(gs[0, 1])
    ax2.plot(times, alts, color='springgreen', linewidth=2)
    ax2.axhline(y=TARGET_ALT, color='white', linestyle='--', alpha=0.5, label=f'目标 {TARGET_ALT}m')
    min_alt, max_alt = min(alts), max(alts)
    padding = max(5, (max_alt - min_alt) * 0.5)
    if padding == 0: padding = 5
    ax2.set_ylim(min_alt - padding, max_alt + padding)
    ax2.set_title('高度剖面', fontsize=12)
    ax2.set_ylabel('高度 (m)')
    ax2.grid(True, linestyle='--', alpha=0.3)

    ax3 = fig.add_subplot(gs[1, 1])
    ax3.plot(times, vels, color='gold', linewidth=2)
    ax3.axhline(y=TARGET_VEL, color='white', linestyle='--', alpha=0.5)
    ax3.set_title('速度剖面', fontsize=12)
    ax3.set_ylabel('速度 (m/s)')
    ax3.grid(True, linestyle='--', alpha=0.3)

    ax4 = fig.add_subplot(gs[2, 1])
    ax4.plot(times, pitches, color='hotpink', linewidth=2, label='俯仰角 (°)')
    ax4.set_ylabel('俯仰角 (°)')
    
    ax4_twin = ax4.twinx()
    ax4_twin.plot(times, alphas, color='dodgerblue', linewidth=2, linestyle='-.', label='迎角 (°)')
    ax4_twin.axhline(y=0, color='white', linestyle='--', alpha=0.3)
    ax4_twin.set_ylabel('迎角 (°)')
    ax4.set_title('纵向气动姿态', fontsize=12)
    fig.legend(loc='center right', bbox_to_anchor=(0.98, 0.4))
    ax4.grid(True, linestyle='--', alpha=0.3)
    
    ax5 = fig.add_subplot(gs[3, 1])
    ax5.plot(times, flap_L, color='cyan', linewidth=2.5, label='主襟翼偏角 L/R')
    # 显著标出扰流板的动作
    ax5.plot(times, spoil_F, color='orange', linewidth=2, linestyle='--', label='前扰流板 (压机头/减升力)')
    ax5.plot(times, spoil_R, color='lime', linewidth=2, linestyle=':', label='后扰流板 (辅助控制)')
    ax5.set_title('🛫 测谎仪：三大气动舵面全火力出击', fontsize=12)
    ax5.set_xlabel('时间 (s)')
    ax5.set_ylabel('偏转角 (°)')
    ax5.legend(loc='upper left', ncol=3, fontsize=9)
    ax5.grid(True, linestyle='--', alpha=0.3)

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()

# ==========================================
# 5. 主程序入口
# ==========================================
if __name__ == "__main__":
    
    aircraft_params = {
        'S': 88.58, 'b': 18.9, 'c_bar': 4.6, 'mass': 14000,
        'Ixx': 313220.6, 'Iyy': 273435.3, 'Izz': 392903.9, 'Ixz': 0.0
    }

    flight_db = HybridAeroDatabase()
    flight_db._load_from_pickle('X47B.pkl')
    engine_db = EngineDatabase()
    
    engine_path = 'engine.pkl' if os.path.exists('engine.pkl') else 'engine_cache.pkl'
    engine_db.load1(engine_path)

    sim = FlightSimulator6DOF(flight_db, engine_db, aircraft_params)
    sim.set_initial_state(h_m=TARGET_ALT, V_mps=TARGET_VEL, theta_deg=1.5)
    
    total_segments = 40        
    segment_duration = 10.0    
    global_time_offset = 0.0   
    
    rrt_dt = 0.5     
    sim_dt = 0.02    
    
    full_expert_trajectory = []
    
    global_last_action = {
        'd_flap_L': 0.0, 'd_flap_R': 0.0, 'd_ail_L': 0.0, 'd_ail_R': 0.0, 
        'd_spoil_F': 0.0, 'd_spoil_R': 0.0
    }
    
    for segment in range(total_segments):
        print(f"\n=== 🚀 正在规划第 {segment + 1}/{total_segments} 段 (时间: {global_time_offset}s - {global_time_offset + segment_duration}s) ===")
        
        rrt_planner = KinodynamicRRT(sim, t_max=segment_duration, rrt_dt=rrt_dt, sim_dt=sim_dt)
        best_trajs = rrt_planner.generate_expert_trajectories(max_iter=1500, num_trajectories=1, initial_action=global_last_action)
        
        print(f"👉 当前气动缓存池已累积 {len(flight_db._cache)} 种有效姿态")
        
        if not best_trajs:
            print(f"❌ 第 {segment + 1} 段规划失败！拼接提前终止。")
            break
            
        best_segment = best_trajs[0]
        
        for i, pt in enumerate(best_segment):
            if segment > 0 and i == 0: continue
            full_expert_trajectory.append({
                'state': pt['state'].copy(),
                'time': pt['time'] + global_time_offset,
                'action': pt['action'].copy()
            })
            
        last_state = best_segment[-1]['state']
        sim.state = last_state.copy() 
        global_time_offset += segment_duration
        global_last_action = best_segment[-1]['action'].copy()

    if full_expert_trajectory:
        print("\n========================================================")
        print("🎯 规划成功！正在生成高密度平滑绘图数据...")
        
        sim.set_initial_state(h_m=TARGET_ALT, V_mps=TARGET_VEL, theta_deg=1.5)
        
        dense_trajectory = []
        current_time = 0.0
        
        prev_action = {
            'd_flap_L': 0.0, 'd_flap_R': 0.0, 'd_ail_L': 0.0, 'd_ail_R': 0.0, 
            'd_spoil_F': 0.0, 'd_spoil_R': 0.0
        }
        
        dense_trajectory.append({
            'time': current_time, 
            'state': sim.state.copy(),
            'action': prev_action.copy()
        })
        
        steps_per_action = int(rrt_dt / sim_dt)
        
        for pt in full_expert_trajectory[1:]:
            target_action = pt['action']
            for i in range(steps_per_action):
                progress = (i + 1) / steps_per_action
                interp_action = {}
                for k in target_action:
                    interp_action[k] = prev_action[k] + (target_action[k] - prev_action[k]) * progress
                    
                sim.step_rk4(sim_dt, interp_action)
                current_time += sim_dt
                
                dense_trajectory.append({
                    'time': current_time, 
                    'state': sim.state.copy(),
                    'action': interp_action.copy()
                })
                
            prev_action = target_action.copy()
                
        print(f"✅ 生成完毕！绘制 {len(dense_trajectory)} 帧物理遥测数据。")
        plot_trajectory(dense_trajectory)
    else:
        print("未能生成任何有效轨迹。")