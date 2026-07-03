#coding=utf-8
import os
import pickle
import math
import random
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import warnings
from scipy.interpolate import LinearNDInterpolator, interp1d

warnings.filterwarnings('ignore')

# ==========================================
# 0. 全局任务设置 (在这里修改你的目标高度！)
# ==========================================
TARGET_ALT = 3000.0   # 目标起飞与巡航高度 (米)
TARGET_VEL = 250.0    # 目标巡航速度 (m/s)

# ==========================================
# 1. 数据库模块 (方案A：0.5° 精度哈希缓存版)
# ==========================================
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
        self._cache = {}
        self.available_machs = []

    def _load_from_pickle(self, pickle_path):
        if not os.path.exists(pickle_path):
            print(f"警告: 找不到气动数据库 {pickle_path}")
            return
        with open(pickle_path, 'rb') as f:
            self.models_db = pickle.load(f)
        self.available_machs = list(self.models_db.keys())
        print(f"气动数据库 '{pickle_path}' 加载成功！\n" + "-"*40)

    def get_body_axis_coeffs(self, mach, d_flap_L, d_flap_R, d_ail_L, d_ail_R, d_spoil_F, d_spoil_R, alpha, beta):
        if not self.models_db:
            return dict(zip(self.output_cols, [0.01, 0, 0.1, 0, 0.05, 0]))

        r_alpha = round(alpha * 2) / 2.0  
        r_beta = round(beta * 2) / 2.0
        r_mach = round(mach, 2)
        
        cache_key = (r_mach, d_flap_L, d_flap_R, d_ail_L, d_ail_R, d_spoil_F, d_spoil_R, r_alpha, r_beta)
        if cache_key in self._cache:
            return self._cache[cache_key]

        closest_mach = min(self.available_machs, key=lambda x: abs(x - r_mach))
        model_info = self.models_db[closest_mach]
        query_point = np.array([d_flap_L, d_flap_R, d_ail_L, d_ail_R, d_spoil_F, d_spoil_R, r_alpha, r_beta])
        
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
    
    def load1(self, pickle_path="engine_cache.pkl"):
        if not os.path.exists(pickle_path):
            return
        with open(pickle_path, 'rb') as f:
            self.thrust_interpolator = pickle.load(f)
        print(f"发动机数据库 '{pickle_path}' 加载成功！\n" + "-"*40)

    def get_thrust_newtons(self, alt, mach):
        if self.thrust_interpolator is None: return 50000.0 
        query_point = np.array([[alt, mach]])
        thrust_dan = self.thrust_interpolator(query_point)[0]
        if np.isnan(thrust_dan): thrust_dan = 0.0 
        return thrust_dan * 10.0

# ==========================================
# 2. 6-DOF 动力学引擎模块 (终极版高度自动驾驶仪)
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

    def set_initial_state(self, h_m, V_mps, theta_deg):
        self.state = np.zeros(12)
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
        
        # 🚀 自动油门 (Auto-Throttle)
        base_thrust = self.engine_db.get_thrust_newtons(h, Mach)
        thrust_cmd = base_thrust + (TARGET_VEL - V) * 15000.0  
        thrust = max(0.0, min(thrust_cmd, 150000.0))
        
        Fx = thrust - coeffs['轴向力系数'] * q_dyn * self.S
        Fy = coeffs['横向力系数'] * q_dyn * self.S
        Fz = - coeffs['法向力系数'] * q_dyn * self.S
        
        # 🚀 【自动驾驶仪核心：高度-俯仰 P控制器】
        alt_error = TARGET_ALT - h
        # 基础配平角设为 2.0°。每低于目标 10m，就多抬头 1°；限幅在 -5° 到 10° 之间。
        target_theta_deg = max(min(2.0 + alt_error * 0.1, 10.0), -5.0)
        target_theta = math.radians(target_theta_deg)
        
        # 飞控姿态保持
        pitch_restore = -(theta - target_theta) * 5000000.0  # 强力锁定目标俯仰角
        pitch_damp = -q * 4000000.0                          # 极强俯仰减震
        roll_restore = -phi * 3000000.0                      # 滚转强制回正
        roll_damp = -p * 2000000.0                           # 滚转减震
        yaw_damp = -r * 2000000.0                            # 偏航减震
        
        L_aero = coeffs['滚转力矩系数'] * q_dyn * self.S * self.b + roll_restore + roll_damp
        M_aero = coeffs['俯仰力矩系数'] * q_dyn * self.S * self.c_bar + pitch_restore + pitch_damp
        N_aero = coeffs['偏航力矩系数'] * q_dyn * self.S * self.b + yaw_damp
        
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
# 3. Kinodynamic RRT 规划器模块
# ==========================================
class KinodynamicRRT:
    def __init__(self, simulator, t_max=10.0, rrt_dt=0.5, sim_dt=0.04):
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
        rand_state[1] = random.uniform(-500, 500)               
        rand_state[2] = -random.uniform(TARGET_ALT - 200, TARGET_ALT + 200)              
        rand_state[3] = random.uniform(TARGET_VEL - 10, TARGET_VEL + 10) 
        rand_state[7] = math.radians(random.uniform(0.0, 5.0))
        return rand_state

    def sample_random_controls(self):
        if random.random() < 0.4:
            return self.neutral_control.copy()
            
        flap = random.choice([-8.0, -4.0, -2.0, 0.0, 2.0, 4.0, 8.0])
        spoil_f = random.choice([-5.0, -2.0, 0.0])
        spoil_r = random.choice([0.0, 2.0, 5.0])
        ail = 0.0 

        return {
            'd_flap_L': flap, 'd_flap_R': flap,
            'd_ail_L': ail, 'd_ail_R': -ail,
            'd_spoil_F': spoil_f, 'd_spoil_R': spoil_r
        }

    def simulate_forward(self, start_state, action_dict, duration):
        self.sim.state = start_state.copy()
        steps = int(duration / self.sim_dt)
        for _ in range(steps):
            self.sim.step_rk4(self.sim_dt, action_dict)
            u, w, h = self.sim.state[3], self.sim.state[5], -self.sim.state[2]
            alpha = math.degrees(math.atan2(w, u)) if u != 0 else 0
            
            # 放宽底线，反正有高度维持控制器，只要不翻跟头就行
            if alpha > 30.0 or alpha < -20.0 or h > TARGET_ALT + 5000.0 or h < 500.0:
                return None 
        return self.sim.state.copy()

    def generate_expert_trajectories(self, max_iter=800, num_trajectories=1):
        root_node = {
            'id': 0, 'state': self.sim.state.copy(), 'time': 0.0, 
            'parent_id': -1, 'action_from_parent': self.neutral_control.copy()
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

            best_new_state, best_action, min_dist = None, None, float('inf')
            
            sampled_actions = [self.neutral_control.copy()]
            for _ in range(3): sampled_actions.append(self.sample_random_controls())
            
            for action_dict in sampled_actions:
                new_state = self.simulate_forward(n_near['state'], action_dict, self.rrt_dt)
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
                    if len(successful_leaves) >= 5: break

        if not successful_leaves: return []

        def evaluate_leaf(n):
            state = n['state']
            pn, h = state[0], -state[2]
            u, v, w = state[3], state[4], state[5]
            V = math.sqrt(u**2 + v**2 + w**2)
            phi, theta = state[6], state[7]
            
            if math.degrees(theta) < -25.0 or math.degrees(theta) > 30.0 or h > TARGET_ALT + 3000.0 or h < 500.0:
                return -float('inf') 
                
            attitude_penalty = abs(phi)*10000.0 
            alt_penalty = abs(h - TARGET_ALT) * 100.0 
            speed_penalty = abs(V - TARGET_VEL) * 50.0
            
            return pn - attitude_penalty - speed_penalty - alt_penalty

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
# 4. 可视化模块 (补全所有飞翼控制面曲线)
# ==========================================
def plot_trajectory(trajectory):
    # 基础飞行状态数据
    times, alts, vels, alphas, pitches, pns, pes = [], [], [], [], [], [], []
    # 飞翼控制面数据（补全所有控制面）
    flap_L_angles = []    # 左襟翼偏角
    flap_R_angles = []    # 右襟翼偏角
    ail_L_angles = []     # 左副翼偏角
    ail_R_angles = []     # 右副翼偏角
    spoil_F_angles = []   # 前扰流板偏角
    spoil_R_angles = []   # 后扰流板偏角
    
    # 遍历轨迹提取所有数据
    for pt in trajectory:
        state = pt['state']
        u, v, w = state[3], state[4], state[5]
        V = math.sqrt(u**2 + v**2 + w**2)
        action = pt.get('action', {})
        
        # 基础状态数据
        times.append(pt['time'])
        alts.append(-state[2])
        vels.append(V)
        alphas.append(math.degrees(math.atan2(w, u)) if u != 0 else 0)
        pitches.append(math.degrees(state[7]))
        pns.append(state[0])
        pes.append(state[1])
        
        # 飞翼控制面数据
        flap_L_angles.append(action.get('d_flap_L', 0.0))
        flap_R_angles.append(action.get('d_flap_R', 0.0))
        ail_L_angles.append(action.get('d_ail_L', 0.0))
        ail_R_angles.append(action.get('d_ail_R', 0.0))
        spoil_F_angles.append(action.get('d_spoil_F', 0.0))
        spoil_R_angles.append(action.get('d_spoil_R', 0.0))

    # 绘图样式设置
    plt.style.use('dark_background')
    plt.rcParams['font.sans-serif'] = ['SimHei'] 
    plt.rcParams['axes.unicode_minus'] = False   
    
    # 画布布局调整为3行3列，适配所有曲线
    fig = plt.figure(figsize=(18, 15))
    fig.suptitle(f'带自动驾驶仪的长航时轨迹 ({TARGET_ALT}m 定高巡航)', fontsize=20, fontweight='bold', color='cyan')

    # 1. 3D空间航迹
    ax1 = fig.add_subplot(3, 3, 1, projection='3d')
    ax1.plot(pes, pns, alts, color='cyan', linewidth=2.5, label='飞行轨迹')
    ax1.scatter(pes[0], pns[0], alts[0], color='lime', s=100, label='起点', zorder=5)
    ax1.scatter(pes[-1], pns[-1], alts[-1], color='red', s=100, label='终点', zorder=5)
    ax1.set_title('3D 空间航迹', fontsize=14)
    ax1.set_xlabel('东向位置 (m)')
    ax1.set_ylabel('北向位置 (m)')
    ax1.set_zlabel('高度 (m)')
    ax1.legend()
    ax1.view_init(elev=25, azim=-45) 

    # 2. 高度剖面
    ax2 = fig.add_subplot(3, 3, 2)
    ax2.plot(times, alts, color='springgreen', linewidth=2)
    ax2.axhline(y=TARGET_ALT, color='white', linestyle='--', alpha=0.5, label=f'目标 {TARGET_ALT}m')
    min_alt, max_alt = min(alts), max(alts)
    padding = max(10, (max_alt - min_alt) * 0.5)
    ax2.set_ylim(min_alt - padding, max_alt + padding)
    ax2.set_title('高度剖面 (自动驾驶介入中)', fontsize=12)
    ax2.set_ylabel('高度 (m)')
    ax2.legend()
    ax2.grid(True, linestyle='--', alpha=0.3)

    # 3. 速度剖面
    ax3 = fig.add_subplot(3, 3, 3)
    ax3.plot(times, vels, color='gold', linewidth=2)
    ax3.axhline(y=TARGET_VEL, color='white', linestyle='--', alpha=0.5, label=f'目标 {TARGET_VEL}m/s')
    ax3.set_title('速度剖面 (自动油门生效中)', fontsize=12)
    ax3.set_ylabel('速度 (m/s)')
    ax3.legend()
    ax3.grid(True, linestyle='--', alpha=0.3)

    # 4. 纵向气动姿态
    ax4 = fig.add_subplot(3, 3, 4)
    ax4.plot(times, pitches, color='hotpink', linewidth=2, label='俯仰角 (°)')
    ax4.set_xlabel('时间 (s)')
    ax4.set_ylabel('角度 (°)')
    ax4_twin = ax4.twinx()
    ax4_twin.plot(times, alphas, color='dodgerblue', linewidth=2, linestyle='-.', label='迎角 (°)')
    ax4.set_title('纵向气动姿态 (寻找最佳配平点)', fontsize=12)
    ax4.legend(loc='upper left')
    ax4_twin.legend(loc='upper right')
    ax4.grid(True, linestyle='--', alpha=0.3)

    # 5. 襟翼偏角变化
    ax5 = fig.add_subplot(3, 3, 5)
    ax5.plot(times, flap_L_angles, color='orange', linewidth=2, label='左襟翼')
    ax5.plot(times, flap_R_angles, color='coral', linewidth=2, linestyle='--', label='右襟翼')
    ax5.axhline(y=0, color='white', linestyle='--', alpha=0.5, label='中立位置')
    ax5.set_title('襟翼偏角变化', fontsize=12)
    ax5.set_xlabel('时间 (s)')
    ax5.set_ylabel('偏角 (°)')
    ax5.legend()
    ax5.grid(True, linestyle='--', alpha=0.3)

    # 6. 扰流板偏角变化
    ax6 = fig.add_subplot(3, 3, 6)
    ax6.plot(times, spoil_F_angles, color='purple', linewidth=2, label='前扰流板')
    ax6.plot(times, spoil_R_angles, color='lightgreen', linewidth=2, linestyle='--', label='后扰流板')
    ax6.axhline(y=0, color='white', linestyle='--', alpha=0.5, label='中立位置')
    ax6.set_title('前后扰流板偏角变化', fontsize=12)
    ax6.set_xlabel('时间 (s)')
    ax6.set_ylabel('偏角 (°)')
    ax6.legend()
    ax6.grid(True, linestyle='--', alpha=0.3)

    # 7. 副翼偏角变化（新增补全）
    ax7 = fig.add_subplot(3, 3, 7)
    ax7.plot(times, ail_L_angles, color='deepskyblue', linewidth=2, label='左副翼')
    ax7.plot(times, ail_R_angles, color='red', linewidth=2, linestyle='--', label='右副翼')
    ax7.axhline(y=0, color='white', linestyle='--', alpha=0.5, label='中立位置')
    ax7.set_title('左右副翼偏角变化', fontsize=12)
    ax7.set_xlabel('时间 (s)')
    ax7.set_ylabel('偏角 (°)')
    ax7.legend()
    ax7.grid(True, linestyle='--', alpha=0.3)

    # 隐藏多余的空白子图（保持布局整齐）
    for idx in [8,9]:
        ax_empty = fig.add_subplot(3,3,idx)
        ax_empty.axis('off')

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
    engine_db.load1('engine.pkl') # 注意文件名

    sim = FlightSimulator6DOF(flight_db, engine_db, aircraft_params)
    sim.set_initial_state(h_m=TARGET_ALT, V_mps=TARGET_VEL, theta_deg=2.0)
    
    # 模拟 10 段，每段 10 秒，共计 100 秒
    total_segments = 50        
    segment_duration = 10.0    
    global_time_offset = 0.0   
    
    rrt_dt = 0.5     
    sim_dt = 0.04    
    
    full_expert_trajectory = []
    
    for segment in range(total_segments):
        print(f"\n=== 🚀 正在规划第 {segment + 1}/{total_segments} 段 (时间: {global_time_offset}s - {global_time_offset + segment_duration}s) ===")
        
        # 配合完美的自动驾驶，迭代次数降至 800，运行超快
        rrt_planner = KinodynamicRRT(sim, t_max=segment_duration, rrt_dt=rrt_dt, sim_dt=sim_dt)
        best_trajs = rrt_planner.generate_expert_trajectories(max_iter=800, num_trajectories=1)
        
        print(f"👉 当前气动缓存池已累积 {len(flight_db._cache)} 种有效姿态")
        
        if not best_trajs:
            print(f"❌ 第 {segment + 1} 段规划失败！无法找到安全路线，拼接提前终止。")
            break
            
        best_segment = best_trajs[0]
        
        for i, pt in enumerate(best_segment):
            if segment > 0 and i == 0: continue
            full_expert_trajectory.append({
                'state': pt['state'].copy(),
                'time': pt['time'] + global_time_offset,
                'action': pt['action'] 
            })
            
        last_state = best_segment[-1]['state']
        sim.state = last_state.copy() 
        global_time_offset += segment_duration

    if full_expert_trajectory:
        print("\n========================================================")
        print("🎯 规划成功！正在生成高密度平滑绘图数据...")
        
        action_sequence = [pt['action'] for pt in full_expert_trajectory[1:]]
        
        # 物理复现时恢复同样的初始状态
        sim.set_initial_state(h_m=TARGET_ALT, V_mps=TARGET_VEL, theta_deg=2.0)
        
        dense_trajectory = []
        current_time = 0.0
        # 初始点携带初始中立控制量
        dense_trajectory.append({
            'time': current_time, 
            'state': sim.state.copy(),
            'action': full_expert_trajectory[0]['action']
        })
        
        steps_per_action = int(rrt_dt / sim_dt)
        
        # 修复：高密度轨迹每个点都携带对应的控制量，确保曲线数据正确
        for action in action_sequence:
            for _ in range(steps_per_action):
                sim.step_rk4(sim_dt, action)
                current_time += sim_dt
                dense_trajectory.append({
                    'time': current_time, 
                    'state': sim.state.copy(),
                    'action': action  # 关键修复：每个高密度点都绑定对应控制量
                })
                
        print(f"✅ 生成完毕！绘制 {len(dense_trajectory)} 帧物理遥测数据。")
        plot_trajectory(dense_trajectory)
    else:
        print("未能生成任何有效轨迹。")