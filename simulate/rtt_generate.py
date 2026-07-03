#coding=utf-8
import os
import pickle
import math
import random
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import warnings

warnings.filterwarnings('ignore')

# ==========================================
# 1. 数据库模块 (气动 & 发动机)
# ==========================================
class HybridAeroDatabase:
    def __init__(self):
        self.raw_data = {} 
        self.models_db = {}
        self.output_cols = [
            '轴向力系数', '横向力系数', '法向力系数', 
            '滚转力矩系数', '俯仰力矩系数', '偏航力矩系数'
        ]

    def _load_from_pickle(self, pickle_path):
        if not os.path.exists(pickle_path):
            print(f"警告: 找不到气动数据库 {pickle_path}。请确保文件存在！")
            return
        with open(pickle_path, 'rb') as f:
            self.models_db = pickle.load(f)
        print(f"气动数据库加载成功！\n" + "-"*40)

    def get_body_axis_coeffs(self, model_code, mach, alpha, beta):
        if not self.models_db:
            return dict(zip(self.output_cols, [0.01, 0, 0.1, 0, 0.05, 0]))

        if model_code not in self.models_db:
            raise ValueError(f"找不到模型代号: {model_code}")
            
        available_machs = list(self.models_db[model_code].keys())
        closest_mach = min(available_machs, key=lambda x: abs(x - mach))
        model_info = self.models_db[model_code][closest_mach]
        query_point = np.array([alpha, beta])
        
        if model_info['type'] == 'ND':
            q = query_point[model_info['active_dims']]
            res = model_info['interp'](q)
            res = res[0] if res.ndim > 1 else res
            if np.isnan(res).any():
                res = np.nan_to_num(res, nan=0.0)
        elif model_info['type'] == '1D':
            q = query_point[model_info['active_dim']]
            res = model_info['interp'](q)
        else:
            res = model_info['val']
            
        return dict(zip(self.output_cols, res))


class EngineDatabase:
    def __init__(self):
        self.thrust_interpolator = None
        self.input_cols = ['Alt（m）', 'Ma']
        self.output_cols = ['FN（DaN）']
    
    def load1(self, pickle_path="engine.pkl"):
        if not os.path.exists(pickle_path):
            print(f"警告: 找不到发动机数据库 {pickle_path}。请确保文件存在！")
            return
        with open(pickle_path, 'rb') as f:
            self.thrust_interpolator = pickle.load(f)
        print("发动机数据库加载成功！\n" + "-"*40)

    def get_thrust_newtons(self, alt, mach):
        if self.thrust_interpolator is None:
            return 50000.0 
            
        query_point = np.array([[alt, mach]])
        thrust_dan = self.thrust_interpolator(query_point)[0]
        if np.isnan(thrust_dan):
            thrust_dan = 0.0 
        return thrust_dan * 10.0

# ==========================================
# 2. 6-DOF 动力学引擎模块
# ==========================================
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

    def get_derivatives(self, state, model_code):
        pn, pe, pd, u, v, w, phi, theta, psi, p, q, r = state
        
        V = math.sqrt(u**2 + v**2 + w**2)
        if V == 0: V = 0.001
        
        alpha_rad = math.atan2(w, u)
        beta_rad = math.asin(v / V) if V > v else 0
        alpha_deg = math.degrees(alpha_rad)
        beta_deg = math.degrees(beta_rad)
        
        h = -pd
        rho, a = self.isa_atmosphere(h)
        Mach = V / a
        q_dyn = 0.5 * rho * V**2
        
        # 1.5倍推力以维持高速平飞
        coeffs = self.aero_db.get_body_axis_coeffs(model_code, Mach, alpha_deg, beta_deg)
        thrust = self.engine_db.get_thrust_newtons(h, Mach) * 1.2
        
        Fx = thrust - coeffs['轴向力系数'] * q_dyn * self.S
        Fy = coeffs['横向力系数'] * q_dyn * self.S
        Fz = - coeffs['法向力系数'] * q_dyn * self.S
        
        L_aero = coeffs['滚转力矩系数'] * q_dyn * self.S * self.b
        M_aero = coeffs['俯仰力矩系数'] * q_dyn * self.S * self.c_bar
        N_aero = coeffs['偏航力矩系数'] * q_dyn * self.S * self.b
        M_aero += -q * 200000.0 
        
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

    def step_rk4(self, dt, model_code):
        y0 = self.state.copy()
        k1 = self.get_derivatives(y0, model_code)
        k2 = self.get_derivatives(y0 + 0.5 * dt * k1, model_code)
        k3 = self.get_derivatives(y0 + 0.5 * dt * k2, model_code)
        k4 = self.get_derivatives(y0 + dt * k3, model_code)
        
        self.state = y0 + (dt / 6.0) * (k1 + 2*k2 + 2*k3 + k4)

# ==========================================
# 3. Kinodynamic RRT 规划器模块
# ==========================================
class KinodynamicRRT:
    def __init__(self, simulator, available_actions, t_max=10.0, rrt_dt=0.5, sim_dt=0.02):
        self.sim = simulator
        self.available_actions = available_actions
        self.t_max = t_max
        self.rrt_dt = rrt_dt
        self.sim_dt = sim_dt
        self.tree = []
        
        self.weights = np.array([
            0.1, 0.1, 1.0,    
            0.5, 0.5, 0.5,    
            5.0, 5.0, 5.0,    
            1.0, 1.0, 1.0     
        ])

    def calc_distance(self, state1, state2):
        diff = state1 - state2
        diff[6:9] = (diff[6:9] + np.pi) % (2 * np.pi) - np.pi
        return np.linalg.norm(diff * self.weights)

    # 引导飞机向正北 (X轴) 飞行
    def sample_random_state(self, current_best_pn):
        rand_state = np.zeros(12)
        # 鼓励向北发展
        rand_state[0] = random.uniform(current_best_pn, current_best_pn + 3000) 
        # 限制东西向偏移，鼓励走直线
        rand_state[1] = random.uniform(-500, 500)               
        
        rand_state[2] = -random.uniform(1600, 2500)              
        rand_state[3] = random.uniform(210, 260) 
        rand_state[7] = math.radians(random.uniform(0.0, 5.0))
        return rand_state

    def simulate_forward(self, start_state, action, duration):
        self.sim.state = start_state.copy()
        steps = int(duration / self.sim_dt)
        
        for _ in range(steps):
            self.sim.step_rk4(self.sim_dt, action)
            
            u, w = self.sim.state[3], self.sim.state[5]
            alpha = math.degrees(math.atan2(w, u)) if u != 0 else 0
            h = -self.sim.state[2]
            
            # 基础坠毁约束判定
            if alpha > 20.0 or alpha < -10.0:
                return None 
            if h > 20000.0 or h < 1500.0:
                return None 
                
        return self.sim.state.copy()

    def generate_expert_trajectories(self, max_iter=1000, num_trajectories=1):
        root_node = {
            'id': 0, 'state': self.sim.state.copy(), 'time': 0.0, 
            'parent_id': -1, 'action_from_parent': None
        }
        self.tree = [root_node]
        successful_leaves = []
        
        # 追踪的最远距离改为 North (pn)
        current_best_pn = self.sim.state[0]

        for i in range(max_iter):
            x_rand = self.sample_random_state(current_best_pn)
            valid_nodes = [n for n in self.tree if n['time'] < self.t_max]
            if not valid_nodes:
                break
                
            valid_nodes.sort(key=lambda n: self.calc_distance(n['state'], x_rand))
            n_near = random.choice(valid_nodes[:min(5, len(valid_nodes))])

            best_new_state = None
            best_action = None
            min_dist = float('inf')
            
            # 宽动作搜索：必选 state05 + 随机 7 个
            neutral_action = 'state05'
            sampled_actions = [neutral_action]
            other_actions = [a for a in self.available_actions if a != neutral_action]
            if other_actions:
                sampled_actions.extend(random.sample(other_actions, min(7, len(other_actions))))
            
            for action in sampled_actions:
                new_state = self.simulate_forward(n_near['state'], action, self.rrt_dt)
                if new_state is not None:
                    dist = self.calc_distance(new_state, x_rand)
                    if dist < min_dist:
                        min_dist = dist
                        best_new_state = new_state
                        best_action = action

            if best_new_state is not None:
                new_time = n_near['time'] + self.rrt_dt
                new_node = {
                    'id': len(self.tree),
                    'state': best_new_state,
                    'time': new_time,
                    'parent_id': n_near['id'],
                    'action_from_parent': best_action
                }
                self.tree.append(new_node)
                
                # 更新最北距离
                if best_new_state[0] > current_best_pn:
                    current_best_pn = best_new_state[0]

                if new_time >= self.t_max:
                    successful_leaves.append(new_node)

        if not successful_leaves:
            return []

        # 终极防作弊打分函数
        def evaluate_leaf(n):
            state = n['state']
            pn = state[0] # 奖励向北飞行距离
            
            u, v, w = state[3], state[4], state[5]
            V = math.sqrt(u**2 + v**2 + w**2)
            phi = state[6]   
            theta = state[7] 
            p, q, r = state[9], state[10], state[11] 
            h = -state[2] 
            
            pitch_deg = math.degrees(theta)
            q_deg = math.degrees(q) 
            
            # ==================================================
            # 【关键修改】：大幅放宽“一票否决”的生死线！
            # 只要不是绝对的死亡螺旋，都允许它进入打分环节
            if pitch_deg < -15.0 or pitch_deg > 25.0:  # 从 [-5, 8] 放宽到 [-15, 25]
                return -float('inf') 
            if abs(q_deg) > 15.0:                      # 允许 15度/秒 的甩尾交接
                return -float('inf')
            if h > 3500.0 or h < 1500.0:               # 放宽高度限制，让它自由爬升
                return -float('inf')
            # ==================================================
                
            # 软惩罚保持不变（这才是真正发挥作用的地方）
            speed_penalty = 0.0
            if V < 210.0:
                speed_penalty = (210.0 - V) * 3000.0
                
            attitude_penalty = (
                abs(phi) * 5000.0 +                               
                abs(theta ) * 1000.0 + 
                (abs(p) + abs(q) + abs(r)) * 2000.0               
            )
            alt_penalty = abs(h - 2000.0) * 50.0 
            
            return pn - attitude_penalty - speed_penalty - alt_penalty

        # 过滤并排序合格状态
        valid_leaves = []
        for leaf in successful_leaves:
            score = evaluate_leaf(leaf)
            if score != -float('inf'):
                leaf['score'] = score
                valid_leaves.append(leaf)

        if not valid_leaves:
            print("  ⚠️ 本段虽然撑到了终点，但所有到达终点的状态都严重违规（姿态或高度越界），全部被一票否决！")
            return []

        valid_leaves.sort(key=lambda n: n['score'], reverse=True)
        
        expert_trajectories = []
        for leaf in valid_leaves[:num_trajectories]:
            path = []
            curr_id = leaf['id']
            while curr_id != -1:
                node = self.tree[curr_id]
                path.append({
                    'state': node['state'],
                    'time': node['time'],
                    'action': node['action_from_parent']
                })
                curr_id = node['parent_id']
            path.reverse()
            expert_trajectories.append(path)
            
        return expert_trajectories

# ==========================================
# 4. 可视化模块
# ==========================================
def plot_trajectory(trajectory):
    times, alts, vels, alphas, pitches, pns, pes = [], [], [], [], [], [], []
    
    for pt in trajectory:
        state = pt['state']
        u, v, w = state[3], state[4], state[5]
        V = math.sqrt(u**2 + v**2 + w**2)
        alpha = math.degrees(math.atan2(w, u)) if u != 0 else 0
        
        times.append(pt['time'])
        alts.append(-state[2])
        vels.append(V)
        alphas.append(alpha)
        pitches.append(math.degrees(state[7]))
        pns.append(state[0])
        pes.append(state[1])

    plt.style.use('dark_background')
    plt.rcParams['font.sans-serif'] = ['SimHei'] 
    plt.rcParams['axes.unicode_minus'] = False   
    
    fig = plt.figure(figsize=(15, 10))
    fig.suptitle('RRT 拼接生成的长序列专家轨迹', fontsize=20, fontweight='bold', color='cyan')

    ax1 = fig.add_subplot(1, 2, 1, projection='3d')
    ax1.plot(pes, pns, alts, color='cyan', linewidth=2.5, label='飞行轨迹')
    ax1.scatter(pes[0], pns[0], alts[0], color='lime', s=100, label='起点', zorder=5)
    ax1.scatter(pes[-1], pns[-1], alts[-1], color='red', s=100, label='终点', zorder=5)
    ax1.set_title('3D 空间航迹', fontsize=14)
    ax1.set_xlabel('东向位置 (m)')
    ax1.set_ylabel('北向位置 (m)')
    ax1.set_zlabel('高度 (m)')
    ax1.legend()
    ax1.view_init(elev=25, azim=-45) 

    ax2 = fig.add_subplot(3, 2, 2)
    ax2.plot(times, alts, color='springgreen', linewidth=2)
    ax2.axhline(y=1500, color='red', linestyle='--', alpha=0.5, label='下限 1500m')
    ax2.set_title('高度剖面', fontsize=12)
    ax2.set_ylabel('高度 (m)')
    ax2.legend()
    ax2.grid(True, linestyle='--', alpha=0.3)

    ax3 = fig.add_subplot(3, 2, 4)
    ax3.plot(times, vels, color='gold', linewidth=2)
    ax3.set_title('速度剖面', fontsize=12)
    ax3.set_ylabel('速度 (m/s)')
    ax3.grid(True, linestyle='--', alpha=0.3)

    ax4 = fig.add_subplot(3, 2, 6)
    ax4.plot(times, pitches, color='hotpink', linewidth=2, label='俯仰角 (°)')
    ax4.set_xlabel('时间 (s)')
    ax4.set_ylabel('角度 (°)')
    
    ax4_twin = ax4.twinx()
    ax4_twin.plot(times, alphas, color='dodgerblue', linewidth=2, linestyle='-.', label='迎角 (°)')
    ax4_twin.axhline(y=15, color='red', linestyle=':', alpha=0.5)
    ax4_twin.axhline(y=-10, color='red', linestyle=':', alpha=0.5)
    
    ax4.set_title('纵向气动姿态', fontsize=12)
    fig.legend(loc='lower right', bbox_to_anchor=(0.95, 0.05))
    ax4.grid(True, linestyle='--', alpha=0.3)

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
    engine_db.load1('engine.pkl')

    sim = FlightSimulator6DOF(flight_db, engine_db, aircraft_params)
    sim.set_initial_state(h_m=2000.0, V_mps=250.0, theta_deg=0.0)
    
    available_actions = ["state01", "state02", "state03", "state04", "state05", "state06", "state07", "state08", "state09", "state10",
                        "state11", "state12", "state13", "state14", "state15", "state16", "state17", "state18", "state19", "state20",
                        "state21", "state22", "state23", "state24", "state25", "state26", "state27", "state28", "state29", "state30",
                        "state31", "state32", "state33", "state34", "state35", "state36", "state37", "state38", "state39", "state40",
                        "state41", "state42", "state43", "state44", "state45", "state46"]
    
    total_segments = 10        
    segment_duration = 10.0    
    global_time_offset = 0.0   
    
    full_expert_trajectory = []
    
    for segment in range(total_segments):
        print(f"\n=== 🚀 开始规划第 {segment + 1}/{total_segments} 段轨迹 (时间: {global_time_offset}s - {global_time_offset + segment_duration}s) ===")
        
        rrt_planner = KinodynamicRRT(
            simulator=sim, 
            available_actions=available_actions,
            t_max=segment_duration,
            rrt_dt=0.5,   
            sim_dt=0.02   
        )
        
        best_trajs = rrt_planner.generate_expert_trajectories(max_iter=2000, num_trajectories=1)
        
        if not best_trajs:
            print(f"❌ 第 {segment + 1} 段规划失败！无法找到存活到终点的路线，拼接提前终止。")
            break
            
        best_segment = best_trajs[0]
        
        for i, pt in enumerate(best_segment):
            if segment > 0 and i == 0:
                continue
            stitched_pt = {
                'state': pt['state'].copy(),
                'time': pt['time'] + global_time_offset,
                'action': pt['action'] 
            }
            full_expert_trajectory.append(stitched_pt)
            
        last_state = best_segment[-1]['state']
        sim.state = last_state.copy() 
        global_time_offset += segment_duration
        print(f"✅ 第 {segment + 1} 段完成，当前飞行距离 (North): {last_state[0]:.1f} m")

    if full_expert_trajectory:
        final_time = full_expert_trajectory[-1]['time']
        print(f"\n🎉 规划流程结束！成功获取一条长达 {final_time:.1f} 秒的专家连续轨迹。")
        plot_trajectory(full_expert_trajectory)
    else:
        print("未能生成任何有效轨迹，请检查气动数据或初始状态。")