#coding=utf-8
import numpy as np
import math
import gymnasium as gym
from gymnasium import spaces
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.env_util import make_vec_env
import warnings

from fly import NeuralAeroDatabase, EngineDatabase, FlightSimulator6DOF

warnings.filterwarnings('ignore')

def get_current_derivatives(sim, controls):
    return math.degrees(sim.get_derivatives(sim.state, controls)[10]) 

# =================================================================
# 👑 统一个体：将 RBF-NSMC 彻底嵌入训练环境
# =================================================================
class RBF_Integral_NSMC:
    def __init__(self):
        self.c1 = 4.0; self.ki1 = 2.0; self.K = 10.0; self.eta = 5.0; self.phi = 0.5
        e1_c = np.linspace(-1.0, 1.0, 5); e2_c = np.linspace(-1.0, 1.0, 5)
        self.centers = np.array(np.meshgrid(e1_c, e2_c)).T.reshape(-1, 2)
        self.width = 1.0; self.W = np.zeros(self.centers.shape[0])
        self.Gamma = 50.0; self.kappa = 0.05      
        
    def reset_states(self):
        # 训练环境每次 reset 必须清空 RBF 权重，防止 Episode 间污染
        self.W = np.zeros(self.centers.shape[0])

    def compute_control(self, e_theta, int_e_theta, q, theta_c_dot, theta_c_ddot, f2_nom, ce0, dt):
        q_c = -self.c1 * e_theta - self.ki1 * int_e_theta + theta_c_dot
        q_c_dot = -self.c1 * (q - theta_c_dot) - self.ki1 * e_theta + theta_c_ddot
        s = q - q_c  
        x_nn = np.array([e_theta / 10.0, s / 20.0])
        dist_sq = np.sum((self.centers - x_nn)**2, axis=1)
        h = np.exp(-dist_sq / (2 * self.width**2))
        self.W += (self.Gamma * (s / 20.0) * h - self.kappa * self.W) * dt
        f_nn = np.dot(self.W, h) 
        u_eq = (-f2_nom + q_c_dot - f_nn * 10.0) / ce0 
        u_sw = (-self.K * s - self.eta * math.tanh(s / self.phi)) / ce0
        return np.clip(u_eq + u_sw, -20.0, 20.0)

class MIMOX47BEnv(gym.Env):
    def __init__(self):
        super(MIMOX47BEnv, self).__init__()
        aircraft_params = {'S': 88.58, 'b': 18.9, 'c_bar': 4.6, 'mass': 14000,
                           'Ixx': 313220.6, 'Iyy': 273435.3, 'Izz': 392903.9, 'Ixz': 0.0}
        self.flight_db = NeuralAeroDatabase()
        self.flight_db._load_from_pickle('aero_surrogate.pth')
        self.engine_db = EngineDatabase()
        self.engine_db.load1('engine.pkl')

        self.sim = FlightSimulator6DOF(self.flight_db, self.engine_db, aircraft_params)
        self.nsmc = RBF_Integral_NSMC() # 实例化 NSMC
        
        self.dt = 0.02; self.action_repeat = 10; self.max_steps = 1000  
        self.g = 9.80665
        
        # 观测空间：引入总能量误差和能量分配误差
        high = np.array([5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 2.0, 2.0, 2.0], dtype=np.float32)
        self.observation_space = spaces.Box(-high, high, dtype=np.float32)
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(2,), dtype=np.float32)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.current_step = 0
        
        # 👑 100% 同构：严格重置所有底层滤波器、积分器、延迟与网络权重
        self.integral_e_theta = 0.0 
        self.nsmc.reset_states()
        
        self.target_alt = np.random.uniform(2000.0, 4000.0)
        self.target_vel = np.random.uniform(180.0, 240.0)
        
        init_h = self.target_alt + np.random.uniform(-150.0, 150.0) 
        init_v = self.target_vel + np.random.uniform(-40.0, 40.0)
        
        self.last_V = init_v
        self.sim.set_initial_state(h_m=init_h, V_mps=init_v, theta_deg=3.0, alpha_deg=3.0)
        
        # 指令平滑与迟滞参数初始化
        self.last_action = np.array([0.0, 0.0], dtype=np.float32)
        self.smoothed_action = np.array([0.0, 0.0], dtype=np.float32)
        self.pitch_c = 3.0       
        self.pitch_c_dot = 0.0   
        self.omega_n = 2.0       
        self.zeta = 0.9    
        self.actual_throttle = 0.5 
        
        return self._get_obs(), {}

    def _get_obs(self):
        u, v, w = self.sim.state[3], self.sim.state[4], self.sim.state[5]
        phi, theta = self.sim.state[6], self.sim.state[7]
        current_h = -self.sim.state[2]
        V = math.sqrt(u**2 + v**2 + w**2)
        
        current_vz = u*math.sin(theta) - v*math.sin(phi)*math.cos(theta) - w*math.cos(phi)*math.cos(theta)
        current_ax = (V - self.last_V) / (self.dt * self.action_repeat)
        self.last_V = V
        
        err_h = self.target_alt - current_h
        err_v = self.target_vel - V
        
        # =================================================================
        # 👑 引入 TECS 能量解耦感知
        # =================================================================
        E_t_target = self.target_alt + (self.target_vel**2) / (2 * self.g)
        E_t_current = current_h + (V**2) / (2 * self.g)
        err_Et = E_t_target - E_t_current  # 总能量误差
        
        E_d_target = self.target_alt - (self.target_vel**2) / (2 * self.g)
        E_d_current = current_h - (V**2) / (2 * self.g)
        err_Ed = E_d_target - E_d_current  # 能量分配误差
        
        alpha = math.degrees(math.atan2(w, u))
        gamma = math.degrees(theta) - alpha
        
        obs = np.array([
            err_Et / 100.0,      # PPO 将学会：看到 err_Et 大，就推油门
            err_Ed / 100.0,      # PPO 将学会：看到 err_Ed 大，就拉机头
            err_h / 500.0,       
            current_vz / 10.0,         
            err_v / 50.0,              
            current_ax / 5.0,          
            gamma / 10.0,               
            alpha / 10.0,               
            math.degrees(self.sim.state[10]) / 10.0
        ], dtype=np.float32)
        return np.nan_to_num(obs, nan=0.0, posinf=1.0, neginf=-1.0)

    def step(self, action):
        self.current_step += 1
        
        # 严格一致的动作平滑机制
        action_diff = action - self.last_action
        self.last_action = action.copy()
        self.smoothed_action = 0.5 * self.smoothed_action + 0.5 * action
        
        target_pitch_ppo = ((self.smoothed_action[0] + 1.0) / 2.0) * 10.0 - 2.0
        target_throttle_ppo = ((self.smoothed_action[1] + 1.0) / 2.0) * 0.9 + 0.1
        
        for _ in range(self.action_repeat):
            # 严格一致的发动机涡轮迟滞
            tau_engine = 0.5 
            self.actual_throttle += (target_throttle_ppo - self.actual_throttle) * (self.dt / tau_engine)
            
            # 严格一致的二阶指令滤波器
            pitch_c_ddot = self.omega_n**2 * (target_pitch_ppo - self.pitch_c) - 2 * self.zeta * self.omega_n * self.pitch_c_dot
            self.pitch_c += self.pitch_c_dot * self.dt
            self.pitch_c_dot += pitch_c_ddot * self.dt
            
            u, v, w = self.sim.state[3], self.sim.state[4], self.sim.state[5]
            phi, theta = self.sim.state[6], self.sim.state[7]
            
            current_pitch = math.degrees(theta)
            current_q = math.degrees(self.sim.state[10])
            current_p = math.degrees(self.sim.state[9])
            current_d_ail = np.clip(1.0 * (0.0 - math.degrees(phi)) - 0.5 * current_p, -10.0, 10.0)
            
            # 物理探针
            controls_0 = {'d_flap_L': 0.0, 'd_flap_R': 0.0, 'd_ail_L': current_d_ail, 'd_ail_R': -current_d_ail, 'throttle': self.actual_throttle}
            controls_1 = {'d_flap_L': 1.0, 'd_flap_R': 1.0, 'd_ail_L': current_d_ail, 'd_ail_R': -current_d_ail, 'throttle': self.actual_throttle}
            f2_nom = get_current_derivatives(self.sim, controls_0) 
            q_dot_1 = get_current_derivatives(self.sim, controls_1)
            ce0_nom = q_dot_1 - f2_nom 
            if abs(ce0_nom) < 1e-2: ce0_nom = -1e-2 if ce0_nom <= 0 else 1e-2
            
            # 严格一致的 NSMC 追踪
            e_theta = current_pitch - self.pitch_c
            self.integral_e_theta = np.clip(self.integral_e_theta + e_theta * self.dt, -10.0, 10.0)
            
            u_total = self.nsmc.compute_control(e_theta, self.integral_e_theta, current_q, self.pitch_c_dot, pitch_c_ddot, f2_nom, ce0_nom, self.dt)
            
            controls = {'d_flap_L': u_total, 'd_flap_R': u_total, 'd_ail_L': current_d_ail, 'd_ail_R': -current_d_ail, 'throttle': self.actual_throttle}
            self.sim.step(self.dt, controls)
            
        current_h = -self.sim.state[2]
        u, v, w = self.sim.state[3], self.sim.state[4], self.sim.state[5]
        V = math.sqrt(u**2 + v**2 + w**2)
        
        # =================================================================
        # 👑 TECS 能量解耦奖励：基于能量，而非孤立的高/速
        # =================================================================
        E_t_target = self.target_alt + (self.target_vel**2) / (2 * self.g)
        E_t_current = current_h + (V**2) / (2 * self.g)
        err_Et = abs(E_t_target - E_t_current)
        
        E_d_target = self.target_alt - (self.target_vel**2) / (2 * self.g)
        E_d_current = current_h - (V**2) / (2 * self.g)
        err_Ed = abs(E_d_target - E_d_current)
        
        # 奖励总能误差（督促推/收油门）和 分配误差（督促俯仰姿态）
        R_Et = math.exp(-0.5 * (err_Et / 50.0)**2) * 3.0 
        R_Ed = math.exp(-0.5 * (err_Ed / 50.0)**2) * 3.0 
        
        penalty = 0.5 * (action_diff[0]**2) + 0.2 * (action_diff[1]**2) 
        reward = R_Et + R_Ed - penalty
        
        terminated = current_h < 500.0 or current_h > 10000.0 or V < 100.0 or V > 400.0
        truncated = self.current_step >= self.max_steps
            
        return self._get_obs(), float(reward), terminated, truncated, {}

if __name__ == "__main__":
    from stable_baselines3.common.vec_env import SubprocVecEnv
    print("=====================================================")
    print("  启动【TECS能量解耦】PPO训练：消除 Sim2Eval 100% 缝隙")
    print("=====================================================")
    
    num_cpu = 14 
    env = make_vec_env(MIMOX47BEnv, n_envs=num_cpu, vec_env_cls=SubprocVecEnv)
    
    model = PPO(
        "MlpPolicy", env, verbose=1, learning_rate=3e-4, 
        n_steps=2048, batch_size=256, gamma=0.99, ent_coef=0.01,
        policy_kwargs=dict(net_arch=dict(pi=[256, 256], vf=[256, 256]))
    )
    
    eval_callback = EvalCallback(
        env, best_model_save_path='./rl_models/mimo_tecs/',
        log_path='./rl_logs/', eval_freq=2000, deterministic=True, render=False
    )
    
    model.learn(total_timesteps=2000000, callback=eval_callback)