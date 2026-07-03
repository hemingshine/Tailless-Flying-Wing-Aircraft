#coding=utf-8
import numpy as np
import math
import gymnasium as gym
from gymnasium import spaces
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.env_util import make_vec_env
import warnings

# 导入你的神经引擎
from fly import NeuralAeroDatabase, EngineDatabase, FlightSimulator6DOF

warnings.filterwarnings('ignore')

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
        
        self.dt = 0.02           
        self.action_repeat = 10          
        self.max_steps = 1000  
        
        # =================================================================
        # 👑 核心升级 1：九维全感知观测空间！
        # 加入了生死攸关的 Vz (爬升率) 和 Ax (纵向加速度)，补齐物理学的 D 项！
        # =================================================================
        high = np.array([5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 2.0, 2.0, 2.0], dtype=np.float32)
        self.observation_space = spaces.Box(-high, high, dtype=np.float32)
        
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(2,), dtype=np.float32)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.current_step = 0
        
        self.integral_h = 0.0 
        self.integral_v = 0.0
        
        self.last_action = np.array([0.0, 0.0], dtype=np.float32)
        
        self.target_alt = np.random.uniform(1500.0, 3500.0)
        self.target_vel = np.random.uniform(220.0, 280.0)
        
        init_h = self.target_alt + np.random.uniform(-300.0, 300.0)
        init_v = self.target_vel + np.random.uniform(-30.0, 30.0)
        
        self.last_V = init_v # 用于计算加速度
        
        self.sim.set_initial_state(h_m=init_h, V_mps=init_v, theta_deg=2.0, alpha_deg=2.0)
        
        return self._get_obs(), {}

    def _get_obs(self):
        u, v, w = self.sim.state[3], self.sim.state[4], self.sim.state[5]
        phi, theta = self.sim.state[6], self.sim.state[7]
        current_h = -self.sim.state[2]
        V = math.sqrt(u**2 + v**2 + w**2)
        
        # 精确计算物理导数
        current_vz = u*math.sin(theta) - v*math.sin(phi)*math.cos(theta) - w*math.cos(phi)*math.cos(theta)
        current_ax = (V - self.last_V) / (self.dt * self.action_repeat)
        self.last_V = V
        
        err_h = self.target_alt - current_h
        err_v = self.target_vel - V
        
        alpha = math.degrees(math.atan2(w, u))
        gamma = math.degrees(theta) - alpha
        
        obs = np.array([
            err_h / 500.0,             # 高度误差 (P项)
            current_vz / 10.0,         # 爬升率 (D项) -> 彻底消灭高度震荡！
            self.integral_h / 1000.0,  # 高度积分 (I项)
            err_v / 50.0,              # 速度误差 (P项)
            current_ax / 5.0,          # 加速度 (D项) -> 彻底消灭油门震荡！
            self.integral_v / 100.0,   # 速度积分 (I项)
            gamma / 10.0,               
            alpha / 10.0,               
            math.degrees(self.sim.state[10]) / 10.0
        ], dtype=np.float32)
        return obs

    def step(self, action):
        self.current_step += 1
        
        # 平滑惩罚：现在 PPO 有了阻尼感知，不需要地狱级惩罚也能飞得很稳
        action_diff = action - self.last_action
        self.last_action = action.copy()
        penalty = 0.2 * (action_diff[0]**2) + 0.1 * (action_diff[1]**2)
        

        # =================================================================
        # 👑 核心升级 2：动作定心 (Action Centering)
        # action = [0, 0] 时，飞机输出：俯仰 2°，油门 0.5。即标准巡航姿态！
        # =================================================================
        target_pitch = action[0] * 5.0 + 2.0     # 范围 [-3°, 7°]
        target_throttle = action[1] * 0.4 + 0.5  # 范围 [0.1, 0.9]
        
        # 内环追踪 (在训练环境用 PID 快速模拟 NSMC 的刚性)
        for _ in range(self.action_repeat):
            u, v, w = self.sim.state[3], self.sim.state[4], self.sim.state[5]
            phi, theta = self.sim.state[6], self.sim.state[7]
            
            current_pitch = math.degrees(theta)
            current_q = math.degrees(self.sim.state[10])
            current_p = math.degrees(self.sim.state[9])
            current_roll = math.degrees(phi)
            
            err_theta = target_pitch - current_pitch
            # 使用强力 PD 参数逼近 NSMC 的瞬态响应
            d_flap = -5.0 * err_theta + 1.2 * current_q 
            d_flap = np.clip(d_flap, -20.0, 20.0)
            
            d_ail = np.clip(1.0 * (0.0 - current_roll) - 0.5 * current_p, -10.0, 10.0)
            
            controls = {'d_flap_L': d_flap, 'd_flap_R': d_flap, 'd_ail_L': d_ail, 'd_ail_R': -d_ail, 'throttle': target_throttle}
            self.sim.step(self.dt, controls)
            
        current_h = -self.sim.state[2]
        u, v, w = self.sim.state[3], self.sim.state[4], self.sim.state[5]
        V = math.sqrt(u**2 + v**2 + w**2)
        
        err_h = self.target_alt - current_h
        err_v = self.target_vel - V


        self.integral_h = np.clip(self.integral_h + err_h * (self.dt * self.action_repeat), -1000.0, 1000.0)
        self.integral_v = np.clip(self.integral_v + err_v * (self.dt * self.action_repeat), -100.0, 100.0)
        
        # =================================================================
        # 👑 核心升级 3：高斯平滑奖励场
        # 绝不使用阶跃奖励，让梯度永远连续，PPO 会顺着钟形曲线一路滑向最优解
        # =================================================================
        R_h = math.exp(-0.5 * (err_h / 20.0)**2) * 3.0
        R_v = math.exp(-0.5 * (err_v / 5.0)**2) * 2.0
        
        reward = R_h + R_v - penalty
        
        terminated = current_h < 500.0 or current_h > 5000.0 or V < 100.0 or V > 400.0
        truncated = self.current_step >= self.max_steps
            
        return self._get_obs(), float(reward), terminated, truncated, {}

if __name__ == "__main__":
    from stable_baselines3.common.vec_env import SubprocVecEnv
    
    print("=====================================================")
    print("  启动【MIMO 双通道】PPO 训练：九维上帝视角的王牌机长")
    print("=====================================================")
    
    num_cpu = 14 
    env = make_vec_env(MIMOX47BEnv, n_envs=num_cpu, vec_env_cls=SubprocVecEnv)
    
    # 恢复标准 PPO 强悍参数，因为观测空间已经物理完备！
    model = PPO(
        "MlpPolicy", env, verbose=1, learning_rate=3e-4, 
        n_steps=2048, batch_size=256, gamma=0.99, 
        policy_kwargs=dict(net_arch=dict(pi=[256, 256], vf=[256, 256]))
    )
    
    eval_callback = EvalCallback(
        env, best_model_save_path='./rl_models/mimo_ultimate/',
        log_path='./rl_logs/', eval_freq=2000, 
        deterministic=True, render=False
    )
    
    print(f"\n🚀 观测空间降维打击已开启，准备见证绝对平稳的收敛...")
    model.learn(total_timesteps=2000000, callback=eval_callback)