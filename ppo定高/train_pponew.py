#coding=utf-8
import numpy as np
import math
import gymnasium as gym
from gymnasium import spaces
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.env_util import make_vec_env
import warnings

# 导入你修改过加入了 throttle 的神经引擎
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
        self.max_steps = 1000  # 稍微延长步数，给速度控制留出响应时间
        
        # =========================================================
        # 【状态空间：7 维感知】
        # 为了让 PPO 学会你说的三种映射，它必须观测：
        # [高度误差, 高度积分, 速度误差, 速度积分, 航迹角(gamma), 迎角(alpha), 俯仰角速度(q)]
        # =========================================================
        high = np.array([5.0, 5.0, 5.0, 5.0, 2.0, 2.0, 2.0], dtype=np.float32)
        self.observation_space = spaces.Box(-high, high, dtype=np.float32)
        
        # =========================================================
        # 【动作空间：2 维输出】
        # action[0]: 目标迎角 (Target AoA)
        # action[1]: 目标油门 (Target Throttle)
        # =========================================================
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(2,), dtype=np.float32)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.current_step = 0
        
        # 【必备：双重积分记忆】
        self.integral_h = 0.0 
        self.integral_v = 0.0
        self.integral_alpha = 0.0
        self.last_action = np.array([0.0, 0.0])
        
        # 随机化目标高度与目标速度
        self.target_alt = np.random.uniform(1500.0, 3500.0)
        self.target_vel = np.random.uniform(220.0, 280.0)
        
        init_h = self.target_alt + np.random.uniform(-300.0, 300.0)
        init_v = self.target_vel + np.random.uniform(-30.0, 30.0)
        
        self.sim.set_initial_state(h_m=init_h, V_mps=init_v, theta_deg=0.7, alpha_deg=0.7)
        
        return self._get_obs(), {}

    def _get_obs(self):
        u, v, w = self.sim.state[3], self.sim.state[4], self.sim.state[5]
        phi, theta = self.sim.state[6], self.sim.state[7]
        current_h = -self.sim.state[2]
        V = math.sqrt(u**2 + v**2 + w**2)
        
        err_h = self.target_alt - current_h
        err_v = self.target_vel - V
        
        # 航迹角 gamma = theta - alpha
        alpha = math.degrees(math.atan2(w, u))
        gamma = math.degrees(theta) - alpha
        
        obs = np.array([
            err_h / 500.0,       
            self.integral_h / 1000.0,  
            err_v / 50.0,
            self.integral_v / 100.0,
            gamma / 10.0,               # PPO 将自动发掘 err_h -> gamma 的关系
            alpha / 10.0,               # PPO 将自动发掘 gamma -> alpha 的关系
            math.degrees(self.sim.state[10]) / 10.0
        ], dtype=np.float32)
        return obs

    def step(self, action):
        self.current_step += 1
        
        # 动作解算
        # a[0] -> 目标迎角 [-2度, +6度]
        target_alpha = ((action[0] + 1.0) / 2.0) * 8.0 - 2.0
        # a[1] -> 目标油门 [0.1, 1.0] (不设为0防熄火)
        target_throttle = ((action[1] + 1.0) / 2.0) * 0.9 + 0.1
        
        for _ in range(self.action_repeat):
            u, v, w = self.sim.state[3], self.sim.state[4], self.sim.state[5]
            current_alpha = math.degrees(math.atan2(w, u))
            current_q = math.degrees(self.sim.state[10])
            current_p = math.degrees(self.sim.state[9])
            current_roll = math.degrees(self.sim.state[6])
            
            # 【内环】：纯数学 PID 追踪目标迎角 (Alpha)
            err_alpha = target_alpha - current_alpha
            self.integral_alpha = np.clip(self.integral_alpha + err_alpha * self.dt, -10.0, 10.0)
            
            # 注意：迎角PID与俯仰角PID方向一致，但参数需微调
            d_flap = -2.5 * err_alpha + 0.8 * current_q - 0.5 * self.integral_alpha
            d_flap = np.clip(d_flap, -15.0, 15.0)
            
            d_ail = np.clip(1.0 * (0.0 - current_roll) - 0.5 * current_p, -10.0, 10.0)
            
            # 传给物理引擎：包含襟翼、副翼和【油门】
            controls = {'d_flap_L': d_flap, 'd_flap_R': d_flap, 'd_ail_L': d_ail, 'd_ail_R': -d_ail, 'throttle': target_throttle}
            self.sim.step(self.dt, controls)
            
        current_h = -self.sim.state[2]
        u, v, w = self.sim.state[3], self.sim.state[4], self.sim.state[5]
        V = math.sqrt(u**2 + v**2 + w**2)
        
        err_h = self.target_alt - current_h
        err_v = self.target_vel - V
        
        self.integral_h = np.clip(self.integral_h + err_h * (self.dt * self.action_repeat), -1000.0, 1000.0)
        self.integral_v = np.clip(self.integral_v + err_v * (self.dt * self.action_repeat), -100.0, 100.0)
        
        # =========================================================
        # 【双轨奖励机制】：既要高度，又要速度，还要平稳
        # =========================================================
        # 1. 高度追踪奖励
        R_h = math.exp(-((err_h / 20.0)**2)) * 1.5
        if abs(err_h) < 5.0: R_h += 1.0
        
        # 2. 速度追踪奖励
        R_v = math.exp(-((err_v / 5.0)**2)) * 1.5
        if abs(err_v) < 1.0: R_v += 1.0
        
        # 3. 动作平滑度惩罚 (防机长疯狂抽搐摇杆和推拉油门)
        action_diff = np.abs(action - self.last_action)
        self.last_action = action.copy()
        penalty = (action_diff[0] * 0.1) + (action_diff[1] * 0.05)
        
        reward = R_h + R_v - penalty
        
        terminated = current_h < 500.0 or current_h > 5000.0 or V < 100.0 or V > 400.0
        truncated = self.current_step >= self.max_steps
            
        return self._get_obs(), float(reward), terminated, truncated, {}


if __name__ == "__main__":
    from stable_baselines3.common.vec_env import SubprocVecEnv
    
    print("=====================================================")
    print("  启动【MIMO 双通道】PPO 训练：同时掌握高度与能量")
    print("=====================================================")
    
    num_cpu = 14 
    env = make_vec_env(MIMOX47BEnv, n_envs=num_cpu, vec_env_cls=SubprocVecEnv)
    
    # 网络架构加宽，因为解耦双变量映射需要更高的网络容量
    model = PPO(
        "MlpPolicy", env, verbose=1, learning_rate=3e-4, 
        n_steps=2048, batch_size=512, gamma=0.99, 
        policy_kwargs=dict(net_arch=[256, 256])
    )
    
    eval_callback = EvalCallback(
        env, best_model_save_path='./rl_models/mimo/',
        log_path='./rl_logs/', eval_freq=2000, 
        deterministic=True, render=False
    )
    
    print(f"\n🚀 正在让 PPO 自行顿悟能量守恒与轨迹几何映射...")
    # 由于需要摸索两个控制变量的解耦，建议给它 300万步的顿悟时间
    model.learn(total_timesteps=3000000, callback=eval_callback)