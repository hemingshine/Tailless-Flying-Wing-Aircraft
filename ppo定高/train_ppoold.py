#coding=utf-8
import numpy as np
import math
import gymnasium as gym
from gymnasium import spaces
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.env_util import make_vec_env

# 导入神经引擎和模拟器
from fly import NeuralAeroDatabase, EngineDatabase, FlightSimulator6DOF

class GeneralX47BEnv(gym.Env):
    def __init__(self):
        super(GeneralX47BEnv, self).__init__()
        
        aircraft_params = {
            'S': 88.58, 'b': 18.9, 'c_bar': 4.6, 'mass': 14000,
            'Ixx': 313220.6, 'Iyy': 273435.3, 'Izz': 392903.9, 'Ixz': 0.0
        }
        self.flight_db = NeuralAeroDatabase()
        self.flight_db._load_from_pickle('aero_surrogate.pth')
        self.engine_db = EngineDatabase()
        self.engine_db.load1('engine.pkl')

        self.sim = FlightSimulator6DOF(self.flight_db, self.engine_db, aircraft_params)
        
        self.dt_physics = 0.02           
        self.action_repeat = 10          
        self.max_steps =800 # 增加步数，因为有的目标可能很远
        self.current_step = 0
        self.target_alt = 2500.0 # 占位，在 reset 中会被随机替换
        
        # 【核心改动 1】：状态空间增加到 6 维，加入了归一化的速度参数
        # [高度误差/1000, 速度误差分布, 垂直速度/10, 俯仰角/10, 俯仰角速度/10, 迎角/10]
        high = np.array([5.0, 2.0, 5.0, 2.0, 2.0, 2.0], dtype=np.float32)
        self.observation_space = spaces.Box(-high, high, dtype=np.float32)
        
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(1,), dtype=np.float32)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.current_step = 0
        self.integral_theta = 0.0 
        
        # 【核心改动 2】：领域随机化 (Domain Randomization)
        # 1. 目标高度在 1500m 到 3500m 之间随机
        self.target_alt = np.random.uniform(1500.0, 3500.0)
        
        # 2. 初始高度在 1000m 到 4000m 之间随机 (保证和目标高度至少有 200m 落差)
        init_h = self.target_alt
        while abs(init_h - self.target_alt) < 200.0:
            init_h = np.random.uniform(1000.0, 4000.0)
            
        # 3. 初始速度在 200m/s 到 300m/s 之间随机
        init_v = np.random.uniform(200.0, 300.0)
        
        # 赋予飞机随机的出生状态
        self.sim.set_initial_state(h_m=init_h, V_mps=init_v, theta_deg=0.7, alpha_deg=0.7)
        
        return self._get_obs(), {}

    def _get_obs(self):
        u, v, w = self.sim.state[3], self.sim.state[4], self.sim.state[5]
        phi, theta = self.sim.state[6], self.sim.state[7]
        
        current_h = -self.sim.state[2]
        V = math.sqrt(u**2 + v**2 + w**2)
        vz = u*math.sin(theta) - v*math.sin(phi)*math.cos(theta) - w*math.cos(phi)*math.cos(theta)
        
        err_h = self.target_alt - current_h
        pitch_deg = math.degrees(theta)
        q_deg = math.degrees(self.sim.state[10])
        alpha_deg = math.degrees(math.atan2(w, u))
        
        obs = np.array([
            err_h / 1000.0,       # 高度误差放缩，最大正负 3.0 左右
            (V - 250.0) / 50.0,   # 速度归一化，围绕 250m/s 上下波动
            vz / 10.0,
            pitch_deg / 10.0,
            q_deg / 10.0,
            alpha_deg / 10.0
        ], dtype=np.float32)
        
        return obs

    def step(self, action):
        self.current_step += 1
        
        # 动作映射 [-1, 1] -> [-2度, +5度] 增加下压机头的幅度，以应对需要下降的任务
        target_pitch = ((action[0] + 1.0) / 2.0) * 7.0 - 2.0
        
        # 内环 PID
        for _ in range(self.action_repeat):
            current_pitch = math.degrees(self.sim.state[7])
            current_roll = math.degrees(self.sim.state[6])
            current_q = math.degrees(self.sim.state[10])
            current_p = math.degrees(self.sim.state[9])
            
            err_theta = target_pitch - current_pitch
            self.integral_theta = np.clip(self.integral_theta + err_theta * self.dt_physics, -5, 5.0)
            
            d_flap = -1.5 * err_theta + 0.8 * current_q - 0.2 * self.integral_theta
            d_flap = np.clip(d_flap, -10.0, 10.0)
            
            d_ail = 1.0 * (0.0 - current_roll) - 0.5 * current_p
            d_ail = np.clip(d_ail, -10.0, 10.0)
            
            controls = {'d_flap_L': d_flap, 'd_flap_R': d_flap, 'd_ail_L': d_ail, 'd_ail_R': -d_ail}
            self.sim.step(self.dt_physics, controls)
            
        obs = self._get_obs()
        current_h = -self.sim.state[2]
        err_h = self.target_alt - current_h
        u, v, w = self.sim.state[3], self.sim.state[4], self.sim.state[5]
        phi, theta = self.sim.state[6], self.sim.state[7]
        vz = u*math.sin(theta) - v*math.sin(phi)*math.cos(theta) - w*math.cos(phi)*math.cos(theta)
        base_reward = math.exp(-abs(err_h) / 500.0) * 0.5 + math.exp(-((err_h / 20.0)**2)) * 1.5
        
        # 2. 完美驻留奖励 (Deadband Bonus)
        # 如果高度误差小于 5 米，给予额外的巨大奖励，鼓励它死死咬住目标
        if abs(err_h) < 5.0:
            base_reward += 1.0
            
        # 3. 稳态误差累积惩罚 (Integral Penalty)
        # 为了防止它在 20m 处躺平，如果在 20m 外停留，给予轻微的持续惩罚
        if abs(err_h) > 10.0:
             base_reward -= 0.1
             
        # 4. 动作平滑度约束
        # 只惩罚动作的变化率 (Jerk)，不惩罚动作的绝对值，让它敢于打舵面去配平！
        if not hasattr(self, 'last_action'):
            self.last_action = action[0]
        action_diff = abs(action[0] - self.last_action)
        self.last_action = action[0]
        
        # 动作惩罚大幅降低，重点惩罚来回抖动
        penalty = action_diff * 0.02 
        
        # =========================================================
        # 【神来之笔】：动态垂直阻尼 (Dynamic Vertical Damping)
        # =========================================================
        # 利用高斯函数：当误差 err_h 很大时，权重接近 0；当误差接近 0 时，权重最大(0.1)
        damping_weight = math.exp(-((err_h / 50.0)**2)) * 0.1
        
        # 只有在接近目标高度时，才对垂直速度进行强力惩罚，强制它轻柔“停靠”
        penalty += abs(vz) * damping_weight
        
        reward = base_reward - penalty
        
        # # 【核心改动 3】：复合奖励函数
        # # 远距离时：使用绝对值倒数提供平缓的吸引力 (防止梯度消失)
        # # 近距离时 (<50m)：使用高斯函数提供巨额的精确驻留奖励
        # reward = math.exp(-abs(err_h) / 500.0) * 0.5 + math.exp(-((err_h / 20.0)**2)) * 1.5
        
        # reward -= abs(action[0]) * 0.05 # 动作平滑惩罚
        
        terminated = False
        truncated = False
        
        # 放宽安全边界，因为随机出生点可能很高或很低
        if current_h < 500.0 or current_h > 5000.0:
            terminated = True
            reward -= 50.0
            
        if self.current_step >= self.max_steps:
            truncated = True
            
        return obs, float(reward), terminated, truncated, {}


if __name__ == "__main__":
    # 导入多进程封装器
    from stable_baselines3.common.vec_env import SubprocVecEnv
    
    print("===========================================")
    print("  启动【通用型】PPO 训练 (16核并行加速版)")
    print("===========================================")
    
    # 获取你的 CPU 线程数
    num_cpu = 14 
    
    # 使用 SubprocVecEnv 开启真正的多进程并发
    env = make_vec_env(GeneralX47BEnv, n_envs=num_cpu, vec_env_cls=SubprocVecEnv)
    
    # 稍微调大 batch_size 以喂饱 GPU 的训练端
    model = PPO(
        "MlpPolicy", env, verbose=1, learning_rate=3e-4, 
        n_steps=2048, batch_size=1024, gamma=0.99, 
        policy_kwargs=dict(net_arch=[256, 256])
    )
    
    eval_callback = EvalCallback(
        env, best_model_save_path='./rl_models/',
        log_path='./rl_logs/', eval_freq=2000, 
        deterministic=True, render=False
    )
    
    print(f"\n🚀 火力全开！已启动 {num_cpu} 个并行物理引擎...")
    model.learn(total_timesteps=1000000, callback=eval_callback)