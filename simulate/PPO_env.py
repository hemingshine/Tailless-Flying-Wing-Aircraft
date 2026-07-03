import sys

import numpy as np
import math
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical
sys.path.append(r"C:\\Users\\13438\\Desktop\\飞行轨迹\simulate\\flight_simulate.py")
# 导入你写好的类（假设你的原文件名为 flight_simulate.py）
from flight_simulate import FlightSimulator6DOF, HybridAeroDatabase, EngineDatabase

class FlightEnvRL:
    """
    带飞行包线约束与动作平滑的 6-DOF 强化学习环境 (脱离 Gym 依赖，更轻量)
    """
    def __init__(self, aero_db, engine_db, global_params, model_codes, max_steps=1000, dt=0.02):
        self.aero_db = aero_db
        self.engine_db = engine_db
        self.global_params = global_params
        self.model_codes = model_codes
        
        self.dt = dt
        self.max_steps = max_steps
        self.current_step = 0
        
        # 动作空间维度和状态空间维度
        self.action_dim = len(self.model_codes)
        self.state_dim = 10 
        
        self.sim = None
        self.last_pn = 0.0
        self.last_pe = 0.0
        self.last_action = None # 用于动作平滑惩罚

    def reset(self):
        self.sim = FlightSimulator6DOF(self.aero_db, self.engine_db, self.global_params)
        # 初始状态：高度10000m, 速度250m/s, 俯仰角3度
        self.sim.set_initial_state(h_m=2000.0, V_mps=250.0, theta_deg=3.0)
        
        self.current_step = 0
        self.last_pn = self.sim.state[0]
        self.last_pe = self.sim.state[1]
        self.last_action = None
        
        return self._get_normalized_obs()

    # def step(self, action_idx):
    #     model_code = self.model_codes[action_idx]
        
    #     # RK4步进
    #     result = self.sim.step_rk4(self.dt, model_code)
    #     self.current_step += 1
        
    #     current_pn = self.sim.state[0]
    #     current_pe = self.sim.state[1]
    #     altitude = result['Altitude']
    #     pitch = result['Pitch'] # 已经是度数
        
    #     # 1. 基础奖励：计算这一步水平面飞行的距离，并大幅缩放！
    #     delta_n = current_pn - self.last_pn
    #     delta_e = current_pe - self.last_pe
    #     distance_flown = math.sqrt(delta_n**2 + delta_e**2)
        
    #     # 【修改点 1】：将距离缩放。比如每次前进 5 米，除以 10 后变成 0.5
    #     reward = distance_flown / 10.0 
        
    #     self.last_pn = current_pn
    #     self.last_pe = current_pe
        
    #     # 2. 动作平滑惩罚
    #     # if self.last_action is not None and action_idx != self.last_action:
    #     #     reward -= 0.5  # 【修改点 2】：惩罚等比例缩小
    #     # self.last_action = action_idx
        
    #     # 3. 飞行包线约束与终止条件
    #     done = False
        
    #     # 约束1：高度 (0 ~ 20000m)
    #     if altitude <= 0.0 or altitude > 20000.0:
    #         reward -= 50.0 # 【修改点 3】：将 -5000 改为 -50。不要用核弹惩罚，AI知道疼就行。
    #         done = True
            
    #     # 约束2：俯仰角 (-3° ~ 15°)
    #     if pitch < -3.0 or pitch > 15.0:
    #         reward -= 50.0 # 同上
    #         done = True

    #     # 约束3：最大时间步
    #     if self.current_step >= self.max_steps:
    #         done = True
            
    #     return self._get_normalized_obs(), reward, done

    def step(self, action_idx):
        model_code = self.model_codes[action_idx]
        
        # 【核心修改 1】：进一步降低动作保持时间，每 0.1 秒让 AI 微调一次动作！
        frame_skip = 5 
        
        accumulated_reward = 0.0
        done = False
        target_alt = 2000.0 
        
        for _ in range(frame_skip):
            result = self.sim.step_rk4(self.dt, model_code)
            self.current_step += 1
            
            current_pn = self.sim.state[0]
            current_pe = self.sim.state[1]
            altitude = result['Altitude']
            pitch = result['Pitch'] 
            
            # 获取迎角 (Alpha) 和 俯仰角速度 (q)
            u, w = self.sim.state[3], self.sim.state[5]
            alpha = math.degrees(math.atan2(w, u)) if u != 0 else 0.0
            q = self.sim.state[10] # 俯仰角速度 rad/s
            
            delta_n = current_pn - self.last_pn
            delta_e = current_pe - self.last_pe
            
            forward_reward = delta_n / 10.0
            drift_penalty = abs(delta_e) / 20.0 
            
            alt_error = abs(altitude - target_alt)
            if alt_error < 100.0:
                alt_reward = 0.2  
            else:
                alt_reward = - (alt_error / 100.0) * 0.1 
                
            # 【核心修改 2】：添加“角速度惩罚”，惩罚机头剧烈甩动！
            # q 越大，扣分越狠，逼迫它飞得平稳
            q_penalty = abs(q) * 2.0
            
            accumulated_reward += (forward_reward - drift_penalty + alt_reward - q_penalty)
            
            self.last_pn = current_pn
            self.last_pe = current_pe
            
            # 【核心修改 3】：加入迎角 (Alpha) 死亡红线！
            # 正常飞机迎角很少超过 15度 或低于 -10度。一旦超过，直接处死！
            if altitude <= 1500.0 or altitude > 5000.0 or pitch < -20.0 or pitch > 20.0 or alpha > 25.0 or alpha < -20.0:
                accumulated_reward -= 50.0 
                done = True
                
                # 黑匣子打印（加入 q 的观察）
                if self.current_step % 200 == 0 or altitude <= 1500.0 or abs(pitch) >= 20.0 or alpha > 15.0 or alpha < -10.0:
                     print(f"[{'坠毁' if altitude<=1500 else '失控'}] 高度:{altitude:.1f}m, 俯仰:{pitch:.1f}°, 迎角:{alpha:.1f}°, 角速度:{math.degrees(q):.1f}°/s")
                break
                
            if self.current_step >= self.max_steps:
                done = True
                break

        self.last_action = action_idx
        return self._get_normalized_obs(), accumulated_reward, done

    def _get_normalized_obs(self):
        """将状态归一化到 [-1, 1] 附近，加速神经网络收敛"""
        state = self.sim.state
        pn, pe, pd, u, v, w, phi, theta, psi, p, q, r = state
        
        h = -pd
        V = math.sqrt(u**2 + v**2 + w**2)
        alpha = math.degrees(math.atan2(w, u)) if u != 0 else 0.0
        beta = math.degrees(math.asin(v / V)) if V != 0 else 0.0
        
        # 高度归一化: 映射 0~20000m 到 -1~1
        h_norm = (h - 10000.0) / 10000.0 
        # 速度归一化: 假设极限速度 500m/s，映射到大概 -1~1
        v_norm = (V - 250.0) / 250.0 
        
        # 角度归一化: 除以预期最大角度
        alpha_norm = alpha / 30.0
        beta_norm = beta / 30.0
        phi_norm = math.degrees(phi) / 180.0
        # 俯仰角期望在 -3 到 15 之间，用 20 除一下即可
        theta_norm = math.degrees(theta) / 20.0 
        psi_norm = math.degrees(psi) / 180.0
        
        # 角速度归一化 (rad/s)
        p_norm, q_norm, r_norm = p / 2.0, q / 2.0, r / 2.0 
        
        obs = np.array([
            h_norm, v_norm, alpha_norm, beta_norm,
            phi_norm, theta_norm, psi_norm,
            p_norm, q_norm, r_norm
        ], dtype=np.float32)
        
        return obs
    


    # --- PPO 经验回放池 ---
class RolloutBuffer:
    def __init__(self):
        self.actions = []
        self.states = []
        self.logprobs = []
        self.rewards = []
        self.is_terminals = []
    
    def clear(self):
        del self.actions[:]
        del self.states[:]
        del self.logprobs[:]
        del self.rewards[:]
        del self.is_terminals[:]

# --- Actor-Critic 神经网络 ---
class ActorCritic(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(ActorCritic, self).__init__()
        
        # 独立的 Actor 网络
        self.actor = nn.Sequential(
            nn.Linear(state_dim, 128),
            nn.Tanh(),
            nn.Linear(128, 128),
            nn.Tanh(),
            nn.Linear(128, 64),
            nn.Tanh(),
            nn.Linear(64, action_dim),
            nn.Softmax(dim=-1)
        )
        
        # 独立的 Critic 网络
        self.critic = nn.Sequential(
            nn.Linear(state_dim, 128),
            nn.Tanh(),
            nn.Linear(128, 128),
            nn.Tanh(),
            nn.Linear(128, 64),
            nn.Tanh(),
            nn.Linear(64, 1)
        )
        
    def act(self, state):
        state = torch.from_numpy(state).float().unsqueeze(0)
        action_probs = self.actor(state)
        dist = Categorical(action_probs)
        
        action = dist.sample()
        action_logprob = dist.log_prob(action)
        
        return action.item(), action_logprob.item()
    
    def evaluate(self, state, action):
        action_probs = self.actor(state)
        dist = Categorical(action_probs)
        action_logprobs = dist.log_prob(action)
        dist_entropy = dist.entropy()
        
        state_values = self.critic(state)
        
        return action_logprobs, state_values, dist_entropy
    
# --- PPO 核心算法类 ---
class PPOAgent:
    def __init__(self, state_dim, action_dim, lr=3e-4, gamma=0.99, K_epochs=4, eps_clip=0.2):
        self.gamma = gamma
        self.eps_clip = eps_clip
        self.K_epochs = K_epochs
        
        self.buffer = RolloutBuffer()
        
        self.policy = ActorCritic(state_dim, action_dim)
        self.optimizer = optim.Adam(self.policy.parameters(), lr=lr)
        
        self.policy_old = ActorCritic(state_dim, action_dim)
        self.policy_old.load_state_dict(self.policy.state_dict())
        
        self.MseLoss = nn.MSELoss()

    def select_action(self, state):
        with torch.no_grad():
            action, action_logprob = self.policy_old.act(state)
            
        self.buffer.states.append(state)
        self.buffer.actions.append(action)
        self.buffer.logprobs.append(action_logprob)
        
        return action

    def update(self):
        # 将 buffer 数据转为 tensor
        old_states = torch.tensor(np.array(self.buffer.states), dtype=torch.float32)
        old_actions = torch.tensor(np.array(self.buffer.actions), dtype=torch.float32)
        old_logprobs = torch.tensor(np.array(self.buffer.logprobs), dtype=torch.float32)
        rewards = self.buffer.rewards
        is_terminals = self.buffer.is_terminals

        # ------------------------------------------------------------------
        # 1. 计算 GAE (广义优势估计)
        # ------------------------------------------------------------------
        with torch.no_grad():
            # 用 Critic 评估所有旧状态的价值
            _, state_values, _ = self.policy.evaluate(old_states, old_actions)
            state_values = state_values.squeeze().numpy()
            
            # 为了计算最后一步的优势，追加一个 0
            state_values = np.append(state_values, 0.0) 
            
        advantages = []
        gae = 0
        gamma = self.gamma
        lam = 0.95 # GAE 的 lambda 参数，通常取 0.95 到 0.99
        
        for i in reversed(range(len(rewards))):
            if is_terminals[i]:
                gae = 0 # 回合结束，优势清零
                next_value = 0
            else:
                next_value = state_values[i + 1]
                
            # TD Error: r_t + gamma * V(s_{t+1}) - V(s_t)
            delta = rewards[i] + gamma * next_value - state_values[i]
            # GAE: delta_t + gamma * lambda * GAE_{t+1}
            gae = delta + gamma * lam * gae
            advantages.insert(0, gae)
            
        advantages = torch.tensor(advantages, dtype=torch.float32)
        
        # 【修改点1】：先计算 Returns (用未归一化的优势计算真实的目标价值！)
        returns = advantages + torch.tensor(state_values[:-1], dtype=torch.float32)

        # 【修改点2】：然后再归一化 Advantage (为 Actor 提供稳定的梯度方向！)
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-7)

        # ------------------------------------------------------------------
        # 2. Mini-batch PPO 更新 (核心改进点)
        # ------------------------------------------------------------------
        batch_size = len(old_states)
        mini_batch_size = 64 # 推荐的 mini-batch 大小
        
        for _ in range(self.K_epochs):
            # 生成随机索引来打乱数据
            indices = torch.randperm(batch_size)
            
            for start in range(0, batch_size, mini_batch_size):
                end = start + mini_batch_size
                mb_indices = indices[start:end]
                
                mb_states = old_states[mb_indices]
                mb_actions = old_actions[mb_indices]
                mb_old_logprobs = old_logprobs[mb_indices]
                mb_advantages = advantages[mb_indices]
                mb_returns = returns[mb_indices]
                
                # 重新评估当前的 Mini-batch
                logprobs, mb_state_values, dist_entropy = self.policy.evaluate(mb_states, mb_actions)
                mb_state_values = torch.squeeze(mb_state_values)
                
                # 寻找比率 (pi_theta / pi_theta_old)
                ratios = torch.exp(logprobs - mb_old_logprobs)
                
                # 计算 Surrogate Loss 并进行 Clip
                surr1 = ratios * mb_advantages
                surr2 = torch.clamp(ratios, 1 - self.eps_clip, 1 + self.eps_clip) * mb_advantages
                
                # Actor 损失 (带截断)
                actor_loss = -torch.min(surr1, surr2).mean()
                # Critic 损失 (MSE)
                critic_loss = self.MseLoss(mb_state_values, mb_returns)
                
                # 组合总 Loss
                loss = actor_loss + 0.5 * critic_loss - 0.05 * dist_entropy.mean()
                
                # 梯度下降
                self.optimizer.zero_grad()
                loss.backward()
                # (可选) 梯度裁剪，防止物理环境产生的极大梯度破坏网络
                torch.nn.utils.clip_grad_norm_(self.policy.parameters(), max_norm=0.5)
                self.optimizer.step()
            
        # 更新完毕，老策略同步为新策略并清空 Buffer
        self.policy_old.load_state_dict(self.policy.state_dict())
        self.buffer.clear()

        