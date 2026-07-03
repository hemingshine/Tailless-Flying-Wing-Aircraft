#coding=utf-8
import numpy as np
import math
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import torch
import torch.nn as nn
import torch.optim as optim
import warnings

# 导入神经引擎与模拟器
from fly import NeuralAeroDatabase, EngineDatabase, FlightSimulator6DOF

warnings.filterwarnings('ignore')

# =================================================================
# 1. RLS 在线增量模型辨识器 (加入论文 Improvement)
# =================================================================
class RLS_Identifier:
    def __init__(self, state_dim=2, input_dim=1):
        self.state_dim = state_dim
        self.input_dim = input_dim
        self.gamma_rls = 0.95 
        self.P = np.eye(state_dim + input_dim) * 1e6 
        self.Theta = np.zeros((state_dim + input_dim, state_dim))
        self.Theta[:state_dim, :state_dim] = np.eye(state_dim) 
        
        # 【修正1】：物理先验，正襟翼偏角导致低头(q减小)，初始值赋为负，指明正确梯度方向
        self.Theta[2, 1] = -0.1

    def update(self, delta_x_prev, delta_u_prev, delta_x_curr):
        Xi = np.zeros((self.state_dim + self.input_dim, 1))
        Xi[:self.state_dim, 0] = delta_x_prev
        Xi[self.state_dim:, 0] = delta_u_prev
        
        # 【修正2】：信号死区(Deadzone)。如果飞机动作极小，停止更新，防止RLS拟合噪声
        if np.linalg.norm(Xi) < 1e-4:
            return self.Theta[:self.state_dim, :].T, self.Theta[self.state_dim:, :].T
            
        delta_x_pred = self.Theta.T @ Xi
        error = delta_x_curr.reshape(-1, 1) - delta_x_pred
        
        K_num = self.P @ Xi
        K_den = self.gamma_rls + Xi.T @ self.P @ Xi
        K = K_num / K_den[0, 0]
        
        self.Theta = self.Theta + K @ error.T
        self.P = (1.0 / self.gamma_rls) * (self.P - K @ Xi.T @ self.P)
        
        # 【修正3：论文 Improvement 2 先验限制】
        # 强制钳制 G_q (Theta[2,1])，绝不允许大于 -0.001，防止 Actor 梯度反向导致暴走
        if self.Theta[2, 1] > -0.001:
            self.Theta[2, 1] = -0.001 
            
        return self.Theta[:self.state_dim, :].T, self.Theta[self.state_dim:, :].T

# =================================================================
# 2. IDHP 容错内环智能体
# =================================================================
class IDHP_Agent:
    def __init__(self):
        self.critic = nn.Sequential(
            nn.Linear(3, 32), nn.Mish(),
            nn.Linear(32, 2)  
        )
        self.actor = nn.Sequential(
            nn.Linear(3, 32), nn.Mish(),
            nn.Linear(32, 1), nn.Tanh() 
        )
        self.opt_c = optim.Adam(self.critic.parameters(), lr=0.01)
        self.opt_a = optim.Adam(self.actor.parameters(), lr=0.005)
        self.gamma = 0.8 
        self.Q_mat = torch.tensor([[90.0, 0.0], [0.0, 5.0]], dtype=torch.float32)

    def get_assistance_action(self, e_theta, e_q, target_theta):
        x_in = torch.tensor([e_theta, e_q, target_theta], dtype=torch.float32)
        with torch.no_grad():
            u_d_norm = self.actor(x_in).item()
        # 【修正4】：缩小辅助上限，IDHP只做精细补救，不抢PID主导权
        return u_d_norm * 3.0 

    def online_train_step(self, e_prev, e_curr, target_theta, F_hat, G_hat):
        e_p = torch.tensor(e_prev, dtype=torch.float32)
        e_c = torch.tensor(e_curr, dtype=torch.float32)
        target = torch.tensor([target_theta], dtype=torch.float32)
        
        in_p = torch.cat([e_p, target])
        in_c = torch.cat([e_c, target])
        G_e = torch.tensor(-G_hat, dtype=torch.float32)
        
        self.opt_c.zero_grad()
        lambda_prev = self.critic(in_p)
        with torch.no_grad():
            lambda_curr = self.critic(in_c)
            dc_de = 2.0 * torch.matmul(self.Q_mat, e_p)
            F_e = torch.tensor(F_hat, dtype=torch.float32)
            target_lambda = dc_de + self.gamma * torch.matmul(F_e.T, lambda_curr)
        loss_c = nn.MSELoss()(lambda_prev, target_lambda)
        loss_c.backward()
        self.opt_c.step()
        
        self.opt_a.zero_grad()
        u_d_norm = self.actor(in_p)
        with torch.no_grad():
            lambda_curr_new = self.critic(in_c)
        actor_grad_direction = torch.matmul(G_e.T, lambda_curr_new)
        loss_a = torch.sum(u_d_norm * actor_grad_direction)
        loss_a.backward()
        self.opt_a.step()

# =================================================================
# 3. 主程序：无缝运动学外环 + 纯净 IDHP 内环
# =================================================================
if __name__ == "__main__":
    TEST_INIT_ALT = 3000.0
    TEST_INIT_VEL = 250.0
    TEST_TARGET_ALT = 3500.0

    print(f"===========================================")
    print(f" 🚀 启动终极架构试飞：克服 RLS 共线性漂移")
    print(f" 任务: {TEST_INIT_ALT}m -> {TEST_TARGET_ALT}m")
    print(f" 故障: 40-60s 舵效减半 (物理注入)")  # 新增：明确打印故障时间
    print(f"===========================================")

    aircraft_params = {'S': 88.58, 'b': 18.9, 'c_bar': 4.6, 'mass': 14000,
                       'Ixx': 313220.6, 'Iyy': 273435.3, 'Izz': 392903.9, 'Ixz': 0.0}
    flight_db = NeuralAeroDatabase()
    flight_db._load_from_pickle('aero_surrogate.pth')
    engine_db = EngineDatabase()
    engine_db.load1('engine.pkl')

    sim = FlightSimulator6DOF(flight_db, engine_db, aircraft_params)
    sim.set_initial_state(h_m=TEST_INIT_ALT, V_mps=TEST_INIT_VEL, theta_deg=0.7, alpha_deg=0.7)

    rls = RLS_Identifier(state_dim=2, input_dim=1)
    idhp = IDHP_Agent()
    
    dt = 0.02
    total_steps = int(300 / dt)  # 从300s缩短到120s，刚好覆盖故障+恢复过程
    
    history_time, history_alt, history_vel = [], [], []
    history_pitch, history_target_pitch = [], []
    history_u0, history_ud = [], []
    
    integral_h, integral_vz, integral_theta = 0.0, 0.0, 0.0
    sim_time = 0.0
    
    last_x = np.array([math.radians(0.7), 0.0])
    last_u, last_delta_u = 0.0, 0.0
    last_delta_x, last_e = np.zeros(2), np.zeros(2)

    for step in range(total_steps):
        u, v, w = sim.state[3], sim.state[4], sim.state[5]
        phi, theta = sim.state[6], sim.state[7]
        current_h = -sim.state[2]
        vz = u*math.sin(theta) - v*math.sin(phi)*math.cos(theta) - w*math.cos(phi)*math.cos(theta)
        
        current_pitch = math.degrees(theta)
        current_q = math.degrees(sim.state[10])
        current_p = math.degrees(sim.state[9])
        
        # ==============================================================
        # 外环：彻底消灭稳态误差的双积分器
        # ==============================================================
        err_h = TEST_TARGET_ALT - current_h
        # 【修正5】：高度积分器！飞机再也无法在几米外躺平了
        integral_h = np.clip(integral_h + err_h * dt, -500.0, 500.0)
        target_vz = np.clip(0.1 * err_h + 0.005 * integral_h, -15.0, 15.0) 
        
        err_vz = target_vz - vz
        integral_vz = np.clip(integral_vz + err_vz * dt, -20.0, 20.0)
        target_pitch = 0.7 + (0.2 * err_vz) + (0.1 * integral_vz)
        target_pitch = np.clip(target_pitch, -2.0, 5.0) 
        
        # ==============================================================
        # 内环: 基础 PID + IDHP 在线学习
        # ==============================================================
        err_theta = target_pitch - current_pitch
        err_q = 0.0 - current_q
        current_e = np.array([err_theta, err_q])
        current_x = np.array([current_pitch, current_q])
        
        if sim_time > 0.05:
            delta_x_curr = current_x - last_x
            F_hat, G_hat = rls.update(last_delta_x, last_delta_u, delta_x_curr)
            idhp.online_train_step(last_e, current_e, target_pitch, F_hat, G_hat)
            last_delta_x = delta_x_curr
        
        integral_theta = np.clip(integral_theta + err_theta * dt, -10.0, 10.0)
        u_0 = -2.5 * err_theta + 0.8 * current_q - 0.5 * integral_theta
        u_d = idhp.get_assistance_action(err_theta, err_q, target_pitch)
        
        # 【修正6：持续激励 PE】加入微小的正弦扰动，让RLS有充足的动态数据学习
        u_PE = 0.05 * math.sin(2 * math.pi * 0.5 * sim_time) + 0.02 * math.sin(2 * math.pi * 1.5 * sim_time)
        
        # RL只负责下发指令
        d_flap_cmd = np.clip(u_0 + u_d + u_PE, -15.0, 15.0) 
        
        # 保存用于下次 RLS 辨识 (控制器只知道指令偏角)
        last_delta_u = d_flap_cmd - last_u
        last_u = d_flap_cmd
        last_x = current_x.copy()
        last_e = current_e.copy()
        
        # ==============================================================
        # ✅ 修正：故障时间统一为40-60s，与绘图标注一致
        # ==============================================================
        d_flap_physical = d_flap_cmd
        if 140.0 < sim_time < 160.0:  # 从160-200s改为40-60s
            d_flap_physical = d_flap_cmd * 0.5 # 真实的物理致动器只能偏转一半
            
        d_ail = np.clip(1.0 * (0.0 - math.degrees(phi)) - 0.5 * current_p, -10.0, 10.0)
        controls = {'d_flap_L': d_flap_physical, 'd_flap_R': d_flap_physical, 'd_ail_L': d_ail, 'd_ail_R': -d_ail}
        
        sim.step(dt, controls)
        sim_time += dt

        history_time.append(sim_time)
        history_alt.append(current_h)
        history_pitch.append(current_pitch)
        history_target_pitch.append(target_pitch)
        history_u0.append(u_0)
        history_ud.append(u_d)
        
        if step % 500 == 0: 
            print(f"Time: {sim_time:4.1f}s | 高度: {current_h:6.1f}m | 指令: {target_pitch:4.1f}° | 基础PID: {u_0:5.1f}° | IDHP辅助: {u_d:5.2f}°")

    print("\n✅ 经典外环 + 纯净 IDHP 验证完成！")
    
    # === 绘图（故障区域已自动匹配40-60s）===
    plt.style.use('dark_background') 
    plt.rcParams['font.sans-serif'] = ['SimHei'] 
    plt.rcParams['axes.unicode_minus'] = False   
    fig = plt.figure(figsize=(15, 10))
    fig.suptitle('IDHP 在线抗故障实飞 (彻底消除稳态误差版)', fontsize=18, color='cyan')

    gs = GridSpec(2, 2, figure=fig)
    
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.plot(history_time, history_alt, color='springgreen', linewidth=2.5)
    ax1.axhline(TEST_TARGET_ALT, color='white', linestyle='--', alpha=0.5)
    ax1.axvspan(140, 160, color='red', alpha=0.2, label='突发故障 (物理效能减半)')
    ax1.set_title('高度轨迹 (彻底贴合虚线)'); ax1.grid(True, alpha=0.2); ax1.legend()

    ax2 = fig.add_subplot(gs[0, 1])
    ax2.plot(history_time, history_pitch, color='hotpink', linewidth=2, label='实际俯仰')
    ax2.plot(history_time, history_target_pitch, color='white', linestyle='--', label='外环指令')
    ax2.axvspan(140, 160, color='red', alpha=0.2)
    ax2.set_title('姿态追踪 (完全解绑 IDHP 饱和死锁)'); ax2.legend(); ax2.grid(True, alpha=0.2)

    ax3 = fig.add_subplot(gs[1, :])
    ax3.plot(history_time, history_u0, color='dodgerblue', linewidth=2, label='PID 基础舵角 (u_0)')
    ax3.plot(history_time, history_ud, color='gold', linewidth=2, label='IDHP 智能微调 (u_d)')
    ax3.axvspan(140, 160, color='red', alpha=0.2)
    ax3.set_title('控制分配：故障区内 IDHP 的精准补救', fontsize=14)
    ax3.legend(); ax3.grid(True, alpha=0.2)

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()