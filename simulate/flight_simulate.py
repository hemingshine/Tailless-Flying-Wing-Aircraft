#coding=utf-8
import os
import pickle
from turtle import pd
import numpy as np
import math
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

import pandas as pd
import numpy as np
import os
import pickle
from scipy.interpolate import LinearNDInterpolator, interp1d
import warnings

warnings.filterwarnings('ignore')
class HybridAeroDatabase:
    def __init__(self):
        # 结构设计变更：self.models_db['模型代号'][马赫数值] = 对应的插值器
        self.raw_data = {} 
        self.models_db = {}
        
        # 注意：在插值引擎眼里，自变量只剩下迎角和侧滑角了，马赫数变成了“分类标签”
        self.output_cols = [
            '轴向力系数', '横向力系数', '法向力系数', 
            '滚转力矩系数', '俯仰力矩系数', '偏航力矩系数'
        ]

    def _load_from_pickle(self, pickle_path):
        with open(pickle_path, 'rb') as f:
            self.models_db = pickle.load(f)
        print(f"加载成功！\n" + "-"*40)

    def get_body_axis_coeffs(self, model_code, mach, alpha, beta):
        """
        马赫数就近取整查表，迎角和侧滑角连续插值计算
        """
        if model_code not in self.models_db:
            raise ValueError(f"找不到模型代号: {model_code}")
            
        # 1. 获取该模型所有可用的马赫数 (例如: [0.4, 0.6, 0.8])
        available_machs = list(self.models_db[model_code].keys())
        
        # 2. 【核心逻辑】：找到离当前飞行马赫数最近的那个表
        # 例如输入 0.49 会找到 0.4，输入 0.51 会找到 0.6
        closest_mach = min(available_machs, key=lambda x: abs(x - mach))
        
        # 3. 提取对应的插值器
        model_info = self.models_db[model_code][closest_mach]
        query_point = np.array([alpha, beta]) # 注意现在查询点只有迎角和侧滑角
        
        # 4. 根据类型进行姿态角的插值计算
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
        # 推力插值器
        self.thrust_interpolator = None
        
        # 输入自变量：高度、马赫数
        self.input_cols = ['Alt（m）', 'Ma']
        # 输出因变量：净推力
        self.output_cols = ['FN（DaN）']
    
    def load1(self,pickle_path="engine.pkl"):
        print(f"检测到发动机缓存文件 '{pickle_path}'，正在秒速加载...")
        with open(pickle_path, 'rb') as f:
            self.thrust_interpolator = pickle.load(f)
        print("发动机数据库加载成功！\n" + "-"*40)

    def get_thrust_newtons(self, alt, mach):
        """
        在仿真循环中调用此方法。
        输入：高度(m), 马赫数, 油门百分比(%)
        输出：推力 (标准单位：牛顿 N)
        """
        if self.thrust_interpolator is None:
            raise RuntimeError("请先调用 load_or_build() 加载数据库！")
            
        query_point = np.array([[alt, mach]])
        thrust_dan = self.thrust_interpolator(query_point)[0]
        
        # 处理可能超出发动机包线（越界插值返回 NaN）的情况
        if np.isnan(thrust_dan):
            # 越界时你可以设为0，或者根据需求外推
            thrust_dan = 0.0 
            
        # 【极其重要】：将 DaN (十牛) 转换为 N (牛)
        thrust_n = thrust_dan * 10.0
        
        return thrust_n

#-----------------------------------------------------------------
'''
数据库构建
'''

class FlightSimulator6DOF:
    def __init__(self, aero_db, engine_db, global_params):
        """
        初始化 6-DOF 飞行模拟器
        :param aero_db: 实例化的气动数据库对象 (带 get_body_axis_coeffs 方法)
        :param engine_db: 实例化的发动机数据库对象 (带 get_thrust_newtons 方法)
        :param global_params: 包含质量、惯量、外形尺寸的字典
        """
        self.aero_db = aero_db
        self.engine_db = engine_db
        
        # --- 飞机的固有全局参数 ---
        self.S = global_params['S']          # 参考面积 (m^2)
        self.b = global_params['b']          # 参考展长 (m)
        self.c_bar = global_params['c_bar']  # 平均气动弦长 (m)
        self.mass = global_params['mass']    # 质量 (kg) (定质量模拟)
        
        # 转动惯量 (kg*m^2)
        self.Ixx = global_params['Ixx']
        self.Iyy = global_params['Iyy']
        self.Izz = global_params['Izz']
        self.Ixz = global_params['Ixz']
        
# 惯量矩阵相关的常数预计算 (严格遵循 Stevens & Lewis 飞行动力学标准公式)
        self.Gamma = self.Ixx * self.Izz - self.Ixz**2
        
        self.c1 = ((self.Iyy - self.Izz) * self.Izz - self.Ixz**2) / self.Gamma
        self.c2 = ((self.Ixx - self.Iyy + self.Izz) * self.Ixz) / self.Gamma
        self.c3 = self.Izz / self.Gamma
        self.c4 = self.Ixz / self.Gamma
        self.c5 = (self.Izz - self.Ixx) / self.Iyy
        self.c6 = self.Ixz / self.Iyy
        self.c7 = ((self.Ixx - self.Iyy) * self.Ixx + self.Ixz**2) / self.Gamma
        self.c8 = self.Ixx / self.Gamma
        
        self.g = 9.80665 # 重力加速度 (m/s^2)

        # --- 状态变量初始化 (12个状态) ---
        # 顺序: [pn, pe, pd, u, v, w, phi, theta, psi, p, q, r]
        self.state = np.zeros(12)

    def set_initial_state(self, h_m, V_mps, theta_deg):
        """设置初始配平状态：定高平飞"""
        self.state[2] = -h_m              # pd = -高度
        self.state[3] = V_mps             # u = 前向速度
        self.state[7] = math.radians(theta_deg) # 初始俯仰角
        
    def isa_atmosphere(self, altitude_m):
        """国际标准大气模型 (ISA) -> 返回密度和音速"""
        # 仅适用对流层 (< 11000m) 简化版，如需高空可扩充
        T0 = 288.15
        p0 = 101325.0
        rho0 = 1.225
        L = 0.0065 # 温度递减率
        R = 287.05
        gamma = 1.4
        
        if altitude_m < 11000:
            T = T0 - L * altitude_m
            p = p0 * (T / T0) ** (self.g * 0.0289644 / (8.3144598 * L))
            rho = p / (R * T)
        else:
            # 高空同温层简化近似
            T = 216.65
            rho = 0.36391 * math.exp(-(altitude_m - 11000) / 6341.6)
            
        a = math.sqrt(gamma * R * T) # 音速
        return rho, a

    def get_derivatives(self, state, model_code):
        """核心物理引擎：计算 12 个状态变量的导数"""
        pn, pe, pd, u, v, w, phi, theta, psi, p, q, r = state
        
        # 1. 运动学基础数据提取
        V = math.sqrt(u**2 + v**2 + w**2)
        if V == 0: V = 0.001 # 防除零错
        
        alpha_rad = math.atan2(w, u)
        beta_rad = math.asin(v / V)
        alpha_deg = math.degrees(alpha_rad)
        beta_deg = math.degrees(beta_rad)
        
        # 2. 大气环境与马赫数
        h = -pd
        rho, a = self.isa_atmosphere(h)
        Mach = V / a
        q_dyn = 0.5 * rho * V**2
        
        # 3. 查表获取气动系数与推力
        coeffs = self.aero_db.get_body_axis_coeffs(model_code, Mach, alpha_deg, beta_deg)
        thrust = self.engine_db.get_thrust_newtons(h, Mach)
        
        # 4. 计算机体轴受力 (严格注意符号约定！)
        # 假设数据中 C_A 为正时向后，C_N 为正时向上
        Fx = thrust - coeffs['轴向力系数'] * q_dyn * self.S
        Fy = coeffs['横向力系数'] * q_dyn * self.S
        Fz = - coeffs['法向力系数'] * q_dyn * self.S
        
        # 5. 计算气动力矩
        L_aero = coeffs['滚转力矩系数'] * q_dyn * self.S * self.b
        M_aero = coeffs['俯仰力矩系数'] * q_dyn * self.S * self.c_bar
        N_aero = coeffs['偏航力矩系数'] * q_dyn * self.S * self.b
        M_aero += -q * 200000.0
        
        # 6. 牛顿-欧拉平动方程 (求速度导数 dot_u, dot_v, dot_w)
        dot_u = (Fx / self.mass) - self.g * math.sin(theta) - q*w + r*v
        dot_v = (Fy / self.mass) + self.g * math.cos(theta) * math.sin(phi) - r*u + p*w
        dot_w = (Fz / self.mass) + self.g * math.cos(theta) * math.cos(phi) - p*v + q*u
        
        # 7. 牛顿-欧拉转动方程 (求角速度导数 dot_p, dot_q, dot_r)
        # 这里使用了非对角惯量矩阵(考虑 Ixz)的解析解形式
        dot_p = (self.c1 * r * q + self.c2 * p * q + self.c3 * L_aero + self.c4 * N_aero)
        dot_q = (self.c5 * p * r - self.c6 * (p**2 - r**2) + M_aero / self.Iyy)
        dot_r = (self.c7 * p * q - self.c2 * q * r + self.c4 * L_aero + self.c8 * N_aero)
        
        # 8. 运动学方程 (求位置和姿态角导数)
        # 机体速度转地面速度 (NED)
        dot_pn = u*math.cos(theta)*math.cos(psi) + v*(math.sin(phi)*math.sin(theta)*math.cos(psi) - math.cos(phi)*math.sin(psi)) + w*(math.cos(phi)*math.sin(theta)*math.cos(psi) + math.sin(phi)*math.sin(psi))
        dot_pe = u*math.cos(theta)*math.sin(psi) + v*(math.sin(phi)*math.sin(theta)*math.sin(psi) + math.cos(phi)*math.cos(psi)) + w*(math.cos(phi)*math.sin(theta)*math.sin(psi) - math.sin(phi)*math.cos(psi))
        dot_pd = -u*math.sin(theta) + v*math.sin(phi)*math.cos(theta) + w*math.cos(phi)*math.cos(theta)
        
        # 机体角速度转欧拉角速率
        dot_phi = p + math.tan(theta) * (q*math.sin(phi) + r*math.cos(phi))
        dot_theta = q*math.cos(phi) - r*math.sin(phi)
        dot_psi = (q*math.sin(phi) + r*math.cos(phi)) / math.cos(theta)
        
        return np.array([dot_pn, dot_pe, dot_pd, dot_u, dot_v, dot_w, dot_phi, dot_theta, dot_psi, dot_p, dot_q, dot_r])

    def step_rk4(self, dt, model_code):
        """四阶龙格-库塔 (RK4) 积分步进器"""
        y0 = self.state.copy()
        
        # RK4 核心的 4 次采样
        k1 = self.get_derivatives(y0, model_code)
        k2 = self.get_derivatives(y0 + 0.5 * dt * k1, model_code)
        k3 = self.get_derivatives(y0 + 0.5 * dt * k2, model_code)
        k4 = self.get_derivatives(y0 + dt * k3, model_code)
        
        # 状态更新
        self.state = y0 + (dt / 6.0) * (k1 + 2*k2 + 2*k3 + k4)
        
        # 返回更新后的高度、速度、攻角等便于观察的参数
        u, v, w = self.state[3], self.state[4], self.state[5]
        V = math.sqrt(u**2 + v**2 + w**2)
        alpha = math.degrees(math.atan2(w, u))
        h = -self.state[2]
        
        return {"Time": 0, "Altitude": h, "Velocity": V, "Alpha": alpha, "Pitch": math.degrees(self.state[7])}

def plot1():

    # --- 设置美观的绘图风格 ---
    plt.style.use('dark_background')  # 科技感暗色背景
    # 解决中文显示问题 (Windows自带黑体)
    plt.rcParams['font.sans-serif'] = ['SimHei'] 
    plt.rcParams['axes.unicode_minus'] = False   
    
    # 创建一个宽 15，高 10 的超大画板
    fig = plt.figure(figsize=(15, 10))
    fig.suptitle('6-DOF 飞行轨迹与姿态遥测数据', fontsize=20, fontweight='bold', color='cyan')

    # ==========================================
    # 子图 1: 3D 飞行航迹 (左侧占据一整列)
    # ==========================================
    ax1 = fig.add_subplot(1, 2, 1, projection='3d')
    # 绘制 3D 曲线，使用亮青色，带一点点发光效果 (linewidth加粗)
    ax1.plot(history_pe, history_pn, history_alt, color='cyan', linewidth=2.5, label='飞行轨迹')
    
    # 标记起点和终点
    ax1.scatter(history_pe[0], history_pn[0], history_alt[0], color='lime', s=100, label='起点 (Takeoff)', zorder=5)
    ax1.scatter(history_pe[-1], history_pn[-1], history_alt[-1], color='red', s=100, label='终点 (Current)', zorder=5)

    ax1.set_title('3D 空间航迹 (NED 坐标)', fontsize=14)
    ax1.set_xlabel('东向位置 East (m)')
    ax1.set_ylabel('北向位置 North (m)')
    ax1.set_zlabel('飞行高度 Altitude (m)')
    ax1.legend()
    # 调整默认视角，让飞机看起来是斜向上飞的
    ax1.view_init(elev=25, azim=-45) 

    # ==========================================
    # 子图 2: 高度变化 (右上)
    # ==========================================
    ax2 = fig.add_subplot(3, 2, 2)
    ax2.plot(history_time, history_alt, color='springgreen', linewidth=2)
    ax2.set_title('高度剖面 (Altitude Profile)', fontsize=12)
    ax2.set_ylabel('高度 (m)')
    ax2.grid(True, linestyle='--', alpha=0.3)

    # ==========================================
    # 子图 3: 速度变化 (右中)
    # ==========================================
    ax3 = fig.add_subplot(3, 2, 4)
    ax3.plot(history_time, history_vel, color='gold', linewidth=2)
    ax3.set_title('速度剖面 (Velocity Profile)', fontsize=12)
    ax3.set_ylabel('速度 (m/s)')
    ax3.grid(True, linestyle='--', alpha=0.3)

    # ==========================================
    # 子图 4: 姿态与迎角 (右下) - 双Y轴设计
    # ==========================================
    ax4 = fig.add_subplot(3, 2, 6)
    # 主Y轴画俯仰角 (Pitch)
    line1 = ax4.plot(history_time, history_pitch, color='hotpink', linewidth=2, label='俯仰角 (Pitch)')
    ax4.set_xlabel('时间 (s)')
    ax4.set_ylabel('俯仰角 (°)', color='hotpink')
    ax4.tick_params(axis='y', labelcolor='hotpink')
    
    # 副Y轴画迎角 (Alpha)
    ax4_twin = ax4.twinx()
    line2 = ax4_twin.plot(history_time, history_alpha, color='dodgerblue', linewidth=2, linestyle='-.', label='迎角 (Alpha)')
    ax4_twin.set_ylabel('迎角 (°)', color='dodgerblue')
    ax4_twin.tick_params(axis='y', labelcolor='dodgerblue')
    
    ax4.set_title('纵向气动姿态 (Longitudinal Attitude)', fontsize=12)
    ax4.grid(True, linestyle='--', alpha=0.3)

    # 调整布局并显示
    plt.tight_layout(rect=[0, 0.03, 1, 0.95]) # 留出标题空间
    plt.show()




# ================= 测试运行用例 =================
if __name__ == "__main__":
    
    # 1. 填入你的飞行器全局参数！
    # aircraft_params = {
    #     'S': 3.857,        # 参考面积 m^2 (瞎猜的数据，需替换)
    #     'b': 4.2,        # 参考展长 m
    #     'c_bar': 1.380462,     # 平均气动弦长 m
    #     'mass': 14000,  # 飞机总质量 kg
    #     'Ixx': 313220.6,   # 滚转转动惯量
    #     'Iyy': 273435.3,   # 俯仰转动惯量
    #     'Izz': 392903.9,  # 偏航转动惯量
    #     'Ixz': 0.0     # 交叉转动惯量
    # }
    aircraft_params = {
        'S': 88.58, 'b': 18.9, 'c_bar': 4.6, 'mass': 14000,
        'Ixx': 313220.6, 'Iyy': 273435.3, 'Izz': 392903.9, 'Ixz': 0.0
    }

    flight_db=HybridAeroDatabase()
    flight_db._load_from_pickle(pickle_path='X47B.pkl')
    engine_db=EngineDatabase()
    engine_db.load1('engine.pkl')

    sim = FlightSimulator6DOF(flight_db, engine_db, aircraft_params)
    
    # 2. 设定初始状态：高度 5000m, 速度 250m/s (约 Ma 0.8), 初始俯仰角 2度
    sim.set_initial_state(h_m=2000.0, V_mps=250.0, theta_deg=0)
    
    # 1. 创建空列表，用于记录飞行黑匣子数据
    history_time = []
    history_alt = []
    history_vel = []
    history_alpha = []
    history_pitch = []
    history_pn = []  # 北向位置
    history_pe = []  # 东向位置

    # 3. 开始模拟主循环！
    dt = 0.02 # 仿真步长：20毫秒 (50Hz)
    sim_time = 0.0
    
    print("开始飞行模拟...")
    for step in range(1000): # 模拟 100 步
        # 你的控制律在这里生效：比如你想在这里改变模型外形
        current_model_code = 'state05' 
        
        # RK4 向前推演一步
        result = sim.step_rk4(dt, current_model_code)
        sim_time += dt

        history_time.append(sim_time)
        history_alt.append(result['Altitude'])
        history_vel.append(result['Velocity'])
        history_alpha.append(result['Alpha'])
        history_pitch.append(result['Pitch'])
        history_pn.append(sim.state[0]) # state[0] 是北向位置 pn
        history_pe.append(sim.state[1]) # state[1] 是东向位置 pe
        # 每隔几步打印一次状态
        if step % 10 == 0:
            print(f"Time: {sim_time:.2f}s | 高度: {result['Altitude']:.1f}m | 速度: {result['Velocity']:.1f}m/s | 迎角: {result['Alpha']:.2f}° | 俯仰角: {result['Pitch']:.2f}°")
    plot1()