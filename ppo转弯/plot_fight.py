#coding=utf-8
import os
import numpy as np
import math
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import seaborn as sns
import warnings

# 导入真实的物理引擎和气动数据库
from fly import NeuralAeroDatabase, EngineDatabase, FlightSimulator6DOF

warnings.filterwarnings('ignore')

def plot_split_real_coupling():
    print("=====================================================")
    print(" 🛩️ Extracting X-47B real aerodynamic and actuator response data...")
    print("=====================================================")

    # 1. 真实物理引擎初始化
    aircraft_params = {'S': 88.58, 'b': 18.9, 'c_bar': 4.6, 'mass': 14000, 
                       'Ixx': 313220.6, 'Iyy': 273435.3, 'Izz': 392903.9, 'Ixz': 0.0}
    flight_db = NeuralAeroDatabase()
    
    db_path = 'aero_surrogate.pth' if os.path.exists('aero_surrogate.pth') else 'X47B_coeffs.pkl'
    flight_db._load_from_pickle(db_path)
    
    engine_db = EngineDatabase()
    engine_db.load1('engine.pkl')
    
    sim = FlightSimulator6DOF(flight_db, engine_db, aircraft_params)
    
    base_controls = {
        'd_flap_L': 0.0, 'd_flap_R': 0.0, 
        'd_ail_L': 0.0, 'd_ail_R': 0.0, 
        'd_spoil_L': 0.0, 'd_spoil_R': 0.0, 
        'throttle': 0.85
    }

    # ==========================================================
    # 全局字体和样式设置 (纯英文论文标准样式，进一步放大字体)
    # ==========================================================
    plt.style.use('default')
    plt.rcParams['font.family'] = 'serif'
    plt.rcParams['font.serif'] = ['Times New Roman', 'DejaVu Serif']
    plt.rcParams['axes.unicode_minus'] = False 
    
    # 全局超大字体配置
    plt.rcParams['axes.labelsize'] = 18    # X/Y 轴标签字体大小
    plt.rcParams['xtick.labelsize'] = 16   # X 轴刻度字体大小
    plt.rcParams['ytick.labelsize'] = 16   # Y 轴刻度字体大小
    plt.rcParams['legend.fontsize'] = 16   # 图例字体大小
    
    # ==========================================================
    # 图片 1: B 矩阵热力图
    # ==========================================================
    print("📊 [1/2] Generating B Matrix heatmap...")
    sim.set_initial_state(h_m=2500.0, V_mps=250.0, theta_deg=1.0, alpha_deg=1.0)
    b_matrix = sim.get_control_effectiveness_matrix(base_controls)
    
    fig1 = plt.figure(figsize=(12, 7))
    ax1 = fig1.add_subplot(111)
    
    # 纯英文标签
    surfaces = ['Left Flap', 'Right Flap)', 
                'Left Aileron', 'Right Aileron', 
                'Left Spoiler', 'Right Spoiler']
    moments = ['Roll Accel', 'Pitch Accel', 'Yaw Accel']

    # annot_kws 调整热力图内部数字大小到 18
    sns.heatmap(b_matrix, annot=True, fmt=".2f", cmap="coolwarm", center=0, 
                xticklabels=surfaces, yticklabels=moments, 
                annot_kws={"size": 20}, 
                cbar_kws={'label': 'Control Effectiveness (deg/s² per deg)'}, ax=ax1, linewidths=1, linecolor='white')
    
    # 调整 Colorbar 字体大小
    cbar = ax1.collections[0].colorbar
    cbar.ax.yaxis.label.set_size(20)
    cbar.ax.tick_params(labelsize=20)
    
    plt.tight_layout()
    
    # 同时保存 PDF (矢量图) 和 PNG (300dpi 高清)
    fig1_name = 'fig1_B_matrix_heatmap'
    fig1.savefig(f'{fig1_name}.pdf', format='pdf', bbox_inches='tight')
    fig1.savefig(f'{fig1_name}.png', format='png', bbox_inches='tight', dpi=300)
    print(f"✅ Figure 1 saved to: {fig1_name}.pdf / .png")

    # ==========================================================
    # 数据提取: 六通道独立单打物理仿真
    # ==========================================================
    print("📈 [2/2] Running independent step response simulations...")
    dt = 0.02
    total_time = 4.0 
    
    surf_keys = ['d_flap_L', 'd_flap_R', 'd_ail_L', 'd_ail_R', 'd_spoil_L', 'd_spoil_R']
    labels = ['Left Flap (+15°)', 'Right Flap (+15°)', 
              'Left Aileron (+15°)', 'Right Aileron (+15°)', 
              'Left Spoiler (+15°)', 'Right Spoiler (+15°)']
    
    # 颜色配对：同一种舵面的左右使用同色系，实线为左，虚线为右
    colors = ['#1f77b4', '#1f77b4', '#d62728', '#d62728', '#2ca02c', '#2ca02c']
    styles = ['-', '--', '-', '--', '-', '--']
    
    results = {}
    
    for key in surf_keys:
        # 严格重置物理引擎状态到绝对干净的起点
        sim.state = np.zeros(12, dtype=np.float64)
        sim.set_initial_state(h_m=2500.0, V_mps=250.0, theta_deg=1.0, alpha_deg=1.0)
        
        sim_time = 0.0
        hist_t, hist_phi, hist_theta, hist_psi = [], [], [], []
        
        for step in range(int(total_time / dt)):
            controls = base_controls.copy()
            # 在 t=1.0s 时，单独对当前遍历的舵面施加 15 度阶跃指令
            if sim_time >= 1.0:
                controls[key] = 15.0
                
            sim.step(dt, controls)
            
            s = sim.state
            hist_t.append(sim_time)
            hist_phi.append(math.degrees(s[6]))   # 滚转 (Roll)
            hist_theta.append(math.degrees(s[7])) # 俯仰 (Pitch)
            hist_psi.append(math.degrees(s[8]))   # 偏航 (Yaw)
            
            sim_time += dt
            
        results[key] = {'t': hist_t, 'phi': hist_phi, 'theta': hist_theta, 'psi': hist_psi}

    # ==========================================================
    # 图片 2: 1行3列 纯学术风格矢量图
    # ==========================================================
    # 尺寸设定，保持一行三列
    fig2 = plt.figure(figsize=(20, 6))
    gs = GridSpec(1, 3, figure=fig2)

    # 子图 1: Roll Response
    ax_roll = fig2.add_subplot(gs[0, 0])
    for i, key in enumerate(surf_keys):
        ax_roll.plot(results[key]['t'], results[key]['phi'], color=colors[i], linestyle=styles[i], linewidth=2.5, label=labels[i])
    ax_roll.axvline(1.0, color='gray', linestyle=':', alpha=0.8)
    ax_roll.set_xlabel('Time [s]', fontweight='bold', fontsize=18)
    ax_roll.set_ylabel('Roll Angle (Phi) [deg]', fontweight='bold', fontsize=18)
    ax_roll.grid(True, linestyle='--', alpha=0.5)

    # 子图 2: Pitch Response
    ax_pitch = fig2.add_subplot(gs[0, 1])
    for i, key in enumerate(surf_keys):
        ax_pitch.plot(results[key]['t'], results[key]['theta'], color=colors[i], linestyle=styles[i], linewidth=2.5, label=labels[i])
    ax_pitch.axvline(1.0, color='gray', linestyle=':', alpha=0.8)
    ax_pitch.set_xlabel('Time [s]', fontweight='bold', fontsize=18)
    ax_pitch.set_ylabel('Pitch Angle (Theta) [deg]', fontweight='bold', fontsize=18)
    ax_pitch.grid(True, linestyle='--', alpha=0.5)

    # 子图 3: Yaw Response
    ax_yaw = fig2.add_subplot(gs[0, 2])
    for i, key in enumerate(surf_keys):
        ax_yaw.plot(results[key]['t'], results[key]['psi'], color=colors[i], linestyle=styles[i], linewidth=2.5, label=labels[i])
    ax_yaw.axvline(1.0, color='gray', linestyle=':', alpha=0.8)
    ax_yaw.set_xlabel('Time [s]', fontweight='bold', fontsize=18)
    ax_yaw.set_ylabel('Yaw Angle (Psi) [deg]', fontweight='bold', fontsize=18)
    ax_yaw.grid(True, linestyle='--', alpha=0.5)

    # 将图例移回图片内部：loc='best' 会自动寻找最不遮挡曲线的位置，framealpha 半透明防止彻底遮住底层网格线
    ax_yaw.legend(loc='best', fontsize=16, frameon=True, framealpha=0.85)

    # 布局自动收紧，确保排版紧凑不溢出
    plt.tight_layout()
    
    # 导出双格式
    fig2_name = 'fig2_individual_surface_coupling'
    fig2.savefig(f'{fig2_name}.pdf', format='pdf', bbox_inches='tight')
    fig2.savefig(f'{fig2_name}.png', format='png', bbox_inches='tight', dpi=300)
    print(f"✅ Figure 2 saved to: {fig2_name}.pdf / .png")

    plt.show()

if __name__ == "__main__":
    plot_split_real_coupling()