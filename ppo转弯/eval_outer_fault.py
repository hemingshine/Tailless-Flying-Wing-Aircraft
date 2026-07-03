import os
import math
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from stable_baselines3 import PPO

# 导入外环环境
from train_outerfault import make_outer_env

def run_simulation(ftc_enabled):
    """
    核心仿真逻辑完全未修改，仅封装为函数以支持 FTC ON/OFF 的双重对比运行。
    """
    model_path = "./logs/best_model_outer1/best_model.zip"
    if not os.path.exists(model_path):
        print(f"❌ 找不到模型文件: {model_path}，请确认路径是否正确。")
        return None

    # 初始化环境
    env_creator = make_outer_env(seed=888)
    env = env_creator()
    
    # 加载模型 (纯推理模式)
    model = PPO.load(model_path[:-4], device='cpu')
    
    obs, _ = env.reset()
    
    # ==========================================
    # 🎯 设定极为严苛的测试工况
    # ==========================================
    env.target_yaw = 90.0   
    env.target_alt = 3200.0 
    
    # 强制物理引擎进入平飞初始态
    env.inner_env.sim.set_initial_state(3000.0, 200.0, theta_deg=2.0)
    env.inner_env.sim.state[6] = 0.0  
    env.inner_env.sim.state[8] = 0.0  
    for _ in range(5): 
        env.inner_env._update_history()

    # ★ 清洁化评估：配置 FTC 的开启或关闭状态
    env.inner_env.domain_rand = False
    env.inner_env.eff = {'pitch': 1.0, 'roll': 1.0, 'yaw': 1.0}
    env.inner_env._fault_t = 1e9
    env.inner_env.ftc_enabled = ftc_enabled  # <-- 这里动态接受开启/关闭状态
    
    env.prev_yaw_error = ((env.target_yaw - math.degrees(env.inner_env.sim.state[8]) + 180) % 360) - 180
    env.prev_alt_error = env.target_alt - (-env.inner_env.sim.state[2])
    obs = env._get_obs()

    # ==========================================
    # 🔥 物理层故障注入 (Monkey Patch)
    # ==========================================
    fault_start_time = 75.0  
    sim_time = [0.0]         
    original_sim_step = env.inner_env.sim.step
    
    def faulty_sim_step(dt_val, controls):
        if sim_time[0] >= fault_start_time and sim_time[0] <= 100:
            controls['d_ail_L'] *= 0.15
            controls['d_ail_R'] *= 0.15
            
        sim_time[0] += dt_val
        return original_sim_step(dt_val, controls)
        
    env.inner_env.sim.step = faulty_sim_step
    # ==========================================
    
    history = {
        't': [], 'yaw': [], 'target_yaw': [],
        'cmd_roll': [], 'actual_roll': [], 
        'cmd_pitch': [], 'actual_pitch': [], 
        'delta_e': [], 'delta_a': [], 'delta_r': [], 
        'beta': [], 'alpha': [], 'altitude': []
    }
    
    max_test_steps = 6000 
    
    for step in range(max_test_steps):
        t = step * env.outer_dt
        
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, _ = env.step(action)
        
        state = env.inner_env.sim.state
        u, v, w = state[3], state[4], state[5]
        V = max(math.sqrt(u**2 + v**2 + w**2), 1.0)
        
        history['t'].append(t)
        history['yaw'].append(math.degrees(state[8]))
        history['target_yaw'].append(env.target_yaw)
        
        history['cmd_roll'].append(env.cmd_phi)
        history['actual_roll'].append(math.degrees(state[6]))
        history['cmd_pitch'].append(env.cmd_theta)
        history['actual_pitch'].append(math.degrees(state[7]))
        
        history['delta_e'].append(env.inner_env.prev_actions['e'])
        history['delta_a'].append(env.inner_env.prev_actions['a'])
        history['delta_r'].append(env.inner_env.prev_actions['r'])
        
        history['alpha'].append(math.degrees(math.atan2(w, u)))
        history['beta'].append(math.degrees(math.asin(np.clip(v / V, -1.0, 1.0))))
        history['altitude'].append(-state[2])
        
        if terminated or truncated:
            break

    return history


def main():
    print("=====================================================")
    print("  ✈️  X-47B 终极试飞: 故障容错 (FTC) A/B 对比测试")
    print("=====================================================")
    
    print("🛫 正在运行：基础模型 (FTC OFF) 试飞...")
    hist_off = run_simulation(ftc_enabled=False)
    
    print("🛫 正在运行：容错模型 (FTC ON) 试飞...")
    hist_on = run_simulation(ftc_enabled=True)
    
    if not hist_off or not hist_on:
        return

    # ==========================================
    # 📊 计算并打印量化指标 (聚焦故障发生后的表现)
    # ==========================================
    fault_start_time = 75.0
    
    idx_off = [i for i, t in enumerate(hist_off['t']) if t >= fault_start_time and t<=100]
    idx_on = [i for i, t in enumerate(hist_on['t']) if t >= fault_start_time and t<=100]
    
    def calc_rmse(hist, idx, key, target_val):
        if not idx: return float('nan')
        err = np.array([hist[key][i] for i in idx]) - target_val
        return float(np.sqrt(np.mean(err**2)))

    rmse_yaw_off = calc_rmse(hist_off, idx_off, 'yaw', 90.0)
    rmse_yaw_on  = calc_rmse(hist_on, idx_on, 'yaw', 90.0)
    
    rmse_alt_off = calc_rmse(hist_off, idx_off, 'altitude', 3200.0)
    rmse_alt_on  = calc_rmse(hist_on, idx_on, 'altitude', 3200.0)
    
    rmse_beta_off = calc_rmse(hist_off, idx_off, 'beta', 0.0)
    rmse_beta_on  = calc_rmse(hist_on, idx_on, 'beta', 0.0)

    max_alpha_off, min_alpha_off = np.max(hist_off['alpha']), np.min(hist_off['alpha'])
    max_alpha_on, min_alpha_on = np.max(hist_on['alpha']), np.min(hist_on['alpha'])

    print("\n" + "="*68)
    print(" 📊 模型外环性能量化对比 (重点统计故障注入后 RMS 误差)")
    print("="*68)
    print(f" {'评估指标':<20} | {'FTC OFF':<12} | {'FTC ON':<12} | {'对比评价'}")
    print("-" * 68)
    print(f" 航向角 (Yaw) RMSE    | {rmse_yaw_off:10.3f}° | {rmse_yaw_on:10.3f}° | " + 
          ("FTC 占优" if rmse_yaw_on < rmse_yaw_off else "无明显差异"))
    print(f" 高度 (Altitude) RMSE | {rmse_alt_off:10.3f}m | {rmse_alt_on:10.3f}m | " + 
          ("FTC 占优" if rmse_alt_on < rmse_alt_off else "无明显差异"))
    print(f" 侧滑角 (Beta) RMSE   | {rmse_beta_off:10.3f}° | {rmse_beta_on:10.3f}° | " + 
          ("FTC 占优" if rmse_beta_on < rmse_beta_off else "无明显差异"))
    print("-" * 68)
    print(f" 迎角 (Alpha) 最大值  | {max_alpha_off:10.2f}° | {max_alpha_on:10.2f}° | " + 
          ("FTC 更安全" if max_alpha_on < max_alpha_off else "相近"))
    print(f" 迎角 (Alpha) 最小值  | {min_alpha_off:10.2f}° | {min_alpha_on:10.2f}° | " + 
          ("安全" if (min_alpha_off > -3 and min_alpha_on > -3) else "逼近下限"))
    print("="*68 + "\n")

    print("🛬 试飞完毕！正在生成全英文 宽版矢量学术排版图表...")
    
    # ==========================================
    # 📊 绘制专业遥测图表 (纯英文、无标题、大字体、单图包含多曲线解耦)
    # ==========================================
    plt.style.use('bmh')
    
    # 强制纯英文衬线字体，满足国际期刊要求
    plt.rcParams['font.family'] = 'serif'
    plt.rcParams['font.serif'] = ['Times New Roman', 'DejaVu Serif']
    plt.rcParams['axes.unicode_minus'] = False
    
    # 极端放大字体配置
    plt.rcParams['axes.labelsize'] = 20    # 坐标轴标签极大
    plt.rcParams['xtick.labelsize'] = 18   # X刻度
    plt.rcParams['ytick.labelsize'] = 18   # Y刻度
    plt.rcParams['legend.fontsize'] = 16   # 图例清晰
    
    c_off = '#1f77b4'  # Blue
    c_on = '#2ca02c'   # Green
    c_cmd = 'black'    # Black line
    
    t_end = 150.0
    
    # 构建 3行 布局：前两行为 3列，最后一行横跨 3列 的宽图
    fig = plt.figure(figsize=(24, 15))
    gs = GridSpec(3, 3, figure=fig, hspace=0.3, wspace=0.25)
    
    # -----------------------------------------------------
    # Row 1: Outer Loop Tracking (Yaw, Roll, Pitch)
    # -----------------------------------------------------
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.plot(hist_off['t'], hist_off['target_yaw'], color=c_cmd, ls='--', lw=2.5, label='Target')
    ax1.plot(hist_off['t'], hist_off['yaw'], color=c_off, lw=3.0, alpha=0.9, label='FTC OFF')
    ax1.plot(hist_on['t'], hist_on['yaw'], color=c_on, lw=3.0, alpha=0.9, label='FTC ON')
    ax1.set_ylabel('Yaw / Heading [deg]', fontweight='bold')

    ax2 = fig.add_subplot(gs[0, 1])
    ax2.plot(hist_off['t'], hist_off['cmd_roll'], color=c_off, ls=':', lw=2.5, alpha=0.7, label='Cmd (OFF)')
    ax2.plot(hist_on['t'], hist_on['cmd_roll'], color=c_on, ls=':', lw=2.5, alpha=0.7, label='Cmd (ON)')
    ax2.plot(hist_off['t'], hist_off['actual_roll'], color=c_off, lw=3.0, alpha=0.9, label='Actual (OFF)')
    ax2.plot(hist_on['t'], hist_on['actual_roll'], color=c_on, lw=3.0, alpha=0.9, label='Actual (ON)')
    ax2.set_ylabel('Roll Angle [deg]', fontweight='bold')

    ax3 = fig.add_subplot(gs[0, 2])
    ax3.plot(hist_off['t'], hist_off['cmd_pitch'], color=c_off, ls=':', lw=2.5, alpha=0.7, label='Cmd (OFF)')
    ax3.plot(hist_on['t'], hist_on['cmd_pitch'], color=c_on, ls=':', lw=2.5, alpha=0.7, label='Cmd (ON)')
    ax3.plot(hist_off['t'], hist_off['actual_pitch'], color=c_off, lw=3.0, alpha=0.9, label='Actual (OFF)')
    ax3.plot(hist_on['t'], hist_on['actual_pitch'], color=c_on, lw=3.0, alpha=0.9, label='Actual (ON)')
    ax3.set_ylabel('Pitch Angle [deg]', fontweight='bold')

    # -----------------------------------------------------
    # Row 2: Altitude & Safety (Alt, Beta, Alpha)
    # -----------------------------------------------------
    ax4 = fig.add_subplot(gs[1, 0])
    ax4.plot(hist_off['t'], [3200]*len(hist_off['t']), color=c_cmd, ls='--', lw=2.5, label='Target')
    ax4.plot(hist_off['t'], hist_off['altitude'], color=c_off, lw=3.0, alpha=0.9, label='FTC OFF')
    ax4.plot(hist_on['t'], hist_on['altitude'], color=c_on, lw=3.0, alpha=0.9, label='FTC ON')
    ax4.set_ylabel('Altitude [m]', fontweight='bold')

    ax5 = fig.add_subplot(gs[1, 1])
    ax5.axhline(0, color=c_cmd, ls='--', lw=2.5, label='Ideal (0°)')
    ax5.plot(hist_off['t'], hist_off['beta'], color=c_off, lw=3.0, alpha=0.9, label='FTC OFF')
    ax5.plot(hist_on['t'], hist_on['beta'], color=c_on, lw=3.0, alpha=0.9, label='FTC ON')
    ax5.set_ylabel('Sideslip (Beta) [deg]', fontweight='bold')

    ax6 = fig.add_subplot(gs[1, 2])
    ax6.plot(hist_off['t'], hist_off['alpha'], color=c_off, lw=3.0, alpha=0.9, label='FTC OFF')
    ax6.plot(hist_on['t'], hist_on['alpha'], color=c_on, lw=3.0, alpha=0.9, label='FTC ON')
    ax6.axhline(12.0, color='red', linestyle=':', lw=2.5, label='Stall Upper')
    ax6.axhline(-3.0, color='red', linestyle=':', lw=2.5, label='Stall Lower')
    ax6.set_ylabel('Angle of Attack (Alpha) [deg]', fontweight='bold')

    # -----------------------------------------------------
    # Row 3: Actuator Deflections (合并为一张超宽大图)
    # 采用高区分度的主副配色，避免线段混杂
    # -----------------------------------------------------
    ax7 = fig.add_subplot(gs[2, :])
    
    # 升降副翼 - 蓝系
    c_ele_off = '#aec7e8' # 浅蓝
    c_ele_on  = '#1f77b4' # 深蓝
    # 副翼 - 绿系
    c_ail_off = '#98df8a' # 浅绿
    c_ail_on  = '#2ca02c' # 深绿
    # 阻力舵 - 橙系
    c_spo_off = '#ffbb78' # 浅橙
    c_spo_on  = '#ff7f0e' # 深橙

    ax7.plot(hist_off['t'], hist_off['delta_e'], color=c_ele_off, ls='--', lw=2.5, label='Elevon (OFF)')
    ax7.plot(hist_on['t'],  hist_on['delta_e'],  color=c_ele_on,  ls='-',  lw=3.0, label='Elevon (ON)')

    ax7.plot(hist_off['t'], hist_off['delta_a'], color=c_ail_off, ls='--', lw=2.5, label='Aileron (OFF)')
    ax7.plot(hist_on['t'],  hist_on['delta_a'],  color=c_ail_on,  ls='-',  lw=3.0, label='Aileron (ON)')

    ax7.plot(hist_off['t'], hist_off['delta_r'], color=c_spo_off, ls='--', lw=2.5, label='Spoiler (OFF)')
    ax7.plot(hist_on['t'],  hist_on['delta_r'],  color=c_spo_on,  ls='-',  lw=3.0, label='Spoiler (ON)')

    ax7.set_ylabel('Actuator Deflections [deg]', fontweight='bold')
    ax7.set_xlabel('Time [s]', fontweight='bold')

    # 全局格式化与智能图例生成
    for ax in fig.axes:
        # 添加故障背景区 (灰色，比红色温和不遮挡曲线)
        ax.axvspan(fault_start_time, 100, color='red', alpha=0.15, label='Fault Zone' if ax == ax1 else "")
        ax.set_xlim(0, t_end)
        ax.grid(True, linestyle='--', alpha=0.5)
        
        # 提取当前子图的无重复图例
        handles, labels = ax.get_legend_handles_labels()
        by_label = dict(zip(labels, handles))
        
        if ax == ax7:
            # 宽图中的舵偏图例横向铺开
            ax.legend(by_label.values(), by_label.keys(), loc='best', ncol=3, framealpha=0.85)
        else:
            # 常规图例
            ax.legend(by_label.values(), by_label.keys(), loc='best', framealpha=0.85)

    # 智能紧凑排版
    plt.tight_layout()
    
    # 📂 双格式自动保存机制
    os.makedirs('./logs/', exist_ok=True)
    save_path = './logs/eval_outer_fault_comparison_combined_actuators'
    
    plt.savefig(f'{save_path}.pdf', format='pdf', bbox_inches='tight')
    plt.savefig(f'{save_path}.png', format='png', bbox_inches='tight', dpi=300)
    
    print(f"\n✅ 图表已保存:\n - 矢量图: {save_path}.pdf\n - 位图: {save_path}.png")
    plt.show()

if __name__ == "__main__":
    main()