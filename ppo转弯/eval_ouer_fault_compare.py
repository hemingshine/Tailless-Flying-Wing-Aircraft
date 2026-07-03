import os
import math
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from stable_baselines3 import PPO

# 导入外环环境
from train_outerfault import make_outer_env

def run_simulation(h_init, v_init, target_alt):
    """
    仿真引擎：支持自由设定初始高度、速度以及目标高度。
    默认全程开启 FTC 容错模块，并在 t=25s (大坡度转弯中) 注入 85% 滚转舵效丧失。
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
    # 🎯 设定目标工况
    # ==========================================
    env.target_yaw = 90.0   
    env.target_alt = target_alt 
    
    # 强制物理引擎进入指定的能量初始态
    env.inner_env.sim.set_initial_state(h_init, v_init, theta_deg=2.0)
    env.inner_env.sim.state[6] = 0.0  
    env.inner_env.sim.state[8] = 0.0  
    for _ in range(5): 
        env.inner_env._update_history()

    # ★ 配置 FTC 默认开启，关闭随机扰动以确保曲线干净对比
    env.inner_env.domain_rand = False
    env.inner_env.eff = {'pitch': 1.0, 'roll': 1.0, 'yaw': 1.0}
    env.inner_env._fault_t = 1e9
    env.inner_env.ftc_enabled = True  # 默认开启 FTC 补偿
    
    # 安全性检查：确认内环是否真正挂载了 FTC 模块
    if not hasattr(env.inner_env, 'ftc'):
        print(f"⚠️ 严重警告: 未在 inner_env 中检测到 ftc 模块！请检查 train_outerfault.py 是否导入了错误的内环！")

    env.prev_yaw_error = ((env.target_yaw - math.degrees(env.inner_env.sim.state[8]) + 180) % 360) - 180
    env.prev_alt_error = env.target_alt - (-env.inner_env.sim.state[2])
    obs = env._get_obs()

    # ==========================================
    # 🔥 物理层故障注入 (Monkey Patch)
    # 本次故障提早到 t=25s 触发
    # ==========================================
    fault_start_time = 25.0  
    sim_time = [0.0]         
    original_sim_step = env.inner_env.sim.step
    
    def faulty_sim_step(dt_val, controls):
        if sim_time[0] >= fault_start_time:
            # 损失 85% 副翼舵效 (滚转通道瘫痪)
            controls['d_ail_L'] *= 0.15
            controls['d_ail_R'] *= 0.15
            
        sim_time[0] += dt_val
        return original_sim_step(dt_val, controls)
        
    env.inner_env.sim.step = faulty_sim_step
    # ==========================================
    
    history = {
        't': [], 'yaw': [], 'target_yaw': [],
        'ftc_I_roll': [], 'alpha': [], 'altitude': []
    }
    
    # 测试时长
    max_test_steps = 1500 
    
    for step in range(max_test_steps):
        t = step * env.outer_dt
        
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, _ = env.step(action)
        
        state = env.inner_env.sim.state
        u, w = state[3], state[5]
        
        history['t'].append(t)
        history['yaw'].append(math.degrees(state[8]))
        history['target_yaw'].append(env.target_yaw)
        history['alpha'].append(math.degrees(math.atan2(w, u))) # 记录迎角
        history['altitude'].append(-state[2])                   # 记录高度
        
        # 鲁棒提取 FTC 模块内部对 Roll 轴的补偿积分状态 (I 项)
        ftc_roll_I = 0.0
        try:
            if hasattr(env.inner_env, 'ftc') and env.inner_env.ftc is not None:
                ftc_roll_I = env.inner_env.ftc.roll.I
        except AttributeError:
            pass
            
        history['ftc_I_roll'].append(ftc_roll_I)
        
        if terminated or truncated:
            break

    return history


def main():
    print("=====================================================")
    print("  ✈️  X-47B 终极试飞: 不同能量状态下的航迹控制与容错补偿")
    print("=====================================================")
    
    # 定义三种典型的试飞工况 (高度, 速度, 目标高度)
    cond1 = {'h': 3000.0, 'v': 200.0, 't_h': 3800.0, 'name': 'Cond 1: 3000m, 200m/s, 3800m'}
    cond2 = {'h': 3900.0, 'v': 180.0, 't_h': 3800.0, 'name': 'Cond 2: 5000m, 180m/s, 4500m'}
    cond3 = {'h': 3200.0, 'v': 220.0, 't_h': 3500.0, 'name': 'Cond 3: 4000m, 220m/s, 4200m'}
    
    print(f"🛫 正在运行 {cond1['name']} ...")
    hist1 = run_simulation(cond1['h'], cond1['v'], cond1['t_h'])
    
    print(f"🛫 正在运行 {cond2['name']} ...")
    hist2 = run_simulation(cond2['h'], cond2['v'], cond2['t_h'])
    
    print(f"🛫 正在运行 {cond3['name']} ...")
    hist3 = run_simulation(cond3['h'], cond3['v'], cond3['t_h'])
    
    if not hist1 or not hist2 or not hist3:
        return

    # ==========================================
    # 📊 深度计算并打印核心 FTC 量化指标
    # ==========================================
    fault_start_time = 25.0
    recovery_threshold = 2.0  # 恢复容差带定为 ±2 度
    
    def extract_ftc_metrics(hist, target_yaw=90.0, target_alt=None):
        # 切割时间窗口
        idx_fault_onset = [i for i, t in enumerate(hist['t']) if fault_start_time <= t <= fault_start_time + 15.0]
        idx_steady = [i for i, t in enumerate(hist['t']) if t > fault_start_time + 15.0]
        idx_post_fault = [i for i, t in enumerate(hist['t']) if t >= fault_start_time]
        
        # 1. 最大瞬态航向偏差 (Max Transient Yaw Error)
        transient_yaw_errors = np.abs(np.array([hist['yaw'][i] for i in idx_fault_onset]) - target_yaw)
        max_transient_yaw_err = np.max(transient_yaw_errors) if len(transient_yaw_errors) > 0 else 0
        
        # 2. 故障恢复时间 (Settling Time)
        settling_time = float('inf')
        for i in idx_post_fault:
            if abs(hist['yaw'][i] - target_yaw) <= recovery_threshold:
                # 检查后续是否都稳定在这个容差内 (简单起见，向后看20步)
                future_errs = [abs(hist['yaw'][j] - target_yaw) for j in range(i, min(i+20, len(hist['yaw'])))]
                if all(e <= recovery_threshold for e in future_errs):
                    settling_time = hist['t'][i] - fault_start_time
                    break
                    
        # 3. 重建稳态误差 (Steady-State RMSE)
        if idx_steady:
            steady_yaw_err = np.array([hist['yaw'][i] for i in idx_steady]) - target_yaw
            steady_rmse = float(np.sqrt(np.mean(steady_yaw_err**2)))
        else:
            steady_rmse = float('nan')
            
        # 4. 安全代价与能量损耗 (Max Alpha & Alt Drop)
        max_alpha = np.max([hist['alpha'][i] for i in idx_post_fault])
        # 掉高 = 故障发生时的初始高度 - 故障后的最低高度
        alt_at_fault = hist['altitude'][idx_post_fault[0]]
        min_alt_after = np.min([hist['altitude'][i] for i in idx_post_fault])
        max_alt_drop = max(0, alt_at_fault - min_alt_after)
        
        return max_transient_yaw_err, settling_time, steady_rmse, max_alpha, max_alt_drop

    # 计算三组指标
    m1 = extract_ftc_metrics(hist1, target_alt=cond1['t_h'])
    m2 = extract_ftc_metrics(hist2, target_alt=cond2['t_h'])
    m3 = extract_ftc_metrics(hist3, target_alt=cond3['t_h'])

    print("\n" + "="*95)
    print(" 📊 FTC 深度性能评估矩阵 (大坡度转弯 + 85% 副翼瘫痪)")
    print("="*95)
    print(f" {'飞行工况 (高度, 速度, 目标高)':<32} | {'最大瞬态偏差':<13} | {'恢复调节时间':<13} | {'重建稳态RMSE':<13} | {'最大迎角 / 掉高'}")
    print("-" * 95)
    
    def format_row(name, m):
        settle_str = f"{m[1]:.1f} s" if m[1] != float('inf') else ">超时"
        return f" {name:<32} | {m[0]:>8.2f}°     | {settle_str:>9}     | {m[2]:>8.2f}°     | {m[3]:5.1f}° / {m[4]:4.0f}m"
        
    print(format_row(cond1['name'], m1))
    print(format_row(cond2['name'], m2))
    print(format_row(cond3['name'], m3))
    print("="*95 + "\n")

    print("🛬 试飞完毕！正在生成全英文 1行3列 矢量大图...")
    
    # ==========================================
    # 📊 绘制专业遥测图表 (无标题，纯英文，1x3 横向排版，大号字体)
    # ==========================================
    plt.style.use('bmh')
    
    # 将图表字体切换为标准学术衬线字体
    plt.rcParams['font.family'] = 'serif'
    plt.rcParams['font.serif'] = ['Times New Roman', 'DejaVu Serif']
    plt.rcParams['axes.unicode_minus'] = False
    
    # 全局超大字体配置
    plt.rcParams['axes.labelsize'] = 25    # X/Y 轴标签
    plt.rcParams['xtick.labelsize'] = 25   # X 轴刻度
    plt.rcParams['ytick.labelsize'] = 25   # Y 轴刻度
    plt.rcParams['legend.fontsize'] = 19   # 图例字体
    
    c_cond1 = '#1f77b4'  # 经典蓝
    c_cond2 = '#ff7f0e'  # 警戒橙
    c_cond3 = '#2ca02c'  # 安全绿
    c_cmd   = 'black'    # 黑色指令线
    
    t_end = 150.0
    
    # 增加图片高与宽以容纳更大的字体
    fig = plt.figure(figsize=(24, 7))
    gs = GridSpec(1, 3, figure=fig)
    
    # --- 左图 (1)：三种工况下的航迹角 (Yaw) 追踪 ---
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.plot(hist1['t'], hist1['target_yaw'], color=c_cmd, ls='--', lw=2.0, label='Target (90°)')
    ax1.plot(hist1['t'], hist1['yaw'], color=c_cond1, lw=2.5, alpha=0.9, label=cond1['name'])
    ax1.plot(hist2['t'], hist2['yaw'], color=c_cond2, lw=2.5, alpha=0.9, label=cond2['name'])
    ax1.plot(hist3['t'], hist3['yaw'], color=c_cond3, lw=2.5, alpha=0.9, label=cond3['name'])
    
    ax1.axvspan(fault_start_time, t_end, color='red', alpha=0.1, label='Fault Zone (85% Loss)')
    ax1.set_xlabel('Time [s]', fontweight='bold')
    ax1.set_ylabel('Yaw / Heading [deg]', fontweight='bold')
    ax1.set_xlim(0, t_end)
    ax1.legend(loc='lower right', framealpha=0.85)
    
    # --- 中图 (2)：底层容错模块 (FTC) 自适应补偿积分曲线 ---
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.plot(hist1['t'], hist1['ftc_I_roll'], color=c_cond1, lw=2.5, alpha=0.9, label=cond1['name'])
    ax2.plot(hist2['t'], hist2['ftc_I_roll'], color=c_cond2, lw=2.5, alpha=0.9, label=cond2['name'])
    ax2.plot(hist3['t'], hist3['ftc_I_roll'], color=c_cond3, lw=2.5, alpha=0.9, label=cond3['name'])
    
    ax2.axvspan(fault_start_time, t_end, color='red', alpha=0.1)
    ax2.set_xlabel('Time [s]', fontweight='bold')
    ax2.set_ylabel('FTC Integral Compensation', fontweight='bold')
    ax2.set_xlim(0, t_end)
    ax2.legend(loc='lower right', framealpha=0.85)
    
    # --- 右图 (3)：迎角安全与代偿代价 (Alpha) ---
    ax3 = fig.add_subplot(gs[0, 2])
    ax3.plot(hist1['t'], hist1['alpha'], color=c_cond1, lw=2.5, alpha=0.9, label=cond1['name'])
    ax3.plot(hist2['t'], hist2['alpha'], color=c_cond2, lw=2.5, alpha=0.9, label=cond2['name'])
    ax3.plot(hist3['t'], hist3['alpha'], color=c_cond3, lw=2.5, alpha=0.9, label=cond3['name'])
    
    # 增加失速红线 (加粗)
    ax3.axhline(12.0, color='r', linestyle='--', lw=2.0, alpha=0.7, label='Stall Upper Limit')
    ax3.axhline(-3.0, color='r', linestyle='--', lw=2.0, alpha=0.7, label='Stall Lower Limit')
    
    ax3.axvspan(fault_start_time, t_end, color='red', alpha=0.1)
    ax3.set_xlabel('Time [s]', fontweight='bold')
    ax3.set_ylabel('Angle of Attack (Alpha) [deg]', fontweight='bold')
    ax3.set_xlim(0, t_end)
    ax3.legend(loc='upper right', framealpha=0.85)
    
    plt.tight_layout()
    
    # ==========================================
    # 📂 双格式自动保存机制 (包含 PDF 矢量图)
    # ==========================================
    os.makedirs('./logs/', exist_ok=True)
    save_path = './logs/eval_turn_3conditions_comparison'
    
    # 保存为无损的 PDF，用于论文 LaTeX 排版，bbox_inches='tight' 防止标签被裁
    plt.savefig(f'{save_path}.pdf', format='pdf', bbox_inches='tight')
    # 同步保存一份高清 300dpi 的 PNG 方便预览
    plt.savefig(f'{save_path}.png', format='png', bbox_inches='tight', dpi=300)
    
    print(f"✅ 图表已保存:\n - 矢量图: {save_path}.pdf\n - 位图: {save_path}.png")
    
    plt.show()

if __name__ == "__main__":
    main()