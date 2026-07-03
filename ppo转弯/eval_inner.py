import os
import math
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from stable_baselines3 import PPO

# 导入你的环境构建函数
from train_inner_fault import make_env

def run_simulation_loop(paths_s3, model3_path, test_mode='7stage'):
    """
    封装原版 eval_inner 的核心执行逻辑。
    新增 test_mode 参数以支持 '7stage' (原七阶段大纲) 和 'sine' (正弦追踪)。
    """
    if not os.path.exists(model3_path):
        print(f"❌ 找不到 Stage 3 主模型: {model3_path}") 
        return None

    env_creator = make_env(stage=3, model_paths=paths_s3, seed=1024)
    env = env_creator()
    model3 = PPO.load(model3_path[:-4], device='cpu')
    
    obs, _ = env.reset()
    
    # 强制平飞初始化，抹除域随机化的影响，提供统一的基准起点
    env.sim.set_initial_state(3000.0, 200.0, theta_deg=2.0)
    env.sim.state[6] = 0.0 
    for _ in range(5): env._update_history()
    
    history = {
        't': [], 'theta': [], 'phi': [], 'beta': [], 'alpha': [],
        'target_theta': [], 'target_phi': [],
        'd_flap': [], 'd_ail': [], 'd_spoil': [],
        'velocity': [], 'altitude': []  
    }
    env.domain_rand = False
    env.ftc_enabled = True 
    
    # 👑 根据测试模式决定仿真时长
    total_steps = 10000 if test_mode == '7stage' else 5000  # 7阶段跑100s，正弦跑50s(5个完整周期)
    
    # 指令平滑器状态变量
    cmd_theta, cmd_phi = 2.0, 0.0
    filter_alpha = 0.02 # 模拟 0.5 秒的摇杆推拉缓冲
    
    for step in range(total_steps):
        t = step * env.dt
        
        if test_mode == '7stage':
            # ====================================================
            # 👑 全新 7 阶段试飞大纲 (Ultimate Flight Profile)
            # ====================================================
            if t < 10.0:
                raw_target_theta, raw_target_phi = 2.0, 0.0
            elif t < 25.0:
                raw_target_theta, raw_target_phi = 8.0, 0.0
            elif t < 40.0:
                raw_target_theta, raw_target_phi = 5.0, 15.0
            elif t < 55.0:
                raw_target_theta, raw_target_phi = -3.0, 0.0
            elif t < 70.0:
                raw_target_theta, raw_target_phi = 2.0, -15.0
            elif t < 85.0:
                raw_target_theta, raw_target_phi = 9.0, -13.0
            else:
                raw_target_theta, raw_target_phi = 2.0, 0.0
                
            # 👑 指令平滑化
            cmd_theta = (1 - filter_alpha) * cmd_theta + filter_alpha * raw_target_theta
            cmd_phi = (1 - filter_alpha) * cmd_phi + filter_alpha * raw_target_phi
            
        elif test_mode == 'sine':
            # ====================================================
            # 🌊 连续正弦追踪大纲 (Sine Wave Tracking)
            # 俯仰角: 基准 2.0°, 振幅 ±4.0°, 频率 0.1Hz
            # ====================================================
            cmd_theta = 2.0 + 4.0 * math.sin(2 * math.pi * 0.1 * t)
            cmd_phi = 0.0
            
        env.target_theta = cmd_theta
        env.target_phi = cmd_phi
        env.target_beta = 0.0 
        
        action, _ = model3.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, _ = env.step(action)
        
        state = env.sim.state
        u, v, w = state[3], state[4], state[5]
        V = max(math.sqrt(u**2 + v**2 + w**2), 1.0)
        
        history['t'].append(t)
        history['theta'].append(math.degrees(state[7]))
        history['phi'].append(math.degrees(state[6]))
        history['beta'].append(math.degrees(math.asin(np.clip(v / V, -1.0, 1.0))))
        history['alpha'].append(math.degrees(math.atan2(w, u)))
        
        history['target_theta'].append(cmd_theta)
        history['target_phi'].append(cmd_phi)
        
        history['d_flap'].append(env.prev_actions['e'])
        history['d_ail'].append(env.prev_actions['a'])
        history['d_spoil'].append(env.prev_actions['r'])
        
        history['velocity'].append(V)
        history['altitude'].append(-state[2])
        
        if terminated:
            print(f"💥 发生坠机截断于 {t:.2f} 秒！")
            break

    return history

def run_simulation_loop1(paths_s3, model3_path, test_mode='7stage'):
    """
    供 Fault 模型专用的测试入口，同样增加了 test_mode 支持。
    """
    from train_inner_compare import make_env
    if not os.path.exists(model3_path):
        print(f"❌ 找不到 Stage 3 主模型: {model3_path}") 
        return None
    env_creator = make_env(stage=3, model_paths=paths_s3, seed=1024)
    env = env_creator()
    model3 = PPO.load(model3_path[:-4], device='cpu')
    
    obs, _ = env.reset()
    
    env.sim.set_initial_state(3000.0, 200.0, theta_deg=2.0)
    env.sim.state[6] = 0.0 
    for _ in range(5): env._update_history()
    history = {
        't': [], 'theta': [], 'phi': [], 'beta': [], 'alpha': [],
        'target_theta': [], 'target_phi': [],
        'd_flap': [], 'd_ail': [], 'd_spoil': [],
        'velocity': [], 'altitude': []  
    }
    
    total_steps = 10000 if test_mode == '7stage' else 5000
    cmd_theta, cmd_phi = 2.0, 0.0
    filter_alpha = 0.02 
    
    for step in range(total_steps):
        t = step * env.dt
        
        if test_mode == '7stage':
            if t < 10.0:
                raw_target_theta, raw_target_phi = 2.0, 0.0
            elif t < 25.0:
                raw_target_theta, raw_target_phi = 8.0, 0.0
            elif t < 40.0:
                raw_target_theta, raw_target_phi = 5.0, 15.0
            elif t < 55.0:
                raw_target_theta, raw_target_phi = -3.0, 0.0
            elif t < 70.0:
                raw_target_theta, raw_target_phi = 2.0, -15.0
            elif t < 85.0:
                raw_target_theta, raw_target_phi = 9.0, -13.0
            else:
                raw_target_theta, raw_target_phi = 2.0, 0.0
                
            cmd_theta = (1 - filter_alpha) * cmd_theta + filter_alpha * raw_target_theta
            cmd_phi = (1 - filter_alpha) * cmd_phi + filter_alpha * raw_target_phi
            
        elif test_mode == 'sine':
            cmd_theta = 2.0 + 4.0 * math.sin(2 * math.pi * 0.1 * t)
            cmd_phi = 0.0
            
        env.target_theta = cmd_theta
        env.target_phi = cmd_phi
        env.target_beta = 0.0 
        
        action, _ = model3.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, _ = env.step(action)
        
        state = env.sim.state
        u, v, w = state[3], state[4], state[5]
        V = max(math.sqrt(u**2 + v**2 + w**2), 1.0)
        
        history['t'].append(t)
        history['theta'].append(math.degrees(state[7]))
        history['phi'].append(math.degrees(state[6]))
        history['beta'].append(math.degrees(math.asin(np.clip(v / V, -1.0, 1.0))))
        history['alpha'].append(math.degrees(math.atan2(w, u)))
        
        history['target_theta'].append(cmd_theta)
        history['target_phi'].append(cmd_phi)
        
        history['d_flap'].append(env.prev_actions['e'])
        history['d_ail'].append(env.prev_actions['a'])
        history['d_spoil'].append(env.prev_actions['r'])
        
        history['velocity'].append(V)
        history['altitude'].append(-state[2])
        
        if terminated:
            print(f"💥 发生坠机截断于 {t:.2f} 秒！")
            break

    return history


def calc_and_print_metrics(hist_normal, hist_fault, title_tag):
    """
    通用量化指标计算与打印封装，保持代码整洁。
    """
    def calc_rmse(hist, key, target_key=None, target_val=None):
        if target_key:
            err = np.array(hist[key]) - np.array(hist[target_key])
        else:
            err = np.array(hist[key]) - target_val
        return float(np.sqrt(np.mean(err**2)))

    rmse_theta_n = calc_rmse(hist_normal, 'theta', target_key='target_theta')
    rmse_theta_f = calc_rmse(hist_fault, 'theta', target_key='target_theta')
    
    rmse_phi_n = calc_rmse(hist_normal, 'phi', target_key='target_phi')
    rmse_phi_f = calc_rmse(hist_fault, 'phi', target_key='target_phi')
    
    rmse_beta_n = calc_rmse(hist_normal, 'beta', target_val=0.0)
    rmse_beta_f = calc_rmse(hist_fault, 'beta', target_val=0.0)

    max_alpha_n, min_alpha_n = np.max(hist_normal['alpha']), np.min(hist_normal['alpha'])
    max_alpha_f, min_alpha_f = np.max(hist_fault['alpha']), np.min(hist_fault['alpha'])

    print("\n" + "="*65)
    print(f" 📊 [{title_tag}] 模型控制性能量化对比 (RMS 误差 & 迎角极值)")
    print("="*65)
    print(f" {'评估指标':<18} | {'Normal 模型':<12} | {'Fault 模型':<12} | {'对比评价'}")
    print("-" * 65)
    print(f" 俯仰角 (Theta) RMSE  | {rmse_theta_n:10.3f}° | {rmse_theta_f:10.3f}° | " + 
          ("Fault 占优" if rmse_theta_f < rmse_theta_n else "Normal 占优"))
    print(f" 滚转角 (Phi)   RMSE  | {rmse_phi_n:10.3f}° | {rmse_phi_f:10.3f}° | " + 
          ("Fault 占优" if rmse_phi_f < rmse_phi_n else "Normal 占优"))
    print(f" 侧滑角 (Beta)  RMSE  | {rmse_beta_n:10.3f}° | {rmse_beta_f:10.3f}° | " + 
          ("Fault 占优" if rmse_beta_f < rmse_beta_n else "Normal 占优"))
    print("-" * 65)
    print(f" 迎角 (Alpha) 最大值  | {max_alpha_n:10.2f}° | {max_alpha_f:10.2f}° | " + 
          ("Fault 更安全" if max_alpha_f < max_alpha_n else "Normal 更安全"))
    print(f" 迎角 (Alpha) 最小值  | {min_alpha_n:10.2f}° | {min_alpha_f:10.2f}° | " + 
          ("均在安全限内" if (min_alpha_n > -3 and min_alpha_f > -3) else "逼近下限"))
    print("="*65 + "\n")


def plot_2x4_comparison(hist_n7, hist_f7, hist_n_sine, hist_f_sine):
    """
    绘制 2 行 4 列的综合学术横向排版图。纯英文标注，移除所有标题，字体放大。
    Row 1: 7-stage Profile
    Row 2: Sine Wave Tracking
    """
    c_normal = '#1f77b4'   # Blue
    c_fault = '#2ca02c'    # Green
    c_cmd = 'k--'          # Black dashed
    
    # 为大号字体留出充足空间，扩展画布到 24x10
    fig, axes = plt.subplots(2, 4, figsize=(24, 10))
    
    # ==========================================
    # Row 0: 7-Stage Flight Profile
    # ==========================================
    
    # 0,0: Pitch
    axes[0, 0].plot(hist_n7['t'], hist_n7['target_theta'], c_cmd, lw=2.5, label='Command')
    axes[0, 0].plot(hist_n7['t'], hist_n7['theta'], color=c_normal, lw=2.5, alpha=0.9, label='Normal')
    axes[0, 0].plot(hist_f7['t'], hist_f7['theta'], color=c_fault, lw=2.5, alpha=0.8, label='Fault')
    axes[0, 0].set_ylabel('Pitch (θ) [deg]', fontweight='bold')
    axes[0, 0].legend(loc='lower right', framealpha=0.85)
    
    # 0,1: Roll
    axes[0, 1].plot(hist_n7['t'], hist_n7['target_phi'], c_cmd, lw=2.5, label='Command')
    axes[0, 1].plot(hist_n7['t'], hist_n7['phi'], color=c_normal, lw=2.5, alpha=0.9, label='Normal')
    axes[0, 1].plot(hist_f7['t'], hist_f7['phi'], color=c_fault, lw=2.5, alpha=0.8, label='Fault')
    axes[0, 1].set_ylabel('Roll (φ) [deg]', fontweight='bold')
    axes[0, 1].legend(loc='lower right', framealpha=0.85)
    
    # 0,2: Sideslip
    axes[0, 2].axhline(0, color='black', linestyle='--', lw=2.0, label='Ideal (0°)')
    axes[0, 2].plot(hist_n7['t'], hist_n7['beta'], color=c_normal, lw=2.5, alpha=0.9, label='Normal')
    axes[0, 2].plot(hist_f7['t'], hist_f7['beta'], color=c_fault, lw=2.5, alpha=0.8, label='Fault')
    axes[0, 2].set_ylabel('Sideslip (β) [deg]', fontweight='bold')
    axes[0, 2].legend(loc='upper right', framealpha=0.85)
    
    # 0,3: AoA
    axes[0, 3].plot(hist_n7['t'], hist_n7['alpha'], color=c_normal, lw=2.5, alpha=0.9, label='Normal')
    axes[0, 3].plot(hist_f7['t'], hist_f7['alpha'], color=c_fault, lw=2.5, alpha=0.8, label='Fault')
    axes[0, 3].axhline(12.0, color='red', linestyle=':', lw=2.5, label='Stall Upper')
    axes[0, 3].axhline(-3.0, color='red', linestyle=':', lw=2.5, label='Stall Lower')
    axes[0, 3].set_ylabel('AoA (α) [deg]', fontweight='bold')
    axes[0, 3].legend(loc='upper right', framealpha=0.85)

    # ==========================================
    # Row 1: Sine Tracking Profile
    # ==========================================
    
    # 1,0: Pitch
    axes[1, 0].plot(hist_n_sine['t'], hist_n_sine['target_theta'], c_cmd, lw=2.5, label='Command')
    axes[1, 0].plot(hist_n_sine['t'], hist_n_sine['theta'], color=c_normal, lw=2.5, alpha=0.9, label='Normal')
    axes[1, 0].plot(hist_f_sine['t'], hist_f_sine['theta'], color=c_fault, lw=2.5, alpha=0.8, label='Fault')
    axes[1, 0].set_xlabel('Time [s]', fontweight='bold')
    axes[1, 0].set_ylabel('Pitch (θ) [deg]', fontweight='bold')
    axes[1, 0].legend(loc='lower right', framealpha=0.85)
    
    # 1,1: Roll
    axes[1, 1].plot(hist_n_sine['t'], hist_n_sine['target_phi'], c_cmd, lw=2.5, label='Command')
    axes[1, 1].plot(hist_n_sine['t'], hist_n_sine['phi'], color=c_normal, lw=2.5, alpha=0.9, label='Normal')
    axes[1, 1].plot(hist_f_sine['t'], hist_f_sine['phi'], color=c_fault, lw=2.5, alpha=0.8, label='Fault')
    axes[1, 1].set_xlabel('Time [s]', fontweight='bold')
    axes[1, 1].set_ylabel('Roll (φ) [deg]', fontweight='bold')
    axes[1, 1].legend(loc='lower right', framealpha=0.85)
    
    # 1,2: Sideslip
    axes[1, 2].axhline(0, color='black', linestyle='--', lw=2.0, label='Ideal (0°)')
    axes[1, 2].plot(hist_n_sine['t'], hist_n_sine['beta'], color=c_normal, lw=2.5, alpha=0.9, label='Normal')
    axes[1, 2].plot(hist_f_sine['t'], hist_f_sine['beta'], color=c_fault, lw=2.5, alpha=0.8, label='Fault')
    axes[1, 2].set_xlabel('Time [s]', fontweight='bold')
    axes[1, 2].set_ylabel('Sideslip (β) [deg]', fontweight='bold')
    axes[1, 2].legend(loc='upper right', framealpha=0.85)
    
    # 1,3: AoA
    axes[1, 3].plot(hist_n_sine['t'], hist_n_sine['alpha'], color=c_normal, lw=2.5, alpha=0.9, label='Normal')
    axes[1, 3].plot(hist_f_sine['t'], hist_f_sine['alpha'], color=c_fault, lw=2.5, alpha=0.8, label='Fault')
    axes[1, 3].axhline(12.0, color='red', linestyle=':', lw=2.5, label='Stall Upper')
    axes[1, 3].axhline(-3.0, color='red', linestyle=':', lw=2.5, label='Stall Lower')
    axes[1, 3].set_xlabel('Time [s]', fontweight='bold')
    axes[1, 3].set_ylabel('AoA (α) [deg]', fontweight='bold')
    axes[1, 3].legend(loc='upper right', framealpha=0.85)

    # 紧凑排版
    plt.tight_layout()


def run_comprehensive_flight_test():
    print("=====================================================")
    print("  🚀 X-47B 飞翼全轴内环综合机动追踪 (双模型 bmh 风格对比)")
    print("=====================================================")
    
    paths_s3_normal = {'dir': './logs/best_model_stage1fault/best_model', 'lat': './logs/best_model_stage2fault/best_model'}
    model3_path_normal = "./logs/best_model_stage3fault/best_model.zip"
    
    paths_s3_fault = {'dir': 'ppo_dir_stage1_paper', 'lat': 'ppo_lat_stage2_paper'}
    model3_path_fault = "./logs_paper/best_model_stage3/best_model.zip"
    
    # ---- 任务 1：7 阶段综合大纲测试 ----
    print("\n[任务 1/2] 🛫 正在运行：7 阶段综合大纲测试...")
    hist_normal_7stage = run_simulation_loop(paths_s3_normal, model3_path_normal, test_mode='7stage')
    hist_fault_7stage = run_simulation_loop1(paths_s3_fault, model3_path_fault, test_mode='7stage')
    
    if hist_normal_7stage is None or hist_fault_7stage is None:
        print("❌ 模型读取不全，跳过可视化绘图。")
        return

    calc_and_print_metrics(hist_normal_7stage, hist_fault_7stage, title_tag="7-Stage Profile")

    # ---- 任务 2：正弦连续追踪测试 ----
    print("\n[任务 2/2] 🛫 正在运行：动态正弦追踪测试...")
    hist_normal_sine = run_simulation_loop(paths_s3_normal, model3_path_normal, test_mode='sine')
    hist_fault_sine = run_simulation_loop1(paths_s3_fault, model3_path_fault, test_mode='sine')
    
    calc_and_print_metrics(hist_normal_sine, hist_fault_sine, title_tag="Sine Tracking")

    # ====================================================
    # 👑 全局画板样式初始化 (学术大字体版)
    # ====================================================
    plt.style.use('bmh')
    
    # 将图表字体切换为标准学术衬线字体
    plt.rcParams['font.family'] = 'serif'
    plt.rcParams['font.serif'] = ['Times New Roman', 'DejaVu Serif']
    plt.rcParams['axes.unicode_minus'] = False
    
    # 全局超大字体配置
    plt.rcParams['axes.labelsize'] = 25    # X/Y 轴标签
    plt.rcParams['xtick.labelsize'] = 25   # X 轴刻度
    plt.rcParams['ytick.labelsize'] = 25   # Y 轴刻度
    plt.rcParams['legend.fontsize'] = 25   # 图例字体
    
    print("🛬 试飞完毕！正在生成 2x4 学术论文排版大图...")
    
    # 直接生成无标题、全英文的 2x4 对比图
    plot_2x4_comparison(hist_normal_7stage, hist_fault_7stage, 
                        hist_normal_sine, hist_fault_sine)
    
    # ==========================================
    # 📂 双格式自动保存机制 (包含 PDF 矢量图)
    # ==========================================
    os.makedirs('./logs/', exist_ok=True)
    save_path = './logs/eval_inner_2x4_comparison'
    
    # 保存为无损的 PDF，用于论文 LaTeX 排版，bbox_inches='tight' 防止标签被裁
    plt.savefig(f'{save_path}.pdf', format='pdf', bbox_inches='tight')
    # 同步保存一份高清 300dpi 的 PNG 方便预览
    plt.savefig(f'{save_path}.png', format='png', bbox_inches='tight', dpi=300)
    
    print(f"✅ 图表已保存:\n - 矢量图: {save_path}.pdf\n - 位图: {save_path}.png")
    
    # 集中展现
    plt.show()

if __name__ == "__main__":
    run_comprehensive_flight_test()