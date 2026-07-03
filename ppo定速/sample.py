#coding=utf-8
import numpy as np
import time
import os
from fly_simulate import HybridAeroDatabase

print("===========================================")
print("  启动高保真气动数据采样器 (智能高斯分布版)")
print("===========================================")

flight_db = HybridAeroDatabase()
try:
    flight_db._load_from_pickle('X47B.pkl')
except Exception as e:
    print(f"找不到 X47B.pkl，请确保在同一目录下: {e}")
    exit()

N_SAMPLES = 500000 # 加大采样基数
print(f"\n开始尝试采样 {N_SAMPLES} 条数据，寻找安全凸包...")

start_time = time.time()
X_data = np.zeros((N_SAMPLES, 9))

# 【核心优化】：智能高斯采样 (Normal Distribution)
# 真实飞行中，绝大部分时间舵面和侧滑角都在 0 度附近，Alpha 在 5 度附近
X_data[:, 0] = np.random.uniform(0.5, 0.9, N_SAMPLES)                        # Mach: 均匀分布
X_data[:, 1] = np.clip(np.random.normal(0, 5.0, N_SAMPLES), -30, 30)         # d_flap_L: 集中在0附近
X_data[:, 2] = np.clip(np.random.normal(0, 5.0, N_SAMPLES), -30, 30)         # d_flap_R
X_data[:, 3] = np.clip(np.random.normal(0, 3.0, N_SAMPLES), -20, 20)         # d_ail_L
X_data[:, 4] = np.clip(np.random.normal(0, 3.0, N_SAMPLES), -20, 20)         # d_ail_R
X_data[:, 5] = np.clip(np.random.normal(0, 2.0, N_SAMPLES), -25, 25)         # d_spoil_F 
X_data[:, 6] = np.clip(np.random.normal(0, 2.0, N_SAMPLES), -25, 25)         # d_spoil_R
X_data[:, 7] = np.clip(np.random.normal(2.0, 5.0, N_SAMPLES), -10, 30)       # Alpha: 集中在2度平飞附近
X_data[:, 8] = np.clip(np.random.normal(0, 2.0, N_SAMPLES), -10, 10)         # Beta: 侧滑角极少出现大值

Y_data = np.zeros((N_SAMPLES, 6))
valid_indices = []

for i in range(N_SAMPLES):
    try:
        res = flight_db.get_body_axis_coeffs(
            X_data[i,0], X_data[i,1], X_data[i,2], X_data[i,3], 
            X_data[i,4], X_data[i,5], X_data[i,6], X_data[i,7], X_data[i,8]
        )
        if abs(res['法向力系数']) > 1e-5: # 排除真空点
            Y_data[i] = [
                res['轴向力系数'], res['横向力系数'], res['法向力系数'],
                res['滚转力矩系数'], res['俯仰力矩系数'], res['偏航力矩系数']
            ]
            valid_indices.append(i)
    except:
        pass

    if (i+1) % 50000 == 0:
        print(f"  进度: {i+1} / {N_SAMPLES} | 已捕获有效数据: {len(valid_indices)} 条")

X_valid = X_data[valid_indices]
Y_valid = Y_data[valid_indices]

print(f"\n采样完成！耗时 {time.time()-start_time:.1f} 秒。最终有效数据量: {len(X_valid)} 条。")

# 保存为 numpy 压缩格式
if len(X_valid) > 0:
    save_file = 'aero_dataset.npz'
    # 如果以前采过数据，可以合并
    if os.path.exists(save_file):
        old_data = np.load(save_file)
        X_valid = np.vstack((old_data['X'], X_valid))
        Y_valid = np.vstack((old_data['Y'], Y_valid))
        print(f"检测到历史数据，已合并。当前总数据量: {len(X_valid)} 条。")
        
    np.savez(save_file, X=X_valid, Y=Y_valid)
    print(f"✅ 数据已安全保存至 '{save_file}'，现在您可以去执行训练脚本了！")
else:
    print("❌ 警告：没有捕获到任何有效数据！")