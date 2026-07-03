import pickle
import numpy as np
from scipy.interpolate import RegularGridInterpolator
import time

def upsample_for_casadi():
    print(">>> 正在读取低密度数据库 unified_db.pkl ...")
    start_time = time.time()
    
    with open('unified_db.pkl', 'rb') as f:
        db = pickle.load(f)

    # 1. 提取并扩展气动网格 (所有维度强制扩充至 >= 8 个点)
    low_aero_grids = db['aero_grids']
    high_aero_grids = [np.linspace(g[0], g[-1], max(8, len(g))) for g in low_aero_grids]
    
    target_shape = [len(g) for g in high_aero_grids]
    print(f"目标高维 B-Spline 网格大小: {target_shape} (共 {np.prod(target_shape)} 个节点)")
    
    # 构建高维查询点
    high_aero_mesh = np.meshgrid(*high_aero_grids, indexing='ij')
    high_aero_points = np.column_stack([m.ravel() for m in high_aero_mesh])
    
    high_aero_data = {}
    for col_name, data in db['aero_data'].items():
        print(f"  -> 正在使用张量插值平滑扩充 [{col_name}] ...")
        interp = RegularGridInterpolator(low_aero_grids, data)
        # 将低密度数据平滑映射到高密度网格上
        high_aero_data[col_name] = interp(high_aero_points).reshape(target_shape)

    # 2. 提取并扩展发动机网格 (保证推力也平滑)
    low_eng_grids = db['eng_grids']
    high_eng_grids = [np.linspace(g[0], g[-1], max(8, len(g))) for g in low_eng_grids]
    
    high_eng_mesh = np.meshgrid(*high_eng_grids, indexing='ij')
    high_eng_points = np.column_stack([m.ravel() for m in high_eng_mesh])
    
    print("  -> 正在平滑扩充发动机推力面...")
    eng_interp = RegularGridInterpolator(low_eng_grids, db['eng_data'])
    high_eng_data = eng_interp(high_eng_points).reshape([len(g) for g in high_eng_grids])

    # 3. 保存专供 B样条使用的高保真数据库
    db_bspline = {
        'aero_grids': high_aero_grids,
        'aero_data': high_aero_data,
        'eng_grids': high_eng_grids,
        'eng_data': high_eng_data
    }
    
    with open('unified_db_bspline.pkl', 'wb') as f:
        pickle.dump(db_bspline, f)
        
    print(f"✅ 升维完成！耗时: {time.time() - start_time:.2f} 秒。")
    print("已生成完美适配 CasADi B-Spline (阶数3) 的 unified_db_bspline.pkl！")

if __name__ == "__main__":
    upsample_for_casadi()