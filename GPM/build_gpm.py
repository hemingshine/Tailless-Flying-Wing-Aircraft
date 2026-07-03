import pandas as pd
import numpy as np
import pickle
from scipy.interpolate import griddata, NearestNDInterpolator
import os
import warnings
import concurrent.futures
import multiprocessing
import time

warnings.filterwarnings('ignore')

# =====================================================================
# 独立的工作函数（必须放在全局作用域，以便多进程进行 Pickle 序列化分发）
# =====================================================================
def interpolate_worker(args):
    """
    单个核心的工作任务：负责计算【某个马赫数】下的【某个气动系数】的 6D 空间投影
    """
    mach, col_name, X_raw, y_raw, grid_points, shape = args
    print(f"  [进程启动] 核心分配 -> M={mach} | {col_name} ...")
    
    # 1. 德劳内三角剖分与线性插值 (最耗时的部分)
    vals = griddata(X_raw, y_raw, grid_points, method='linear')
    
    # 2. 边缘兜底：最近邻插值填补 NaN
    if np.isnan(vals).any():
        nn_interp = NearestNDInterpolator(X_raw, y_raw)
        vals[np.isnan(vals)] = nn_interp(grid_points[np.isnan(vals)])
        
    print(f"  [进程完成] 核心释放 <- M={mach} | {col_name} 完成！")
    return mach, col_name, vals.reshape(shape)

# =====================================================================
# 主构建程序
# =====================================================================
def build_7d_database_parallel(aero_excel='C1-X47B.xlsx', engine_excel='engine.xlsx', pkl_name='unified_db.pkl'):
    start_time = time.time()
    
    # 获取 CPU 核心数
    max_workers = multiprocessing.cpu_count()
    print(f">>> 🚀 启动多核并行构建 7D 统一代理数据库 <<<")
    print(f">>> 检测到 CPU 核心数: {max_workers}，火力全开！ <<<")
    
    mach_levels = [0.4, 0.6, 0.8]
    aero_grids = [
        np.linspace(-30, 30, 5),    # 襟翼
        np.linspace(-10, 20, 5),    # 副翼
        np.linspace(-25, 0, 4),     # 前扰流
        np.linspace(0, 25, 4),      # 后扰流
        np.linspace(-3, 15, 6),     # 迎角
        np.linspace(-10, 15, 5)     # 侧滑角
    ]
    full_grids = [np.array(mach_levels)] + aero_grids
    aero_mesh = np.meshgrid(*aero_grids, indexing='ij')
    aero_points = np.column_stack([m.ravel() for m in aero_mesh])
    shape_6d = tuple(len(g) for g in aero_grids)
    
    out_cols = ['轴向力系数', '横向力系数', '法向力系数', '滚转力矩系数', '俯仰力矩系数', '偏航力矩系数']
    in_cols = ['左襟翼偏角（°）', '左副翼偏角（°）', '前扰流板偏角（°）', '后扰流板偏角（°）', '迎角（°）', '侧滑角（°）']

    # 1. 主进程：先将 Excel 里的散点提取成 Numpy 数组，准备给各个进程分发“弹药”
    task_args = []
    
    for mach in mach_levels:
        sheet_name = f'M={mach}'
        print(f"正在主进程内存中提取 {sheet_name} 数据矩阵...")
        
        df_raw = pd.read_excel(aero_excel, header=None, sheet_name=sheet_name)
        header_idx = df_raw.apply(lambda row: row.astype(str).str.contains('左襟翼偏角').any(), axis=1).idxmax()
        df_aero = df_raw.iloc[header_idx + 1:].copy()
        df_aero.columns = df_raw.iloc[header_idx].values
        df_aero.columns = df_aero.columns.astype(str).str.strip()
        
        if '模型代号' in df_aero.columns:
            df_aero = df_aero[~df_aero['模型代号'].astype(str).str.contains('min|max', case=False, na=False)]
            
        for col in in_cols: df_aero[col] = pd.to_numeric(df_aero[col], errors='coerce').ffill()
        for col in out_cols: df_aero[col] = pd.to_numeric(df_aero[col], errors='coerce')
        df_aero = df_aero.dropna(subset=in_cols + out_cols)
        
        X_raw = df_aero[in_cols].values
        
        # 将每个系数封装为一个独立任务
        for col in out_cols:
            y_raw = df_aero[col].values
            task_args.append((mach, col, X_raw, y_raw, aero_points, shape_6d))

    # 2. 进程池并行计算 (核心提速点)
    print(f"\n[并行计算] 开始将 {len(task_args)} 个高维插值任务投入进程池...")
    
    # 预准备结果容器
    mach_data_layers = {mach: {col: None for col in out_cols} for mach in mach_levels}
    
    with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as executor:
        # map 会阻塞直到所有进程计算完毕
        results = executor.map(interpolate_worker, task_args)
        
        for mach, col_name, result_grid in results:
            mach_data_layers[mach][col_name] = result_grid

    # 3. 将并行算好的 2D 切片堆叠成 7D 张量
    print("\n[张量合成] 正在将马赫切片堆叠为 7D 张量...")
    aero_7d_data = {}
    for col in out_cols:
        # 按照 mach_levels 的顺序提取出对应的 6D 网格，沿着第 0 维(Mach) stack 起来
        layers = [mach_data_layers[m][col] for m in mach_levels]
        aero_7d_data[col] = np.stack(layers, axis=0) 

    # 4. 解析引擎数据 (2D 很小，主进程瞬间秒杀)
    print("[引擎解析] 正在处理发动机推力数据...")
    df_raw_eng = pd.read_excel(engine_excel, header=None, sheet_name='原始数据')
    eng_idx = df_raw_eng.apply(lambda row: row.astype(str).str.contains('Alt（m）').any(), axis=1).idxmax()
    df_eng = df_raw_eng.iloc[eng_idx + 1:].copy()
    df_eng.columns = df_raw_eng.iloc[eng_idx].values
    df_eng.columns = df_eng.columns.astype(str).str.strip()
    
    eng_in, eng_out = ['Alt（m）', 'Ma'], 'FN（DaN）'
    for col in eng_in + [eng_out]: df_eng[col] = pd.to_numeric(df_eng[col], errors='coerce')
    df_eng = df_eng.dropna(subset=eng_in + [eng_out])
    
    eng_grids = [np.linspace(0, 10000, 11), np.linspace(0.1, 0.9, 9)]
    eng_mesh = np.meshgrid(*eng_grids, indexing='ij')
    eng_points = np.column_stack([m.ravel() for m in eng_mesh])
    eng_vals = griddata(df_eng[eng_in].values, df_eng[eng_out].values, eng_points, method='linear')
    if np.isnan(eng_vals).any():
         eng_vals[np.isnan(eng_vals)] = NearestNDInterpolator(df_eng[eng_in].values, df_eng[eng_out].values)(eng_points[np.isnan(eng_vals)])
    
    # 5. 保存
    db_pack = {
        'aero_grids': full_grids, 
        'aero_data': aero_7d_data,
        'eng_grids': eng_grids,
        'eng_data': eng_vals.reshape((11, 9)) * 10.0
    }
    with open(pkl_name, 'wb') as f: 
        pickle.dump(db_pack, f)
        
    total_time = time.time() - start_time
    print(f"\n✅ 7D 统一数据库构建完成！总耗时: {total_time:.2f} 秒。文件已保存至: {pkl_name}")

if __name__ == "__main__":
    # Windows 环境下由于没有 fork 机制，启动并行进程池必须在这个 protected block 里！
    build_7d_database_parallel()