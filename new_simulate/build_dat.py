#coding=utf-8
import pandas as pd
import numpy as np
import pickle
import os
import warnings
from scipy.interpolate import LinearNDInterpolator, interp1d

warnings.filterwarnings('ignore')

class HybridAeroDatabase:
    def __init__(self):
        self.raw_data = {} 
        self.models_db = {}
        
        self.control_cols = [
            '左襟翼偏角（°）', '右襟翼偏角（°）', 
            '左副翼偏角（°）', '右副翼偏角（°）', 
            '前扰流板偏角（°）', '后扰流板偏角（°）'
        ]
        self.state_cols = ['迎角（°）', '侧滑角（°）']
        self.input_cols = self.control_cols + self.state_cols
        
        self.output_cols = [
            '轴向力系数', '横向力系数', '法向力系数', 
            '滚转力矩系数', '俯仰力矩系数', '偏航力矩系数'
        ]

    def load_or_build(self, excel_path, pickle_path="X47B.pkl"):
        """优先从本地缓存加载，没有缓存才去读Excel。"""
        if os.path.exists(pickle_path):
            print(f"检测到气动数据库缓存 '{pickle_path}'，正在秒速加载...")
            self._load_from_pickle(pickle_path)
        else:
            print(f"未检测到缓存，准备从 Excel 构建气动数据库 (这需要解析多维特征)...")
            
            target_sheets = ['M=0.4', 'M=0.6', 'M=0.8']
            for sheet in target_sheets:
                print(f"开始处理 Sheet: {sheet}")
                self._build_from_excel(excel_path, sheet_name=sheet) 
            
            self._compile_interpolators()
            self._save_to_pickle(pickle_path)

    def _build_from_excel(self, excel_path, sheet_name=0):
        print(f"正在读取数据: {excel_path} (Sheet: {sheet_name}) ...")
        
        df_raw = pd.read_excel(excel_path, header=None, sheet_name=sheet_name)
        is_header_row = df_raw.apply(lambda row: row.astype(str).str.contains('左襟翼偏角').any(), axis=1)
        
        if not is_header_row.any():
            raise ValueError("没有找到包含 '左襟翼偏角（°）' 的表头行！请检查 Excel 格式。")
            
        header_idx = is_header_row.idxmax()
        df = df_raw.iloc[header_idx + 1:].copy()
        df.columns = df_raw.iloc[header_idx].values
        df.columns = df.columns.astype(str).str.strip()
        
        if '模型代号' in df.columns:
            df = df[~df['模型代号'].astype(str).str.contains('min|max', case=False, na=False)]
        
        fill_cols = self.control_cols + ['马赫数', '迎角（°）', '侧滑角（°）']
        for col in fill_cols:
            df[col] = pd.to_numeric(df[col], errors='coerce').ffill()
            
        df = df.dropna(subset=['迎角（°）', '马赫数'])
        
        for col in self.output_cols:
            df[col] = pd.to_numeric(df[col], errors='coerce')

        for index, row in df.iterrows():
            mach_val = round(float(row['马赫数']), 4)
            x_val = [float(row[col]) for col in self.input_cols]
            y_val = [float(row[col]) for col in self.output_cols]
            
            if mach_val not in self.raw_data:
                self.raw_data[mach_val] = {'X': [], 'Y': []}
                
            self.raw_data[mach_val]['X'].append(x_val)
            self.raw_data[mach_val]['Y'].append(y_val)

    def _compile_interpolators(self):
        print("正在按马赫数编译多维(舵面+姿态)插值模型...")
        
        for mach_val, data in self.raw_data.items():
            X = np.array(data['X'])
            Y = np.array(data['Y'])
            
            X_unique, unique_indices = np.unique(X, axis=0, return_index=True)
            Y_unique = Y[unique_indices]
            
            X_centered = X_unique - np.mean(X_unique, axis=0)
            active_dims = []
            current_rank = 0
            
            for i in range(X_unique.shape[1]):
                test_cols = active_dims + [i]
                rank = np.linalg.matrix_rank(X_centered[:, test_cols])
                if rank > current_rank:
                    active_dims.append(i)
                    current_rank = rank
            
            dim_names = [self.input_cols[i] for i in active_dims]
            print(f"  [M={mach_val}] 识别出 {current_rank} 个独立变化维度: {dim_names}")
            
            # 【新增逻辑】：提取并记录各个活跃维度的上下界，用于约束表
            bounds = []
            if current_rank >= 1:
                for idx in active_dims:
                    bounds.append((np.min(X_unique[:, idx]), np.max(X_unique[:, idx])))

            if current_rank == 1:
                idx = active_dims[0]
                sort_idx = np.argsort(X_unique[:, idx])
                interp_model = {
                    'type': '1D',
                    'active_dims': active_dims,
                    'bounds': bounds, # 保存约束
                    'interp': interp1d(X_unique[sort_idx, idx], Y_unique[sort_idx], axis=0, fill_value="extrapolate")
                }
            elif current_rank > 1:
                interp_model = {
                    'type': 'ND',
                    'active_dims': active_dims,
                    'bounds': bounds, # 保存约束
                    'interp': LinearNDInterpolator(X_unique[:, active_dims], Y_unique)
                }
            else:
                interp_model = {'type': '0D', 'val': Y_unique[0], 'bounds': [], 'active_dims': []}
                
            self.models_db[mach_val] = interp_model
                
        print(f"气动编译完成！共处理了 {len(self.models_db)} 个马赫数截面的高维模型。")
        self.raw_data.clear()

    def _save_to_pickle(self, pickle_path):
        with open(pickle_path, 'wb') as f:
            pickle.dump(self.models_db, f)
        print(f"气动数据已缓存至 '{pickle_path}'\n" + "-"*40)

    def _load_from_pickle(self, pickle_path):
        with open(pickle_path, 'rb') as f:
            self.models_db = pickle.load(f)
        print(f"气动数据库加载成功！\n" + "-"*40)

    def print_input_constraints(self):
        """动态打印当前数据库的取值约束表"""
        if not self.models_db:
            print("数据库未加载，无法打印约束表！")
            return
            
        print("\n" + "="*50)
        print("📊 气动数据库：输入参数取值约束表")
        print("注意：查询时超出以下范围，该马赫数下的插值器将返回默认值 0.0")
        print("="*50)
        
        for mach, model in sorted(self.models_db.items()):
            print(f"【 马赫数 M = {mach} 】 独立维度: {len(model['active_dims'])}")
            if model['type'] == '0D':
                print("  该马赫数下所有参数均未发生变化 (仅有单一静态数据点)。")
            else:
                for i, col_idx in enumerate(model['active_dims']):
                    col_name = self.input_cols[col_idx]
                    min_val, max_val = model['bounds'][i]
                    print(f"  > {col_name:<10}: 允许范围 [ {min_val:>6.2f}  至  {max_val:>6.2f} ]")
            print("-" * 50)
        print("\n")

    def get_body_axis_coeffs(self, mach, d_flap_L, d_flap_R, d_ail_L, d_ail_R, d_spoil_F, d_spoil_R, alpha, beta):
        """核心查询函数。"""
        if not self.models_db:
            raise ValueError("气动数据库为空，请先调用 load_or_build()！")
            
        available_machs = list(self.models_db.keys())
        closest_mach = min(available_machs, key=lambda x: abs(x - mach))
        model_info = self.models_db[closest_mach]
        
        query_point = np.array([d_flap_L, d_flap_R, d_ail_L, d_ail_R, d_spoil_F, d_spoil_R, alpha, beta])
        
        if model_info['type'] == 'ND':
            q = query_point[model_info['active_dims']]
            res = model_info['interp'](q)
            res = res[0] if res.ndim > 1 else res
            if np.isnan(res).any():
                res = np.nan_to_num(res, nan=0.0)
                
        elif model_info['type'] == '1D':
            q = query_point[model_info['active_dims'][0]]
            res = model_info['interp'](q)
            
        else:
            res = model_info['val']
            
        return dict(zip(self.output_cols, res))


class EngineDatabase:
    def __init__(self):
        self.thrust_interpolator = None
        self.input_cols = ['Alt（m）', 'Ma']
        self.output_cols = ['FN（DaN）']

    def load_or_build(self, excel_path, pickle_path="engine_cache.pkl", sheet_name='原始数据'):
        """缓存秒加载逻辑，包含自动保存机制"""
        if os.path.exists(pickle_path):
            print(f"检测到发动机缓存文件 '{pickle_path}'，正在秒速加载...")
            with open(pickle_path, 'rb') as f:
                self.thrust_interpolator = pickle.load(f)
            print("发动机数据库加载成功！\n" + "-"*40)
        else:
            print(f"未检测到发动机缓存，准备从 Excel 构建 (这可能需要几秒钟)...")
            self._build_from_excel(excel_path, sheet_name)
            
            # 【这里就是发动机数据库保存为pkl缓存的地方】
            with open(pickle_path, 'wb') as f:
                pickle.dump(self.thrust_interpolator, f)
            print(f"发动机数据已缓存至 '{pickle_path}'！\n" + "-"*40)

    def _build_from_excel(self, excel_path, sheet_name=0):
        print(f"正在读取发动机数据: {excel_path} ...")
        
        df_r = pd.read_excel(excel_path, sheet_name=sheet_name)
        is_header_row = df_r.apply(lambda row: row.astype(str).str.contains('Alt（m）').any(), axis=1)
        
        if not is_header_row.any():
            raise ValueError("没有找到包含 'Alt（m）' 的表头行！")
            
        header_idx = is_header_row.idxmax()
        df = df_r.iloc[header_idx + 1:].copy()
        df.columns = df_r.iloc[header_idx].values
        df.columns = df.columns.astype(str).str.strip()
        
        for col in self.input_cols + self.output_cols:
            if col not in df.columns:
                raise KeyError(f"缺少发动机关键参数列：'{col}'")
            df[col] = pd.to_numeric(df[col], errors='coerce')
            
        df = df.dropna(subset=self.input_cols + self.output_cols)
        
        X = df[self.input_cols].values
        Y = df[self.output_cols[0]].values
        
        print("正在构建推力插值器...")
        self.thrust_interpolator = LinearNDInterpolator(X, Y)

    def get_thrust_newtons(self, alt, mach):
        if self.thrust_interpolator is None:
            raise RuntimeError("请先调用 load_or_build() 加载数据库！")
            
        query_point = np.array([[alt, mach]])
        thrust_dan = self.thrust_interpolator(query_point)[0]
        
        if np.isnan(thrust_dan):
            thrust_dan = 0.0 
            
        thrust_n = thrust_dan * 10.0
        return thrust_n


# =================综合测试用例=================
if __name__ == "__main__":
    
    # --- 1. 气动数据库测试 ---
    print(">>> 气动数据库初始化 <<<")
    aero_db = HybridAeroDatabase()
    aero_db.load_or_build('C1-X47B.xlsx', pickle_path='X47B.pkl')
    
    # 动态打印出你独有的取值约束表！
    aero_db.print_input_constraints()
    
    # coeffs = aero_db.get_body_axis_coeffs(
    #     mach=0.4, 
    #     d_flap_L=-4.0, 
    #     d_flap_R=-4.0, 
    #     d_ail_L=1.0, 
    #     d_ail_R=1.0, 
    #     d_spoil_F=6.0, 
    #     d_spoil_R=6.0, 
    #     alpha=1.0, 
    #     beta=0.0
    # )
    # print(f"气动系数查询结果: {coeffs}\n")
    
    
    # --- 2. 发动机数据库测试 ---
    print(">>> 发动机数据库初始化 <<<")
    engine_db = EngineDatabase()
    
    # 请确保同级目录下有发动机对应的excel，如果没有请修改文件名。
    # 这里只要运行一次，就会自动触发上面类里面的 pickle.dump，生成 engine_cache.pkl
    engine_db.load_or_build('engine.xlsx', pickle_path='engine.pkl')
    
    thrust = engine_db.get_thrust_newtons(alt=5000, mach=0.6)
    print(f"当前状态下，发动机推力为: {thrust} 牛顿 (N)")