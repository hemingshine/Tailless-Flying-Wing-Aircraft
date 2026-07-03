#coding=utf-8
import pandas as pd
import numpy as np
import pickle
import os
import pandas as pd
import numpy as np
from scipy.interpolate import LinearNDInterpolator
import pickle
import os
import warnings

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

    def load_or_build(self, excel_path, pickle_path="X47B.pkl"):
        """优先从本地缓存秒读，没有缓存才去读Excel。"""
        if os.path.exists(pickle_path):
            print(f"检测到混合型数据库缓存 '{pickle_path}'，正在秒速加载...")
            self._load_from_pickle(pickle_path)
        else:
            print(f"未检测到缓存，准备从 Excel 构建数据库...")
            self._build_from_excel(excel_path, 'M=0.4')
            self._build_from_excel(excel_path, 'M=0.6')
            self._build_from_excel(excel_path, 'M=0.8')
            
            self._compile_interpolators()
            self._save_to_pickle(pickle_path)

    def _build_from_excel(self, excel_path, sheet_name=0):
        print(f"正在读取数据: {excel_path} (Sheet: {sheet_name}) ...")
        
        df_raw = pd.read_excel(excel_path, header=None, sheet_name=sheet_name)
        is_header_row = df_raw.apply(lambda row: row.astype(str).str.contains('模型代号').any(), axis=1)
        
        if not is_header_row.any():
            raise ValueError("没有找到包含 '模型代号' 的表头行！")
            
        header_idx = is_header_row.idxmax()
        df = df_raw.iloc[header_idx + 1:].copy()
        df.columns = df_raw.iloc[header_idx].values
        df.columns = df.columns.astype(str).str.strip()
        
        df['迎角（°）'] = pd.to_numeric(df['迎角（°）'], errors='coerce')
        df = df.dropna(subset=['迎角（°）']).ffill()
        
        # 将马赫数也加入强制转换，防止文本格式出错
        for col in ['马赫数', '迎角（°）', '侧滑角（°）'] + self.output_cols:
            df[col] = df[col].astype(float)

        for index, row in df.iterrows():
            model_name = str(row['模型代号']).strip()
            mach_val = round(float(row['马赫数']), 4)
            alpha = float(row['迎角（°）'])
            beta = float(row['侧滑角（°）'])
            
            # 【关键修改】：将数据按 模型代号 -> 马赫数 进行双层分组
            if model_name not in self.raw_data:
                self.raw_data[model_name] = {}
            if mach_val not in self.raw_data[model_name]:
                self.raw_data[model_name][mach_val] = {'X': [], 'Y': []}
                
            # X 现在只存 [迎角, 侧滑角]
            self.raw_data[model_name][mach_val]['X'].append([alpha, beta])
            self.raw_data[model_name][mach_val]['Y'].append([float(row[col]) for col in self.output_cols])

    def _compile_interpolators(self):
        print("正在按马赫数分层编译插值模型...")
        
        for model_name, mach_dict in self.raw_data.items():
            self.models_db[model_name] = {}
            
            for mach_val, data in mach_dict.items():
                X = np.array(data['X'])
                Y = np.array(data['Y'])
                
                X_unique, unique_indices = np.unique(X, axis=0, return_index=True)
                Y_unique = Y[unique_indices]
                
                # 检查迎角(列0)和侧滑角(列1)谁在变
                active_dims = [i for i in range(X_unique.shape[1]) if np.max(X_unique[:, i]) > np.min(X_unique[:, i])]
                
                if len(active_dims) == 1:
                    idx = active_dims[0]
                    sort_idx = np.argsort(X_unique[:, idx])
                    interp_model = {
                        'type': '1D',
                        'active_dim': idx,
                        'interp': interp1d(X_unique[sort_idx, idx], Y_unique[sort_idx], axis=0, fill_value="extrapolate")
                    }
                elif len(active_dims) > 1:
                    interp_model = {
                        'type': 'ND',
                        'active_dims': active_dims,
                        'interp': LinearNDInterpolator(X_unique[:, active_dims], Y_unique)
                    }
                else:
                    interp_model = {'type': '0D', 'val': Y_unique[0]}
                    
                self.models_db[model_name][mach_val] = interp_model
                
        print(f"编译完成！共处理了 {len(self.models_db)} 种模型代号。")
        self.raw_data.clear()

    def _save_to_pickle(self, pickle_path):
        with open(pickle_path, 'wb') as f:
            pickle.dump(self.models_db, f)
        print(f"数据已缓存至 '{pickle_path}'\n" + "-"*40)

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
        
        # 输入自变量：高度、马赫数、油门百分比
        self.input_cols = ['Alt（m）', 'Ma']
        # 输出因变量：净推力
        self.output_cols = ['FN（DaN）']

    def load_or_build(self, excel_path, pickle_path="engine_cache.pkl", sheet_name='原始数据'):
        """缓存秒加载逻辑"""
        if os.path.exists(pickle_path):
            print(f"检测到发动机缓存文件 '{pickle_path}'，正在秒速加载...")
            with open(pickle_path, 'rb') as f:
                self.thrust_interpolator = pickle.load(f)
            print("发动机数据库加载成功！\n" + "-"*40)
        else:
            print(f"未检测到发动机缓存，准备从 Excel 构建 (这可能需要几秒钟)...")
            self._build_from_excel(excel_path, sheet_name)
            
            with open(pickle_path, 'wb') as f:
                pickle.dump(self.thrust_interpolator, f)
            print(f"发动机数据已缓存至 '{pickle_path}'！\n" + "-"*40)

    def _build_from_excel(self, excel_path, sheet_name=0):
        """从 Excel 清洗数据并构建高维插值器"""
        print(f"正在读取发动机数据: {excel_path} ...")
        
        # 读取数据 (假设发动机数据表头在第一行，如果也有说明文字，请参考气动代码盲读找表头的逻辑)
        df_r = pd.read_excel(excel_path, sheet_name=sheet_name)
        is_header_row = df_r.apply(lambda row: row.astype(str).str.contains('').any(), axis=1)
        
        if not is_header_row.any():
            raise ValueError("没有找到包含 'Alt（m）' 的表头行！")
            
        header_idx = is_header_row.idxmax()
        df = df_r.iloc[header_idx + 1:].copy()
        df.columns = df_r.iloc[header_idx].values
        df.columns = df.columns.astype(str).str.strip()
        df.columns = df.columns.astype(str).str.strip()
        
        # 确保必需的列存在
        for col in self.input_cols + self.output_cols:
            if col not in df.columns:
                raise KeyError(f"缺少发动机关键参数列：'{col}'")
            # 强制转换为数字，非数字变成 NaN，然后清理掉
            df[col] = pd.to_numeric(df[col], errors='coerce')
            
        df = df.dropna(subset=self.input_cols + self.output_cols)
        
        # 提取输入 (N行 x 3列) 和输出 (N行 x 1列)
        X = df[self.input_cols].values
        Y = df[self.output_cols[0]].values
        
        print("正在构建推力插值器...")
        self.thrust_interpolator = LinearNDInterpolator(X, Y)

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

# =================测试用例=================
# if __name__ == "__main__":
    # engine_db = EngineDatabase()
    
    # # 假设你的发动机文件叫 engine_data.xlsx
    # engine_db.load_or_build('engine.xlsx', pickle_path='engine.pkl')
    
    # # 模拟飞行查询：高度 5000米，马赫 0.6，油门(RNL) 85%
    # thrust = engine_db.get_thrust_newtons(1000, 0.6)
    # print(f"当前状态下，发动机推力为: {thrust} 牛顿 (N)")
    

# =================测试用例=================
if __name__ == "__main__":
    db = HybridAeroDatabase()
    
    # 以后你只需要调用这一行代码！
    # 第一个参数是你的原始 Excel，第二个参数是你希望保存的缓存文件名。
    # 第一次运行会比较慢，第二次运行就会瞬间完成。
    db.load_or_build('C1-X47B.xlsx', pickle_path='X47B.pkl')
    
    # 如果你修改了原始的 Excel 文件，只需要把本地的 'X47B_aero_cache.pkl' 删掉，
    # 程序就会自动重新读取 Excel 并生成新的缓存。
    
    # 测试查询
    coeffs = db.get_body_axis_coeffs('state01', 0.8, 3.0, 0.0)
    print(coeffs)