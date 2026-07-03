#coding=utf-8
"""
validate_surrogate.py —— 气动代理模型验证
产出:
  1) 6 通道(Cx,Cy,Cz,Cl,Cm,Cn) 预测-真值散点图 (每格标 R²/RMSE)
  2) 每通道 R²/RMSE/MAE/RMSE% 验证表 (渲染成图 + 存 CSV + 控制台打印)
测试集:
  默认从高保真 HybridAeroDatabase 新采一批【独立】测试点(最严谨);
  若 fly_simulate / X47B.pkl 不可用, 自动回退用 aero_dataset.npz 固定种子留出 10%。
"""
import os
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

# ============================ 配置 ============================
SURROGATE_PATH = 'aero_surrogate.pth'
DATASET_PATH   = 'aero_dataset.npz'
USE_FRESH_TEST = True       # True: 从真值模型采独立测试集; False: 用数据集留出集
N_FRESH        = 50000      # 独立测试点采样基数
TEST_FRAC      = 0.10       # USE_FRESH_TEST=False 时的留出比例
PLOT_MAX_PTS   = 4000       # 散点最多画多少点(指标仍用全量算)
SEED           = 12345

# 通道: (数学符号, 英文名, 数据集真值模型里的中文键)
CHANNELS = [
    ('$C_X$', 'Axial force',  '轴向力系数'),
    ('$C_Y$', 'Side force',   '横向力系数'),
    ('$C_Z$', 'Normal force', '法向力系数'),
    ('$C_l$', 'Roll moment',  '滚转力矩系数'),
    ('$C_m$', 'Pitch moment', '俯仰力矩系数'),
    ('$C_n$', 'Yaw moment',   '偏航力矩系数'),
]


# ===================== 网络结构(须与 surrage.py 完全一致) =====================
class AeroSurrogate(nn.Module):
    def __init__(self):
        super(AeroSurrogate, self).__init__()

        class ResBlock(nn.Module):
            def __init__(self, in_dim, out_dim):
                super().__init__()
                self.net = nn.Sequential(
                    nn.Linear(in_dim, out_dim),
                    nn.BatchNorm1d(out_dim),
                    nn.GELU(),
                    nn.Dropout(0.2),
                )
                self.shortcut = nn.Linear(in_dim, out_dim) if in_dim != out_dim else nn.Identity()

            def forward(self, x):
                return self.net(x) + self.shortcut(x)

        self.net = nn.Sequential(
            ResBlock(9, 256),
            ResBlock(256, 512),
            ResBlock(512, 256),
            ResBlock(256, 128),
            nn.Linear(128, 6),
        )

    def forward(self, x):
        return self.net(x)


def load_surrogate(path, device):
    if not os.path.exists(path):
        raise FileNotFoundError(f"找不到代理模型 {path}")
    ckpt = torch.load(path, map_location=device)
    model = AeroSurrogate().to(device)
    model.load_state_dict(ckpt['model_state_dict'])
    model.eval()                                   # ★ BatchNorm 用运行统计, 关 Dropout
    stats = {k: ckpt[k].to(device).float() for k in ('x_mean', 'x_std', 'y_mean', 'y_std')}
    return model, stats


@torch.no_grad()
def predict(model, stats, X, device, bs=8192):
    """X: (N,9) 物理量 -> 返回 (N,6) 物理量预测"""
    Xt = torch.as_tensor(X, dtype=torch.float32, device=device)
    Xn = (Xt - stats['x_mean']) / stats['x_std']
    outs = []
    for i in range(0, len(Xn), bs):
        yn = model(Xn[i:i + bs])
        outs.append(yn)
    Yn = torch.cat(outs, dim=0)
    Y = Yn * stats['y_std'] + stats['y_mean']      # 反归一化回物理量
    return Y.cpu().numpy()


# ===================== 测试集获取 =====================
def sample_fresh_test(n, seed):
    """复刻 sample.py 的采样分布, 从高保真模型生成【独立】测试集。"""
    from fly_simulate import HybridAeroDatabase           # 仅此分支需要
    db = HybridAeroDatabase()
    db._load_from_pickle('X47B.pkl')
    rng = np.random.default_rng(seed)
    X = np.zeros((n, 9))
    X[:, 0] = rng.uniform(0.5, 0.9, n)
    X[:, 1] = np.clip(rng.normal(0, 5.0, n), -30, 30)
    X[:, 2] = np.clip(rng.normal(0, 5.0, n), -30, 30)
    X[:, 3] = np.clip(rng.normal(0, 3.0, n), -20, 20)
    X[:, 4] = np.clip(rng.normal(0, 3.0, n), -20, 20)
    X[:, 5] = np.clip(rng.normal(0, 2.0, n), -25, 25)
    X[:, 6] = np.clip(rng.normal(0, 2.0, n), -25, 25)
    X[:, 7] = np.clip(rng.normal(2.0, 5.0, n), -10, 30)
    X[:, 8] = np.clip(rng.normal(0, 2.0, n), -10, 10)
    keys = [c[2] for c in CHANNELS]
    Xv, Yv = [], []
    for i in range(n):
        try:
            r = db.get_body_axis_coeffs(*X[i])
            if abs(r['法向力系数']) > 1e-5:
                Yv.append([r[k] for k in keys]); Xv.append(X[i])
        except Exception:
            pass
    return np.asarray(Xv), np.asarray(Yv)


def get_test_data():
    if USE_FRESH_TEST:
        try:
            print('采集独立测试集 (高保真模型)...')
            X, Y = sample_fresh_test(N_FRESH, SEED)
            print(f'  独立测试点: {len(X)} 条')
            return X, Y, 'independent'
        except Exception as e:
            print(f'  采集独立测试集失败({e}); 回退到数据集留出集。')
    # 回退: 数据集固定种子留出
    if not os.path.exists(DATASET_PATH):
        raise FileNotFoundError(f"找不到数据集 {DATASET_PATH}")
    d = np.load(DATASET_PATH)
    X, Y = d['X'], d['Y']
    rng = np.random.default_rng(SEED)
    idx = rng.permutation(len(X))
    n_test = int(TEST_FRAC * len(X))
    ti = idx[:n_test]
    print(f'  数据集留出测试点: {len(ti)} 条 (共 {len(X)})')
    return X[ti], Y[ti], 'held-out'


# ===================== 指标 =====================
def per_channel_metrics(Yt, Yp):
    rows = []
    for j in range(Yt.shape[1]):
        yt, yp = Yt[:, j], Yp[:, j]
        ss_res = np.sum((yt - yp) ** 2)
        ss_tot = np.sum((yt - yt.mean()) ** 2)
        r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else float('nan')
        rmse = float(np.sqrt(np.mean((yt - yp) ** 2)))
        mae = float(np.mean(np.abs(yt - yp)))
        rng = yt.max() - yt.min()
        rmse_pct = 100.0 * rmse / rng if rng > 0 else float('nan')
        rows.append({'r2': r2, 'rmse': rmse, 'mae': mae, 'rmse_pct': rmse_pct})
    return rows


# ===================== 可视化 =====================
def paper_style():
    plt.rcParams.update({
        'font.family': 'serif', 'mathtext.fontset': 'dejavuserif',
        'font.size': 10, 'axes.titlesize': 11, 'axes.labelsize': 10,
        'legend.fontsize': 9, 'xtick.labelsize': 9, 'ytick.labelsize': 9,
        'axes.linewidth': 0.8,
    })


def plot_scatter(Yt, Yp, metrics, src):
    # 适度拉大画布，给外部标签留出空间
    fig = plt.figure(figsize=(9.6, 6.0))
    # 取消硬编码的 hspace 和 wspace，交给 tight_layout 自动计算
    gs = GridSpec(2, 3, figure=fig)
    
    rng = np.random.default_rng(0)
    m = len(Yt)
    pi = rng.choice(m, size=min(PLOT_MAX_PTS, m), replace=False)
    
    for j, (sym, name, _) in enumerate(CHANNELS):
        ax = fig.add_subplot(gs[j // 3, j % 3])
        yt, yp = Yt[pi, j], Yp[pi, j]
        ax.scatter(yt, yp, s=5, alpha=0.25, color='#1f5fb4', edgecolors='none', rasterized=True)
        lo = min(Yt[:, j].min(), Yp[:, j].min())
        hi = max(Yt[:, j].max(), Yp[:, j].max())
        pad = 0.05 * (hi - lo + 1e-9)
        lim = (lo - pad, hi + pad)
        ax.plot(lim, lim, 'r--', lw=1.0)            # y=x 理想线
        ax.set_xlim(lim); ax.set_ylim(lim)
        ax.set_aspect('equal', adjustable='box')
        ax.set_title(f'{sym}  ({name})')
        ax.set_xlabel('True'); ax.set_ylabel('Predicted')
        
        # --- 修复 1：限制坐标轴最大刻度数量，并使用科学计数法防止字体重叠 ---
        ax.xaxis.set_major_locator(plt.MaxNLocator(4))
        ax.yaxis.set_major_locator(plt.MaxNLocator(4))
        ax.ticklabel_format(style='sci', scilimits=(-3, 3), axis='both')
        
        # --- 修复 3：针对极小 RMSE 动态应用科学计数法，避免 0.000 截断 ---
        mt = metrics[j]
        rmse_val = mt['rmse']
        rmse_str = f"{rmse_val:.2e}" if rmse_val < 0.001 else f"{rmse_val:.4f}"
        
        ax.text(0.05, 0.95, f"$R^2$={mt['r2']:.4f}\nRMSE={rmse_str}",
                transform=ax.transAxes, va='top', ha='left', fontsize=9,
                bbox=dict(boxstyle='round,pad=0.3', fc='white', ec='0.6', alpha=0.85))
        ax.grid(alpha=0.25, lw=0.5)
        
    # --- 修复 2：自动收紧布局，彻底解决 "Predicted" 和子图刻度重叠的问题 ---
    plt.tight_layout()
    
    fig.savefig('./surrogate_scatter.png', dpi=300, bbox_inches='tight')
    fig.savefig('./surrogate_scatter.pdf', bbox_inches='tight')
    print('散点图已存: surrogate_scatter.png / .pdf')


def render_table(metrics, src):
    fig, ax = plt.subplots(figsize=(6.6, 2.0))
    ax.axis('off')
    col = ['Channel', '$R^2$', 'RMSE', 'MAE', 'RMSE (%)']
    cell = []
    for (sym, name, _), m in zip(CHANNELS, metrics):
        cell.append([f'{sym} ({name})', f"{m['r2']:.4f}", f"{m['rmse']:.4e}",
                     f"{m['mae']:.4e}", f"{m['rmse_pct']:.2f}"])
    # 总体均值行
    mr2 = np.mean([m['r2'] for m in metrics])
    cell.append(['Mean', f'{mr2:.4f}', '—', '—', '—'])
    tb = ax.table(cellText=cell, colLabels=col, loc='center', cellLoc='center')
    tb.auto_set_font_size(False); tb.set_fontsize(9); tb.scale(1, 1.45)
    for (r, c), cellobj in tb.get_celld().items():
        cellobj.set_edgecolor('0.7')
        if r == 0:
            cellobj.set_facecolor('#2f4b7c'); cellobj.set_text_props(color='white', fontweight='bold')
        elif r == len(cell):
            cellobj.set_facecolor('#dfe6f0'); cellobj.set_text_props(fontweight='bold')
        elif r % 2 == 0:
            cellobj.set_facecolor('#f4f6fa')
    fig.savefig('./surrogate_table.png', dpi=300, bbox_inches='tight')
    fig.savefig('./surrogate_table.pdf', bbox_inches='tight')
    print('验证表已存: surrogate_table.png / .pdf')


def save_csv(metrics):
    import csv
    with open('./surrogate_metrics.csv', 'w', newline='', encoding='utf-8-sig') as f:
        w = csv.writer(f)
        w.writerow(['Channel', 'R2', 'RMSE', 'MAE', 'RMSE_pct'])
        for (sym, name, _), m in zip(CHANNELS, metrics):
            w.writerow([f'{name}', f"{m['r2']:.6f}", f"{m['rmse']:.6e}",
                        f"{m['mae']:.6e}", f"{m['rmse_pct']:.4f}"])
    print('指标已存: surrogate_metrics.csv')


def print_table(metrics, src):
    print('\n' + '=' * 72)
    print(f' 气动代理模型验证表  (测试集: {src})')
    print('=' * 72)
    print(f" {'Channel':<20}{'R^2':>10}{'RMSE':>14}{'MAE':>14}{'RMSE%':>9}")
    print('-' * 72)
    for (sym, name, _), m in zip(CHANNELS, metrics):
        print(f" {name:<20}{m['r2']:>10.4f}{m['rmse']:>14.4e}{m['mae']:>14.4e}{m['rmse_pct']:>9.2f}")
    print('-' * 72)
    print(f" {'Mean R^2':<20}{np.mean([m['r2'] for m in metrics]):>10.4f}")
    print('=' * 72)


# ===================== 主流程 =====================
def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'计算硬件: {device}')
    model, stats = load_surrogate(SURROGATE_PATH, device)
    Xte, Yte, src = get_test_data()
    if len(Xte) == 0:
        print('测试集为空, 退出。'); return
    Yp = predict(model, stats, Xte, device)

    metrics = per_channel_metrics(Yte, Yp)
    print_table(metrics, src)

    paper_style()
    plot_scatter(Yte, Yp, metrics, src)
    render_table(metrics, src)
    save_csv(metrics)
    plt.show()


if __name__ == '__main__':
    main()