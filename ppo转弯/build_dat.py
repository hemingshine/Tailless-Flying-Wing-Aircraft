#coding=utf-8
"""
build_dat.py —— 用全部真实行做多元回归，拟合"基线(α多项式)+线性操纵/侧滑导数"模型
(多进程读取)。这是转向"物理解析气动模型"的第一步：

  C_j(α, 舵面, β) = poly3(α) + Σ_k deriv_jk·v_k + Cβ_j·β  (+ α×舵面交互)

报告：每系数线性模型 R²(线性够不够)、拟出的操纵导数 / 侧滑导数(喂给 fly.py 的 Numba 模型)。
转动阻尼 Cl_p/Cm_q/Cn_r 数据没有，后续在 fly.py 用 DATCOM 估算补。
"""
import os
import time
import pickle
import numpy as np
import pandas as pd
import multiprocessing as mp
from functools import partial

EXCEL_PATH = 'C1-X47B.xlsx'
OUT_PKL = 'X47B_coeffs.pkl'
COEF = ['Cx', 'Cy', 'Cz', 'Cl', 'Cm', 'Cn']

CTRL_COLS = ['左襟翼偏角（°）', '右襟翼偏角（°）', '左副翼偏角（°）', '右副翼偏角（°）',
             '前扰流板偏角（°）', '后扰流板偏角（°）', '马赫数']
ROW_COLS = ['迎角（°）', '侧滑角（°）']
OUT_COLS = ['轴向力系数', '横向力系数', '法向力系数', '滚转力矩系数', '俯仰力矩系数', '偏航力矩系数']

# 特征顺序(fly.py 的 Numba 模型必须严格照此重建)
FEATURES = ['1', 'a', 'a2', 'a3', 'sym_flap', 'diff_flap', 'sym_ail', 'diff_ail',
            'spoil_F', 'spoil_R', 'beta', 'a*sym_flap', 'a*diff_ail']


def _resolve(cols, target):
    t = target.strip()
    for c in cols:
        if str(c).strip() == t:
            return c
    key = t.replace('（°）', '').replace('(°)', '').strip()
    for c in cols:
        if key and key in str(c).strip():
            return c
    return None


def read_sheet(sheet, path):
    try:
        raw = pd.read_excel(path, header=None, sheet_name=sheet)
    except Exception as e:
        return ('ERR', sheet, repr(e))
    mask = raw.apply(lambda r: r.astype(str).str.contains('模型代号').any(), axis=1)
    if not mask.any():
        return ('SKIP', sheet, '无表头')
    hi = mask.idxmax()
    df = raw.iloc[hi + 1:].copy()
    df.columns = raw.iloc[hi].astype(str).str.strip().values
    cols = list(df.columns)
    cmap = [_resolve(cols, c) for c in CTRL_COLS]
    rmap = [_resolve(cols, c) for c in ROW_COLS]
    omap = [_resolve(cols, c) for c in OUT_COLS]
    if any(m is None for m in cmap + rmap + omap):
        return ('MISSING', sheet, [c for c, m in zip(CTRL_COLS + ROW_COLS + OUT_COLS,
                                                      cmap + rmap + omap) if m is None])
    df[cmap] = df[cmap].apply(pd.to_numeric, errors='coerce').ffill()
    df[rmap + omap] = df[rmap + omap].apply(pd.to_numeric, errors='coerce')
    df = df.dropna(subset=cmap + [rmap[0]] + omap)
    if len(df) == 0:
        return ('EMPTY', sheet, 0)
    return ('OK', sheet, np.column_stack([df[c].values for c in (cmap + rmap + omap)]).astype(np.float64))


def design_matrix(A):
    fL, fR, aL, aR, spF, spR = [A[:, i] for i in range(6)]
    a = A[:, 7]; beta = A[:, 8]
    sf = (fL + fR) / 2; df = (fL - fR) / 2
    sa = (aL + aR) / 2; da = (aL - aR) / 2
    one = np.ones_like(a)
    return np.column_stack([one, a, a**2, a**3, sf, df, sa, da, spF, spR, beta, a * sf, a * da])


if __name__ == '__main__':
    print("=" * 56)
    print("  多元回归拟合气动导数 (多进程读取)")
    print("=" * 56)
    if not os.path.exists(EXCEL_PATH):
        print(f"找不到 {EXCEL_PATH}"); raise SystemExit
    t0 = time.time()
    sheets = pd.ExcelFile(EXCEL_PATH).sheet_names
    with mp.Pool(min(len(sheets), max(1, mp.cpu_count() - 1))) as pool:
        results = pool.map(partial(read_sheet, path=EXCEL_PATH), sheets)
    blocks = []
    for r in results:
        if r[0] == 'OK':
            blocks.append(r[2]); print(f"  OK {r[1]}: {len(r[2])} 行")
        else:
            print(f"  - {r[1]}: {r[0]} {r[2] if len(r) > 2 else ''}")
    A = np.vstack(blocks)
    machs = np.unique(np.round(A[:, 6], 4))
    print(f"\n合并 {len(A)} 行, mach={machs} ({time.time()-t0:.1f}s)")

    model = {'features': FEATURES}
    for m in machs:
        sub = A[np.abs(A[:, 6] - m) < 1e-6]
        if len(sub) < 30:
            print(f"  M={m}: 仅{len(sub)}行,跳过"); continue
        Xf = design_matrix(sub); Y = sub[:, 9:15]
        B, *_ = np.linalg.lstsq(Xf, Y, rcond=None)   # (P,6)
        model[float(m)] = B
        pred = Xf @ B
        print(f"\n  ===== M={m} (n={len(sub)}) 线性模型 R² =====")
        for j, nm in enumerate(COEF):
            ss_res = np.sum((Y[:, j] - pred[:, j])**2)
            ss_tot = np.sum((Y[:, j] - Y[:, j].mean())**2) + 1e-12
            r2 = 1 - ss_res / ss_tot
            print(f"    {nm}: R²={r2:6.3f}", end="")
        print()
        fi = {f: i for i, f in enumerate(FEATURES)}
        print(f"    操纵导数: Cm_δflap={B[fi['sym_flap'],4]:+.5f}  "
              f"Cl_δail={B[fi['diff_ail'],3]:+.5f}  "
              f"Cn_spoilF={B[fi['spoil_F'],5]:+.5f}  Cn_spoilR={B[fi['spoil_R'],5]:+.5f}")
        print(f"    侧滑导数: Cn_β={B[fi['beta'],5]:+.5f}  Cl_β={B[fi['beta'],3]:+.5f}  "
              f"Cy_β={B[fi['beta'],1]:+.5f}   (Cn_β>0 才方向稳定)")

    with open(OUT_PKL, 'wb') as f:
        pickle.dump(model, f)
    print(f"\n保存 {OUT_PKL}")
    print("看各系数 R²：若力矩 Cl/Cm/Cn 的 R²>0.7 -> 线性解析模型可行,我据此写 fly.py 的 Numba 气动模型。")