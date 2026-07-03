#coding=utf-8
"""
ablationvis.py —— 消融实验可视化 (数据驱动, 与"怎么训出来"无关)

支持单 seed 与多 seed:
  results[key] = {'label', 't', 'ref', 'y'}            # 单次
  results[key] = {'label', 't', 'ref', 'ys':[y0,y1..]} # 多 seed(每个 y 是同一 t/ref 下一次实现)
多 seed 时: 指标取 seed 间 mean±std, 柱状图带误差棒, 表格/CSV 带 ±std。
第一个 key 必须是 full(基线)。
"""
import csv
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

# ============================ 配置(可被 runner 覆盖) ============================
SIGNAL_LABEL = 'Altitude'
SIGNAL_UNIT  = 'm'
FAULT_WINDOW = (40.0, 80.0)        # 评估窗(阴影 + 段内 RMSE)
TRANSIENT    = (40.0, 45.0)        # 瞬态窗(算最大瞬态偏差)
SAVE_PREFIX  = './ablation'
SHOW         = True                # 批量出多张图时 runner 置 False, 避免 plt.show 阻塞

FULL_COLOR = '#d62728'
ABL_COLORS = ['#1f77b4', '#2ca02c', '#ff7f0e', '#9467bd', '#8c564b']


# ============================ 指标 ============================
def _mask(t, a, b):
    return (t >= a) & (t <= b)

def compute_metrics(t, y, ref):
    t = np.asarray(t); y = np.asarray(y); ref = np.asarray(ref)
    err = y - ref
    rmse_all = float(np.sqrt(np.mean(err ** 2)))
    mf = _mask(t, *FAULT_WINDOW)
    rmse_fault = float(np.sqrt(np.mean(err[mf] ** 2))) if np.any(mf) else float('nan')
    mt = _mask(t, *TRANSIENT)
    max_trans = float(np.max(np.abs(err[mt]))) if np.any(mt) else float('nan')
    return {'rmse_all': rmse_all, 'rmse_fault': rmse_fault, 'max_trans': max_trans}


def _agg(t, ys, ref):
    """多 seed 聚合: 对齐长度, 逐 seed 算指标取 mean/std, 并给出曲线 mean/std。"""
    t = np.asarray(t); ref = np.asarray(ref)
    ys = [np.asarray(y) for y in ys]
    L = min([len(t), len(ref)] + [len(y) for y in ys])
    t = t[:L]; ref = ref[:L]; ys = [y[:L] for y in ys]
    Y = np.stack(ys, axis=0)                         # (S, L)
    per = [compute_metrics(t, y, ref) for y in ys]
    def ms(key):
        a = np.array([p[key] for p in per], dtype=float)
        return float(np.nanmean(a)), float(np.nanstd(a))
    Eabs = np.abs(Y - ref)                           # (S, L)
    return {'t': t, 'ref': ref,
            'y_mean': Y.mean(0), 'y_std': Y.std(0),
            'e_mean': Eabs.mean(0), 'e_std': Eabs.std(0),
            'n_seed': len(ys),
            'rmse_all': ms('rmse_all'), 'rmse_fault': ms('rmse_fault'),
            'max_trans': ms('max_trans')}


# ============================ 论文风格 ============================
def paper_style():
    plt.rcParams.update({
        'font.family': 'serif', 'mathtext.fontset': 'dejavuserif',
        'font.size': 10, 'axes.titlesize': 10.5, 'axes.labelsize': 10,
        'legend.fontsize': 8.5, 'xtick.labelsize': 9, 'ytick.labelsize': 9,
        'axes.linewidth': 0.8, 'lines.linewidth': 1.5,
        'axes.grid': True, 'grid.alpha': 0.3, 'grid.linewidth': 0.5,
    })


# ============================ 可视化 ============================
def plot_ablation(results):
    paper_style()
    keys = list(results.keys())
    full_key = keys[0]
    cmap = {full_key: FULL_COLOR}
    ai = 0
    for k in keys[1:]:
        cmap[k] = ABL_COLORS[ai % len(ABL_COLORS)]; ai += 1

    # 聚合(单 seed 也走同一路径: ys=[y])
    agg = {k: _agg(v['t'], (v['ys'] if 'ys' in v else [v['y']]), v['ref']) for k, v in results.items()}
    full_rf_m, full_rf_s = agg[full_key]['rmse_fault']
    n_seed = agg[full_key]['n_seed']

    # ---------- 图1: 跟踪叠加 + 误差叠加 + 组件贡献柱状 ----------
    fig = plt.figure(figsize=(13, 6.2))
    gs = GridSpec(2, 2, figure=fig,wspace=0.18)

    ax_trk = fig.add_subplot(gs[0, 0])
    ax_trk.plot(agg[full_key]['t'], agg[full_key]['ref'], 'k--', lw=1.2, label='Reference')
    for k in keys:
        a = agg[k]; lw = 2.0 if k == full_key else 1.3
        ax_trk.plot(a['t'], a['y_mean'], color=cmap[k], lw=lw, alpha=0.9, label=results[k]['label'])
        if n_seed > 1:
            ax_trk.fill_between(a['t'], a['y_mean'] - a['y_std'], a['y_mean'] + a['y_std'],
                                color=cmap[k], alpha=0.12, linewidth=0)
    ax_trk.axvspan(*FAULT_WINDOW, color='red', alpha=0.08)
    ax_trk.set_xlabel('Time (s)'); ax_trk.set_ylabel(f'{SIGNAL_LABEL} ({SIGNAL_UNIT})')
    ax_trk.set_title('(a) Tracking response')
    ax_trk.legend(loc='best', framealpha=0.9, fontsize=7.5)

    ax_err = fig.add_subplot(gs[0, 1])
    for k in keys:
        a = agg[k]; lw = 2.0 if k == full_key else 1.3
        ax_err.plot(a['t'], a['e_mean'], color=cmap[k], lw=lw, alpha=0.9, label=results[k]['label'])
        if n_seed > 1:
            ax_err.fill_between(a['t'], np.clip(a['e_mean'] - a['e_std'], 0, None), a['e_mean'] + a['e_std'],
                                color=cmap[k], alpha=0.12, linewidth=0)
    ax_err.axvspan(*FAULT_WINDOW, color='red', alpha=0.08)
    ax_err.set_xlabel('Time (s)'); ax_err.set_ylabel(f'|{SIGNAL_LABEL} error| ({SIGNAL_UNIT})')
    ax_err.set_title('(b) Tracking error')
    ax_err.legend(loc='upper left', framealpha=0.9, fontsize=7.5)

    # 组件贡献柱状: 段内 RMSE mean ± std(误差棒)
    # ax_bar = fig.add_subplot(gs[1, :])
    # labels = [results[k]['label'] for k in keys]
    # means = [agg[k]['rmse_fault'][0] for k in keys]
    # stds = [agg[k]['rmse_fault'][1] for k in keys]
    # colors = [cmap[k] for k in keys]
    # ypos = np.arange(len(keys))[::-1]
    # ax_bar.barh(ypos, means, color=colors, alpha=0.85, edgecolor='0.3', height=0.62,
    #             xerr=stds if n_seed > 1 else None,
    #             error_kw=dict(ecolor='0.25', lw=1.0, capsize=3))
    # for yp, k in zip(ypos, keys):
    #     m, sdv = agg[k]['rmse_fault']
    #     pct = 'baseline' if k == full_key else (f'+{(m-full_rf_m)/full_rf_m*100:.0f}%' if full_rf_m > 0 else '')
    #     txt = f'{m:.2f}' + (f'±{sdv:.2f}' if n_seed > 1 else '') + (f' ({pct})' if pct else '')
    #     ax_bar.text((m + sdv if n_seed > 1 else m), yp, '  ' + txt, va='center', ha='left', fontsize=8.5)
    # ax_bar.set_yticks(ypos); ax_bar.set_yticklabels(labels, fontsize=9)
    # seedtag = f'  (mean$\\pm$std over {n_seed} seeds)' if n_seed > 1 else ''
    # ax_bar.set_xlabel(f'Within-window {SIGNAL_LABEL} RMSE ({SIGNAL_UNIT})  — lower is better{seedtag}')
    # ax_bar.set_title('(c) Component contribution (degradation when removed)')
    # ax_bar.set_xlim(0, (max(np.array(means) + np.array(stds))) * 1.32)
    # ax_bar.grid(axis='y', alpha=0)

    fig.savefig(f'{SAVE_PREFIX}_curves.png', dpi=300, bbox_inches='tight')
    fig.savefig(f'{SAVE_PREFIX}_curves.pdf', bbox_inches='tight')
    print(f'图已存: {SAVE_PREFIX}_curves.png / .pdf')

    # ---------- 图2: 指标表 ----------
    def fmt(ms_tuple):
        m, s = ms_tuple
        return f'{m:.3f}' + (f'±{s:.3f}' if n_seed > 1 else '')
    fig2, ax = plt.subplots(figsize=(8.4, 0.5 + 0.42 * len(keys)))
    ax.axis('off')
    col = ['Variant', f'RMSE all ({SIGNAL_UNIT})', f'RMSE win. ({SIGNAL_UNIT})',
           f'Max trans. ({SIGNAL_UNIT})', 'ΔRMSE_win.']
    cell = []
    for k in keys:
        a = agg[k]
        m = a['rmse_fault'][0]
        d = 'baseline' if k == full_key else (f'+{(m-full_rf_m)/full_rf_m*100:.1f}%' if full_rf_m > 0 else '—')
        cell.append([results[k]['label'], fmt(a['rmse_all']), fmt(a['rmse_fault']),
                     fmt(a['max_trans']), d])
    tb = ax.table(cellText=cell, colLabels=col, loc='center', cellLoc='center')
    tb.auto_set_font_size(False); tb.set_fontsize(9); tb.scale(1, 1.5)
    for (r, c), cobj in tb.get_celld().items():
        cobj.set_edgecolor('0.7')
        if r == 0:
            cobj.set_facecolor('#2f4b7c'); cobj.set_text_props(color='white', fontweight='bold')
        elif r == 1:
            cobj.set_facecolor('#e8edf5'); cobj.set_text_props(fontweight='bold')
        elif r % 2 == 0:
            cobj.set_facecolor('#f4f6fa')
    fig2.savefig(f'{SAVE_PREFIX}_table.png', dpi=300, bbox_inches='tight')
    fig2.savefig(f'{SAVE_PREFIX}_table.pdf', bbox_inches='tight')
    print(f'表已存: {SAVE_PREFIX}_table.png / .pdf')

    # CSV + 控制台
    with open(f'{SAVE_PREFIX}_metrics.csv', 'w', newline='', encoding='utf-8-sig') as f:
        w = csv.writer(f)
        w.writerow(['Variant', 'n_seed', 'RMSE_all_mean', 'RMSE_all_std',
                    'RMSE_win_mean', 'RMSE_win_std', 'MaxTrans_mean', 'MaxTrans_std', 'dRMSE_win_pct'])
        for k in keys:
            a = agg[k]; m = a['rmse_fault'][0]
            dd = 0.0 if k == full_key else ((m - full_rf_m) / full_rf_m * 100 if full_rf_m > 0 else float('nan'))
            w.writerow([results[k]['label'], a['n_seed'],
                        f'{a["rmse_all"][0]:.4f}', f'{a["rmse_all"][1]:.4f}',
                        f'{a["rmse_fault"][0]:.4f}', f'{a["rmse_fault"][1]:.4f}',
                        f'{a["max_trans"][0]:.4f}', f'{a["max_trans"][1]:.4f}', f'{dd:.2f}'])
    print(f'指标已存: {SAVE_PREFIX}_metrics.csv')
    print(f'\n[{SIGNAL_LABEL}] 变体              RMSE_all     RMSE_win     ΔRMSE_win  (n_seed={n_seed})')
    for k in keys:
        a = agg[k]; m = a['rmse_fault'][0]
        dd = 'baseline' if k == full_key else (f'+{(m-full_rf_m)/full_rf_m*100:.1f}%' if full_rf_m > 0 else '—')
        print(f"  {results[k]['label']:<22}{fmt(a['rmse_all']):>14}{fmt(a['rmse_fault']):>14}   {dd}")

    if SHOW:
        plt.show()
    else:
        plt.close(fig); plt.close(fig2)


# ============================ 演示(合成多 seed 数据) ============================
def _demo_results():
    t = np.arange(0, 200, 0.2)
    ref = 3000.0 + 30.0 * np.sin(2 * np.pi * (5 / 200.0) * t)

    def mk(amp_base, amp_fault, seed):
        rng = np.random.default_rng(seed)
        e = amp_base * np.sin(0.7 * t + 1.0) + amp_base * 0.3 * rng.standard_normal(len(t))
        mf = (t >= FAULT_WINDOW[0]) & (t <= FAULT_WINDOW[1])
        e[mf] += amp_fault * np.sin(1.2 * t[mf]) + amp_fault * 0.5
        return ref + e

    def runs(ab, af):
        return [mk(ab, af, s) for s in range(5)]

    return {
        'full':          {'label': 'Full (ours)',            't': t, 'ref': ref, 'ys': runs(0.6, 0.8)},
        'no_temporal':   {'label': 'w/o Temporal feat.',     't': t, 'ref': ref, 'ys': runs(0.9, 2.4)},
        'no_curriculum': {'label': 'w/o 3-stage curriculum', 't': t, 'ref': ref, 'ys': runs(1.1, 3.6)},
        'no_tecs':       {'label': 'w/o energy mgmt.',       't': t, 'ref': ref, 'ys': runs(0.8, 3.0)},
        'no_preview':    {'label': 'w/o Look-ahead',         't': t, 'ref': ref, 'ys': runs(1.4, 2.0)},
    }


if __name__ == '__main__':
    plot_ablation(_demo_results())