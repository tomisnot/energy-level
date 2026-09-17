# -*- coding: utf-8 -*-
"""图8（定稿）：正常下降沿密扫 25–125 ns，每 5 ns。

  (a) 换后态的振荡曲线：随 τ_f 变宽，曲线**整体右移（相位落后）**且**峰被压低**
      —— 直观展示下降沿确实明显影响交换过程
  (b) 把同一批点按**提取出的相位** Δφ 重排：交换质量随相位下降
      —— 并标出它在 |Δφ|≈1.0 rad 处触底后回升（诚实呈现，非单调）
另出图9：(时间, τ_f) 热图，看条纹倾斜与亮带下降。
"""
import io
import json
import os
import sys

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8',
                              errors='replace')
plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['font.size'] = 11
plt.rcParams['axes.linewidth'] = 1.0
plt.rcParams['savefig.bbox'] = 'tight'

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
TMP = os.path.join(ROOT, 'scripts', 'tmp')
FIGDIR = os.path.join(ROOT, 'ui_design', 'screenshots', 'edge_study')
RES = os.path.join(TMP, '_edge_dense_res.json')


def load_curves():
    """从各 sweep 的 npz 重取振荡曲线（指纹匹配）。"""
    sys.path.insert(0, os.path.join(ROOT, 'scripts', 'research'))
    from _edge_dense_collect import collect
    return collect()


def main():
    R = json.load(io.open(RES, encoding='utf-8'))
    rows = sorted(R['rows'], key=lambda r: r['tau'])
    f0 = R['f0']
    D = load_curves()
    x = np.abs(np.array([r['dphi'] for r in rows]))
    A0 = np.array([r['A0'] for r in rows])
    CA = np.array([r['A0'] + r['C'] for r in rows])
    tau = np.array([r['tau'] for r in rows], float)

    fig = plt.figure(figsize=(16.0, 5.6))
    gs = fig.add_gridspec(2, 3, width_ratios=[1.25, 1, 1], hspace=0.95,
                          wspace=0.30)
    axA = fig.add_subplot(gs[:, 0])
    axB = fig.add_subplot(gs[0, 1:])
    axC = fig.add_subplot(gs[1, 1:])
    axes = [axA, axB, axC]

    # ---------------- (a) 振荡曲线 ----------------
    ax = axes[0]
    pick = [25, 50, 75, 100, 125]
    cols = plt.get_cmap('viridis')(np.linspace(0.05, 0.9, len(pick)))
    for c, tf in zip(cols, pick):
        if tf not in D:
            continue
        d = D[tf]
        m = (d['t'] >= 300) & (d['t'] <= 1150)
        r = next(rr for rr in rows if rr['tau'] == tf)
        ax.plot(d['t'][m], d['a'][m], '-', color=c, lw=2.0,
                label=r'$\tau_f$=%3d ns   $A_0$=%.3f   $|\Delta\phi|$=%.2f'
                      % (tf, r['A0'], abs(r['dphi'])))
        mm = (d['t'] >= 400) & (d['t'] <= 800)
        k = int(np.argmax(d['a'][mm]))
        ax.plot(d['t'][mm][k], d['a'][mm][k], 'v', color=c, ms=7)
    ax.set_xlabel('时间 (ns)')
    ax.set_ylabel(r'换后态布居  $P_{(60S_{-1/2})(60P_{+3/2})}$')
    ax.set_title('(a) 下降沿越宽：振荡峰右移且被压低', fontsize=12)
    ax.legend(fontsize=8.6, loc='lower right')
    ax.grid(alpha=0.22)
    ax.annotate('峰位右移 →', xy=(0.66, 0.955), xycoords='axes fraction',
                fontsize=10, color='0.35', ha='center')

    # ---------------- (b) 可达最大值 ----------------
    xg = np.linspace(0, x.max() * 1.03, 200)
    ax = axes[1]
    o = np.argsort(x)
    xs, a0s, cas = x[o], A0[o], CA[o]
    ax.plot(xs, cas, 's-', color='#1f77b4', lw=2, ms=5.5,
            label=r'可达最大值 $C+A_0$')
    try:
        from scipy.interpolate import UnivariateSpline
        spc = UnivariateSpline(xs, cas, k=3, s=len(xs) * 2.0e-6)
        ax.plot(xg, spc(xg), ':', color='#1f77b4', lw=1.8,
                label='平滑样条（$R^2$=%.2f）'
                      % (1 - ((cas - spc(xs)) ** 2).sum()
                         / ((cas - cas.mean()) ** 2).sum()))
    except Exception:                       # noqa: BLE001
        pass
    ax.set_xlabel(r'沿造成的相位差  $|\Delta\phi|$  (rad)')
    ax.set_ylabel(r'$C+A_0$')
    ax.set_title('(b) 相位越大，可达最大值越低（整体降 %.1f%%）'
                 % (100 * (CA[0] - CA[-1]) / CA[0]), fontsize=11.5)
    ax.legend(fontsize=8.8, loc='lower right'); ax.grid(alpha=0.22)

    # ---------------- (c) 交换振幅 ----------------
    ax = axes[2]
    ax.plot(xs, a0s, 'o-', color='#d62728', lw=2, ms=6.5,
            label=r'交换振幅 $A_0$')
    try:
        from scipy.interpolate import UnivariateSpline
        sp = UnivariateSpline(xs, a0s, k=3, s=len(xs) * 2.0e-6)
        yg = sp(xg)
        imin = int(np.argmin(yg))
        phi_min, y_min = float(xg[imin]), float(yg[imin])
        res = a0s - sp(xs)
        r2 = 1 - (res ** 2).sum() / ((a0s - a0s.mean()) ** 2).sum()
        ax.plot(xg, yg, ':', color='#d62728', lw=1.8,
                label='平滑样条（谷底 |Δφ|=%.2f rad，$R^2$=%.2f）' % (phi_min, r2))
        ax.annotate('谷底：再加大相位反而回升', xy=(phi_min, y_min),
                    xytext=(0.06, 0.62), textcoords='axes fraction', fontsize=9,
                    color='0.3',
                    arrowprops=dict(arrowstyle='->', color='0.5', lw=1.1))
    except Exception:                       # noqa: BLE001
        phi_min, r2 = float('nan'), float('nan')
    ax.set_xlabel(r'沿造成的相位差  $|\Delta\phi|$  (rad)')
    ax.set_ylabel(r'交换振幅 $A_0$')
    ax.set_title('(c) 交换振幅随相位下降 %.1f%%（对应 τ_f 25→125 ns）'
                 % (100 * (A0[0] - A0[-1]) / A0[0]), fontsize=11.5)
    ax.legend(fontsize=8.8, loc='lower right'); ax.grid(alpha=0.22)
    ax2 = ax.twiny()
    ax2.set_xlim(ax.get_xlim())
    tick_tau = [25, 50, 75, 100, 125]
    tick_phi = [abs(next(r for r in rows if r['tau'] == t)['dphi']) for t in tick_tau]
    ax2.set_xticks(tick_phi)
    ax2.set_xticklabels([r'$\tau_f$=%d' % t for t in tick_tau], fontsize=8.5)
    ax2.tick_params(length=0)

    fig.suptitle('图8  正常下降沿密扫（25–125 ns，每 5 ns，共 21 点）：'
                 '相位是交换变差的直接标度', fontsize=14, y=1.0)
    for ext in ('png', 'pdf'):
        fig.savefig(os.path.join(FIGDIR, 'fig8_fall_dense.%s' % ext), dpi=300)
    print('  图 → fig8_fall_dense.png/.pdf')

    # ---------------- 图9：热图 ----------------
    fig2, ax = plt.subplots(figsize=(9.8, 5.4))
    grid = np.vstack([D[t]['a'] for t in sorted(D)])
    tt = D[sorted(D)[0]]['t']
    m = (tt >= 300) & (tt <= 1500)
    im = ax.pcolormesh(tt[m], np.array(sorted(D)), grid[:, m], cmap='inferno',
                       shading='auto')
    plt.colorbar(im, ax=ax, label='换后态布居')
    ax.set_xlabel('时间 (ns)'); ax.set_ylabel(r'下降沿宽度 $\tau_f$ (ns)')
    ax.set_title('图9  全貌：换后态布居的 (时间, $\\tau_f$) 演化\n'
                 '亮条纹随 $\\tau_f$ 倾斜 = 相位被沿改写；亮带变暗 = 振幅被削',
                 fontsize=12)
    for ext in ('png', 'pdf'):
        fig2.savefig(os.path.join(FIGDIR, 'fig9_dense_heatmap.%s' % ext), dpi=300)
    print('  图 → fig9_dense_heatmap.png/.pdf')

    # ---------------- 打印定量 ----------------
    print('\n  === 定量 ===')
    print('     corr(τ_f, |Δφ|) = %+.3f  （相位随沿宽线性增长）'
          % np.corrcoef(tau, x)[0, 1])
    print('     corr(|Δφ|, A0)  = %+.3f' % np.corrcoef(x, A0)[0, 1])
    print('     平滑样条谷底 |Δφ| = %.2f rad（$R^2$=%.3f）' % (phi_min, r2))
    print('     |Δφ| 范围 0 → %.2f rad；A0 从 %.4f 降到 %.4f（−%.1f%%）'
          % (x.max(), A0[0], A0[-1], 100 * (A0[0] - A0[-1]) / A0[0]))
    print('     A0 最低 %.4f 出现在 τ_f = %d ns（|Δφ|=%.2f）'
          % (A0.min(), int(tau[int(np.argmin(A0))]), x[int(np.argmin(A0))]))


if __name__ == '__main__':
    main()