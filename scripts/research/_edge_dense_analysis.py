# -*- coding: utf-8 -*-
"""下降沿密扫（正常下降沿，25–125 ns，每 5 ns）的分析与出图。

产出图（fig8_fall_dense，两个子图，按使用者要求）：
  (a) 某一态（换后态）的振荡曲线 + Hilbert 上包络
      —— 直观展示下降沿越长、振荡被压得越低
  (b) 交换振幅 A0 与"可达最大值"对**提取出的相位 Δφ** 的关系
      —— 直观展示"相位越大，交换越差"

相位定义：对每点做锁相投影 a(t) = C + A0·cos(2πf t + φ0)（f 为共同主频），
以最短沿（τ_f=25 ns）为相位零点，给出 Δφ。
"""
import glob
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
os.makedirs(FIGDIR, exist_ok=True)
K_AFT = '(60S_{1/2} m_J=-0.5) x (60P_{3/2} m_J=+1.5)'
K_BEF = '(60P_{3/2} m_J=-0.5) x (60S_{1/2} m_J=+0.5)'
IRES, T0, TMAX = 0.42, 280.0, 2000.0
W = (340.0, 2000.0)


def normal(tau):
    n = max(8, int(round(tau)))
    br = [(0.0, 0.0), (119.999, 0.0), (120.0, 1.0), (T0, 1.0)]
    for k in range(1, n + 1):
        br.append((T0 + tau * k / n, 1.0 - (1.0 - IRES) * k / n))
    br.append((TMAX, IRES))
    out, last = [], None
    for t, v in br:
        if last is not None and t <= last:
            continue
        out.append((t, v)); last = t
    return out


def env_of(tau):
    br = normal(tau)
    t = np.array([b[0] for b in br], float)
    return t, np.sqrt(np.array([b[1] for b in br], float))


def collect():
    taus = json.load(io.open(os.path.join(TMP, '_edge_dense.json'),
                             encoding='utf-8'))['taus']
    ref = {tau: env_of(float(tau)) for tau in taus}
    out = {}
    for sd in sorted(glob.glob(os.path.join(ROOT, 'runtime', 'sweeps', 'sw*')),
                     key=os.path.getmtime):
        sp = os.path.join(sd, 'spec.json')
        if not os.path.exists(sp):
            continue
        spec = json.load(io.open(sp, encoding='utf-8'))
        for i, p in enumerate(spec.get('points') or []):
            wa = p.get('dynamic.waves_A')
            if not wa or len(wa) != 3 or wa[2].get('env', [None])[0] != 'file':
                continue
            tt = np.array(wa[2]['env'][1], float)
            aa = np.array(wa[2]['env'][2], float)
            for tau, (rt, ra) in ref.items():
                if rt.size != tt.size or not np.allclose(rt, tt, atol=1e-6) \
                        or not np.allclose(ra, aa, atol=1e-9):
                    continue
                if tau in out:
                    continue
                f = os.path.join(sd, 'pt_%04d.npz' % i)
                if not os.path.exists(f):
                    continue
                z = np.load(f, allow_pickle=False)
                if 'series_t' not in z.files:
                    continue
                labels = [str(x) for x in z['labels']]
                if K_AFT not in labels:
                    continue
                t = np.asarray(z['series_t'], float)
                P = np.asarray(z['series_pop'], float)
                out[tau] = dict(t=t, a=P[:, labels.index(K_AFT)],
                                b=P[:, labels.index(K_BEF)], sid=os.path.basename(sd))
    return out


def lockin(t, y, f0=None, lo=340.0, hi=2000.0):
    m = (t >= lo) & (t <= hi)
    tt, yy = t[m] - lo, y[m]
    n = tt.size
    w = np.hanning(n)
    if f0 is None:
        sp = np.abs(np.fft.rfft((yy - yy.mean()) * w))
        fr = np.fft.rfftfreq(n, d=float(np.median(np.diff(tt))))
        k = int(np.argmax(sp[1:]) + 1)
        f0 = float(fr[k])
        if 0 < k < sp.size - 1:
            a_, b_, c_ = sp[k - 1], sp[k], sp[k + 1]
            den = a_ - 2 * b_ + c_
            f0 += (0.5 * (a_ - c_) / den if den else 0.0) * (fr[1] - fr[0])
    G = np.column_stack([np.ones(n), np.cos(2 * np.pi * f0 * tt),
                         np.sin(2 * np.pi * f0 * tt)])
    coef, *_ = np.linalg.lstsq(G, yy, rcond=None)
    return dict(C=float(coef[0]), A0=float(np.hypot(coef[1], coef[2])),
                phi=float(np.arctan2(-coef[2], coef[1])), f0=float(f0),
                rms=float(np.sqrt(np.mean((G @ coef - yy) ** 2))))


def main():
    D = collect()
    taus = sorted(D)
    if not taus:
        print('  ⚠ 没有数据（扫描可能还没跑完）')
        return
    print('  点数 %d：τ_f = %s' % (len(taus), taus))

    # 共同主频（用最长窗口的各点中位）
    f0 = float(np.median([lockin(D[t]['t'], D[t]['a'])['f0'] for t in taus]))
    rows = []
    for t in taus:
        d = D[t]
        L = lockin(d['t'], d['a'], f0=f0, lo=W[0], hi=W[1])
        k = int(np.argmax(d['a']))
        rows.append(dict(tau=t, **L, pk=float(d['a'][k]),
                         s=float(d['a'][k] + d['b'][k]),
                         frac=float(d['a'][k] / (d['a'][k] + d['b'][k]))))
    ref = rows[0]
    for r in rows:
        r['dphi'] = float((r['phi'] - ref['phi'] + np.pi) % (2 * np.pi) - np.pi)
    # 以首个相位（最短沿）为零点，故 dphi<=0 表示"落后"
    print('\n  %-8s %8s %8s %9s %9s %9s' %
          ('τ_f', '观测峰', 'A0', 'C+A0', 'Δφ(rad)', 'a/s'))
    print('  ' + '-' * 60)
    for r in sorted(rows, key=lambda x: x['tau']):
        print('  %-8d %8.4f %8.4f %9.4f %9.3f %9.4f'
              % (r['tau'], r['pk'], r['A0'], r['A0'] + r['C'], r['dphi'], r['frac']))

    x = np.abs(np.array([r['dphi'] for r in rows]))
    A0 = np.array([r['A0'] for r in rows])
    ca = np.array([r['A0'] + r['C'] for r in rows])
    tau = np.array([r['tau'] for r in rows], float)
    print('\n  === 定量关系 ===')
    print('     corr(|Δφ|, A0)    = %+.3f' % np.corrcoef(x, A0)[0, 1])
    print('     corr(|Δφ|, C+A0)  = %+.3f' % np.corrcoef(x, ca)[0, 1])
    print('     corr(τ_f,  |Δφ|)  = %+.3f' % np.corrcoef(tau, x)[0, 1])
    print('     |Δφ| 范围 %.3f → %.3f rad（%.2f 个周期）'
          % (x.min(), x.max(), (x.max() - x.min()) / (2 * np.pi)))
    print('     A0 从 %.4f 降到 %.4f（相对 %.1f%%）'
          % (A0[0], A0[-1], 100 * (A0[0] - A0[-1]) / A0[0]))

    # ================= 出图：两个子图 =================
    fig, axes = plt.subplots(1, 2, figsize=(14.6, 5.2))

    # ---- (a) 振荡曲线 ----
    ax = axes[0]
    pick = [25, 50, 75, 100, 125]
    pick = [p for p in pick if p in D]
    cmap = plt.get_cmap('viridis')
    for i, tau_ in enumerate(pick):
        d = D[tau_]
        col = cmap(i / max(1, len(pick) - 1))
        m = d['t'] >= 280.0
        ax.plot(d['t'][m], d['a'][m], '-', color=col, lw=1.5, alpha=0.9,
                label=r'$\tau_f$ = %d ns  ($A_0$=%.3f)'
                      % (tau_, next(r['A0'] for r in rows if r['tau'] == tau_)))
        # 上包络（解析信号）
        try:
            from scipy.signal import hilbert
            z = hilbert(d['a'][m] - d['a'][m].mean())
            ax.plot(d['t'][m], d['a'][m].mean() + np.abs(z), ':', color=col,
                    lw=1.0, alpha=0.75)
        except ImportError:
            pass
    ax.set_xlabel('时间 (ns)'); ax.set_ylabel(r'换后态布居  $P_{(60S_{-0.5})(60P_{+1.5})}$')
    ax.set_xlim(280, 1500)
    ax.set_title('(a) 正常下降沿越宽，换后态的振荡被压得越低', fontsize=12)
    ax.legend(fontsize=9, loc='upper right', ncol=1)
    ax.grid(alpha=0.22)

    # ---- (b) 相位 vs 交换质量 ----
    ax = axes[1]
    o = np.argsort(x)
    ax.plot(x[o], A0[o], 'o-', color='#d62728', lw=2, ms=6,
            label=r'交换振幅 $A_0$')
    ax.plot(x[o], ca[o], 's--', color='#1f77b4', lw=1.8, ms=5,
            label=r'可达最大值 $C+A_0$')
    k1 = np.polyfit(x, A0, 1)
    xs = np.linspace(0, x.max() * 1.05, 50)
    ax.plot(xs, np.polyval(k1, xs), ':', color='#d62728', lw=1.6,
            label=r'线性拟合：$A_0$ 每 rad 掉 %.3f（corr=%.2f）'
                  % (-k1[0], np.corrcoef(x, A0)[0, 1]))
    ax.set_xlabel(r'沿造成的相位差  $|\Delta\phi|$  (rad)')
    ax.set_ylabel('交换质量（幅度）')
    ax.set_title('(b) 相位越大，交换越差（21 点密扫）', fontsize=12)
    ax.legend(fontsize=9, loc='upper right')
    ax.grid(alpha=0.22)
    # 标注几个点
    for r in rows:
        if r['tau'] in (25, 75, 125):
            ax.annotate(r'$\tau_f$=%d' % r['tau'], (abs(r['dphi']), r['A0']),
                        xytext=(5, 6), textcoords='offset points', fontsize=8.5,
                        color='0.3')
    fig.suptitle('图8  正常下降沿密扫（25–125 ns，每 5 ns）：相位是交换变差的直接标度',
                 fontsize=13, y=1.02)
    for ext in ('png', 'pdf'):
        fig.savefig(os.path.join(FIGDIR, 'fig8_fall_dense.%s' % ext), dpi=300)
    print('\n  图 → fig8_fall_dense.png / .pdf')

    # ---- 附：时间-τ_f 热图（全貌） ----
    fig2, ax2 = plt.subplots(figsize=(9.5, 5.2))
    grid = np.vstack([D[t]['a'] for t in taus])
    tt = D[taus[0]]['t']
    m = (tt >= 280) & (tt <= 1500)
    im = ax2.pcolormesh(tt[m], np.array(taus), grid[:, m], cmap='inferno',
                        shading='auto')
    plt.colorbar(im, ax=ax2, label='换后态布居')
    ax2.set_xlabel('时间 (ns)'); ax2.set_ylabel(r'下降沿宽度 $\tau_f$ (ns)')
    ax2.set_title('图9  全貌：换后态布居随 (时间, τ_f) 的演化\n'
                  '条纹随 τ_f 倾斜 = 相位被沿改写；亮带高度下降 = 振幅被削',
                  fontsize=12)
    for ext in ('png', 'pdf'):
        fig2.savefig(os.path.join(FIGDIR, 'fig9_dense_heatmap.%s' % ext), dpi=300)
    print('  图 → fig9_dense_heatmap.png / .pdf')

    json.dump({'rows': rows, 'f0': f0},
              io.open(os.path.join(TMP, '_edge_dense_res.json'), 'w',
                      encoding='utf-8'), ensure_ascii=False, indent=1)
    print('  数据 → scripts/tmp/_edge_dense_res.json')


if __name__ == '__main__':
    main()