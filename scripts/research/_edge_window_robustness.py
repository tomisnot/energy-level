# -*- coding: utf-8 -*-
"""方法稳健性：锁相提取的 (A0, C, f0, φ0) 对分析窗口有多敏感？

如果 A0 随窗口漂移，那"沿宽削掉 10% 幅度"可能只是窗口伪像。
这里对前人 12 条曲线，用 4 个不同窗口各算一遍，比较 A0 的离散度。
"""
import io
import json
import os
import sys

import numpy as np

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8',
                              errors='replace')
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
TMP = os.path.join(ROOT, 'scripts', 'tmp')
TAGS = ['F/tf=0', 'F/tf=10', 'F/tf=20', 'F/tf=35', 'F/tf=50', 'F/tf=70',
        'F/tf=100', 'E1/hold0@0.6', 'E1/hold25@0.6', 'E1/hold50@0.6',
        'E1/hold100@0.6', 'E3/stair50']
WINDOWS = [(340.0, 2000.0), (340.0, 1200.0), (1000.0, 2000.0), (600.0, 1800.0),
           (340.0, 800.0)]


def lockin(t, y, lo, hi):
    m = (t >= lo) & (t <= hi)
    tt, yy = t[m] - lo, y[m]
    n = tt.size
    w = np.hanning(n)
    sp = np.abs(np.fft.rfft((yy - yy.mean()) * w))
    fr = np.fft.rfftfreq(n, d=float(np.median(np.diff(tt))))
    k = int(np.argmax(sp[1:]) + 1)
    if 0 < k < sp.size - 1:
        a, b, c = sp[k - 1], sp[k], sp[k + 1]
        den = a - 2 * b + c
        dk = 0.5 * (a - c) / den if den else 0.0
        f0 = float(fr[k] + dk * (fr[1] - fr[0]))
    else:
        f0 = float(fr[k])
    G = np.column_stack([np.ones(n), np.cos(2 * np.pi * f0 * tt),
                         np.sin(2 * np.pi * f0 * tt)])
    coef, *_ = np.linalg.lstsq(G, yy, rcond=None)
    return (float(coef[0]), float(np.hypot(coef[1], coef[2])), f0,
            float(np.arctan2(-coef[2], coef[1])))


def main():
    print('  %-16s %s' % ('tag', '  '.join('A0[%g-%g]' % w for w in WINDOWS)))
    print('  ' + '-' * 84)
    A0rows, allvals = [], []
    for tag in TAGS:
        p = os.path.join(TMP, '_fall_mech_%s.npz' % tag.replace('/', '_'))
        if not os.path.exists(p):
            continue
        z = np.load(p)
        t, a = np.asarray(z['t'], float), np.asarray(z['aft'], float)
        vals = [lockin(t, a, lo, hi)[1] for lo, hi in WINDOWS]
        A0rows.append((tag, vals))
        allvals.append(vals)
        print('  %-16s %s' % (tag, '  '.join('%9.4f' % v for v in vals)))

    V = np.array(allvals)
    print('\n  === A0 对窗口的敏感度 ===')
    sp = V.max(axis=1) - V.min(axis=1)
    for (tag, _), s in zip(A0rows, sp):
        print('    %-16s 跨窗口极差 %.4f（相对 %.1f%%）'
              % (tag, s, 100 * s / np.mean(V[A0rows.index((tag, _))])))
    print('    全体最大极差 %.4f' % sp.max())

    # 关键：沿宽趋势在各窗口下是否一致
    print('\n  === 沿宽趋势在各窗口下的一致性（F 族 A0 随 τ_f 的斜率）===')
    fam = [('F/tf=0', 0), ('F/tf=10', 10), ('F/tf=20', 20), ('F/tf=35', 35),
           ('F/tf=50', 50), ('F/tf=70', 70), ('F/tf=100', 100)]
    idx = {tg: i for i, (tg, _) in enumerate(A0rows)}
    xs = np.array([tf for _, tf in fam], float)
    for j, (lo, hi) in enumerate(WINDOWS):
        ys = np.array([V[idx[tg], j] for tg, _ in fam if tg in idx])
        if ys.size == xs.size:
            k = np.polyfit(xs, ys, 1)[0]
            print('    窗口 %5.0f–%4.0f : 斜率 %+.3e /ns  ⇒ 100 ns 掉 %.1f%%'
                  % (lo, hi, k, 100 * abs(k) * 100 / ys[0]))

    out = os.path.join(TMP, '_edge_window_robustness.json')
    json.dump({'windows': WINDOWS,
               'A0': {tg: v for tg, v in A0rows}},
              io.open(out, 'w', encoding='utf-8'), indent=1)
    print('\n  数据 → %s' % out)


if __name__ == '__main__':
    main()