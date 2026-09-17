# -*- coding: utf-8 -*-
"""对判决扫描的 9 条轨迹做窗口稳健性检查。

存在的风险：不同沿宽改变了振荡相位，"在固定窗口上做锁相"可能把**相位差**
读成**幅度差**。本脚本用 5 个窗口（含只取第一个交换周期的短窗）各算一遍 A0，
只有**跨窗口一致**的差异才能当物理结论。
"""
import glob
import io
import json
import os
import sys

import numpy as np

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8',
                              errors='replace')
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(ROOT, 'scripts', 'research'))
from _edge_profiles import profiles, one          # noqa: E402

K_AFT = '(60S_{1/2} m_J=-0.5) x (60P_{3/2} m_J=+1.5)'
CACHE = os.path.join(ROOT, 'scripts', 'tmp', '_edge_final.json')
WINDOWS = [(340, 2000), (340, 1200), (600, 1800), (1000, 2000), (340, 950)]


def lockin(t, y, lo, hi):
    m = (t >= lo) & (t <= hi)
    tt, yy = t[m] - lo, y[m]
    n = tt.size
    if n < 64:
        return None
    w = np.hanning(n)
    sp = np.abs(np.fft.rfft((yy - yy.mean()) * w))
    fr = np.fft.rfftfreq(n, d=float(np.median(np.diff(tt))))
    k = int(np.argmax(sp[1:]) + 1)
    if 0 < k < sp.size - 1:
        a, b, c = sp[k - 1], sp[k], sp[k + 1]
        den = a - 2 * b + c
        f0 = float(fr[k] + (0.5 * (a - c) / den if den else 0.0) * (fr[1] - fr[0]))
    else:
        f0 = float(fr[k])
    G = np.column_stack([np.ones(n), np.cos(2 * np.pi * f0 * tt),
                         np.sin(2 * np.pi * f0 * tt)])
    coef, *_ = np.linalg.lstsq(G, yy, rcond=None)
    return (float(coef[0]), float(np.hypot(coef[1], coef[2])), f0)


def main():
    fin = json.load(io.open(CACHE, encoding='utf-8'))['rows']
    want = {r['tag']: r for r in fin}
    # 指纹 → (sid, k)
    ref = {}
    for kind in ('A', 'B', 'C'):
        for tag, br, n in profiles(kind):
            env = one(tag, br, n)['dynamic.waves_A'][2]['env']
            ref[tag] = np.array(env[1], float)
    loc = {}
    for d in sorted(glob.glob(os.path.join(ROOT, 'runtime', 'sweeps', 'sw*')),
                    key=os.path.getmtime):
        sp = os.path.join(d, 'spec.json')
        if not os.path.exists(sp):
            continue
        spec = json.load(io.open(sp, encoding='utf-8'))
        for i, p in enumerate(spec.get('points') or []):
            wa = p.get('dynamic.waves_A')
            if not wa or len(wa) != 3 or wa[2].get('env', [None])[0] != 'file':
                continue
            tt = np.array(wa[2]['env'][1], float)
            for tag, rt in ref.items():
                if rt.size == tt.size and np.allclose(rt, tt, atol=1e-6):
                    f = os.path.join(d, 'pt_%04d.npz' % i)
                    if os.path.exists(f) and (tag not in loc or
                                              os.path.getmtime(f) > loc[tag][2]):
                        loc[tag] = (f, i, os.path.getmtime(f))

    print('  %-18s %s' % ('tag', '  '.join('A0[%d-%d]' % w for w in WINDOWS)))
    print('  ' + '-' * 78)
    table = {}
    for tag in sorted(want):
        if tag not in loc:
            continue
        z = np.load(loc[tag][0], allow_pickle=False)
        labels = [str(x) for x in z['labels']]
        if K_AFT not in labels:
            continue
        t = np.asarray(z['series_t'], float)
        a = np.asarray(z['series_pop'], float)[:, labels.index(K_AFT)]
        vals = []
        for lo, hi in WINDOWS:
            L = lockin(t, a, lo, hi)
            vals.append(L[1] if L else np.nan)
        table[tag] = vals
        print('  %-18s %s' % (tag, '  '.join(
            '  %.4f  ' % v if not np.isnan(v) else '   --    ' for v in vals)))

    # 关键斜率：A0 是否随"越走越远"系统下降（跨窗口一致？）
    print('\n  === 跨窗口一致性 ===')
    for tag, v in table.items():
        v = np.array(v)
        v = v[~np.isnan(v)]
        if v.size:
            print('    %-18s 极差 %.4f（相对 %.1f%%）' %
                  (tag, v.max() - v.min(), 100 * (v.max() - v.min()) / v.mean()))

    print('\n  === 判决量：A 族 vs C 族，逐窗口 ===')
    for j, (lo, hi) in enumerate(WINDOWS):
        g = {k: table[k][j] for k in table if not np.isnan(table[k][j])}
        step = g.get('C/step'); d50 = g.get('C/down50')
        tri = {k: g[k] for k in g if k.startswith('A/')}
        if step and d50 and tri:
            best = max(tri.values())
            print('    窗口 %4d-%4d : 单向掉 %.1f%% | 往返最好 A0=%.4f（%+.1f%%）'
                  % (lo, hi, 100 * (step - d50) / step, best,
                     100 * (best - step) / step))

    out = os.path.join(ROOT, 'scripts', 'tmp', '_edge_final_windows.json')
    json.dump({'windows': WINDOWS, 'A0': table}, io.open(out, 'w', encoding='utf-8'),
              indent=1)
    print('\n  数据 → %s' % out)


if __name__ == '__main__':
    main()