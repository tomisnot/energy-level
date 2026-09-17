# -*- coding: utf-8 -*-
"""严格对拍：**同一条边沿**在两条通道（前人离线脚本 / 本轮 MCP）上的 A0。

为什么必须做：我先前把"前人 F 族"与"本轮 C 族"当作同一轨迹比较，得到
"30 倍差异"的结论——那是**方法错误**：
  · 两条通道的 A0 用了**不同分析窗**（前人 600–2000，本轮 340–2000）
  · 我生成 C 族折线时把 50 ns 下坡压缩成了 ~0.67 ns 的突降
  ⇒ 那次比较无效，必须重做。

本脚本把两边统一到**同一分析窗（340–2000 ns）**，并明确指出两边
实际光强轨迹的差别（下坡起点 255 vs 280 ns）。
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
W = (340.0, 2000.0)
K_AFT = '(60S_{1/2} m_J=-0.5) x (60P_{3/2} m_J=+1.5)'


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
        f0 = float(fr[k] + (0.5 * (a - c) / den if den else 0.0) * (fr[1] - fr[0]))
    else:
        f0 = float(fr[k])
    G = np.column_stack([np.ones(n), np.cos(2 * np.pi * f0 * tt),
                         np.sin(2 * np.pi * f0 * tt)])
    coef, *_ = np.linalg.lstsq(G, yy, rcond=None)
    return float(np.hypot(coef[1], coef[2])), float(coef[0])


def main():
    d = json.load(io.open(os.path.join(TMP, '_fall_mech.json'), encoding='utf-8'))
    rows = []
    for r in d['res']:
        p = os.path.join(TMP, r['npz'])
        if not os.path.exists(p):
            continue
        z = np.load(p)
        t, a = np.asarray(z['t'], float), np.asarray(z['aft'], float)
        A0, C = lockin(t, a, *W)
        rows.append(dict(tag=r['tag'], A0=A0, C=C, sum=C + A0, pk=r['pk']))

    ref = next(r for r in rows if r['tag'] == 'F/tf=0')
    print('  === 前人离线通道（统一窗 340–2000 ns）===')
    print('  %-14s %8s %8s %9s %9s' % ('tag', 'A0', 'C', 'C+A0', '相对损失'))
    for r in sorted(rows, key=lambda x: x['tag']):
        print('  %-14s %8.4f %8.4f %9.4f %+8.1f%%'
              % (r['tag'], r['A0'], r['C'], r['sum'],
                 100 * (ref['A0'] - r['A0']) / ref['A0']))

    # 本轮判决实验同窗重算（**不 import** _edge_final_analysis：它会重设 stdout）
    sys.path.insert(0, os.path.join(ROOT, 'scripts', 'research'))
    from _edge_profiles import profiles, one
    ref_env = {}
    for kind in ('A', 'B', 'C'):
        for tag, br, n in profiles(kind):
            env = one(tag, br, n)['dynamic.waves_A'][2]['env']
            ref_env[tag] = np.array(env[1], float)
    T = {}
    import glob
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
            for tag, rt in ref_env.items():
                if rt.size != tt.size or not np.allclose(rt, tt, atol=1e-6):
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
                a = np.asarray(z['series_pop'], float)[:, labels.index(K_AFT)]
                A0, C = lockin(t, a, *W)
                if tag not in T or os.path.getmtime(f) > T[tag][2]:
                    T[tag] = (A0, C, os.path.getmtime(f))

    print('\n  === 本轮 MCP 通道（同窗 340–2000 ns，同锁相代码）===')
    print('  %-18s %8s %8s %9s %9s' % ('tag', 'A0', 'C', 'C+A0', '相对损失'))
    base = T['C/step'][0]
    for k in ('C/step', 'C/down25', 'C/down50', 'B/hold10@0.71',
              'B/hold25@0.71', 'B/hold50@0.71', 'A/tri25_d1.00',
              'A/tri50_d1.00', 'A/tri100_d1.00'):
        if k not in T:
            continue
        A0, C, _ = T[k]
        print('  %-18s %8.4f %8.4f %9.4f %+8.1f%%'
              % (k, A0, C, A0 + C, 100 * (base - A0) / base))

    print('\n  === 结论 ===')
    f50 = next(r for r in rows if r['tag'] == 'F/tf=50')
    cd50 = T.get('C/down50')
    if cd50:
        print('    F/tf=50 (前人在 280 ns 处 50 ns 下坡)   A0 = %.4f' % f50['A0'])
        print('    C/down50 (本轮在 255 ns 处 50 ns 下坡)  A0 = %.4f' % cd50[0])
        print('    两者差 %.1f%% ⇒ %s'
              % (100 * abs(f50['A0'] - cd50[0]) / f50['A0'],
                 '一致（同一物理，两条通道对拍通过）'
                 if abs(f50['A0'] - cd50[0]) / f50['A0'] < 0.05 else
                 '**不一致，需查**（唯一差别 = 下坡起点 255 vs 280 ns）'))


if __name__ == '__main__':
    main()