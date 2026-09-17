# -*- coding: utf-8 -*-
"""非理想因素敏感性：谁改相位、谁改交换能力（a/s）。

判据：相位 = 锁相出的 φ0（共同主频）；交换能力 = 峰值时刻两通道分配 a/s。
对照点 = P0/pi基准（π 偏振、线性 50 ns 下坡、高斯 σ=20 π 脉冲）。
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
TMP = os.path.join(ROOT, 'scripts', 'tmp')
K_AFT = '(60S_{1/2} m_J=-0.5) x (60P_{3/2} m_J=+1.5)'
K_BEF = '(60P_{3/2} m_J=-0.5) x (60S_{1/2} m_J=+0.5)'
W = (340.0, 2000.0)


def lockin(t, y, f0=None, lo=W[0], hi=W[1]):
    m = (t >= lo) & (t <= hi)
    tt, yy = t[m] - lo, y[m]
    n = tt.size
    w = np.hanning(n)
    if f0 is None:
        sp = np.abs(np.fft.rfft((yy - yy.mean()) * w))
        fr = np.fft.rfftfreq(n, d=float(np.median(np.diff(tt))))
        k = int(np.argmax(sp[1:]) + 1)
        f0 = float(fr[k])
    G = np.column_stack([np.ones(n), np.cos(2 * np.pi * f0 * tt),
                         np.sin(2 * np.pi * f0 * tt)])
    coef, *_ = np.linalg.lstsq(G, yy, rcond=None)
    return dict(C=float(coef[0]), A0=float(np.hypot(coef[1], coef[2])),
                phi=float(np.arctan2(-coef[2], coef[1])), f0=float(f0))


def main():
    tags = json.load(io.open(os.path.join(TMP, '_nonideal_v2.json'),
                             encoding='utf-8'))['tags']
    # 指纹：按 spec 里 waves_A 的**光包络折线**建索引
    ref = {}
    for sd in sorted(glob.glob(os.path.join(ROOT, 'runtime', 'sweeps', 'sw*')),
                     key=os.path.getmtime):
        sp = os.path.join(sd, 'spec.json')
        if not os.path.exists(sp):
            continue
        spec = json.load(io.open(sp, encoding='utf-8'))
        if spec.get('tag') != 'nonideal-v2':
            continue
        pts = spec.get('points') or []
        for i, p in enumerate(pts):
            f = os.path.join(sd, 'pt_%04d.npz' % i)
            if not os.path.exists(f):
                continue
            ref[tags[i] if i < len(tags) else 'pt%d' % i] = f

    rows = []
    for tag, f in ref.items():
        z = np.load(f, allow_pickle=False)
        if 'series_t' not in z.files:
            continue
        labels = [str(x) for x in z['labels']]
        if K_AFT not in labels:
            continue
        t = np.asarray(z['series_t'], float)
        P = np.asarray(z['series_pop'], float)
        a = P[:, labels.index(K_AFT)]
        b = P[:, labels.index(K_BEF)]
        k = int(np.argmax(a))
        L = lockin(t, a)
        rows.append(dict(tag=tag, **L, pk=float(a[k]),
                         s=float(a[k] + b[k]),
                         frac=float(a[k] / (a[k] + b[k]))))
    rows.sort(key=lambda r: r['tag'])
    if not rows:
        print('  ⚠ 无数据'); return

    base = next((r for r in rows if '基准' in r['tag']), rows[0])
    print('  基准（π 偏振 + 线性 50 ns）: A0=%.4f  C=%.4f  φ0=%.3f  a/s=%.4f  f0=%.4f MHz'
          % (base['A0'], base['C'], base['phi'], base['frac'], base['f0'] * 1000))
    print()
    print('  %-18s %8s %8s %9s %10s %9s %9s'
          % ('tag', 'A0', 'C+A0', 'a/s', 'Δφ(rad)', 'ΔA0(%)', 'Δ(a/s)(%)'))
    print('  ' + '-' * 84)
    for r in rows:
        dphi = (r['phi'] - base['phi'] + np.pi) % (2 * np.pi) - np.pi
        dA = 100 * (r['A0'] - base['A0']) / base['A0']
        dF = 100 * (r['frac'] - base['frac']) / base['frac']
        r['dphi'] = float(dphi); r['dA'] = float(dA); r['dF'] = float(dF)
        print('  %-18s %8.4f %8.4f %9.4f %10.3f %+9.2f %+9.2f'
              % (r['tag'], r['A0'], r['A0'] + r['C'], r['frac'], dphi, dA, dF))

    print('\n  === 三组对照的"效力"排序（|Δφ| 与 |Δ(a/s)| 取组内最大）===')
    groups = {'P1/张量光移': 'P1/', 'S/同一积分的形状': 'S', 'M/微波边沿形状': 'M'}
    out = []
    for nm, pref in groups.items():
        g = [r for r in rows if r['tag'].startswith(pref) and r is not base]
        if not g:
            continue
        md = max(abs(r['dphi']) for r in g)
        mf = max(abs(r['dF']) for r in g)
        ma = max(abs(r['dA']) for r in g)
        out.append((nm, md, mf, ma))
    print('  %-20s %12s %14s %12s' % ('组', 'max|Δφ|', 'max|Δ(a/s)|%', 'max|ΔA0|%'))
    for nm, md, mf, ma in sorted(out, key=lambda x: -x[1]):
        print('  %-20s %12.3f %14.2f %12.2f' % (nm, md, mf, ma))
    print('\n  注：Δφ 只是振荡相位（卷绕在 2π 内），用来排序；')
    print('      交换能力的直接量是 a/s —— 它变了才说明"能力"真被改。')

    json.dump({'rows': rows, 'base': base['tag']},
              io.open(os.path.join(TMP, '_nonideal_v2_res.json'), 'w',
                      encoding='utf-8'), ensure_ascii=False, indent=1)
    print('\n  数据 → scripts/tmp/_nonideal_v2_res.json')


if __name__ == '__main__':
    main()