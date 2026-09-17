# -*- coding: utf-8 -*-
"""锁相式单模提取：把"交换能力"从多通道拍里干净地拿出来。

为什么不用 Hilbert：布居里至少两个频率分量（主交换 + 慢拍），解析信号把
拍当包络，会给出大于观测峰值的"包络"（E1/hold100 给 0.83 > 峰值 0.78 = 非物理）。

做法（lock-in）：在稳态段对 a(t) 加 Hann 窗做 FFT，取主峰频率 f0，
再用最小二乘投影到 {cos2πf0t, sin2πf0t, 1} 三个基上 ⇒ 得到
**该频率分量的幅度 A0 与相位 φ0**。A0 就是"交换能力的幅度"，
与相位采样无关，是判定"下降沿是否真的削掉交换"的正确量。

判据：
  Q1  A0 随 τ_f 是否系统下降？（真的削）还是基本平（只是相位移动）
  Q2  观测峰值 max_t a 与 A0 的关系是否 max ≈ C + A0（纯相位采样）
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


def lockin(t, y, t_lo=340.0, t_hi=2000.0):
    m = (t >= t_lo) & (t <= t_hi)
    tt, yy = t[m] - t_lo, y[m]
    n = tt.size
    w = np.hanning(n)
    sp = np.abs(np.fft.rfft((yy - yy.mean()) * w))
    fr = np.fft.rfftfreq(n, d=float(np.median(np.diff(tt))))
    k = int(np.argmax(sp[1:]) + 1)
    # 抛物线插值细化峰位
    if 0 < k < sp.size - 1:
        a, b, c = sp[k - 1], sp[k], sp[k + 1]
        dk = 0.5 * (a - c) / (a - 2 * b + c) if (a - 2 * b + c) != 0 else 0.0
        f0 = float(fr[k] + dk * (fr[1] - fr[0]))
    else:
        f0 = float(fr[k])
    # 三基最小二乘
    G = np.column_stack([np.ones(n), np.cos(2 * np.pi * f0 * tt),
                         np.sin(2 * np.pi * f0 * tt)])
    coef, *_ = np.linalg.lstsq(G, yy, rcond=None)
    C, cc, ss = coef
    A0 = float(np.hypot(cc, ss))
    phi0 = float(np.arctan2(-ss, cc))
    resid = float(np.sqrt(np.mean((G @ coef - yy) ** 2)))
    return dict(C=float(C), A0=A0, phi0=phi0, f0=f0, rms=resid,
                rms_rel=resid / A0 if A0 else None)


def main():
    rows = []
    for tag in TAGS:
        p = os.path.join(TMP, '_fall_mech_%s.npz' % tag.replace('/', '_'))
        if not os.path.exists(p):
            continue
        z = np.load(p)
        t = np.asarray(z['t'], float)
        a = np.asarray(z['aft'], float)
        b = np.asarray(z['bef'], float)
        r = lockin(t, a)
        rb = lockin(t, b)
        m = t >= 340.0
        rows.append(dict(tag=tag, pk=float(a.max()), t_pk=float(t[int(np.argmax(a))]),
                         **r, A0_b=rb['A0'], C_b=rb['C'],
                         max_in_win=float(a[m].max()),
                         mean_a=float(a[m].mean()), end_a=float(a[-1])))

    print('  %-16s %8s %8s %8s %8s %8s %8s %7s' %
          ('tag', '观测峰', 'A0', 'C', 'C+A0', 'rms/A0', 'f0(MHz)', 'φ0'))
    print('  ' + '-' * 88)
    for r in sorted(rows, key=lambda x: x['tag']):
        print('  %-16s %8.4f %8.4f %8.4f %8.4f %8.2e %8.4f %7.3f'
              % (r['tag'], r['pk'], r['A0'], r['C'], r['C'] + r['A0'],
                 r['rms_rel'], r['f0'] * 1000, r['phi0']))

    # ---- Q1：A0 随 τ_f ----
    print('\n  === Q1 交换幅度 A0 随下降沿宽 τ_f（F 族，唯一变量）===')
    fam = [(0, 'F/tf=0'), (10, 'F/tf=10'), (20, 'F/tf=20'), (35, 'F/tf=35'),
           (50, 'F/tf=50'), (70, 'F/tf=70'), (100, 'F/tf=100')]
    xs, a0s, pks = [], [], []
    for tf, tag in fam:
        r = next(x for x in rows if x['tag'] == tag)
        xs.append(tf); a0s.append(r['A0']); pks.append(r['pk'])
    a0s, pks, xs = np.array(a0s), np.array(pks), np.array(xs)
    print('     τ_f    :', '  '.join('%7.0f' % x for x in xs))
    print('     A0     :', '  '.join('%7.4f' % x for x in a0s))
    print('     观测峰 :', '  '.join('%7.4f' % x for x in pks))
    print('     A0 相对下降 %.1f%%   观测峰相对下降 %.1f%%'
          % (100 * (a0s[0] - a0s[-1]) / a0s[0],
             100 * (pks[0] - pks[-1]) / pks[0]))
    print('     corr(τ_f, A0) = %+.3f' % np.corrcoef(xs, a0s)[0, 1])

    # ---- Q2：峰值是否 ≈ C + A0 ----
    print('\n  === Q2 观测峰 vs (C + A0) ===')
    print('  %-16s %8s %8s %9s' % ('tag', '观测峰', 'C+A0', '差'))
    dif = []
    for r in sorted(rows, key=lambda x: x['tag']):
        dd = r['pk'] - (r['C'] + r['A0'])
        dif.append(dd)
        print('  %-16s %8.4f %8.4f %+9.4f' % (r['tag'], r['pk'],
                                              r['C'] + r['A0'], dd))
    print('     |差| 中位 %.2e  最大 %.2e' % (float(np.median(np.abs(dif))),
                                              float(np.max(np.abs(dif)))))

    out = os.path.join(TMP, '_edge_lockin.json')
    json.dump({'rows': rows}, io.open(out, 'w', encoding='utf-8'),
              ensure_ascii=False, indent=1)
    print('\n  数据 → %s' % out)


if __name__ == '__main__':
    main()