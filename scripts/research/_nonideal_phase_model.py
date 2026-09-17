# -*- coding: utf-8 -*-
"""决定性验证：相位 ∝ 哪个积分？

三个"同一积分 χ=∫(I−I_res)dt = 14.5"却形状不同的轨迹，相位却差 0.42 rad
⇒ 相位不是 ∫δdt（那三者相同）。候选：
    ∫δdt   （一阶、线性项 —— 三者相同 ⇒ 不能解释）
    ∫δ²dt  （二阶、曲率项 —— 三者不同 ⇒ 可能）
以及下降沿密扫 21 点上的同款检验：Δφ 是否 ∝ ∫δ²dt。
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
IRES = 0.42
T0, TMAX = 280.0, 2000.0
CHI = 14.5


def profiles():
    def lin():
        tau = 2.0 * CHI / (1.0 - IRES)
        return ([(0.0, 0.0), (119.999, 0.0), (120.0, 1.0), (T0, 1.0)]
                + [(T0 + tau * k / 50, 1.0 - (1.0 - IRES) * k / 50)
                   for k in range(1, 51)] + [(TMAX, IRES)])

    def exp():
        tau = CHI / (1.0 - IRES)
        br = [(0.0, 0.0), (119.999, 0.0), (120.0, 1.0), (T0, 1.0)]
        for k in range(1, 61):
            x = T0 + 4.0 * k
            br.append((x, IRES + (1.0 - IRES) * np.exp(-(x - T0) / tau)))
        br.append((TMAX, IRES))
        return br

    def step():
        tau = CHI / (1.0 - IRES)
        return [(0.0, 0.0), (119.999, 0.0), (120.0, 1.0), (T0, 1.0),
                (T0 + tau, 1.0), (T0 + tau + 0.001, IRES), (TMAX, IRES)]
    return {'S1/lin_chi14.5': lin(), 'S2/exp_chi14.5': exp(),
            'S3/step_chi14.5': step()}


def moments(br):
    """∫(I−I_res)dt、∫(I−I_res)²dt、∫I dt（都在 T0 之后）"""
    t = np.array([b[0] for b in br], float)
    v = np.array([b[1] for b in br], float)
    m = t >= T0
    tt = np.clip(t[m], T0, TMAX)
    vv = v[m]
    d = vv - IRES
    return (float(np.trapezoid(d, tt)), float(np.trapezoid(d ** 2, tt)),
            float(np.trapezoid(vv, tt)))


def main():
    R = json.load(io.open(os.path.join(TMP, '_nonideal_v2_res.json'),
                          encoding='utf-8'))['rows']
    by = {r['tag']: r for r in R}
    base = by['P0/pi基准']
    print('  === 组②：同一 χ，不同时间结构 ===')
    print('  %-16s %10s %10s %10s %10s %10s' %
          ('tag', '∫δdt', '∫δ²dt', 'Δφ(rad)', 'a/s', 'A0'))
    rows = []
    for tag, br in profiles().items():
        m1, m2, mi = moments(br)
        r = by.get(tag)
        if not r:
            continue
        rows.append(dict(tag=tag, m1=m1, m2=m2, mi=mi, dphi=r['dphi'],
                         frac=r['frac'], A0=r['A0']))
        print('  %-16s %10.2f %10.1f %10.3f %10.4f %10.4f'
              % (tag, m1, m2, r['dphi'], r['frac'], r['A0']))
    if len(rows) >= 3:
        m1 = np.array([r['m1'] for r in rows])
        m2 = np.array([r['m2'] for r in rows])
        dphi = np.array([r['dphi'] for r in rows])
        fr = np.array([r['frac'] for r in rows])
        a0 = np.array([r['A0'] for r in rows])
        print('\n     ∫δdt 三者极差 %.3f（相同）⇒ 不能解释相位差' % (m1.max() - m1.min()))
        print('     ∫δ²dt 范围 %.1f–%.1f（差 %.1f 倍）' % (m2.min(), m2.max(), m2.max() / m2.min()))
        for nm, y in (('Δφ', dphi), ('a/s', fr), ('A0', a0)):
            print('     corr(∫δ²dt, %s) = %+.3f ; corr(∫δdt, %s) = %+.3f'
                  % (nm, np.corrcoef(m2, y)[0, 1], nm, np.corrcoef(m1, y)[0, 1]))

    # 降沿密扫上验证 Δφ ∝ ∫δ²dt
    D = json.load(io.open(os.path.join(TMP, '_edge_dense_res.json'),
                          encoding='utf-8'))['rows']
    print('\n  === 下降沿密扫 21 点：Δφ vs ∫δ²dt ===')
    print('  %-8s %10s %10s %10s %10s' % ('τ_f', '∫δ²dt', 'Δφ', 'A0', 'a/s'))
    xs, ys, a0s, frs = [], [], [], []
    for r in sorted(D, key=lambda x: x['tau']):
        tau = float(r['tau'])
        br = [(0.0, 0.0), (119.999, 0.0), (120.0, 1.0), (T0, 1.0)]
        n = max(8, int(round(tau)))
        for k in range(1, n + 1):
            br.append((T0 + tau * k / n, 1.0 - (1.0 - IRES) * k / n))
        br.append((TMAX, IRES))
        _, m2, _ = moments(br)
        xs.append(m2); ys.append(r['dphi']); a0s.append(r['A0']); frs.append(r['frac'])
        if r['tau'] % 25 == 0:
            print('  %-8d %10.1f %10.3f %10.4f %10.4f'
                  % (r['tau'], m2, r['dphi'], r['A0'], r['frac']))
    xs, ys, a0s, frs = map(np.array, (xs, ys, a0s, frs))
    # 过原点线性拟合
    k = float((xs * ys).sum() / (xs * xs).sum())
    resid = ys - k * xs
    print('     过原点拟合 Δφ = %.5f·∫δ²dt（残差 rms %.3f rad）' % (k, float(np.sqrt((resid**2).mean()))))
    print('     corr(∫δ²dt, Δφ)  = %+.3f' % np.corrcoef(xs, ys)[0, 1])
    print('     corr(∫δ²dt, A0)  = %+.3f ; corr(∫δ²dt, a/s) = %+.3f'
          % (np.corrcoef(xs, a0s)[0, 1], np.corrcoef(xs, frs)[0, 1]))
    print('     corr(|Δφ|,  A0) = %+.3f ; corr(|Δφ|,  a/s) = %+.3f'
          % (np.corrcoef(np.abs(ys), a0s)[0, 1], np.corrcoef(np.abs(ys), frs)[0, 1]))

    out = os.path.join(TMP, '_nonideal_phase_model.json')
    json.dump({'shape_test': rows,
               'dense_fit': {'k': k, 'corr_phi': float(np.corrcoef(xs, ys)[0, 1]),
                             'Q': xs.tolist(), 'dphi': ys.tolist(),
                             'A0': a0s.tolist(), 'frac': frs.tolist()}},
              io.open(out, 'w', encoding='utf-8'), ensure_ascii=False, indent=1)
    print('\n  数据 → scripts/tmp/_nonideal_phase_model.json')


if __name__ == '__main__':
    main()