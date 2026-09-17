# -*- coding: utf-8 -*-
"""最终综合：把全部批次（判决 A/B/C、平滑度 S、时序对照 T、前人离线 F/E）

统一到**同一分析窗（340–2000 ns）与同一锁相法**下汇总，输出：
  · 总表（A0 / C / C+A0 / a_s / Δφ / 相对损失）
  · 对拍检验（F/tf=50 vs C/down50 vs T/ramp50@255 vs T/ramp50@305）
  · 时序依赖检验（step/ramp 在不同 t_jump）
  · 候选控制量的相关矩阵
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

TMP = os.path.join(ROOT, 'scripts', 'tmp')
W = (340.0, 2000.0)
K_AFT = '(60S_{1/2} m_J=-0.5) x (60P_{3/2} m_J=+1.5)'
K_BEF = '(60P_{3/2} m_J=-0.5) x (60S_{1/2} m_J=+0.5)'
KINDS = ('A', 'B', 'C', 'ctrl', 'smooth')


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
    return dict(C=float(coef[0]), A0=float(np.hypot(coef[1], coef[2])),
                phi0=float(np.arctan2(-coef[2], coef[1])), f0=f0)


def collect():
    """指纹 → 统计算量。"""
    ref = {}
    for kind in KINDS:
        for tag, br, n in profiles(kind):
            env = one(tag, br, n)['dynamic.waves_A'][2]['env']
            ref[tag] = np.array(env[1], float)
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
            for tag, rt in ref.items():
                if rt.size != tt.size or not np.allclose(rt, tt, atol=1e-6):
                    continue
                f = os.path.join(sd, 'pt_%04d.npz' % i)
                if not os.path.exists(f) or tag in out:
                    continue
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
                L = lockin(t, a, *W)
                if L is None:
                    continue
                k = int(np.argmax(a))
                out[tag] = dict(tag=tag, **L,
                                pk=float(a[k]), bef=float(b[k]),
                                s=float(a[k] + b[k]),
                                as_frac=float(a[k] / (a[k] + b[k])))
    # 前人离线（统一窗重算）
    d = json.load(io.open(os.path.join(TMP, '_fall_mech.json'), encoding='utf-8'))
    for r in d['res']:
        p = os.path.join(TMP, r['npz'])
        if not os.path.exists(p):
            continue
        z = np.load(p)
        t, a = np.asarray(z['t'], float), np.asarray(z['aft'], float)
        b = np.asarray(z['bef'], float)
        L = lockin(t, a, *W)
        k = int(np.argmax(a))
        out[r['tag']] = dict(tag=r['tag'], legacy=True, **L,
                             pk=float(a[k]), bef=float(b[k]),
                             s=float(a[k] + b[k]),
                             as_frac=float(a[k] / (a[k] + b[k])))
    return out


GROUPS = [
    ('前人离线 F 族（单向下降）', ['F/tf=0', 'F/tf=10', 'F/tf=20', 'F/tf=35',
                                  'F/tf=50', 'F/tf=70', 'F/tf=100']),
    ('本轮 C 族（单向下降）', ['C/step', 'C/down25', 'C/down50']),
    ('本轮 A 族（对称往返）', ['A/tri25_d1.00', 'A/tri50_d1.00', 'A/tri100_d1.00']),
    ('本轮 B 族（停留式）', ['B/hold10@0.71', 'B/hold25@0.71', 'B/hold50@0.71']),
    ('平滑度族（同一轨迹，折点数↓）', ['S/ramp_n1', 'S/ramp_n2', 'S/ramp_n4',
                                       'S/ramp_n16', 'S/ramp_n64', 'S/ramp_n1000']),
    ('时序对照（同样回到 I_res）', ['T/step@255', 'T/step@305', 'T/step@400',
                                    'T/step@600', 'T/ramp50@255', 'T/ramp50@305']),
]


def main():
    T = collect()
    print('  === 总表（统一窗 340–2000 ns，统一锁相法）===')
    print('  %-22s %8s %8s %9s %8s %9s %9s'
          % ('tag', 'A0', 'C', 'C+A0', 'a/s', 'f0(MHz)', 'φ0'))
    print('  ' + '-' * 82)
    for nm, tags in GROUPS:
        print('  【%s】' % nm)
        for k in tags:
            r = T.get(k)
            if not r:
                print('  %-22s   (缺)' % k)
                continue
            print('  %-22s %8.4f %8.4f %9.4f %8.3f %9.4f %9.3f'
                  % (k, r['A0'], r['C'], r['A0'] + r['C'], r['as_frac'],
                     r['f0'] * 1000, r['phi0']))

    print('\n  === 对拍检验（同一物理：I3 → I_res 的 50 ns 单向下降）===')
    keys = ['F/tf=50', 'C/down50', 'T/ramp50@255', 'T/ramp50@305']
    vals = [(k, T[k]['A0']) for k in keys if k in T]
    for k, v in vals:
        print('     %-16s A0 = %.4f   C+A0 = %.4f' % (k, v, T[k]['A0'] + T[k]['C']))
    if len(vals) > 1:
        arr = np.array([v for _, v in vals])
        print('     极差 %.4f（相对 %.1f%%）⇒ %s'
              % (arr.max() - arr.min(), 100 * (arr.max() - arr.min()) / arr.mean(),
                 '**对拍通过**（同一物理，两条通道+两种实现一致）'
                 if (arr.max() - arr.min()) / arr.mean() < 0.03 else
                 '**不一致，需查**'))

    print('\n  === 时序依赖检验（同样回到 I_res，只改边沿时刻）===')
    for pre in ('T/step@', 'T/ramp50@'):
        g = [(k, T[k]['A0'], T[k]['as_frac']) for k in sorted(T)
             if k.startswith(pre)]
        if g:
            a = np.array([x[1] for x in g])
            print('     %s  A0 = %s  极差 %.4f（相对 %.1f%%）'
                  % (pre, ' '.join('%.4f' % x[1] for x in g),
                     a.max() - a.min(), 100 * (a.max() - a.min()) / a.mean()))

    print('\n  === 候选控制量相关矩阵（对本轮全部点）===')
    keys = [t for t in T if not T[t].get('legacy')]
    A0 = np.array([T[t]['A0'] for t in keys])
    base = T['C/step']['A0']
    loss = (base - A0) / base
    cands = {
        r'$|\Delta\phi|$': np.abs(np.array([T[t]['phi0'] - T['C/step']['phi0']
                                            for t in keys])),
        '折点数 log': np.array([np.log(int(t.split('_n')[-1])) if '_n' in t else 0.0
                                for t in keys]),
        '边沿时刻': np.array([float(t.split('@')[-1]) if '@' in t else 0.0
                              for t in keys]),
    }
    for nm, x in cands.items():
        if x.std() > 0:
            print('     corr(%s, 相对损失) = %+.3f' % (nm, np.corrcoef(x, loss)[0, 1]))
    print('     corr(a/s, 相对损失)     = %+.3f'
          % np.corrcoef([T[t]['as_frac'] for t in keys], loss)[0, 1])

    out = os.path.join(TMP, '_edge_all.json')
    json.dump({'rows': T}, io.open(out, 'w', encoding='utf-8'),
              ensure_ascii=False, indent=1)
    print('\n  数据 → %s' % out)


if __name__ == '__main__':
    main()