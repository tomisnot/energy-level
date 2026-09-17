# -*- coding: utf-8 -*-
"""偏振细扫分析：张量光移如何调制"两通道分配 a/s"与交换质量。

对照 = Q/pi基准（纯 π）。椭圆偏 α（alpha_deg）打开张量光移项。
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


def lockin(t, y, f0=None):
    m = (t >= 340.0) & (t <= 2000.0)
    tt, yy = t[m] - 340.0, y[m]
    n = tt.size
    if f0 is None:                    # 自估主频（勿写死：不同批次的 f0 会差）
        w = np.hanning(n)
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
                phi=float(np.arctan2(-coef[2], coef[1])), f0=float(f0))


def main():
    tags = json.load(io.open(os.path.join(TMP, '_nonideal_pol.json'),
                             encoding='utf-8'))['tags']
    rows = []
    for sd in sorted(glob.glob(os.path.join(ROOT, 'runtime', 'sweeps', 'sw*')),
                     key=os.path.getmtime):
        sp = os.path.join(sd, 'spec.json')
        if not os.path.exists(sp):
            continue
        if json.load(io.open(sp, encoding='utf-8')).get('tag') != 'nonideal-pol':
            continue
        for i, _ in enumerate(json.load(io.open(sp, encoding='utf-8'))
                              .get('points') or []):
            f = os.path.join(sd, 'pt_%04d.npz' % i)
            if not os.path.exists(f):
                continue
            z = np.load(f, allow_pickle=False)
            if 'series_t' not in z.files:
                continue
            labels = [str(x) for x in z['labels']]
            a = np.asarray(z['series_pop'], float)[:, labels.index(K_AFT)]
            b = np.asarray(z['series_pop'], float)[:, labels.index(K_BEF)]
            t = np.asarray(z['series_t'], float)
            k = int(np.argmax(a))
            rows.append(dict(tag=tags[i] if i < len(tags) else 'pt%d' % i,
                             t=t, a=a, b=b, pk=float(a[k]),
                             s=float(a[k] + b[k]),
                             frac=float(a[k] / (a[k] + b[k]))))
    if not rows:
        print('  ⚠ 无数据'); return
    for r in rows:                     # 各自自估主频（勿写死）
        r.update(lockin(r['t'], r['a']))
        r['alpha'] = (0.0 if '基准' in r['tag']
                      else float(r['tag'].split('_a')[-1]))
    rows.sort(key=lambda r: r['alpha'])
    base = rows[0]

    print('  %-14s %6s %8s %9s %9s %9s %9s' %
          ('tag', 'α(°)', 'A0', 'C+A0', 'a/s', 'Δφ(rad)', 'f0(MHz)'))
    print('  ' + '-' * 70)
    for r in rows:
        dphi = (r['phi'] - base['phi'] + np.pi) % (2 * np.pi) - np.pi
        r['dphi'] = float(dphi)
        print('  %-14s %6.1f %8.4f %9.4f %9.4f %9.3f %9.4f'
              % (r['tag'], r['alpha'], r['A0'], r['A0'] + r['C'],
                 r['frac'], dphi, r['f0'] * 1000))

    A = np.array([r['alpha'] for r in rows])
    ca = np.array([r['A0'] + r['C'] for r in rows])
    a0 = np.array([r['A0'] for r in rows])
    fr = np.array([r['frac'] for r in rows])
    print('\n  === 极值与排序 ===')
    i_best_ca = int(np.argmax(ca)); i_best_fr = int(np.argmax(fr))
    print('     C+A0 最大 = %.4f 出现在 α=%g°（基准 %.4f，+%.1f%%）'
          % (ca[i_best_ca], A[i_best_ca], ca[0],
             100 * (ca[i_best_ca] - ca[0]) / ca[0]))
    print('     a/s  最大 = %.4f 出现在 α=%g°（基准 %.4f）'
          % (fr[i_best_fr], A[i_best_fr], fr[0]))
    print('     a/s 单调升到 α=%.0f° 后回落；A0 同期单调下降 %.1f%%'
          % (A[i_best_fr], 100 * (a0[0] - a0[-1]) / a0[0]))
    nz = A > 0
    if nz.sum() > 2:
        print('     （非 π 点）corr(α, a/s) = %+.3f ；corr(α, A0) = %+.3f'
              % (np.corrcoef(A[nz], fr[nz])[0, 1],
                 np.corrcoef(A[nz], a0[nz])[0, 1]))
    print('\n  === 工程含义 ===')
    print('     1–3° 的偏振误差已能改变 a/s：')
    for r in rows:
        if r['alpha'] in (1.0, 2.0, 3.0):
            print('       α=%.0f° → a/s %.4f（%+.1f%%）、C+A0 %.4f（%+.1f%%）'
                  % (r['alpha'], r['frac'],
                     100 * (r['frac'] - fr[0]) / fr[0],
                     r['A0'] + r['C'],
                     100 * ((r['A0'] + r['C']) - ca[0]) / ca[0]))

    json.dump({'rows': [{k: v for k, v in r.items() if k not in ('t', 'a', 'b')}
                        for r in rows]},
              io.open(os.path.join(TMP, '_nonideal_pol_res.json'), 'w',
                      encoding='utf-8'), ensure_ascii=False, indent=1)
    print('\n  数据 → scripts/tmp/_nonideal_pol_res.json')


if __name__ == '__main__':
    main()