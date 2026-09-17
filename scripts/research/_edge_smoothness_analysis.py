# -*- coding: utf-8 -*-
"""判决：下降沿的损伤来自"形状"还是"平滑度"？

同一条物理轨迹（I3 → I_res，50 ns 单向下坡），只改**折线采样点数 nseg**。
FileEnv 在折点间线性插值 ⇒ nseg 小 = 光强变化呈台阶（"跳过"某些强度），
nseg 大 = 平滑扫过。

  · 若 A0 随 nseg 单调下降并收敛 ⇒ **损伤来自平滑扫过共振**（前人 §8.1 的预期）
  · 若 A0 与 nseg 无关 ⇒ 形状/相位是全部原因，台阶无影响

对照锚：前人 F/tf=50（1001 折点）A0 = 0.3632（本会话 _edge_lockin）
        C/down50（6 折点）A0 = 0.3640（判决实验）
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
SDIR = os.path.join(ROOT, 'runtime', 'sweeps', 'sw20260916_104631_130')
K_AFT = '(60S_{1/2} m_J=-0.5) x (60P_{3/2} m_J=+1.5)'


def lockin(t, y, lo=340.0, hi=2000.0):
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


def main():
    tags = json.load(io.open(os.path.join(ROOT, 'scripts', 'tmp',
                                          '_edge_smoothness.json'),
                             encoding='utf-8'))['tags']
    rows = []
    for f in sorted(glob.glob(os.path.join(SDIR, 'pt_*.npz'))):
        k = int(os.path.basename(f)[3:7])
        z = np.load(f, allow_pickle=False)
        if 'series_t' not in z.files:
            continue
        labels = [str(x) for x in z['labels']]
        if K_AFT not in labels:
            continue
        t = np.asarray(z['series_t'], float)
        a = np.asarray(z['series_pop'], float)[:, labels.index(K_AFT)]
        L = lockin(t, a)
        if L is None:
            continue
        rows.append(dict(k=k, tag=tags[k] if k < len(tags) else 'pt%d' % k,
                         nseg=int((tags[k].split('_n')[-1]) if k < len(tags) else 0),
                         pk=float(a.max()), **L))
    rows.sort(key=lambda r: r['nseg'])

    ref = next((r for r in rows if r['nseg'] == 1), rows[0])
    print('  %-14s %6s %9s %9s %9s %9s' %
          ('tag', 'nseg', '观测峰', 'A0', '相对损失', 'f0(MHz)'))
    print('  ' + '-' * 62)
    for r in rows:
        print('  %-14s %6d %9.4f %9.4f %+8.1f%% %9.4f'
              % (r['tag'], r['nseg'], r['pk'], r['A0'],
                 100 * (ref['A0'] - r['A0']) / ref['A0'], r['f0'] * 1000))

    print('\n  === 判据 ===')
    n = np.array([r['nseg'] for r in rows], float)
    a0 = np.array([r['A0'] for r in rows])
    lo, hi = a0[:2].mean(), a0[-2:].mean()
    print('     最粗折线 (nseg≤2) A0 = %.4f' % lo)
    print('     最细折线 (nseg≥256) A0 = %.4f' % hi)
    print('     细-粗 = %+.4f（相对 %+.1f%%）' % (hi - lo, 100 * (hi - lo) / lo))
    print('     corr(log nseg, A0) = %+.3f' % np.corrcoef(np.log(n), a0)[0, 1])
    print('     ⇒ %s' % ('**损伤确实来自"平滑扫过"**：越平滑越差'
                        if hi < lo - 0.01 else
                        '**与平滑度基本无关**：形状/相位是全部原因'))

    print('\n  === 与两种"同一轨迹"实现对照 ===')
    print('     前人 F/tf=50（1001 折点，离线通道）A0 = 0.3632')
    print('     C/down50（6 折点）      A0 = 0.3640  ← 两者几乎相同')
    if rows:
        near = min(rows, key=lambda r: abs(r['A0'] - 0.3632))
        print('     本轮 nseg=%d           A0 = %.4f' % (near['nseg'], near['A0']))
    print('     ⇒ 说明折点数（6 vs 1001）**不影响**结果 ⇒ 该"30 倍差异"由其它因素造成')

    out = os.path.join(ROOT, 'scripts', 'tmp', '_edge_smoothness_res.json')
    json.dump({'rows': rows}, io.open(out, 'w', encoding='utf-8'),
              ensure_ascii=False, indent=1)
    print('\n  数据 → %s' % out)


if __name__ == '__main__':
    main()