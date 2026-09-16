# -*- coding: utf-8 -*-
"""跑出工作子能级的时间结构：什么时候漏、漏到哪、与光强包络什么关系。"""
import glob
import io
import os
import sys

import numpy as np

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8',
                              errors='replace')
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
JOBS = {0.0: 'job3', 50.0: 'job5', 100.0: 'job7'}
IDX = {'60P(-1.5)': 2, '60P(+0.5)': 4}
SIXP = list(range(6, 22))

for tr in sorted(JOBS):
    g = glob.glob(os.path.join(ROOT, 'runtime', 'ai_results',
                               'ai_%s_*.npz' % JOBS[tr]))
    if not g:
        continue
    d = np.load(sorted(g)[-1], allow_pickle=True)
    t = np.asarray(d['t'], float)
    A = np.real(np.asarray(d['pops_A'], float))
    B = np.real(np.asarray(d['pops_B'], float))
    at = lambda arr, tq: float(arr[int(np.argmin(np.abs(t - tq)))])
    print('\n=== τ_r = %g ns ===' % tr)
    print('  %-10s %9s %9s %9s %9s %9s %9s'
          % ('t(ns)', 'A:60P-1.5', 'A:60P+0.5', 'A:6P', 'B:60P-1.5',
             'B:60P+0.5', 'B:6P'))
    for tq in (0, 70, 120, 150, 200, 250, 280, 300, 350, 500, 800):
        print('  %-10g %9.4f %9.4f %9.4f %9.4f %9.4f %9.4f'
              % (tq, at(A[:, 2], tq), at(A[:, 4], tq), at(A[:, SIXP].sum(axis=1), tq),
                 at(B[:, 2], tq), at(B[:, 4], tq), at(B[:, SIXP].sum(axis=1), tq)))
    # 漏出量的时间峰
    out = 0.5 * (A[:, [2, 4]].sum(axis=1) + B[:, [2, 4]].sum(axis=1))
    sx = 0.5 * (A[:, SIXP].sum(axis=1) + B[:, SIXP].sum(axis=1))
    i1, i2 = int(np.argmax(out)), int(np.argmax(sx))
    print('  → max(60P其它m)=%.4f @ %.0f ns ； max(6P)=%.4f @ %.0f ns'
          % (out[i1], t[i1], sx[i2], t[i2]))
    # 光处于平台期(120-270)内的平均值 vs 全程
    m = (t >= 120) & (t <= 270)
    print('  → 光平台期(120–270ns)内 60P其它m 均值 %.4f / 6P 均值 %.4f'
          % (out[m].mean(), sx[m].mean()))