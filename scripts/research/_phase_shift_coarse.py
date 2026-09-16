# -*- coding: utf-8 -*-
"""用已有 5 档（4 ns 网格）算相位时移与同刻布居差（独立于本次细网格复现）。"""
import glob
import io
import os
import sys

import numpy as np

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8',
                              errors='replace')
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
JOBS = {0.0: 'job3', 25.0: 'job4', 50.0: 'job5', 75.0: 'job6', 100.0: 'job7'}
BEF = '(60P_{3/2} m_J=-0.5) x (60S_{1/2} m_J=+0.5)'
AFT = '(60S_{1/2} m_J=-0.5) x (60P_{3/2} m_J=+1.5)'

D = {}
for tr, job in JOBS.items():
    g = glob.glob(os.path.join(ROOT, 'runtime', 'ai_results',
                               'ai_%s_*.npz' % job))
    if not g:
        print('  缺 %s' % job)
        continue
    d = np.load(sorted(g)[-1], allow_pickle=True)
    lab = [str(x['label']) for x in d['states_2d']]
    D[tr] = (np.asarray(d['t'], float),
             np.real(np.asarray(d['pops_2d'], float))[:, lab.index(BEF)],
             np.real(np.asarray(d['pops_2d'], float))[:, lab.index(AFT)])

t = D[0.0][0]
at = lambda a, tq: float(a[int(np.argmin(np.abs(t - tq)))])
print('  五档 D(t)=P(换后)−P(换前) 在关键时刻：')
print('  %-7s %8s %8s %8s %8s %8s' % ('tau_r', 't=120', 't=150', 't=200',
                                      't=250', 't=400'))
for tr in sorted(D):
    tt, b, a = D[tr]
    d = a - b
    print('  %-7g %8.4f %8.4f %8.4f %8.4f %8.4f'
          % (tr, at(d, 120), at(d, 150), at(d, 200), at(d, 250), at(d, 400)))

w = (t >= 250) & (t <= 800)
ref = (D[0.0][2] - D[0.0][1])[w]
ref = ref - ref.mean()
print('\n  相对阶跃(τ=0) 的互相关时移（正 = 滞后）：')
for tr in sorted(D):
    if tr == 0.0:
        continue
    x = (D[tr][2] - D[tr][1])[w]
    x = x - x.mean()
    c = np.correlate(x, ref, mode='full')
    lag = (np.arange(c.size) - (len(ref) - 1)) * (t[1] - t[0])
    print('    τ=%5.0f ns  →  %+6.1f ns' % (tr, float(lag[int(np.argmax(c))])))

print('\n  同刻布居差（τ=0 vs τ=25）：看差别从哪一刻开始')
for tq in (120, 150, 200, 220, 250, 280, 300, 350):
    b0, a0 = at(D[0.0][1], tq), at(D[0.0][2], tq)
    b1, a1 = at(D[25.0][1], tq), at(D[25.0][2], tq)
    print('    t=%4g  换前 %.4f→%.4f (Δ%+.4f)   换后 %.4f→%.4f (Δ%+.4f)'
          % (tq, b0, b1, b1 - b0, a0, a1, a1 - a0))
