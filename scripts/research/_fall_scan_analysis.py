# -*- coding: utf-8 -*-
"""下降沿宽度 τ_f 扫描分析（run_20260915_182145_000008，job11..job15，R=12/θ=30）。

固定上升沿=理想阶跃，只改衔接处下降沿宽度 τ_f ∈ {0,10,20,50,100} ns。
看：交换峰值、峰值时刻、以及第二支 π 期间 B:60P(+0.5) 的漏出量。
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
RID = 'run_20260915_182145_000008'
BEF = '(60P_{3/2} m_J=-0.5) x (60S_{1/2} m_J=+0.5)'
AFT = '(60S_{1/2} m_J=-0.5) x (60P_{3/2} m_J=+1.5)'

j = json.load(io.open(os.path.join(ROOT, 'runtime', 'ai_results', RID + '.json'),
                      encoding='utf-8'))
rows = (j.get('result') or {}).get('rows') or []
print('  %-7s %10s %8s %10s %12s %12s %12s'
      % ('τ_f', 'maxP(后)', '@t', 'max(P后−P前)', 'B:60P+0.5@250',
         'max漏(其它m)', '@t'))
for r in rows:
    g = glob.glob(os.path.join(ROOT, 'runtime', 'ai_results',
                               'ai_%s_*.npz' % r['job']))
    if not g:
        continue
    d = np.load(sorted(g)[-1], allow_pickle=True)
    lab = [str(x['label']) for x in d['states_2d']]
    t = np.asarray(d['t'], float)
    A = np.real(np.asarray(d['pops_A'], float))
    B = np.real(np.asarray(d['pops_B'], float))
    P2 = np.real(np.asarray(d['pops_2d'], float))
    a, b = P2[:, lab.index(AFT)], P2[:, lab.index(BEF)]
    i_pk = int(np.argmax(a))
    out = 0.5 * (A[:, [2, 4]].sum(axis=1) + B[:, [2, 4]].sum(axis=1))
    i_o = int(np.argmax(out))
    at = lambda arr, tq: float(arr[int(np.argmin(np.abs(t - tq)))])
    print('  %-7g %10.4f %8.0f %10.4f %12.4f %12.4f %12.0f'
          % (r['tf'], a[i_pk], t[i_pk], (a - b).max(), at(B[:, 4], 250),
             out[i_o], t[i_o]))