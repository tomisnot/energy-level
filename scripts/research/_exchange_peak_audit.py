# -*- coding: utf-8 -*-
"""交换峰值与"跑出工作子能级"审计。

工作子能级（交换涉及的 4 个，两原子同集）：60S(m=-0.5)=idx0、60S(m=+0.5)=idx1、
60P(m=-0.5)=idx3、60P(m=+1.5)=idx5。
"之外" = 60P(m=-1.5)=idx2、60P(m=+0.5)=idx4、以及 6P 的 16 个 idx6..21。

指标：
  · 交换峰值 = max_t P(换后态)，及其时刻；以及 max_t [P后−P前]
  · 峰值时刻的逐原子布居账（工作内 / 60P 其它 m / 6P）
  · 全程最大"跑出"量
"""
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
WORK = [0, 1, 3, 5]          # 工作子能级（逐原子）
P_OTHER_M = [2, 4]           # 60P 的另两个 m
SIXP = list(range(6, 22))    # 6P 全部

print('  %-7s %10s %8s %10s %8s %10s %10s %10s'
      % ('τ_r', 'maxP(后)', '@t', 'max(P后−P前)', '@t', '工作内', '60P其它m', '6P'))
rows = []
for tr in sorted(JOBS):
    g = glob.glob(os.path.join(ROOT, 'runtime', 'ai_results',
                               'ai_%s_*.npz' % JOBS[tr]))
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
    i_ct = int(np.argmax(a - b))
    w = 0.5 * (A[:, WORK].sum(axis=1) + B[:, WORK].sum(axis=1))
    om = 0.5 * (A[:, P_OTHER_M].sum(axis=1) + B[:, P_OTHER_M].sum(axis=1))
    sx = 0.5 * (A[:, SIXP].sum(axis=1) + B[:, SIXP].sum(axis=1))
    rows.append(dict(tr=tr, pk=float(a[i_pk]), t_pk=float(t[i_pk]),
                     ct=float((a - b)[i_ct]), t_ct=float(t[i_ct]),
                     work_pk=float(w[i_pk]), other_pk=float(om[i_pk]),
                     sixp_pk=float(sx[i_pk]),
                     work_min=float(w.min()), other_max=float(om.max()),
                     sixp_max=float(sx.max())))
    print('  %-7g %10.4f %8.0f %10.4f %8.0f %10.4f %10.4f %10.4f'
          % (tr, a[i_pk], t[i_pk], (a - b)[i_ct], t[i_ct],
             w[i_pk], om[i_pk], sx[i_pk]))

print('\n  === 全程"跑出工作子能级"的最大量 ===')
print('  %-7s %14s %14s %14s' % ('τ_r', 'max(60P其它m)', 'max(6P)', 'min(工作内)'))
for r in rows:
    print('  %-7g %14.4f %14.4f %14.4f'
          % (r['tr'], r['other_max'], r['sixp_max'], r['work_min']))

print('\n  === 交换峰值（max P(换后态)）随 τ_r 的变化 ===')
pk = {r['tr']: r['pk'] for r in rows}
for tr in sorted(pk):
    print('    τ=%5.0f ns → 峰值 %.4f  @ %.0f ns' % (tr, pk[tr], [r for r in rows if r['tr'] == tr][0]['t_pk']))
if 0.0 in pk:
    print('\n  相对阶跃(τ=0)的峰值偏移：')
    for tr in sorted(pk):
        if tr != 0.0:
            print('    τ=%5.0f ns → %+.4f' % (tr, pk[tr] - pk[0.0]))