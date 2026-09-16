# -*- coding: utf-8 -*-
"""峰值时刻的两体逐态账：交换峰值缺的那 ~22% 到底在哪些态上。

用已有离线数据（不需软件、不需大内存）。
对 (τ_r, τ_f) 若干点，取 max P(换后态) 的时刻，把 484 个两体态按类别归并。
"""
import glob
import io
import os
import sys

import numpy as np

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8',
                              errors='replace')
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
JOBS = {(0.0, 0.0): 'job19', (0.0, 100.0): 'job22', (50.0, 0.0): 'job27',
        (75.0, 20.0): 'job32'}
K_AFT = '(60S_{1/2} m_J=-0.5) x (60P_{3/2} m_J=+1.5)'
K_BEF = '(60P_{3/2} m_J=-0.5) x (60S_{1/2} m_J=+0.5)'


def cls(lab):
    """把两体态标签归类。"""
    A, B = lab.split(') x (')
    A, B = A + ')', B


def which(lab):
    def fam(s):
        if '60S' in s:
            return 'S'
        if '60P' in s:
            return 'P'
        if '6P' in s:
            return '6P'
        return '?'
    A, B = lab.split(') x (')
    return fam(A + ')'), fam(B)


for (tr, tf), job in JOBS.items():
    g = glob.glob(os.path.join(ROOT, 'runtime', 'ai_results', 'ai_%s_*.npz' % job))
    if not g:
        print('  缺 %s' % job)
        continue
    d = np.load(sorted(g)[-1], allow_pickle=True)
    lab = [str(x['label']) for x in d['states_2d']]
    t = np.asarray(d['t'], float)
    P2 = np.real(np.asarray(d['pops_2d'], float))
    i_aft, i_bef = lab.index(K_AFT), lab.index(K_BEF)
    ip = int(np.argmax(P2[:, i_aft]))
    p = P2[ip]
    print('\n=== τ_r=%g, τ_f=%g （%s）峰值时刻 t=%.0f ns ===' % (tr, tf, job, t[ip]))
    print('  换后态 %.4f   换前态 %.4f   两者和 %.4f'
          % (p[i_aft], p[i_bef], p[i_aft] + p[i_bef]))
    # 按 (族,族) 归并
    acc = {}
    for i, s in enumerate(lab):
        k = '×'.join(which(s))
        acc[k] = acc.get(k, 0.0) + float(p[i])
    print('  按族归并: ' + '  '.join('%s=%.4f' % (k, v)
                                     for k, v in sorted(acc.items(), key=lambda x: -x[1])
                                     if v > 1e-3))
    # 交换两通道之外，最大的 6 个态
    order = np.argsort(-p)
    top = [(lab[i], float(p[i])) for i in order[:8]]
    print('  峰值时刻布居最大的 8 个态:')
    for s, v in top:
        tag = ' ★换后' if s == K_AFT else (' ★换前' if s == K_BEF else '')
        print('    %.4f  %s%s' % (v, s, tag))
    # 4 个工作子能级之内的两体组合合计
    work_sub = ('m_J=-0.5)', 'm_J=+0.5)', 'm_J=+1.5)')
    tot_in = 0.0
    for i, s in enumerate(lab):
        A, B = s.split(') x (')
        okA = ('60S' in A or '60P' in A) and any(w in A + ')' for w in work_sub)
        okB = ('60S' in B or '60P' in B) and any(w in B for w in work_sub)
        if okA and okB:
            tot_in += float(p[i])
    print('  仅由 4 个工作子能级组成的两体态合计 = %.4f' % tot_in)