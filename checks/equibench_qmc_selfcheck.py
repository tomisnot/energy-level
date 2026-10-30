# -*- coding: utf-8 -*-
"""SSE-QMC 第一阶段：stoquastic 检查、统一结果契约和路由接线。"""
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from checks.checklib import Checker
from src.core.manybody.qmc import check_stoquastic, sse_run
from src.core.manybody.eq_route import solve_equilibrium

C = Checker('equibench_qmc')
calib = {
    'terms': ['I1'],
    'atoms': [dict(id=0, pos=(0., 0., 0.)), dict(id=1, pos=(1., 0., 0.))],
    'pairs': [dict(i=0, j=1, V=0.4, J=0., A=0.)],
}

info = check_stoquastic(calib, Om=np.array([-0.8, -0.8]),
                        delta=np.array([0.2, 0.2]), terms=['I1'])
C.check('stoquastic 检查', info['ok'] and info['average_sign'] == 1.0)
r = sse_run(calib, betas=[0.5, 1.0], Om=np.array([-0.8, -0.8]),
            delta=np.array([0.2, 0.2]), terms=['I1'], sweeps=120,
            burn_in=40, measure_every=2, seed=7)
C.group('SSE 结果契约', [
    ('backend', r['backend'] == 'sse-qmc'),
    ('betas', r['betas'].shape == (2,)),
    ('energy', r['E'].shape == (2,)),
    ('error', r['E_err'].shape == (2,)),
    ('sign', np.allclose(r['average_sign'], 1.0)),
    ('samples', np.all(r['samples'] > 0)),
])
try:
    check_stoquastic(calib, Om=np.array([0.8, 0.8]), delta=np.zeros(2), terms=['I1'])
    C.check('符号问题守卫', False, '应拒绝非 stoquastic 矩阵元')
except ValueError as e:
    C.check('符号问题守卫', 'sign problem' in str(e), str(e))

res = solve_equilibrium(dict(calib=calib, Om=np.array([-0.8, -0.8]),
                             delta=np.array([0.2, 0.2]), n_sites=2,
                             dmrg_ok=False), want='thermal', backend='qmc',
                        betas=[1.0], qmc_sweeps=60, qmc_burn_in=20,
                        qmc_measure_every=2, seed=3)
C.check('eq_route qmc', res['backend'] == 'qmc' and res['E'].shape == (1,))

# Independent two-level thermal anchor: H = 0.2 |r><r|, exact E(β=1).
one = {'terms': [], 'atoms': [dict(id=0, pos=(0., 0., 0.))], 'pairs': []}
qr = sse_run(one, betas=[1.0], Om=np.zeros(1), delta=np.array([0.2]),
             terms=[], sweeps=1000, burn_in=100, measure_every=2, seed=7)
exact_E = 0.2 * np.exp(-0.2) / (1.0 + np.exp(-0.2))
C.check('SSE 热力学锚', abs(float(qr['E'][0]) - exact_E) < 0.15,
        'ours=%.6g exact=%.6g' % (qr['E'][0], exact_E))
sys.exit(0 if C.summary() else 1)
