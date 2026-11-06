# -*- coding: utf-8 -*-
"""TEBD/TDVP 含时 MPS 与 ED 的小系统交叉验证。"""
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from checks.checklib import Checker
from src.core.manybody.engine_ed import evolve_manybody
from src.core.manybody.engine_mps import evolve_mps
from src.core.manybody.runspec_mb import resolve_solver

C = Checker('manybody_mps')


def _calib(long_range=False):
    pairs = []
    for i in range(4):
        for j in range(i + 1, 4):
            if long_range or j == i + 1:
                pairs.append(dict(i=i, j=j, V=0.2 / (j - i), J=0j, A=0j))
    return dict(terms=['I1'],
                atoms=[dict(id=i, pos=(float(i), 0.0, 0.0)) for i in range(4)],
                pairs=pairs)


t = np.linspace(0.0, 20.0, 9)
Om = np.ones(4)
de = np.zeros(4)
calib = _calib()
ed = evolve_manybody(calib, psi0='gggg', t_grid=t, Om=Om, delta=de,
                     terms=['I1'], want_corr=False)
for alg in ('tebd', 'tdvp'):
    got = evolve_mps(calib, psi0='gggg', t_grid=t, Om=Om, delta=de,
                     terms=['I1'], algorithm=alg, chi_max=32,
                     truncerr=1e-12, want_corr=False)
    C.close('%s 与 ED 布居' % alg,
            float(np.max(np.abs(got['pops_atom'] - ed['pops_atom']))), 0.0,
            rtol=0.0, atol=2e-8)
    C.close('%s 范数守恒' % alg, float(np.max(np.abs(got['loss']))), 0.0,
            rtol=0.0, atol=2e-8)
    C.check('%s backend' % alg, got['backend'] == '%s-mps' % alg,
            'got=%s' % got['backend'])

long_calib = _calib(long_range=True)
try:
    evolve_mps(long_calib, psi0='gggg', t_grid=t[:2], Om=Om, delta=de,
               algorithm='tebd', want_corr=False)
    C.fail('TEBD 长程守卫', '应明确拒绝非最近邻 MPO')
except ValueError as e:
    C.check('TEBD 长程守卫', True, '拒绝并建议 TDVP：%s' % e)

# 分段驱动：同一段内复用 engine，只在段边界重建；语义必须与整段等价。
_segs = [dict(t0=0.0, Om=Om, delta=de), dict(t0=10.0, Om=Om, delta=de)]
_seg = evolve_mps(calib, psi0='gggg', t_grid=t, segments=_segs,
                  algorithm='tdvp', chi_max=32, truncerr=1e-12,
                  max_step_ns=1.0, want_corr=False)
_const = evolve_mps(calib, psi0='gggg', t_grid=t, Om=Om, delta=de,
                    algorithm='tdvp', chi_max=32, truncerr=1e-12,
                    max_step_ns=1.0, want_corr=False)
C.group('分段驱动不改变结果', [
    ('max|Δpops| < 1e-10（实测 %.2e）'
     % np.max(np.abs(_seg['pops_atom'] - _const['pops_atom'])),
     np.max(np.abs(_seg['pops_atom'] - _const['pops_atom'])) < 1e-10),
    ('段边界各重建一次（实测 %d）' % _seg['error_report']['timings_s']['engine_builds'],
     _seg['error_report']['timings_s']['engine_builds'] == 2),
    ('段内发生复用（实测 %d）' % _seg['error_report']['timings_s']['engine_reuses'],
     _seg['error_report']['timings_s']['engine_reuses'] > 0),
])

for solver, expected in [('auto', 'ed'), ('tebd', 'tebd'), ('tdvp', 'tdvp'), ('mps', 'tdvp')]:
    w = []
    C.check('solver=%s' % solver, resolve_solver(solver, 4, 1 << 12, w) == expected)
C.check('auto 大维度→TDVP', resolve_solver('auto', 25, 1 << 12, []) == 'tdvp')

sys.exit(0 if C.summary() else 1)
