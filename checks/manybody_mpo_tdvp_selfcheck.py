# -*- coding: utf-8 -*-
"""独立判据：MPO-TDVP 后端与 ED 短时对拍。"""
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from checks.checklib import Checker
from src.core.manybody.engine_ed import evolve_manybody
from src.core.manybody.mpo_tdvp import evolve_mpo_tdvp

C = Checker('manybody_mpo_tdvp')


def _cal():
    return {
        'atoms': [{'id': i, 'pos': [float(i), 0.0, 0.0]} for i in range(4)],
        'pairs': [{'i': i, 'j': i + 1, 'V': 0.1, 'J': 0.4, 'A': 0.0}
                  for i in range(3)],
    }


def main():
    cal = _cal()
    t = np.array([0.0, 1.0, 2.0])
    got = evolve_mpo_tdvp(cal, psi0='rggg', t_grid=t, chi_max=32)
    ref = evolve_manybody(cal, psi0='rggg', t_grid=t,
                          Om=np.zeros(4, complex), delta=np.zeros(4),
                          terms=['I1', 'I2'], want_corr=False)
    C.check('N4 布居 vs ED', np.max(np.abs(got['pops_atom'] - ref['pops_atom'])) < 1e-12)
    C.check('backend 标记', got['backend'] == 'mpo-tdvp')
    C.check('error_report 含内部步数',
            got['error_report']['n_internal_steps'] == 2)

    # B3：kagome/非恒等 MPS order 下，原子序初态与 pops 必须映射回来
    cal_o = {'atoms': [{'id': i, 'pos': p} for i, p in enumerate(
        [[2., 0., 0.], [0., 0., 0.], [1., 0., 0.], [3., 0., 0.]])],
        'pairs': [{'i': i, 'j': i + 1, 'V': 0.1, 'J': 0.4} for i in range(3)]}
    t2 = np.array([0.0, 1.0, 2.0])
    got_o = evolve_mpo_tdvp(cal_o, psi0='rggg', t_grid=t2, chi_max=32)
    ref_o = evolve_manybody(cal_o, psi0='rggg', t_grid=t2,
                            terms=['I1', 'I2'], want_corr=False)
    C.group('非恒等 atom/MPS order（B3）',
            [('order 非恒等', got_o['error_report'].get('mps_order') != [0, 1, 2, 3]),
             ('布居按原子序对拍',
              np.max(np.abs(got_o['pops_atom'] - ref_o['pops_atom'])) < 1e-4)])

    # 产品路径与拒绝面
    from src.core import protocol
    p = {'manybody.calib': cal, 'dynamic.mode': 2, 'dynamic.t_max': 1.0,
         'dynamic.npts': 2, 'manybody.want': '', 'manybody.solver': 'tdvp-mpo',
         'manybody.psi0': 'rggg', 'manybody.chi_max': 16,
         'manybody.max_step_ns': 1.0}
    out = protocol.evolve_run(protocol.build_run(None, p))
    C.check('产品路径可运行', np.shape(out['pops_atom']) == (2, 4))
    for label, kw in (('segments', dict(segments=[(0, 1)])),
                      ('Pauli', dict(pauli_strings=[{'label': 'x',
                                                     'terms': 'X'}])),
                      ('return_state', dict(return_state=True))):
        try:
            evolve_mpo_tdvp(cal, psi0='rggg', t_grid=np.array([0.0, 1.0]),
                            **kw)
            ok = False
        except ValueError:
            ok = True
        C.check('%s 被拒绝' % label, ok)
    return 0 if C.summary() else 1


if __name__ == '__main__':
    sys.exit(main())
