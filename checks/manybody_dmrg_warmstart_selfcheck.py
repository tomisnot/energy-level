# -*- coding: utf-8 -*-
"""DMRG 同进程热启动：上一段原生 MPS → 下一点，E0 一致且不隐式重排。"""
import os
import sys
import time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from checks.checklib import Checker
from checks.mb_benchlib import random_calib

C = Checker('manybody_dmrg_warmstart')

try:
    import tenpy  # noqa: F401
except Exception:
    C.skip('TeNPy DMRG 热启动', 'TeNPy unavailable')
    sys.exit(0 if C.summary() else 1)

from src.core.manybody.dmrg import dmrg_rydberg_cluster


def _calib(n=8):
    cal = random_calib(n, seed=17, complex_couplings=False)
    cal['terms'] = ['I1', 'I2']
    for p in cal['pairs']:
        p['V'] = 0.05
        p['J'] = 0.4
        p['A'] = 0.0
    return cal


def main():
    import numpy as np
    n = 8
    cal = _calib(n)
    Om = np.zeros(n)
    de = np.zeros(n)
    t0 = time.perf_counter()
    cold = dmrg_rydberg_cluster(cal, Om=Om, delta=de, chi_max=32,
                                truncerr=1e-10, max_sweeps=4,
                                charge_sector=4, return_state=True)
    t_cold = time.perf_counter() - t0
    state = cold['state']
    t0 = time.perf_counter()
    warm = dmrg_rydberg_cluster(cal, Om=Om, delta=de, chi_max=32,
                                truncerr=1e-10, max_sweeps=4,
                                charge_sector=4, return_state=True, psi0=state)
    t_warm = time.perf_counter() - t0
    C.group('原生 MPS 热启动', [
        ('E0 冷/热一致（Δ=%.2e）' % abs(cold['E0'] - warm['E0']),
         abs(cold['E0'] - warm['E0']) < 1e-8),
        ('order 原样保留', cold['order'] == warm['order']),
        ('热启动有态', warm.get('state', {}).get('kind') == 'mps'),
    ])
    C.check('冷/热启动都完成（%.2fs / %.2fs）' % (t_cold, t_warm),
            np.isfinite(cold['E0']) and np.isfinite(warm['E0']))
    # 顺序不匹配必须拒绝，而不是悄悄重排。
    bad = dict(state)
    bad['order'] = list(reversed(state['order']))
    try:
        dmrg_rydberg_cluster(cal, Om=Om, delta=de, chi_max=32,
                             max_sweeps=1, charge_sector=4,
                             return_state=True, psi0=bad, order=list(range(n)))
        C.check('热启动顺序不匹配被拒绝', False, '应拒绝')
    except ValueError as exc:
        C.check('热启动顺序不匹配被拒绝', 'order' in str(exc).lower(),
                str(exc)[:100])
    return 0 if C.summary() else 1


if __name__ == '__main__':
    sys.exit(main())
