# -*- coding: utf-8 -*-
"""系综驱动的独立判据：n=1、种子、无条件/后选分离、失败计数。"""
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from checks.checklib import Checker
from src.core import protocol
from src.core.manybody import run_ensemble

C = Checker('manybody_ensemble')


def _calib():
    return {
        'atoms': [{'id': i, 'pos': [float(i), 0.0, 0.0],
                   'Omega_mhz': 0.5, 'delta_mhz': 0.0, 'gamma_mhz': 0.0}
                  for i in range(3)],
        'pairs': [{'i': 0, 'j': 1, 'V': 0.1, 'J': 0.2, 'A': 0.0},
                  {'i': 1, 'j': 2, 'V': 0.1, 'J': 0.2, 'A': 0.0}],
    }


def _params():
    return {'manybody.calib': _calib(),
            'dynamic.mode': 2, 'dynamic.t_max': 0.5, 'dynamic.npts': 3,
            'manybody.solver': 'ed', 'manybody.psi0': 'ggg',
            'manybody.want': ''}


def _shift(rng):
    cal = _calib()
    for a in cal['atoms']:
        a['pos'][0] += float(rng.normal(0.0, 0.01))
    return cal


def main():
    p = _params()
    fixed = run_ensemble(p, lambda rng: _calib(), n_samples=1, seed=7,
                         observables=('pops_atom',), return_samples=True)
    direct = protocol.evolve_run(protocol.build_run(None, p))
    got = np.asarray(fixed['unconditional']['pops_atom']['mean'])
    C.close('n=1 与单次运行逐位一致',
            float(np.max(np.abs(got - direct['pops_atom']))), 0.0,
            rtol=0.0, atol=0.0)
    C.check('n=1 的 stderr 是 NaN',
            bool(np.all(np.isnan(fixed['unconditional']['pops_atom']['stderr']))))

    a = run_ensemble(p, _shift, n_samples=4, seed=11,
                     observables=('pops_atom',), return_samples=True)
    b = run_ensemble(p, _shift, n_samples=4, seed=11,
                     observables=('pops_atom',))
    C.close('同种子均值可复现',
            float(np.max(np.abs(a['unconditional']['pops_atom']['mean']
                              - b['unconditional']['pops_atom']['mean']))),
            0.0, rtol=0.0, atol=0.0)
    C.check('样本数/有效数正确',
            a['n_samples'] == 4 and a['n_valid'] == 4 and a['rejected'] == 0)
    C.check('无条件 n 与样本一致',
            a['unconditional']['pops_atom']['n'] == 4)
    C.check('stderr 形状与均值一致',
            np.shape(a['unconditional']['pops_atom']['stderr'])
            == np.shape(a['unconditional']['pops_atom']['mean']))

    cond = run_ensemble(p, _shift, n_samples=6, seed=3,
                        observables=('pops_atom',),
                        conditional=lambda res: float(res['pops_atom'][-1, 0]) > 0.0)
    C.check('后选与无条件分列',
            cond['unconditional']['pops_atom']['n'] == 6
            and cond['conditional'] is not None
            and 0 <= cond['conditional']['n'] <= 6)
    C.check('后选接受率可读',
            np.isfinite(cond['conditional']['acceptance']))

    bad = run_ensemble(p, lambda rng: {'atoms': _calib()['atoms'][:2], 'pairs': []},
                       n_samples=2, seed=0, observables=('pops_atom',))
    C.check('非法样本被计数而不是静默丢弃',
            bad['n_valid'] == 0 and bad['rejected'] == 2
            and bad['unconditional']['pops_atom']['n'] == 0)

    def flip(rng):
        return _calib(), {'manybody.psi0': 'grg'}
    C.check('sampler 可覆盖初态（初始缺陷）',
            run_ensemble(p, flip, n_samples=2, seed=0,
                         observables=('pops_atom',))['n_valid'] == 2)
    return 0 if C.summary() else 1


if __name__ == '__main__':
    sys.exit(main())
