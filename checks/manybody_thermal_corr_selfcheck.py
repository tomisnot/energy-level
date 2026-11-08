# -*- coding: utf-8 -*-
"""有限温 ED 关联的路线判据：solve_equilibrium 的 corr/n_atom 与独立 Gibbs 对拍。"""
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from scipy import sparse as sp

from checks.checklib import Checker
from src.core.manybody.assemble import build_H_sparse
from src.core.manybody.eq_route import solve_equilibrium

C = Checker('manybody_thermal_corr')


def _calib(n=4):
    return {'atoms': [{'id': i, 'pos': [float(i), 0.0, 0.0]} for i in range(n)],
            'pairs': [{'i': i, 'j': i + 1, 'V': 0.1, 'J': 0.4, 'A': 0.0}
                      for i in range(n - 1)]}


def _reference(cal, n, beta):
    H = build_H_sparse(cal, np.zeros(n), np.zeros(n), ['I1', 'I2'])
    Hd = H.toarray()
    evals, evecs = np.linalg.eigh(Hd)
    w = np.exp(-beta * (evals - evals.min())); w /= w.sum()
    rho = (evecs * w) @ evecs.conj().T
    bits = np.array([((np.arange(1 << n) >> (n - 1 - i)) & 1)
                     for i in range(n)])
    p = np.diag(rho).real
    n_atom = bits @ p
    corr = (bits * p) @ bits.T - np.outer(n_atom, n_atom)
    return Hd, n_atom, corr


def main():
    n = 4
    cal = _calib(n)
    system = dict(calib=cal, n_sites=n, Om=np.zeros(n), delta=np.zeros(n),
                  terms=['I1', 'I2'])
    beta = 0.5
    res = solve_equilibrium(system, want='thermal', backend='ed', betas=[beta])
    Hd, n_ref, corr_ref = _reference(cal, n, beta)
    n_atom = np.asarray(res['n_atom']).reshape(-1)
    corr = np.asarray(res['corr']).reshape(n, n)
    C.group('ED 热态关联', [
        ('E/C/F 有限', all(np.isfinite(res[k]).all() for k in ('E', 'C', 'F'))),
        ('n_atom 与独立 Gibbs <1e-12',
         np.max(np.abs(n_atom - n_ref)) < 1e-12),
        ('corr 与独立 Gibbs <1e-12',
         np.max(np.abs(corr - corr_ref)) < 1e-12),
        ('corr 轴声明为 beta×site×site',
         res['data_schema']['quantities']['corr']['axes'] == ['beta', 'site', 'site']),
    ])
    from src.core.runspec import result_summary
    summ = result_summary({'n_atoms': n, 'want': 'thermal', 'equilibrium': True}, res)
    C.group('热态摘要保形（B6）', [
        ('corr 首/末矩阵', np.shape(summ.get('corr_first_last')) == (2, n, n)),
        ('corr total', summ.get('corr_total') == 1),
        ('n_atom 保留 beta×site', np.shape(summ.get('n_atom')) == (1, n)),
        ('summary 块含 corr', 'corr' in (summ.get('summary') or {})),
    ])

    C.check('corr 无 unavailable 标记', 'corr_unavailable_reason' not in res)
    # N>12：仍给 E/C/F，但明确不提供精确 corr（不物化稠密本征矢）。
    out = solve_equilibrium({'H': sp.eye(16), 'n_sites': 13},
                            want='thermal', backend='ed', betas=[0.5])
    C.group('N>12 的精确 corr 显式拒绝', [
        ('有 unavailable 原因', 'corr_unavailable_reason' in out),
        ('仍给 E/C/F', all(k in out for k in ('E', 'C', 'F'))),
        ('不填 corr 零', 'corr' not in out),
    ])
    return 0 if C.summary() else 1


if __name__ == '__main__':
    sys.exit(main())
