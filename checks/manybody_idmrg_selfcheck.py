# -*- coding: utf-8 -*-
"""iDMRG 独立判据：临界横场 Ising 的每 site 能量与解析 -4/π。"""
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from checks.checklib import Checker
from src.core.manybody.idmrg import idmrg_ground_state

C = Checker('manybody_idmrg')


def _ising_spec():
    # H = -σx σx - σz，用 2-site 单胞表示均匀链
    return {'cell': [{'site': 0}, {'site': 1}], 'terms': [
        {'kind': 'onsite', 'site': 0, 'op': 'Z', 'coeff': -1.0},
        {'kind': 'onsite', 'site': 1, 'op': 'Z', 'coeff': -1.0},
        {'kind': 'pair', 'i': 0, 'j': 1, 'dx': 0,
         'op_i': 'X', 'op_j': 'X', 'coeff': -1.0},
        {'kind': 'pair', 'i': 1, 'j': 0, 'dx': 1,
         'op_i': 'X', 'op_j': 'X', 'coeff': -1.0},
    ]}


def main():
    res = idmrg_ground_state(_ising_spec(), chi_max=32, max_sweeps=12)
    exact = -4.0 / np.pi
    C.close('临界 Ising E/site ≡ -4/π', res['E_per_site'], exact,
            rtol=0.0, atol=2e-5)
    C.check('E/cell = 2·E/site',
            abs(res['E_per_cell'] - 2 * res['E_per_site']) < 1e-12)
    C.check('每 cell 占据数形状', np.shape(res['n_atom']) == (2,))
    C.check('有限键维/截断误差可读',
            res['chi'] >= 2 and np.isfinite(res['trunc_err']))
    C.check('无限标记', res['infinite'] is True and res['n_sites'] is None)

    # 结构拒绝：单站点单胞不能跑两站点 iDMRG；负 dx/未知 op 也必须拒绝。
    one = {'cell': [{'site': 0}], 'terms': [
        {'kind': 'onsite', 'site': 0, 'op': 'Z', 'coeff': 1.0}]}
    bad_neg = _ising_spec()
    bad_neg['terms'][2]['dx'] = -1
    bad_op = _ising_spec()
    bad_op['terms'][0]['op'] = 'Q'
    for label, spec in (('单站点单胞', one), ('负 dx', bad_neg),
                        ('未知算符', bad_op)):
        try:
            idmrg_ground_state(spec, chi_max=8, max_sweeps=2)
            ok = False
        except ValueError:
            ok = True
        C.check('%s 被拒绝' % label, ok)

    # B1：P_i M_j 是总激发数守恒项，charge_sector 下必须放行。
    from src.core.manybody.idmrg import infinite_model
    pm_spec = {'cell': [{'site': 0}, {'site': 1}], 'terms': [
        {'kind': 'onsite', 'site': 0, 'op': 'N', 'coeff': 0.2},
        {'kind': 'pair', 'i': 0, 'j': 1, 'dx': 0,
         'op_i': 'P', 'op_j': 'M', 'coeff': 0.5},
        {'kind': 'pair', 'i': 1, 'j': 0, 'dx': 1,
         'op_i': 'P', 'op_j': 'M', 'coeff': 0.5},
    ]}
    try:
        infinite_model(pm_spec, charge_sector=1)
        ok_pm = True
    except ValueError:
        ok_pm = False
    bad_pm = {'cell': pm_spec['cell'], 'terms': [
        {'kind': 'onsite', 'site': 0, 'op': 'P', 'coeff': 1.0}]}
    try:
        infinite_model(bad_pm, charge_sector=1)
        bad_ok = False
    except ValueError:
        bad_ok = True
    C.group('charge_sector 的组合判据（B1）',
            [('P_i M_j 放行', ok_pm), ('单站点 P 仍拒绝', bad_ok)])

    # 经 build/run 产品路径：字段和 data_schema 必须出现。
    from src.core.manybody.eq_engine import build_equilibrium, run_equilibrium
    p = {'manybody.idmrg': _ising_spec(), 'manybody.eq_backend': 'idmrg',
         'manybody.chi_max': 16, 'manybody.max_sweeps': 4}
    run = build_equilibrium(None, p, want='gs')
    out = run_equilibrium(run)
    C.check('产品路径返回 E_per_site/E_per_cell',
            'E_per_site' in out and 'E_per_cell' in out)
    C.check('产品路径 data_schema 含 E_per_site',
            'E_per_site' in out['data_schema']['quantities'])
    return 0 if C.summary() else 1


if __name__ == '__main__':
    sys.exit(main())
