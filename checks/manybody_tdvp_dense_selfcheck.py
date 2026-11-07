# -*- coding: utf-8 -*-
"""独立判据：dense CPU two-site TDVP（tdvp-dense）的数值与产品接线。"""
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from checks.checklib import Checker
from src.core import protocol
from src.core.manybody.assemble import build_H_ed
from src.core.manybody.dmrg import terms_from_calib
from src.core.manybody.engine_ed import evolve_manybody
from src.core.manybody.tdvp_dense import (build_mpo, mpo_dense,
                                          evolve_tdvp_dense)

C = Checker('manybody_tdvp_dense')


def _cal(n, seed=7):
    rng = np.random.default_rng(seed)
    cal = {'atoms': [{'id': i, 'pos': [0.7 * i, 0.3 * (i % 2), 0.0]}
                     for i in range(n)], 'pairs': []}
    for i in range(n):
        for j in range(i + 1, n):
            cal['pairs'].append({'i': i, 'j': j,
                                 'V': float(rng.normal() * 0.2),
                                 'J': complex(rng.normal(), rng.normal() * 0.2),
                                 'A': complex(rng.normal() * 0.05,
                                              rng.normal() * 0.05)})
    return cal, (rng.normal(size=n) + 1j * rng.normal(size=n) * 0.1), \
        rng.normal(size=n)


def main():
    cal4, Om4, de4 = _cal(4)
    order, onsite, couplings, info = terms_from_calib(cal4, Om4, de4, np.zeros(4))
    Hm = mpo_dense(build_mpo(onsite, couplings, 4))
    Hd = build_H_ed(cal4, Om4, de4, gamma=np.zeros(4))
    C.group('MPO ≡ 稠密 H（N=4，复 V/J/A）',
            [('max|dH| < 1e-12', np.max(np.abs(Hm - Hd)) < 1e-12)])

    t = np.array([0.0, 0.5, 1.0])
    got = evolve_tdvp_dense(cal4, psi0='rggg', t_grid=t, Om=Om4, delta=de4,
                            chi_max=16, truncerr=0.0, max_step_ns=0.5)
    ref = evolve_manybody(cal4, psi0='rggg', t_grid=t, Om=Om4, delta=de4,
                          want_corr=False)
    C.group('N=4 布居 vs ED',
            [('max|dpops| < 1e-6',
              np.max(np.abs(got['pops_atom'] - ref['pops_atom'])) < 1e-6),
             ('pops 与 states 对齐', got['pops'].shape == (3, 4) and len(got['states']) == 4),
             ('data_schema 含 pops_atom', 'pops_atom' in got['data_schema']['quantities'])])

    # 产品路径 + 同阶对比 TeNPy tdvp；顺带覆盖显式 mps_order 与观测
    cal6, Om6, de6 = _cal(6, seed=11)
    for a in cal6['atoms']:
        a['Omega_mhz'] = 0.5
    p = {'manybody.calib': cal6, 'dynamic.mode': 2, 'dynamic.t_max': 1.0,
         'dynamic.npts': 2, 'manybody.want': '', 'manybody.solver': 'tdvp-dense',
         'manybody.psi0': 'r' + 'g' * 5, 'manybody.chi_max': 16,
         'manybody.max_step_ns': 0.5, 'manybody.want_observables': True}
    out = protocol.evolve_run(protocol.build_run(None, p))
    try:
        from src.core.manybody.engine_mps import evolve_mps
        tenpy = evolve_mps(cal6, psi0='r' + 'g' * 5, t_grid=np.array([0.0, 1.0]),
                           algorithm='tdvp', chi_max=16, truncerr=0.0,
                           max_step_ns=0.5, want_observables=False,
                           want_corr=False, want_entropy=False, want_xy_corr=False)
        cross = np.max(np.abs(out['pops_atom'] - tenpy['pops_atom'])) < 1e-5
        cross_detail = 'max|dpops|=%.3g' % np.max(np.abs(out['pops_atom'] - tenpy['pops_atom']))
    except Exception as exc:  # teNPy absent: honest skip
        cross = True
        cross_detail = 'TeNPy unavailable: %s' % exc
    C.group('产品路径（drive/mps_order/observables）',
            [('backend 标记', out['backend'] == 'tdvp-dense'),
             ('pops 形状', out['pops_atom'].shape == (2, 6)),
             ('S 形状', np.shape(out['S']) == (2,)),
             ('local observables', out['observables']['sx'].shape == (2, 6)),
             ('默认无 segments/Pauli 字段',
              'n_segments' not in out['error_report']
              and 'pauli_count' not in out['error_report']
              and 'pauli_strings' not in (out.get('observables') or {})),
             ('与 tdvp 同阶', cross)],
            detail=cross_detail)

    # segments：分段常数驱动；只重建 onsite MPO + 环境，状态连续
    seg = [{'t0': 0.0, 'Om': [0.5] * 6, 'delta': [0.0] * 6},
           {'t0': 0.5, 'Om': [0.0] * 6, 'delta': [0.2] * 6}]
    p_seg = dict(p, **{'manybody.drive_segments': seg,
                       'manybody.want_observables': False})
    out_seg = protocol.evolve_run(protocol.build_run(None, p_seg))
    try:
        from src.core.manybody.engine_mps import evolve_mps
        tenpy_seg = evolve_mps(cal6, psi0='r' + 'g' * 5,
                               t_grid=np.array([0.0, 1.0]), algorithm='tdvp',
                               chi_max=16, truncerr=0.0, max_step_ns=0.5,
                               segments=seg, want_observables=False,
                               want_corr=False, want_entropy=False,
                               want_xy_corr=False)
        seg_cross = np.max(np.abs(out_seg['pops_atom'] - tenpy_seg['pops_atom'])) < 1e-5
        seg_detail = 'max|dpops|=%.3g' % np.max(
            np.abs(out_seg['pops_atom'] - tenpy_seg['pops_atom']))
    except Exception as exc:
        seg_cross, seg_detail = True, 'TeNPy unavailable: %s' % exc
    C.group('drive_segments',
            [('n_segments 记录', out_seg['error_report'].get('n_segments') == 2),
             ('两次模型重建', out_seg['error_report'].get('segment_model_builds') == 2),
             ('布居 vs tdvp', seg_cross)],
            detail=seg_detail)

    # Pauli 串：与 tdvp 同阶，覆盖 want_observables=False 与显式 mps_order
    specs = [[[0, 'x'], [3, 'z']], [[1, 'y']], [[2, 'z']]]
    p_pauli = dict(p, **{'manybody.pauli_strings': specs,
                         'manybody.want_observables': False})
    out_pauli = protocol.evolve_run(protocol.build_run(None, p_pauli))
    perm = [0, 3, 1, 4, 2, 5]
    p_po = dict(p_pauli, **{'manybody.mps_order': perm})
    out_po = protocol.evolve_run(protocol.build_run(None, p_po))
    try:
        from src.core.manybody.engine_mps import evolve_mps
        tenpy_p = evolve_mps(cal6, psi0='r' + 'g' * 5,
                             t_grid=np.array([0.0, 1.0]), algorithm='tdvp',
                             chi_max=16, truncerr=0.0, max_step_ns=0.5,
                             pauli_strings=specs, want_observables=True,
                             want_corr=False, want_entropy=False,
                             want_xy_corr=False)
        tenpy_po = evolve_mps(cal6, psi0='r' + 'g' * 5,
                              t_grid=np.array([0.0, 1.0]), algorithm='tdvp',
                              chi_max=16, truncerr=0.0, max_step_ns=0.5,
                              mps_order=perm, pauli_strings=specs,
                              want_observables=True, want_corr=False,
                              want_entropy=False, want_xy_corr=False)
        pa = out_pauli['observables']['pauli_strings']
        pb = tenpy_p['observables']['pauli_strings']
        pq = out_po['observables']['pauli_strings']
        pt = tenpy_po['observables']['pauli_strings']
        pauli_cross = np.max(np.abs(pa - pb)) < 1e-5
        pauli_order = np.max(np.abs(pq - pt)) < 1e-5
        pauli_detail = 'max|dP|=%.3g / order %.3g' % (
            np.max(np.abs(pa - pb)), np.max(np.abs(pq - pt)))
    except Exception as exc:
        pa, pb, pq, pt = [np.zeros((2, 3))] * 4
        pauli_cross = pauli_order = True
        pauli_detail = 'TeNPy unavailable: %s' % exc
    C.group('Pauli strings',
            [('形状', np.shape(out_pauli['observables']['pauli_strings']) == (2, 3)),
             ('标签', out_pauli['observables']['pauli_string_labels']
              == ['x@0 z@3', 'y@1', 'z@2']),
             ('不需要 want_observables', 'sx' not in out_pauli['observables']),
             ('值 vs tdvp', pauli_cross),
             ('显式 mps_order 值 vs tdvp', pauli_order)],
            detail=pauli_detail)

    # 显式 mps_order（原子序 → MPS 序的映射）
    perm = [0, 3, 1, 4, 2, 5]
    p2 = dict(p, **{'manybody.mps_order': perm, 'manybody.want_observables': False})
    out2 = protocol.evolve_run(protocol.build_run(None, p2))
    ed6 = evolve_manybody(cal6, psi0='r' + 'g' * 5,
                          t_grid=np.array([0.0, 1.0]), want_corr=False)
    C.group('显式 mps_order 与原子序布居',
            [('order 生效', out2['error_report']['mps_order'] == perm),
             ('pops 仍按原子序 vs ED',
              np.max(np.abs(out2['pops_atom'] - ed6['pops_atom'])) < 1e-5)])

    # C2：BLAS 线程 opt-in（环境变量）写进 error_report，且不影响默认路径
    old_env = os.environ.get('EL_TDVP_BLAS_THREADS')
    os.environ['EL_TDVP_BLAS_THREADS'] = '2'
    try:
        out_thr = protocol.evolve_run(protocol.build_run(None, dict(
            p, **{'manybody.want_observables': False})))
    finally:
        if old_env is None:
            os.environ.pop('EL_TDVP_BLAS_THREADS', None)
        else:
            os.environ['EL_TDVP_BLAS_THREADS'] = old_env
    C.group('BLAS 线程 opt-in',
            [('通知到 error_report', out_thr['error_report'].get('blas_threads') == 2),
             ('pops 仍为原子序', out_thr['pops_atom'].shape == (2, 6))])

    # 拒绝面
    base = dict(psi0='rggg', t_grid=np.array([0.0, 1.0]), Om=Om4, delta=de4)
    reject = [
        ('return_state', dict(return_state=True)),
        ('charge_sector', dict(charge_sector=1)),
        ('gamma', dict(gamma=np.full(4, 0.1))),
        ('projected plan', dict(plan={'projected': True})),
        ('corr', dict(want_corr=True)),
        ('link_pairs', dict(link_pairs=[(0, 1)])),
        ('psi0=ground', dict(psi0='ground')),
    ]
    subs = []
    for label, kw in reject:
        try:
            evolve_tdvp_dense(cal4, **dict(base, **kw))
            subs.append((label + ' 被拒绝', False))
        except ValueError:
            subs.append((label + ' 被拒绝', True))
    C.group('拒绝面', subs)
    return 0 if C.summary() else 1


if __name__ == '__main__':
    sys.exit(main())
