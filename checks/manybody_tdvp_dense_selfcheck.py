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
    run_pauli = protocol.build_run(None, p_pauli)
    out_pauli = protocol.evolve_run(run_pauli)
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

    # MCP 摘要 / preview：从 data_schema 与 describe_drive 派生，不手写 Pauli/段
    from src.core.runspec import result_summary
    summ = result_summary(run_pauli, out_pauli)
    p_sum = summ.get('pauli_strings') or {}
    C.group('MCP 摘要 / preview',
            [('含时摘要含 Pauli', len(p_sum) == 3),
             ('含时摘要含 summary 块', 'summary' in summ),
             ('Pauli 终值对齐',
              abs((complex(p_sum['x@0 z@3'][0], p_sum['x@0 z@3'][1])
                   if isinstance(p_sum.get('x@0 z@3'), list)
                   else complex(p_sum.get('x@0 z@3', 0)))
                  - out_pauli['observables']['pauli_strings'][-1][0]) < 1e-9)])
    p_prev = dict(p_seg, **{'manybody.pauli_strings': specs})
    prev = protocol.preview(None, p_prev)
    C.group('MCP preview 的 segments/Pauli',
            [('n_segments', prev.get('n_segments') == 2),
             ('分段最大 Ω', abs(float(prev.get('max_Omega_mhz')) - 0.5) < 1e-9),
             ('分段最大 δ', abs(float(prev.get('max_delta_mhz')) - 0.2) < 1e-9),
             ('pauli_count', prev.get('pauli_count') == 3),
             ('estimate 的 ED 口径有提示',
              protocol.estimate(None, p_prev).get('cost_model') == 'ed'
              and 'tdvp-dense' in str(protocol.estimate(None, p_prev).get('backend_note')))])

    # B2：静态（无 segments）的 dt 收敛先做基线；不能把分段阶梯混进来。
    t2 = np.array([0.0, 20.0])
    ed_static = evolve_manybody(cal4, psi0='rggg', t_grid=t2,
                                Om=np.full(4, 15.4, complex), delta=de4,
                                want_corr=False)
    err_static = []
    for dt in (10.0, 5.0, 2.0):
        r = evolve_tdvp_dense(cal4, psi0='rggg', t_grid=t2,
                              Om=np.full(4, 15.4, complex), delta=de4,
                              chi_max=64, truncerr=1e-12, max_step_ns=dt)
        err_static.append(float(np.max(np.abs(
            r['pops_atom'][-1] - ed_static['pops_atom'][-1]))))
    C.group('静态 dt 收敛基线（B2）',
            [('dt 减半不是一阶（10→5）', err_static[1] < 0.65 * err_static[0]),
             ('dt 减半不是一阶（5→2）', err_static[2] < 0.65 * err_static[1])],
            detail='errors=%s' % ['%.2e' % x for x in err_static])

    # R2：分段控制点的 midpoint 规则；与高分辨参考比，midpoint 应优于 left。
    T, tau, d0, nseg = 20.0, 5.0, 2.0, 20
    def _ramp_segments(ns):
        tt = np.linspace(0.0, T, ns + 1)
        return tt, [{'t0': float(x), 'Om': [0.5] * 4,
                     'delta': [float(d0 * np.exp(-x / tau))] * 4}
                    for x in tt[:-1]]
    tt_ref, seg_ref = _ramp_segments(400)
    ref = evolve_manybody(cal4, psi0='rggg', t_grid=np.array([0.0, T]),
                          Om=np.full(4, 0.5, complex), delta=np.zeros(4),
                          terms=['I2'], segments=seg_ref,
                          drive_interp='midpoint', want_corr=False)
    tt, seg = _ramp_segments(nseg)
    ramp = {}
    for rule in ('left', 'midpoint'):
        ramp[rule] = evolve_manybody(
            cal4, psi0='rggg', t_grid=tt, Om=np.full(4, 0.5, complex),
            delta=np.zeros(4), terms=['I2'], segments=seg,
            drive_interp=rule, want_corr=False)
    e_left = float(np.max(np.abs(ramp['left']['pops_atom'][-1]
                                 - ref['pops_atom'][-1])))
    e_mid = float(np.max(np.abs(ramp['midpoint']['pops_atom'][-1]
                                - ref['pops_atom'][-1])))
    default = evolve_manybody(cal4, psi0='rggg', t_grid=tt,
                              Om=np.full(4, 0.5, complex), delta=np.zeros(4),
                              terms=['I2'], segments=seg, want_corr=False)
    C.group('drive_interp=midpoint（R2）',
            [('midpoint 优于 left', e_mid < e_left),
             ('默认 None 等价 left',
              np.array_equal(default['pops_atom'], ramp['left']['pops_atom']))],
            detail='left=%.3e midpoint=%.3e' % (e_left, e_mid))

    # R5：原生 dense MPS return_state；可继续喂给同一后端
    st = evolve_tdvp_dense(cal4, psi0='rggg', t_grid=np.array([0.0, 1.0]),
                           Om=Om4, delta=de4, chi_max=16, truncerr=1e-12,
                           max_step_ns=1.0, return_state=True)
    state = st.get('state') or {}
    again = evolve_tdvp_dense(cal4, psi0=state, t_grid=np.array([1.0, 2.0]),
                              Om=Om4, delta=de4, chi_max=16, truncerr=1e-12,
                              max_step_ns=1.0)
    C.group('原生 dense return_state（R5）',
            [('state backend/order', state.get('backend') == 'tdvp-dense'
              and state.get('order') == [0, 1, 2, 3]),
             ('续跑边界布居连续',
              np.max(np.abs(st['pops_atom'][-1] - again['pops_atom'][0])) < 1e-12)])

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
