# -*- coding: utf-8 -*-
"""含时多体 MPS 引擎：TeNPy TEBD / TDVP。

TEBD 只接受 MPS 顺序上的最近邻 MPO；任意点云和长程偶极耦合自动使用
两站点 TDVP。两条路径共享 ``dmrg.terms_from_calib`` 的 Hamiltonian 映射，
所以与 ED 的 V/J/A、复交换相位和逐原子驱动保持同一约定。
"""
import time
from .results import time_result
from .runtime import segment_drive, segment_intervals, reject_options
from .couplings import active_terms
import numpy as np

from ..evolve_util import CONV
from .dmrg import (terms_from_calib, cluster_model, dmrg_run,
                   _require_tenpy)
from .state import product_vectors, make_mps_state, validate_mps_state
from .matrix_free import check_number_conserving
from .observables import loop_observables, link_loop_observables, PauliPlan

__all__ = ['evolve_mps']


def _product_state(n, spec):
    """Return local vectors; arbitrary product superpositions are physical MPS input."""
    return product_vectors(spec, n)


_segment_for = segment_drive


def _compile_link_groups(order, link_pairs):
    """Group link correlations by their left MPS endpoint.

    ``MPS.correlation_function`` can evaluate several right endpoints in one
    sweep.  Calling it once per link is especially expensive for 2-D arrays,
    where the atom order is a long cylinder.  The groups retain the original
    ``link_pairs`` order through the ``q`` indices.
    """
    if link_pairs is None:
        return []
    order = np.asarray(order, dtype=int)
    inv = np.empty(len(order), dtype=int)
    inv[order] = np.arange(len(order), dtype=int)
    grouped = {}
    for q, (a, b) in enumerate(link_pairs):
        ia, ib = int(inv[int(a)]), int(inv[int(b)])
        if ia <= ib:
            key = ('P', 'M', ia)
            target = ib
        else:
            # <P_i M_j> = <M_j P_i> for different sites; placing the
            # smaller MPS endpoint first lets TeNPy sweep only to the right.
            key = ('M', 'P', ib)
            target = ia
        grouped.setdefault(key, []).append((target, q))
    groups = []
    for (op1, op2, left), vals in grouped.items():
        vals.sort(key=lambda x: x[0])
        groups.append((op1, op2, int(left),
                       np.asarray([v[0] for v in vals], dtype=int),
                       np.asarray([v[1] for v in vals], dtype=int)))
    return groups


def _measure(psi, order, n, cut, want_corr, want_observables=False,
             want_xy=False, link_pairs=None, link_groups=None, timing=None,
             want_entropy=True):
    """MPS 顺序测量并映射回原子顺序。"""
    # Local MPS contractions assume canonical tensors.  Finite precision in
    # TDVP can invalidate that assumption even with an exact U(1) charge.
    # Recondition a copy, leaving the evolution engine's cached environments
    # untouched; do not turn gauge error into apparent charge/energy drift.
    if n > 1 and hasattr(psi, 'norm_test') and np.max(np.abs(psi.norm_test())) > 1e-12:
        psi = psi.copy()
        psi.canonical_form(renormalize=False, cutoff=0.)
    p_mps = np.asarray(psi.expectation_value('Nhat'), dtype=float)
    order = np.asarray(order, dtype=int)
    p_atom = np.zeros(n, float)
    p_atom[order] = p_mps
    S = (float(np.asarray(psi.entanglement_entropy(bonds=[cut])).ravel()[0])
         if want_entropy and n >= 2 else 0.0)
    corr = None
    t_stage = time.perf_counter()
    if want_corr:
        # One batched contraction replaces n**2 independent Python calls.
        # Nhat is Hermitian, so TeNPy can fill the lower triangle by symmetry.
        cc = psi.correlation_function(
            'Nhat', 'Nhat', sites1=list(range(n)),
            sites2=list(range(n)), hermitian=True)
        corr = np.zeros((n, n), float)
        corr[np.ix_(order, order)] = (
            np.real(cc) - p_mps[:, None] * p_mps[None, :])
    if timing is not None:
        timing['corr_measure_s'] += time.perf_counter() - t_stage
    t_stage = time.perf_counter()
    obs = None
    if want_observables:
        sz_mps = 1.0 - 2.0 * p_mps
        sp_mps = np.asarray(psi.expectation_value('P'), dtype=complex)
        sm_mps = np.asarray(psi.expectation_value('M'), dtype=complex)
        sx_mps, sy_mps = 2 * sm_mps.real, 2 * sm_mps.imag
        sx = np.zeros(n); sy = np.zeros(n); sz = np.zeros(n)
        sp = np.zeros(n, complex); sm = np.zeros(n, complex)
        sx[order], sy[order], sz[order] = sx_mps, sy_mps, 1.0 - 2.0 * p_mps
        sp[order], sm[order] = sp_mps, sm_mps
        phase = np.angle(sm); phase[np.abs(sm) < 1e-14] = np.nan
        xy = None
        links = None
        if want_xy:
            xy = np.full((n, n), np.nan + 0j, complex)
            cc = psi.correlation_function(
                'P', 'M', sites1=list(range(n)), sites2=list(range(n)))
            xy[np.ix_(order, order)] = cc
        if timing is not None:
            timing['local_measure_s'] += time.perf_counter() - t_stage
        t_stage = time.perf_counter()
        if link_pairs is not None:
            links = np.full(len(link_pairs), np.nan + 0j, complex)
            groups = link_groups if link_groups is not None else _compile_link_groups(order, link_pairs)
            for op1, op2, left, targets, qs in groups:
                vals = psi.correlation_function(
                    op1, op2, sites1=[left], sites2=targets.tolist())
                links[qs] = vals[0]
        if timing is not None:
            timing['link_measure_s'] += time.perf_counter() - t_stage
        obs = dict(sx=sx, sy=sy, sz=sz, sp=sp, sm=sm,
            phase=phase, xy_corr=xy, link_corr=links)
    norm = float(np.real(psi.overlap(psi)))
    if not np.isfinite(norm) or norm <= 0:
        raise RuntimeError('MPS norm measurement is non-finite or non-positive')
    return p_atom, S, corr, norm, obs


@time_result()
def evolve_mps(calib, psi0=None, t_grid=None, Om=None, delta=None, gamma=None,
               terms=None, segments=None, progress_arr=None, log=None,
               algorithm='tdvp', chi_max=200, truncerr=1e-10, max_sweeps=60,
               want_corr=True, want_entropy=True, diag=False,
               max_dim=None, plan=None, want_observables=True, loops=None,
               want_xy_corr=True,
               link_pairs=None,
               max_step_ns=None,
               charge_sector=None, return_state=False, krylov_dim=20, krylov_tol=1e-10,
               adaptive_switch_chi=16, adaptive_min_steps=1, mps_order=None,
               pauli_strings=None,
               **kwargs):
    """用 TeNPy 的 TEBD 或两站点 TDVP 演化多体二能级阵列。

    时间单位仍为 ns、Hamiltonian 为 MHz；内部把每个积分步长乘以
    ``CONV=2π×10⁻³``。``t_grid`` 只定义观测输出时刻；给出
    ``max_step_ns`` 时，每个输出区间会自动细分为不超过该值的内部步。
    ``algorithm='tebd'`` 要求所有耦合在 MPS 顺序上相邻，否则明确报错并建议
    ``tdvp``；TDVP 可处理任意 MPO 长程耦合。``tdvp-single`` 使用单站点
    TDVP（不会主动增长已有键维），``tdvp-adaptive`` 在键维达到
    ``adaptive_switch_chi`` 后从两站点切换到单站点。
    """
    reject_options(kwargs)
    _require_tenpy()
    from tenpy.networks.mps import MPS
    from tenpy.algorithms.tebd import TEBDEngine
    from tenpy.algorithms.tdvp import TwoSiteTDVPEngine, SingleSiteTDVPEngine
    from tenpy.models.model import NearestNeighborModel

    log = log or (lambda *a: None)
    if plan is not None and plan.get('projected'):
        raise ValueError('PXP/blockade projected Plan 当前使用 reduced-ED；'
                         '受限 MPO 编译器尚未启用，请选择 ed')
    t = np.asarray(t_grid, float)
    if t.ndim != 1 or t.size < 2 or not np.isfinite(t).all() or np.any(np.diff(t) < 0):
        raise ValueError('MPS 多体演化时间网格须为至少两个单调不减的点')
    if max_step_ns is not None:
        max_step_ns = float(max_step_ns)
        if max_step_ns <= 0.0:
            raise ValueError('MPS max_step_ns 必须 > 0')
    n = len(calib.get('atoms') or [])
    if n == 0:
        raise ValueError('MPS 多体演化：calib 没有原子')
    if pauli_strings and not want_observables:
        raise ValueError('manybody.pauli_strings 需要 want_observables=true')
    if link_pairs is not None:
        link_pairs = [(int(a), int(b)) for a, b in link_pairs]
        if any(a < 0 or a >= n or b < 0 or b >= n for a, b in link_pairs):
            raise ValueError('link_pairs 含越界原子下标')
    Om = np.zeros(n, complex) if Om is None else np.asarray(Om, complex)
    delta = np.zeros(n) if delta is None else np.asarray(delta, float)
    gamma = np.zeros(n) if gamma is None else np.asarray(gamma, float)
    if Om.shape != (n,) or delta.shape != (n,) or gamma.shape != (n,):
        raise ValueError('MPS 多体演化的 Om/delta/gamma 长度必须等于原子数')
    alg = str(algorithm or 'tdvp').lower()
    if alg in ('mps', 'mps_mpo'):
        alg = 'tdvp'
    if alg in ('tdvp-1site', 'tdvp1', 'single-site-tdvp'):
        alg = 'tdvp-single'
    if alg in ('tdvp-hybrid', 'adaptive-tdvp'):
        alg = 'tdvp-adaptive'
    if alg not in ('tebd', 'tdvp', 'tdvp-single', 'tdvp-adaptive'):
        raise ValueError('MPS algorithm 只支持 tebd/tdvp/tdvp-single/tdvp-adaptive，得到 %r' % algorithm)
    if np.any(np.abs(gamma) > 0.0):
        # TeNPy canonicalization normalizes MPS tensors during truncation.  A
        # no-jump norm is therefore not a trustworthy loss observable yet.
        # Refuse the request instead of returning a deceptively normalized run.
        raise ValueError('TEBD/TDVP 当前只支持厄米多体 Hamiltonian（gamma=0）；'
                         '非厄米损失请使用 ED')

    t_total0 = time.perf_counter()
    t_model0 = time.perf_counter()
    native_state = isinstance(psi0, dict) and psi0.get('kind') == 'mps'
    if native_state:
        validate_mps_state(psi0, n)
        saved_charge = psi0.get('charge_sector')
        if charge_sector is None:
            charge_sector = saved_charge
        elif saved_charge is not None and charge_sector != saved_charge:
            raise ValueError('初态与运行 charge_sector 不一致')
    if charge_sector is not None:
        if (isinstance(charge_sector, (bool, np.bool_)) or int(charge_sector) != charge_sector
                or not 0 <= int(charge_sector) <= n):
            raise ValueError('charge_sector 必须是 [0,N] 中的整数')
        charge_sector = int(charge_sector)
        check_number_conserving(calib, Om, terms)
        for segment in segments or []:
            check_number_conserving(calib, segment.get('Om'), terms)
    selected = dict(calib, terms=list(active_terms(calib) if terms is None else terms))
    order, onsite, couplings, info = terms_from_calib(selected, Om, delta, gamma,order=mps_order)
    order = np.asarray(order, dtype=int)
    link_groups = _compile_link_groups(order, link_pairs)
    pauli_plan = None
    pauli_labels = []
    if pauli_strings:
        from .observables import pauli_specs_to_mps_order
        mapped = pauli_specs_to_mps_order(pauli_strings, order)
        pauli_labels = [spec['label'] for spec in pauli_strings]
        pauli_plan = PauliPlan(n, mapped, kind='mps')
    conserve = 'Sz' if charge_sector is not None else None
    model = cluster_model(onsite, couplings, conserve=conserve)
    model_build_s = time.perf_counter() - t_model0
    if alg == 'tebd':
        max_range = max((b - a for _c, a, _oa, b, _ob in couplings), default=1)
        if max_range > 1:
            raise ValueError('TEBD 要求 MPS 顺序上的最近邻耦合；当前最大程=%d，请改用 tdvp'
                             % max_range)
        model = NearestNeighborModel.from_MPOModel(model)

    t_state0 = time.perf_counter()
    p0 = None if native_state else _product_state(n, psi0)
    if native_state:
        if list(psi0['order']) != order.tolist():
            raise ValueError('初态 MPS order 与当前模型不一致；不会静默重排')
        psi = psi0['psi'].copy()
        for old, new in zip(psi.sites, model.lat.mps_sites()):
            old.leg.test_equal(new.leg)
            if old.state_labels != new.state_labels:
                raise ValueError('初态与模型局部基标签不一致')
    elif p0 is None:
        if np.any(np.abs(gamma) > 0.0):
            raise ValueError("psi0='ground' 与非厄米 gamma 不兼容，请给乘积初态")
        r = dmrg_run(model, chi_max=chi_max, truncerr=truncerr,
                     max_sweeps=max_sweeps, want_state=True,
                     product_state=([1]*charge_sector + [0]*(n-charge_sector)
                                    if charge_sector is not None else None))
        psi = r.pop('psi')
    else:
        p_mps = [p0[int(a)] for a in order]
        psi = MPS.from_product_state(model.lat.mps_sites(), p_mps,
                                     bc='finite', dtype=complex)
    if charge_sector is not None:
        actual = float(np.real(np.sum(psi.expectation_value('Nhat'))))
        if abs(actual-charge_sector) > 1e-8:
            raise ValueError('初态占据数 %.16g 不等于 charge_sector=%d' % (actual, charge_sector))

    # Real static DMRG states must be promoted before exp(-i H t), including
    # one-site TDVP where updating a real container could otherwise lose phase.
    from .mps_dtype import select_mps_dtype
    psi = select_mps_dtype(psi,'complex',complex_required=True)
    state_init_s = time.perf_counter() - t_state0
    cut = int(n // 2)
    if want_entropy and n >= 2 and not (1 <= cut <= n - 1):
        raise ValueError('MPS entropy cut 越界')
    pops_atom = np.zeros((len(t), n), float)
    S = np.zeros(len(t), float)
    norms = np.zeros(len(t), float)
    corr_last = None
    obs_series = None
    if want_observables:
        obs_series = dict(sx=np.zeros((len(t), n)), sy=np.zeros((len(t), n)),
                          sz=np.zeros((len(t), n)), sp=np.zeros((len(t), n), complex),
                          sm=np.zeros((len(t), n), complex),
                          phase=np.full((len(t), n), np.nan))
    link_series = (np.full((len(t), len(link_pairs)), np.nan + 0j, complex)
                   if link_pairs is not None else None)
    pauli_series = (np.zeros((len(t), len(pauli_strings)), complex)
                    if pauli_strings else None)
    xy_last = None
    timing = dict(measure_s=0.0, record_s=0.0, corr_measure_s=0.0,
                  local_measure_s=0.0, link_measure_s=0.0,
                  engine_build_s=0.0, engine_run_s=0.0, segment_model_s=0.0)

    def record(k):
        nonlocal corr_last, xy_last
        t_record0 = time.perf_counter()
        pa, ss, cc, nn, oo = _measure(
            psi, order, n, cut, want_corr and k == len(t) - 1,
            want_observables, want_xy=(want_observables and want_xy_corr
                                       and k == len(t) - 1),
            link_pairs=link_pairs, link_groups=link_groups, timing=timing,
            want_entropy=want_entropy)
        pops_atom[k] = pa
        S[k] = ss if want_entropy else 0.0
        norms[k] = nn
        if cc is not None:
            corr_last = cc
        if oo is not None:
            for key in ('sx', 'sy', 'sz', 'sp', 'sm', 'phase'):
                obs_series[key][k] = oo[key]
            if oo['xy_corr'] is not None:
                xy_last = oo['xy_corr']
            if oo.get('link_corr') is not None:
                link_series[k] = oo['link_corr']
        if pauli_series is not None:
            pauli_series[k] = pauli_plan.expect(psi)
        timing['measure_s'] += time.perf_counter() - t_record0
        timing['record_s'] += time.perf_counter() - t_record0

    record(0)
    # TeNPy engines cache the MPS/MPO environments.  For a time-independent
    # Hamiltonian, reusing the engine avoids rebuilding those caches at every
    # output time.  A changing dt or drive segment still creates a fresh
    # engine, preserving the old semantics for nonuniform grids and drives.
    eng = None
    eng_dt = None
    eng_seg = object()
    engine_builds = 0
    engine_reuses = 0
    n_internal_steps = 0
    active_alg = ('tdvp' if alg == 'tdvp-adaptive' else alg)
    adaptive_switch_chi = int(adaptive_switch_chi)
    adaptive_min_steps = int(adaptive_min_steps)
    if adaptive_switch_chi < 2:
        raise ValueError('adaptive_switch_chi 必须 >= 2')
    if adaptive_min_steps < 0:
        raise ValueError('adaptive_min_steps 必须 >= 0')
    adaptive_switched_at = None
    for k in range(1, len(t)):
        steps = []
        for left, right in segment_intervals(t[k-1], t[k], segments):
            nsub = 1 if max_step_ns is None else max(1, int(np.ceil((right-left)/max_step_ns)))
            grid = np.linspace(left, right, nsub+1)
            steps.extend(zip(grid[:-1], grid[1:]))
        for ta, tb in steps:
            if progress_arr is not None and float(progress_arr[1]) != 0.0:
                raise RuntimeError('用户取消')
            dt_ns = tb - ta
            mid = 0.5 * (ta + tb)
            seg = _segment_for(mid, segments, n)
            seg_key = (None if not segments else
                       max([float(s['t0']) for s in segments
                            if float(s['t0']) <= mid],
                           default=min(float(s['t0']) for s in segments)))
            Om_k, de_k = (Om, delta) if seg is None else seg
            # A segment change is the only reason to rebuild the model.  Within
            # one piecewise-constant segment the engine/environments are valid
            # across internal steps (same as the constant-model path).
            seg_changed = bool(segments) and (eng is None or seg_key != eng_seg)
            if seg_changed:
                t_seg_model0 = time.perf_counter()
                _, on_k, cp_k, _ = terms_from_calib(selected, Om_k, de_k, gamma, order=order)
                model_k = cluster_model(on_k, cp_k, conserve=conserve)
                if alg == 'tebd':
                    model_k = NearestNeighborModel.from_MPOModel(model_k)
                timing['segment_model_s'] += time.perf_counter() - t_seg_model0
                eng = None
            elif not segments:
                model_k = model
            # Reuse only when model/dt/algorithm and segment are unchanged.
            can_reuse = (eng is not None
                         and np.isclose(eng_dt, dt_ns, rtol=1e-12, atol=1e-15)
                         and active_alg == getattr(eng, '_el_algorithm', active_alg)
                         and (not segments or seg_key == eng_seg))
            if can_reuse:
                engine_reuses += 1
            else:
                t_engine0 = time.perf_counter()
                opts = dict(dt=CONV * dt_ns, N_steps=1)
                if active_alg != 'tdvp-single':
                    opts['trunc_params'] = dict(chi_max=int(chi_max),
                                                trunc_cut=float(truncerr))
                if active_alg == 'tebd':
                    opts.update(order=2)
                    eng = TEBDEngine(psi, model_k, opts)
                elif active_alg == 'tdvp-single':
                    opts['lanczos_params'] = dict(N_max=int(krylov_dim),
                                                  E_tol=float(krylov_tol),
                                                  hermitian=not bool(np.any(np.abs(gamma) > 0.0)),
                                                  normalize=False)
                    eng = SingleSiteTDVPEngine(psi, model_k, opts)
                else:
                    opts['lanczos_params'] = dict(N_max=int(krylov_dim),
                                                  E_tol=float(krylov_tol),
                                                  hermitian=not bool(np.any(np.abs(gamma) > 0.0)),
                                                  normalize=False)
                    eng = TwoSiteTDVPEngine(psi, model_k, opts)
                eng._el_algorithm = active_alg
                eng_dt = dt_ns
                eng_seg = seg_key
                engine_builds += 1
                timing['engine_build_s'] += time.perf_counter() - t_engine0
            t_run0 = time.perf_counter()
            eng.run()
            timing['engine_run_s'] += time.perf_counter() - t_run0
            n_internal_steps += 1
            if (alg == 'tdvp-adaptive' and active_alg == 'tdvp' and
                    n_internal_steps >= adaptive_min_steps and
                    max(getattr(psi, 'chi', [1])) >= adaptive_switch_chi):
                active_alg = 'tdvp-single'
                adaptive_switched_at = n_internal_steps
                eng = None
                eng_dt = None
        record(k)
        if progress_arr is not None:
            progress_arr[0] = (k + 1) / len(t)
        if log is not None and (k == 1 or k == len(t) - 1):
            log('[manybody:%s] %d/%d t=%.6g ns chi=%s'
                % (alg, k + 1, len(t), t[k], max(getattr(psi, 'chi', [1]))))

    backend = 'tebd-mps' if alg == 'tebd' else 'tdvp-mps'
    states = [{'label': 'atom%d@%s' % (a['id'], tuple(a.get('pos', ()))),
               'atom_id': a['id']} for a in calib['atoms']]
    err = dict(backend=backend, algorithm=alg, n_atoms=n, n_t=len(t),
               chi_max=int(chi_max), truncerr=float(truncerr),
               max_chi=int(max(psi.chi)), max_norm_dev=float(np.max(np.abs(norms - 1.0))),
               max_step_ns=max_step_ns, n_internal_steps=int(n_internal_steps),
               tdvp_mode=('two-site' if alg == 'tdvp' else
                          'one-site' if alg == 'tdvp-single' else
                          'adaptive-two-to-one' if alg == 'tdvp-adaptive' else None),
               adaptive_switch_chi=(adaptive_switch_chi if alg == 'tdvp-adaptive' else None),
               adaptive_switched_at=(adaptive_switched_at if alg == 'tdvp-adaptive' else None),
               timings_s=dict(timing, model_build_s=float(model_build_s),
                              state_init_s=float(state_init_s),
                              engine_builds=int(engine_builds),
                              engine_reuses=int(engine_reuses),
                              total_s=float(time.perf_counter() - t_total0)),
               mps_order=list(order), entropy_cut=cut if want_entropy else None,
               n_legs=int(info.get('n_legs', 0)),
               charge_sector=charge_sector, conserve=conserve,
               long_range=(alg == 'tdvp'))
    lp, lw = loop_observables(xy_last, loops) if (
        want_observables and xy_last is not None and loops is not None) else ([], [])
    loop_series, loop_winding_series = link_loop_observables(
        link_series, link_pairs, loops) if (
            want_observables and link_series is not None and loops is not None) else (
                np.empty((0, 0), float), np.empty((0, 0), float))
    obs = (dict(obs_series, xy_corr=xy_last, loop_phase=lp,
                loop_winding=lw, loop_phase_series=loop_series,
                loop_winding_series=loop_winding_series,
                link_corr=link_series,
                link_pairs=link_pairs) if want_observables else None)
    if obs is not None and pauli_series is not None:
        obs['pauli_strings'] = pauli_series
        obs['pauli_string_labels'] = pauli_labels
    return dict(t=t, pops=pops_atom, states=states, method='manybody', backend=backend,
                pops_atom=pops_atom, S=S if want_entropy else None, corr=corr_last, loss=1.0 - norms,
                error_report=err,
                **({'state': make_mps_state(psi, order, charge_sector)} if return_state else {}),
                observables=obs,
                diag=(dict(err, terms=sorted(set(selected.get('terms') or []))) if diag else None))
