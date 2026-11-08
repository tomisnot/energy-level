# -*- coding: utf-8 -*-
"""Finite MPO-TDVP reference backend.

This backend is intentionally separate from the TEBD/swap GPU path.  It
uses TeNPy's trusted ``TwoSiteTDVPEngine`` (MPO environments + two-site
Lanczos + canonical SVD) and exposes EL's common time-result contract.
It is the correctness reference for the dense/GPU kernels; the default
solver is unchanged and this backend is selected explicitly.
"""
import numpy as np

from .runtime import check_cancel, report_progress, reject_options
from .results import time_result

__all__ = ['evolve_mpo_tdvp']
_CONV = 2.0 * np.pi * 1.0e-3


@time_result(approximation='two-site MPO-TDVP')
def evolve_mpo_tdvp(calib, psi0=None, t_grid=None, Om=None, delta=None,
                    gamma=None, terms=None, segments=None, progress_arr=None,
                    log=None, chi_max=200, truncerr=1e-10,
                    max_step_ns=None, krylov_dim=20, krylov_tol=1e-10,
                    want_observables=False, return_state=False,
                    charge_sector=None, mps_order=None, pauli_strings=None,
                    memory_budget_mb=None, max_sweeps=None, plan=None,
                    **kwargs):
    reject_options(kwargs)
    if segments:
        raise ValueError('MPO-TDVP 当前仅支持静态驱动；segments 尚未接入')
    if return_state or pauli_strings or charge_sector is not None or mps_order is not None:
        raise ValueError('MPO-TDVP 当前不支持 return_state/Pauli/固定荷/显式 order')
    if plan is not None and plan.get('projected'):
        raise ValueError('MPO-TDVP 暂不支持 projected Plan')
    from tenpy.networks.mps import MPS
    from tenpy.algorithms.tdvp import TwoSiteTDVPEngine
    from .dmrg import cluster_model, terms_from_calib
    from .couplings import active_terms

    if gamma is not None and np.any(np.asarray(gamma) != 0):
        raise ValueError('MPO-TDVP 当前只支持厄米演化（gamma=0）')
    log = log or (lambda *a: None)
    t = np.asarray(t_grid, float)
    if t.ndim != 1 or t.size < 2 or np.any(np.diff(t) < 0):
        raise ValueError('MPO-TDVP 时间网格须至少两个单调点')
    n = len(calib.get('atoms') or [])
    if n < 2:
        raise ValueError('MPO-TDVP 需要 N>=2')
    Om = np.zeros(n, complex) if Om is None else np.asarray(Om, complex)
    delta = np.zeros(n) if delta is None else np.asarray(delta, float)
    if Om.shape != (n,) or delta.shape != (n,):
        raise ValueError('MPO-TDVP 的 Om/delta 长度必须等于原子数')
    selected = dict(calib, terms=list(terms or active_terms(calib)))
    order, onsite, couplings, info = terms_from_calib(
        selected, Om, delta, np.zeros(n), order=None)
    order = np.asarray(order, dtype=int)
    model = cluster_model(onsite, couplings, conserve=None)
    if psi0 is None or (isinstance(psi0, str) and psi0 == 'ground'):
        psi = MPS.from_product_state(model.lat.mps_sites(), [0] * n, bc='finite')
        if n >= 3:
            from tenpy.algorithms.dmrg import TwoSiteDMRGEngine
            eng = TwoSiteDMRGEngine(psi, model, dict(
                max_sweeps=int(max_sweeps or 20),
                trunc_params=dict(chi_max=int(chi_max),
                                  trunc_cut=float(truncerr))))
            eng.run()
        else:
            psi.canonical_form(renormalize=False)
    else:
        if psi0 is None:
            raise ValueError('MPO-TDVP 需要乘积初态或 psi0=ground')
        if isinstance(psi0, str):
            p0 = [0 if c == 'g' else 1 for c in psi0]
            if len(p0) != n:
                raise ValueError('MPO-TDVP psi0 长度必须等于原子数')
        else:
            p0 = [int(x) for x in psi0]
        if len(p0) != n:
            raise ValueError('MPO-TDVP psi0 长度必须等于原子数')
        p0 = [p0[int(a)] for a in order]          # atom order -> MPS order
        psi = MPS.from_product_state(model.lat.mps_sites(), p0, bc='finite')
        psi.canonical_form(renormalize=False)
    pops = np.zeros((len(t), n), float)
    S_history = np.zeros(len(t), float)
    n_steps = 0

    def _atom_pops():
        vals = np.asarray(psi.expectation_value('Nhat'), float)
        out = np.empty(n, float)
        out[order] = vals
        return out

    pops[0] = _atom_pops()
    S_history[0] = float(psi.entanglement_entropy(bonds=[n // 2])[0])
    for k in range(1, len(t)):
        span = float(t[k] - t[k - 1])
        nsub = 1 if max_step_ns is None else max(
            1, int(np.ceil(span / float(max_step_ns))))
        for ta, tb in zip(np.linspace(t[k - 1], t[k], nsub + 1)[:-1],
                          np.linspace(t[k - 1], t[k], nsub + 1)[1:]):
            check_cancel(progress_arr)
            dt_ns = float(tb - ta)
            eng = TwoSiteTDVPEngine(
                psi, model,
                dict(dt=_CONV * dt_ns, max_dt=1.0,
                     trunc_params=dict(chi_max=int(chi_max),
                                       trunc_cut=float(truncerr)),
                     lanczos_params=dict(N_max=int(krylov_dim),
                                         E_tol=float(krylov_tol))))
            eng.run()
            n_steps += 1
            check_cancel(progress_arr)
        pops[k] = _atom_pops()
        S_history[k] = float(psi.entanglement_entropy(bonds=[n // 2])[0])
        if log and (k == 1 or k == len(t) - 1):
            log('[mpo-tdvp] %d/%d t=%.6g chi=%s'
                % (k + 1, len(t), t[k], max(psi.chi)))
        report_progress(progress_arr, k / len(t))
    report_progress(progress_arr, 1.0)
    states = [dict(label='atom%d' % i, atom_id=i) for i in range(n)]
    return dict(t=t, pops_atom=pops, pops=pops, states=states,
                method='manybody', backend='mpo-tdvp',
                S=S_history, corr=None, loss=None,
                error_report=dict(algorithm='two-site-tdvp-mpo',
                                  n_atoms=n, chi_max=int(chi_max),
                                  truncerr=float(truncerr),
                                  n_internal_steps=int(n_steps),
                                  max_chi=int(max(psi.chi)),
                                  mps_order=order.tolist(),
                                  backend='tenpy-two-site-tdvp'),
                observables=None, diag=None)
