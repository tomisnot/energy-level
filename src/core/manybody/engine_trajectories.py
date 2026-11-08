# -*- coding: utf-8 -*-
"""MPS Monte-Carlo wave-function (quantum-trajectory) evolution."""
from .runtime import (time_grid, check_cancel, report_progress, reject_options,
                      atom_to_mps)
from .backends import validate_time_capabilities
from .results import time_result
from .couplings import active_terms
import numpy as np
from scipy.linalg import expm
from ..evolve_util import CONV
from .dmrg import terms_from_calib, cluster_model, _require_tenpy
from .state import product_vectors
from .engine_mps import _measure, _compile_link_groups
from .observables import link_loop_observables

__all__ = ["evolve_trajectories"]
_OPS = {
    "Id": np.eye(2, dtype=complex),
    "M": np.array([[0, 1], [0, 0]], complex),
    "P": np.array([[0, 0], [1, 0]], complex),
    "Nhat": np.array([[0, 0], [0, 1]], complex),
    "SZ": np.diag([1, -1]).astype(complex),
    "SX": np.array([[0, 1], [1, 0]], complex),
    "SY": np.array([[0, -1j], [1j, 0]], complex),
}

def _matrix(spec):
    if "matrix" in spec:
        a = np.asarray(spec["matrix"], complex)
    elif "real" in spec or "imag" in spec:
        a = np.asarray(spec.get("real", [[0, 0], [0, 0]]), float) + 1j * np.asarray(spec.get("imag", [[0, 0], [0, 0]]), float)
    else:
        name = str(spec.get("operator", "M"))
        if name not in _OPS:
            raise ValueError("未知 collapse operator=%s；请提供 2x2 matrix" % name)
        a = _OPS[name]
    if a.shape != (2, 2):
        raise ValueError("collapse operator 必须是 2x2 矩阵")
    return a

def _collapse_spec(collapse_ops, n, order):
    out = []
    for q, raw in enumerate(collapse_ops or []):
        if not isinstance(raw, dict):
            raise ValueError("collapse_ops 每项必须是 dict")
        site, rate = int(raw.get("site", -1)), float(raw.get("rate", 0.0))
        if not 0 <= site < n: raise ValueError("collapse operator site 越界")
        if rate < 0: raise ValueError("collapse rate 必须 >=0")
        L = _matrix(raw)
        out.append(dict(site=site, mps_site=int(order.index(site)), rate=rate, L=L, name="C%d" % q))
    return out

def _register_ops(model, cops):
    site = model.lat.mps_sites()[0]
    for c in cops:
        if c["name"] not in site.opnames:
            site.add_op(c["name"], c["L"])
        site.add_op(c["name"] + "dL", c["L"].conj().T @ c["L"])

def _norm(psi):
    return float(np.sqrt(max(0.0, float(np.real(psi.overlap(psi))))))

def _normalize(psi):
    nrm = _norm(psi)
    if nrm > 0:
        psi.set_B(0, psi.get_B(0, "B", copy=True) / nrm, "B")
        psi.norm = 1.0
    return nrm

@time_result(loss_kind='no_jump_weight_deficit', approximation='trajectory ensemble')
def evolve_trajectories(calib, psi0=None, t_grid=None, Om=None, delta=None,
                        terms=None, gamma=None, collapse_ops=None, n_trajectories=32,
                        algorithm="tdvp-trajectories", chi_max=100,
                        truncerr=1e-9, seed=0, max_step_ns=0.25,
                        want_observables=True, link_pairs=None, loops=None, segments=None,
                        plan=None, progress_arr=None, log=None, krylov_dim=20,
                        krylov_tol=1e-10, pauli_strings=None, **kwargs):
    """MPS quantum trajectories with local arbitrary 2x2 collapse channels.

    Rates are supplied in the public physical convention (1/ns = rate*1e-3
    for the MHz-like input units).  Hamiltonian evolution is Hermitian TDVP;
    local no-jump/jump operations use a controlled first-order split step.
    """
    reject_options(kwargs)
    validate_time_capabilities('tdvp-trajectories', segments=segments, plan=plan, gamma=gamma)
    t_grid = time_grid(t_grid)
    check_cancel(progress_arr)
    _require_tenpy()
    from tenpy.networks.mps import MPS
    from tenpy.algorithms.tdvp import TwoSiteTDVPEngine
    t = np.asarray(t_grid, float)
    if t.ndim != 1 or len(t) < 2 or np.any(np.diff(t) < 0): raise ValueError("trajectory 时间网格必须单调且至少两个点")
    if max_step_ns <= 0: raise ValueError("max_step_ns 必须 >0")
    n = len(calib.get("atoms") or [])
    if n == 0: raise ValueError("trajectory calib 没有原子")
    if pauli_strings and not want_observables:
        raise ValueError('manybody.pauli_strings 需要 want_observables=true')
    nt = int(n_trajectories)
    if nt < 1: raise ValueError("n_trajectories 必须 >=1")
    if str(algorithm).lower() not in ("tdvp-trajectories", "trajectories", "tdvp"): raise ValueError("trajectory algorithm 只支持 tdvp-trajectories")
    selected = dict(calib, terms=active_terms(calib) if terms is None else list(terms))
    order, onsite, couplings, info = terms_from_calib(selected, Om, delta)
    link_groups = _compile_link_groups(order, link_pairs)
    pauli_plan = None
    pauli_labels = []
    if pauli_strings:
        from .observables import PauliPlan, pauli_specs_to_mps_order
        mapped = pauli_specs_to_mps_order(pauli_strings, order)
        pauli_labels = [spec['label'] for spec in pauli_strings]
        pauli_plan = PauliPlan(n, mapped, kind='mps')
    model = cluster_model(onsite, couplings)
    if gamma is not None:
        ga = np.asarray(gamma, float)
        if ga.shape != (n,) or np.any(ga < 0): raise ValueError("trajectory gamma 必须是非负长度 N 数组")
        collapse_ops = list(collapse_ops or []) + [{"site": i, "rate": float(ga[i]), "operator": "M"} for i in range(n) if ga[i] > 0]
    has_hamiltonian = bool(np.any(np.abs(np.asarray(onsite, complex))) or len(couplings))
    cops = _collapse_spec(collapse_ops, n, order)
    _register_ops(model, cops)
    p0 = product_vectors(psi0, n)
    if p0 is None: raise ValueError("trajectory 初态必须是 product/local_states；ground 请先用基态后端准备")
    p_mps = atom_to_mps(p0, order)
    rng = np.random.default_rng(seed)
    vals = np.zeros((nt, len(t), n), float); jumps = np.zeros(nt, dtype=int); survival = np.ones((nt, len(t)), float)
    obs_acc = None
    link_acc = (np.zeros((nt, len(t), len(link_pairs)), complex)
                if want_observables and link_pairs is not None else None)
    pauli_acc = (np.zeros((nt, len(t), len(pauli_strings)), complex)
                 if pauli_plan is not None else None)
    for itr in range(nt):
        check_cancel(progress_arr)
        psi = MPS.from_product_state(model.lat.mps_sites(), p_mps, bc="finite", dtype=complex)
        norm_weight = 1.0
        for k in range(len(t)):
            report_progress(progress_arr, (itr+k/max(1,len(t)-1))/nt)
            if want_observables:
                p, _ss, _cc, _nn, oo = _measure(
                    psi, order, n, n // 2, False, True, want_xy=False,
                    link_pairs=link_pairs, link_groups=link_groups)
                if obs_acc is None:
                    obs_acc = {key: np.zeros((nt, len(t), n), dtype=(complex if key in ('sp', 'sm') else float))
                               for key in ('sx', 'sy', 'sz', 'sp', 'sm')}
                for key in ('sx', 'sy', 'sz', 'sp', 'sm'):
                    obs_acc[key][itr, k] = oo[key]
                if link_acc is not None:
                    link_acc[itr, k] = oo['link_corr']
                if pauli_acc is not None:
                    pauli_acc[itr, k] = pauli_plan.expect(psi)
            else:
                p = np.asarray(psi.expectation_value("Nhat"), float)
            if want_observables:
                vals[itr, k] = p
            else:
                vals[itr, k, order] = p
            survival[itr, k] = norm_weight
            if k == len(t) - 1: break
            interval = float(t[k + 1] - t[k]); nsub = max(1, int(np.ceil(interval / max_step_ns))); h = interval / nsub
            for _ in range(nsub):
                check_cancel(progress_arr)
                opts = dict(dt=CONV*h, N_steps=1, trunc_params=dict(chi_max=int(chi_max), trunc_cut=float(truncerr)), lanczos_params=dict(N_max=int(krylov_dim), E_tol=float(krylov_tol), hermitian=True, normalize=False))
                if has_hamiltonian:
                    TwoSiteTDVPEngine(psi, model, opts).run()
                    _normalize(psi)
                probs = []
                for c in cops:
                    x = np.asarray(psi.expectation_value(c["name"] + "dL"), float)
                    probs.append(max(0.0, h*c["rate"]*1e-3*float(x[c["mps_site"]])))
                total = float(sum(probs)); u = rng.random()
                if total > 0.15: raise ValueError("trajectory 内步过大：单步总跳跃概率 %.3f > 0.15；请减小 max_step_ns" % total)
                if cops and u < total:
                    z = u/total; acc = 0.0; chosen = cops[-1]
                    for c, pr in zip(cops, probs):
                        acc += pr/total
                        if z <= acc: chosen = c; break
                    psi.apply_local_op(chosen["mps_site"], chosen["name"], unitary=False, renormalize=False); _normalize(psi); jumps[itr] += 1
                else:
                    for c in cops:
                        K = expm(-0.5*h*c["rate"]*1e-3*(c["L"].conj().T @ c["L"]))
                        kname = "K_%d_%d_%d" % (itr, k, _)
                        site = model.lat.mps_sites()[0]
                        if kname not in site.opnames: site.add_op(kname, K)
                        psi.apply_local_op(c["mps_site"], kname, unitary=False, renormalize=False)
                    q = _norm(psi); norm_weight *= q*q; _normalize(psi)
    report_progress(progress_arr, 1.)
    mean = vals.mean(axis=0); sem = vals.std(axis=0, ddof=1)/np.sqrt(nt) if nt > 1 else np.zeros_like(mean)
    obs = None
    if want_observables:
        mean_obs = {key: obs_acc[key].mean(axis=0) for key in obs_acc}
        mean_obs['phase'] = np.angle(mean_obs['sm'])
        mean_obs['phase'][np.abs(mean_obs['sm']) < 1e-14] = np.nan
        mean_links = link_acc.mean(axis=0) if link_acc is not None else None
        lps, lws = link_loop_observables(mean_links, link_pairs, loops) if mean_links is not None and loops is not None else (
            np.empty((0, 0), float), np.empty((0, 0), float))
        mean_obs.update(xy_corr=None, loop_phase=(lps[-1].tolist() if lps.size else []),
                        loop_winding=(lws[-1].tolist() if lws.size else []),
                        loop_phase_series=lps, loop_winding_series=lws,
                        link_corr=mean_links, link_pairs=link_pairs,
                        approximation='trajectory ensemble average')
        if pauli_acc is not None:
            mean_obs['pauli_strings'] = pauli_acc.mean(axis=0)
            mean_obs['pauli_string_labels'] = pauli_labels
        obs = mean_obs
    return dict(t=t, pops=mean, pops_atom=mean, stderr=sem, states=[{"label":"atom%d"%i,"atom_id":i} for i in range(n)], method="manybody", backend="tdvp-trajectories", loss=1-survival.mean(axis=0), survival_weight=survival.mean(axis=0), jump_counts=jumps, observables=obs, error_report=dict(backend="tdvp-trajectories", n_atoms=n, n_trajectories=nt, chi_max=int(chi_max), truncerr=float(truncerr), max_step_ns=float(max_step_ns), first_order_jump_split=True, max_jump_probability=float(max_step_ns*1e-3*sum(c["rate"] for c in cops)), mps_order=list(order), n_legs=int(info.get("n_legs",0))))
