# -*- coding: utf-8 -*-
"""CUDA MPS evolution: nearest-neighbour and arbitrary-pair swap TEBD.

Hamiltonian calibration is translated here; model-independent MPS kernels
live in mps_torch.  No lattice shape, coupling type or experiment is hardcoded.
"""
from collections import OrderedDict
import time
from .results import time_result
import numpy as np

from ..evolve_util import CONV
from .dmrg import terms_from_calib
from .state import product_vectors
from .observables import link_loop_observables, loop_observables
from .schema_mb import validate_drive_segments, validate_link_pairs, validate_loops
from .mps_torch import TorchMPS

__all__ = ['evolve_mps_gpu']


def _require_torch_cuda():
    try:
        import torch
    except ImportError as exc:
        raise ImportError('GPU MPS needs a CUDA build of PyTorch') from exc
    if not torch.cuda.is_available():
        raise RuntimeError('GPU MPS needs an available CUDA device')
    return torch


def _ops(torch, dtype, device):
    arrays = dict(Nhat=[[0, 0], [0, 1]], P=[[0, 0], [1, 0]],
                  M=[[0, 1], [0, 0]], Z=[[1, 0], [0, -1]])
    return {k: torch.tensor(v, dtype=dtype, device=device) for k, v in arrays.items()}


def _gate(torch, h, dt_ns):
    # Works for one-site matrices and batches of two-site matrices.
    w, v = torch.linalg.eigh(h)
    return (v * torch.exp((-1j * CONV * dt_ns) * w).unsqueeze(-2)) @ v.mH


def _initial_tensors(torch, psi0, n, atom_order, dtype, device):
    if isinstance(psi0, dict) and psi0.get('kind') == 'mps':
        if psi0.get('backend') == 'tenpy':
            raise ValueError('GPU MPS 需要显式 tensors；不隐式转换 TeNPy 带荷状态')
        given_order = list(psi0.get('order', range(n)))
        if given_order != list(atom_order):
            raise ValueError('MPS initial state order must match error_report.mps_order / cylinder_order')
        raw = psi0.get('tensors', [])
        if len(raw) != n:
            raise ValueError('MPS initial tensors count must equal atom count')
        ts = [torch.as_tensor(a, dtype=dtype, device=device) for a in raw]
        if any(a.ndim != 3 or a.shape[1] != 2 for a in ts):
            raise ValueError('two-level MPS input tensors have shape (left, 2, right)')
        return ts
    if isinstance(psi0, (int, np.integer)) and not isinstance(psi0, bool):
        if not 0 <= int(psi0) < (1 << n):
            raise ValueError('product basis index out of range')
        psi0 = ''.join('r' if int(psi0) & (1 << (n-1-i)) else 'g' for i in range(n))
    vecs = product_vectors(psi0, n)
    if vecs is None:
        raise ValueError("GPU MPS 'ground' requires a prepared MPS; provide kind=mps tensors from a ground-state solver")
    return [torch.as_tensor(vecs[a], dtype=dtype, device=device).reshape(1, 2, 1)
            for a in atom_order]


@time_result()
def evolve_mps_gpu(calib, psi0=None, t_grid=None, Om=None, delta=None, gamma=None,
                   terms=None, segments=None, progress_arr=None, log=None,
                   max_step_ns=None, chi_max=100, truncerr=1e-10,
                   want_observables=True, link_pairs=None, loops=None,
                   want_xy_corr=False, want_corr=False, want_entropy=False,
                   dtype='float64', device='cuda', order=2, algorithm='tebd',
                   plan=None, memory_budget_mb=None, max_dim=None, diag=False,
                   profile_timing=False, svd_driver=None,
                   return_state=False, pauli_strings=None,
                   **kwargs):
    """Finite two-level MPS evolution with first/second-order Trotter splitting.

    algorithm='tebd' requires nearest neighbours in MPS order;
    algorithm='swap-tebd' accepts arbitrary pair Hamiltonians and geometry.
    Both support complex V/J/A calibration, site-dependent complex drives,
    exact piecewise-constant drive boundaries, selected directed links, loops,
    optional final full correlations, and optional entropy at the middle cut.
    Each SVD uses a canonical Schmidt basis.  Reported discarded weight includes
    routing SWAPs.  Small chi is often faster on CPU: CUDA is an explicit choice.
    """
    torch = _require_torch_cuda()
    if plan is not None and plan.get('projected'):
        raise ValueError('projected PXP/blockade HamiltonianPlan requires reduced ED; GPU MPS cannot ignore the projection')
    if kwargs:
        raise TypeError('unsupported GPU MPS options: %s' % ', '.join(sorted(kwargs)))
    if algorithm not in ('tebd', 'swap-tebd'):
        raise ValueError('GPU MPS algorithm must be tebd or swap-tebd')
    if order not in (1, 2):
        raise ValueError('GPU TEBD order must be 1 or 2')
    t = np.asarray(t_grid, float)
    if t.ndim != 1 or t.size < 2 or not np.isfinite(t).all() or np.any(np.diff(t) < 0):
        raise ValueError('GPU MPS t_grid must contain >=2 finite monotone times')
    max_step_ns = float(max_step_ns if max_step_ns is not None else max(np.max(np.diff(t)), 1.))
    if not np.isfinite(max_step_ns) or max_step_ns <= 0:
        raise ValueError('max_step_ns must be finite and > 0')
    n = len(calib.get('atoms') or [])
    if n < 1:
        raise ValueError('GPU MPS calib has no atoms')
    if pauli_strings and not want_observables:
        raise ValueError('manybody.pauli_strings 需要 want_observables=true')
    Om = np.zeros(n, complex) if Om is None else np.asarray(Om, complex)
    delta = np.zeros(n) if delta is None else np.asarray(delta, float)
    gamma = np.zeros(n) if gamma is None else np.asarray(gamma, float)
    if any(x.shape != (n,) or not np.isfinite(x).all() for x in (Om, delta, gamma)):
        raise ValueError('Om/delta/gamma must be finite arrays of length n_atoms')
    if np.any(gamma):
        raise ValueError('GPU MPS supports Hermitian evolution only (gamma=0); use ED/trajectories for loss')
    segments = validate_drive_segments(segments, n)
    loops = validate_loops(loops, n)
    pairs = list(validate_link_pairs(link_pairs, n) or [])
    for loop in loops or []:
        for a, b in zip(loop, loop[1:] + loop[:1]):
            if (a, b) not in pairs and (b, a) not in pairs:
                pairs.append((a, b))
    types = {'float32': torch.complex64, 'complex64': torch.complex64,
             'float64': torch.complex128, 'complex128': torch.complex128}
    if dtype not in types:
        raise ValueError('dtype must be float32/float64/complex64/complex128')
    tdtype = types[dtype]
    if svd_driver not in (None, 'gesvdj', 'gesvd'):
        raise ValueError('svd_driver must be gesvdj, gesvd, or None')
    dev = torch.device(device)
    if dev.type != 'cuda':
        raise ValueError('GPU MPS device must be a CUDA device')
    profile_timing = bool(profile_timing)
    if profile_timing:
        torch.cuda.synchronize(dev)
    started = time.perf_counter()
    ops = _ops(torch, tdtype, dev)
    selected = dict(calib)
    selected['terms'] = list(calib.get('terms', ['I1', 'I2', 'I3']) if terms is None else terms)
    atom_order, onsite, couplings, info = terms_from_calib(selected, Om, delta, gamma)
    atom_order = list(map(int, atom_order))
    inv = {a: i for i, a in enumerate(atom_order)}
    pauli_mapped = []
    pauli_labels = []
    if pauli_strings:
        from .observables import pauli_specs_to_mps_order
        pauli_labels = [spec['label'] for spec in pauli_strings]
        pauli_mapped = pauli_specs_to_mps_order(pauli_strings, atom_order)
    if algorithm == 'tebd' and any(b-a != 1 for _, a, _, b, _ in couplings):
        raise ValueError('GPU TEBD requires MPS 最近邻 couplings; select swap-tebd-gpu or CPU TDVP for long-range pairs')
    ts = _initial_tensors(torch, psi0, n, atom_order, tdtype, dev)
    # Conservative per-SVD scratch estimate, plus stored MPS; cannot promise
    # CUDA allocator/workspace exactness but prevents unbounded chi requests.
    if memory_budget_mb is not None:
        if float(memory_budget_mb) <= 0:
            raise ValueError('memory_budget_mb must be positive')
        effective_chi = min(int(chi_max), 1 << min(n//2, 30))
        effective_chi = max(effective_chi, max(a.shape[-1] for a in ts))
        estimate = (2*n + 64) * effective_chi**2 * ts[0].element_size()
        free, _total = torch.cuda.mem_get_info(dev)
        budget = min(float(memory_budget_mb)*1024**2, .85*free)
        if estimate > budget:
            raise ValueError('GPU MPS estimated tensors/SVD workspace %.1f MiB exceeds budget %.1f MiB; reduce chi_max' % (estimate/1024**2, budget/1024**2))
    state = TorchMPS(torch, ts, chi_max, truncerr, svd_driver=svd_driver)
    pair_h = {}
    for c, a, op1, b, op2 in couplings:
        key = (int(a), int(b))
        if key not in pair_h:
            pair_h[key] = torch.zeros((4, 4), dtype=tdtype, device=dev)
        pair_h[key] += complex(c) * torch.kron(ops[op1], ops[op2])
    # Preserve deterministic schedules independently of calibration pair order.
    pair_keys = sorted(pair_h)
    ident = torch.eye(2, dtype=tdtype, device=dev)
    drive_models = {}

    def model_for(segment_id):
        if segment_id in drive_models:
            return drive_models[segment_id]
        on = onsite
        if segment_id >= 0:
            seg = segments[segment_id]
            om = np.asarray(seg.get('Om', np.zeros(n)), float)*np.exp(1j*np.asarray(seg.get('phase', np.zeros(n)), float))
            de = np.asarray(seg.get('delta', np.zeros(n)), float)
            _o, on, _c, _i = terms_from_calib(selected, om, de, gamma)
        hs = torch.stack([complex(dd)*ops['Nhat'] + complex(oo)*ops['P'] + np.conj(oo)*ops['M'] for dd, oo in on])
        if algorithm == 'tebd' and n > 1:
            bonds = []
            for i in range(n-1):
                H = pair_h.get((i, i+1), hs.new_zeros((4, 4)))
                wi = 1. if i == 0 else .5
                wj = 1. if i+1 == n-1 else .5
                bonds.append(H + wi*torch.kron(hs[i], ident) + wj*torch.kron(ident, hs[i+1]))
            result = (None, torch.stack(bonds))
        else:
            result = (hs, torch.stack([pair_h[k] for k in pair_keys]) if pair_keys else None)
        drive_models[segment_id] = result
        return result

    timing = dict(model_build_s=0., gate_build_s=0., gate_apply_s=0., measure_s=0., total_s=0.)
    if profile_timing:
        torch.cuda.synchronize(dev)
    timing['model_build_s'] = time.perf_counter()-started
    N = len(t)
    pops = np.zeros((N, n))
    norms = np.zeros(N)
    entropy = np.zeros(N) if want_entropy else None
    obs_series = ({k: np.zeros((N, n), dtype=complex if k in ('sm', 'sp') else float)
                   for k in ('sx', 'sy', 'sz', 'sm', 'sp', 'phase')} if want_observables else None)
    link_series = np.zeros((N, len(pairs)), complex) if want_observables else None
    pauli_series = (np.zeros((N, len(pauli_strings)), complex)
                    if want_observables and pauli_strings else None)
    xy_last = corr_last = None
    mps_pairs = [(inv[a], inv[b]) for a, b in pairs]

    def record(k):
        nonlocal xy_last, corr_last
        tic = time.perf_counter()
        local_ops = {'Nhat': ops['Nhat']}
        if want_observables:
            local_ops['M'] = ops['M']
        local, links, norm = state.expectations(local_ops, mps_pairs if want_observables else (), (ops['P'], ops['M']))
        p = local['Nhat'].real.cpu().numpy()
        pops[k, atom_order] = p
        norms[k] = float(norm.item())
        if want_observables:
            sm = local['M'].cpu().numpy()
            for key, val in (('sm', sm), ('sp', sm.conj()), ('sx', 2*sm.real), ('sy', 2*sm.imag),
                             ('sz', 1-2*p), ('phase', np.where(abs(sm)>1e-14, np.angle(sm), np.nan))):
                obs_series[key][k, atom_order] = val
            link_series[k] = links.cpu().numpy()
        if pauli_series is not None:
            pauli_series[k] = state.pauli_expectation(pauli_mapped, ops)
        if want_entropy:
            entropy[k] = state.entropy(n//2)
        if k == N-1:
            all_pairs = [(i, j) for i in range(n) for j in range(n)]
            if want_observables and want_xy_corr:
                _, vals, _ = state.expectations({}, all_pairs, (ops['P'], ops['M']))
                xy_last = np.zeros((n, n), complex)
                xy_last[np.ix_(atom_order, atom_order)] = vals.cpu().numpy().reshape(n, n)
            if want_corr:
                _, vals, _ = state.expectations({}, all_pairs, (ops['Nhat'], ops['Nhat']))
                arr = vals.real.cpu().numpy().reshape(n, n)-p[:, None]*p[None, :]
                corr_last = np.zeros((n, n))
                corr_last[np.ix_(atom_order, atom_order)] = arr
        torch.cuda.synchronize(dev)
        timing['measure_s'] += time.perf_counter()-tic

    cache = OrderedDict()
    nsteps = 0
    segments = sorted(segments or [], key=lambda s: float(s['t0']))
    boundaries = [float(s['t0']) for s in segments]
    record(0)
    for it in range(N-1):
        cuts = [float(t[it])] + [b for b in boundaries if t[it] < b < t[it+1]] + [float(t[it+1])]
        for ta, tb in zip(cuts[:-1], cuts[1:]):
            if tb == ta:
                continue
            count = max(1, int(np.ceil((tb-ta)/max_step_ns)))
            h = (tb-ta)/count
            active = max([j for j, b in enumerate(boundaries) if b <= (ta+tb)/2], default=0) if segments else -1
            key = (active, float(h).hex())
            if key not in cache:
                tic = time.perf_counter()
                on_h, cp_h = model_for(active)
                if algorithm == 'tebd' and n > 1:
                    full = _gate(torch, cp_h, h)
                    half = _gate(torch, cp_h, h/2) if order == 2 else full
                    gs = (None, half, full)
                else:
                    og = _gate(torch, on_h, h/2 if order == 2 else h)
                    pg = _gate(torch, cp_h, h/2 if order == 2 else h) if cp_h is not None else None
                    gs = (og, pg, None)
                if profile_timing:
                    torch.cuda.synchronize(dev)
                timing['gate_build_s'] += time.perf_counter()-tic
                cache[key] = gs
                if len(cache) > 32:
                    cache.popitem(last=False)
            gs = cache[key]
            tic = time.perf_counter()
            for sub in range(count):
                if progress_arr is not None and float(progress_arr[1]) != 0:
                    raise RuntimeError('用户取消')
                if algorithm == 'tebd' and n > 1:
                    if order == 2:
                        layers = ((range(0, n-1, 2), gs[1]), (range(1, n-1, 2), gs[2]), (range(0, n-1, 2), gs[1]))
                    else:
                        layers = ((range(n-1), gs[2]),)
                    for bonds, gates in layers:
                        # Disjoint bonds commute.  Visit the next bond nearest
                        # the current orthogonality centre to reduce QR gauge
                        # shifts; this is a generic MPS scheduling optimization.
                        pending = list(bonds)
                        while pending:
                            i = min(pending, key=lambda q: abs(int(q) - state.center))
                            pending.remove(i)
                            state.apply_two(i, gates[i])
                else:
                    for i in range(n):
                        state.apply_one(i, gs[0][i])
                    for q, (a, b) in enumerate(pair_keys):
                        state.apply_pair(a, b, gs[1][q])
                    if order == 2:
                        for q in range(len(pair_keys)-1, -1, -1):
                            a, b = pair_keys[q]
                            state.apply_pair(a, b, gs[1][q])
                        for i in range(n):
                            state.apply_one(i, gs[0][i])
                nsteps += 1
                if progress_arr is not None:
                    progress_arr[0] = (it + (sub+1)/count)/max(1, N-1)
            if profile_timing:
                torch.cuda.synchronize(dev)
            timing['gate_apply_s'] += time.perf_counter()-tic
        record(it+1)
        if log is not None and (it == 0 or it == N-2):
            log('[manybody:%s-gpu] %d/%d t=%.6g ns chi=%d' % (algorithm, it+2, N, t[it+1], state.max_chi))
    torch.cuda.synchronize(dev)
    timing['total_s'] = time.perf_counter()-started
    obs = None
    if want_observables:
        lp, lw = link_loop_observables(link_series, pairs, loops)
        final_lp, final_lw = (loop_observables(xy_last, loops) if xy_last is not None else
                             (lp[-1].tolist(), lw[-1].tolist()) if loops else ([], []))
        obs = dict(obs_series, xy_corr=xy_last, link_corr=link_series, link_pairs=pairs,
                   loop_phase=final_lp, loop_winding=final_lw,
                   loop_phase_series=lp, loop_winding_series=lw)
        if pauli_series is not None:
            obs['pauli_strings'] = pauli_series
            obs['pauli_string_labels'] = pauli_labels
    backend = '%s-torch-cuda' % algorithm
    err = dict(backend=backend, algorithm=algorithm, n_atoms=n, n_t=N, chi_max=int(chi_max),
               truncerr=float(truncerr), max_chi=state.max_chi, max_norm_dev=float(abs(norms-1).max()),
               discarded_weight_sum=float(state.discarded_weight.item()),
               n_internal_steps=nsteps, n_two_site_updates=state.gate_count, n_swaps=state.swap_count,
               max_step_ns=max_step_ns, order=order, device=str(dev), dtype=str(tdtype),
               svd_driver=svd_driver, profile_timing=profile_timing,
               mps_order=atom_order, entropy_cut=n//2 if want_entropy else None,
               nearest_neighbour_only=(algorithm == 'tebd'),
               timings_s=timing)
    return dict(t=t, pops=pops, pops_atom=pops, states=[dict(label='atom%d'%i, atom_id=i) for i in range(n)],
                **({'state': dict(kind='mps', backend='torch', order=atom_order,
                                  tensors=[a.detach().cpu().numpy().copy() for a in state.tensors])}
                   if return_state else {}),
                method='manybody', backend=backend, norms=norms, loss=1-norms,
                S=entropy if want_entropy else None, corr=corr_last, observables=obs, error_report=err,
                diag=err.copy() if diag else None)
