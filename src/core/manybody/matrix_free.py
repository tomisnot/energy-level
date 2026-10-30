# -*- coding: utf-8 -*-
"""逐键矩阵自由作用（CPU）与带 Ritz 残差检查的 CUDA 块 Krylov 求解。"""
import numpy as np
import scipy.sparse.linalg as spla
import itertools
import math

__all__ = ['hamiltonian_linear_operator', 'torch_lanczos_ground_state',
           'number_sector_basis', 'check_number_conserving']


def number_sector_basis(n, number):
    """Sorted MSB bit codes with fixed occupation, without enumerating 2**n."""
    if isinstance(number, (bool, np.bool_)) or int(number) != number:
        raise ValueError('charge_sector 必须是整数')
    n, number = int(n), int(number)
    if not 0 <= n <= 63 or not 0 <= number <= n:
        raise ValueError('charge_sector 越界或 n 不在 int64 位编码范围 [0,63]')
    states = np.fromiter((sum(1 << i for i in c)
                          for c in itertools.combinations(range(n), number)),
                         dtype=np.int64, count=math.comb(n, number))
    states.sort()
    return states


def check_number_conserving(calib, Om=None, terms=None):
    """Refuse dropping transitions out of a declared fixed-number space."""
    use = set(calib.get('terms', ()) if terms is None else terms)
    if Om is not None and np.any(np.asarray(Om, complex) != 0):
        raise ValueError('固定 charge_sector 要求 Omega=0')
    if 'I3' in use and any(complex(p.get('A', 0)) != 0
                            for p in calib.get('pairs', ())):
        raise ValueError('固定 charge_sector 不支持 I3 双激发项')


def hamiltonian_linear_operator(calib, Om=None, delta=None, gamma=None,
                                terms=None, basis_indices=None):
    atoms = calib.get('atoms') or []
    n = len(atoms); dim_full = 1 << n
    Om = np.zeros(n, complex) if Om is None else np.asarray(Om, complex)
    delta = np.zeros(n) if delta is None else np.asarray(delta, float)
    gamma = np.zeros(n) if gamma is None else np.asarray(gamma, float)
    use = set(calib.get('terms') if terms is None else terms)
    basis = np.arange(dim_full, dtype=np.int64) if basis_indices is None else np.asarray(basis_indices, np.int64)
    if (basis.ndim != 1 or len(basis) == 0 or np.any(basis < 0)
            or np.any(basis >= dim_full)):
        raise ValueError('basis_indices 必须是非空有效位编码')
    permutation = np.argsort(basis)
    sorted_basis = basis[permutation]
    if np.any(np.diff(sorted_basis) == 0):
        raise ValueError('basis_indices 不能重复')
    bits = [((basis >> (n - 1 - i)) & 1).astype(np.int8) for i in range(n)]
    diag = np.zeros(len(basis), complex)
    for i in range(n):
        diag += (delta[i] - 0.5j * gamma[i] * 1e-3 / (2 * np.pi * 1e-3)) * bits[i]
    if 'I1' in use:
        for p in calib.get('pairs') or []:
            diag += p.get('V', 0.0) * bits[p['i']] * bits[p['j']]
    masks = [np.int64(1) << (n - 1 - i) for i in range(n)]

    def scatter(y, v, selected, mask, amplitude):
        """One XOR transition at a time; O(D) scratch, no bond*D table."""
        columns = np.flatnonzero(selected)
        targets = basis[columns] ^ mask
        locations = np.searchsorted(sorted_basis, targets)
        valid = locations < len(sorted_basis)
        valid[valid] &= sorted_basis[locations[valid]] == targets[valid]
        # XOR is injective within one channel, so indexed += has no repeats.
        y[permutation[locations[valid]]] += amplitude * v[columns[valid]]

    def matvec(v):
        v = np.asarray(v, complex).reshape(-1); y = diag * v
        for i, om in enumerate(Om):
            if om == 0: continue
            scatter(y, v, bits[i] == 0, masks[i], om / 2.)
            scatter(y, v, bits[i] == 1, masks[i], np.conj(om) / 2.)
        for p in calib.get('pairs') or []:
            i, j = int(p['i']), int(p['j']); mi, mj = masks[i], masks[j]
            bi, bj = bits[i], bits[j]
            if 'I2' in use and p.get('J', 0) != 0:
                scatter(y, v, (bi == 1) & (bj == 0), mi ^ mj, np.conj(p['J']))
                scatter(y, v, (bi == 0) & (bj == 1), mi ^ mj, p['J'])
            if 'I3' in use and p.get('A', 0) != 0:
                scatter(y, v, (bi == 0) & (bj == 0), mi ^ mj, p['A'])
                scatter(y, v, (bi == 1) & (bj == 1), mi ^ mj, np.conj(p['A']))
        return y

    return spla.LinearOperator((len(basis), len(basis)), matvec=matvec,
                               dtype=np.complex128)


def torch_lanczos_ground_state(calib, Om=None, delta=None, gamma=None,
                               terms=None, basis_indices=None, m=48, seed=0,
                               dtype='float64', device='cuda', tol=None,
                               max_restarts=8, return_state=False, memory_budget_mb=None):
    """Block Krylov/Rayleigh-Ritz with full reorthogonalization on CUDA.

    No Hilbert-space dense/CSR Hamiltonian is formed.  Two starting vectors
    retain degenerate low-energy subspaces.  Ritz residuals certify each
    returned energy; ``gap`` is NaN when both lowest pairs have not converged.
    The optional state and the Krylov basis have explicit memory costs.
    """
    import torch
    if not torch.cuda.is_available() or str(device).startswith('cpu'):
        raise RuntimeError('CUDA 不可用')
    atoms = calib.get('atoms') or []; n = len(atoms); dim_full = 1 << n
    basis = np.arange(dim_full, dtype=np.int64) if basis_indices is None else np.asarray(basis_indices, np.int64)
    pos = {int(s): i for i, s in enumerate(basis)}; dim = len(basis)
    Om = np.zeros(n, complex) if Om is None else np.asarray(Om, complex)
    delta = np.zeros(n) if delta is None else np.asarray(delta, float)
    gamma = np.zeros(n) if gamma is None else np.asarray(gamma, float)
    if np.any(gamma):
        raise ValueError('GPU 基态求解要求厄米 Hamiltonian（gamma=0）')
    if dtype not in ('float64', 'complex128', 'float32', 'complex64'):
        raise ValueError('dtype 必须为 float32/float64/complex64/complex128')
    if int(m) < 2 or int(max_restarts) < 0:
        raise ValueError('m 必须 >=2，max_restarts 必须 >=0')
    nritz = min(2, dim)
    max_basis = min(dim, max(int(m), nritz))
    itemsize = 16 if dtype in ('float64', 'complex128') else 8
    if memory_budget_mb is not None:
        required = dim * itemsize * (2 * max_basis + 8)
        # Each two-site channel can retain at most dim source/target indices.
        channels = len(calib.get('pairs') or []) * 2 + n * 2
        required += dim * channels * 16
        if required > float(memory_budget_mb) * 1024**2:
            raise MemoryError('GPU Krylov/transition 保守内存估算超过 memory_budget_mb')
    use = set(calib.get('terms') if terms is None else terms)
    bits = [((basis >> (n - 1 - i)) & 1).astype(float) for i in range(n)]
    diag = np.zeros(dim, complex)
    for i in range(n):
        diag += (delta[i] - 0.5j * gamma[i] * 1e-3 / (2 * np.pi * 1e-3)) * bits[i]
    if 'I1' in use:
        for p in calib.get('pairs') or []:
            diag += p.get('V', 0.0) * bits[p['i']] * bits[p['j']]
    transitions = []
    def add_transition(src, mask, amp):
        tgt = src ^ np.int64(mask)
        rows = []; cols = []
        for j, s in enumerate(src):
            i = pos.get(int(tgt[j]))
            if i is not None:
                rows.append(i); cols.append(pos[int(s)])
        if rows: transitions.append((np.asarray(rows, np.int64), np.asarray(cols, np.int64), complex(amp)))
    for i, om in enumerate(Om):
        if om != 0:
            mask = np.int64(1) << (n - 1 - i)
            add_transition(basis[bits[i] == 0], mask, om / 2.0)
            add_transition(basis[bits[i] == 1], mask, np.conj(om) / 2.0)
    for p in calib.get('pairs') or []:
        i, j = int(p['i']), int(p['j']); mi=np.int64(1)<<(n-1-i); mj=np.int64(1)<<(n-1-j)
        bi, bj = bits[i], bits[j]
        if 'I2' in use and p.get('J', 0) != 0:
            add_transition(basis[(bi==1)&(bj==0)], mi|mj, np.conj(p['J']))
            add_transition(basis[(bi==0)&(bj==1)], mi|mj, p['J'])
        if 'I3' in use and p.get('A', 0) != 0:
            add_transition(basis[(bi==0)&(bj==0)], mi|mj, p['A'])
            add_transition(basis[(bi==1)&(bj==1)], mi|mj, np.conj(p['A']))
    dev = torch.device(device)
    cdtype = torch.complex128 if str(dtype) in ('float64','complex128') else torch.complex64
    dt = torch.float64 if cdtype == torch.complex128 else torch.float32
    d = torch.as_tensor(diag, device=dev, dtype=cdtype)
    trans = [(torch.as_tensor(r, device=dev), torch.as_tensor(c, device=dev),
              torch.as_tensor(a, device=dev, dtype=cdtype)) for r,c,a in transitions]
    gen = torch.Generator(device=dev); gen.manual_seed(int(seed))
    def action(q):
        z = d * q
        for r,c,a in trans:
            z.index_add_(0, r, a * q[c])
        return z

    tolerance = float(tol if tol is not None else (1e-9 if itemsize == 16 else 2e-5))
    if tolerance <= 0 or not np.isfinite(tolerance):
        raise ValueError('tol 必须为正有限数')
    tiny = 1e-13 if itemsize == 16 else 1e-6
    starts = [torch.randn(dim, device=dev, dtype=dt, generator=gen).to(cdtype)
              + 1j * torch.randn(dim, device=dev, dtype=dt, generator=gen).to(cdtype)
              for _ in range(nritz)]
    converged = False
    for restart in range(int(max_restarts) + 1):
        Q, HQ = [], []
        candidates = list(starts)
        expanded = 0
        while len(Q) < max_basis:
            if not candidates:
                if expanded < len(HQ):
                    candidates.append(HQ[expanded].clone())
                    expanded += 1
                else:
                    # Fill an invariant block's orthogonal complement so
                    # multiplicities are not lost after exact breakdown.
                    candidates.append(torch.randn(dim, device=dev, dtype=dt,
                                                  generator=gen).to(cdtype))
            q = candidates.pop(0).clone()
            for _ in range(2):
                for v in Q:
                    q -= torch.vdot(v, q) * v
            norm = torch.linalg.vector_norm(q)
            if float(norm.item()) <= tiny:
                if expanded >= len(HQ) and not candidates:
                    break
                continue
            q /= norm
            Q.append(q)
            HQ.append(action(q))
        qmat, hmat = torch.stack(Q, 1), torch.stack(HQ, 1)
        projected = qmat.mH @ hmat
        projected = (projected + projected.mH) / 2
        values, vectors = torch.linalg.eigh(projected)
        count = min(nritz, len(values))
        ritz = qmat @ vectors[:, :count]
        hritz = hmat @ vectors[:, :count]
        residual_vectors = hritz - ritz * values[:count][None, :]
        residuals = torch.linalg.vector_norm(residual_vectors, dim=0)
        threshold = tolerance * torch.maximum(torch.ones_like(values[:count]), values[:count].abs())
        flags = residuals <= threshold
        converged = count == nritz and bool(torch.all(flags).item())
        if converged or len(Q) == dim:
            break
        starts = [ritz[:, i].clone() for i in range(count)]
        starts += [residual_vectors[:, i].clone() for i in range(count)
                   if not bool(flags[i].item())]
    ev = values[:count].detach().cpu().numpy()
    res = residuals.detach().cpu().numpy()
    out = dict(E0=float(ev[0]), gap=float(ev[1]-ev[0]) if converged and count > 1 else float('nan'),
               evals=ev, residuals=res, converged=converged,
               backend='torch-matrixfree-lanczos', dim=dim, krylov_dim=len(Q),
               restarts=restart, tolerance=tolerance)
    if not converged:
        out['warning'] = '最低本征对未全部达到残差容限；gap 未报告'
    if return_state:
        out['vec'] = ritz[:, 0].detach().cpu().numpy()
    return out
