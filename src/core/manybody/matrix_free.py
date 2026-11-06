# -*- coding: utf-8 -*-
"""逐键矩阵自由作用（CPU）与带 Ritz 残差检查的 CUDA 块 Krylov 求解。"""
import numpy as np
import scipy.sparse.linalg as spla
from .couplings import active_terms
import math

__all__ = ['hamiltonian_linear_operator', 'torch_lanczos_ground_state',
           'number_sector_basis', 'check_number_conserving']


def number_sector_basis(n, number):
    """Sorted MSB bit codes with fixed occupation, without enumerating 2**n.

    组合枚举用 numpy 逐位倍增（O(C(n,k))），不再跑 Python 的
    ``sum(1 << i for i in c)`` 生成器；n=24 半填充从数十秒降到亚秒。
    """
    if isinstance(number, (bool, np.bool_)) or int(number) != number:
        raise ValueError('charge_sector 必须是整数')
    n, number = int(n), int(number)
    if not 0 <= n <= 63 or not 0 <= number <= n:
        raise ValueError('charge_sector 越界或 n 不在 int64 位编码范围 [0,63]')
    k = number
    if k == 0:
        return np.zeros(1, np.int64)
    if k == n:
        return np.array([(1 << n) - 1], np.int64)
    codes = np.zeros(1, np.int64)
    counts = np.zeros(1, np.int8)
    for i in range(n):
        new = codes | np.int64(1 << i)
        cnew = counts + 1
        keep = cnew <= k
        codes = np.concatenate((codes, new[keep]))
        counts = np.concatenate((counts, cnew[keep]))
    out = codes[counts == k]
    out.sort()
    return out


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
    """CPU 矩阵自由作用：通道语义来自 ``channels.channel_spec``。

    与 ``assemble.build_H_sparse`` 共用同一份通道声明；这里只负责
    “代码空间 → 基索引”的映射与散加。全空间不再做 searchsorted。
    """
    from .channels import ChannelKernel, channel_spec
    spec = channel_spec(calib, Om, delta, gamma, terms)
    kernel = ChannelKernel(spec, basis_indices)
    return spla.LinearOperator((kernel.dim, kernel.dim), matvec=kernel.matvec,
                               dtype=np.complex128)


def _channel_arrays(basis, n, Om, pairs, use):
    """把每个"一次 XOR 翻转"通道的 (rows, cols, amp) 一次算齐（向量化）。

    消费与 ``assemble.build_H_sparse`` / ``ChannelKernel`` 同一份
    ``channel_spec``；返回 ``(loc, cols, amp, count)``，供 GPU 转移表使用。
    """
    from .channels import channel_spec, selector_mask
    calib = dict(atoms=[{}] * int(n), pairs=list(pairs or []))
    spec = channel_spec(calib, Om, np.zeros(int(n)), np.zeros(int(n)),
                        terms=use)
    basis = np.asarray(basis, np.int64)
    bits = [((basis >> (int(n) - 1 - i)) & 1).astype(np.int8)
            for i in range(int(n))]
    # 位置表：把 searchsorted 的 O(log D) 换成一次查表（n≤26 时表 ≤ 268 MB）
    pos = None
    if int(n) <= 26:
        pos = np.full(1 << int(n), -1, np.int32)
        pos[basis] = np.arange(basis.size, dtype=np.int32)
    out = []
    for selector, mask, amp in spec.channels:
        cols = np.flatnonzero(selector_mask(bits, selector))
        if cols.size == 0:
            continue
        targets = basis[cols] ^ mask
        if pos is not None:
            loc = pos[targets]
            ok = loc >= 0
        else:
            loc = np.searchsorted(basis, targets)
            ok = loc < basis.size
            ok[ok] &= basis[loc[ok]] == targets[ok]
        if ok.any():
            out.append((loc[ok].astype(np.int32),
                        cols[ok].astype(np.int32),
                        complex(amp), int(ok.sum())))
    return out


def _group_channels(chans, budget_bytes, entry_bytes):
    """把通道按内存预算分组：一组一次 index_add_，把 kernel 数从 #通道 降到 #组。"""
    groups, cur, cur_bytes = [], [], 0
    for item in chans:
        need = item[3] * entry_bytes
        if cur and cur_bytes + need > budget_bytes:
            groups.append(cur)
            cur, cur_bytes = [], 0
        cur.append(item)
        cur_bytes += need
    if cur:
        groups.append(cur)
    return groups


def torch_lanczos_ground_state(calib, Om=None, delta=None, gamma=None,
                               terms=None, basis_indices=None, m=48, seed=0,
                               dtype='float64', device='cuda', tol=None,
                               max_restarts=8, return_state=False, memory_budget_mb=None):
    """Block Krylov/Rayleigh-Ritz with full reorthogonalization on CUDA.

    No Hilbert-space dense/CSR Hamiltonian is formed.  Two starting vectors
    retain degenerate low-energy subspaces.  Ritz residuals certify each
    returned energy; ``gap`` is NaN when both lowest pairs have not converged.

    实现要点：
    · 通道表向量化（searchsorted / 位置表），matvec 只做 gather + ``index_add_``；
    · 通道按内存预算分组，每步只发 #组 个 ``index_add_``；
    · 全重正交用一次 GEMM；
    · 全实模型（Ω=0、J/A 全实）走实数算术，显存与带宽减半。
    """
    import torch
    if not torch.cuda.is_available() or str(device).startswith('cpu'):
        raise RuntimeError('CUDA 不可用')
    atoms = calib.get('atoms') or []; n = len(atoms); dim_full = 1 << n
    basis = np.arange(dim_full, dtype=np.int64) if basis_indices is None else np.asarray(basis_indices, np.int64)
    Om = np.zeros(n, complex) if Om is None else np.asarray(Om, complex)
    delta = np.zeros(n) if delta is None else np.asarray(delta, float)
    gamma = np.zeros(n) if gamma is None else np.asarray(gamma, float)
    if np.any(gamma):
        raise ValueError('GPU 基态求解要求厄米 Hamiltonian（gamma=0）')
    if dtype not in ('float64', 'complex128', 'float32', 'complex64'):
        raise ValueError('dtype 必须为 float32/float64/complex64/complex128')
    if int(m) < 2 or int(max_restarts) < 0:
        raise ValueError('m 必须 >=2，max_restarts 必须 >=0')
    use = set(active_terms(calib) if terms is None else terms)

    # 排序基（searchsorted 要求），并记住回排
    order = np.argsort(basis)
    basis_sorted = basis[order]
    if np.any(np.diff(basis_sorted) == 0):
        raise ValueError('basis_indices 不能重复')
    dim = len(basis_sorted)

    bits = [((basis_sorted >> (n - 1 - i)) & 1).astype(float) for i in range(n)]
    from .channels import channel_spec
    spec = channel_spec(calib, Om, delta, gamma, terms)
    diag = spec.diagonal([b.astype(np.int8) for b in bits])

    real_model = (not np.any(Om)) and all(
        abs(complex(p.get('J', 0)).imag) < 1e-15 and abs(complex(p.get('A', 0)).imag) < 1e-15
        for p in calib.get('pairs') or [])

    chans = _channel_arrays(basis_sorted, n, Om, calib.get('pairs') or [], use)
    nnz = sum(c[3] for c in chans)

    nritz = min(2, dim)
    max_basis = min(dim, max(int(m), nritz))
    itemsize = 16 if dtype in ('float64', 'complex128') else 8
    entry_bytes = 4 + 4 + (8 if real_model and itemsize == 16 else itemsize)
    required = dim * itemsize * (2 * max_basis + 8) + nnz * entry_bytes
    if memory_budget_mb is not None and required > float(memory_budget_mb) * 1024**2:
        raise MemoryError('GPU Krylov/transition 保守内存估算超过 memory_budget_mb')

    dev = torch.device(device)
    if real_model and itemsize == 16:
        cdtype = torch.float64
    elif real_model:
        cdtype = torch.float32
    else:
        cdtype = torch.complex128 if itemsize == 16 else torch.complex64
    dt = torch.float64 if itemsize == 16 else torch.float32
    d = torch.as_tensor(diag.real if real_model else diag, device=dev, dtype=cdtype)

    # 转移表：按预算分组，一组一次 index_add_
    budget_bytes = int(0.5 * float(memory_budget_mb) * 1024**2) if memory_budget_mb else 384 * 1024**2
    groups, dev_groups = _group_channels(chans, max(64 * 1024**2, budget_bytes), entry_bytes), []
    for grp in groups:
        rows = torch.as_tensor(np.concatenate([g[0] for g in grp]), device=dev)
        cols = torch.as_tensor(np.concatenate([g[1] for g in grp]), device=dev)
        fill = float(np.real(grp[0][2])) if real_model else grp[0][2]
        amps_np = np.concatenate([np.full(g[3], float(np.real(g[2])) if real_model else g[2],
                                          dtype=np.float64 if real_model else np.complex128)
                                  for g in grp])
        amps = torch.as_tensor(amps_np, device=dev, dtype=cdtype)
        dev_groups.append((rows, cols, amps))

    def action(q):
        z = d * q
        for rows, cols, amps in dev_groups:
            z.index_add_(0, rows, q.index_select(0, cols) * amps)
        return z

    gen = torch.Generator(device=dev); gen.manual_seed(int(seed))

    def _rand():
        if real_model:
            return torch.randn(dim, device=dev, dtype=dt, generator=gen).to(cdtype)
        return (torch.randn(dim, device=dev, dtype=dt, generator=gen).to(cdtype)
                + 1j * torch.randn(dim, device=dev, dtype=dt, generator=gen).to(cdtype))

    tolerance = float(tol if tol is not None else (1e-9 if itemsize == 16 else 2e-5))
    if tolerance <= 0 or not np.isfinite(tolerance):
        raise ValueError('tol 必须为正有限数')
    tiny = 1e-13 if itemsize == 16 else 1e-6

    Qbuf = torch.empty(dim, max_basis, device=dev, dtype=cdtype)
    Hbuf = torch.empty(dim, max_basis, device=dev, dtype=cdtype)
    starts = [_rand() for _ in range(nritz)]

    converged = False
    restarts = 0
    values = residuals = None
    for restart in range(int(max_restarts) + 1):
        restarts = restart
        kQ = 0
        cand = list(starts)
        used = 0                     # 已用作"下一个候选"的 H 向量数
        while kQ < max_basis:
            if cand:
                q = cand.pop(0).clone()
            elif used < kQ:          # ← 关键：把算过但没用过的 Hq 逐个续上
                q = Hbuf[:, used].clone()
                used += 1
            else:
                q = _rand()
            for _ in range(2):                    # 两次修正的经典 GS（一次 GEMM）
                if kQ:
                    Qv = Qbuf[:, :kQ]
                    q = q - Qv @ (Qv.mH @ q)
            norm = torch.linalg.vector_norm(q)
            if float(norm.item()) <= tiny:
                if used >= kQ and not cand:
                    break
                continue
            q = q / norm
            Qbuf[:, kQ] = q
            Hbuf[:, kQ] = action(q)
            kQ += 1

        qmat, hmat = Qbuf[:, :kQ], Hbuf[:, :kQ]
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
        if converged or kQ == dim:
            break
        starts = [ritz[:, i].clone() for i in range(count)]
        starts += [residual_vectors[:, i].clone() for i in range(count)
                   if not bool(flags[i].item())]

    ev = values[:count].detach().cpu().numpy()
    res = residuals.detach().cpu().numpy()
    out = dict(E0=float(ev[0]), gap=float(ev[1] - ev[0]) if converged and count > 1 else float('nan'),
               evals=ev, residuals=res, converged=converged,
               backend='torch-matrixfree-lanczos', dim=dim, krylov_dim=kQ,
               restarts=restart, tolerance=tolerance)
    if not converged:
        out['warning'] = '最低本征对未全部达到残差容限；gap 未报告'
    if return_state:
        vec = ritz[:, 0].detach().cpu().numpy()
        if order.size and not np.array_equal(order, np.arange(order.size)):
            back = np.empty_like(vec)
            back[order] = vec
            vec = back
        out['vec'] = vec.astype(np.complex128) if real_model else vec
    return out
