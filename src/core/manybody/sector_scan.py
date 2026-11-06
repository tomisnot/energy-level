# -*- coding: utf-8 -*-
"""逐扇区扫描：不知道电荷扇区时，精确解该付多少代价。

这是**编排层**，不引入新的数值算法：每个扇区直接复用

    number_sector_basis(n, k)                  —— 组合基（O(C(n,k))）
    hamiltonian_linear_operator(..., basis_indices) —— CPU 矩阵自由算子
    torch_lanczos_ground_state(..., basis_indices)  —— GPU 块 Krylov（带 Ritz 残差认证）

然后取 E0 最小的扇区。它对任意**数守恒**模型通用（不限于 XY），也不需要外部先验
k；这正是"不给真值扇区"的公平口径：付的是 Σ_k C(n,k) = 2^n 的总基，而不是
偷偷用一个扇区的价钱。

约定：
· 只对 `charge_sector` 有意义的模型使用（Ω=0、无 I3 双激发）；否则
  `check_number_conserving` 会拒绝，而不是静默扫描。
· 各扇区独立求解，结果里保留每个扇区的 E0 / 维度 / 残差 / 耗时，便于审计。
· 扫描是**可并行**的（扇区互不依赖），本函数默认串行；调用方可用线程/进程包一层。
"""
from __future__ import annotations

import time

import numpy as np

from .cost_mb import effective_dim_from_sector
from .equilibrium import ground_state
from .matrix_free import (check_number_conserving, hamiltonian_linear_operator,
                          number_sector_basis, torch_lanczos_ground_state)

__all__ = ['scan_ground_sectors']


def _solve_one(calib, delta, Om, gamma, terms, idx, *, backend, m, tol, dtype,
               device, memory_budget_mb, dense_max, seed):
    """单扇区：小维走稠密 eigh，大维走 GPU 块 Krylov（不可用则 CPU eigsh）。"""
    dim = len(idx)
    # 极小扇区：直接稠密（几十维，列化成本可忽略）
    if backend in ('auto', 'ed') and dim <= 64:
        H = hamiltonian_linear_operator(calib, Om, delta, gamma, terms,
                                        basis_indices=idx)
        Hd = np.column_stack([H @ np.eye(1, dim, j).ravel() for j in range(dim)])
        w, V = np.linalg.eigh(Hd)
        return dict(E0=float(w[0]), gap=float(w[1] - w[0]) if dim > 1 else 0.0,
                    vec=V[:, 0], residuals=np.zeros(1), converged=True,
                    backend='eq-eigh')
    # 其余一律 GPU 块 Krylov（对小维也比"把 LinearOperator 列化"快一个量级）
    if backend in ('auto', 'gpu-lanczos') and device and not str(device).startswith('cpu'):
        try:
            return torch_lanczos_ground_state(
                calib, Om, delta, gamma, terms, basis_indices=idx, m=m, seed=seed,
                dtype=dtype, device=device, tol=tol,
                memory_budget_mb=memory_budget_mb, return_state=False)
        except RuntimeError:
            if backend == 'gpu-lanczos':
                raise
    H = hamiltonian_linear_operator(calib, Om, delta, gamma, terms, basis_indices=idx)
    return ground_state(H, dense_max=int(dense_max), k=2, seed=seed)


def scan_ground_sectors(calib, delta=None, Om=None, gamma=None, terms=None, *,
                        sectors=None, backend='auto', m=48, tol=None,
                        dtype='float64', device='cuda', memory_budget_mb=None,
                        dense_max=2048, seed=0, progress=None):
    """对 k=0…N（或给定的 sectors）逐个求基态，返回能量最低的扇区。

    返回
    ----
    dict:
        E0            最低能量
        sector        取得最低能量的扇区
        per_sector    [{sector, dim, E0, gap, residual, converged, backend, seconds}, ...]
        total_seconds 总耗时
        sectors_run   实际跑过的扇区
        converged     所有扇区都达到收敛判据
    """
    atoms = calib.get('atoms') or []
    n = len(atoms)
    if n <= 0:
        raise ValueError('calib 需要 atoms')
    delta = np.zeros(n) if delta is None else np.asarray(delta, float)
    Om = np.zeros(n, complex) if Om is None else np.asarray(Om, complex)
    gamma = np.zeros(n) if gamma is None else np.asarray(gamma, float)
    if np.any(gamma):
        raise ValueError('扫描基态要求厄米 Hamiltonian（gamma=0）')
    check_number_conserving(calib, Om, terms)
    ks = list(range(n + 1)) if sectors is None else [int(k) for k in sectors]
    for k in ks:
        if not 0 <= k <= n:
            raise ValueError('sector 必须在 [0, N]')

    # CUDA 上下文/句柄的首次初始化要数秒，不能算进算法耗时（网络侧同样不计）
    warmup = 0.0
    if device and not str(device).startswith('cpu'):
        try:
            import torch
            if torch.cuda.is_available():
                t_w = time.perf_counter()
                torch.zeros(1, device=device)
                torch.cuda.synchronize()
                warmup = time.perf_counter() - t_w
        except Exception:      # noqa: BLE001 - 没有 CUDA 时静默走 CPU
            pass

    t_start = time.perf_counter()
    rows = []
    best = None
    for k in ks:
        t0 = time.perf_counter()
        idx = number_sector_basis(n, k)
        r = _solve_one(calib, delta, Om, gamma, terms, idx, backend=backend, m=m,
                       tol=tol, dtype=dtype, device=device,
                       memory_budget_mb=memory_budget_mb, dense_max=dense_max,
                       seed=seed)
        dt = time.perf_counter() - t0
        res = np.asarray(r.get('residuals', [0.0])).reshape(-1)
        row = dict(sector=k, dim=len(idx), E0=float(r['E0']),
                   gap=float(r.get('gap', float('nan'))),
                   residual=float(res[0]) if res.size else 0.0,
                   converged=bool(r.get('converged', True)),
                   backend=str(r.get('backend', '')), seconds=dt)
        rows.append(row)
        if best is None or row['E0'] < best['E0']:
            best = dict(row)
        if progress is not None:
            progress(row)
    total = time.perf_counter() - t_start
    return dict(E0=best['E0'], sector=best['sector'], per_sector=rows,
                total_seconds=total, warmup_seconds=warmup, sectors_run=ks,
                converged=all(r['converged'] for r in rows),
                dimension_sum=int(sum(r['dim'] for r in rows)),
                full_dimension=int(1 << n))
