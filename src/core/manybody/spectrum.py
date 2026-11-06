# -*- coding: utf-8 -*-
"""源态 Lanczos 谱：H、|0⟩、φ = O|0⟩ → 展宽谱 + 未展宽求积 + 收敛诊断。

这是**通用**能力，不知道任何物理模型：

- 输入只要求一个可作用在态矢上的线性算子（``matvec``，通常来自
  ``matrix_free.hamiltonian_linear_operator`` 或稀疏矩阵）与一个源态 ``phi``；
- 输出 ``S(ω) = -1/π Im ⟨φ|(ω+iη-(H-E0))^{-1}|φ⟩``（洛伦兹展宽）、
  Ritz 节点与权重（未展宽，可用于窗口积分）、以及正交性/矩的收敛诊断；
- 归一化约定：``∫S dω = ‖φ‖²``；调用方想按"每站点"归一就自己除。

为什么不用 KPM：KPM 给的是迹（DOS），这里要的是**单个源态**的 Lehmann 谱；
为什么不在引擎里写谱线"识别"：峰值/束缚态判定属于领域解释，留在调用方。
"""
from __future__ import annotations

import time

import numpy as np
import scipy.sparse as sp

__all__ = ['lanczos', 'quadrature', 'lorentz_spectrum', 'source_spectrum']


def _matvec(H):
    if sp.issparse(H):
        return lambda v: H @ v
    if callable(H):
        return H
    arr = np.asarray(H)
    return lambda v: arr @ v


def lanczos(H, phi, n_krylov=200, *, reortho=True, memory_budget_mb=512.0):
    """对 ``H`` 从 ``phi`` 出发做 Lanczos，返回三对角与诊断。

    返回 dict:
      ``alphas`` (k,)、``betas`` (k-1,)、``norm2``（‖φ‖²）、``n``（实际步数）、
      ``ortho_loss``（最后一次与首个基矢的最大重叠，衡量失去正交性）、
      ``converged``（beta 小于 tol 或达到 n_krylov）、``seconds``。

    内存：全重正交需要 k×dim 的基矢。超出 ``memory_budget_mb`` 时**自动降级**
    为不重正交，并在 ``notes`` 里如实说明（不静默改变算法）。
    """
    from .resources import budget_bytes, require_workspace
    v = np.asarray(phi, dtype=complex).reshape(-1)
    dim = v.size
    if dim == 0:
        raise ValueError('源态为空')
    norm = float(np.linalg.norm(v))
    if not np.isfinite(norm) or norm <= 0:
        raise ValueError('源态范数为零/非有限——检查算符作用结果')
    k = int(n_krylov)
    if k < 1:
        raise ValueError('n_krylov 须为正')
    k = min(k, dim)
    note = ''
    if reortho:
        need = dim * k * 16 * 2
        limit = budget_bytes(memory_budget_mb)
        if limit is not None and need > limit:
            reortho = False
            note = ('全重正交工作区约 %.0f MiB > 预算 %.0f MiB ⇒ 降级为不重正交'
                    % (need / 2 ** 20, float(memory_budget_mb)))
        else:
            require_workspace(need, memory_budget_mb, 'Lanczos 全重正交')
    t0 = time.perf_counter()
    mv = _matvec(H)
    q = v / norm
    q_prev = np.zeros_like(q)
    alphas, betas = [], []
    basis = [] if reortho else None
    n = 0
    converged = False
    beta = 0.0
    for j in range(k):
        if basis is not None:
            basis.append(q)
        w = np.asarray(mv(q), dtype=complex)
        alpha = float(np.vdot(q, w).real)
        w = w - alpha * q - (beta * q_prev if j else 0.0)
        if basis is not None:
            for b in basis:                    # 全重正交（两次即可，这里一次够用）
                w = w - np.vdot(b, w) * b
            for b in basis:
                w = w - np.vdot(b, w) * b
        beta = float(np.linalg.norm(w))
        alphas.append(alpha)
        n = j + 1
        if beta <= 1e-12 or n >= dim:
            converged = True
            break
        if j + 1 < k:
            betas.append(beta)
        q_prev, q = q, w / beta
    ortho = 0.0
    if basis:
        ortho = float(abs(np.vdot(basis[0], q)))
    return {'alphas': np.asarray(alphas, float),
            'betas': np.asarray(betas, float),
            'norm2': norm ** 2, 'n': int(n), 'converged': bool(converged),
            'ortho_loss': ortho, 'reortho': bool(reortho), 'note': note,
            'seconds': round(time.perf_counter() - t0, 3)}


def quadrature(alphas, betas, norm2, shift=0.0):
    """三对角 → 未展宽求积节点/权重（``Σ w = ‖φ‖²``）。

    ``shift`` 是参考能量（通常基态 E0）：返回的是**相对**能量 ``E_n - shift``，
    与 ``lorentz_spectrum`` 的横轴一致。
    """
    alphas = np.asarray(alphas, float)
    betas = np.asarray(betas, float)
    if alphas.size == 0:
        return np.zeros(0), np.zeros(0)
    if betas.size:
        evals, evecs = np.linalg.eigh(np.diag(alphas) + np.diag(betas, 1) + np.diag(betas, -1))
    else:
        evals, evecs = alphas.copy(), np.ones((1, 1))
    weights = float(norm2) * np.abs(evecs[0, :]) ** 2
    return evals - float(shift), weights


def lorentz_spectrum(alphas, betas, norm2, omega, eta, shift=0.0):
    """连分式求 ``S(ω) = -1/π Im ⟨φ|(ω+shift+iη-H)^{-1}|φ⟩``。"""
    alphas = np.asarray(alphas, float)
    betas = np.asarray(betas, float)
    omega = np.asarray(omega, float)
    if alphas.size == 0:
        return np.zeros_like(omega)
    z = omega + float(shift) + 1j * float(eta)
    g = z - alphas[-1]
    for j in range(alphas.size - 2, -1, -1):
        g = z - alphas[j] - (betas[j] ** 2) / g
    return -float(norm2) * np.imag(1.0 / g) / np.pi


def source_spectrum(H, phi, *, shift=0.0, eta=0.05, n_krylov=None,
                    omega_min=None, omega_max=None, n_omega=800,
                    reortho=True, memory_budget_mb=512.0):
    """一步到位：Lanczos → 求积 + 展宽谱 + 诊断。

    ``shift`` 通常是基态能量 E0（于是横轴是相对能量）；``eta`` 是洛伦兹半宽。
    频率窗缺省由 Ritz 节点范围外扩 10% 决定；**窗是显示参数，不是物理结论**。
    """
    from .resources import require_workspace
    phi = np.asarray(phi, dtype=complex).reshape(-1)
    k = int(n_krylov) if n_krylov else min(phi.size, 200)
    lz = lanczos(H, phi, k, reortho=reortho, memory_budget_mb=memory_budget_mb)
    nodes, weights = quadrature(lz['alphas'], lz['betas'], lz['norm2'], shift)
    if omega_min is None or omega_max is None:
        if nodes.size:
            lo, hi = float(nodes.min()), float(nodes.max())
        else:
            lo, hi = 0.0, 1.0
        pad = max(0.05 * (hi - lo), 4.0 * float(eta))
        omega_min = lo - pad if omega_min is None else float(omega_min)
        omega_max = hi + pad if omega_max is None else float(omega_max)
    omega = np.linspace(float(omega_min), float(omega_max), int(n_omega))
    eta = float(eta)
    require_workspace(omega.size * 32, memory_budget_mb, '展宽谱网格')
    spectral = lorentz_spectrum(lz['alphas'], lz['betas'], lz['norm2'], omega, eta, shift)
    # 矩校验：展宽谱在整窗上的积分应接近 ‖φ‖²（窗够宽时）
    _trapz = getattr(np, 'trapezoid', None) or getattr(np, 'trapz')
    integral = float(_trapz(spectral, omega)) if omega.size > 1 else 0.0
    return {
        'omega': omega, 'spectral': spectral,
        'evals': nodes, 'spectrum_weights': weights,
        'source_norm': float(lz['norm2']), 'eta': eta, 'shift': float(shift),
        'krylov': {'n': lz['n'], 'n_requested': k, 'converged': lz['converged'],
                   'ortho_loss': lz['ortho_loss'], 'reortho': lz['reortho'],
                   'note': lz['note'], 'seconds': lz['seconds']},
        'error_report': {'source_norm': float(lz['norm2']),
                         'spectral_integral': integral,
                         'integral_ratio': (integral / lz['norm2']) if lz['norm2'] else None,
                         'eta_mhz': eta,
                         'krylov_n': lz['n'],
                         'ortho_loss': lz['ortho_loss'],
                         'reortho': lz['reortho']},
    }
