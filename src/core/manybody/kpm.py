# -*- coding: utf-8 -*-
"""KPM / Chebyshev 谱方法 + 随机迹（典型性）——general-H 的有限温 & 态密度（DOS）。

为什么是它（前沿普查后选定）：ED 受 2^N 尺寸限、QMC 对非 stoquastic/复有符号问题、
NQS 被本项目禁用（要蒸馏精确法）。KPM 只需**稀疏 matvec**，**对任意厄米 H 无符号问题**、
资源线性、GPU 友好，给**有限温热力学（E/C/F/χ）**与**态密度 DOS**——正是通用多体平衡态缺的
一维（有限温 + 谱）。文献：Weisse et al. CMP2006; Fehrmann&Struck 2D-Rydberg 热 2025;
arXiv:2503.18413（Rydberg 有限温）。

约定同 `equilibrium`/`assemble`：H 为 2^N 稀疏/稠密厄米（可复、非 stoquastic 均可），能量 MHz。
- 先自动定界并缩放到 x=(E−c)/r ∈[−1,1]（H̃=(H−c)/r）。
- 矩 μ_n=(1/D)Tr(T_n(H̃))：**随机 Z2 向量（Hutchinson 迹）估计**或 `exact`（谱和）。
- 任意 f(H) 的归一迹 (1/D)Tr(f(H̃)) = Σ_n a_n μ_n，a_n 由 `chebinterpolate` 得 f 的 Chebyshev 系数。
- 有限温：f=e^{−β(E)}，E=⟨H⟩=Tr(H e^{−βH})/Tr(e^{−βH})、C、F、χ 全从同一 μ_n 出。
- DOS：Jackson 核抑制 Gibbs 振荡。⚠ 随机迹有统计噪声：给 `nvec`，判据按容差比对精确。
"""
import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import eigsh
from numpy.polynomial.chebyshev import chebinterpolate

__all__ = ['kpm_moments', 'kpm_thermal', 'kpm_dos', 'jackson_kernel']


def _dense(H):
    return np.asarray(H.todense()) if sp.issparse(H) else np.asarray(H)


def _bounds(H):
    """谱界 [emin,emax]（eigsh SA/LA；小维直接 eigvalsh）+ 缩放中心 c、半径 r。"""
    D = H.shape[0]
    if D <= 4096:
        w = np.linalg.eigvalsh(_dense(H))
        emin, emax = float(w[0]), float(w[-1])
    else:
        Hc = H if sp.issparse(H) else sp.csr_matrix(H)
        emin = float(eigsh(Hc, k=1, which='SA', return_eigenvectors=False)[0])
        emax = float(eigsh(Hc, k=1, which='LA', return_eigenvectors=False)[0])
    c = 0.5 * (emin + emax)
    r = 0.5 * (emax - emin) * (1.0 + 1e-3) + 1e-12
    return emin, emax, c, r


def _cheb_coeffs(func, num):
    """f(x)=Σ_{n=0}^{num} a_n T_n(x) 的系数 a_n（DCT，numpy 处理所有归一化约定）。"""
    p = chebinterpolate(func, num)
    a = np.asarray(p.coef if hasattr(p, 'coef') else p, dtype=float)
    if a.size < num + 1:
        a = np.concatenate([a, np.zeros(num + 1 - a.size)])
    return a[:num + 1]


def jackson_kernel(num):
    n = np.arange(num + 1)
    return ((num - n + 1) * np.cos(np.pi * n / (num + 1))
            + np.sin(np.pi * n / (num + 1)) / np.tan(np.pi / (num + 1))) / (num + 1)


def kpm_moments(H, num=300, nvec=48, seed=0, exact=False):
    """→ (mu[num+1], c, r)：mu_n=(1/D)Tr(T_n(H̃))，H̃=(H−c)/r∈[−1,1]。"""
    D = H.shape[0]
    _, _, c, r = _bounds(H)
    mu = np.zeros(num + 1)
    if exact:
        lam = np.linalg.eigvalsh(_dense((H - c * np.eye(D, dtype=_dense(H).dtype)) / r))
        T0 = np.ones_like(lam); T1 = lam.copy()
        mu[0] = T0.mean()
        if num >= 1:
            mu[1] = T1.mean()
        for n in range(2, num + 1):
            T2 = 2 * lam * T1 - T0
            mu[n] = T2.mean()
            T0, T1 = T1, T2
        return mu, c, r
    # 随机 Z2 迹（Hutchinson）：E[⟨z|A|z⟩]=Tr(A)
    Hc = H if sp.issparse(H) else sp.csr_matrix(H)
    I = sp.eye(D, format='csr')
    Htil = (Hc - c * I) * (1.0 / r)
    rng = np.random.default_rng(seed)
    acc = np.zeros(num + 1)
    for _ in range(nvec):
        z = rng.choice(np.array([-1.0, 1.0]), size=D)
        t_prev = z.copy()                                  # T0 z = z
        t_cur = Htil @ z                                   # T1 z
        acc[0] += (t_prev @ z).real
        if num >= 1:
            acc[1] += (t_cur @ z).real
        for n in range(2, num + 1):
            t_next = 2 * (Htil @ t_cur) - t_prev
            acc[n] += (t_next @ z).real
            t_prev, t_cur = t_cur, t_next
    mu = acc / (nvec * D)
    return mu, c, r


def _norm_trace_f(mu, a):
    """(1/D)Tr(f(H̃)) = Σ_n a_n μ_n（μ0=1 对应 f=I）。"""
    return float(np.dot(a, mu))


def kpm_thermal(H, betas, num=300, nvec=48, seed=0, exact=False):
    """有限温：→ dict(E,C,F,Z,chi_n?) per beta（betas 为 (1/MHz,) 数组，标量亦可）。

    E(β)=⟨H⟩、C(β)=β²Var(H)、F=−lnZ/β、Z=Tr e^{−βH}（全由 μ_n + 各 f 的 Chebyshev 系数得）。
    """
    mu, c, r = kpm_moments(H, num, nvec, seed, exact)
    D = H.shape[0]
    betas = np.atleast_1d(np.asarray(betas, float))
    E = np.zeros_like(betas); C = np.zeros_like(betas)
    F = np.zeros_like(betas); Z = np.zeros_like(betas)
    for i, b in enumerate(betas):
        fZ = lambda x, b=b: np.exp(-b * (r * x + c))
        fE = lambda x, b=b: (r * x + c) * np.exp(-b * (r * x + c))
        fE2 = lambda x, b=b: (r * x + c) ** 2 * np.exp(-b * (r * x + c))
        z = _norm_trace_f(mu, _cheb_coeffs(fZ, num)) * D
        e1 = _norm_trace_f(mu, _cheb_coeffs(fE, num)) * D
        e2 = _norm_trace_f(mu, _cheb_coeffs(fE2, num)) * D
        Z[i] = z
        E[i] = e1 / z
        var = max(e2 / z - E[i] ** 2, 0.0)
        C[i] = b ** 2 * var
        F[i] = -np.log(z) / b if b > 0 else float('nan')
    return dict(betas=betas, E=E, C=C, F=F, Z=Z, num=num, nvec=nvec,
                exact=bool(exact), backend='kpm-chebyshev')


def kpm_dos(H, energies, num=300, nvec=48, seed=0, exact=False, use_jackson=True):
    """态密度 ρ(E)（归一 ∫ρ dE = 1）。energies: 目标能量点（MHz）。"""
    mu, c, r = kpm_moments(H, num, nvec, seed, exact)
    x = (np.asarray(energies, float) - c) / r
    xn = np.clip(x, -1 + 1e-9, 1 - 1e-9)
    G = jackson_kernel(num) if use_jackson else np.ones(num + 1)
    # D(x)=1/(π√(1-x²)) [μ0 + 2Σ_{n≥1} G_n μ_n T_n(x)]
    series = mu[0] * np.ones_like(xn)
    if num >= 1:
        T0 = np.ones_like(xn); T1 = xn.copy()
        series = series + 2 * G[1] * mu[1] * T1
        for n in range(2, num + 1):
            T2 = 2 * xn * T1 - T0
            series = series + 2 * G[n] * mu[n] * T2
            T0, T1 = T1, T2
    rho_x = series / (np.pi * np.sqrt(1 - xn ** 2))
    rho = rho_x / r                                  # 变量换到 E：ρ(E)=D(x)/r
    return np.maximum(rho, 0.0)
