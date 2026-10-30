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
from .resources import budget_bytes, require_workspace

__all__ = ['kpm_moments', 'kpm_thermal', 'kpm_dos', 'kpm_moments_gpu', 'jackson_kernel']


def kpm_moments_gpu(H, num=300, nvec=48, seed=0, device='cuda', chunk=32,
                    memory_budget_mb=None, dtype='float64'):
    """CUDA Hutchinson moments with block probes; keeps only two Krylov blocks.

    Bounds are obtained through the existing CPU routine.  The operator is a
    torch sparse COO matrix when possible, so no dense GPU copy is required.
    """
    import torch
    if not torch.cuda.is_available() or str(device).startswith('cpu'):
        raise RuntimeError('CUDA 不可用')
    if dtype not in ('float32','float64'):
        raise ValueError('dtype 必须为 float32/float64')
    if int(num) < 1 or int(nvec) < 1:
        raise ValueError('num 和 nvec 必须为正整数')
    D = int(H.shape[0]); _, _, c, r = _bounds(H)
    Hs = H.tocoo() if sp.issparse(H) else sp.coo_matrix(np.asarray(H))
    itemsize = 8 if dtype == 'float32' else 16
    base_bytes = (len(Hs.data)+D)*(itemsize+16)*3
    budget = budget_bytes(memory_budget_mb)
    require_workspace(base_bytes+D*itemsize*6, memory_budget_mb, 'GPU KPM 算子和单个 probe')
    if budget is not None:
        chunk = min(int(chunk), max(1,(budget-base_bytes)//(D*itemsize*6)))
    dev = torch.device(device)
    rows = torch.as_tensor(Hs.row, device=dev, dtype=torch.long)
    cols = torch.as_tensor(Hs.col, device=dev, dtype=torch.long)
    tdtype = torch.complex64 if dtype == 'float32' else torch.complex128
    vals = torch.as_tensor((Hs.data / r), device=dev, dtype=tdtype)
    A = torch.sparse_coo_tensor(torch.stack([rows, cols]), vals, (D, D)).coalesce()
    diag = torch.arange(D,device=dev)
    shift = torch.sparse_coo_tensor(torch.stack([diag,diag]),
        torch.full((D,), c/r,device=dev,dtype=tdtype),(D,D))
    A = (A-shift).coalesce()
    chunk = max(1, min(int(chunk), int(nvec)))
    gen = torch.Generator(device=dev); gen.manual_seed(int(seed))
    acc = torch.zeros(num + 1, device=dev, dtype=torch.float64)
    done = 0
    while done < int(nvec):
        b = min(chunk, int(nvec) - done)
        z = torch.randint(0, 2, (D, b), device=dev, generator=gen,
                           dtype=torch.int8).to(torch.float64).mul_(2).sub_(1)
        zc = z.to(tdtype); t0 = zc; t1 = torch.sparse.mm(A, zc)
        acc[0] += (t0.conj() * zc).real.sum(dtype=torch.float64)
        if num >= 1: acc[1] += (t1.conj() * z).real.sum(dtype=torch.float64)
        for n in range(2, int(num) + 1):
            t2 = 2 * torch.sparse.mm(A, t1) - t0
            acc[n] += (t2.conj() * z).real.sum(dtype=torch.float64)
            t0, t1 = t1, t2
        done += b
    mu = (acc / (int(nvec) * D)).detach().cpu().numpy()
    return mu, c, r


def _dense(H):
    return np.asarray(H.todense()) if sp.issparse(H) else np.asarray(H)


def _bounds(H):
    """谱界 [emin,emax]（eigsh SA/LA；小维直接 eigvalsh）+ 缩放中心 c、半径 r。"""
    D = H.shape[0]
    if D <= 16:
        w = np.linalg.eigvalsh(_dense(H))
        emin, emax = float(w[0]), float(w[-1])
    else:
        Hc = H if sp.issparse(H) else sp.csr_matrix(H)
        # Fixed starting vectors keep the scaling (and hence seeded moments)
        # reproducible after switching from full spectra to iterative bounds.
        start = np.random.default_rng(0).standard_normal(D)
        emin = float(eigsh(Hc, k=1, which='SA', v0=start.copy(), return_eigenvectors=False)[0])
        emax = float(eigsh(Hc, k=1, which='LA', v0=start.copy(), return_eigenvectors=False)[0])
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
    if int(num) < 1 or int(nvec) < 1:
        raise ValueError('num 和 nvec 必须为正整数')
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


def _thermal_from_moments(H, betas, moments, num):
    mu, c, r = moments; D = H.shape[0]
    betas = np.atleast_1d(np.asarray(betas, float)); E=np.zeros_like(betas); C=np.zeros_like(betas); F=np.zeros_like(betas); Z=np.zeros_like(betas); lnZ=np.zeros_like(betas)
    for i,b in enumerate(betas):
        # Shift by a lower spectral bound; ratios remain unchanged and F is
        # restored analytically. Avoid exp(-beta*E) overflow at low temperature.
        shift = c-r
        fZ=lambda x,b=b: np.exp(-b*(r*x+c-shift))
        fE=lambda x,b=b: (r*x+c)*np.exp(-b*(r*x+c-shift))
        fE2=lambda x,b=b: (r*x+c)**2*np.exp(-b*(r*x+c-shift))
        z=_norm_trace_f(mu,_cheb_coeffs(fZ,num))*D
        e1=_norm_trace_f(mu,_cheb_coeffs(fE,num))*D
        e2=_norm_trace_f(mu,_cheb_coeffs(fE2,num))*D
        if not np.isfinite(z) or z <= 0:
            raise ValueError('KPM 配分函数非正；请增加 moments/probes 或降低 beta')
        E[i]=e1/z; C[i]=b*b*max(e2/z-E[i]**2,0.)
        F[i]=shift-np.log(z)/b if b>0 else np.nan
        lnZ[i]=np.log(z)-b*shift
        with np.errstate(over="ignore",under="ignore"):
            Z[i]=np.exp(lnZ[i])
    return dict(betas=betas,E=E,C=C,F=F,Z=Z,lnZ=lnZ)


def _dos_from_moments(mu, c, r, energies, num, use_jackson=True):
    x=(np.asarray(energies,float)-c)/r; xn=np.clip(x,-1+1e-9,1-1e-9); G=jackson_kernel(num) if use_jackson else np.ones(num+1)
    series=mu[0]*np.ones_like(xn)
    if num>=1:
        t0=np.ones_like(xn); t1=xn.copy(); series += 2*G[1]*mu[1]*t1
        for n in range(2,num+1):
            t2=2*xn*t1-t0; series += 2*G[n]*mu[n]*t2; t0,t1=t1,t2
    return np.maximum(series/(np.pi*np.sqrt(1-xn**2))/r,0.0)


def kpm_thermal(H, betas, num=300, nvec=48, seed=0, exact=False):
    """有限温：→ dict(E,C,F,Z,chi_n?) per beta（betas 为 (1/MHz,) 数组，标量亦可）。

    E(β)=⟨H⟩、C(β)=β²Var(H)、F=−lnZ/β、Z=Tr e^{−βH}（全由 μ_n + 各 f 的 Chebyshev 系数得）。
    """
    mu, c, r = kpm_moments(H, num, nvec, seed, exact)
    result = _thermal_from_moments(H, betas, (mu,c,r), num)
    return dict(result, num=num, nvec=nvec, exact=bool(exact), backend='kpm-chebyshev')


def kpm_dos(H, energies, num=300, nvec=48, seed=0, exact=False, use_jackson=True):
    """态密度 ρ(E)（归一 ∫ρ dE = 1）。energies: 目标能量点（MHz）。"""
    mu, c, r = kpm_moments(H, num, nvec, seed, exact)
    return _dos_from_moments(mu,c,r,energies,num,use_jackson)
