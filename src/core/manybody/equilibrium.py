# -*- coding: utf-8 -*-
"""多体**平衡态**求解器（M-EQ）：对任意 2^N 稀疏/稠密 H（MHz，复、厄米），
给出基态（Lanczos / 虚时幂法）与精确有限温热力学（配分函数）。

角色：`equilib` 家族，独立于含时 `engine_ed`——只吃 `assemble.build_H_sparse`
产出的 H（或任何同基序矩阵），产基态能量/矢、能隙、逐原子/关联/纠缠观测量，以及
Gibbs 态热力学（E(β)、F(β)、C(β)、⟨Ô⟩_β、χ_β）。**单位**：H/能量=MHz；
β=1/MHz（无量纲指数 = β·H），温度 T=1/β（MHz）。观测量复用 `observables`（位序
bit i ↔ 原子 i，MSB，bit=1↔|r⟩）。

这些求解器**本身是被测对象**——`checks/equibench_*` 用独立暴力 ED / 解析闭式 /
自由费米子 / 二聚体 / 2D 海森堡（Lieb-Mattis 单重态）来钉它们的正确性
（见 docs/标杆判据库-平衡态多体算法正确性实例.md）。
"""
import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import eigsh, expm_multiply

from .observables import bit_matrix, entropy_at_cut

__all__ = ['ground_state', 'ground_state_imagtime', 'thermal_from_h',
           'gs_observables', 'detect_gpu', 'batched_ground_state']


def _to_array_dim(H):
    n = H.shape[0]
    d = int(np.log2(n))
    if 1 << d != n:
        raise ValueError('H 维度 %d 非 2^N' % n)
    return d


def ground_state(H, dense_max=2048, seed=0, k=2, v0=None):
    """基态：小维稠密 `eigh`（精确），大维 `eigsh`(Lanczos, which='SA')。

    返回 dict(E0, gap, vec, backend)。E0=最小本征值(MHz)，gap=第一激发−基态。
    Lanczos 用固定 v0（种子）保确定性；退化基态时 vec 任取一线性组合（观测量按
    该 vec 计，不做子空间平均——调用方若基态简并需自行处理）。
    """
    dim = H.shape[0]
    is_lo = isinstance(H, sp.linalg.LinearOperator)   # LinearOperator 只能稀疏路
    if dim <= dense_max and not is_lo:
        Hd = np.asarray(H.todense()) if sp.issparse(H) else np.asarray(H)
        w, V = np.linalg.eigh(Hd)
        return dict(E0=float(w[0]), gap=float(w[1] - w[0]) if len(w) > 1 else 0.0,
                    vec=V[:, 0], backend='eq-eigh', evals=w[:k])
    rng = np.random.default_rng(seed)
    if v0 is None:
        v0 = rng.standard_normal(dim) + 1j * rng.standard_normal(dim)
    if is_lo:
        Hd = H
    else:
        Hd = H if sp.issparse(H) else sp.csr_matrix(H)
    w, V = eigsh(Hd, k=min(k, dim - 1), which='SA', v0=v0, maxiter=dim * 20,
                 tol=0)
    order = np.argsort(w)
    w, V = w[order], V[:, order]
    return dict(E0=float(w[0]),
                gap=float(w[1] - w[0]) if len(w) > 1 else 0.0,
                vec=V[:, 0], backend='eq-lanczos', evals=w)


def ground_state_imagtime(H, beta_step=0.02, steps=4000, seed=0,
                          conv=1e-10, dense_start=None):
    """虚时幂法基态：|ψ⟩ ← e^{−τ H}|ψ⟩ 反复 + 归一，Rayleigh 商给 E₀。

    τ=beta_step(1/MHz)，从随机态出发（种子确定性）。适合 gap>0 的实例；收敛到
    与 `ground_state` 一致（迭代精度量级）。返回 dict(E0, vec, converged, E_prev)。
    """
    dim = H.shape[0]
    Hd = H if sp.issparse(H) else sp.csr_matrix(H)
    rng = np.random.default_rng(seed)
    v = rng.standard_normal(dim) + 1j * rng.standard_normal(dim)
    v /= np.linalg.norm(v)
    # 减去对角均值以稳定 e^{−τH}（不改变本征矢、只移能标）
    diagmean = float(np.real(np.asarray(Hd.diagonal()).mean()))
    Hd_shift = Hd - diagmean * sp.eye(dim, format='csr') if sp.issparse(Hd) \
        else Hd - diagmean * np.eye(dim)
    E_prev = None
    converged = False
    for it in range(steps):
        v = expm_multiply(-beta_step * Hd_shift, v)
        nv = np.linalg.norm(v)
        if nv == 0:
            break
        v = v / nv
        E = float(np.real(v.conj() @ (Hd @ v)))
        if E_prev is not None and abs(E - E_prev) < conv * max(1.0, abs(E)):
            converged = True
            break
        E_prev = E
    return dict(E0=E_prev, vec=v, converged=converged, backend='eq-imagtime')


def gs_observables(H, n_atoms=None):
    """基态观测量：E0、逐原子 ⟨n_i⟩、末态关联 C_ij、中切纠缠熵 S、总Sz²（自旋判据）。"""
    d = _to_array_dim(H)
    n = n_atoms if n_atoms is not None else d
    gs = ground_state(H)
    psi = gs['vec']
    dim = 1 << n
    bits = bit_matrix(n, dim)
    p = np.abs(psi) ** 2
    out = dict(E0=gs['E0'], gap=gs['gap'], backend=gs['backend'],
               n_atom=(bits @ p), corr=((bits * p) @ bits.T
                                        - np.outer(bits @ p, bits @ p)),
               S=entropy_at_cut(psi, n, n // 2) if n >= 2 else 0.0)
    return out


def thermal_from_h(H, beta, n_atoms=None):
    """精确有限温 Gibbs：由谱分解 e^{−βH}，给 E(β)、F(β)、C(β)、⟨n_i⟩_β、χ_β(Σn)。

    β=1/MHz（β→∞ 趋于基态）。小维稠密 `eigh`（精确，O(2^N)）。
    返回 dict(E, F, C, Z, n_atom(dim=n), chi_n, evals)。
    """
    d = _to_array_dim(H)
    n = n_atoms if n_atoms is not None else d
    dim = 1 << n
    Hd = np.asarray(H.todense()) if sp.issparse(H) else np.asarray(H)
    evals, evecs = np.linalg.eigh(Hd)
    e0 = evals.min()
    w = np.exp(-beta * (evals - e0))
    Z_shift = w.sum()                            # Σ e^{−β(e−e0)}，数值稳定
    lnZ = float(np.log(Z_shift) - beta * e0)     # 物理 lnZ = ln Σ e^{−βe}
    pm = w / Z_shift                             # 各本征态占据
    E = float(e0 + pm @ (evals - e0))
    F = float(-lnZ / beta)
    var_e = float(pm @ (evals - e0) ** 2 - (pm @ (evals - e0)) ** 2)
    C = float(beta ** 2 * var_e)
    # 组态对角占据 P_s = Σ_m pm_m |evecs[s,m]|²
    Ps = (np.abs(evecs) ** 2) @ pm
    bits = bit_matrix(n, dim)
    n_atom = bits @ Ps
    m_tot = bits.sum(axis=0)                     # 总激发数算符（对角）
    mean_m = float(m_tot @ Ps)
    var_m = float((m_tot ** 2) @ Ps - mean_m ** 2)
    chi_n = float(beta * var_m)
    return dict(E=E, F=F, C=C, lnZ=lnZ, n_atom=n_atom, chi_n=chi_n,
                evals=evals)


# =====================================================================
# GPU 批量 ED：参数网格上成千个独立 H 的基态一次批量化求解。
#   蒸馏造标签 = “很多点 × 各自一次本征”→ 大批量本征是 GPU 甜点。
#   优先 CuPy（numpy-API 对齐），其次 torch-cuda；均无则 numpy 批量化回退。
# =====================================================================
def detect_gpu():
    """→ 'cupy' | 'torch' | None（轻量探测，不制造副作用）。"""
    try:
        import cupy as cp
        cp.cuda.Device(0).asnumpy()
        return 'cupy'
    except Exception:
        pass
    try:
        import torch
        if torch.cuda.is_available():
            return 'torch'
    except Exception:
        pass
    return None


def batched_ground_state(H_batch, backend='auto', vram_budget=1.2e9):
    """H_batch: (B, dim, dim) 复厄米稠密矩阵。→ 每个 H 基态 E0/gap/⟨n_i⟩。

    backend: 'auto'（有 GPU 优先 cupy>torch，否则 numpy）| 'cupy' | 'torch' | 'numpy'。
    GPU 按 VRAM 预算分块（每块 B_chunk×dim×dim×16B）；本征确定性（与 CPU 逐点一致）。
    返回 dict(E0(B,), gap(B,), n_atom(B,n), backend)。
    """
    Hb = np.asarray(H_batch)
    if Hb.ndim != 3 or Hb.shape[1] != Hb.shape[2]:
        raise ValueError('H_batch 需形如 (B,dim,dim)，实到 %r' % (Hb.shape,))
    B, dim, _ = Hb.shape
    n = int(round(np.log2(dim)))
    if (1 << n) != dim:
        raise ValueError('dim=%d 非 2^N' % dim)
    bits = bit_matrix(n, dim).astype(np.float64)          # (n, dim)
    be = detect_gpu() if backend == 'auto' else backend

    def _numpy():
        w, V = np.linalg.eigh(Hb)                          # (B,dim),(B,dim,dim) 批量化
        gs = V[:, :, 0]
        p = np.abs(gs) ** 2
        return w[:, 0].copy(), (w[:, 1] - w[:, 0]).copy(), p @ bits.T, 'numpy-batched-eigh'

    if be == 'cupy':
        import cupy as cp
        chunk = max(1, int(vram_budget // max(dim * dim * 16, 1)))
        e0s, gaps, nas = [], [], []
        bits_c = cp.asarray(bits.T)                         # (dim, n)
        for i in range(0, B, chunk):
            Hc = cp.asarray(Hb[i:i + chunk])
            w, V = cp.linalg.eigh(Hc)
            gs = V[:, :, 0]
            p = cp.abs(gs) ** 2
            e0s.append(cp.asnumpy(w[:, 0]))
            gaps.append(cp.asnumpy(w[:, 1] - w[:, 0]))
            nas.append(cp.asnumpy(p @ bits_c))
            del Hc, w, V, gs, p
        return dict(E0=np.concatenate(e0s), gap=np.concatenate(gaps),
                    n_atom=np.concatenate(nas), backend='cupy-batched-eigh')

    if be == 'torch':
        import torch
        dev = torch.device('cuda')
        bits_t = torch.as_tensor(bits.T, device=dev, dtype=torch.float64)
        e0s, gaps, nas = [], [], []
        chunk = max(1, int(vram_budget // max(dim * dim * 16, 1)))
        with torch.no_grad():
            for i in range(0, B, chunk):
                Ht = torch.as_tensor(Hb[i:i + chunk], device=dev, dtype=torch.complex128)
                w, V = torch.linalg.eigh(Ht)
                gs = V[:, :, 0]
                p = (gs.abs() ** 2).to(torch.float64)
                e0s.append(w[:, 0].real.cpu().numpy())
                gaps.append((w[:, 1] - w[:, 0]).real.cpu().numpy())
                nas.append((p @ bits_t).cpu().numpy())
                del Ht, w, V, gs, p
        return dict(E0=np.concatenate(e0s), gap=np.concatenate(gaps),
                    n_atom=np.concatenate(nas), backend='torch-batched-eigh')

    e0, gap, n_atom, name = _numpy()
    return dict(E0=e0, gap=gap, n_atom=n_atom, backend=name)
