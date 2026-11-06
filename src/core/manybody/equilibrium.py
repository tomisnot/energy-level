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
（见 docs/history/标杆判据库-平衡态多体算法正确性实例.md）。
"""
import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import eigsh, expm_multiply

from .observables import bit_matrix, entropy_at_cut
from .matrix_free import torch_lanczos_ground_state

__all__ = ['ground_state', 'ground_state_imagtime', 'thermal_from_h',
           'thermal_from_eigh', 'gs_observables', 'thermal_from_spectrum',
           'detect_gpu', 'batched_ground_state', 'torch_lanczos_ground_state']


def _to_array_dim(H):
    n = H.shape[0]
    d = int(np.log2(n))
    if 1 << d != n:
        raise ValueError('H 维度 %d 非 2^N' % n)
    return d


def ground_state(H, dense_max=2048, seed=0, k=2, v0=None,
                 sector=None, sector_labels=None, basis_indices=None,
                 sector_check=True, return_evecs=False):
    """基态：小维稠密 `eigh`（精确），大维 `eigsh`(Lanczos, which='SA')。

    返回 dict(E0, gap, vec, backend)。E0=最小本征值(MHz)，gap=第一激发−基态。
    Lanczos 用固定 v0（种子）保确定性；退化基态时 vec 任取一线性组合（观测量按
    该 vec 计，不做子空间平均——调用方若基态简并需自行处理）。
    """
    sector_idx = None
    if basis_indices is not None:
        sector_idx = np.asarray(basis_indices, dtype=int)
        H = H[np.ix_(sector_idx, sector_idx)] if not isinstance(H, sp.linalg.LinearOperator) else H
    elif sector is not None:
        if sector_labels is None:
            raise ValueError('sector 模式需要 sector_labels 或 basis_indices')
        labels = np.asarray(sector_labels)
        Hs = H if sp.issparse(H) else sp.csr_matrix(H)
        if sector_check:
            coo = Hs.tocoo(); nz = coo.data != 0
            if np.any(labels[coo.row[nz]] != labels[coo.col[nz]]):
                raise ValueError('H 不守恒 sector_labels，拒绝静默切片')
        sector_idx = np.where(labels == int(sector))[0]
        if len(sector_idx) == 0:
            raise ValueError('请求 sector 为空')
        H = Hs[sector_idx][:, sector_idx]
    dim = H.shape[0]
    is_lo = isinstance(H, sp.linalg.LinearOperator)   # LinearOperator 只能稀疏路
    if dim <= 0 or int(k) < 1:
        raise ValueError('Hamiltonian 不能为空且 k 必须 >=1')
    if (dim <= dense_max and not is_lo) or dim <= 3:
        Hd = (np.column_stack([H @ np.eye(1, dim, j).ravel() for j in range(dim)])
              if is_lo else np.asarray(H.todense()) if sp.issparse(H) else np.asarray(H))
        w, V = np.linalg.eigh(Hd)
        out = dict(E0=float(w[0]), gap=float(w[1] - w[0]) if len(w) > 1 else 0.0,
                   vec=V[:, 0], backend='eq-eigh', evals=w[:k])
        if return_evecs:
            out['evecs'] = V[:, :k]
        if sector_idx is not None: out.update(sector=int(sector) if sector is not None else None,
                                               sector_idx=sector_idx, dim_sub=len(sector_idx))
        return out
    rng = np.random.default_rng(seed)
    if v0 is None:
        v0 = rng.standard_normal(dim)
        if np.issubdtype(H.dtype, np.complexfloating):
            v0 = v0 + 1j * rng.standard_normal(dim)
    if is_lo:
        Hd = H
    else:
        Hd = H if sp.issparse(H) else sp.csr_matrix(H)
    w, V = eigsh(Hd, k=min(k, dim - 1), which='SA', v0=v0, maxiter=dim * 20,
                 tol=0)
    order = np.argsort(w)
    w, V = w[order], V[:, order]
    out = dict(E0=float(w[0]),
                gap=float(w[1] - w[0]) if len(w) > 1 else 0.0,
                vec=V[:, 0], backend='eq-lanczos', evals=w)
    if return_evecs:
        out['evecs'] = V
    if sector_idx is not None: out.update(sector=int(sector) if sector is not None else None,
                                          sector_idx=sector_idx, dim_sub=len(sector_idx))
    return out


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


def thermal_from_spectrum(evals, betas, return_weights=False):
    """Exact Gibbs thermodynamics for an arbitrary spectrum (energies in MHz)."""
    values = np.asarray(evals,float)
    betas = np.atleast_1d(np.asarray(betas,float))
    if values.ndim != 1 or not len(values) or not np.isfinite(values).all():
        raise ValueError('能谱必须是非空有限数组')
    if not np.isfinite(betas).all() or np.any(betas <= 0):
        raise ValueError('betas 必须是有限正数')
    e0 = values.min(); relative = values-e0
    weights = np.exp(-betas[:,None]*relative)
    shifted_Z = weights.sum(axis=1)
    probabilities = weights/shifted_Z[:,None]
    relative_mean = probabilities @ relative
    variance = np.maximum(probabilities @ relative**2-relative_mean**2, 0.)
    lnZ = np.log(shifted_Z)-betas*e0
    result = dict(betas=betas,E=e0+relative_mean,C=betas**2*variance,F=-lnZ/betas,lnZ=lnZ)
    if return_weights:
        result['weights'] = probabilities
    return result


def _occupation_correlations(bits, probabilities):
    """(dim,n) 位表 + (dim,B) 概率 → n_atom (B,n)、连通 corr (B,n,n)。

    ``bits`` 是占据数（0/1）而不是 σz；因此相关矩阵是
    ``⟨n_i n_j⟩−⟨n_i⟩⟨n_j⟩``，与 ``⟨σz_iσz_j⟩`` 只差常数/线性项。
    """
    bits = np.asarray(bits, float)
    probabilities = np.asarray(probabilities, float)
    b = probabilities.shape[1]
    n_atom = (bits.T @ probabilities).T                 # (B,n)
    corr = np.empty((b, bits.shape[1], bits.shape[1]), float)
    for k in range(b):
        p = probabilities[:, k]
        corr[k] = (bits * p[:, None]).T @ bits - np.outer(n_atom[k], n_atom[k])
    return n_atom, corr


def thermal_from_eigh(evals, evecs, betas, bits):
    """一次 eigh 的精确 Gibbs 热态：E/C/F、⟨n_i⟩、连通占据关联。

    ``bits`` 为 (dim,n) 占据表。返回的 ``n_atom`` 是 (B,n)，``corr`` 是
    (B,n,n)；单 β 也不例外，调用方需要 1D 时自行取 [0]。
    """
    thermal = thermal_from_spectrum(evals, betas, return_weights=True)
    weights = np.asarray(thermal['weights'], float)        # (B,dim)
    probabilities = (np.abs(np.asarray(evecs, complex)) ** 2) @ weights.T
    n_atom, corr = _occupation_correlations(bits, probabilities)
    return dict(thermal, n_atom=n_atom, corr=corr)


def thermal_from_h(H, beta, n_atoms=None, basis_indices=None):
    """精确有限温 Gibbs：由谱分解 e^{−βH}，给 E/F/C、⟨n_i⟩、χ_β(Σn)、连通关联。

    β=1/MHz（β→∞ 趋于基态）。小维稠密 `eigh`（精确，O(2^N)）。
    返回 dict(E, F, C, lnZ, n_atom(n,), corr(n,n), chi_n, evals)。
    """
    d = _to_array_dim(H)
    n = n_atoms if n_atoms is not None else d
    if basis_indices is not None:
        idx = np.asarray(basis_indices, dtype=int)
        H = H[np.ix_(idx, idx)]
        bits = np.asarray([((idx >> (n-1-i)) & 1) for i in range(n)])
    else:
        bits = bit_matrix(n, 1 << n)
    Hd = np.asarray(H.todense()) if sp.issparse(H) else np.asarray(H)
    evals, evecs = np.linalg.eigh(Hd)
    thermal = thermal_from_spectrum(evals,[beta],return_weights=True)
    pm = thermal['weights'][0]
    E,F,C,lnZ = (float(thermal[key][0]) for key in ('E','F','C','lnZ'))
    # 组态对角占据 P_s = Σ_m pm_m |evecs[s,m]|²
    Ps = (np.abs(evecs) ** 2) @ pm
    n_atom, corr = _occupation_correlations(bits.T, Ps[:, None])
    n_atom = n_atom[0]
    m_tot = bits.sum(axis=0)                     # 总激发数算符（对角）
    mean_m = float(m_tot @ Ps)
    var_m = float((m_tot ** 2) @ Ps - mean_m ** 2)
    chi_n = float(beta * var_m)
    return dict(E=E, F=F, C=C, lnZ=lnZ, n_atom=n_atom, corr=corr[0],
                chi_n=chi_n, evals=evals)


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
