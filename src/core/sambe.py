# -*- coding: utf-8 -*-
"""Sambe 引擎总模块（t74：核心层 + 单体引擎 + 两体引擎合一）。

一个算法一个家。本模块自上而下分三段：
  1. 核心层（原 sambe.py）：光子块几何 photon_geometry、扩展空间结构
     SambeSetup（单/两体同一实现）、纯冻结段传播、嵌入/物理投影、
     φ-帧相干扇和重构、稠密分解等——与物理维 N 无关；
  2. 单体引擎（原 evolve_sambe.py）：M 判定/探针 + 包络自适应分段
     （中点冻结精确指数 + Duhamel 缺陷修正）；
  3. 两体引擎（原 evolve_sambe_pair.py）：两体 Sambe 完整 V_dd 无 secular
     （稠密缺陷 / 稀疏 IP Volterra-GMRES / frozen）。
物理模型（唯一定义）：实验室系 H(t)=H0+Σ env·cos(ω t)·A，无 RWA、
无旋转系、无近/远分流；AC 位移 / Bloch–Siegert / 远失谐效应全部由
精确传播自然涌现。
调用方一律 `from .sambe import …`。
"""
import numpy as np
import scipy.sparse as sp
import time as _time
from scipy.sparse.linalg import expm_multiply, LinearOperator, gmres

from .evolve_util import (_CONV, _POL_Q, _as_vector)  # t74 工具/常量单源
from .field_build import _group_by_freq, _validate_fields
from .tensor import (_pair_loss_vectors, _as_product_psi,
                     _reduce_A, _reduce_B)
from .tensor_fields import _pair_groups


def _group_mats(g):
    """组的耦合矩阵列表（兼容三引擎原生键：mats / As（单体）/ K2_w（两体））。"""
    for k in ('mats', 'As', 'K2_w'):
        if k in g:
            return g[k]
    return []


def _group_envs(g):
    """组的包络列表（兼容 envs / envs_w）。"""
    return g.get('envs', g.get('envs_w', []))


def photon_geometry(E, groups, Ms):
    """Sambe 光子块几何（任意 n_phys，与态空间维度无关）。"""
    E = np.asarray(E, float)
    n_phys = E.size
    G = len(groups)
    if isinstance(Ms, (int, np.integer)):
        Ms = [int(Ms)] * G
    else:
        Ms = [int(m) for m in Ms]
        if len(Ms) != G:
            raise ValueError(f'M 长度 {len(Ms)} != 载频组数 {G}')
    Ds = [2 * m + 1 for m in Ms]                    # 每组块数
    P = 1
    for d in Ds:
        P *= d
    idx = np.arange(P)
    strides = []
    nvecs = []
    base = np.zeros(P)
    stride = 1
    for gi, g in enumerate(groups):
        Dg = Ds[gi]
        Mg = Ms[gi]
        nvec = (idx // stride) % Dg
        strides.append(stride)
        nvecs.append(nvec)
        # Sambe 块对角 = E_i − (n_g−M_g)·ω_g（与耦合方向 n→n+1 配套）
        base = base - (nvec - Mg) * g['omega_mhz']
        stride *= Dg
    diag = np.tile(E, P) + np.repeat(base, n_phys)
    # 重构相位系数：θ_p(t) = nphase[p]·t（rad，t 单位 ns）
    nphase = np.zeros(P)
    for nvec, g in zip(nvecs, groups):
        nphase = nphase + nvec * (_CONV * g['omega_mhz'])
    # 中央光子块（各组量子数 n=0，即中心化索引 M_g）：物理态嵌入位置
    p_center = sum(Mg * st for Mg, st in zip(Ms, strides))
    return dict(P=P, dim=n_phys * P, strides=strides, nvecs=nvecs,
                diag=diag, nphase=nphase, p_center=p_center, Ms=Ms,
                Ds=Ds, _idx=np.arange(P))


class SambeSetup:
    """N 无关 Sambe 扩展空间结构（t73 合并单体/两体两份逐行复制的实现）。

    参数：
      E          : 物理能量（单体 (N,) / 两体 (N²,)，MHz）
      groups     : 载频组（键兼容 mats/As/K2_w + envs/envs_w；矩阵
                   dense/sparse 均可，尺寸 = 物理维）
      M          : int 或逐载频 list
      dense_max  : 扩展维 ≤ 此值走稠密（单体 2200 / 两体 700 / 两场 3000）
      static     : 常数耦合矩阵（两体 V_dd；kron(I_P, ·) 块对角），可 None
      gamma      : NonHermitian 对角（寿命 −iΓ/2，长度 = len(E)），可 None
      opt_terms  : 非光子常数项 [(矩阵, env[, label])]（两场变体光场）

    段算符：K(mid) = diag(E − n·ω − iΓ/2) + kron(I_P, static)
            + Σ_g Σ_w env_w(mid)·kron(T_g, ·5·K_gw + 转置厄米)
            + Σ_o env_o(mid)·kron(I_P, Ko_o)
    """

    def __init__(self, E, groups, M, dense_max=2200, static=None,
                 gamma=None, opt_terms=None):
        E = np.asarray(E, float)
        self.E = E
        self.n_phys = E.size
        self.N = self.n_phys          # 单体引擎消费名
        self.N2 = self.n_phys         # 两体引擎消费名
        self.groups = groups
        geo = photon_geometry(E, groups, M)
        self.Ms = geo['Ms']
        self.Ds = list(geo['Ds'])
        self.D = 2 * self.Ds[0] + 1 if len(set(self.Ds)) == 1 else None
        self.M = Ms = geo['Ms']
        self.P = geo['P']
        self.dim = geo['dim']
        self.strides = geo['strides']
        self.nvecs = geo['nvecs']
        self.diag = geo['diag']
        self.nphase = geo['nphase']
        self.p_center = geo['p_center']
        # gamma → NonHermitian 对角（寿命通道，落点=丢失）
        if gamma is not None and np.any(gamma):
            self.diag = (self.diag.astype(complex)
                         - 0.5j * np.tile(np.asarray(gamma, float),
                                          self.P))
        # static（V_dd）块对角嵌入：kron(I_P, ·)
        if static is None:
            static_sp = None
        elif sp.issparse(static):
            static_sp = (sp.kron(sp.eye(self.P, format='csr'),
                                 static.tocsr().astype(complex)).tocsr()
                         if static.nnz else None)
        else:
            arr = np.asarray(static, complex)
            static_sp = (sp.kron(sp.eye(self.P, format='csr'),
                                 sp.csr_matrix(arr)).tocsr()
                         if np.any(arr != 0) else None)
        self.has_static = static_sp is not None
        K_static = sp.diags(self.diag, format='csr')
        if static_sp is not None:
            K_static = (K_static + static_sp).tocsr()
        self.K_static = realify(K_static)
        # 逐组逐波形光子邻接耦合：kron(T_g, ·5·K) + kron(T_gᵀ, ·5·Kᵀ)
        # （T_g = n→n+1 光子位移 0/1 矩阵；每矩阵元恰一个贡献源）
        idx = geo['_idx']
        self.Kc_ext = []
        self._group_active = []
        for gi, g in enumerate(groups):
            mats = _group_mats(g)
            stride = self.strides[gi]
            Dg = self.Ds[gi]
            rs, cs = [], []
            for s in range(Dg - 1):
                p_src = idx[(idx // stride) % Dg == s]
                rs.append(p_src + stride)
                cs.append(p_src)
            T = sp.coo_matrix(
                (np.ones(np.concatenate(rs).size),
                 (np.concatenate(rs), np.concatenate(cs))),
                shape=(self.P, self.P)).tocsr()
            wl = []
            any_nz = False
            for K in mats:
                Ksp = (K if sp.issparse(K)
                       else sp.csr_matrix(np.asarray(K, float)))
                Kh = 0.5 * Ksp.astype(complex)
                Kcw = realify(
                    (sp.kron(T, Kh) + sp.kron(T.T, Kh.T)).tocsr())
                wl.append(Kcw)
                any_nz = any_nz or (Kcw.nnz > 0)
            self.Kc_ext.append(wl)
            self._group_active.append(any_nz)
        # 非光子常数项（两场变体光场：不进光子块，kron(I_P, ·)）
        self.opt_Ko = []
        self.opt_envs_w = []
        self.opt_labels = []
        for ot in (opt_terms or []):
            if isinstance(ot, dict):        # 两场变体原生格式
                Ko, wf = ot['Ko'], ot['wf']
                lab = ot.get('label', 'opt')
            else:
                Ko, wf = ot[0], ot[1]
                lab = ot[2] if len(ot) > 2 else 'opt'
            Ko2 = (Ko.tocsr().astype(complex) if sp.issparse(Ko)
                   else sp.csr_matrix(np.asarray(Ko, complex)))
            self.opt_Ko.append(
                sp.kron(sp.eye(self.P, format='csr'), Ko2).tocsr())
            self.opt_envs_w.append(wf)
            self.opt_labels.append(lab)
        self.dense_max = dense_max
        self.dense = self.dim <= dense_max
        if self.dense:
            # 段算符装配用稠密副本；Kc_ext/opt_Ko 本体保持稀疏——缺陷修正
            # 走稀疏 matvec（nnz 极小），比稠密 (dim²×n_nodes) 快 2 个量级
            self.K_static = K_static.toarray()
            self.Kc_dense = [([Kc.toarray() for Kc in wl]
                              if act else None)
                             for wl, act in zip(self.Kc_ext,
                                                self._group_active)]
            self.opt_Ko_d = [Ko.toarray() for Ko in self.opt_Ko]
        else:
            self.Kc_dense = None
            self.opt_Ko_d = None

    def _c_mids_wf(self, groups, mid):
        """逐波形中点包络——flat list: [g0w0, g0w1, ...]。"""
        return [env_at(wf, mid) for g in groups for wf in _group_envs(g)]

    def _c_at_nodes_wf(self, groups, nodes):
        """逐波形节点包络——(total_wf, n_nodes)。"""
        nodes = np.asarray(nodes, float)
        rows = []
        for g in groups:
            for wf in _group_envs(g):
                rows.append(np.asarray(wf(nodes), float))
        return np.array(rows, float)

    def hamiltonian_seg(self, mid, groups=None):
        """段算符 K(mid)（稠密 ('dense', K) / 稀疏 ('sparse', K)）。

        统一逐波实现（env_w × Kcw_w 逐波形配对——同频合并 bug 的修复点）。
        """
        if groups is None:
            groups = self.groups
        if self.dense:
            K = np.array(self.K_static, copy=True)
            for gi, g in enumerate(groups):
                Kc_list = self.Kc_dense[gi]
                if Kc_list is None:
                    continue
                for wi, wf in enumerate(_group_envs(g)):
                    c_w = env_at(wf, mid)
                    if c_w != 0.0:
                        K = K + c_w * Kc_list[wi]
            for Ko, wf in zip(self.opt_Ko_d, self.opt_envs_w):
                c_o = env_at(wf, mid)
                if c_o != 0.0:
                    K = K + c_o * Ko
            return ('dense', K)
        K = self.K_static.copy()
        for gi, g in enumerate(groups):
            for wi, wf in enumerate(_group_envs(g)):
                c_w = env_at(wf, mid)
                if c_w != 0.0:
                    K = (K + c_w * self.Kc_ext[gi][wi]).tocsr()
        for Ko, wf in zip(self.opt_Ko, self.opt_envs_w):
            c_o = env_at(wf, mid)
            if c_o != 0.0:
                K = (K + c_o * Ko).tocsr()
        return ('sparse', K.tocsr())


def env_at(wf, t):
    """标量包络取值——收敛各引擎反复出现的
    float(np.asarray(wf(np.atleast_1d(t))[, float]).flat[0]) 惯用式。"""
    return float(np.asarray(wf(np.atleast_1d(t)), float).flat[0])


def realify(M):
    """虚部全零的复数矩阵转实数（BLAS/eigh 提速 ~10×）。
    耗散（−iΓ/2 对角）存在时虚部非零，自动保留复数。"""
    if sp.issparse(M):
        if M.data.size and not np.any(np.imag(M.data)):
            return sp.csr_matrix((np.real(M.data), M.indices, M.indptr),
                                 shape=M.shape)
        return M
    arr = np.asarray(M)
    if np.iscomplexobj(arr) and not np.any(np.imag(arr)):
        return np.real(arr)
    return M


def embed(psi, setup, n_phys=None):
    """物理态 → 扩展空间中央光子块（n=0；嵌入时刻重构精确还原 ψ）。"""
    if n_phys is None:
        n_phys = setup.N2
    X = np.zeros(setup.dim, complex)
    off = n_phys * setup.p_center
    X[off:off + n_phys] = psi
    return X


def reconstruct_psi(setup, X, tq):
    """扩展态 → 物理振幅（(t,t') 投影 ψ_i = Σ_n X_{n,i}·e^{−inω t}）。"""
    Xm = np.asarray(X, complex).reshape(setup.P, setup.n_phys)
    return np.exp(-1j * setup.nphase * float(tq)) @ Xm


def physical_pops(setup, X, tq):
    """扩展态 → 物理布居（|ψ|²）。"""
    return np.abs(reconstruct_psi(setup, X, tq)) ** 2


def dense_decomp(K):
    """稠密段算符本征分解（Hermitian → eigh；否则一般 eig + 逆）。
    强制 ndarray 避免 numpy.matrix 广播问题。"""
    K_arr = np.asarray(K)
    if np.allclose(K_arr, K_arr.conj().T, rtol=0.0, atol=1e-10):
        evals, evecs = np.linalg.eigh(K_arr)
        return evals, evecs, evecs.conj().T
    evals, evecs = np.linalg.eig(K_arr)
    return evals, evecs, np.linalg.inv(evecs)


def prop_step(setup, hamiltonian_seg, ta, tb, X):
    """冻结段传播（稠密路径：本征分解精确指数）。"""
    dt = tb - ta
    evals, evecs, left = dense_decomp(hamiltonian_seg[1])
    a0 = np.asarray(left @ X).ravel()
    phase = np.asarray(np.exp(-1j * _CONV * dt * evals)).ravel()
    return np.asarray(evecs) @ (a0 * phase)


# ============================================================
# Gauss–Legendre 节点缓存（缺陷修正求积；原 evolve_sambe._gl_nodes）
# ============================================================
_GL_CACHE = {}


def gl_nodes(n):
    """n 点 Gauss-Legendre 节点/权重（[−1,1]，缓存）。"""
    gl = _GL_CACHE.get(n)
    if gl is None:
        from numpy.polynomial.legendre import leggauss
        gl = leggauss(n)
        _GL_CACHE[n] = gl
    return gl


def frozen_seg_propagate(setup, ta, tb, X, h, groups, eigh_cache=None,
                         sparse_step=None):
    """纯冻结段传播（t73：单/两体同一实现）——固定 h 内部细分，每段中点
    K 对角化 + 精确指数（无缺陷修正；误差 = 段内包络变化，与 THz 解耦）。

    eigh_cache：[key, val] 可变容器——包络值键相同的相邻段共享分解
    （square 平台/全关区不再重复 eigh；仅 Hermitian 路径缓存）。键含
    **全部**波形（含两场变体光场 env——旧两体版只键控微波组，光场 env
    变化时会错用陈旧分解，此处一并修复）。
    sparse_step：稀疏 K 段传播策略 callable(K, dt, X) → X（单体注入
    expm_multiply；缺省稠密化 eigh——大 dim 调用方自慎）。
    """
    nsub = max(1, int(np.ceil((tb - ta) / float(h))))
    hh = (tb - ta) / nsub
    for _s in range(nsub):
        t1 = ta + _s * hh
        mid = t1 + 0.5 * hh
        kind, K = setup.hamiltonian_seg(mid, groups)
        if kind == 'diag':
            X = X * np.exp(-1j * _CONV * hh * setup.diag)
            continue
        if kind == 'sparse' and sparse_step is not None:
            X = sparse_step(K, hh, X)
            continue
        if kind == 'sparse':
            K = K.toarray()
        if np.iscomplexobj(K) and np.any(np.imag(K) != 0):
            # 耗散（NonHermitian）→ 通用分解（不缓存）
            evals, evecs, left = dense_decomp(K)
            X = evecs @ ((left @ X) * np.exp(-1j * _CONV * hh * evals))
            continue
        Kd = np.real(K)
        if eigh_cache is not None:
            key = tuple(
                round(env_at(wf, mid), 12)
                for g in groups for wf in _group_envs(g))
            if eigh_cache[0] == key and eigh_cache[1] is not None:
                evals, evecs = eigh_cache[1]
            else:
                evals, evecs = np.linalg.eigh(Kd)
                eigh_cache[0] = key
                eigh_cache[1] = (evals, evecs)
        else:
            evals, evecs = np.linalg.eigh(Kd)
        a0 = evecs.conj().T @ X
        X = evecs @ (a0 * np.exp(-1j * _CONV * hh * evals))
    return X


def sector_sum_amp(phi, P, n_phys):
    """φ-帧物理振幅 = 相干扇区和 Σ_p φ[p·n_phys + i]（t73b 单/两体统一）。

    由 ψ_i = Σ_p e^{−iC·n_p·ω·t}·X[p,i]、X = e^{−iC·D·t}·φ、
    D[p,i] = E_i − n_p·ω 可得全部时间相位恰好对消：
    ψ_i = e^{−iC·E_i·t}·Σ_p φ[p,i]——全局相位不影响布居。
    注意：非相干版 Σ_p|φ[p,i]|² 丢掉 O(Ω/ω) 的扇区间交叉项
    （旧 mixed 单原子的系统误差）。
    """
    phi = np.asarray(phi, complex).reshape(P, n_phys)
    return phi.sum(axis=0)


def sector_sum_pops(phi, P, n_phys):
    """φ-帧物理布居 |Σ_p φ[p,i]|²（mixed 单/两体统一重构）。"""
    return np.abs(sector_sum_amp(phi, P, n_phys)) ** 2


# ============================================================
# 单体引擎
# ============================================================
from .params import (_M_START, _M_MAX, M_EST_BW_MHZ,
                     _DENSE_MAX, _ENV_PROBE_N,
                     _DT_MIN_FRAC, _SPARSE_SEG_MAX, _SPARSE_MAX_DEPTH,
                     _NODE_LADDER, _VOLT_MAX_ITER, _VOLT_PICARD_FRAC,
                     _VOLT_PRE_SWEEPS, R_CRIT, MW_MAX_MHZ)  # t74c 参数单源
# t80：删死导入 `_DENSE_MAX_PAIR`（从未接线，稠密上限一律由 `_DENSE_MAX` 决定）；
#      `MW_MAX_MHZ` 原有 6 处行内 `1e5` 副本，已改回引用本常量。
# M 判定的近共振带宽（MHz）：离共振 Rydberg-Rydberg 巨矩阵元（如光学频下
# 60S↔60P 的 Ω~19 GHz）不参与 r 判定——其物理贡献是二阶微扰（~1e-9 相对），
# 光子块阶数由与载频近共振的对决定（Mizuta 2025 / Kuwahara–Mori–Saito 2016
# 收敛界针对共振耦合算子范数；离共振对的 Sambe 收敛在 M=1 已足够）。
# 扩展维 ≤ 此值走稠密 eigh，否则稀疏 expm_multiply
# （2026-08-29 提到 2200：逐载频 M 后 22 态 3 载频 = 45 扇区×22 = 990 维，
#  M+1 探针 setup 75 扇区×22 = 1650 维——仍须稠密，否则掉进 expm_multiply
#  THz 对角爆炸。2200 维实对称 eigh ~0.5 s，可接受。）
# 包络采样（活性判断/恒常判断）
# Richardson 递归深度保护
# 临时调试日志（非 None 时记录 (ta, dt, rms)）
_DBG_LOG = None


# ============================================================
# 场构造（无 near_cutoff / 无 AC 注入 / 无 scatter——全部补丁机制切割）
# ============================================================
# ============================================================
# Sambe 扩展空间（t73：结构统一走 SambeSetup——单/两体同一实现）
# ============================================================
class _SambeSetup(SambeSetup):
    """单体兼容包装（几何/耦合/段算符全部继承统一实现）。

    基组 |i⟩⊗|n_1..n_G⟩，扁平索引 i + N·p（逐载频 M 混合进制张量积；
    理论依据 Ho–Chu 多模 Floquet + Mizuta 2025 截断误差上界——详见
    SambeSetup docstring）。
    """

    def __init__(self, E, groups, M):
        super().__init__(E, groups, M, dense_max=_DENSE_MAX)


def _seg_decomp(setup, groups, ta, tb):
    """段算符分解：中点全零包络 → 冻结对角（单体既有快路径）；
    否则装配 K(mid)（dense/sparse 由 SambeSetup 按扩展维自动分流）。"""
    mid = 0.5 * (ta + tb)
    for gi, g in enumerate(groups):
        if not setup._group_active[gi]:
            continue
        for wf in _group_envs(g):
            if env_at(wf, mid) != 0.0:
                return SambeSetup.hamiltonian_seg(setup, mid)
    return ('diag', None)


def _embed(psi, setup):
    """物理态 → 扩展空间中央光子块（量子数 n=0；嵌入自由度取单块非零，
    重构求和 Σ_n c_n e^{−inωt} 在嵌入时刻精确还原 ψ）。"""
    return embed(psi, setup)


def _apply_prop(prop, setup, psi, ta, tq):
    """段传播 + (t,t') 重构：ψ(ta) → ψ(tq[])（(nq, N) 复矩阵）。

    重构 ψ_i(t) = Σ_p e^{i·nphase[p]·t}·φ_{p,i}(t)——Sambe 展开的物理投影。
    """
    tq = np.asarray(tq, float)
    if prop[0] == 'diag':
        ph = np.exp(-1j * _CONV * np.outer(tq - ta, setup.E))   # (nq, N)
        return psi[None, :] * ph
    if prop[0] == 'dense':
        evals, evecs = prop[1]
        a0 = evecs.conj().T @ _embed(psi, setup)
        dt = tq - ta
        A = np.exp(-1j * _CONV * np.outer(evals, dt))           # (dim, nq)
        X = evecs @ (a0[:, None] * A)                           # (dim, nq)
        Xm = X.reshape(setup.P, setup.N, len(tq))
        # 重构 ψ_i(t) = Σ_n c_n(t)·e^{−inωt}（Sambe 展开，负相位）
        PH = np.exp(-1j * np.outer(setup.nphase, tq))           # (P, nq)
        return np.einsum('pq,piq->iq', PH, Xm).T
    # 稀疏路径：逐查询 expm_multiply（后备；大基组/多频场景）
    out = np.zeros((len(tq), setup.N), complex)
    X0 = _embed(psi, setup)
    for k, t in enumerate(tq):
        X = expm_multiply(-1j * _CONV * (t - ta) * prop[1], X0)
        Xm = X.reshape(setup.P, setup.N)
        out[k] = (np.exp(-1j * setup.nphase * t) @ Xm)
    return out


def _pops_of(ps):
    return np.abs(ps) ** 2


# ============================================================
# 光子块阶数：常数包络探针（单次对角化，截断差随时变包络的上界）
# ============================================================
def _const_env(c):
    """常数包络工厂（探针用；独立函数避免循环闭包陷阱）。"""
    return lambda t: np.full_like(np.asarray(t, float), c)


def _peak_const_groups(groups, T):
    """全部包络 → 峰值常数的镜像组（探针用）。"""
    tg = np.linspace(0.0, T, _ENV_PROBE_N)
    out = []
    for g in groups:
        peaks = [float(np.max(np.abs(np.asarray(wf(tg), float))))
                 for wf in g['envs']]
        out.append(dict(
            omega_mhz=g['omega_mhz'], As=list(g['As']),
            envs=[_const_env(pk) for pk in peaks],
            labels=g['labels']))
    return out


def _probe_pops(E, probe_groups, M, psi0, T):
    """探针场景单次对角化布居（五个采样时刻）。"""
    st = _SambeSetup(E, probe_groups, M)
    kind, K = st.hamiltonian_seg(0.5 * T, probe_groups)
    if kind == 'dense':
        prop = ('dense', np.linalg.eigh(K))
    else:
        prop = ('sparse', K)
    ps = _apply_prop(prop, st, np.asarray(psi0, complex), 0.0,
                     T * np.array([0.1, 0.3, 0.5, 0.7, 1.0]))
    return np.abs(ps) ** 2


def _group_res_peak(g, E_mhz, bw_mhz):
    """组内与载频**近共振**（|ΔE−ω|<bw）的对的最大 |Ω|；E_mhz None → 全局峰值。

    离共振 Rydberg-Rydberg 巨矩阵元（60S↔60P 在光学频 Ω~19 GHz，|ΔE−ω|~295 THz）
    被排除——其物理贡献为二阶微扰，光子块阶数只由共振对决定。"""
    if E_mhz is None:
        return max(float(np.abs(A).max()) for A in g['As'])
    E = np.asarray(E_mhz, float)
    peak = 0.0
    for A in g['As']:
        Anz = np.abs(A) > 1e-12
        rows, cols = np.nonzero(Anz)
        if rows.size == 0:
            continue
        det = np.abs((E[rows] - E[cols]) - g['omega_mhz'])
        m = det < bw_mhz
        if m.any():
            peak = max(peak, float(np.abs(A[rows[m], cols[m]]).max()))
    return peak


def _m_analytic(groups, tol, E_mhz=None):
    """逐载频解析下界：(r_g/4)^{2(M_g−1)} ≤ tol（r_g = 近共振对最大 Ω/ω）。

    返回与 groups 等长的 M 列表——**每组独立**：光学载频（近共振 Ω~104 MHz，
    r~3.5e-7）→ M=1（3 块，单光子阶即收敛）；微波 r~5e-4 → M=2（5 块）。
    离共振巨矩阵元（光学频 60S↔60P，Ω~19 GHz）不参与判定（_group_res_peak）。
    统一 M 会把光学拖到微波阶数（5³=125 扇区），逐载频 M=(1,1,2) → 45 扇区。
    """
    Ms = []
    for g in groups:
        vmax = _group_res_peak(g, E_mhz, M_EST_BW_MHZ)
        r = vmax / max(abs(g['omega_mhz']), 1e-9)
        M = _M_START - 1                      # 从 M=1（3 块）起步
        if r > 0.0:
            # M=1 的截断误差 ~ (r/4)²（单光子阶，|k|≥2 谐波）；
            # 若已 ≤ tol 停留 M=1；否则按 (r/4)^{2(M−1)} 升阶。
            if (r / 4.0) ** 2 > tol:
                M = 2
                while (r / 4.0) ** (2 * (M - 1)) > tol and M < _M_MAX:
                    M += 1
        Ms.append(max(1, M))
    return Ms


def setup_P(Ms):
    """逐载频 M 列表 → 扇区数 Π(2M_g+1)。"""
    p = 1
    for m in Ms:
        p *= (2 * int(m) + 1)
    return p


# ============================================================
# 包络自适应分段主驱动（冻结中点精确指数 + Richardson 外推 + 减半控制）
# ============================================================
# Gauss–Legendre 节点缓存收归 sambe（t73）；此处别名保持旧导出点
from .sambe import gl_nodes as _gl_nodes, _GL_CACHE  # noqa: E402


def _frozen_apply(evals, evecs, a0, ta, tq):
    """冻结传播子作用：e^{−iC K(c_mid)(tq−ta)}·X(ta)。"""
    return evecs @ (a0 * np.exp(-1j * _CONV * evals * (tq - ta)))


def _defect_sweep(setup, groups, ta, tb, nodes, ws, evals, evecs,
                  psi_nodes):
    """单次缺陷扫描（本征基可分离形式，O(dim·n)）：

    δ(t) = −iC∫_{ta}^{t} U_f(t,s)·ΔK(s)·ψ(s) ds，
    本征基分量 δ_r(t) = −iC·e^{−iCε_r t}·S_r，
    S_r = Σ_q w_q·e^{+iCε_r t_q}·(evecs†·ΔK(t_q)·ψ(t_q))_r——对固定节点组
    一次求和即得任意目标时刻的 δ（无嵌套）；节点级 δ 满足 Volterra 上限
    （仅用 ≤ 该节点的求积点，累积和实现）。返回 (S, delta_nodes)：
    S = 全区间和（供目标时刻求值）。
    """
    nodes = np.asarray(nodes, float)
    ws = np.asarray(ws, float)
    Psi = np.asarray(psi_nodes, complex)                 # (n, dim)
    V = np.zeros_like(Psi)
    mid = 0.5 * (ta + tb)                                  # 段中点（用于逐波修正）
    for g, Kc_list in zip(groups, setup.Kc_dense):
        if Kc_list is None:
            continue
        # 逐波配对：dc_w(q) = env_w(nodes[q]) − env_w(mid)，乘对应 Kcw（修复合并bug）
        V_g = np.zeros_like(Psi)
        has_active = False
        for wf, Kcw in zip(_group_envs(g), Kc_list):
            dc_w = np.asarray(wf(nodes), float) - env_at(wf, mid)
            if Kcw is not None and np.max(np.abs(dc_w)) > 1e-14:
                V_g = V_g + dc_w[:, None] * (Psi @ Kcw.T)
                has_active = True
        if has_active:
            V = V + V_g
    G = V @ evecs.conj()                                 # (n, dim) evecs†·v_q
    Ph = np.exp(1j * _CONV * np.outer(evals, nodes))     # (dim, n)
    Csum = np.cumsum(Ph * (G * ws[:, None]).T, axis=1)   # (dim, n) 累积和
    S = Csum[:, -1]                                      # 全区间和（本征基）
    Phn = np.exp(-1j * _CONV * np.outer(evals, nodes))   # (dim, n)
    # 节点级 δ（物理基，满足 Volterra 上限）
    delta_nodes = (evecs @ (-1j * _CONV * (Phn * Csum))).T   # (n, dim)
    return S, delta_nodes


def _defect_at(setup, S, evals, evecs, tq):
    """由缺陷扫描结果 S（本征基）求物理基 δ(tq)。"""
    return evecs @ (-1j * _CONV * np.exp(-1j * _CONV * evals * tq) * S)


def _propagate_adaptive(setup, groups, psi0, t_grid, tol, progress_arr, log,
                        frozen_h_ns=None):
    """自适应分段演化：全程在 Sambe 扩展空间连续传播（边界不重嵌入，
    无微运动泄漏；幺正守恒），仅在输出时刻投影物理态。

    段传播 = 冻结中点精确指数 + Duhamel 缺陷修正（真实包络差在段内 Gauss
    求积，块对角载波相位由冻结指数精确承载）；求积节点数自适应倍增（被积
    函数含本征拍频振荡，节点须解析），节点加密差即误差估计：≤ tol 接受，
    否则真中点细分（细分同时降低所需节点数）——可细分到输出网格以下，
    直到残差 ≤ tol 或触及绝对下限。

    **frozen_h_ns（t36 方案 1）**：>0 时切换"纯冻结"模式——固定 h 内部细分，
    每段只做中点对角化 + 精确指数（无缺陷修正/无 Gauss 节点/无 Richardson）。
    缺陷修正的 Gauss 核含 THz 光子块本征跨度 → 在 THz 场景段被细分到
    h_floor 仍不收敛；纯冻结误差 = 段内包络变化（σ 缓变场景 h=1ns 布居
    偏差 ~1e-7），与 THz 解耦。默认 None = 原缺陷路径（行为不变）。
    """
    T = float(t_grid[-1])
    t = np.asarray(t_grid, float)
    N = setup.N
    psi0 = np.asarray(psi0, complex)
    pops = np.zeros((len(t), N))
    pops[0] = _pops_of(psi0)
    report = dict(n_segments=0, max_richardson_diff=0.0, dt_min_hit=False,
                  max_norm_dev=0.0)
    if frozen_h_ns is not None and frozen_h_ns > 0:
        report['frozen_h_ns'] = float(frozen_h_ns)
    h_floor = max(T / _DT_MIN_FRAC, 1e-9)

    def _propagate_seg(ta, tb, X):
        """自适应传播 X(ta) → X(tb)（无输出；返回修正后链状态）。"""
        if progress_arr[1] != 0.0:
            raise RuntimeError('用户取消')
        if frozen_h_ns is not None and frozen_h_ns > 0:
            # t36/t73：纯冻结——共享实现（t73 起带 eigh 缓存：包络键相同
            # 的相邻段共享分解；稀疏段仍走单体 expm_multiply 策略）
            nsub = max(1, int(np.ceil((tb - ta) / float(frozen_h_ns))))
            report['n_segments'] += nsub
            return frozen_seg_propagate(
                setup, ta, tb, X, frozen_h_ns, groups,
                sparse_step=lambda K, dt, Xk: expm_multiply(
                    -1j * _CONV * dt * K, Xk))
        dec = _seg_decomp(setup, groups, ta, tb)
        if dec[0] == 'diag':
            report['n_segments'] += 1
            return X * np.exp(-1j * _CONV * (tb - ta) * setup.diag)
        kind, K = dec
        if kind == 'sparse':
            return expm_multiply(-1j * _CONV * (tb - ta) * K, X)
        evals, evecs = np.linalg.eigh(K)
        a0 = evecs.conj().T @ X

        def frozen_at(tq):
            return _frozen_apply(evals, evecs, a0, ta, tq)

        psi_prev = None
        psi_out = frozen_at(tb)
        est = 0.0
        for n_nodes in (32, 64, 128):
            glx, glw = _gl_nodes(n_nodes)
            nodes = ta + 0.5 * (tb - ta) * (glx + 1.0)
            ws = 0.5 * (tb - ta) * glw
            # 节点冻结态（批量）
            F = evecs @ (a0[:, None] * np.exp(
                -1j * _CONV * np.outer(evals, nodes - ta)))   # (dim, n)
            psi_nodes = F.T
            # 两级缺陷扫描：一级用冻结被积态，二级用修正后节点态（Volterra）
            S1, d_nodes1 = _defect_sweep(
                setup, groups, ta, tb, nodes, ws, evals, evecs,
                psi_nodes)
            S2, _ = _defect_sweep(
                setup, groups, ta, tb, nodes, ws, evals, evecs,
                psi_nodes + d_nodes1)
            psi_out = frozen_at(tb) + _defect_at(setup, S2, evals, evecs, tb)
            if psi_prev is not None:
                est = float(np.linalg.norm(psi_out - psi_prev))
                if est <= tol:
                    break
            psi_prev = psi_out
        if est > tol and (tb - ta) > h_floor:
            tm = 0.5 * (ta + tb)
            Xm = _propagate_seg(ta, tm, X)
            return _propagate_seg(tm, tb, Xm)
        report['max_richardson_diff'] = max(
            report['max_richardson_diff'], est)
        report['n_segments'] += 1
        if est > tol:
            report['dt_min_hit'] = True
        if _DBG_LOG is not None:
            _DBG_LOG.append((ta, tb - ta, est))
        return psi_out

    X = _embed(psi0, setup)
    tprev = 0.0
    for i in range(1, len(t)):
        progress_arr[0] = min(tprev / T, 0.999)
        X = _propagate_seg(tprev, float(t[i]), X)
        ps = _reconstruct(setup, X, t[i])
        p = _pops_of(ps)
        pops[i] = p
        report['max_norm_dev'] = max(report['max_norm_dev'],
                                    float(abs(p.sum() - 1.0)))
        tprev = float(t[i])
    psi_f = _reconstruct(setup, X, T)
    progress_arr[0] = 1.0
    return pops, psi_f, report


def _reconstruct(setup, X, tq):
    """扩展态 → 物理态（(t,t') 投影 ψ_i = Σ_n X_{n,i} e^{−inω t}）。"""
    return reconstruct_psi(setup, X, tq)


def _env_all_constant(groups, T):
    for g in groups:
        for wf in g['envs']:
            v = np.asarray(wf(np.linspace(0.0, T, 64)), float)
            if np.ptp(v) > 1e-9:
                return False
    return True


# ============================================================
# 单次全传播（给定光子块阶数；供主入口与 M 收敛对比）
# ============================================================
def _run_sambe(E, groups, M, psi0, t, tol, progress_arr, log,
               frozen_h_ns=None):
    """返回 (pops, psi_f, report, setup)。frozen_h_ns：t36 纯冻结模式段长。"""
    setup = _SambeSetup(E, groups, M)
    T = float(t[-1])
    if _env_all_constant(groups, T):
        # 全常数包络 → 单次扩展空间对角化（成本与时长无关）
        kind, K = setup.hamiltonian_seg(0.5 * T)
        if kind == 'dense':
            prop = ('dense', np.linalg.eigh(K))
        elif kind == 'sparse':
            prop = ('sparse', K)
        else:
            prop = ('diag', None)
        ps = _apply_prop(prop, setup, psi0, 0.0, t)
        pops = np.abs(ps) ** 2
        report = dict(n_segments=1, max_richardson_diff=0.0,
                      dt_min_hit=False,
                      max_norm_dev=float(np.max(
                          np.abs(pops.sum(axis=1) - 1.0))))
        return pops, ps[-1], report, setup
    pops, psi_f, report = _propagate_adaptive(
        setup, groups, psi0, t, tol, progress_arr, log, frozen_h_ns)
    return pops, psi_f, report, setup


# ============================================================
# 主入口：method='exact'（无耗散；含耗散转参考路径）
# ============================================================
def evolve_sambe(states, fields, psi0, t_grid, tol=1e-6, decays=None,
                 progress_arr=None, log=None, record_amp=False,
                 E_shift=None, frozen_h_ns=None):
    """第一性原理单原子含时演化（无模型近似）。

    fields : kind='exact' 场列表（build_fields_exact 输出或手工同构）
    tol    : 布居容差（包络冻结 + 光子块截断两处共用）
    decays : 非空 → 转 evolve_reference（Lindblad，第一期执行路径）
    frozen_h_ns : t36 方案 1——>0 时启用"纯冻结分段"（砍缺陷修正，固定
        段长 h；适合 THz 光子块 × 时变包络场景如 STIRAP，缺陷修正的
        Gauss 核在 THz 本征跨度下死局）。None = 原缺陷路径（默认不变）。
    返回 dict(t, pops, states, om_mhz, method='exact', backend, loss,
              error_report, [psi_amp | rho])——与 evolve.evolve 同构。
    """
    from .coupling import stage_progress
    log = log or (lambda *a: None)
    t = np.asarray(t_grid, float)
    if progress_arr is None:
        progress_arr = np.zeros(3)
    stage_progress(progress_arr, 0, 0.0)     # 阶段 0：M 判定/探针/装配
    _validate_fields(fields)
    psi0v = _as_vector(states, psi0)

    if decays:
        log(f'[exact] 含 {len(decays)} 个衰变通道 → 转参考路径'
            '（H0 相互作用绘景 + 自适应主方程积分，同为第一性原理）')
        from .evolve_reference import evolve_reference   # t74：独立模块
        return evolve_reference(states, fields, psi0v, t, decays=decays,
                                progress_arr=progress_arr,
                                record_amp=record_amp)

    E = np.array([s['E_mhz'] for s in states], float)
    if E_shift is not None:
        E = E + np.asarray(E_shift, float)
    E = E - float(E.min())                    # 公共偏移 = 全局相位（精确等价）

    if not fields:
        stage_progress(progress_arr, 2, 1.0)
        pops = np.zeros((len(t), len(states)))
        pops[:] = _pops_of(psi0v)
        out = dict(t=t, pops=pops, states=states, om_mhz=[],
                   method='exact', backend='none', loss=np.zeros(len(t)),
                   error_report={'n_segments': 0, 'photon_blocks': {},
                                 'max_richardson_diff': 0.0})
        if record_amp:
            out['psi_amp'] = psi0v.copy()
        return out

    groups = _group_by_freq(fields)
    T = float(t[-1])
    tol = float(tol)

    # ---- 光子块收敛：逐载频常数包络探针 ----
    # 每载频独立递增（固定其他组）：光学 r~1e-7 → M=1 即收敛；
    # 微波 r~3e-4 → M=1..2。避免统一 M 把光学拖到微波阶数（125 扇区）。
    Ms = _m_analytic(groups, tol, E)
    probe_groups = _peak_const_groups(groups, T)
    G = len(groups)
    for gi in range(G):
        stage_progress(progress_arr, 0, 0.2 + 0.6 * gi / max(G, 1))
        # 近共振峰 Ω（去离共振 Rydberg 巨矩阵元污染）；r 极小（光学量级
        # <1e-5）→ 解析界已证单光子阶收敛，探针验证纯属浪费——跳过。
        vmax = _group_res_peak(groups[gi], E, M_EST_BW_MHZ)
        r_g = vmax / max(abs(groups[gi]['omega_mhz']), 1e-9)
        if r_g < 1e-5:
            continue
        # 微波类组：本组做 M vs M+1 验证（其余组固定）
        while Ms[gi] < _M_MAX:
            M_cur = list(Ms)
            M_nxt = list(Ms)
            M_nxt[gi] = M_cur[gi] + 1
            stage_progress(progress_arr, 0,
                           0.2 + 0.6 * (gi + 0.5) / max(G, 1))
            cur = _probe_pops(E, probe_groups, M_cur, psi0v, T)
            nxt = _probe_pops(E, probe_groups, M_nxt, psi0v, T)
            d = float(np.max(np.abs(nxt - cur)))
            if d <= tol:
                break                            # 本组 M_g 已收敛
            Ms[gi] = M_nxt[gi]
            log(f'[exact] 光子块 M[{gi}]={M_cur[gi]}→{M_nxt[gi]}：'
                f'探针截断差 {d:.2e}，递增')
            if Ms[gi] >= _M_MAX:
                log(f'[exact] 警告：载频 {groups[gi]["omega_mhz"]:.6g}MHz 光子块'
                    f'达上限 M={_M_MAX}，截断误差可能 > tol')
                break
    log(f'[exact] 光子块（逐载频）M={Ms}'
        f'（扇区数 Π(2M_g+1)={setup_P(Ms)}）')
    if frozen_h_ns is not None and frozen_h_ns > 0:
        # t37：frozen 段 eigh 是小矩阵——全程单线程（OpenBLAS 24 线程在此
        # 尺度是净损失 16ms→6ms @234 维；逐段开关上下文开销更大，故包全程）
        import threadpoolctl
        stage_progress(progress_arr, 1, 0.0)  # 阶段 1：传播（段循环写 t/T）
        with threadpoolctl.threadpool_limits(limits=1):
            pops, psi_f, report, setup = _run_sambe(
                E, groups, Ms, psi0v, t, tol, progress_arr, log, frozen_h_ns)
    else:
        stage_progress(progress_arr, 1, 0.0)
        pops, psi_f, report, setup = _run_sambe(
            E, groups, Ms, psi0v, t, tol, progress_arr, log, frozen_h_ns)
    stage_progress(progress_arr, 2, 1.0)      # 阶段 2：收尾
    if frozen_h_ns is not None and frozen_h_ns > 0:
        log(f'[exact] 纯冻结模式完成：{report["n_segments"]} 段'
            f'（h={frozen_h_ns}ns），范数偏差 {report["max_norm_dev"]:.2e}')
    else:
        log(f'[exact] 完成：{report["n_segments"]} 段，'
            f'最大 Richardson 差 {report["max_richardson_diff"]:.2e}'
            f'（tol={tol:.1e}），范数偏差 '
            f'{report["max_norm_dev"]:.2e}')

    report['photon_blocks'] = {f'{g["omega_mhz"]:.6g}MHz': setup.Ds[gi]
                               for gi, g in enumerate(groups)}
    report['tol'] = float(tol)
    out = dict(t=t, pops=pops, states=states,
               om_mhz=[float(max(np.abs(A).max() for A in g['As']))
                       for g in groups],
               method='exact',
               backend='sambe-dense' if setup.dense else 'sambe-sparse',
               loss=np.zeros(len(t)), error_report=report)
    if record_amp:
        out['psi_amp'] = psi_f
    return out


# ---- t19 函数级改名兼容别名（旧名 = 新名，外部零改动）----
evolve_exact = evolve_sambe


# ============================================================
# 两体引擎
# ============================================================
# 缺陷求积节点阶梯
# 包络采样（活性判断/恒常判断）


# ============================================================
# 两体 Sambe 扩展空间（t73：继承 SambeSetup 统一实现——
# 几何/块对角/V_dd 嵌入/光子邻接耦合/段算符装配与单体同一份代码）
# ============================================================
class _PairSambeSetup(SambeSetup):
    """两体兼容包装（保留两体构造签名；K2_w/envs_w 键由统一层识别）。"""

    def __init__(self, E_pair, N, groups, M, vdd_sp, gamma_pair_mhz,
                 opt_terms=None, dense_max=None):
        super().__init__(E_pair, groups, M,
                         dense_max=(_DENSE_MAX if dense_max is None
                                    else int(dense_max)),
                         static=vdd_sp, gamma=gamma_pair_mhz,
                         opt_terms=opt_terms)


# t73：实数化/嵌入/物理投影/稠密分解/冻结步 收归 sambe 单源（别名保持旧名）
_embed_pair = embed
_physical_pops = physical_pops
_dense_decomp = dense_decomp
_prop_step = prop_step

def _defect_pass(setup, hamiltonian_seg, ta, tb, X, groups, nodes, ws, dense_eig):
    """本征基 Duhamel 缺陷扫描。

    ΔK = Σ_g Σ_w (env_w(nodes)−env_w(mid))·Kcw_{gw}
         + Σ_o (env_o(nodes)−env_o(mid))·Ko_o      ← 光场项（两场变体下
           光场包络变化必须一并修正，否则只剩冻结中点零阶精度）

    实现：逐节点循环改为矩阵乘批量化（原逐 q 循环等价，速度 ~n 倍）：
      W[:,q] = ΔK(nodes[q])·ψ_f(q)  →  W = Σ (M @ Psif)·dc[None,:]
      δ = −iC·evecs @ [ (left@W) · e^{−iC·evals·(tb−nodes)} ] @ ws
    Kcw/Ko 一律走稀疏 matvec（nnz 极小），与路径稠密/稀疏无关。
    """
    mid = 0.5 * (ta + tb)

    # 逐波形包络差（微波 Kcw + 光场 Ko），统一为 [(矩阵, dc数组)]
    terms = []
    max_dc = 0.0
    for gi, g in enumerate(groups):
        for wi, wf in enumerate(g['envs_w']):
            dc = (np.asarray(wf(nodes), float)
                  - env_at(wf, mid))
            dmax = float(np.abs(dc).max())
            max_dc = max(max_dc, dmax)
            if dmax > 1e-14:
                terms.append((setup.Kc_ext[gi][wi], dc))
    for Ko, wf in zip(setup.opt_Ko, setup.opt_envs_w):
        dc = (np.asarray(wf(nodes), float)
              - env_at(wf, mid))
        dmax = float(np.abs(dc).max())
        max_dc = max(max_dc, dmax)
        if dmax > 1e-14:
            terms.append((Ko, dc))
    if max_dc < 1e-12 or not terms:
        return np.zeros(setup.dim, complex), max_dc

    evals, evecs, left = dense_eig
    a0 = np.asarray(left @ X).ravel()
    Psif = np.asarray(evecs) @ (
        a0[:, None] * np.exp(-1j * _CONV * np.outer(evals, nodes - ta)))
    W = np.zeros_like(Psif)
    for M, dc in terms:
        W = W + (M @ Psif) * dc[None, :]
    B = np.asarray(left @ W)
    delta = np.asarray(evecs) @ (
        (B * np.exp(-1j * _CONV * np.outer(evals, tb - nodes))) @ ws)
    return -1j * _CONV * delta, max_dc

def _volterra_rhs(setup, phi, c_nodes_wf, groups=None):
    """R(s_q)·φ_q = (K_static − D)φ + Σ_g Σ_w env_w(s_q)·Kcw_{gw}·φ。
    
    统一逐波实现。c_nodes_wf 形状为 (n_total_wfs, n_nodes)。
    """
    y = setup.K_static @ phi - setup.diag[:, None] * phi
    if c_nodes_wf is not None:
        if groups is None:
            groups = setup.groups
        idx_wf = 0
        for gi, g in enumerate(groups):
            Kcw_list = setup.Kc_ext[gi]
            if not Kcw_list:
                continue
            for wi, Kcw in enumerate(Kcw_list):
                nz = Kcw.nnz if sp.issparse(Kcw) else np.any(Kcw)
                if nz:
                    env_at_nodes = c_nodes_wf[idx_wf]
                    y = y + env_at_nodes[None, :] * (Kcw @ phi)
                idx_wf += 1
    return y


_VOLT_RULE_CACHE = {}


def _volterra_rule(n):
    """n 点 GL 节点/权重 + 部分积分矩阵 Wpart[q,s] = ∫_{-1}^{x_q} L_s(x)dx。
    注意：Volterra 节点方程必须用全部 q' 的部分积分权重（Lagrange 基尾部落在
    t_q 之后仍有贡献），截断为 q'≤q 的 cumsum 会引入一阶误差。"""
    hit = _VOLT_RULE_CACHE.get(n)
    if hit is not None:
        return hit
    x, w = _gl_nodes(n)
    V = np.vander(x, n, increasing=True).T     # V[p, s] = x_s^p
    B = np.linalg.inv(V)                       # B[s, p]
    pows = np.stack([x ** (p + 1) for p in range(n)], axis=1)   # (n, n) [q,p]
    Ip = pows - ((-1.0) ** np.arange(1, n + 1))[None, :]
    Ip = Ip / np.arange(1, n + 1)[None, :]
    Wpart = Ip @ B.T                           # (n, n)：[q, s]
    _VOLT_RULE_CACHE[n] = (x, w, Wpart)
    return x, w, Wpart


def _subspace_segment(setup, Ksp, X, ta, tb, tol, groups, m_guess=None,
                      warm_Q=None, const_c=None):
    """段传播（稀疏路径）：对角相互作用绘景 + Volterra–Picard。

    φ = e^{+iD(t−ta)}ψ：φ(t) = X + (−iC)∫ e^{+iD(t−s)}R(s)φ(s) ds，
    离散为逐节点方程，Picard 迭代（cumsum 结构）；节点倍增差 + Picard
    残差联合估计，超限由调用方细分。R 只含非对角（V_dd + 驱动），
    范数 ~百 MHz 量级，与巨大对角能量标度（~THz）无关 → 收敛快。
    返回 (psi_out, est_eff, 0, None)（后两项保持接口兼容）。"""
    C = _CONV
    h = tb - ta
    X = np.asarray(X, complex)
    prev_phi_tb = None
    est_eff = np.inf
    phi_tb_ip = None
    for n_nodes in _NODE_LADDER:
        glx, glw, Wpart = _volterra_rule(n_nodes)
        nodes = ta + 0.5 * h * (glx + 1.0)
        ws = 0.5 * h * glw
        Wp = Wpart * (0.5 * h)                 # 映射回 [ta, t_q] 积分
        if const_c is not None:
            n_total_wfs = sum(len(g.get('envs_w', ['_'])) 
                              for g in groups) if groups else 0
            if n_total_wfs > 0 and len(const_c) > 0:
                base_vals = np.repeat(np.asarray(const_c, float),
                                      n_nodes // max(n_total_wfs, 1))
                c_nodes_wf = base_vals.reshape(-1, n_nodes)
            else:
                c_nodes_wf = None
        else:
            c_nodes_wf = setup._c_at_nodes_wf(groups, nodes)
        Pi = np.exp(1j * C * np.outer(setup.diag, nodes - ta))    # (dim,n)
        Pinv = np.exp(-1j * C * np.outer(setup.diag, nodes - ta))
        # 节点配置方程为线性系统 (I−T)φ = b：GMRES 直解（无收缩率约束，
        # 对任意 h 稳健）；预条件 = 因果（cumsum）Picard 扫掠近似求逆。
        dim_n = setup.dim * n_nodes

        def _t_full(vec):
            phi = vec.reshape(setup.dim, n_nodes)
            y = _volterra_rhs(setup, phi, c_nodes_wf, groups)
            S = np.einsum('qs,ds->dq', Wp, Pinv * y)
            return ((-1j * C) * (Pi * S)).ravel()

        b_rhs = np.broadcast_to(X[:, None], (setup.dim, n_nodes)).copy()
        Aop = LinearOperator((dim_n, dim_n),
                             matvec=lambda v: v - _t_full(v),
                             dtype=complex)

        def _pre(v):
            r = v.reshape(setup.dim, n_nodes)
            phi = np.zeros_like(r)
            for _s in range(_VOLT_PRE_SWEEPS):
                y = _volterra_rhs(setup, phi, c_nodes_wf, groups)
                S = np.cumsum(Pinv * y * ws[None, :], axis=1)
                phi = r + (-1j * C) * (Pi * S)
            return phi.ravel()

        Mop = LinearOperator((dim_n, dim_n), matvec=_pre, dtype=complex)
        bvec = b_rhs.ravel()
        sol, info = gmres(Aop, bvec, M=Mop, rtol=0.0,
                          atol=max(tol * _VOLT_PICARD_FRAC *
                                   max(float(np.linalg.norm(bvec)), 1.0),
                                   1e-15),
                          restart=min(40, dim_n),
                          maxiter=_VOLT_MAX_ITER)
        if info != 0:                       # 未收敛 → 交细分（不返回垃圾态）
            est_eff = np.inf
            break
        phi = sol.reshape(setup.dim, n_nodes)
        res = max(float(np.linalg.norm((Aop @ sol) - bvec)) /
                  max(float(np.linalg.norm(bvec)), 1.0), 1e-16)
        # 端点：全区间求积（同一节点集，与阶梯自洽）
        y = _volterra_rhs(setup, phi, c_nodes_wf, groups)
        S_tb = (Pinv * y * ws[None, :]).sum(axis=1)
        phi_tb_ip = X + (-1j * C) * np.exp(
            1j * C * setup.diag * h) * S_tb
        if prev_phi_tb is not None:
            est_eff = max(float(np.linalg.norm(phi_tb_ip - prev_phi_tb)),
                          res)
            if est_eff <= tol:
                break
        prev_phi_tb = phi_tb_ip
    if not np.isfinite(est_eff):
        return X, np.inf, 0, None               # 细分信号（不返回垃圾态）
    X_tb = np.exp(-1j * C * setup.diag * h) * phi_tb_ip
    return X_tb, est_eff, 0, None


# ============================================================
# 光子块阶数探针（常数峰值包络单次传播，M vs M+1）
# ============================================================
def _pair_m_probe(E_pair, N, groups, M, vdd_sp, gamma_pair_mhz, psi0_2d,
                  T, tol, log=None, progress=None):
    """M 递增收敛：常数峰值包络传播，M vs M+1 布居差 ≤ tol。
    稀疏路径：探针窗口取 max(10ns, T/4)（截断泄漏随时间增长，窗口过短会
    低估所需 M），按 1ns 分段链式子空间传播（与主循环同一机制）。
    progress：可选 callable(frac)——每轮探针后上报（阶段 0 内进度）。"""
    def run_at(Mc):
        stp = _PairSambeSetup(E_pair, N, groups, Mc, vdd_sp, gamma_pair_mhz)
        c_list = []
        for g in groups:
            pk = 0.0
            for wf in g['envs_w']:
                v = np.asarray(wf(np.linspace(0.0, T, _ENV_PROBE_N)), float)
                pk = max(pk, float(np.max(np.abs(v))))
            c_list.append(pk)
        K = stp.hamiltonian_seg(T / 2.0 if T > 0 else 1.0, groups)
        X = _embed_pair(psi0_2d, stp)
        if stp.dense:
            X = _prop_step(stp, K, 0.0, T, X)
            return _physical_pops(stp, X, T), 1.0
        T_pr = min(T, max(10.0, 0.25 * T))
        # 截断泄漏幅值随时间线性增长 → 布居差 ~ T²：探针窗口短于 T 时
        # 按 (T/T_pr)² 外推（保守上界，防低估）
        ext_scale = (T / T_pr) ** 2
        h_seg = 1.0
        n_seg = max(1, int(np.ceil(T_pr / h_seg)))
        hs = T_pr / n_seg
        ta = 0.0
        m_guess = None
        for _s in range(n_seg):
            X, _, m_guess, _ = _subspace_segment(
                stp, K[1], X, ta, ta + hs, tol, groups, m_guess=m_guess)
            ta += hs
        return _physical_pops(stp, X, T_pr), ext_scale

    Mc = max(2, M)
    prev = None
    ext_prev = 1.0
    _plog = log or (lambda *a: None)
    _iter = 0
    while True:
        cur, ext = run_at(Mc)
        _iter += 1
        if progress is not None:
            progress(min(0.99, 0.3 * _iter))
        if prev is not None:
            d = float(np.max(np.abs(cur - prev))) * max(ext, ext_prev)
            _plog(f'[pair_exact] 探针 M={Mc} vs M-1 差={d:.3e}'
                  f'（外推 ×{max(ext, ext_prev):.1f}）')
            if d <= tol:
                return Mc - 1, d
        prev = cur
        ext_prev = ext
        if Mc >= _M_MAX:
            return Mc, np.nan
        Mc += 1


# ============================================================
# 传播核心（evolve_sambe_pair / evolve_two_field_pair 共用）
# ============================================================
def _pair_propagate(E_pair_rot, N, groups, opt_terms, V, gamma_pair,
                    psi0_2d, t, tol, M, progress_arr, log, dense_max,
                    frozen_h_ns=None):
    """冻结段传播主循环（稠密本征基缺陷 / 稀疏 IP Volterra）。

    两体 Sambe 与两体两场变体的唯一差别在"喂进来的 groups / opt_terms /
    E_pair_rot 怎么构造"（载频是否全部进光子块），传播数学完全一致——
    故此处只保留一份实现（2026-08-31 核心-调度分层纪律）。

    **frozen_h_ns（t48 阶段 A）**：>0 时切"纯冻结"模式（t36 单原子同款）——
    固定 h 内部细分，每段只做中点 K 稠密 eigh + 精确指数（无缺陷修正/
    无 Gauss 节点）——THz 光子块下缺陷 Gauss 死局（t46 稀疏亦死）的解法：
    误差由包络段内变化决定（与 THz 解耦）。默认 None = 原缺陷路径不变。
    """
    T = float(t[-1])
    N2 = N * N
    from .coupling import stage_progress
    stage_progress(progress_arr, 0, 0.3)     # 阶段 0：扩展空间装配
    setup = _PairSambeSetup(E_pair_rot, N, groups, M, V, gamma_pair,
                            opt_terms=opt_terms, dense_max=dense_max)
    stage_progress(progress_arr, 0, 0.9)
    log(f'[pair_sambe] 扩展维 = {N2} × {setup.P} = {setup.dim}'
        f'（{"稠密" if setup.dense else "稀疏 Krylov"}）'
        + (f'；纯冻结 h={frozen_h_ns}ns' if frozen_h_ns else ''))

    # ---- 全常数包络快速路径（稠密）：一次本征分解覆盖全部输出时刻 ----
    all_const = True
    probes = [wf for g in groups for wf in g['envs_w']]
    probes = probes + list(setup.opt_envs_w)
    for wf in probes:
        v = np.asarray(wf(np.linspace(0.0, T, 64)), float)
        if np.ptp(v) > 1e-9:
            all_const = False
            break
    report = dict(n_segments=0, max_richardson_diff=0.0, dt_min_hit=False,
                  max_norm_dev=0.0)

    if (all_const or (not groups and not setup.opt_Ko)) and setup.dense:
        K = setup.hamiltonian_seg(T / 2.0 if T > 0 else 1.0, groups)
        X0 = _embed_pair(psi0_2d, setup)
        # 稠密：一次本征分解 + 全输出时刻向量化重构（支持非厄米）
        evals, evecs, left = _dense_decomp(K[1])
        a0 = left @ X0
        A = np.exp(-1j * _CONV * np.outer(evals, t))      # (dim, n_t)
        Xall = evecs @ (a0[:, None] * A)                    # (dim, n_t)
        Xm = Xall.reshape(setup.P, setup.N2, len(t))
        PH = np.exp(-1j * np.outer(setup.nphase, t))        # (P, n_t)
        psi_all = np.einsum('pq,pkq->kq', PH, Xm)           # (N2, n_t)
        pops_2d = np.abs(psi_all).T ** 2
        X_final = Xall[:, -1]
        report['n_segments'] = 1
        stage_progress(progress_arr, 1, 1.0)
        return pops_2d, X_final, report, setup

    # ---- 自适应分段主循环（稠密：本征基缺陷；稀疏：IP Volterra）----
    pops_2d = np.zeros((len(t), N2))
    h_floor = max(T / _DT_MIN_FRAC, 1e-9)

    def seg_K(ta, tb):
        return setup.hamiltonian_seg(0.5 * (ta + tb), groups)

    # t57b：frozen 的 eigh 复用缓存——K 恒定相邻段（包络值相同）共享分解
    # （引擎级 const 合并：square 平台/全关区的段不再重复 eigh；仅无耗散
    #  Hermitian 路径——耗散 NonHermitian 不缓存）
    _eigh_cache = [None, None]  # t73：[env键, (evals, evecs)]——共享 frozen 用
    _seg_depth = [0]         # EP6 保护：稀疏自适应递归深度计数

    def _propagate_seg(ta, tb, X):
        if progress_arr[1] != 0.0:
            raise RuntimeError('用户取消')
        # EP6 保护（船长）：dt_min_hit = 某叶子细分到 h_floor 仍不收敛——
        # 稀疏 IP Volterra 病理（THz 光子块死局，t46）——立即报错而非继续
        # 爆炸细分（原行为：每输出区间细分到 h_floor → 2e9 段静默卡死）
        if report['dt_min_hit']:
            raise RuntimeError(
                f'[pair_sambe] 稀疏自适应不收敛（细分到 h_floor 仍 est>tol）'
                f'——THz 光子块 IP Volterra 死局（t46/EP6）；'
                f'请用 frozen_h_ns 冻结模式')
        if frozen_h_ns is not None and frozen_h_ns > 0:
            # t48/t73：纯冻结——共享实现（eigh 缓存键含全部波形，含两场
            # 变体光场 env——修复旧键漏光场的隐患）
            nsub = max(1, int(np.ceil((tb - ta) / float(frozen_h_ns))))
            report['n_segments'] += nsub
            return frozen_seg_propagate(
                setup, ta, tb, X, frozen_h_ns, groups,
                eigh_cache=_eigh_cache)
        K = seg_K(ta, tb)
        if setup.dense:
            dense_eig = _dense_decomp(K[1])
            evals, evecs, left = dense_eig
            a0 = left @ X
            psi_frozen = evecs @ (
                a0 * np.exp(-1j * _CONV * (tb - ta) * evals))
            est = 0.0
            psi_out = psi_frozen
            prev = None
            for n_nodes in _NODE_LADDER:
                glx, glw = _gl_nodes(n_nodes)
                nodes = ta + 0.5 * (tb - ta) * (glx + 1.0)
                ws = 0.5 * (tb - ta) * glw
                delta, max_dc = _defect_pass(
                    setup, K, ta, tb, X, groups, nodes, ws, dense_eig)
                if max_dc < 1e-12:           # 包络平坦：冻结即精确
                    est = 0.0
                    psi_out = psi_frozen
                    break
                cur = psi_frozen + delta
                if prev is not None:
                    est = float(np.linalg.norm(cur - prev))
                    psi_out = cur
                    if est <= tol:
                        break
                else:
                    psi_out = cur
                prev = cur
        else:
            # 稀疏：对角相互作用绘景 + Volterra–Picard
            psi_out, est, _mf, _qw = _subspace_segment(
                setup, K[1], X, ta, tb, tol, groups)
        if (not np.isfinite(est) or est > tol) and (tb - ta) > h_floor:
            # EP6 保护（船长）：稀疏自适应细分深度/段数上限——病理场景
            # （est 永不收敛 → 二分爆炸到 h_floor，静默卡死 >4min）明确报错。
            # 正常场景 est 在深度 1-3 收敛；深度 12 对应 h~5e-4ns（远细于
            # 任何物理需求）。报错建议 frozen_h_ns（THz 光子块正解，t48）。
            if _seg_depth[0] >= _SPARSE_MAX_DEPTH:
                raise RuntimeError(
                    f'[pair_sambe] 稀疏自适应细分过深（>'
                    f'{_SPARSE_MAX_DEPTH} 层，est 不收敛）——THz 光子块 IP '
                    f'Volterra 死局（t46/EP6）；请用 frozen_h_ns 冻结模式')
            if report['n_segments'] > _SPARSE_SEG_MAX:
                raise RuntimeError(
                    f'[pair_sambe] 稀疏自适应细分爆炸（est 不收敛，已超 '
                    f'{_SPARSE_SEG_MAX} 段）——场景病态（t46/EP6）；'
                    f'请用 frozen_h_ns 冻结模式')
            _seg_depth[0] += 1
            try:
                tm = 0.5 * (ta + tb)
                Xm = _propagate_seg(ta, tm, X)
                return _propagate_seg(tm, tb, Xm)
            finally:
                _seg_depth[0] -= 1
        report['max_richardson_diff'] = max(
            report['max_richardson_diff'],
            float(est) if np.isfinite(est) else 1e3)
        report['n_segments'] += 1
        if not np.isfinite(est) or est > tol:
            report['dt_min_hit'] = True
        return psi_out

    X = _embed_pair(psi0_2d, setup)
    tprev = 0.0
    stage_progress(progress_arr, 1, 0.0)     # 阶段 1：传播（逐输出区间写 t/T）
    _t_wall = _time.perf_counter()
    for i in range(1, len(t)):
        progress_arr[0] = min(tprev / T, 0.999)
        X = _propagate_seg(tprev, float(t[i]), X)
        if report['n_segments'] % 20 == 0:
            log(f'[pair_sambe] 进度 t={t[i]:.2f}/{T:.2f} ns，'
                f'{report["n_segments"]} 段'
                f'（{_time.perf_counter() - _t_wall:.0f}s）')
        pops_2d[i] = _physical_pops(setup, X, t[i])
        tprev = float(t[i])
    pops_2d[0] = np.abs(psi0_2d) ** 2
    return pops_2d, X, report, setup


# ============================================================
# 主入口
# ============================================================
def evolve_sambe_pair(states, fields_A, fields_B, psi0_A, psi0_B, t_grid,
                      vdd_matrix=None, R_um=None, theta_deg=0.0,
                      phi_deg=0.0, cache=None, tol=1e-6,
                      decays_A=None, decays_B=None,
                      progress_arr=None, log=None, record_amp=False,
                      frozen_h_ns=None, per_freq_M=None, device='cpu',
                      ac_ize=False):
    """两体第一性原理含时演化（完整 V_dd，无 secular/RWA/旋转系）。

    签名与返回契约对齐 pair_evolve.pair_evolve（GUI 直接复用）；
    耗散仅支持寿命通道（落点=丢失，NonHermitian），显式落点/纯退相
    抛 NotImplementedError（同 pair_evolve v1 限制）。

    ac_ize : t62 目标 6——True 时自动 AC 化：光场远强对（r=Ω/ω>r_crit）
        剔除 + far_sos 位移注入物理能级（E_pair 自动含 δ_i+δ_j）。假设
        A/B 场同构（交换全过程场景）；异构场用 A 侧位移近似（log 提示）。
        默认 False 零行为变化。
    """
    from . import pair_core as _pc
    log = log or (lambda *a: None)
    if t_grid is None:
        raise ValueError('t_grid 必须提供')
    t = np.asarray(t_grid, float)
    T = float(t[-1])
    N = len(states)
    N2 = N * N
    E0 = np.array([s['E_mhz'] for s in states], float)

    # ---- 目标 6：AC 化（远强对剔除 + far_sos 位移注入物理 E）----
    E_shift = np.zeros(N)
    if ac_ize:
        from .evolve_mixed import ac_ize_physical_E, R_CRIT
        E2a, A_newA = ac_ize_physical_E(E0,
                                         [f for f in fields_A
                                          if f['omega_L_mhz'] > MW_MAX_MHZ])
        E2b, A_newB = ac_ize_physical_E(E0,
                                         [f for f in fields_B
                                          if f['omega_L_mhz'] > MW_MAX_MHZ])
        # A/B 同构假设：E_shift 取 A 侧（B 侧位移并入 A 侧——同场时同 δ）
        E_shift = E2a - E0
        if np.max(np.abs((E2b - E0) - E_shift)) > 1.0:
            log(f'[pair_exact] ⚠ A/B 光场 AC 位移不同（max 差 '
                f'{np.max(np.abs((E2b-E0)-E_shift)):.1f} MHz）——取 A 侧'
                f'（E_pair 位移近似；异构场建议 per_freq_M 大 M 或手工 AC 化）')
        n_opt = 0
        fa2, fb2 = [], []
        for f, Anew in zip([f for f in fields_A if f['omega_L_mhz'] > MW_MAX_MHZ],
                           A_newA):
            fa2.append(dict(f, A=Anew)); n_opt += 1
        for f, Anew in zip([f for f in fields_B if f['omega_L_mhz'] > MW_MAX_MHZ],
                           A_newB):
            fb2.append(dict(f, A=Anew))
        # 重组 fields（光场剔除后 + 微波原样）
        fields_A = [f for f in fields_A if f['omega_L_mhz'] <= MW_MAX_MHZ] + fa2
        fields_B = [f for f in fields_B if f['omega_L_mhz'] <= MW_MAX_MHZ] + fb2
        log(f'[pair_exact] AC 化 {n_opt} 组光场：E_shift 范围 '
            f'{E_shift.min():+.2f}..{E_shift.max():+.2f} MHz'
            f'（远强对剔除——t58 r 判据 r_crit={R_CRIT}）')

    groups, _ = _pair_groups(fields_A, fields_B)
    # 维度校验：场耦合矩阵 = N² 两体基
    for g in groups:
        if g['K2_w'][0].shape != (N2, N2):
            raise ValueError('两体耦合矩阵维度与 states 不一致')

    # ---- V_dd（vdd_matrix 优先，否则 pair_core.vdd_sparse）----
    if vdd_matrix is not None:
        V = vdd_matrix
    elif R_um is not None:
        if cache is None:
            raise ValueError(
                '两体演化需要 cache（core.ArcCache 实例）来计算 V_dd；'
                '请传入 cache，或显式提供 vdd_matrix。')
        V = _pc.vdd_sparse(cache, [dict(s) for s in states],
                           R_um, theta_deg, phi_deg)
    else:
        V = None
    if V is not None and not sp.issparse(V):
        V = sp.csr_matrix(np.asarray(V, complex))

    # ---- 耗散（寿命通道 → NonHermitian；速率向量复用 pair_evolve）----
    GammaA = GammaB = None
    has_decay = bool(decays_A or decays_B)
    if has_decay:
        GammaA, GammaB, _ = _pair_loss_vectors(
            states, decays_A, decays_B, N, N2)
        log('[pair_exact] 两体耗散：NonHermitian 寿命通道'
            + (f'（A {int((GammaA > 0).sum())} 态'
               if np.any(GammaA > 0) else '（A 无')
            + ('，B %d 态）' % int((GammaB > 0).sum())
               if np.any(GammaB > 0) else '，B 无）')
            + '，落点=丢失')
    gamma_pair = None
    if has_decay:
        # K 对角 −iΓ_MHz/2 经传播因子 _CONV → 幅衰率 _CONV·Γ_MHz/2，
        # 须等于物理速率 gam_ns/2 → Γ_MHz = gam_ns/_CONV（同 pair_evolve
        # 的 gam/_CONV 约定，gam 单位 /ns）
        gamma_pair = (GammaA + GammaB) / _CONV

    # ---- 初态与能量 ----
    psi0_A_vec = _as_vector(states, psi0_A)
    psi0_B_vec = _as_vector(states, psi0_B)
    psi0_2d = _as_product_psi(psi0_A_vec, psi0_B_vec, N)
    E = np.array([s['E_mhz'] for s in states], float)
    if ac_ize:
        # 目标 6：AC 位移注入物理能级——E_pair[i,j] 含 δ_i+δ_j（A/B 同构）
        E = E + E_shift
    E_pair = np.zeros(N2)
    for i in range(N):
        for j in range(N):
            E_pair[i * N + j] = E[i] + E[j]

    if progress_arr is None:
        progress_arr = np.zeros(3)
    from .coupling import stage_progress
    stage_progress(progress_arr, 0, 0.05)    # 阶段 0：探针/装配

    # ---- M 探针 / 逐载频光子数（t52：per_freq_M 可调）----
    # per_freq_M：每载频组独立光子数 M_g（list，长度 = 载频组数）或 int（全体）
    # ——光场 r=Ω/ω~1e-7 → M=1 即收敛；微波 r~3e-4 → M=1（实测 5.75e-5）或
    # 保守 M=2。None = 均匀 M 探针（历史行为）。
    if per_freq_M is None:
        M, mdiff = _pair_m_probe(E_pair, N, groups, 2, V, gamma_pair,
                                 psi0_2d, T, tol, log=log,
                                 progress=lambda f: stage_progress(
                                     progress_arr, 0, 0.05 + 0.6 * f))
        log(f'[pair_exact] 光子块收敛：M={M}'
            + (f'（探针差 {mdiff:.2e}）' if mdiff == mdiff else '（达上限）'))
    else:
        G = len(groups)
        if isinstance(per_freq_M, (list, tuple, np.ndarray)):
            if len(per_freq_M) != G:
                raise ValueError(
                    f'per_freq_M 长度 {len(per_freq_M)} ≠ 载频组数 {G}')
            M = [int(m) for m in per_freq_M]
        else:
            M = [int(per_freq_M)] * G
        log(f'[pair_exact] 逐载频光子块 M={M}'
            f'（用户指定 per_freq_M；dim = N²·Π(2M_g+1)）')

    if device == 'cuda':
        # t57：GPU 全驻留路径（torch；const 段合并移植自 tempsimu/_t50_full）
        from .evolve_sambe_pair_gpu import gpu_propagate_pair
        setup = _PairSambeSetup(E_pair, N, groups, M, V, gamma_pair,
                                dense_max=10 ** 9)
        stage_progress(progress_arr, 1, 0.0)
        pops_2d, n_eigh = gpu_propagate_pair(
            setup, groups, psi0_2d, t, frozen_h_ns or 8.0, log=log)
        stage_progress(progress_arr, 2, 0.5)
        X_final = None
        report = dict(n_segments=n_eigh, max_richardson_diff=0.0,
                      dt_min_hit=False, max_norm_dev=0.0)
    else:
        pops_2d, X_final, report, setup = _pair_propagate(
            E_pair, N, groups, None, V, gamma_pair, psi0_2d, t, tol, M,
            progress_arr, log, _DENSE_MAX, frozen_h_ns)
        stage_progress(progress_arr, 2, 0.3)  # 阶段 2：收尾（loss/约化）

    # ---- loss 曲线（NonHermitian 速率积分，同 pair_evolve 语义）----
    loss_A = np.zeros(len(t))
    loss_B = np.zeros(len(t))
    if has_decay:
        dt_arr = np.diff(t)
        rate_A = pops_2d @ (GammaA * 1.0)          # (n_t,)，GammaA 已 /ns
        rate_B = pops_2d @ GammaB
        loss_A[1:] = np.cumsum(0.5 * (rate_A[1:] + rate_A[:-1]) * dt_arr)
        loss_B[1:] = np.cumsum(0.5 * (rate_B[1:] + rate_B[:-1]) * dt_arr)
        norm_dev = float(np.max(np.abs(
            pops_2d.sum(axis=1) + loss_A + loss_B - 1.0)))
    else:
        norm_dev = float(np.max(np.abs(pops_2d.sum(axis=1) - 1.0)))
    report['max_norm_dev'] = max(report['max_norm_dev'], norm_dev)

    # ---- 约化布居 ----
    pops_A = _reduce_A(pops_2d, N)
    pops_B = _reduce_B(pops_2d, N)
    stage_progress(progress_arr, 2, 1.0)

    log(f'[pair_exact] 完成：{report["n_segments"]} 段，'
        f'最大残差 {report["max_richardson_diff"]:.2e}'
        f'（tol={tol:.1e}），范数偏差 {report["max_norm_dev"]:.2e}')

    report['photon_blocks'] = {
        f'{g["omega_mhz"]:.6g}MHz': (2 * M + 1 if isinstance(M, int)
                                     else 2 * M[gi] + 1)
        for gi, g in enumerate(groups)}
    report['tol'] = float(tol)

    out = dict(
        t=t,
        pops_2d=pops_2d,
        pops_A=pops_A,
        pops_B=pops_B,
        loss_A=loss_A,
        loss_B=loss_B,
        states_2d=[{
            'label': f"({states[i]['label']}) x ({states[j]['label']})",
            'E_mhz': E_pair[i * N + j],
            'ket_i': i,
            'ket_j': j,
        } for i in range(N) for j in range(N)],
        method='exact',
        backend='sambe-krylov',
        error_report=report,
    )
    if record_amp:
        Xm = np.asarray(X_final, complex).reshape(setup.P, setup.N2)
        ph = np.exp(-1j * setup.nphase * T)
        out['psi_amp'] = ph @ Xm
    return out


# ---- t19 函数级改名兼容别名（旧名 = 新名，外部零改动）----
evolve_exact_pair = evolve_sambe_pair
_PairSambeSetup.K_seg = _PairSambeSetup.hamiltonian_seg
