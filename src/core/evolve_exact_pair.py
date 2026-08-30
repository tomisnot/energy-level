# -*- coding: utf-8 -*-
"""两体第一性原理含时引擎（evolve_exact_pair.py，method='exact' 双原子）。

物理模型（无 secular、无 RWA、无旋转系、无近/远分流）：

    H_2d(t) = [H_A0 + Σ_f Ω_f env_f(t)cos(ω_f t)·d̂_A] ⊗ I
            + I ⊗ [H_B0 + Σ_f Ω_f env_f(t)cos(ω_f t)·d̂_B] + V_dd

V_dd 完整保留（不做 pair_evolve 的 secular 筛选）：近简并交换、远失谐
vdW 通道效应全部由精确传播自然涌现。单原子侧沿用 evolve_exact 约定
（Rabi 周频 Ω/2π、kind='exact' 场、π 脉冲 = 1/(2Ω) µs）。

数值算法：
  1. 两体 Sambe (t,t') 扩展空间：两体基 k=i·N+j（N²），光子维按
     A/B 全部场的相异载频分组（同频场共享光子维），索引 idx = k + N²·p。
  2. 稠密路径（扩展维 ≤ 700）：冻结中点段传播 = 本征分解精确指数
     （绝对容差判厄米，支持 NonHermitian）+ 本征基 Duhamel 缺陷修正。
  3. 稀疏路径（大基组）：对角相互作用绘景 φ = e^{+iD(t−ta)}ψ（D 为含
     光子偏移的静态对角）+ Volterra–Picard 积分（cumsum 结构，逐节点精确
     采样包络）；R 只含非对角（V_dd + 驱动），收敛速度与巨大对角能量
     标度（~THz）无关；节点倍增差 + Picard 残差联合估计，超限真中点细分。
  4. 误差控制：单旋钮 tol；误差随 error_report 可见。
  5. 寿命耗散：pair_evolve._pair_loss_vectors 速率 → NonHermitian 对角
     （−iΓ/2，Γ_MHz = gam_ns/_CONV），loss_A/loss_B 速率×布居积分。

返回契约与 pair_evolve 完全同键（GUI 直接复用）：
    t, pops_2d, pops_A, pops_B, loss_A, loss_B, states_2d, method, backend
另附 error_report（段数/最大残差/范数偏差/光子块）。
"""
import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import LinearOperator, gmres

from . import evolve
from . import sambe

from . import evolve_exact as _ex
from .evolve_exact import _CONV, _gl_nodes

# 扩展维 ≤ 此值走稠密路径（本征分解），否则稀疏（Krylov 子空间）
_DENSE_MAX = 700
# 缺陷求积节点阶梯
_NODE_LADDER = (16, 32, 64, 128)
# 包络采样（活性判断/恒常判断）
_ENV_PROBE_N = 2001
# 光子块阶数上限
_M_MAX = 8
# 细分绝对下限
_DT_MIN_FRAC = 2e5
# Volterra–Picard：GMRES 求解上限/容差，预条件因果扫掠次数与节点阶梯
_VOLT_MAX_ITER = 60
_VOLT_PICARD_FRAC = 0.05
_VOLT_PRE_SWEEPS = 8


# ============================================================
# 场分组（A/B 全部场按相异载频；同频共享光子维）
# ============================================================
def _pair_groups(fields_A, fields_B):
    """返回 groups：[dict(omega_mhz, K2(稀疏 N², 单位包络两体耦合), envs)]。

    A 侧场嵌入为 A_f⊗I，B 侧为 I⊗A_f（kron 顺序 k=i·N+j）。
    """
    N = None
    groups = []
    for fields, side in ((fields_A or [], 'A'), (fields_B or [], 'B')):
        for f in fields:
            if (f or {}).get('kind') != 'exact':
                raise ValueError(
                    "method='exact' 两体要求场为 kind='exact'（由 "
                    'waveforms.build_fields_exact 构造）。')
            A = np.asarray(f['A'], float)
            if N is None:
                N = A.shape[0]
            if A.shape != (N, N):
                raise ValueError('exact 场 A 矩阵维度不一致')
            if not np.any(np.abs(A) > 1e-15):
                continue                       # 无耦合场 → 不占光子维
            Asp = sp.csr_matrix(A)
            I = sp.eye(N, format='csr')
            K2 = (sp.kron(Asp, I) if side == 'A'
                  else sp.kron(I, Asp)).tocsr()
            w = float(f['omega_L_mhz'])
            hit = None
            for g in groups:
                if abs(g['omega_mhz'] - w) <= 1e-9 * max(abs(w), 1.0):
                    hit = g
                    break
            if hit is None:
                hit = dict(omega_mhz=w, K2_w=[K2], envs_w=[f['waveform']])
                groups.append(hit)
            else:
                hit['K2_w'].append(K2)
                hit['envs_w'].append(f['waveform'])
    return groups, N


# ============================================================
# 两体 Sambe 扩展空间（稀疏结构预计算 + 按段装配）
# ============================================================
class _PairSambeSetup:
    """两体 Sambe 结构。索引 idx = k + N²·p（两体基最快）。

    K_static = diag(E_pair + 光子偏移) + V_dd（块对角，逐光子块）
               − i·Γ_pair/2（寿命 NonHermitian）
    Kc_ext_g = 光子邻接（n→n±1，系数 1/2）⊗ 两体单位包络耦合（逐组）。
    """

    def __init__(self, E_pair, N, groups, M, vdd_sp, gamma_pair_mhz):
        self.N = N
        self.N2 = N * N
        self.groups = groups
        self.M = M
        self.D = 2 * M + 1
        geo = sambe.photon_geometry(E_pair, groups, M)
        self.P = geo['P']
        self.dim = geo['dim']
        self.strides = geo['strides']
        self.nvecs = geo['nvecs']
        self.diag = geo['diag']
        self.nphase = geo['nphase']
        self.p_center = geo['p_center']
        _idx = geo['_idx']  # photon grid index (pair Kc_ext construction)
        # 对角 + Γ/2（寿命 NonHermitian）— V_dd 在调用方处理
        if gamma_pair_mhz is not None and np.any(gamma_pair_mhz):
            self.diag = self.diag.astype(complex)
            self.diag = self.diag - 0.5j * np.tile(
                np.asarray(gamma_pair_mhz, float), self.P)
        # V_dd 块对角嵌入（kron(I_P, V_dd)：k + N²·p 顺序 ✓）
        if vdd_sp is None:
            vdd_ext = None
        elif sp.issparse(vdd_sp):
            vdd_ext = (sp.kron(sp.eye(self.P, format='csr'),
                               vdd_sp.tocsr().astype(complex))
                       if vdd_sp.nnz else None)
        else:
            arr = np.asarray(vdd_sp, complex)
            vdd_ext = (sp.kron(sp.eye(self.P, format='csr'),
                               sp.csr_matrix(arr))
                       if np.any(arr != 0) else None)
        K_static = sp.diags(self.diag, format='csr')
        if vdd_ext is not None:
            K_static = (K_static + vdd_ext).tocsr()
        self.K_static = _realify(K_static)
        # 逐组逐波形光子邻接耦合：kron(T_g + T_gᵀ, 0.5·K2_w_gf)
        self.Kc_ext = [[] for _ in groups]
        for gi, g in enumerate(groups):
            stride = self.strides[gi]
            rs, cs = [], []
            for s in range(self.D - 1):
                p_src = _idx[(_idx // stride) % self.D == s]
                rs.append(p_src + stride)
                cs.append(p_src)
            T = sp.coo_matrix(
                (np.ones(np.concatenate(rs).size),
                 (np.concatenate(rs), np.concatenate(cs))),
                shape=(self.P, self.P)).tocsr()
            for K2w in g['K2_w']:
                Kh = 0.5 * K2w.astype(complex)
                Kcw = _realify(
                    (sp.kron(T, Kh) + sp.kron(T.T, Kh.T)).tocsr())
                self.Kc_ext[gi].append(Kcw)
        self.dense = self.dim <= _DENSE_MAX
        if self.dense:
            self.K_static = K_static.toarray()
            self.Kc_ext = [[Kc.toarray() if not sp.issparse(Kc) else Kc
                            for Kc in wl]
                           for wl in self.Kc_ext]

    def _c_mids_wf(self, groups, mid):
        """逐波形中点包络和——返回 flat list: [g0w0, g0w1, ..., gNwM]。"""
        result = []
        for g in groups:
            for wf in g['envs_w']:
                result.append(float(
                    np.asarray(wf(np.atleast_1d(mid))).flat[0]))
        return result

    def _c_at_nodes_wf(self, groups, nodes):
        """逐波形节点包络和——返回 (total_wf, n_nodes)。"""
        nodes = np.asarray(nodes, float)
        rows = []
        for g in groups:
            for wf in g['envs_w']:
                rows.append(np.asarray(wf(nodes), float))
        return np.array(rows, float)

    def K_seg(self, mid, groups=None):
        """段算符 K_static + Σ_g Σ_w env_w(mid)·Kcw_{gw}（稀疏/稠密）。
        
        统一逐波实现，无分支。单波形组自然退化为等效计算。
        """
        K = self.K_static
        if groups is None:
            groups = self.groups
        for gi, g in enumerate(groups):
            Kcw_list = self.Kc_ext[gi]
            if not Kcw_list:
                continue
            first_Kcw = Kcw_list[0]
            if isinstance(first_Kcw, np.ndarray):
                Kc_sum_g = np.zeros_like(first_Kcw)
            else:
                Kc_sum_g = sp.csr_matrix(first_Kcw.shape, dtype=complex)
            active_g = False
            for wf, Kcw in zip(g['envs_w'], Kcw_list):
                c_w = float(np.asarray(wf(np.atleast_1d(mid))).flat[0])
                if c_w != 0.0:
                    Kc_sum_g = Kc_sum_g + c_w * Kcw
                    active_g = True
            if active_g:
                K = K + Kc_sum_g
        if self.dense:
            return ('dense', K)
        return ('sparse', K.tocsr())


def _realify(M):
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


def _embed_pair(psi_2d, setup):
    X = np.zeros(setup.dim, complex)
    off = setup.N2 * setup.p_center
    X[off:off + setup.N2] = psi_2d
    return X


def _physical_pops(setup, X, tq):
    """扩展态 → 两体物理布居（光子块相位投影）。"""
    Xm = np.asarray(X, complex).reshape(setup.P, setup.N2)
    ph = np.exp(-1j * setup.nphase * float(tq))
    psi = ph @ Xm
    return np.abs(psi) ** 2


def _dense_decomp(K):
    """稠密段算符本征分解。强制转为 ndarray 避免 numpy.matrix 广播问题。"""
    K_arr = np.asarray(K)
    if np.allclose(K_arr, K_arr.conj().T, rtol=0.0, atol=1e-10):
        evals, evecs = np.linalg.eigh(K_arr)
        return evals, evecs, evecs.conj().T
    evals, evecs = np.linalg.eig(K_arr)
    return evals, evecs, np.linalg.inv(evecs)

def _prop_step(setup, K_seg, ta, tb, X):
    """冻结段传播（稠密路径：本征分解精确指数）。"""
    dt = tb - ta
    evals, evecs, left = _dense_decomp(K_seg[1])
    a0 = np.asarray(left @ X).ravel()
    phase = np.asarray(np.exp(-1j * _CONV * dt * evals)).ravel()
    return np.asarray(evecs) @ (a0 * phase)

def _defect_pass(setup, K_seg, ta, tb, X, groups, nodes, ws, dense_eig):
    """本征基 Duhamel 缺陷扫描。统一逐波实现：ΔK = Σ_g Σ_w (env_w(nodes)-env_w(mid))·Kcw_{gw}。"""
    n = len(nodes)
    mid = 0.5 * (ta + tb)
    
    # Build per-waveform envelope diff arrays: dict((gi,wi) -> (n,) array)
    dc_wf_dict = {}
    max_dc = 0.0
    idx_wf = 0
    for gi, g in enumerate(groups):
        Kcw_list = setup.Kc_ext[gi]
        for wi, wf in enumerate(g['envs_w']):
            env_vals = np.asarray(wf(nodes), float)
            env_mid = float(np.asarray(
                wf(np.atleast_1d(mid)), float).flat[0])
            dc_wf = env_vals - env_mid
            dc_wf_dict[(gi, wi)] = dc_wf
            dmax = float(np.abs(dc_wf).max())
            if dmax > max_dc:
                max_dc = dmax
            idx_wf += 1
    if max_dc < 1e-12:
        return np.zeros(setup.dim, complex), max_dc
    
    evals, evecs, left = dense_eig
    a0 = np.asarray(left @ X).ravel()
    phi_tq = np.exp(-1j * _CONV * np.outer(evals, nodes - ta))
    Psif = np.asarray(evecs) @ (a0[:, None] * phi_tq)
    
    delta = np.zeros(setup.dim, complex)
    for gi, g in enumerate(groups):
        Kcw_list = setup.Kc_ext[gi]
        if not Kcw_list:
            continue
        for q in range(n):
            v = np.zeros(setup.dim, complex)
            for wi in range(len(Kcw_list)):
                dc_w = dc_wf_dict[(gi, wi)]
                if abs(dc_w[q]) > 1e-14:
                    Kcw_dense = Kcw_list[wi].toarray() if sp.issparse(Kcw_list[wi]) else Kcw_list[wi]
                    v = v + dc_w[q] * (Kcw_dense @ Psif[:, q])
            if np.any(v != 0.0):
                b0 = np.asarray(left @ v).ravel()
                up = np.asarray(evecs) @ (b0 * np.exp(
                    -1j * _CONV * evals * (tb - nodes[q])))
                delta = delta + ws[q] * up
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
                  T, tol, log=None):
    """M 递增收敛：常数峰值包络传播，M vs M+1 布居差 ≤ tol。
    稀疏路径：探针窗口取 max(10ns, T/4)（截断泄漏随时间增长，窗口过短会
    低估所需 M），按 1ns 分段链式子空间传播（与主循环同一机制）。"""
    def run_at(Mc):
        stp = _PairSambeSetup(E_pair, N, groups, Mc, vdd_sp, gamma_pair_mhz)
        c_list = []
        for g in groups:
            pk = 0.0
            for wf in g['envs_w']:
                v = np.asarray(wf(np.linspace(0.0, T, _ENV_PROBE_N)), float)
                pk = max(pk, float(np.max(np.abs(v))))
            c_list.append(pk)
        K = stp.K_seg(T / 2.0 if T > 0 else 1.0, groups)
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
    while True:
        cur, ext = run_at(Mc)
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
# 主入口
# ============================================================
def evolve_exact_pair(states, fields_A, fields_B, psi0_A, psi0_B, t_grid,
                      vdd_matrix=None, R_um=None, theta_deg=0.0,
                      phi_deg=0.0, cache=None, tol=1e-6,
                      decays_A=None, decays_B=None,
                      progress_arr=None, log=None, record_amp=False):
    """两体第一性原理含时演化（完整 V_dd，无 secular/RWA/旋转系）。

    签名与返回契约对齐 pair_evolve.pair_evolve（GUI 直接复用）；
    耗散仅支持寿命通道（落点=丢失，NonHermitian），显式落点/纯退相
    抛 NotImplementedError（同 pair_evolve v1 限制）。
    """
    from . import pair_evolve as _pe
    from . import pair_core as _pc
    log = log or (lambda *a: None)
    if t_grid is None:
        raise ValueError('t_grid 必须提供')
    t = np.asarray(t_grid, float)
    T = float(t[-1])
    N = len(states)
    N2 = N * N

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
        GammaA, GammaB, _ = _pe._pair_loss_vectors(
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
    psi0_A_vec = evolve._as_vector(states, psi0_A)
    psi0_B_vec = evolve._as_vector(states, psi0_B)
    psi0_2d = _pe._as_product_psi(psi0_A_vec, psi0_B_vec, N)
    E = np.array([s['E_mhz'] for s in states], float)
    E_pair = np.zeros(N2)
    for i in range(N):
        for j in range(N):
            E_pair[i * N + j] = E[i] + E[j]

    if progress_arr is None:
        progress_arr = np.zeros(2)

    # ---- M 探针 ----
    M, mdiff = _pair_m_probe(E_pair, N, groups, 2, V, gamma_pair,
                             psi0_2d, T, tol, log=log)
    log(f'[pair_exact] 光子块收敛：M={M}'
        + (f'（探针差 {mdiff:.2e}）' if mdiff == mdiff else '（达上限）'))

    setup = _PairSambeSetup(E_pair, N, groups, M, V, gamma_pair)

    # ---- 全常数包络快速路径（稠密）----
    all_const = True
    c_peaks = []
    for g in groups:
        cs = []
        for wf in g['envs_w']:
            v = np.asarray(wf(np.linspace(0.0, T, 64)), float)
            if np.ptp(v) > 1e-9:
                all_const = False
            cs.append(float(v.mean()))
        c_peaks.append(sum(cs))
    report = dict(n_segments=0, max_richardson_diff=0.0, dt_min_hit=False,
                  max_norm_dev=0.0)

    def _record_chain(X, tq):
        return _physical_pops(setup, X, tq)

    if (all_const or not groups) and setup.dense:
        K = setup.K_seg(T / 2.0 if T > 0 else 1.0, groups)
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
    else:
        # ---- 自适应分段主循环（稠密：本征基缺陷；稀疏：IP Volterra）----
        pops_2d = np.zeros((len(t), N2))
        h_floor = max(T / _DT_MIN_FRAC, 1e-9)

        def seg_K(ta, tb):
            return setup.K_seg(0.5 * (ta + tb), groups)

        def rec2(ta, tb, X):
            if progress_arr[1] != 0.0:
                raise RuntimeError('用户取消')
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
                    c_nodes_wf = setup._c_at_nodes_wf(groups, nodes)
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
                tm = 0.5 * (ta + tb)
                Xm = rec2(ta, tm, X)
                return rec2(tm, tb, Xm)
            report['max_richardson_diff'] = max(
                report['max_richardson_diff'],
                float(est) if np.isfinite(est) else 1e3)
            report['n_segments'] += 1
            if not np.isfinite(est) or est > tol:
                report['dt_min_hit'] = True
            return psi_out

        X = _embed_pair(psi0_2d, setup)
        tprev = 0.0
        for i in range(1, len(t)):
            progress_arr[0] = min(tprev / T, 0.999)
            X = rec2(tprev, float(t[i]), X)
            pops_2d[i] = _record_chain(X, t[i])
            tprev = float(t[i])
        pops_2d[0] = np.abs(psi0_2d) ** 2
        X_final = X

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
    pops_A = _pe._reduce_A(pops_2d, N)
    pops_B = _pe._reduce_B(pops_2d, N)

    log(f'[pair_exact] 完成：{report["n_segments"]} 段，'
        f'最大残差 {report["max_richardson_diff"]:.2e}'
        f'（tol={tol:.1e}），范数偏差 {report["max_norm_dev"]:.2e}')

    report['photon_blocks'] = {f'{g["omega_mhz"]:.6g}MHz': 2 * M + 1
                               for g in groups}
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
