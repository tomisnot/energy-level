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

两场变体（t30，路径 A）：`build_two_field_pair` + `evolve_two_field_pair`
——光场载波由旋转系吸收、不进光子块，只有微波占光子维 → 22 态 dim=2420
稠密可行（替代全载频进块的 60500 维不可用路径）。见模块内"两场扩展"注释块。
"""
import numpy as np
import scipy.sparse as sp
import time as _time
from scipy.sparse.linalg import LinearOperator, gmres

from . import evolve
from . import sambe

from . import evolve_sambe as _ex
from .evolve_sambe import _CONV, _gl_nodes

# 扩展维 ≤ 此值走稠密路径（本征分解），否则稀疏（Krylov 子空间）
_DENSE_MAX = 700
_SPARSE_SEG_MAX = 50000         # EP6 保护：稀疏自适应细分总段数上限（防爆炸卡死）
_SPARSE_MAX_DEPTH = 4           # EP6 保护：稀疏自适应二分递归深度上限（正常 est 深度 ≤3）
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
# 两场扩展（t30）：光场旋转系 + 微波-only 光子块的两体变体
# ============================================================
# 背景（docs/核心资产盘点表.md §五 / 任务书 §14 路径A）：两体 Sambe 把所有
# 载频（含 THz 光场）塞进光子块 → dim = N²·Π(2M+1) 达 60500，不可用。两场
# 变体把光场载波用旋转系解析吸收（同单原子 evolve_twofield），光子块只承载
# 微波 → dim = N²·P_mw = 484×5 = 2420，稠密 eigh 可行。
#
# 旋转系约定（与 evolve_twofield.build_two_field 逐字对齐）：
#   E_rot = E + ω_opt·1_low（低能流形整体抬升），光场耦合项 = ½·Ω（RWA 共振项）；
#   两体 E_pair_rot[i·N+j] = E_rot_A[i] + E_rot_B[j]，光场按侧嵌入 A⊗I / I⊗A。
#
# V_dd 在旋转系中的处理（本变体唯一的额外近似，与光学 RWA 同级）：
#   Ṽ_dd[r,c] = V[r,c]·exp(iC·(ΔE_lab − ΔE_rot)·t)
#   —— r/c 的"光场旋转流形占据数"相同（ΔE_rot = ΔE_lab）→ 静态保留，含最
#      重要的 |60S,60P⟩↔|60P,60S⟩ 交换（ΔE_lab=0，两侧均无 6P 占据）；
#   —— 相差 ±1（如 |60S,60S⟩↔|60S,6P⟩）→ 以 ω_opt（THz）快振荡，RWA 丢弃；
#      净效应 ≤ |V|²/Δ ~ (4 MHz)²/3e8 MHz ~ 5e-8 MHz（同光学反旋项论证）。
#      否则就得给光场开光子块，本变体失去意义。
# ============================================================

# 两场变体稠密上限：22 态 N²·P = 484×5 = 2420 的稠密 eigh 可行（秒级/段）
_TF_DENSE_MAX = 3000
# 微波近共振带宽（≡ evolve_twofield.MW_NRES_BW_MHZ）
_TF_MW_NRES_BW_MHZ = 5e3
# V_dd 旋转系保留判据：|ΔE_rot − ΔE_lab| ≤ 此值（MHz）视为静态
_VDD_RWA_TOL_MHZ = 1.0


def _embed2d(A, side, N):
    """单原子 A（N×N）→ 两体嵌入（k=i·N+j）：'A' 侧 A⊗I，'B' 侧 I⊗A。"""
    Asp = sp.csr_matrix(np.asarray(A, float))
    I = sp.eye(N, format='csr')
    return (sp.kron(Asp, I) if side == 'A' else sp.kron(I, Asp)).tocsr()


def build_two_field_pair(meta_A, meta_B, log=None):
    """A/B 两侧 build_two_field 输出 → 两体两场 meta。

    返回 dict：
      N, N2      : 单体 / 两体维度
      E_pair_rot : (N²,) 旋转系两体能量 E_rot_A[i] + E_rot_B[j]（k = i·N+j）
      mw_groups  : 微波组（A/B 按载频合并，同频共享光子维）
                   dict(omega_mhz, K2_w=[两体稀疏], envs_w=[wf], labels=[])
      opt_terms  : 光场项 [dict(Ko=两体稀疏(N²,N²), wf, label, side)]
    """
    log = log or (lambda *a: None)
    N = int(meta_A['N'])
    if int(meta_B['N']) != N:
        raise ValueError('两侧 meta 的 N 不一致')
    E_A = np.asarray(meta_A['E_rot'], float)
    E_B = np.asarray(meta_B['E_rot'], float)
    E_pair_rot = (E_A[:, None] + E_B[None, :]).ravel()

    groups, opt_terms = [], []
    for side, meta in (('A', meta_A), ('B', meta_B)):
        # 光场：逐侧两体嵌入（旋转系内为常数矩阵，不占光子块）
        for tup in meta['opt_terms']:
            A_opt, wf = tup[0], tup[1]
            omega_f = float(tup[2])
            name = tup[3] if len(tup) > 3 else 'opt'
            omega_ref = float(tup[4]) if len(tup) > 4 else omega_f
            if abs(omega_f - omega_ref) > 1e-9 * max(abs(omega_f), 1.0):
                raise NotImplementedError(
                    f'两体两场 Sambe 不支持多光场拍频：光场 {name!r} '
                    f'ω={omega_f:.6g} MHz 与共享旋转参考 {omega_ref:.6g} MHz '
                    f'相差 {omega_f - omega_ref:+.6g} MHz（拍频相位 '
                    'exp(i·C·δ·t) 无法在冻结段+包络缺陷修正中表示）。'
                    '请改用 hard 引擎（evolve_hard_pair）。')
            opt_terms.append(dict(Ko=_embed2d(A_opt, side, N), wf=wf,
                                  label=f'{side}:{name}', side=side))
        # 微波：按载频合并（A/B 同频共享光子维，同 _pair_groups 语义）
        for g in meta['mw_groups']:
            w = float(g['omega_mhz'])
            labels = g.get('labels') or [''] * len(g['As'])
            for A, wf, lab in zip(g['As'], g['envs'], labels):
                hit = None
                for hg in groups:
                    if abs(hg['omega_mhz'] - w) <= 1e-9 * max(abs(w), 1.0):
                        hit = hg
                        break
                if hit is None:
                    hit = dict(omega_mhz=w, K2_w=[], envs_w=[], labels=[])
                    groups.append(hit)
                hit['K2_w'].append(_embed2d(A, side, N))
                hit['envs_w'].append(wf)
                hit['labels'].append(f'{side}:{lab}')
    log(f'[two_field_pair] 微波组 {len(groups)}'
        f'（{sum(len(g["envs_w"]) for g in groups)} 波形）'
        f'，光场项 {len(opt_terms)}')
    return dict(N=N, N2=N * N, E_pair_rot=E_pair_rot, mw_groups=groups,
                opt_terms=opt_terms)


def vdd_rot_filter(V, E_pair_rot, E_pair_lab, log=None,
                   tol_mhz=_VDD_RWA_TOL_MHZ):
    """V_dd 旋转系 RWA 筛选：只保留 ΔE_rot = ΔE_lab 的元素（旋转系内静态）。

    返回 (V_filtered, info)；info 记录丢弃元素的数量 / max|V| / 净效应上界
    （|V|²/Δ，Δ 取丢弃元素的实验室系失谐）——供调用方日志与自检核对。
    """
    info = dict(n_keep=0, n_drop=0, max_v_drop=0.0, max_effect_mhz=0.0)
    if V is None:
        return None, info
    coo = (V.tocsr().astype(complex) if sp.issparse(V)
           else sp.csr_matrix(np.asarray(V, complex))).tocoo()
    if coo.nnz == 0:
        return coo.tocsr(), info
    d_rot = E_pair_rot[coo.row] - E_pair_rot[coo.col]
    d_lab = E_pair_lab[coo.row] - E_pair_lab[coo.col]
    keep = np.abs(d_rot - d_lab) <= tol_mhz
    info['n_keep'] = int(keep.sum())
    info['n_drop'] = int((~keep).sum())
    if info['n_drop']:
        v_drop = np.abs(coo.data[~keep])
        det = np.abs(d_lab[~keep])
        info['max_v_drop'] = float(v_drop.max())
        info['max_effect_mhz'] = float(
            np.max(v_drop ** 2 / np.maximum(det, 1e-9)))
        if log is not None:
            log(f'[two_field_pair] V_dd 旋转系 RWA：丢弃 {info["n_drop"]}/'
                f'{coo.nnz} 个元素（旋转流形占据数变化 ±1 → 以 ω_opt 振荡；'
                f'max|V|={info["max_v_drop"]:.3g} MHz，净效应 ≤ '
                f'|V|²/Δ = {info["max_effect_mhz"]:.3g} MHz）')
    Vf = sp.coo_matrix((coo.data[keep], (coo.row[keep], coo.col[keep])),
                       shape=coo.shape).tocsr()
    return Vf, info


def _pair_mw_Ms(E_pair, groups, tol, bw=_TF_MW_NRES_BW_MHZ, m_min=2,
                m_max=_M_MAX):
    """逐微波载频解析光子块阶数（同单原子两场 evolve_two_field 的判据）。

    近共振峰值 r = max|A_近共振| / ω；M 取满足 (r/4)^{2(M−1)} ≤ tol 的最小值，
    下限 m_min=2（多光子显式保留）。
    """
    Ms = []
    for g in groups:
        r = 0.0
        for K2 in g['K2_w']:
            coo = K2.tocoo()
            if coo.nnz == 0:
                continue
            det = np.abs((E_pair[coo.row] - E_pair[coo.col])
                         - g['omega_mhz'])
            m = det < bw
            if m.any():
                r = max(r, float(np.abs(coo.data[m]).max())
                        / max(abs(g['omega_mhz']), 1e-9))
        M = m_min
        while (r / 4.0) ** (2 * (M - 1)) > tol and M < m_max:
            M += 1
        Ms.append(M)
    return Ms


# ============================================================
# 两体 Sambe 扩展空间（稀疏结构预计算 + 按段装配）
# ============================================================
class _PairSambeSetup:
    """两体 Sambe 结构。索引 idx = k + N²·p（两体基最快）。

    K_static = diag(E_pair + 光子偏移) + V_dd（块对角，逐光子块）
               − i·Γ_pair/2（寿命 NonHermitian）
    Kc_ext_g = 光子邻接（n→n±1，系数 1/2）⊗ 两体单位包络耦合（逐组）。
    """

    def __init__(self, E_pair, N, groups, M, vdd_sp, gamma_pair_mhz,
                 opt_terms=None, dense_max=None):
        self.N = N
        self.N2 = N * N
        self.groups = groups
        self.M = M
        # 稠密上限：默认 _DENSE_MAX（全载频进光子块，维数易爆炸）；两场变体
        # 显式传 _TF_DENSE_MAX（光场不占光子块 → 2420 维稠密可行）
        self.dense_max = _DENSE_MAX if dense_max is None else int(dense_max)
        geo = sambe.photon_geometry(E_pair, groups, M)
        # 逐组块数（M 可为 int 或逐载频 list；photon_geometry 统一展开）
        self.Ds = list(geo['Ds'])
        self.D = 2 * self.Ds[0] + 1 if len(set(self.Ds)) == 1 else None
        self.P = geo['P']
        self.dim = geo['dim']
        self.strides = geo['strides']
        self.nvecs = geo['nvecs']
        self.diag = geo['diag']
        self.nphase = geo['nphase']
        self.p_center = geo['p_center']
        # 光学场矩阵（不进光子块）：build_two_field_pair 已按侧张量化为
        # A_opt⊗I / I⊗A_opt（N²×N²），此处只需沿光子块复制 kron(I_P, ·)。
        # （t30 修正：旧写法在这里二次张量化成 A⊗I + I⊗A，且未按 P 扩展，
        #   与 K 的 (N²·P)² 维度不相容——该分支从未被调用，故未暴露。）
        self.opt_Ko = []
        self.opt_envs_w = []
        self.opt_labels = []
        for ot in (opt_terms or []):
            Ko2 = ot['Ko']
            Ko2 = (Ko2.tocsr().astype(complex) if sp.issparse(Ko2)
                   else sp.csr_matrix(np.asarray(Ko2, complex)))
            self.opt_Ko.append(
                sp.kron(sp.eye(self.P, format='csr'), Ko2).tocsr())
            self.opt_envs_w.append(ot['wf'])
            self.opt_labels.append(ot.get('label', 'opt'))
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
            Dg = self.Ds[gi]
            rs, cs = [], []
            for s in range(Dg - 1):
                p_src = _idx[(_idx // stride) % Dg == s]
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
        self.dense = self.dim <= self.dense_max
        if self.dense:
            # 段算符装配用稠密副本；Kc_ext/opt_Ko 本体保持稀疏——缺陷修正
            # 走稀疏 matvec（nnz 极小），比稠密 (dim²×n_nodes) 快 2 个量级。
            self.K_static = K_static.toarray()
            self.Kc_dense = [[Kc.toarray() if sp.issparse(Kc)
                              else np.asarray(Kc)
                              for Kc in wl] for wl in self.Kc_ext]
            self.opt_Ko_d = [Ko.toarray() for Ko in self.opt_Ko]
        else:
            self.Kc_dense = None
            self.opt_Ko_d = None

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

    def hamiltonian_seg(self, mid, groups=None):
        """段算符 K_static + Σ_g Σ_w env_w(mid)·Kcw_{gw} + Σ_o env_o(mid)·Ko_o。

        统一逐波实现（同频合并 bug 的修复点：env_w × Kcw_w 逐波形配对）。
        光学场不进光子块，按 kron(I_P, A_opt⊗I / I⊗A_opt) 注入（t30）。
        """
        if groups is None:
            groups = self.groups
        if self.dense:
            K = np.array(self.K_static, copy=True)
            for gi, g in enumerate(groups):
                for wf, Kcw in zip(g['envs_w'], self.Kc_dense[gi]):
                    c_w = float(np.asarray(wf(np.atleast_1d(mid))).flat[0])
                    if c_w != 0.0:
                        K = K + c_w * Kcw
            for Ko, wf in zip(self.opt_Ko_d, self.opt_envs_w):
                c_o = float(np.asarray(wf(np.atleast_1d(mid))).flat[0])
                if c_o != 0.0:
                    K = K + c_o * Ko
            return ('dense', K)
        K = self.K_static.copy()
        for gi, g in enumerate(groups):
            for wf, Kcw in zip(g['envs_w'], self.Kc_ext[gi]):
                c_w = float(np.asarray(wf(np.atleast_1d(mid))).flat[0])
                if c_w != 0.0:
                    K = K + c_w * Kcw
        for Ko, wf in zip(self.opt_Ko, self.opt_envs_w):
            c_o = float(np.asarray(wf(np.atleast_1d(mid))).flat[0])
            if c_o != 0.0:
                K = K + c_o * Ko
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

def _prop_step(setup, hamiltonian_seg, ta, tb, X):
    """冻结段传播（稠密路径：本征分解精确指数）。"""
    dt = tb - ta
    evals, evecs, left = _dense_decomp(hamiltonian_seg[1])
    a0 = np.asarray(left @ X).ravel()
    phase = np.asarray(np.exp(-1j * _CONV * dt * evals)).ravel()
    return np.asarray(evecs) @ (a0 * phase)

def _defect_pass(setup, hamiltonian_seg, ta, tb, X, groups, nodes, ws, dense_eig):
    """本征基 Duhamel 缺陷扫描。

    ΔK = Σ_g Σ_w (env_w(nodes)−env_w(mid))·Kcw_{gw}
         + Σ_o (env_o(nodes)−env_o(mid))·Ko_o      ← t30 补光场项（此前缺，
           两场变体下光场包络变化完全未被修正，只剩冻结中点零阶精度）

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
                  - float(np.asarray(wf(np.atleast_1d(mid)), float).flat[0]))
            dmax = float(np.abs(dc).max())
            max_dc = max(max_dc, dmax)
            if dmax > 1e-14:
                terms.append((setup.Kc_ext[gi][wi], dc))
    for Ko, wf in zip(setup.opt_Ko, setup.opt_envs_w):
        dc = (np.asarray(wf(nodes), float)
              - float(np.asarray(wf(np.atleast_1d(mid)), float).flat[0]))
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
    setup = _PairSambeSetup(E_pair_rot, N, groups, M, V, gamma_pair,
                            opt_terms=opt_terms, dense_max=dense_max)
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
        return pops_2d, X_final, report, setup

    # ---- 自适应分段主循环（稠密：本征基缺陷；稀疏：IP Volterra）----
    pops_2d = np.zeros((len(t), N2))
    h_floor = max(T / _DT_MIN_FRAC, 1e-9)

    def seg_K(ta, tb):
        return setup.hamiltonian_seg(0.5 * (ta + tb), groups)

    # t57b：frozen 的 eigh 复用缓存——K 恒定相邻段（包络值相同）共享分解
    # （引擎级 const 合并：square 平台/全关区的段不再重复 eigh；仅无耗散
    #  Hermitian 路径——耗散 NonHermitian 不缓存）
    _eigh_key = [None]
    _eigh_val = [None]       # (evals, evecs)
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
            # t48：纯冻结——固定 h 内部细分（无缺陷修正；THz 对角由 eigh 精确
            # 承载，误差 = 包络段内变化，与 THz 解耦）
            nsub = max(1, int(np.ceil((tb - ta) / float(frozen_h_ns))))
            hh = (tb - ta) / nsub
            report['n_segments'] += nsub
            for _s in range(nsub):
                t1 = ta + _s * hh
                mid = t1 + 0.5 * hh
                envs_key = tuple(
                    round(float(np.asarray(wf(np.atleast_1d(mid))).flat[0]),
                          12)
                    for g in groups for wf in g['envs_w'])
                use_cache = (_eigh_key[0] == envs_key
                             and _eigh_val[0] is not None)
                if use_cache:
                    evals, evecs = _eigh_val[0]
                else:
                    K = seg_K(t1, t1 + hh)
                    if K[0] == 'sparse':
                        # 冻结需稠密 eigh——稀疏 K 稠密化
                        Kd = K[1].toarray()
                    else:
                        Kd = K[1]
                    if np.iscomplexobj(Kd) and np.any(np.imag(Kd) != 0):
                        # 耗散（NonHermitian）→ 通用分解（不缓存）
                        evals, evecs, left = _dense_decomp(Kd)
                        X = evecs @ ((left @ X)
                                     * np.exp(-1j * _CONV * hh * evals))
                        continue
                    Kd = np.real(Kd)
                    evals, evecs = np.linalg.eigh(Kd)
                    _eigh_key[0] = envs_key
                    _eigh_val[0] = (evals, evecs)
                a0 = evecs.conj().T @ X
                X = evecs @ (a0 * np.exp(-1j * _CONV * hh * evals))
            return X
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
    from . import pair_evolve as _pe
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
                                          if f['omega_L_mhz'] > 1e5])
        E2b, A_newB = ac_ize_physical_E(E0,
                                         [f for f in fields_B
                                          if f['omega_L_mhz'] > 1e5])
        # A/B 同构假设：E_shift 取 A 侧（B 侧位移并入 A 侧——同场时同 δ）
        E_shift = E2a - E0
        if np.max(np.abs((E2b - E0) - E_shift)) > 1.0:
            log(f'[pair_exact] ⚠ A/B 光场 AC 位移不同（max 差 '
                f'{np.max(np.abs((E2b-E0)-E_shift)):.1f} MHz）——取 A 侧'
                f'（E_pair 位移近似；异构场建议 per_freq_M 大 M 或手工 AC 化）')
        n_opt = 0
        fa2, fb2 = [], []
        for f, Anew in zip([f for f in fields_A if f['omega_L_mhz'] > 1e5],
                           A_newA):
            fa2.append(dict(f, A=Anew)); n_opt += 1
        for f, Anew in zip([f for f in fields_B if f['omega_L_mhz'] > 1e5],
                           A_newB):
            fb2.append(dict(f, A=Anew))
        # 重组 fields（光场剔除后 + 微波原样）
        fields_A = [f for f in fields_A if f['omega_L_mhz'] <= 1e5] + fa2
        fields_B = [f for f in fields_B if f['omega_L_mhz'] <= 1e5] + fb2
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
    if ac_ize:
        # 目标 6：AC 位移注入物理能级——E_pair[i,j] 含 δ_i+δ_j（A/B 同构）
        E = E + E_shift
    E_pair = np.zeros(N2)
    for i in range(N):
        for j in range(N):
            E_pair[i * N + j] = E[i] + E[j]

    if progress_arr is None:
        progress_arr = np.zeros(2)

    # ---- M 探针 / 逐载频光子数（t52：per_freq_M 可调）----
    # per_freq_M：每载频组独立光子数 M_g（list，长度 = 载频组数）或 int（全体）
    # ——光场 r=Ω/ω~1e-7 → M=1 即收敛；微波 r~3e-4 → M=1（实测 5.75e-5）或
    # 保守 M=2。None = 均匀 M 探针（历史行为）。
    if per_freq_M is None:
        M, mdiff = _pair_m_probe(E_pair, N, groups, 2, V, gamma_pair,
                                 psi0_2d, T, tol, log=log)
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
        pops_2d, n_eigh = gpu_propagate_pair(
            setup, groups, psi0_2d, t, frozen_h_ns or 8.0, log=log)
        X_final = None
        report = dict(n_segments=n_eigh, max_richardson_diff=0.0,
                      dt_min_hit=False, max_norm_dev=0.0)
    else:
        pops_2d, X_final, report, setup = _pair_propagate(
            E_pair, N, groups, None, V, gamma_pair, psi0_2d, t, tol, M,
            progress_arr, log, _DENSE_MAX, frozen_h_ns)

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


def evolve_two_field_pair(states, fields_meta_A, fields_meta_B, psi0_A,
                          psi0_B, t_grid, vdd_matrix=None, R_um=None,
                          theta_deg=0.0, phi_deg=0.0, cache=None, tol=1e-6,
                          mw_M=None, decays_A=None, decays_B=None,
                          progress_arr=None, log=None, record_amp=False,
                          dense_max=None, frozen_h_ns=None):
    """两体两场引擎：光场旋转系 + 微波-only Sambe 光子块（t30，路径 A）。

    与 evolve_sambe_pair 的差别只在"场怎么进扩展空间"：
      - 光场（>100 GHz，自适应阈值见 evolve_twofield.split_by_freq）：载波由
        旋转系解析吸收 → 不占光子块，耦合项为常数矩阵（RWA 共振项 ½Ω）；
      - 微波：Sambe 光子块（M≥2，多光子显式），A/B 同频共享光子维。
    ⟹ dim = N²·P_mw（22 态 = 484×5 = 2420，稠密可行），替代原两体 Sambe
       把光场也塞进光子块导致的 60500 维不可用路径。

    参数：
      fields_meta_A/B : build_two_field(cache, states, waves) 的输出（各原子）
      mw_M            : None → 逐载频解析判据（M≥2）；int/list → 强制指定
      其余（vdd/R_um/decays/progress）与 evolve_sambe_pair 同义。

    返回契约与 evolve_sambe_pair 完全同键（method='two_field'），
    另附 error_report['vdd_rwa']（V_dd 旋转系筛选统计）。
    """
    from . import pair_core as _pc
    from . import pair_evolve as _pe
    log = log or (lambda *a: None)
    if t_grid is None:
        raise ValueError('t_grid 必须提供')
    t = np.asarray(t_grid, float)
    T = float(t[-1])
    N = len(states)
    N2 = N * N

    meta = build_two_field_pair(fields_meta_A, fields_meta_B, log=log)
    groups = meta['mw_groups']
    opt_terms = meta['opt_terms']
    E_pair_rot = meta['E_pair_rot']
    E = np.array([s['E_mhz'] for s in states], float)
    E_pair_lab = (E[:, None] + E[None, :]).ravel()

    # ---- V_dd（同 evolve_sambe_pair；再做旋转系 RWA 筛选）----
    if vdd_matrix is not None:
        V = vdd_matrix
    elif R_um is not None:
        if cache is None:
            raise ValueError('两体两场需要 cache（core.ArcCache 实例）来'
                             '计算 V_dd；请传入 cache 或显式 vdd_matrix。')
        V = _pc.vdd_sparse(cache, [dict(s) for s in states],
                           R_um, theta_deg, phi_deg)
    else:
        V = None
    if V is not None and not sp.issparse(V):
        V = sp.csr_matrix(np.asarray(V, complex))
    V, vdd_info = vdd_rot_filter(V, E_pair_rot, E_pair_lab, log=log)

    # ---- 耗散（寿命通道 → NonHermitian，同 evolve_sambe_pair）----
    GammaA = GammaB = None
    has_decay = bool(decays_A or decays_B)
    gamma_pair = None
    if has_decay:
        GammaA, GammaB, _ = _pe._pair_loss_vectors(
            states, decays_A, decays_B, N, N2)
        gamma_pair = (GammaA + GammaB) / _CONV
        log('[two_field_pair] 两体耗散：NonHermitian 寿命通道（落点=丢失）')

    # ---- 初态 ----
    psi0_A_vec = evolve._as_vector(states, psi0_A)
    psi0_B_vec = evolve._as_vector(states, psi0_B)
    psi0_2d = _pe._as_product_psi(psi0_A_vec, psi0_B_vec, N)

    if progress_arr is None:
        progress_arr = np.zeros(2)

    # ---- 微波光子块阶数（解析判据，M≥2 多光子显式）----
    if mw_M is None:
        Ms = _pair_mw_Ms(E_pair_rot, groups, tol)
    elif isinstance(mw_M, (int, np.integer)):
        Ms = [int(mw_M)] * len(groups)
    else:
        Ms = [int(m) for m in mw_M]
        if len(Ms) != len(groups):
            raise ValueError(f'mw_M 长度 {len(Ms)} ≠ 微波组数 {len(groups)}')
    if groups:
        log(f'[two_field_pair] 微波光子块 M={Ms}'
            f'（扇区 {int(np.prod([2 * m + 1 for m in Ms]))}）')

    pops_2d, X_final, report, setup = _pair_propagate(
        E_pair_rot, N, groups, opt_terms, V, gamma_pair, psi0_2d, t, tol,
        Ms, progress_arr, log,
        _TF_DENSE_MAX if dense_max is None else dense_max, frozen_h_ns)

    # ---- loss 曲线（NonHermitian 速率积分）----
    loss_A = np.zeros(len(t))
    loss_B = np.zeros(len(t))
    if has_decay:
        dt_arr = np.diff(t)
        rate_A = pops_2d @ GammaA
        rate_B = pops_2d @ GammaB
        loss_A[1:] = np.cumsum(0.5 * (rate_A[1:] + rate_A[:-1]) * dt_arr)
        loss_B[1:] = np.cumsum(0.5 * (rate_B[1:] + rate_B[:-1]) * dt_arr)
        norm_dev = float(np.max(np.abs(
            pops_2d.sum(axis=1) + loss_A + loss_B - 1.0)))
    else:
        norm_dev = float(np.max(np.abs(pops_2d.sum(axis=1) - 1.0)))
    report['max_norm_dev'] = max(report['max_norm_dev'], norm_dev)

    pops_A = _pe._reduce_A(pops_2d, N)
    pops_B = _pe._reduce_B(pops_2d, N)

    log(f'[two_field_pair] 完成：{report["n_segments"]} 段，'
        f'最大残差 {report["max_richardson_diff"]:.2e}'
        f'（tol={tol:.1e}），范数偏差 {report["max_norm_dev"]:.2e}')

    report['photon_blocks'] = {f'{g["omega_mhz"]:.6g}MHz': 2 * m + 1
                               for g, m in zip(groups, Ms)}
    report['tol'] = float(tol)
    report['vdd_rwa'] = vdd_info
    report['opt_terms'] = len(opt_terms)

    out = dict(
        t=t,
        pops_2d=pops_2d,
        pops_A=pops_A,
        pops_B=pops_B,
        loss_A=loss_A,
        loss_B=loss_B,
        states_2d=[{
            'label': f"({states[i]['label']}) x ({states[j]['label']})",
            'E_mhz': E_pair_lab[i * N + j],
            'ket_i': i,
            'ket_j': j,
        } for i in range(N) for j in range(N)],
        method='two_field',
        backend='sambe-pair-twofield',
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
