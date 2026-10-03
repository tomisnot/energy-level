# -*- coding: utf-8 -*-
"""Rydberg 分数陈绝缘体引擎（FCI）—— arXiv:2202.00699 精读复现（2026-10-02）。

论文：S. Weber, R. Bai, N. Makki, J. Mögerle, T. Lahaye, A. Browaeys,
M. Daghofer, N. Lang, H. P. Büchler, "Experimentally accessible scheme for a
fractional Chern insulator in Rydberg atoms", arXiv:2202.00699 (2022)。

为什么单独一个模块（而不塞进 model.py/assemble.py）：多体页现有引擎是**每原子
2 能级**（dim=2^N，assemble.build_H_sparse 的 I1/I2/I3 项开关）。论文模型每格点
**3 个态** |0⟩,|+⟩,|−⟩（V 型里德堡结构，|0⟩=真空，|±⟩=粒子的两种内态），局部
空间 3^L，相互作用内容（自旋轨道跳迁 t^a/t^b/ω + 全物种密度-密度 V^{αβ}，αβ∈
{0,a,b} 含真空项）与 I1/I2/I3 不同构。硬塞会把两套口径搅在一起；本模块按
manybody 家族"一引擎一文件"的惯例自立门户，复用同一套稀疏装配/eigsh 纪律。

物理模型（论文 Eq. 1/2/4/5，能量一律 MHz=E/h，位置 µm）：
    H_0 = Σ_{i≠j} (a_i,b_i)† [[−t^a_ij, ω_ij],[ω*_ij, −t^b_ij]] (a_j,b_j)
          + Δ Σ_i n^a_i
    H_int = Σ_{i<j} Σ_{α,β∈{0,a,b}} V^αβ_ij n^α_i n^β_j   (n^0 = 1−n^a−n^b)
    H_c  = Ω(τ) Σ_i (b†_i + b_i) − δ(τ) Σ_i (n^a_i + n^b_i)   （微波驱动 0↔−）
其中 ω_ij = |ω_ij|·e^{−2iφ_ij}，φ_ij = r_ij=r_j−r_i 的极角（自旋轨道耦合，
绕六边形 NN 环路净相位为 0，非零磁通来自 NNN 自旋翻转跳迁的三角环
e^{∓i2π/3}——A/B 子晶格磁通反号，Haldane 型，最低带 C=±1）。
跳迁/相互作用按壳层 r/l ∈ {1, √3, 2}（NN/NNN/NNNN）取论文 Table I 数值，
更远者按论文理由舍弃（PBC 下避免自相互作用）。

有效模型（论文 Eq. 5，绝热消除 |+⟩ 后的硬核玻色子，用于大系统 ramp）：
    H_eff/t_b = −Σ_⟨ij⟩ b†_i b_j − ζ Σ_⟨⟨⟨ij⟩⟩⟩ b†_i b_j + λ Σ_⟨ij⟩ n_i n_j
                − Σ_⟨⟨ij⟩⟩ (1/3^{3/2} + λ e^{i2πp_ij/3} (1−n̄_ij)) b†_i b_j + h.c.
p_ij = ±1 为绕六边形的弯折手性（顺时针 = +1），n̄_ij = i,j 公共近邻占据。

基编码约定（本模块私有，与 assemble 的 2^N 位串不同）：
- 全模型：code = Σ_i occ_i·4^i（格点 i 在 bit [2i,2i+2)，LSB 在前），
  occ ∈ {0,1,2} ↔ |0⟩,|+⟩(a),|−⟩(b)；int64 承载 L≤31。
- 有效模型：code = Σ_i n_i·2^i（bit i = 格点 i 的 b 占据），同 assemble 的
  LSB 风格（注意 assemble 是 MSB 在前——两套约定各自文档化，不混用）。

ramp 截断：驱动 Ω 破坏总激发数守恒 ⇒ 全空间 3^L；L=16 时 3^16=4300 万本机
不可行，按"激发数 ≤ n_max"截断（默认 6，终态 1/4 填充只需 4；截断外丢掉的
驱动矩阵元计数上报，不静默）。跳迁部分本身守恒 N_exc，逐 sector 块对角。
"""
import itertools

import numpy as np
import scipy.sparse as sp

__all__ = ['PAPER', 'EFFECTIVE', 'honeycomb_sites', 'pair_shells',
           'build_states_full', 'build_H_full', 'build_H_eff',
           'bloch_H', 'band_structure', 'chern_fukui', 'band_metrics',
           'sweep_profile', 'site_density']

# =====================================================================
# 论文参数（Table I：壳层顺序 [r/l=1, √3, 2]；正文 §II：Δ、实验参数）
# =====================================================================
PAPER = dict(
    Delta=18.52,                    # Δ/h MHz，|+⟩−|−⟩ 能差（Ez=0.725 V/cm）
    t_a=(1.26, 0.24, 0.16),         # |+⟩ 粒子跳迁 /h MHz
    t_b=(0.49, 0.09, 0.06),         # |−⟩ 粒子跳迁 /h MHz
    w=(2.38, 0.45, 0.29),           # |ω_ij| 自旋翻转跳迁 /h MHz
    V00=(0.03, 0.00, 0.00),         # 密度-密度 V^{αβ}/h MHz（含真空项）
    V0a=(0.07, 0.01, 0.01),
    V0b=(-0.20, 0.00, 0.00),
    Vaa=(0.19, 0.04, 0.03),
    Vab=(0.25, 0.05, 0.03),
    Vbb=(0.28, 0.05, 0.04),
    l_um=12.0, n_principal=60, Ez=0.725, Bz=-8.0,
    # 论文 Fig.4 的扫描端点（有效模型 24 格点用；全模型 ramp 的 δ 由
    # 团簇自身的 E(N) 重新标定，见 sim 脚本）
    delta0=-1.47, delta1=-0.73, Omega_max=1.0,
)

EFFECTIVE = dict(t_b=PAPER['t_b'][0], zeta=-0.207, lam=0.254)

_SHELL_R = (1.0, np.sqrt(3.0), 2.0)
_SHELL_TOL = 0.06          # r/l 判壳容差（√3−(√3±0.06) 远离相邻壳）
_R_MAX = 2.05              # 只保留 r/l ≤ 2.05 的对（论文：算到 NNNN）


# =====================================================================
# 几何：蜂巢格点 + 壳层配对（含 PBC 最小像）
# =====================================================================
def honeycomb_sites(n1, n2, l_um=12.0):
    """n1×n2 原胞蜂巢片：返回 (pos(L,2) µm, sub(L,) 0/1=A/B)。

    原胞基矢 a1=l(3/2,√3/2)、a2=l(3/2,−√3/2)，胞内 B=A+(l,0)。
    环面簇：PBC 卷绕矢 T1=n1·a1、T2=n2·a2（矩形铺贴）。
    """
    a1 = l_um * np.array([1.5, np.sqrt(3.0) / 2])
    a2 = l_um * np.array([1.5, -np.sqrt(3.0) / 2])
    d0 = l_um * np.array([1.0, 0.0])
    pos, sub = [], []
    for m in range(n1):
        for n in range(n2):
            R = m * a1 + n * a2
            pos += [R, R + d0]
            sub += [0, 1]
    return np.asarray(pos, float), np.asarray(sub, int)


def honeycomb_cluster(M, l_um=12.0):
    """任意卷绕蜂巢簇（论文 Fig.2(a) 的菱形/斜平行四边形环面簇）。

    M=[[c1,c3],[c2,c4]]：卷绕矢 T1=c1·a1+c2·a2、T2=c3·a1+c4·a2（单位：原胞）。
    格点 = 商群 Z²/⟨(c1,c2),(c3,c4)⟩ 的 |det M| 个胞，每胞 A、B 两格点 →
    L=2|det M|。论文 16 格点菱形簇 = M=[[3,1],[1,3]]（T1=3a1+a2、T2=a1+3a2，
    两边等长 → 几何菱形）；对比矩形铺贴 honeycomb_sites(4,2)（4×2，有半卷绕
    坍缩的有限尺寸效应），菱形簇保留完整 NN/NNN/NNNN 壳层计数。
    返回 (pos(L,2) µm, sub(L,), wraps=[[T1],[T2]] µm)。
    """
    M = np.asarray(M, int)
    det = int(round(M[0, 0] * M[1, 1] - M[0, 1] * M[1, 0]))
    if det <= 0:
        raise ValueError('det(M) 必须为正（交换两列可修正定向）')
    a1 = l_um * np.array([1.5, np.sqrt(3.0) / 2])
    a2 = l_um * np.array([1.5, -np.sqrt(3.0) / 2])
    d0 = l_um * np.array([1.0, 0.0])
    Minv = np.linalg.inv(M.astype(float))
    seen, cells = set(), []
    S = int(np.ceil(np.sqrt(det))) + 2       # 搜索半径，保证覆盖全部陪集
    for m in range(-S, S + 1):
        for n in range(-S, S + 1):
            uv = Minv @ np.array([float(m), float(n)])
            u = uv[0] - np.floor(uv[0] + 1e-9)   # 分数坐标 mod 1 → 陪集代表
            v = uv[1] - np.floor(uv[1] + 1e-9)
            mn = M @ np.array([u, v])
            key = (int(round(mn[0])), int(round(mn[1])))
            if key in seen:
                continue
            seen.add(key)
            cells.append(key)
        if len(cells) == det:
            break
    if len(cells) != det:
        raise RuntimeError('陪集枚举失败：得 %d / %d 胞' % (len(cells), det))
    T1 = M[0, 0] * a1 + M[1, 0] * a2             # 列 0 = T1 系数 (c1,c2)
    T2 = M[0, 1] * a1 + M[1, 1] * a2             # 列 1 = T2 系数 (c3,c4)
    pos, sub = [], []
    for m, n in cells:
        R = m * a1 + n * a2
        pos += [R, R + d0]
        sub += [0, 1]
    return np.asarray(pos, float), np.asarray(sub, int), np.array([T1, T2])


def _wrap_dist(d, wraps):
    """位移向量 d 在卷绕矢 wraps（(W,2)）下的最小像 → (R, d_min)。"""
    if wraps is None or len(wraps) == 0:
        return float(np.hypot(*d)), np.asarray(d, float)
    best_R, best_d = np.inf, None
    for m in (-1, 0, 1):
        for n in (-1, 0, 1):
            if m == 0 and n == 0:
                cand, cd = np.inf, d
            else:
                cd = d - m * wraps[0] - n * wraps[1]
                cand = float(np.hypot(*cd))
            if cand < best_R:
                best_R, best_d = cand, cd
    R0 = float(np.hypot(*d))
    return (best_R if best_R < R0 else R0), (best_d if best_R < R0 else d)


def pair_shells(pos, wraps=None, l_um=12.0, r_max=_R_MAX):
    """格点对清单（r/l ≤ r_max）：list of dict(i, j, R_um, phi, shell)。

    shell ∈ {0,1,2} ↔ r/l = {1, √3, 2}（NN/NNN/NNNN）；phi = r_ij=r_j−r_i
    极角（rad，PBC 用最小像方向——自旋轨道相位是物理量，必须跟着最小像走）。
    更远对按论文理由舍弃（计数返回在 n_dropped）。
    """
    L = len(pos)
    out, n_dropped = [], 0
    for i in range(L):
        for j in range(i + 1, L):
            R, d = _wrap_dist(pos[j] - pos[i], wraps)
            ratio = R / l_um
            if ratio > r_max:
                n_dropped += 1
                continue
            shell = None
            for k, r0 in enumerate(_SHELL_R):
                if abs(ratio - r0) < _SHELL_TOL:
                    shell = k
            if shell is None:
                n_dropped += 1
                continue
            out.append(dict(i=i, j=j, R_um=R, phi=float(np.arctan2(d[1], d[0])),
                            shell=shell))
    return out, n_dropped


# =====================================================================
# 全模型：3^L 基矢（可按 N_exc ≤ n_max 截断）+ 稀疏装配
# =====================================================================
def build_states_full(L, n_max=None):
    """基矢 codes（升序 int64）+ 逐态 (n_a, n_b, n_exc)。

    occ: 0=|0⟩, 1=|+⟩(a), 2=|−⟩(b)；code = Σ_i occ_i·4^i（LSB=格点 0）。
    n_max=None → 全空间 3^L；否则只保留 n_a+n_b ≤ n_max（ramp 截断）。
    """
    if n_max is None:
        n_max = L
    codes, nas, nbs = [], [], []
    # 逐激发数层生成：选 N 个格点 × 每点二选一物种
    for N in range(0, min(n_max, L) + 1):
        for sites in itertools.combinations(range(L), N):
            base = sum(1 << (2 * s) for s in sites)
            for species in range(1 << N):        # bit k=1 → 第 k 个格点是 b
                code, na, nb = base, 0, 0
                for k, s in enumerate(sites):
                    if (species >> k) & 1:
                        code += 1 << (2 * s)     # occ 1→2（a→b）
                        nb += 1
                    else:
                        na += 1
                codes.append(code)
                nas.append(na)
                nbs.append(nb)
    codes = np.asarray(codes, np.int64)
    order = np.argsort(codes)
    return codes[order], np.asarray(nas, np.int64)[order], \
        np.asarray(nbs, np.int64)[order]


def _occ_arrays(codes, L):
    """codes → (na, nb) 逐格点占据（向量化）。"""
    na = np.zeros((L, len(codes)), np.int64)
    nb = np.zeros((L, len(codes)), np.int64)
    for i in range(L):
        occ = (codes >> (2 * i)) & 3
        na[i] = (occ == 1)
        nb[i] = (occ == 2)
    return na, nb


def build_H_full(L, pairs, *, Delta, t_a, t_b, w, V, Omega=0.0, delta_mu=0.0,
                 n_max=None, return_states=False):
    """全模型稀疏 H（MHz，CSR；非厄米不可能——驱动是厄米的）。

    H = H_0(跳迁+Δ n^a) + H_int(全物种密度-密度) + ΩΣ(b†+b) − δΣ(n^a+n^b)。
    Omega/delta_mu：标量（ramp 是空间均匀驱动；论文 Fig.4(d) 的非均匀 Ω 留作
    逐点扩展，本模块先给标量口径）。n_max：N_exc ≤ n_max 截断（None=3^L）。
    截断把「产生方向」的驱动矩阵元丢掉 → dropped_drive 计数上报（不静默）。
    返回 dict(H=csr, codes, na, nb, n_exc, dropped_drive, dim)。
    """
    if '00' not in V and 'V00' in V:        # 接受 PAPER 的 V00/V0a/… 风格键
        V = {k[1:]: v for k, v in V.items() if k.startswith('V')}
    codes, na_s, nb_s = build_states_full(L, n_max)
    dim = len(codes)
    na, nb = _occ_arrays(codes, L)          # (L, dim)
    n_tot = (na + nb).sum(axis=0)           # 逐态总激发数 (dim,)，a+b 物种
    rows, cols, vals = [], [], []

    # ---- 对角：Δ n^a + Σ_pairs V^{αβ} n^α_i n^β_j − δ (n^a+n^b) ----
    diag = (Delta * na.sum(axis=0).astype(float)   # Δ 只加给 a 物种
            - delta_mu * n_tot)
    for p in pairs:
        i, j, s = p['i'], p['j'], p['shell']
        nai, naj, nbi, nbj = na[i], na[j], nb[i], nb[j]
        n0i, n0j = 1 - nai - nbi, 1 - naj - nbj
        diag = diag + (V['00'][s] * n0i * n0j
                       + V['0a'][s] * (n0i * naj + nai * n0j)
                       + V['0b'][s] * (n0i * nbj + nbi * n0j)
                       + V['aa'][s] * nai * naj
                       + V['ab'][s] * (nai * nbj + nbi * naj)
                       + V['bb'][s] * nbi * nbj)
    rows.append(np.arange(dim)); cols.append(np.arange(dim)); vals.append(diag)

    # ---- 跳迁（守恒 N_exc）：逐有向对 (j→i) 四种过程 ----
    # pairs 只含 i<j 的单向对，故每生成一个矩阵元必须**同时补 h.c.**
    # （⟨目标|H|源⟩ 与 ⟨源|H|目标⟩=conj），否则 H 只有单侧三角块——非厄米。
    # 注意 ω 项的 h.c. 是同一有向对上的共轭（ω*ᵢⱼ ↔ ωᵢⱼ），而"b_j→a_i"
    # 是另一个物理过程（a 的位置不同），两者都要有。
    idx_of = None
    for p in pairs:
        i, j, s = p['i'], p['j'], p['shell']
        mi, mj = np.int64(1) << (2 * i), np.int64(1) << (2 * j)
        mb_i, mb_j = np.int64(2) << (2 * i), np.int64(2) << (2 * j)
        w_sh, ta, tb = w[s], t_a[s], t_b[s]

        def _emit(rows_t, cols_s, amp):
            rows.append(rows_t); cols.append(cols_s); vals.append(amp)
            rows.append(cols_s); cols.append(rows_t); vals.append(np.conj(amp))

        # a_j→a_i：⟨a_i|H|a_j⟩ = −t^a；要求 occ_j=1(a)，occ_i=0
        sel = (na[j] == 1) & (na[i] + nb[i] == 0)
        if sel.any() and ta != 0.0:
            ss = codes[sel]
            _emit(np.searchsorted(codes, ss - mj + mi), sel.nonzero()[0],
                  np.full(ss.size, -ta, complex))
        # b_j→b_i：⟨b_i|H|b_j⟩ = −t^b
        sel = (nb[j] == 1) & (na[i] + nb[i] == 0)
        if sel.any() and tb != 0.0:
            ss = codes[sel]
            _emit(np.searchsorted(codes, ss - mb_j + mb_i), sel.nonzero()[0],
                  np.full(ss.size, -tb, complex))
        # a_j→b_i：⟨b_i|H|a_j⟩ = ω*_ij = |ω|e^{+2iφ}
        sel = (na[j] == 1) & (na[i] + nb[i] == 0)
        if sel.any() and w_sh != 0.0:
            ss = codes[sel]
            _emit(np.searchsorted(codes, ss - mj + mb_i), sel.nonzero()[0],
                  np.full(ss.size, w_sh * np.exp(+2j * p['phi']), complex))
        # b_j→a_i：⟨a_i|H|b_j⟩ = ω_ij = |ω|e^{−2iφ}
        sel = (nb[j] == 1) & (na[i] + nb[i] == 0)
        if sel.any() and w_sh != 0.0:
            ss = codes[sel]
            _emit(np.searchsorted(codes, ss - mb_j + mi), sel.nonzero()[0],
                  np.full(ss.size, w_sh * np.exp(-2j * p['phi']), complex))

    # ---- 驱动 Ω(b†+b)：耦合 N↔N±1（截断处丢弃产生方向并计数） ----
    dropped_drive = 0
    if Omega != 0.0:
        n_cap = n_max if n_max is not None else L
        for i in range(L):
            mb_i = np.int64(2) << (2 * i)
            # b†_i：occ_i=0 → 2
            sel = np.nonzero((na[i] + nb[i] == 0) & (n_tot + 1 <= n_cap))[0]
            dropped_drive += int(((na[i] + nb[i] == 0)
                                  & (n_tot + 1 > n_cap)).sum())
            if sel.size:
                ss = codes[sel]
                rows.append(np.searchsorted(codes, ss + mb_i)); cols.append(sel)
                vals.append(np.full(sel.size, Omega, complex))
            # b_i：occ_i=2 → 0
            sel = np.nonzero(nb[i] == 1)[0]
            if sel.size:
                ss = codes[sel]
                rows.append(np.searchsorted(codes, ss - mb_i)); cols.append(sel)
                vals.append(np.full(sel.size, Omega, complex))

    H = sp.coo_matrix((np.concatenate(vals),
                       (np.concatenate(rows), np.concatenate(cols))),
                      shape=(dim, dim)).tocsr()
    H.sum_duplicates()
    out = dict(H=H, codes=codes, na=na, nb=nb, n_exc=n_tot,
               dropped_drive=dropped_drive, dim=dim)
    if return_states:
        out['states'] = codes
    return out


# =====================================================================
# 有效模型（Eq. 5）：2^L 硬核玻色子
# =====================================================================
def _middle_site(pos, i, j, wraps=None, l_um=12.0):
    """NNN 对 (i,j) 的"中间格点" k（公共 NN 邻居，恰一个）+ 弯折手性 p。"""
    best, best_R = None, np.inf
    for k in range(len(pos)):
        if k in (i, j):
            continue
        Rik, _ = _wrap_dist(pos[k] - pos[i], wraps)
        Rkj, _ = _wrap_dist(pos[j] - pos[k], wraps)
        if abs(Rik / l_um - 1.0) < _SHELL_TOL and abs(Rkj / l_um - 1.0) < _SHELL_TOL:
            if Rik + Rkj < best_R:
                best, best_R = k, Rik + Rkj
    if best is None:
        raise ValueError('NNN 对 (%d,%d) 没找到公共近邻——几何不是蜂巢？' % (i, j))
    k = best
    # 手性：弯折 i→k→j 顺时针（y 向上坐标系叉积 <0）→ p=+1（论文约定）
    _, dik = _wrap_dist(pos[k] - pos[i], wraps)
    _, dkj = _wrap_dist(pos[j] - pos[k], wraps)
    cross = dik[0] * dkj[1] - dik[1] * dkj[0]
    p = 1 if cross < 0 else -1
    return k, p


def build_H_eff(L, pairs, pos, *, t_b=0.49, zeta=-0.207, lam=0.254,
                Omega=0.0, delta_mu=0.0, l_um=12.0, wraps=None):
    """有效模型 Eq.5 稀疏 H（MHz，CSR，2^L 基；bit i = 格点 i 的 b 占据）。

    按论文常数 ζ、λ；NNN 相关跳迁带 (1−n̄_ij) 密度依赖（n̄=中间格点占据）。
    返回 dict(H=csr, codes, nb(L,dim), n_tot, dim)。
    """
    dim = 1 << L
    codes = np.arange(dim, dtype=np.int64)
    nbit = np.zeros((L, dim), np.int8)
    for i in range(L):
        nbit[i] = (codes >> i) & 1
    n_tot = nbit.sum(axis=0).astype(float)
    rows, cols, vals = [], [], []

    diag = -delta_mu * n_tot

    def _emit(rows_t, cols_s, amp):
        # 同 build_H_full：pairs 单向遍历，逐元补 h.c.（conj 镜像），
        # 依赖 nk 的 NNN 相关项在共轭方向上 nk 相同（k 未动），直接 conj。
        rows.append(rows_t); cols.append(cols_s); vals.append(amp)
        rows.append(cols_s); cols.append(rows_t); vals.append(np.conj(amp))

    for p in pairs:
        i, j, s = p['i'], p['j'], p['shell']
        if s == 0:                                   # NN：−t_b b†b + λ t_b n_i n_j
            diag = diag + lam * t_b * (nbit[i] * nbit[j])
            sel = np.nonzero((nbit[j] == 1) & (nbit[i] == 0))[0]
            if sel.size:
                ss = codes[sel]
                _emit(ss + (1 << i) - (1 << j), sel,
                      np.full(sel.size, -t_b, complex))
        elif s == 1:                                 # NNN：裸 1/3^{3/2} + 相关项
            k, pp = _middle_site(pos, i, j, wraps, l_um)
            nk = nbit[k]
            sel = np.nonzero((nbit[j] == 1) & (nbit[i] == 0))[0]
            if sel.size:
                ss = codes[sel]
                amp = -(t_b / 3.0 ** 1.5) - (lam * t_b) * np.exp(2j * np.pi * pp / 3.0) * (1.0 - nk[sel])
                _emit(ss + (1 << i) - (1 << j), sel, amp.astype(complex))
        elif s == 2:                                 # NNNN：−ζ t_b
            sel = np.nonzero((nbit[j] == 1) & (nbit[i] == 0))[0]
            if sel.size:
                ss = codes[sel]
                _emit(ss + (1 << i) - (1 << j), sel,
                      np.full(sel.size, -zeta * t_b, complex))
    rows.append(np.arange(dim)); cols.append(np.arange(dim)); vals.append(diag)

    if Omega != 0.0:
        for i in range(L):
            sel = np.nonzero(nbit[i] == 0)[0]
            if sel.size:
                ss = codes[sel]
                rows.append(ss + (1 << i)); cols.append(sel)
                vals.append(np.full(sel.size, Omega, complex))
            sel = np.nonzero(nbit[i] == 1)[0]
            if sel.size:
                ss = codes[sel]
                rows.append(ss - (1 << i)); cols.append(sel)
                vals.append(np.full(sel.size, Omega, complex))

    H = sp.coo_matrix((np.concatenate(vals),
                       (np.concatenate(rows), np.concatenate(cols))),
                      shape=(dim, dim)).tocsr()
    H.sum_duplicates()
    return dict(H=H, codes=codes, nb=nbit, n_tot=n_tot, dim=dim)


# =====================================================================
# 单粒子：Bloch 哈密顿量（4 带）、能带、Fukui-Chern、平坦度
# =====================================================================
def _lat_vectors(l_um):
    a1 = l_um * np.array([1.5, np.sqrt(3.0) / 2])
    a2 = l_um * np.array([1.5, -np.sqrt(3.0) / 2])
    d = [np.zeros(2), l_um * np.array([1.0, 0.0])]
    return a1, a2, d


def bloch_H(k, *, Delta, t_a, t_b, w, l_um=12.0):
    """∞ 蜂巢单粒子 Bloch H（4×4，基 [A+, A−, B+, B−]，MHz=E/h）。

    H(k)[(σ,α),(σ',α')] = Σ_D e^{ik·(D+d_{σ'}−d_σ)} h_{(σ α),(σ' α')}(D)，
    D 跑遍 |D+d_{σ'}−d_σ| ≤ 2.05l 的格矢。h：
      ⟨σ,a|σ',a⟩ = −t^a(shell)；⟨σ,b|σ',b⟩ = −t^b；
      ⟨σ,a|σ',b⟩ = ω = |ω|e^{−2iφ}（φ = 极角 of r_{σ→σ'} = D+d_{σ'}−d_σ）；
      ⟨σ,b|σ',a⟩ = ω*；对角 E_a=Δ, E_b=0。
    """
    a1, a2, d = _lat_vectors(l_um)
    Hk = np.zeros((4, 4), complex)
    rng = range(-3, 4)
    for s1 in (0, 1):
        for s2 in (0, 1):
            for m in rng:
                for n in rng:
                    Dvec = m * a1 + n * a2
                    rvec = Dvec + d[s2] - d[s1]
                    R = float(np.hypot(*rvec))
                    if R < 1e-9:
                        continue
                    ratio = R / l_um
                    shell = None
                    for kk, r0 in enumerate(_SHELL_R):
                        if abs(ratio - r0) < _SHELL_TOL:
                            shell = kk
                    if shell is None:
                        continue
                    ph = np.exp(1j * np.dot(k, rvec))
                    phi = np.arctan2(rvec[1], rvec[0])
                    i0, j0 = 2 * s1, 2 * s2
                    # 相位约定：布洛赫基按格点位置 e^{ik·(R+d_s)} 展开，
                    # 全部矩阵元统一乘 ph = e^{ik·(源−汇)}；厄米性由
                    # (s2,s1,−D) 镜像对自动保证，**不要**手动 conj——
                    # （2026-10-02 抓到：b→a 项误用 conj(ph) ⇒ H(k) 非厄米，
                    # eigvalsh 默认读下三角把错谱静默吞掉 → C/平坦度全废）
                    Hk[i0, j0] += -t_a[shell] * ph            # a→a
                    Hk[i0 + 1, j0 + 1] += -t_b[shell] * ph    # b→b
                    Hk[i0, j0 + 1] += w[shell] * np.exp(-2j * phi) * ph   # ⟨a|H|b⟩
                    Hk[i0 + 1, j0] += w[shell] * np.exp(+2j * phi) * ph   # ⟨b|H|a⟩
    Hk[0, 0] += Delta; Hk[2, 2] += Delta    # E_a = Δ：a 物种在偶数位（A_a/B_a）
    return Hk


def band_structure(path_k, params=None, l_um=12.0):
    """k 路径 → 能带 (n_k, 4)（升序）。"""
    P = dict(PAPER) if params is None else params
    ks = [k for seg in path_k for k in seg]
    return np.sort(np.array([np.linalg.eigvalsh(bloch_H(k, Delta=P['Delta'],
                                                        t_a=P['t_a'], t_b=P['t_b'],
                                                        w=P['w'], l_um=l_um))
                             for k in ks]), axis=1)


def kpath_KGM(N_seg=40, l_um=12.0):
    """K→Γ→M→K 折线路径（论文 Fig.1(c)）。返回 [(n_k,2) 段, …]。"""
    a1, a2, _ = _lat_vectors(l_um)
    b1 = 2 * np.pi / (a1[0] * a2[1] - a1[1] * a2[0]) * np.array([a2[1], -a2[0]])
    b2 = 2 * np.pi / (a1[0] * a2[1] - a1[1] * a2[0]) * np.array([-a1[1], a1[0]])
    K = (2 * b1 + b2) / 3.0
    M = (b1 + b2) / 2.0
    G = np.zeros(2)

    def line(u, v, n):
        return [u + (v - u) * t for t in np.linspace(0, 1, n)]

    return [line(K, G, N_seg), line(G, M, N_seg), line(M, K, N_seg)], (K, G, M)


def chern_fukui(Nk=24, band=0, params=None, l_um=12.0):
    """Fukui-Hatsugai-Suzuki 格点 Chern 数（单带，周期 k 网格 [0,2π)³→2D）。"""
    P = dict(PAPER) if params is None else params
    a1, a2, _ = _lat_vectors(l_um)
    b1 = 2 * np.pi / (a1[0] * a2[1] - a1[1] * a2[0]) * np.array([a2[1], -a2[0]])
    b2 = 2 * np.pi / (a1[0] * a2[1] - a1[1] * a2[0]) * np.array([-a1[1], a1[0]])
    U = np.zeros((Nk, Nk, 4), complex)      # 逐 k 存第 band 条带的态矢量
    for i in range(Nk):
        for j in range(Nk):
            k = (i / Nk) * b1 + (j / Nk) * b2
            w, v = np.linalg.eigh(bloch_H(k, Delta=P['Delta'], t_a=P['t_a'],
                                          t_b=P['t_b'], w=P['w'], l_um=l_um))
            U[i, j] = v[:, band]
    c = 0.0 + 0.0j
    for i in range(Nk):
        for j in range(Nk):
            i2, j2 = (i + 1) % Nk, (j + 1) % Nk
            o1 = np.vdot(U[i, j], U[i2, j])
            o2 = np.vdot(U[i2, j], U[i2, j2])
            o3 = np.vdot(U[i2, j2], U[i, j2])
            o4 = np.vdot(U[i, j2], U[i, j])
            c += np.angle(o1 * o2 * o3 * o4)
    return int(round(c.real / (2 * np.pi)))


def band_metrics(params=None, l_um=12.0, Nk=24):
    """最低带平坦度 f = (E2−E1)/W1 与 Berry 曲率涨落 σ_B（论文 §II 定义）。"""
    P = dict(PAPER) if params is None else params
    a1, a2, _ = _lat_vectors(l_um)
    b1 = 2 * np.pi / (a1[0] * a2[1] - a1[1] * a2[0]) * np.array([a2[1], -a2[0]])
    b2 = 2 * np.pi / (a1[0] * a2[1] - a1[1] * a2[0]) * np.array([-a1[1], a1[0]])
    bands, Bcurv = [], []
    for i in range(Nk):
        for j in range(Nk):
            k = (i / Nk) * b1 + (j / Nk) * b2
            w, v = np.linalg.eigh(bloch_H(k, Delta=P['Delta'], t_a=P['t_a'],
                                          t_b=P['t_b'], w=P['w'], l_um=l_um))
            bands.append(w)
            # Berry 曲率（Fukui 铰链法，最低带）：F = Im ln Π U
            if i + 1 < Nk and j + 1 < Nk:
                def band_vec(kk, bd=0):
                    _, vv = np.linalg.eigh(bloch_H(kk, Delta=P['Delta'],
                                                   t_a=P['t_a'], t_b=P['t_b'],
                                                   w=P['w'], l_um=l_um))
                    return vv[:, bd]
                k1 = k + (b1 / Nk); k2 = k + (b2 / Nk); k3 = k + (b1 + b2) / Nk
                u0, u1, u2, u3 = band_vec(k), band_vec(k1), band_vec(k3), band_vec(k2)
                link = (np.vdot(u0, u1) * np.vdot(u1, u2)
                        * np.vdot(u2, u3) * np.vdot(u3, u0))
                Bcurv.append(np.imag(np.log(link)))
    bands = np.array(bands)
    e = np.sort(bands, axis=1)
    W1 = e[:, 0].max() - e[:, 0].min()
    gap12 = e[:, 1].min() - e[:, 0].max()
    Bm = np.mean(Bcurv)
    sigmaB = float(np.sqrt(np.mean((np.array(Bcurv) - Bm) ** 2)) / abs(Bm)) \
        if abs(Bm) > 1e-14 else float('inf')
    return dict(flatness=gap12 / W1 if W1 > 0 else np.inf, band_width=W1,
                gap_12=gap12, sigma_B=sigmaB)


# =====================================================================
# ramp 模型接口：H(τ) 组装（纯构造，求解在 eq_route.solve_equilibrium）
# =====================================================================
def sweep_profile(tau, Omega_max=1.0, delta0=0.0, delta1=0.0):
    """三段式 ramp 剖面（论文 Fig.4(a) 的扫描协议）：τ∈[0,1] → (Ω(τ), δ(τ))。

    ⚠ **这是论文协议，不是通用求解动词**（2026-10-02 晚裁决）。δ 端点的取法
    （真空稳定侧 µ_c(0→1)−offset、N=n_target 平台中央）与 Ω 的三段升降划分，
    都是本文的做法——换一个扫描协议就全变了，通用引擎不该持有唯一的 ramp 波形。
    曾一度收进 `eq_route`（"求解动词收敛"方案 §4.2），已摘回本模块。
    本函数是**零依赖纯函数**（只 numpy），故可随本模块被按文件路径独立加载
    （sim 脚本与自检都走 `spec_from_file_location`，不经 `src.core` 包导入）。

    段1 τ≤1/3：Ω: 0→Ω_max，δ=δ0；段2 ≤2/3：Ω=Ω_max，δ: δ0→δ1 线性；
    段3：Ω: Ω_max→0，δ=δ1。delta0/delta1 = 扫描端点（MHz），**必传**——
    论文数值不该藏进默认签名（藏了就看不见"这次跑的是哪个协议"）。
    """
    tau = np.asarray(tau, float)
    Om = np.where(tau <= 1 / 3, Omega_max * 3 * tau,
                  np.where(tau <= 2 / 3, Omega_max,
                           Omega_max * 3 * (1 - tau)))
    de = np.where(tau <= 1 / 3, delta0,
                  np.where(tau <= 2 / 3,
                           delta0 + (delta1 - delta0) * 3 * (tau - 1 / 3),
                           delta1))
    return Om, de


def ramp_operator(H0, B, n_tot, Om, delta):
    """H(τ) = H0 + Ω·B − δ·N_tot 的 LinearOperator（ramp 编排的模型侧接口）。

    H0：Ω=δ=0 的静态哈密顿量（build_H_full/build_H_eff，Ω=delta=0 装配）；
    B：纯 Ω 驱动矩阵（技巧：t=V=Δ=δ=0、Ω=1 的装配取纯驱动）；
    n_tot：1 维逐基矢总激发数（δ 项是对角的，矩阵-向量乘里按对角作用）。
    求解（warm-start 逐点对角化）在 eq_route.solve_equilibrium(want='ramp')。
    """
    def mv(v):
        return H0 @ v + Om * (B @ v) - delta * (n_tot * v)
    return sp.linalg.LinearOperator((H0.shape[0],) * 2, matvec=mv, dtype=complex)


def site_density(nb_or_pops, p):
    """逐格点平均粒子数：nb(L,dim) 概率加权。"""
    return nb_or_pops @ np.abs(p) ** 2

