# -*- coding: utf-8 -*-
"""4 维对空间 V_dd 全精确标定 + N 体装配（M0）。

物理路线（docs/多体页-调查与开发计划.md §3.2，2026-09-17 裁决）：
- 每原子 2 能级 (g,r)；一对原子张成 4 维空间 [|gg⟩,|gr⟩,|rg⟩,|rr⟩]。
- **一阶**：V_dd = (1/R³)Σ_qq' T_qq'(θ,φ)·d1_q⊗d2_q' 直接投影到 4 维
  （复用 static_pair 的球基张量与 ARC 偶极矩阵元，C3/交换不硬编码）。
- **二阶**：Löwdin downfolding——模型外中间态 |m⟩（伙伴态，single_atom_basis
  自动补）以 Σ_m V_pm V_mq/(E−E_m) 折进 4 维有效块（vdW/C6 自动涌现）。
  分母 |E−E_m| < guard 的近简并中间态**剔除并计数上报**（不许静默）。

相位协变（本模块的灵魂，自检 A0b 精确验证）。H4 用**标准矩阵元序**
H4[p,q] = ⟨p|V|q⟩（注意 vdd_sparse 的存储是转置序 ⟨q|V|p⟩，这里已取共轭
换回），协变律：
    ⟨p|V|q⟩(θ,φ) = e^{i(M_p−M_q)φ} · ⟨p|V|q⟩(θ,0)，  M = m₁+m₂
即 ⟨rr|V|gg⟩ ∝ e^{+2i(m_r−m_g)φ} 型缠绕——偶极相互作用对方向角 φ 的
相位依赖**全程复数保留**，绝不取实部。θ=0 时 T 仅 q+q'=0 分量 ⇒ 总 m
守恒（⟨gg|V|rr⟩ 在 2(m_r−m_g)≠0 时严格为 0，A0e 锚）。

单位：能级/耦合 MHz，位置 µm，偶极 a.u.。本包只读复用 src.core，不改它。
"""
import numpy as np
from scipy import sparse as sp

from .. import static_pair as _sp
from ..evolve_util import CONV          # MHz·ns → rad（2π×1e-3），同各演化引擎

_DD_CONV = _sp._DD_CONV          # MHz·µm³/(a.u.)²
_QS = (1, 0, -1)                 # T 指标 a ∈ {0,1,2} ↔ 球分量 q
_H4_ORDER = ('gg', 'gr', 'rg', 'rr')


# =====================================================================
# 几何与 V_dd 行抽取
# =====================================================================
def geometry(pos_i, pos_j):
    """两原子坐标 → (R_µm, θ_deg, φ_deg)，n = (r_j−r_i)/R（与 static_pair 同约定）。"""
    d = np.asarray(pos_j, float) - np.asarray(pos_i, float)
    R = float(np.linalg.norm(d))
    if R <= 0.0:
        raise ValueError('两原子重合（R=0）')
    theta = float(np.degrees(np.arccos(np.clip(d[2] / R, -1.0, 1.0))))
    phi = float(np.degrees(np.arctan2(d[1], d[0])))
    return R, theta, phi


def _vdd_row(D, T, i, j, N):
    """⟨i⊗j| V_dd |全体 (a⊗b)⟩ 的稀疏行（1×N²，单位 MHz，未除 R³）。

    V_dd = Σ_qq' T_qq' D_q⊗D_q'（D_q[x,y]=⟨x|d_q|y⟩，与 vdd_sparse 同序：
    k=i*N+j，行=bra）。逐 (q,q') 用两条稀疏行做 kron——避开 N⁴ 稠密。
    """
    row = None
    for a in range(3):
        Di = D[_QS[a]][i]                      # csr (1,N)
        if Di.nnz == 0:
            continue
        for b in range(3):
            t = T[a, b]
            if abs(t) < 1e-14:
                continue
            Dj = D[_QS[b]][j]
            if Dj.nnz == 0:
                continue
            r = sp.kron(Di, Dj, format='csr')   # (1, N²)
            row = t * r if row is None else row + t * r
    if row is None:
        row = sp.csr_matrix((1, N * N))
    return row


# =====================================================================
# 逐对标定
# =====================================================================
def calibrate_pair(cache, single, D, E, pair4, R, theta_deg, phi_deg,
                   second_order=True, guard_mhz=1.0):
    """一对原子的 4 维有效哈密顿（MHz，复）。

    pair4 = ((ig_i, ig_j), (ir_i, ig_j), (ig_i, ir_j), (ir_i, ir_j))
            ——单原子基下标 (原子1, 原子2)，顺序 [|gg⟩,|gr⟩,|rg⟩,|rr⟩]。
    返回 dict(H4, delta1(一阶 4×4), delta2(二阶修正), info)。
    """
    N = len(single)
    E_pair = (E[:, None] + E[None, :]).ravel()
    k_idx = np.array([a * N + b for a, b in pair4], dtype=int)
    T = _sp._vdd_tensor(theta_deg, phi_deg)
    scale = _DD_CONV / R ** 3
    # rmat[p, x] = ⟨x|V|p⟩（vdd_sparse 存储序：行=ket）；标准元 ⟨p|V|q⟩
    # = conj(rmat[p, k_q])。单位 MHz（scale 已含 1/R³）。
    rmat = np.stack([np.asarray(_vdd_row(D, T, a, b, N).todense()).ravel()
                     for a, b in pair4]) * scale          # (4, N²)

    # ---- 一阶：4×4 模型块（标准序 ⟨p|V|q⟩，复数原样保留 (θ,φ) 相位） ----
    d1 = np.conj(rmat[:, k_idx])                           # (4,4)

    # ---- 二阶：Löwdin downfolding Σ_m ⟨p|V|m⟩⟨m|V|q⟩·f(E_p,E_q,E_m) ----
    #      ⟨p|V|m⟩ = conj(rm[p,m])，⟨m|V|q⟩ = rm[q,m]
    d2 = np.zeros((4, 4), complex)
    info = dict(n_intermediate=0, n_guarded=0, min_gap_mhz=float('inf'))
    if second_order:
        mask = np.ones(N * N, dtype=bool)
        mask[k_idx] = False
        Em = E_pair[mask]
        rm = rmat[:, mask]                                 # (4, n_m)
        Ek = E_pair[k_idx]
        gap = np.abs(Ek[:, None] - Em[None, :])            # (4, n_m)
        keep = (gap > guard_mhz).all(axis=0)               # 4 端点都远离才用
        info['n_intermediate'] = int(keep.sum())
        info['n_guarded'] = int((~keep).sum())
        if keep.any():
            info['min_gap_mhz'] = float(gap[:, keep].min())
        denom = Ek[:, None] - Em[None, :]                  # E_p − E_m
        inv = np.zeros((4, len(Em)))
        np.divide(1.0, denom, out=inv,
                  where=(keep[None, :] & (denom != 0.0)))  # (4, n_m)
        # f_pq = ½(1/(E_p−E_m) + 1/(E_q−E_m))
        for p in range(4):
            for q in range(4):
                f = 0.5 * (inv[p] + inv[q])
                d2[p, q] = np.sum(np.conj(rm[p]) * rm[q] * f)
    H4 = d1 + d2
    return dict(H4=H4, delta1=d1, delta2=d2, info=info)


# =====================================================================
# 模型级标定（逐对扫描 + 项开关 + 诊断）
# =====================================================================
def calibrate(cache, model, second_order=True, guard_mhz=1.0, log=None):
    """ManyBodyModel → calib dict（可 JSON 序列化，H4 除外另存 npz）。

    项开关 model.terms：I1=V·n_i n_j，I2=J 交换，I3=A 双激发翻转；
    未勾选的项在输出里置 0（raw_* 保留全值，供"被丢项最大幅值"报告）。
    """
    log = log or (lambda *a: None)
    # 1) 单原子基：所有 (n,l,j) 并集 + 伙伴态（自动补 vdW 中间态）
    groups, seen = [], set()
    for a in model.atoms:
        for q in (a['g'], a['r']):
            key = (int(q[0]), int(q[1]), float(q[2]))
            if key not in seen:
                seen.add(key)
                groups.append(dict(n=key[0], l=key[1], j=key[2]))
    single = _sp.single_atom_basis(cache, groups, model.B_gauss,
                                   model.n_window)
    E = np.array([s['E_mhz'] for s in single], float)
    D = {q: _sp._dipole_sparse(cache, single, q) for q in (-1, 0, 1)}

    # 2) 逐原子能级下标与 ε
    atoms = []
    for a in model.atoms:
        ig = _sp.find_sublevel(single, *a['g'])
        ir = _sp.find_sublevel(single, *a['r'])
        if ig is None or ir is None:
            raise ValueError(f'原子 {a["id"]} 的能级不在基组内：'
                             f'g={a["g"]}→{ig}, r={a["r"]}→{ir}')
        atoms.append(dict(id=a['id'], g=list(a['g']), r=list(a['r']),
                          pos=list(a['pos']), ig=ig, ir=ir,
                          eps_mhz=float(E[ir] - E[ig])))

    # 3) 逐对标定
    use = set(model.terms)
    pairs, dropped = [], []
    n = len(model.atoms)
    for i in range(n):
        for j in range(i + 1, n):
            ai, aj = model.atoms[i], model.atoms[j]
            R, th, ph = geometry(ai['pos'], aj['pos'])
            if model.r_cut_um is not None and R > model.r_cut_um:
                dropped.append(dict(i=i, j=j, R_um=R, reason='r_cut'))
                continue
            pair4 = ((atoms[i]['ig'], atoms[j]['ig']),
                     (atoms[i]['ir'], atoms[j]['ig']),
                     (atoms[i]['ig'], atoms[j]['ir']),
                     (atoms[i]['ir'], atoms[j]['ir']))
            res = calibrate_pair(cache, single, D, E, pair4, R, th, ph,
                                 second_order=second_order,
                                 guard_mhz=guard_mhz)
            H4 = res['H4']
            dg, dgr, drg, dr = (H4[0, 0].real, H4[1, 1].real,
                                H4[2, 2].real, H4[3, 3].real)
            rec = dict(i=i, j=j, R_um=R, theta_deg=th, phi_deg=ph,
                       raw_V=float(dr - dgr - drg + dg),   # nn 不可约组合
                       raw_J=complex(H4[1, 2]),            # ⟨r_i g_j|V|g_i r_j⟩
                       raw_A=complex(H4[3, 0]),            # ⟨rr|V|gg⟩
                       delta=dict(gg=dg, gr=dgr, rg=drg, rr=dr),
                       info=res['info'])
            rec['V'] = rec['raw_V'] if 'I1' in use else 0.0
            rec['J'] = rec['raw_J'] if 'I2' in use else 0.0j
            rec['A'] = rec['raw_A'] if 'I3' in use else 0.0j
            pairs.append(rec)
    # 4) 被丢项幅值报告（关项不许静默）
    dropped_amp = dict(
        I1=float(max((abs(p['raw_V']) for p in pairs
                      if 'I1' not in use), default=0.0)),
        I2=float(max((abs(p['raw_J']) for p in pairs
                      if 'I2' not in use), default=0.0)),
        I3=float(max((abs(p['raw_A']) for p in pairs
                      if 'I3' not in use), default=0.0)))
    log(f'[manybody] 标定 {len(pairs)} 对（截断 {len(dropped)}），'
        f'基组 N={len(single)}')
    return dict(template=model.template, terms=list(model.terms),
                atoms=atoms, pairs=pairs, dropped=dropped,
                dropped_amp=dropped_amp,
                single_basis=[dict(n=s['n'], l=s['l'], j=s['j'],
                                   mj=s.get('mj'), E_mhz=s['E_mhz'])
                              for s in single])


def to_json_dict(calib):
    """calib → 可 JSON 化（raw_J/raw_A 拆 re/im）。"""
    c = dict(calib)
    c['pairs'] = [dict(p, raw_J=[p['raw_J'].real, p['raw_J'].imag],
                       raw_A=[p['raw_A'].real, p['raw_A'].imag],
                       J=[p['J'].real, p['J'].imag],
                       A=[p['A'].real, p['A'].imag])
                  for p in calib['pairs']]
    return c


# =====================================================================
# ED 装配（2^N 基，bit=1 ↔ r）——稀疏为主，稠密仅 tiny-N 交叉对照
# =====================================================================
def build_H_sparse(calib, Om=None, delta=None, terms=None, gamma=None):
    """calib → 2^N **稀疏** H（CSR，MHz，复）。Om/delta/gamma：逐原子（缺省 0）。

    H = Σ_i [δ_i n_i + (Ω_i/2)σˣ_i − (i/2)Γ_i n_i]
      + Σ_{i<j} [V·n_i n_j + (J σ⁺_iσ⁻_j + A σ⁺_iσ⁺_j + h.c.)]
    基序 bit i ↔ 原子 i（MSB：bit(s,i)=(s>>(n−1−i))&1），bit=1 ↔ |r⟩。
    矩阵元序 H[t,s]=⟨t|H|s⟩（与 calibrate_pair 的标准序一致）。

    每行 O(N) 非零 ⇒ 取代 2^N×2^N 稠密（N=16 稠密需 68 GB，不可行）。
    非厄米接缝（裁决③：默认关，gamma=None ⇒ 厄米）：gamma 是**布居衰减率 Γ(MHz)**，
    同 evolve_util 约定（物理率 = Γ×1e-3 /ns）；演化用 exp(-i·CONV·H·t)，故 H 虚部
    对角取 -i·Γ×1e-3/(2·CONV) ⇒ 布居恰按 exp(-Γ×1e-3·t) 衰减（与 auto_decays/Lindblad 一致）。
    """
    n = len(calib['atoms'])
    dim = 1 << n
    use = set(calib['terms'] if terms is None else terms)
    Om = np.zeros(n) if Om is None else np.asarray(Om, float)
    delta = np.zeros(n) if delta is None else np.asarray(delta, float)
    gamma = np.zeros(n) if gamma is None else np.asarray(gamma, float)
    s = np.arange(dim, dtype=np.int64)
    bit = [((s >> (n - 1 - i)) & 1).astype(np.int8) for i in range(n)]

    rows, cols, vals = [], [], []
    # ---- 对角：Σ δ_i n_i + Σ V_ij n_i n_j − (i/2)Σ Γ_i n_i ----
    diag = np.zeros(dim, complex)
    for i in range(n):
        # 衰减项：Γ(MHz)→物理率 Γ×1e-3/ns；除以 CONV 抵消演化的 exp(-i·CONV·H·t) 的 2π
        diag += (delta[i] - 0.5j * gamma[i] * 1e-3 / CONV) * bit[i]
    if 'I1' in use:
        for p in calib['pairs']:
            diag = diag + p['V'] * (bit[p['i']] * bit[p['j']])
    rows.append(s); cols.append(s); vals.append(diag)

    # ---- 单原子驱动 Ω_i/2·σˣ_i：(t,s)=(flip_i(s), s)，翻一位 ----
    for i in range(n):
        if Om[i] == 0.0:
            continue
        t = s ^ (np.int64(1) << (n - 1 - i))
        rows.append(t); cols.append(s)
        vals.append(np.full(dim, Om[i] / 2.0, complex))

    # ---- 交换 J（I2）+ 双激发 A（I3）：翻两位 ----
    for p in calib['pairs']:
        i, j = p['i'], p['j']
        mi = np.int64(1) << (n - 1 - i)
        mj = np.int64(1) << (n - 1 - j)
        J = p['J'] if 'I2' in use else 0.0j
        A = p['A'] if 'I3' in use else 0.0j
        if J != 0:
            # s=|r_i g_j⟩→t=|g_i r_j⟩：H[t,s]+=conj(J)；s=|g_i r_j⟩→t=|r_i g_j⟩：+=J
            for sel, v in ((((bit[i] == 1) & (bit[j] == 0)), np.conj(J)),
                           (((bit[i] == 0) & (bit[j] == 1)), J)):
                ss = s[sel]
                rows.append((ss ^ mi) ^ mj); cols.append(ss)
                vals.append(np.full(ss.size, v, complex))
        if A != 0:
            # |gg⟩→|rr⟩：H[t,s]+=A；|rr⟩→|gg⟩：+=conj(A)
            for sel, v in ((((bit[i] == 0) & (bit[j] == 0)), A),
                           (((bit[i] == 1) & (bit[j] == 1)), np.conj(A))):
                ss = s[sel]
                rows.append((ss ^ mi) ^ mj); cols.append(ss)
                vals.append(np.full(ss.size, v, complex))

    return sp.coo_matrix(
        (np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))),
        shape=(dim, dim)).tocsr()          # tocsr 自动合并重复 (row,col)


def build_H_ed(calib, Om=None, delta=None, terms=None, gamma=None):
    """calib → 2^N **稠密** H（MHz，复）——tiny-N 交叉对照用。

    内部委托 `build_H_sparse(...).toarray()`（同一矩阵，M0 约定不变）。
    大 N 请直接用 build_H_sparse（稠密 2^N×2^N 内存不可行）。
    """
    return build_H_sparse(calib, Om, delta, terms, gamma).toarray()
