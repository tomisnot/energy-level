# -*- coding: utf-8 -*-
"""两原子相互作用科学计算核心（纯计算，无 UI）。

知识体系（详见 _misc/two_body_design.md）：
    H_pair(R) = H1 ⊗ 1 + 1 ⊗ H2 + V_dd(R)
  单原子部分在 core 中已对角（含 Zeeman），pair 对角能 = E1+E2；
  偶极-偶极相互作用（几何 R, θ, φ；n=(sinθcosφ,sinθsinφ,cosθ)）：
    V_dd = (1/R³) Σ_qq' d1_q T_qq' d2_q',   T = e_q^T (1 − 3 n n^T) e_q'
  θ=0（n∥z）退化为：
    V_dd = (1/R³)[ -d1_{-1}d2_{+1} - d1_{+1}d2_{-1} - 2 d1_0 d2_0 ]
  一阶项（交换/Förster，∝1/R³）：|SP⟩↔|PS⟩
  二阶项（vdW，∝1/R⁶）：须对角化后从本征支提取（不在 raw 对角）

截断（四重）：
  ① 单原子基：已添加组 + 偶极伙伴 |Δn|≤n_window（自动补 vdW 中间态）
  ② 总磁量子数 m_total 分块（V_dd 保 m1+m2，H1+H2 也保）
  ③ 能量窗口 |E_a+E_b−E_ref| ≤ energy_window_mhz（可选）
  ④ 偶极选择定则（Δl=±1, |Δj|≤1）由 dipole_amplitude 内部处理

单位：能级/耦合 MHz，间距 μm，偶极矩阵元 a.u.(e·a0)。
复用 core：ArcCache / dipole_amplitude / _direct_basis / address_pol_coeffs。
C3/C6 不硬编码：由 ARC 偶极矩阵元（core.ArcCache）直接算出，一阶交换/二阶 vdW 自动涌现；
不同原子/态对（n,l,j）自动给出不同值。
"""
import sys
sys.stdout.reconfigure(encoding='utf-8', errors='replace')
import numpy as np
from scipy.constants import physical_constants as pc
import scipy.sparse as sp
from . import core

# ---------------- 单位换算 ----------------
_HARTREE_HZ = pc['hartree-hertz relationship'][0]   # 6.5797e15 Hz
_A0_M = pc['Bohr radius'][0]                        # 5.2918e-11 m
# d(a.u.) 与 R(μm) → 能量 MHz：V_au = d1·d2/R_au³；V_MHz = V_au·Hartree_Hz/1e6
_DD_CONV = _HARTREE_HZ / 1e6 / (1e-6 / _A0_M) ** 3   # MHz·μm³/(a.u.)²
# V_dd 的球基系数 c_q（见 docstring）
VDD_QCOEFF = {1: -1.0, -1: -1.0, 0: -2.0}


# ---------------- 基组 ----------------
def single_atom_basis(cache, groups, B_gauss=0.0, n_window=3):
    """单原子基：已添加组 + 每组偶极伙伴（|Δn|≤n_window）。返回 core 行列表。

    行字段：n,l,j,kind('mJ'/'F'),mj,mF,amps,E_mhz(MHz),qstr。
    """
    return core._direct_basis(cache, groups, B_gauss, n_window)


def _mag(s):
    """磁量子数：mJ 态用 mj，F(超精细) 态用 mF。"""
    return s['mj'] if s['kind'] == 'mJ' else s['mF']


def find_sublevel(single, n, l, j, m):
    """在单原子基中按 (n,l,j,m) 找索引；找不到返回 None。"""
    for i, s in enumerate(single):
        if (s['n'], s['l'], s['j']) == (n, l, j) and abs(_mag(s) - m) < 1e-9:
            return i
    return None


def pair_indices(single, m_total=None, energy_window_mhz=None,
                 ref_energy_mhz=None):
    """pair 基（原子1 ⊗ 原子2）索引 (i, j)，k = i*N+j 为 pair 基下标。

    截断：总磁量子数 m_total（m1+m2）；能量窗口 |E_a+E_b−E_ref|≤window。
    返回 (pairs, E_pair_mhz)：pairs 为 (i,j) 数组，E_pair = E1+E2。
    """
    N = len(single)
    E = np.array([s['E_mhz'] for s in single])
    pairs, E_pair = [], []
    for i in range(N):
        for j in range(N):
            if m_total is not None and abs(_mag(single[i]) + _mag(single[j]) - m_total) > 1e-9:
                continue
            e = E[i] + E[j]
            if energy_window_mhz is not None:
                if ref_energy_mhz is None:
                    raise ValueError('energy_window_mhz 需要 ref_energy_mhz')
                if abs(e - ref_energy_mhz) > energy_window_mhz:
                    continue
            pairs.append((i, j))
            E_pair.append(e)
    return np.array(pairs, dtype=int), np.array(E_pair)


# ---------------- 偶极-偶极相互作用 ----------------
def _dipole_sparse(cache, single, q):
    """单原子偶极矩阵 D_q（N×N，CSR）：D_q[i,i2] = ⟨i2|d_q|i⟩（a.u.）。"""
    N = len(single)
    rows, cols, vals = [], [], []
    for i in range(N):
        si = single[i]
        for i2 in range(N):
            if i2 == i:
                continue
            s2 = single[i2]
            if abs(si['l'] - s2['l']) != 1 or abs(si['j'] - s2['j']) > 1.0 + 1e-9:
                continue
            d = core.dipole_amplitude(cache, si, s2, q)
            if abs(d) > 1e-12:
                rows.append(i); cols.append(i2); vals.append(d)
    return sp.csr_matrix((vals, (rows, cols)), shape=(N, N))


def _vdd_tensor(theta_deg, phi_deg):
    """偶极-偶极张量 T（球基 q,q'∈{+1,0,-1}）：V=(1/R³)Σ_qq' d1q T_qq' d2q'。

    T_qq' = e_q^T (1 − 3 n n^T) e_q'，n=(sinθcosφ,sinθsinφ,cosθ) 为两原子连线方向；
    e_{+1}=−(x+iy)/√2, e_0=z, e_{−1}=(x−iy)/√2。θ=0 退化为
    T_{+1,−1}=T_{−1,+1}=−1、T_{00}=−2（其余 0）。
    """
    th, ph = np.deg2rad(theta_deg), np.deg2rad(phi_deg)
    n = np.array([np.sin(th) * np.cos(ph), np.sin(th) * np.sin(ph), np.cos(th)])
    G = np.eye(3) - 3.0 * np.outer(n, n)
    s = 1.0 / np.sqrt(2)
    E = [np.array([-s, -1j * s, 0.0]), np.array([0.0, 0.0, 1.0]),
         np.array([s, -1j * s, 0.0])]          # e_{+1}, e_0, e_{-1}
    T = np.empty((3, 3), complex)
    for a in range(3):
        for b in range(3):
            T[a, b] = E[a] @ G @ E[b]
    return T


def vdd_sparse(cache, single, R_um, theta_deg=0.0, phi_deg=0.0):
    """偶极-偶极相互作用稀疏矩阵（MHz），pair 基为笛卡尔积 k=i*N+j。

    几何：间距 R，两原子连线方向 n=(θ,φ)（θ=0 → 沿 z 轴）。
    V[k,k2] = (1/R³) Σ_qq' T_qq' ⟨i2|d_q|i⟩⟨j2|d_q'|j⟩  （k=(i,j), k2=(i2,j2)）
    """
    N = len(single)
    D = {q: _dipole_sparse(cache, single, q) for q in (-1, 0, 1)}
    T = _vdd_tensor(theta_deg, phi_deg)
    qs = (1, 0, -1)
    V = None
    for a in range(3):
        for b in range(3):
            if abs(T[a, b]) > 1e-12:
                term = T[a, b] * sp.kron(D[qs[a]], D[qs[b]])
                V = term if V is None else V + term
    return V * (_DD_CONV / R_um ** 3)


# ---------------- pair 哈密顿与对角化 ----------------
def pair_hamiltonian(cache, groups, R_um, theta_deg=0.0, phi_deg=0.0,
                     B_gauss=0.0, n_window=3, m_total=None,
                     energy_window_mhz=None, ref_energy_mhz=None):
    """构建 pair 哈密顿（MHz）。几何 (R, θ, φ)。

    返回 dict：
      single   : 单原子行列表
      pairs    : (i,j) 数组（pair 基，k = i*N+j）
      E_pair   : 对角能 E_a+E_b（MHz）
      H        : H_pair = diag(E_pair) + V_dd(R,θ,φ)（子空间稠密复矩阵）
    """
    single = single_atom_basis(cache, groups, B_gauss, n_window)
    N = len(single)
    pairs, E_pair = pair_indices(single, m_total, energy_window_mhz,
                                 ref_energy_mhz)
    V = vdd_sparse(cache, single, R_um, theta_deg, phi_deg)
    sel = np.array([i * N + j for i, j in pairs], dtype=int)
    H = V.tocsr()[sel][:, sel].toarray()
    H[np.diag_indices(len(sel))] += E_pair
    return dict(single=single, pairs=pairs, E_pair=E_pair, H=H)


def pair_eigen(cache, groups, R_um, theta_deg=0.0, phi_deg=0.0,
               B_gauss=0.0, n_window=3, m_total=None,
               energy_window_mhz=None, ref_energy_mhz=None):
    """两体静态能级：对角化 pair 哈密顿。几何 (R, θ, φ)。

    返回 dict：single/pairs/E_pair + evals(升序 MHz) + evecs(列=本征态)。
    """
    res = pair_hamiltonian(cache, groups, R_um, theta_deg, phi_deg,
                           B_gauss, n_window, m_total, energy_window_mhz,
                           ref_energy_mhz)
    evals, evecs = np.linalg.eigh(res['H'])
    res['evals'] = evals
    res['evecs'] = evecs
    return res


def pair_component(res, a_row, b_row):
    """各本征态在 |a_row ⊗ b_row⟩ 上的权重 |⟨ψ|ab⟩|²（数组，长=本征态数）。"""
    ia = next(i for i, s in enumerate(res['single'])
              if all(s[k] == a_row[k] for k in ('n', 'l', 'j')) and
              abs(_mag(s) - _mag(a_row)) < 1e-9)
    ib = next(i for i, s in enumerate(res['single'])
              if all(s[k] == b_row[k] for k in ('n', 'l', 'j')) and
              abs(_mag(s) - _mag(b_row)) < 1e-9)
    k = np.where((res['pairs'][:, 0] == ia) & (res['pairs'][:, 1] == ib))[0][0]
    return np.abs(res['evecs'][k, :]) ** 2


# ---------------- 实用提取 ----------------
def exchange_coupling(cache, groups, R_um, theta_deg=0.0, phi_deg=0.0,
                      B_gauss=0.0, n_window=3, m_total=None,
                      ket_a=None, ket_b=None):
    """一阶交换耦合 ⟨a|V_dd|b⟩（MHz）——直接查矩阵元，不需对角化。

    ket_a/ket_b：(n,l,j,m) 四元组；默认 60S↔60P 交换（|SP⟩↔|PS⟩）。
    """
    single = single_atom_basis(cache, groups, B_gauss, n_window)
    N = len(single)
    if ket_a is None:
        ket_a = (60, 0, 0.5, 0.5)   # 60S m=+1/2
        ket_b = (60, 1, 0.5, 0.5)   # 60P m=+1/2
    ia = find_sublevel(single, *ket_a)
    ib = find_sublevel(single, *ket_b)
    k1 = ia * N + ib                    # |SP⟩
    k2 = ib * N + ia                    # |PS⟩
    V = vdd_sparse(cache, single, R_um, theta_deg, phi_deg)
    return float(np.real(V[k1, k2]))


def vdw_shift(cache, groups, R_um, theta_deg=0.0, phi_deg=0.0,
              B_gauss=0.0, n_window=3, m_total=None,
              ref_ket=(60, 0, 0.5, 0.5)):
    """vdW 位移（MHz）：对角化后从 |ref⊗ref⟩ 支（overlap 最大本征态）提取。

    返回 (shift_mhz, E_ref_pair)：shift = E_支 − (E_ref+E_ref)。
    """
    res = pair_eigen(cache, groups, R_um, theta_deg, phi_deg, B_gauss,
                     n_window, m_total=m_total)
    ia = find_sublevel(res['single'], *ref_ket)
    w = pair_component(res, res['single'][ia], res['single'][ia])
    im = int(np.argmax(w))
    E_ref = res['single'][ia]['E_mhz']
    return float(res['evals'][im] - 2 * E_ref), float(2 * E_ref)


def pair_potential(cache, groups, R_scan, B_gauss=0.0, n_window=3,
                   m_total=None):
    """扫 R → (R 数组, 能级数组 [n_R, n_levels])。"""
    evals_list = []
    for R in R_scan:
        res = pair_eigen(cache, groups, R, B_gauss, n_window, m_total=m_total)
        evals_list.append(res['evals'])
    return np.asarray(R_scan), np.asarray(evals_list)


# 自检已移至 pair_selfcheck.py（P 系列），本模块保持纯计算（无检查逻辑）。
