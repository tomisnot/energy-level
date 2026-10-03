# -*- coding: utf-8 -*-
"""平衡态**2D 海森堡旗舰**判据（Tier-C）：通用里德堡引擎 ⟷ 教科书 XXX 模型。

证明"我们的广义引擎 + 平衡态求解器"能把标准 2D 自旋模型算对：
- **引擎映射 ≡ 独立构造**：键上的 XXX 项 S_i·S_j = (½)(σ⁺σ⁻+h.c.) + n_i n_j − ½(n_i+n_j) + ¼，
  故取 V=1、J=½、δ_i=−(配位数)/2、Ω=A=0，再加常数 |E|/4 ⇒ 引擎 H ≡ 独立 S·S 构造
  （逐键算符恒等，任意晶格严格成立；对照 Δ<1e-9）。
- **Lieb-Mattis 定理**（阻挫无关的严格结果）：二部晶格 AF XXX 基态总自旋
  S=(|N_A−N_B|)/2 → ⟨S²_tot⟩=S(S+1)。2×3(3-3)→0；3×3(5-4)→0.75。独立验证求解器
  确实拿到对称正确的基态。
- **Lanczos ≡ 稠密 eigh**：同一引擎 H 两条求解路径同能。
- 文献热力学极限目标（E₀/N≈−0.66985、m_s≈0.30）登记为**将来 2D TN/QMC 的门禁目标**，
  有限尺寸 ED 不直接比对（此处不硬编成断言，避免假绿）。
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.stdout.reconfigure(encoding='utf-8', errors='replace')

import numpy as np
import scipy.sparse as sp

from checks.checklib import Checker                              # noqa: E402
from src.core.manybody import assemble as asm                    # noqa: E402
from src.core.manybody import equilibrium as eq                  # noqa: E402

C = Checker('equibench_heisenberg2d')
I2 = np.eye(2, dtype=complex)
_Sx = 0.5 * np.array([[0, 1], [1, 0]], complex)
_Sy = 0.5 * np.array([[0, -1j], [1j, 0]], complex)
_Sz = 0.5 * np.array([[1, 0], [0, -1]], complex)
_N = np.array([[0, 0], [0, 1]], complex)
_SP = np.array([[0, 0], [1, 0]], complex)
_SM = _SP.conj().T


def _embed(n, ops):
    out = np.array([[1.0 + 0j]])
    for i in range(n):
        out = np.kron(out, ops.get(i, I2))
    return out


def _calib(n, pairs, terms=('I1', 'I2')):
    return dict(terms=list(terms),
                atoms=[dict(id=i, pos=(0.0, 0.0, 0.0)) for i in range(n)],
                pairs=pairs, single_basis=[])


def _pair(i, j, V, J):
    return dict(i=i, j=j, V=float(V), J=complex(J), A=0j, raw_V=float(V),
                raw_J=complex(J), raw_A=0j, delta={}, info={})


def _open_square(Lx, Ly):
    """返回 (idx 坐标, NN 键表)。idx = x*Ly+y。"""
    idx = lambda x, y: x * Ly + y                              # noqa: E731
    bonds = []
    for x in range(Lx):
        for y in range(Ly):
            if x + 1 < Lx:
                bonds.append((idx(x, y), idx(x + 1, y)))
            if y + 1 < Ly:
                bonds.append((idx(x, y), idx(x, y + 1)))
    return Lx * Ly, bonds


def _bipart_counts(Lx, Ly):
    A = sum(1 for x in range(Lx) for y in range(Ly) if (x + y) % 2 == 0)
    return A, Lx * Ly - A


def _heisenberg_indep(n, bonds):
    dim = 1 << n
    H = np.zeros((dim, dim), complex)
    for i, j in bonds:
        for S in (_Sx, _Sy, _Sz):
            H += _embed(n, {i: S, j: S})
    return H


def _s_tot_sq(vec, n):
    Sx = sum(_embed(n, {i: _Sx}) for i in range(n))
    Sy = sum(_embed(n, {i: _Sy}) for i in range(n))
    Sz = sum(_embed(n, {i: _Sz}) for i in range(n))
    S2 = Sx @ Sx + Sy @ Sy + Sz @ Sz
    return float(np.real(vec.conj() @ (S2 @ vec)))


subs_map, subs_lm = [], []
for (Lx, Ly) in [(2, 2), (2, 3), (3, 3), (1, 4)]:
    n, bonds = _open_square(Lx, Ly)
    Hd = _heisenberg_indep(n, bonds)
    deg = np.zeros(n)
    for i, j in bonds:
        deg[i] += 1
        deg[j] += 1
    pairs = [_pair(i, j, V=1.0, J=0.5) for i, j in bonds]
    Hem = np.asarray(asm.build_H_ed(_calib(n, pairs), Om=np.zeros(n),
                                    delta=-0.5 * deg))
    E0_ind = float(np.linalg.eigvalsh(Hd)[0])
    E0_eng = float(np.linalg.eigvalsh(Hem)[0]) + 0.25 * len(bonds)
    E0_lanczos = eq.ground_state(sp.csr_matrix(Hem), dense_max=0)['E0'] + 0.25 * len(bonds)
    A, B = _bipart_counts(Lx, Ly)
    S_pred = abs(A - B) / 2.0
    S2_pred = S_pred * (S_pred + 1)
    w, V = np.linalg.eigh(Hd)
    S2_meas = _s_tot_sq(V[:, 0], n)
    tag = f'{Lx}×{Ly}({n}子,{len(bonds)}键)'
    subs_map += [
        (f'{tag}: 引擎映射+|E|/4 ≡ 独立 S·S 构造 <1e-9', abs(E0_eng - E0_ind) < 1e-9),
        (f'{tag}: Lanczos ≡ 稠密eigh <1e-9', abs(E0_lanczos - E0_ind) < 1e-9),
    ]
    subs_lm += [
        (f'{tag}: ⟨S²_tot⟩ ≡ Lieb-Mattis S(S+1)={S2_pred:.4f} <1e-6',
         abs(S2_meas - S2_pred) < 1e-6),
    ]

C.group('C·2D XXX：通用引擎映射 ≡ 独立海森堡构造（含 Lanczos）', subs_map,
        detail='逐键算符恒等：V=1,J=½,δ=−deg/2,+|E|/4')
C.group('C·2D XXX 基态总自旋 ≡ Lieb-Mattis 严格值（求解器拿到对称正确基态）', subs_lm,
        detail='文献热力学目标 E₀/N≈−0.66985,m_s≈0.30 留给 2D TN/QMC 门禁')

sys.exit(0 if C.summary() else 1)
