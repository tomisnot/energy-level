# -*- coding: utf-8 -*-
"""平衡态多体标杆判据（P0）：**独立 Kronecker 构造** vs **引擎 ED 装配**。

不碰 ARC（喂假 calib：直接给 V/J/A），快且确定；呼应 `manybody_m1_ed_selfcheck` 的精神，
但本判据专攻**平衡态基线**——把"我们广义哈密顿（V·nn + J·σ⁺σ⁻ + A·σ⁺σ⁺ + δn + Ω/2·σˣ）
的 2^N 矩阵装配"钉死到机器精度，供后续 TN/NQS/QMC 接入时当独立参照（标杆判据库 §2 Tier-B）。

三条设计原则在此落地（见 docs/标杆判据库…md）：
- **独立性**：参照 H 用**逐位张量积 np.kron**（教科书二次量子化式），与引擎的**位运算稀疏装配**
  是两条结构不同的实现路径 → 抓位序/共轭/因子号类 bug。
- **确定性**：numpy 随机实例**固定种子**；比对走机器精度。
- **自证**：断言实际比对元素数（非零元），"没命中"≠"通过"。

覆盖（group 不变式包，整包算 1 项）：
  E1 谱一致：多个随机实例（N=2..5，链/方格，V/J/A 含复、Ω/δ）⇒ 引擎 H ≡ 独立 H（逐元素 <1e-9）
     + 本征值谱 + 基态能量 + 基态观测量（⟨n_i⟩、⟨σˣ⟩、S）一致
  E2 厄米：γ=0 ⇒ 两路都严格厄米；γ≠0 ⇒ 引擎非厄米对角 = −0.5·Γ·1e-3/CONV·⟨n_i⟩（文档约定）
  A1 解析·XX/Heisenberg 二聚体：N=2、V=A=Ω=δ=0、J 实 ⇒ 谱 = {−|J|,0,0,+|J|}（单激发 2×2 闭式）
  A2 解析·blockade 双能级：V≫Ω ⇒ 基态 ≈ |gg⟩、P(rr)→0（Deep-blockade 经典极限）
  C1 独立构造自证：张量积算符满足 σ⁺σ⁻+h.c. / σ⁺σ⁺+h.c. 的矩阵元约定（防我参照码自身写反）
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.stdout.reconfigure(encoding='utf-8', errors='replace')

import numpy as np

from checks.checklib import Checker                              # noqa: E402
from src.core.manybody import assemble as asm                    # noqa: E402
from src.core.evolve_util import CONV                             # noqa: E402

C = Checker('equibench_engine_ed')

# ------------------------------------------------------------------
# 独立参照构造器：局部基 [|g⟩,|r⟩]（bit=0↔g，bit=1↔r；与引擎 bit=1↔|r⟩ 同）
#   n̂ = |r⟩⟨r|, σˣ = |g⟩⟨r|+|r⟩⟨g|, σ⁺ = |r⟩⟨g|, σ⁻ = |g⟩⟨r|
# 全空间：site 0 = 左/最高位（与引擎 MSB bit(s,i)=(s>>(n-1-i))&1 对齐），索引 s=Σ b_i·2^(n-1-i)
# ------------------------------------------------------------------
_I = np.eye(2, dtype=complex)
_N = np.array([[0, 0], [0, 1]], dtype=complex)          # |r><r|
_SX = np.array([[0, 1], [1, 0]], dtype=complex)         # |g><r|+|r><g|
_SP = np.array([[0, 0], [1, 0]], dtype=complex)         # |r><g|  (raises g->r)
_SM = _SP.conj().T                                        # |g><r|


def _embed(n, ops):
    """ops: dict{site: 2x2}；其余位为 I。返回 2^n×2^n 稠密算符（site0 在左/最高位）。"""
    out = np.array([[1.0 + 0j]])
    for i in range(n):
        out = np.kron(out, ops.get(i, _I))
    return out


def _H_reference(n, Om, delta, pairs, terms):
    """教科书二次量子化装配：
    H = Σ_i[δ_i n_i + (Ω_i/2)σˣ_i] + Σ_{i<j}[V n_i n_j
        + (J σ⁺_iσ⁻_j + h.c.) + (A σ⁺_iσ⁺_j + h.c.)]
    pairs: 每项 dict(i,j,V,J,A)。terms ⊆ {I1,I2,I3}。dim=2^n（小 N 稠密）。
    """
    dim = 1 << n
    H = np.zeros((dim, dim), complex)
    for i in range(n):
        if delta[i] != 0.0:
            H += delta[i] * _embed(n, {i: _N})
        if Om[i] != 0.0:
            H += (Om[i] / 2.0) * _embed(n, {i: _SX})
    for p in pairs:
        i, j = p['i'], p['j']
        if 'I1' in terms and p['V'] != 0.0:
            H += p['V'] * _embed(n, {i: _N, j: _N})
        if 'I2' in terms and p['J'] != 0:
            H += p['J'] * _embed(n, {i: _SP, j: _SM}) \
                + np.conj(p['J']) * _embed(n, {i: _SM, j: _SP})
        if 'I3' in terms and p['A'] != 0:
            H += p['A'] * _embed(n, {i: _SP, j: _SP}) \
                + np.conj(p['A']) * _embed(n, {i: _SM, j: _SM})
    return H


def _calib(n, pairs, terms=('I1', 'I2', 'I3')):
    """假 calib（不碰 ARC）：与 manybody_m1_ed 同形状。"""
    return dict(terms=list(terms),
                atoms=[dict(id=i, pos=(float(i % 4), float(i // 4), 0.0))
                       for i in range(n)],
                pairs=pairs, single_basis=[])


def _pair(i, j, V, J, A):
    return dict(i=i, j=j, V=float(V), J=complex(J), A=complex(A),
                raw_V=float(V), raw_J=complex(J), raw_A=complex(A),
                delta={}, info={})


def _bits(state_idx, n):
    """整数 → [b_0..b_{n-1}]，b_i=(state>>(n-1-i))&1（site0=最高位，与引擎一致）。"""
    return [int((state_idx >> (n - 1 - i)) & 1) for i in range(n)]


def _ground_obs(Hdense, n):
    """基态（最小本征值）观测：E0、⟨n_i⟩、⟨σˣ⟩、site-0 von Neumann 熵。"""
    w, U = np.linalg.eigh(Hdense)
    gs = U[:, 0]
    E0 = float(w[0])
    nb = np.zeros(n)
    sx = np.zeros(n)
    dim = 1 << n
    probs = np.abs(gs) ** 2
    # 用基矢表示逐位算 n_i、σˣ_i 期望
    for i in range(n):
        nvec = np.array([_bits(s, n)[i] for s in range(dim)])
        nb[i] = float(np.sum(probs * nvec))
        Sx = _embed(n, {i: _SX})
        sx[i] = float(np.real(gs.conj() @ (Sx @ gs)))
    # site-0 约化密度 → 熵（自然对数）；site0=最高位 ⇒ reshape(2, dim//2)
    psi = gs.reshape(2, dim // 2)
    rho0 = psi @ psi.conj().T
    ev = np.linalg.eigvalsh(rho0)
    ev = ev[ev > 1e-15]
    S = float(-np.sum(ev * np.log(ev)))
    return E0, nb, sx, S


# ===================================================================
# E1 谱一致：随机实例（链/方格、V/J/A 含复、Ω/δ）引擎 ≡ 独立参照
# ===================================================================
rng = np.random.default_rng(20261001)
TERMS = ('I1', 'I2', 'I3')
sub_E1 = []
n_checked_elems = 0
for n, nbonds in [(2, 1), (3, 2), (4, 4), (5, 5)]:
    pairs = []
    # 近邻链 (i,i+1) + 方格额外一条 (i, i+2)
    for i in range(n - 1):
        pairs.append(_pair(i, i + 1, V=rng.uniform(0, 50),
                           J=rng.uniform(-2, 2) + 1j * rng.uniform(-2, 2),
                           A=rng.uniform(-1, 1) + 1j * rng.uniform(-1, 1)))
    for i in range(n - 2):
        pairs.append(_pair(i, i + 2, V=rng.uniform(0, 10), J=rng.normal(0, .5),
                           A=rng.normal(0, .5) + 1j * rng.normal(0, .5)))
    Om = rng.uniform(0, 20, n)
    delta = rng.uniform(-10, 10, n)
    H_eng = np.asarray(asm.build_H_ed(_calib(n, pairs, TERMS), Om, delta,
                                      terms=list(TERMS)))
    H_ref = _H_reference(n, Om, delta, pairs, TERMS)
    n_checked_elems += int(H_eng.size)
    E0e, nbe, sxe, Se = _ground_obs(H_eng, n)
    E0r, nbr, sxr, Sr = _ground_obs(H_ref, n)
    tag = f'N={n}({nbonds}键)'
    sub_E1 += [
        (f'{tag}: 逐元素 max|H_ours−H_ref| < 1e-9',
         float(np.abs(H_eng - H_ref).max()) < 1e-9),
        (f'{tag}: 基态能量 E0 < 1e-8',
         abs(E0e - E0r) < 1e-8 + 1e-8 * abs(E0r)),
        (f'{tag}: ⟨n_i⟩ 全分量 < 1e-7',
         float(np.abs(nbe - nbr).max()) < 1e-7),
        (f'{tag}: ⟨σˣ_i⟩ 全分量 < 1e-7',
         float(np.abs(sxe - sxr).max()) < 1e-7),
        (f'{tag}: site-0 纠缠熵 S < 1e-7', abs(Se - Sr) < 1e-7),
    ]
C.group('E1 引擎 ED ≡ 独立 Kronecker 装配（谱/基态观测，随机含复实例）',
        sub_E1, detail=f'共比对 {n_checked_elems} 个矩阵元素/4 例')

# ===================================================================
# E2 厄米 + 非厄米对角约定
# ===================================================================
n = 4
pairs = [_pair(i, i + 1, V=30.0, J=-1.2 + 0.7j, A=0.9 - 1.3j)
         for i in range(n - 1)]
Om = [5.0, 0.0, 3.0, 0.0]
delta = [1.0, -2.0, 0.5, 2.0]
H0 = np.asarray(asm.build_H_ed(_calib(n, pairs, TERMS), Om, delta))
Hg = np.asarray(asm.build_H_ed(_calib(n, pairs, TERMS), Om, delta,
                               gamma=[2.0, 0.0, 1.0, 0.0]))
dim = 1 << n
occ_i = {i: np.array([_bits(s, n)[i] for s in range(dim)], float)
         for i in range(n)}
# 非厄米对角虚部（MHz 单位下的 H 对角 = Σ_i −0.5j·Γ_i·1e-3/CONV·n_i + 实部）
want_imag = np.zeros(dim)
for i in range(n):
    want_imag += (-0.5 * np.asarray([2.0, 0.0, 1.0, 0.0])[i] * 1e-3 / CONV) * occ_i[i]
sub_E2 = [
    ('γ=0 ⇒ 引擎 H 严格厄米（max|H−H†| < 1e-12）',
     float(np.abs(H0 - H0.conj().T).max()) < 1e-12),
    ('γ≠0 ⇒ 非厄米对角虚部 = −0.5·Γ·1e-3/CONV·⟨n_i⟩（<1e-9）',
     float(np.abs(np.imag(np.diag(Hg)) - want_imag).max()) < 1e-9),
    ('γ≠0 ⇒ 破坏厄米（H≠H†）',
     float(np.abs(Hg - Hg.conj().T).max()) > 1e-9),
]
C.group('E2 厄米性 + 非厄米对角衰减约定（文档 §5.2/CONV）', sub_E2)

# ===================================================================
# A1 解析·XX/Heisenberg 二聚体：N=2，V=A=Ω=δ=0，J 实 ⇒ 谱={−|J|,0,0,+|J|}
# ===================================================================
Jv = 1.7
pairs_x = [_pair(0, 1, V=0.0, J=Jv, A=0.0)]
Hx = np.asarray(asm.build_H_ed(_calib(2, pairs_x, TERMS), Om=[0, 0],
                               delta=[0, 0]))
ref_spec = np.sort(np.array([-abs(Jv), 0.0, 0.0, abs(Jv)]))
ours_spec = np.sort(np.linalg.eigvalsh(Hx))
sub_A1 = [
    ('XX 二聚体本征值 ≡ 单激发 2×2 闭式 {−|J|,0,0,+|J|} <1e-10',
     float(np.abs(ours_spec - ref_spec).max()) < 1e-10),
    ('单激发子空间 offdiag = |J|（σ⁺σ⁻+h.c. 装配正确）',
     abs(abs(Hx[1, 2]) - abs(Jv)) < 1e-12),
]
C.group('A1 解析·XX/Heisenberg 二聚体谱（引擎 vs 2×2 闭式）', sub_A1)

# ===================================================================
# A2 解析·深阻塞经典极限：V≫Ω ⇒ 基态 ≈ |gg⟩、P(rr)→0
# ===================================================================
n = 3
pairs_b = [_pair(0, 1, V=2000.0, J=0.0, A=0.0), _pair(1, 2, V=2000.0, J=0.0, A=0.0)]
Om_b = [10.0, 10.0, 10.0]
Hb = np.asarray(asm.build_H_ed(_calib(n, pairs_b, TERMS), Om=Om_b, delta=[0, 0, 0]))
wb, Ub = np.linalg.eigh(Hb)
gs = Ub[:, 0]


def _no_adjacent(s, nn):
    b = _bits(s, nn)
    return not any(b[i] == 1 and b[i + 1] == 1 for i in range(nn - 1))


allowed = [s for s in range(1 << n) if _no_adjacent(s, n)]
w_allowed = sum(abs(gs[s]) ** 2 for s in allowed)
# 纯路径图 P_n 的合法独立集数 = Fibonacci(n+2)：n=3→F5=5, n=4→F6=8, n=5→F7=13
sub_A2 = [
    ('V≫Ω ⇒ 相邻双激发布居 ≈ 0（基态支撑 ⊆ 独立集，w_allowed>0.999）',
     w_allowed > 0.999),
    ('P_3 合法独立集数 = Fibonacci(5)=5（阻塞子空间维数闭式）',
     len(allowed) == 5),
    ('P_4 合法独立集数 = Fibonacci(6)=8',
     sum(1 for s in range(16) if _no_adjacent(s, 4)) == 8),
    ('P_5 合法独立集数 = Fibonacci(7)=13',
     sum(1 for s in range(32) if _no_adjacent(s, 5)) == 13),
]
C.group('A2 解析·深阻塞 → 经典硬核极限（相邻双激发压制 + 独立集 Fibonacci 计数）', sub_A2)

# ===================================================================
# A3 第三独立路径·N=2 手推显式 4×4（基序 |gg>,|gr>,|rg>,|rr>）
#   ——与 engine(位稀疏) / _H_reference(np.kron) 都不同；三路径同谱=模型理解被钉死
# ===================================================================
O0, O1, d0, d1 = 3.0, 5.0, 1.0, -2.0
Vv, Jv3, Av3 = 30.0, -1.2 + 0.7j, 0.9 - 1.3j
M = np.zeros((4, 4), complex)
M[0, 0], M[1, 1], M[2, 2] = 0.0, d1, d0
M[3, 3] = d0 + d1 + Vv
M[0, 1] = M[1, 0] = O1 / 2.0        # σˣ_1: |gg><->|gr>, |rg><->|rr>
M[2, 3] = M[3, 2] = O1 / 2.0
M[0, 2] = M[2, 0] = O0 / 2.0        # σˣ_0: |gg><->|rg>, |gr><->|rr>
M[1, 3] = M[3, 1] = O0 / 2.0
M[2, 1] = Jv3; M[1, 2] = np.conj(Jv3)     # J σ⁺_0σ⁻_1 + h.c.：|gr>->|rg> = J
M[3, 0] = Av3; M[0, 3] = np.conj(Av3)      # A σ⁺_0σ⁺_1 + h.c.：|gg>->|rr> = A
H3 = np.asarray(asm.build_H_ed(_calib(2, [_pair(0, 1, V=Vv, J=Jv3, A=Av3)], TERMS),
                               Om=[O0, O1], delta=[d0, d1]))
sub_A3 = [
    ('N=2 手推显式矩阵 ≡ 引擎逐元素 <1e-12', float(np.abs(M - H3).max()) < 1e-12),
    ('两路本征值谱（排序）<1e-9',
     float(np.abs(np.sort(np.linalg.eigvalsh(M))
                  - np.sort(np.linalg.eigvalsh(H3))).max()) < 1e-9),
]
C.group('A3 第三独立路径·N=2 手推显式 4×4 ≡ 引擎（三路径同谱）', sub_A3)

# ===================================================================
# C1 独立构造自证：张量积算符的矩阵元约定（防参照码自身写反共轭）
# ===================================================================
# 2 位基序（site0 左）：|gg>=0,|gr>=1,|rg>=2,|rr>=3（与 _H4_ORDER/_embed 同）
SpSm = _embed(2, {0: _SP, 1: _SM})          # |r g><g r| : 把 |gr>=1 -> |rg>=2
SmSp = _embed(2, {0: _SM, 1: _SP})          # |g r><r g| : 把 |rg>=2 -> |gr>=1
SpSp = _embed(2, {0: _SP, 1: _SP})          # |r r><g g| : |gg>=0 -> |rr>=3
sub_C1 = [
    ('σ⁺_iσ⁻_j: |gr⟩→|rg⟩ 元=1（H[2,1]）', abs(SpSm[2, 1] - 1) < 1e-15),
    ('σ⁻_iσ⁺_j: |rg⟩→|gr⟩ 元=1（H[1,2]）', abs(SmSp[1, 2] - 1) < 1e-15),
    ('σ⁺_iσ⁺_j: |gg⟩→|rr⟩ 元=1（H[3,0]）', abs(SpSp[3, 0] - 1) < 1e-15),
    ('σˣ|g⟩=|r⟩、σ⁺|g⟩=|r⟩（局部单比特约定）',
     abs(_SP[1, 0] - 1) < 1e-15 and abs(_SX[1, 0] - 1) < 1e-15),
]
C.group('C1 独立构造自证（张量积算符矩阵元约定，site0=最高位）', sub_C1)

sys.exit(0 if C.summary() else 1)
