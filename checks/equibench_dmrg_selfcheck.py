# -*- coding: utf-8 -*-
"""2D 柱形 DMRG 标杆（Tier-C）：同一 2D 阶梯 Ising 的稠密 H 四方对齐。

把 TeNPy 模型的全空间 H（2^N 稠密）同时交给：
- **TeNPy DMRG**（`dmrg_ising_ladder`，被测后端）
- **我们的 ED**（`equilibrium.ground_state`）
- **我们的 GPU 批量 ED**（`equilibrium.batched_ground_state`）
- **独立 np.kron 阶梯构造**（与 TeNPy 格点/排序无关的第三方实现）
四者基态能量必须一致（ED/GPU/kron 机器精度；DMRG 到截断容差）。另加**经典极限解析锚**：
hx=hz=0 时 2 腿×L 阶梯（二部）AF Ising 基态 E0 = −|Jz|/4·(3L−2)（每键 −|Jz|/4，键数=3L−2）。
这把“2D 柱形 DMRG 可用且正确”与“我们的 ED/GPU 求解器在真 2D H 上正确”一次验透。
无 TeNPy 时本判据 SKIP（不误报）。
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.stdout.reconfigure(encoding='utf-8', errors='replace')

import numpy as np
import scipy.sparse as sp

from checks.checklib import Checker                              # noqa: E402
from src.core.manybody import equilibrium as eq                  # noqa: E402

C = Checker('equibench_dmrg')

try:
    from src.core.manybody import dmrg as _dmrg
    _dmrg.ising_ladder_full_hamiltonian(2, 1.0, 0.5)            # 触发 TeNPy 导入探活
    HAS_TENPY = True
except Exception as e:
    HAS_TENPY = False
    print('  [info] TeNPy 不可用（%r）→ 本判据将 SKIP' % e)

I2 = np.eye(2, dtype=complex)
_Sx = 0.5 * np.array([[0, 1], [1, 0]], complex)
_Sz = 0.5 * np.array([[1, 0], [0, -1]], complex)


def _embed(n, ops):
    out = np.array([[1.0 + 0j]])
    for i in range(n):
        out = np.kron(out, ops.get(i, I2))
    return out


def _ladder_bonds(L):
    """2 腿 × 长度 L 阶梯（site=2x+leg），开边界：rung + 2 leg 方向 NN。"""
    bonds = [(2 * x, 2 * x + 1) for x in range(L)]                       # 腿间 rung
    bonds += [(2 * x, 2 * x + 2) for x in range(L - 1)]                 # leg0 沿链
    bonds += [(2 * x + 1, 2 * x + 3) for x in range(L - 1)]            # leg1 沿链
    return bonds


def _ladder_hamiltonian(L, Jz, hx, hz):
    n = 2 * L
    H = np.zeros((1 << n, 1 << n), complex)
    for i in range(n):
        H += -hx * _embed(n, {i: _Sx}) - hz * _embed(n, {i: _Sz})
    for (i, j) in _ladder_bonds(L):
        H += Jz * _embed(n, {i: _Sz, j: _Sz})
    return H


if not HAS_TENPY:
    C.skip('TeNPy 2D 柱形 DMRG 四方对齐', 'tenpy 未安装/导入失败')
    sys.exit(0 if C.summary() else 1)

subs, subs_class = [], []
for (L, Jz, hx, hz) in [(3, 1.0, 0.5, 0.0), (4, 1.0, 1.2, 0.3), (5, 0.8, 0.6, 0.0)]:
    Ht, n = _dmrg.ising_ladder_full_hamiltonian(L, Jz, hx, hz)
    E_tenpy = float(np.linalg.eigvalsh(Ht)[0])
    E_ours = eq.ground_state(sp.csr_matrix(Ht))['E0']
    E_gpu = eq.batched_ground_state(Ht[None, ...])['E0'][0]
    E_ind = float(np.linalg.eigvalsh(_ladder_hamiltonian(L, Jz, hx, hz))[0])
    E_dmrg = _dmrg.dmrg_ising_ladder(L, Jz, hx, hz)['E0']
    tag = 'L=%d(N=%d) Jz=%.1f hx=%.1f hz=%.1f' % (L, n, Jz, hx, hz)
    subs += [
        (f'{tag}: TeNPy 稠密 H ≡ 独立 Kron 阶梯 谱 <1e-8', abs(E_tenpy - E_ind) < 1e-8),
        (f'{tag}: 我们的 ED ≡ TeNPy 稠密 H <1e-8', abs(E_ours - E_tenpy) < 1e-8),
        (f'{tag}: GPU/批量 ED ≡ TeNPy 稠密 H <1e-8', abs(E_gpu - E_tenpy) < 1e-8),
        (f'{tag}: DMRG ≡ 精确基态 <1e-6', abs(E_dmrg - E_tenpy) < 1e-6),
    ]

# 经典极限解析锚：hx=hz=0，二部 AF 阶梯 E0 = −Jz/4 · 键数，键数 = rung L + leg 2(L−1) = 3L−2
for L in (3, 4, 5):
    Jz = 1.0
    nbonds = 3 * L - 2
    E_classical_analytic = -Jz / 4.0 * nbonds
    E_ind0 = float(np.linalg.eigvalsh(_ladder_hamiltonian(L, Jz, 0.0, 0.0))[0])
    E_dmrg0 = _dmrg.dmrg_ising_ladder(L, Jz, 0.0, 0.0)['E0']
    subs_class += [
        (f'L={L}: 独立 Kron E0 ≡ −Jz/4·(3L−2) 经典解析 <1e-9',
         abs(E_ind0 - E_classical_analytic) < 1e-9),
        (f'L={L}: DMRG E0 ≡ 经典解析 <1e-8', abs(E_dmrg0 - E_classical_analytic) < 1e-8),
    ]

# 截断收敛：更长阶梯（ED 不可行，N=16）看 DMRG 随 truncerr 收紧而稳定
loose = _dmrg.dmrg_ising_ladder(8, 1.0, 0.9, 0.0, truncerr=1e-6)['E0']
tight = _dmrg.dmrg_ising_ladder(8, 1.0, 0.9, 0.0, truncerr=1e-12)['E0']
subs_class += [('N=16(L=8): DMRG truncerr 1e-6→1e-12 收敛 |ΔE|<1e-5',
                abs(loose - tight) < 1e-5)]

C.group('C·2D 柱形 DMRG ≡ 我们的 ED ≡ GPU 批量 ≡ 独立 Kron（同一稠密 H 四方对齐）', subs)
C.group('C·2D 阶梯：经典极限解析锚 + chi 收敛', subs_class)

sys.exit(0 if C.summary() else 1)
