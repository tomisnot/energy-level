# -*- coding: utf-8 -*-
"""平衡态**解析闭式**判据（Tier-A）：自由费米子谱 + 有限温 Gibbs 闭式 vs 求解器。

- **XX 链 ↔ 自由费米子**：H=JΣ(σ⁺σ⁻+h.c.)（V=A=Ω=δ=0，开链）经 Jordan-Wigner 精确对易到
  单粒子 hopping 矩阵；多体基态 **E₀ = Σ_k min(0, μ_k)**（μ_k=hopping 矩阵本征值）。
  独立 O(N) 闭式 vs 求解器 O(2^N) 基态 → 强约束（Lanczos 与稠密 eigh 都验）。
- **二聚体热力学**：谱 {−|J|,0,0,+|J|} → E(β)、lnZ、C(β) 闭式 vs `thermal_from_h`。
- **N=1 顺磁 Curie**：两能级 ε → ⟨n⟩_β、χ_β=β⟨n⟩(1−⟨n⟩) 闭式（经典配分）。
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

C = Checker('equibench_analytic_thermo')
TERMS = ('I1', 'I2', 'I3')


def _calib(n, pairs, terms=TERMS):
    return dict(terms=list(terms),
                atoms=[dict(id=i, pos=(float(i), 0.0, 0.0)) for i in range(n)],
                pairs=pairs, single_basis=[])


def _pair(i, j, V=0.0, J=0j, A=0j):
    return dict(i=i, j=j, V=float(V), J=complex(J), A=complex(A), raw_V=float(V),
                raw_J=complex(J), raw_A=complex(A), delta={}, info={})


# ---------- XX 链自由费米子 ----------
subs_ff = []
for n in (4, 5, 6, 7, 8):
    Jx = 0.5
    bonds = [(i, i + 1) for i in range(n - 1)]
    H = asm.build_H_sparse(_calib(n, [_pair(i, j, J=Jx) for i, j in bonds], ('I2',)),
                           Om=np.zeros(n), delta=np.zeros(n))
    E0_dense = eq.ground_state(H)['E0']
    E0_lanczos = eq.ground_state(H, dense_max=0)['E0']
    Adj = np.zeros((n, n))
    for i, j in bonds:
        Adj[i, j] = Adj[j, i] = Jx
    E0_ff = float(np.sum(np.minimum(0.0, np.linalg.eigvalsh(Adj))))   # 独立 O(N) 闭式
    subs_ff += [
        (f'n={n}: 稠密eigh基态 ≡ 自由费米子 Σmin(0,μ) <1e-9',
         abs(E0_dense - E0_ff) < 1e-9),
        (f'n={n}: Lanczos基态 ≡ 自由费米子闭式 <1e-9', abs(E0_lanczos - E0_ff) < 1e-9),
    ]
C.group('A·XX 链自由费米子：E₀(多体求解器) ≡ Σmin(0,μ)(单粒子闭式, O(N))', subs_ff)

# ---------- 二聚体热力学 ----------
Jd = 2.0
Hd = asm.build_H_ed(_calib(2, [_pair(0, 1, J=Jd)], ('I2',)), Om=[0, 0], delta=[0, 0])
subs_dm = []
for beta in (0.05, 0.5, 3.0):
    th = eq.thermal_from_h(sp.csr_matrix(Hd), beta)
    ch = np.cosh(beta * Jd)
    lnZ_an = float(np.log(2 + 2 * ch))
    E_an = -Jd * np.tanh(0.5 * beta * Jd)
    E2_an = Jd ** 2 * ch / (1 + ch)
    C_an = beta ** 2 * (E2_an - E_an ** 2)
    subs_dm += [
        (f'β={beta}: E ≡ -J·tanh(βJ/2) <1e-9', abs(th['E'] - E_an) < 1e-9),
        (f'β={beta}: lnZ ≡ ln(2+2cosh βJ) <1e-9', abs(th['lnZ'] - lnZ_an) < 1e-9),
        (f'β={beta}: C ≡ β²(⟨E²⟩−⟨E⟩²) <1e-9', abs(th['C'] - C_an) < 1e-9),
    ]
C.group('A·XX 二聚体热力学：E/lnZ/C ≡ 四能级谱闭式', subs_dm)

# ---------- N=1 顺磁 Curie ----------
eps = 3.0
H1 = asm.build_H_ed(_calib(1, [], ('I1',)), Om=[0.0], delta=[eps])
sub_cur = []
for beta in (0.2, 1.0, 6.0):
    th = eq.thermal_from_h(sp.csr_matrix(H1), beta)
    n_an = 1.0 / (np.exp(beta * eps) + 1.0)
    chi_an = beta * n_an * (1 - n_an)
    lnZ_an = float(np.log(1 + np.exp(-beta * eps)))
    sub_cur += [
        (f'β={beta}: ⟨n⟩ ≡ 1/(e^βε+1) <1e-12', abs(th['n_atom'][0] - n_an) < 1e-12),
        (f'β={beta}: χ ≡ β⟨n⟩(1−⟨n⟩) (Curie) <1e-12', abs(th['chi_n'] - chi_an) < 1e-12),
        (f'β={beta}: lnZ ≡ ln(1+e^−βε) <1e-12', abs(th['lnZ'] - lnZ_an) < 1e-12),
    ]
C.group('A·N=1 两能级 Gibbs：⟨n⟩/χ/lnZ ≡ 玻尔兹曼闭式', sub_cur)

# β→∞ 基态一致性：thermal E(β→大) → E0（低温极限）
th_low = eq.thermal_from_h(sp.csr_matrix(Hd), beta=200.0)
E0_dimer = eq.ground_state(sp.csr_matrix(Hd))['E0']
C.check('热力学低温极限 E(β→∞) ≡ 基态 E0（<1e-9）',
        abs(th_low['E'] - E0_dimer) < 1e-9,
        f'thermal={th_low["E"]:.8f} E0={E0_dimer:.8f}')

sys.exit(0 if C.summary() else 1)
