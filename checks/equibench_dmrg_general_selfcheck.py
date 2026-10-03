# -*- coding: utf-8 -*-
"""① 一般复耦合喂进 TeNPy MPO 的验证：DMRG 解 general Rydberg 阶梯 == 引擎精确 H。

在 2-leg 阶梯上，把**逐键复耦合 V·nn + J·σ⁺σ⁻(复) + A·σ⁺σ⁺(复, 双激发/对产生) + δn + Ωσˣ**
经算符分解喂进 TeNPy MPO，然后：
- **TeNPy ExactDiag 谱 == 引擎 build_H 的 H 谱**（同图同参，机器精度）→ 证 MPO 分解/复共轭/常数都对；
- **DMRG == 精确基态**（含 A、复 J/A）→ 证 DMRG 后端能解一般（非 Ising）里德堡 H。
无 TeNPy 时 SKIP。
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.stdout.reconfigure(encoding='utf-8', errors='replace')

import numpy as np

from checks.checklib import Checker                              # noqa: E402
from src.core.manybody import assemble as asm                    # noqa: E402
from src.core.manybody import dmrg as dm                         # noqa: E402

C = Checker('equibench_dmrg_general')

try:
    dm.ising_ladder_full_hamiltonian(2)
    HAS_TENPY = True
except Exception as e:
    HAS_TENPY = False
    print('  [info] TeNPy 不可用（%r）→ SKIP' % e)

TERMS = ('I1', 'I2', 'I3')


def _calib(n, pairs):
    return dict(terms=list(TERMS),
                atoms=[dict(id=i, pos=(float(i % 2), float(i // 2), 0.0)) for i in range(n)],
                pairs=pairs, single_basis=[])


def _pair(i, j, V, J, A):
    return dict(i=i, j=j, V=complex(V), J=complex(J), A=complex(A), raw_V=complex(V),
                raw_J=complex(J), raw_A=complex(A), delta={}, info={})


def engine_ladder_H(length, V_r, J_r, A_r, V_l, J_l, A_l, delta, Omega):
    """我们引擎里的同一 2-leg 阶梯（atom=x*2+leg）：rung + leg 分型复耦合。"""
    n = 2 * length
    pairs = []
    for x in range(length):                                   # rung (leg0-leg1)
        pairs.append(_pair(2 * x, 2 * x + 1, V_r, J_r, A_r))
    for x in range(length - 1):                               # leg0, leg1 沿链
        pairs.append(_pair(2 * x, 2 * x + 2, V_l, J_l, A_l))
        pairs.append(_pair(2 * x + 1, 2 * x + 3, V_l, J_l, A_l))
    return np.asarray(asm.build_H_ed(_calib(n, pairs),
                                     Om=[Omega] * n, delta=[delta] * n))


if not HAS_TENPY:
    C.skip('一般复耦合 TeNPy MPO DMRG', 'tenpy 不可用')
    sys.exit(0 if C.summary() else 1)

# 参数组：含非零 A（对产生）、复 J/A、复 V 各向异性类比
cases = [
    dict(length=4, V_r=0.8, J_r=0.3 + 0.2j, A_r=0.0, V_l=0.5, J_l=-0.4 + 0.1j, A_l=0.0,
         delta=0.6, Omega=1.1),
    dict(length=3, V_r=1.2, J_r=0.5j, A_r=0.3 - 0.2j, V_l=0.7, J_l=0.2, A_l=-0.25 + 0.15j,
         delta=-0.3, Omega=0.9),
    dict(length=5, V_r=0.6, J_r=0.0, A_r=0.4 + 0.3j, V_l=0.9, J_l=0.0, A_l=0.0,
         delta=0.4, Omega=1.3),
]
subs = []
for cs in cases:
    L = cs['length']
    Ht, n = dm.rydberg_ladder_full_hamiltonian(**cs)
    He = engine_ladder_H(L, cs['V_r'], cs['J_r'], cs['A_r'], cs['V_l'], cs['J_l'],
                         cs['A_l'], cs['delta'], cs['Omega'])
    st = np.sort(np.linalg.eigvalsh(Ht))
    se = np.sort(np.linalg.eigvalsh(He))
    E0 = float(st[0])
    E_dmrg = dm.dmrg_rydberg_ladder(L, cs['V_r'], cs['J_r'], cs['A_r'], cs['V_l'],
                                    cs['J_l'], cs['A_l'], cs['delta'], cs['Omega'])['E0']
    tag = 'L=%d J_r=%s A_r=%s A_l=%s' % (L, cs['J_r'], cs['A_r'], cs['A_l'])
    subs += [
        (f'{tag}: TeNPy MPO 谱 ≡ 引擎 H 谱 <1e-8', float(np.max(np.abs(st - se))) < 1e-8),
        (f'{tag}: DMRG ≡ 精确 E0 <1e-6', abs(E_dmrg - E0) < 1e-6),
    ]

C.group('① 一般复耦合(V/J/A+σ⁺σ⁺)喂进 TeNPy MPO：MPO≡引擎H 且 DMRG≡精确', subs)

sys.exit(0 if C.summary() else 1)
