# -*- coding: utf-8 -*-
"""平衡态**求解器**交叉一致判据（Tier-B · B2 交叉律）。

被测：`equilibrium.ground_state`（稠密 eigh / Lanczos）与 `ground_state_imagtime`
（虚时幂法）。对同一批随机**厄米**广义里德堡 H（V·nn + J·σ⁺σ⁻ + A·σ⁺σ⁺ + δn + Ω/2·σˣ），
要求四种独立路径给出**同一基态能量**：

    稠密 eigh  ≡  Lanczos(eigsh)  ≡  虚时幂法  ≡  独立 np.kron 构造的稠密 eigh

任一越界即定位到具体求解器/装配。虚时给迭代精度（1e-5），其余机器精度。
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

C = Checker('equibench_gs_solvers')
I2 = np.eye(2, dtype=complex)
_N = np.array([[0, 0], [0, 1]], complex)
_SX = np.array([[0, 1], [1, 0]], complex)
_SP = np.array([[0, 0], [1, 0]], complex)
_SM = _SP.conj().T
TERMS = ('I1', 'I2', 'I3')


def _embed(n, ops):
    out = np.array([[1.0 + 0j]])
    for i in range(n):
        out = np.kron(out, ops.get(i, I2))
    return out


def _H_ref(n, Om, delta, pairs):
    dim = 1 << n
    H = np.zeros((dim, dim), complex)
    for i in range(n):
        H += delta[i] * _embed(n, {i: _N}) + (Om[i] / 2) * _embed(n, {i: _SX})
    for p in pairs:
        i, j = p['i'], p['j']
        H += p['V'] * _embed(n, {i: _N, j: _N})
        H += p['J'] * _embed(n, {i: _SP, j: _SM}) + np.conj(p['J']) * _embed(n, {i: _SM, j: _SP})
        H += p['A'] * _embed(n, {i: _SP, j: _SP}) + np.conj(p['A']) * _embed(n, {i: _SM, j: _SM})
    return H


def _calib(n, pairs):
    return dict(terms=list(TERMS),
                atoms=[dict(id=i, pos=(float(i % 4), float(i // 4), 0.0)) for i in range(n)],
                pairs=pairs, single_basis=[])


def _pair(i, j, V, J, A):
    return dict(i=i, j=j, V=float(V), J=complex(J), A=complex(A), raw_V=float(V),
                raw_J=complex(J), raw_A=complex(A), delta={}, info={})


rng = np.random.default_rng(20261001)
subs = []
for n, nb in [(5, 6), (6, 7), (7, 9)]:
    pairs = []
    for i in range(n - 1):
        pairs.append(_pair(i, i + 1, rng.uniform(0, 40),
                           rng.normal() + 1j * rng.normal(), rng.normal() + 1j * rng.normal()))
    for i in range(n - 2):
        pairs.append(_pair(i, i + 2, rng.uniform(0, 12), rng.normal(), rng.normal()))
    Om = rng.uniform(0, 15, n)
    delta = rng.uniform(-8, 8, n)
    Hd = np.asarray(asm.build_H_ed(_calib(n, pairs), Om, delta, terms=list(TERMS)))
    E0_ref = float(np.linalg.eigvalsh(_H_ref(n, Om, delta, pairs))[0])
    E0_auto = eq.ground_state(sp.csr_matrix(Hd))['E0']            # 小维→稠密 eigh
    E0_lanczos = eq.ground_state(sp.csr_matrix(Hd), dense_max=0)['E0']   # 强制 Lanczos
    it = eq.ground_state_imagtime(sp.csr_matrix(Hd), steps=6000)
    tag = f'N={n}'
    subs += [
        (f'{tag}: 引擎稠密eigh ≡ 独立Kron参照 <1e-8',
         abs(float(np.linalg.eigvalsh(Hd)[0]) - E0_ref) < 1e-8),
        (f'{tag}: Lanczos ≡ 稠密eigh <1e-8', abs(E0_lanczos - E0_auto) < 1e-8),
        (f'{tag}: Lanczos ≡ 独立参照 <1e-8', abs(E0_lanczos - E0_ref) < 1e-8),
        (f'{tag}: 虚时幂法 ≡ 稠密eigh <1e-5', abs(it['E0'] - E0_auto) < 1e-5),
        (f'{tag}: 虚时幂法收敛标志=True', bool(it['converged'])),
    ]

C.group('求解器四路径交叉一致（稠密eigh/Lanczos/虚时/独立Kron，随机含复厄米实例）',
        subs, detail='抓任一求解器或装配的相对漂移')

sys.exit(0 if C.summary() else 1)
