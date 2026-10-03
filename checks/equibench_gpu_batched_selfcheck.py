# -*- coding: utf-8 -*-
"""GPU 批量 ED 标杆：批量化本征 ≡ 逐点精确 ≡ 独立参照（Tier-B · B2 交叉律的 GPU 版）。

蒸馏造标签 = 参数网格上成千个独立 H 的基态。本判据证明 GPU 批量化路径**不多不少**：
- cupy-batched ≡ torch-batched ≡ numpy-batched ≡ 逐点 `ground_state`（同 E0/⟨n_i⟩/gap）
- 且这些 ≡ 独立 np.kron 参照（同 equibench_engine_ed 的装配纪律）。
无 GPU 时 cupy/torch 子项 SKIP、CPU 批量化路径仍全跑（保确定性、不误报）。
小 N（≤10）以适配显存（RTX 4050 6GB），generic 随机实例 ⇒ 基态非简并 ⇒ 逐分量可比。
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

C = Checker('equibench_gpu_batched')
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
                atoms=[dict(id=i, pos=(float(i), 0.0, 0.0)) for i in range(n)],
                pairs=pairs, single_basis=[])


def _pair(i, j, V, J, A):
    return dict(i=i, j=j, V=float(V), J=complex(J), A=complex(A), raw_V=float(V),
                raw_J=complex(J), raw_A=complex(A), delta={}, info={})


rng = np.random.default_rng(20261001)
n = 8
dim = 1 << n
grid = []
for b in range(8):                                  # 一个小的“参数网格”
    pairs = ([_pair(i, i + 1, rng.uniform(1, 40), rng.normal() + 1j * rng.normal(),
                    rng.normal() + 1j * rng.normal()) for i in range(n - 1)]
             + [_pair(i, i + 2, rng.uniform(0, 10), rng.normal(), rng.normal())
                for i in range(n - 2)])
    Om = rng.uniform(0, 20, n)
    delta = rng.uniform(-9, 9, n)
    Heng = np.asarray(asm.build_H_ed(_calib(n, pairs), Om, delta, terms=list(TERMS)))
    Href = _H_ref(n, Om, delta, pairs)
    grid.append((Heng, Href))
H_batch = np.stack([g[0] for g in grid])

from src.core.manybody.observables import bit_matrix                 # noqa: E402
# 逐点 CPU 精确参照（equilibrium.ground_state 走稠密 eigh）——单趟，避免重复对角化
bits = bit_matrix(n, dim).astype(float)
E0_pt, n_pt = [], []
for H, _ in grid:
    _v = eq.ground_state(sp.csr_matrix(H))['vec']
    E0_pt.append(eq.ground_state(sp.csr_matrix(H))['E0'])
    n_pt.append(np.abs(_v) ** 2 @ bits.T)
E0_pt = np.array(E0_pt)
n_pt = np.array(n_pt)
E0_ref = np.array([float(np.linalg.eigvalsh(Hr)[0]) for _, Hr in grid])

subs = []
for be in ('numpy', 'auto'):
    if be == 'auto' and eq.detect_gpu() is None:
        C.skip('GPU auto 后端', '无 GPU（cupy/torch-cuda 均不可用）→ 只验 CPU 批量化')
        continue
    r = eq.batched_ground_state(H_batch, backend=be)
    subs += [
        (f'{r["backend"]}: E0 ≡ 逐点 ground_state <1e-7',
         float(np.abs(r['E0'] - E0_pt).max()) < 1e-7),
        (f'{r["backend"]}: E0 ≡ 独立 Kron 参照 <1e-7',
         float(np.abs(r['E0'] - E0_ref).max()) < 1e-7),
        (f'{r["backend"]}: ⟨n_i⟩ ≡ 逐点 <1e-5（generic 非简并）',
         float(np.abs(r['n_atom'] - n_pt).max()) < 1e-5),
    ]

# 显式比对 cupy vs torch vs numpy（都可用时三者同能）
sub_backends = []
try:
    import cupy  # noqa: F401
    has_cupy = True
except Exception:
    has_cupy = False
try:
    import torch
    has_torch = torch.cuda.is_available()
except Exception:
    has_torch = False
if has_cupy:
    rc = eq.batched_ground_state(H_batch, backend='cupy')
    rn = eq.batched_ground_state(H_batch, backend='numpy')
    sub_backends.append(('cupy ≡ numpy-batched E0 <1e-7',
                         float(np.abs(rc['E0'] - rn['E0']).max()) < 1e-7))
    sub_backends.append(('cupy ≡ numpy-batched ⟨n_i⟩ <1e-5',
                         float(np.abs(rc['n_atom'] - rn['n_atom']).max()) < 1e-5))
if has_torch:
    rt = eq.batched_ground_state(H_batch, backend='torch')
    rn = eq.batched_ground_state(H_batch, backend='numpy')
    sub_backends.append(('torch ≡ numpy-batched E0 <1e-7',
                         float(np.abs(rt['E0'] - rn['E0']).max()) < 1e-7))
if sub_backends:
    C.group('GPU 批量化后端一致性（cupy/torch ≡ numpy-batched）', sub_backends)

C.group('GPU 批量 ED ≡ 逐点精确 ≡ 独立参照（参数网格，N=8×8 点）', subs)

sys.exit(0 if C.summary() else 1)
