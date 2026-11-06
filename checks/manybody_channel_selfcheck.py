# -*- coding: utf-8 -*-
"""通道语义唯一性：稀疏装配、矩阵自由内核、受限基三者同源。

随机非对称复模型（V 实、J/A 复、Ω/δ/Γ 非零），检查：
  · ``build_H_sparse`` 与稠密独立装配的矩阵元一致（已有 ED 判据覆盖，这里做通道侧）；
  · ``hamiltonian_linear_operator`` 的 matvec 与 ``H @ v`` 逐位一致（全空间/受限基）；
  · 受限基的通道表只保留基内目标（越界分量按 0，不静默改基）。
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from scipy import sparse as sp

from checks.checklib import Checker
from checks.mb_benchlib import random_calib, random_state, dense_H
from src.core.manybody.assemble import (calibrate_pair,
                                         decompose_pair_channels)
from src.core.manybody.matrix_free import hamiltonian_linear_operator

C = Checker('manybody_channel')


def main():
    n = 3
    cal = random_calib(n, seed=11)
    # 保证 I3 通道非零（随机 A 可能恰好很小，仍会被算子包含；这里显式给一个）。
    cal['pairs'][0]['A'] = 0.05 - 0.01j
    Om = np.array([0.11 + 0.02j, -0.07j, 0.03])
    de = np.array([0.04, -0.02, 0.01])
    ga = np.array([0.01, 0.0, 0.02])
    use = ['I1', 'I2', 'I3']
    H = dense_H(cal, Om, de, use, ga)
    op = hamiltonian_linear_operator(cal, Om, de, ga, terms=use)
    v = random_state(1 << n, seed=3)
    C.group('全空间 matvec = 稀疏 H @ v', [
        ('shape', (op.shape == H.shape)),
        ('max|Δ| < 1e-12（实测 %.2e）' % np.max(np.abs(op @ v - H @ v)),
         np.max(np.abs(op @ v - H @ v)) < 1e-12),
        ('非厄米 gamma 进入对角', not np.allclose(H, H.conj().T)),
    ])

    idx = np.array([1, 2, 4, 7], np.int64)
    Hs = H[np.ix_(idx, idx)]
    ops = hamiltonian_linear_operator(cal, Om, de, ga, terms=use,
                                      basis_indices=idx)
    C.group('受限基 matvec = H 子块', [
        ('shape', (ops.shape == Hs.shape)),
        ('max|Δ| < 1e-12（实测 %.2e）' % np.max(np.abs(ops @ v[idx] - Hs @ v[idx])),
         np.max(np.abs(ops @ v[idx] - Hs @ v[idx])) < 1e-12),
    ])

    # 基序无关：打乱 basis 后，作用结果应按同一置换回到全空间。
    perm = np.array([7, 1, 4, 2], np.int64)
    Hp = H[np.ix_(perm, perm)]
    opp = hamiltonian_linear_operator(cal, Om, de, ga, terms=use,
                                      basis_indices=perm)
    vp = v[perm]
    C.group('任意基序（不要求排序）', [
        ('max|Δ| < 1e-12（实测 %.2e）' % np.max(np.abs(opp @ vp - Hp @ vp)),
         np.max(np.abs(opp @ vp - Hp @ vp)) < 1e-12),
    ])
    try:
        hamiltonian_linear_operator(cal, Om, de, ga, terms=use,
                                    basis_indices=np.array([0, 0, 1]))
        dup = False
    except ValueError:
        dup = True
    C.check('重复基编码被拒绝', dup)

    # 球基通道分解必须无损重构一阶 4×4 块（原 channel_api 判据并入本文件）。
    single = [dict(n=1, l=0, j=0.5, kind='mJ', mj=0.5, E_mhz=0.0),
              dict(n=1, l=1, j=0.5, kind='mJ', mj=1.5, E_mhz=10.0)]
    D = {1: sp.csr_matrix(np.array([[0, 0], [1.0 + 0.2j, 0]], complex)),
         0: sp.csr_matrix((2, 2), dtype=complex),
         -1: sp.csr_matrix(np.array([[0, 0], [0.3 - 0.1j, 0]], complex))}
    E = np.array([0.0, 10.0])
    pair4 = ((0, 0), (1, 0), (0, 1), (1, 1))
    out = decompose_pair_channels(single, D, pair4, 7.0, 47.0, 23.0)
    ref = calibrate_pair(None, single, D, E, pair4, 7.0, 47.0, 23.0,
                         second_order=False)['delta1']
    err = float(np.max(np.abs(out['total'] - ref)))
    C.group('球基通道分解无损重建', [
        ('max|Δ| < 1e-12（实测 %.2e）' % err, err < 1e-12),
        ('至少一个通道', len(out['channels']) > 0),
        ('通道字段齐', all({'q1', 'q2', 'tensor', 'matrix'} <= set(c)
                           for c in out['channels'])),
    ])
    return 0 if C.summary() else 1


if __name__ == '__main__':
    sys.exit(main())
