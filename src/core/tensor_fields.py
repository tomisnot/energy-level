# -*- coding: utf-8 -*-
"""tensor_fields.py（t63 第 2 层）：两体场提升——单原子 A 矩阵 → N² 两体耦合。

两体 = 单原子核心 + tensor 提升：每个 A（N×N）提升为 N²×N² 的两体耦合
A⊗I + I⊗A（每原子各驱动一次）。复用 tensor.py 的直积工具语义。

- lift_single_A(A, N)：A (N,N) → 两体 (N²,N²) = A⊗I + I⊗A
  索引约定：k = i·N + j（i = A 原子态，j = B 原子态）
  A⊗I：|i,j⟩→|i',j⟩（A 原子跃迁 i→i'，B 不动）v = A[i,i']
  I⊗A：|i,j⟩→|i,j'⟩（B 原子跃迁 j→j'，A 不动）v = A[j,j']
- lift_fields(fields_A, fields_B, N)：两体场对——A 侧场全提升（作用于 A 原子）、
  B 侧场全提升（作用于 B 原子）——同频场自动合并（共享光子块语义）。
"""
import numpy as np


def lift_single_A(A, N):
    """单原子 A (N,N) → 两体 (N²,N²)：A⊗I + I⊗A（Hermitian 对称）。"""
    A = np.asarray(A, float)
    N2 = N * N
    M = np.zeros((N2, N2))
    for i in range(N):
        for ip in range(N):
            v = A[i, ip]
            if v == 0.0:
                continue
            for j in range(N):
                M[i * N + j, ip * N + j] += v      # A 原子 i→ip
                M[j * N + i, j * N + ip] += v      # B 原子 i→ip
    return M


def lift_fields(fields_A, fields_B, N):
    """两体场提升：返回 (lifted_opt, lifted_mw)——同频光场合并为共享光子块。

    fields_A/B：kind='exact' 场列表（omega_L_mhz + A + waveform）。
    lifted_opt：光场列表（同频合并——A = Σ 提升矩阵；envs 逐波保留）
    lifted_mw：微波列表（每场独立——当场 cos）
    """
    N2 = N * N

    def _one(t):
        return np.ones_like(np.atleast_1d(t), float)

    # 光场：同频合并（共享光子块——A 求和 + 每波 env 独立）
    opt_by_w = {}
    mw_list = []
    for side, fields in (('A', fields_A), ('B', fields_B)):
        if not fields:
            continue
        for f in fields:
            w = float(f['omega_L_mhz'])
            A = np.asarray(f['A'], float)
            env = f.get('waveform') or _one
            if w > 1e5:
                g = opt_by_w.setdefault(w, {'A': np.zeros((N2, N2)),
                                            'waves': []})
                g['A'] = g['A'] + lift_single_A(A, N)
                g['waves'].append(env)
            else:
                mw_list.append(dict(kind='exact', A=lift_single_A(A, N),
                                    omega_L_mhz=w, waveform=env))
    opt_list = [dict(kind='exact', A=g['A'], omega_L_mhz=w, waves=g['waves'])
                for w, g in sorted(opt_by_w.items())]
    return opt_list, mw_list
