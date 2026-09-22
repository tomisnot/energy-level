# -*- coding: utf-8 -*-
"""多体观测量（ED 全态可得）：逐原子布居、关联 C_ij、双部纠缠熵。

基序与 assemble.build_H_sparse 一致：产物态索引 s 的 bit i ↔ 原子 i
（MSB：bit(s,i)=(s>>(n−1−i))&1），bit=1 ↔ |r⟩（激发）、0 ↔ |g⟩。

所有函数都从**概率分布 p=|psi|²**（dim,）或**态矢 psi**（dim,）算，
不持有大数组，供 engine_ed 在每个输出时刻流式归约（大 dim 不物化 psi_all）。
"""
import numpy as np

__all__ = ['bit_matrix', 'popcount', 'atom_pops', 'corr_matrix',
           'entropy_at_cut', 'product_label']


def bit_matrix(n, dim):
    """(n, dim) int8：bits[i, s] = 原子 i 在产物态 s 的激发数（1=r / 0=g）。"""
    s = np.arange(dim, dtype=np.int64)
    return np.stack([((s >> (n - 1 - i)) & 1).astype(np.int8)
                     for i in range(n)]) if n else np.zeros((0, dim), np.int8)


def popcount(n, dim):
    """(dim,) int：每个产物态的总激发数（= bit_matrix 列和）。"""
    if n == 0:
        return np.zeros(dim, dtype=np.int64)
    return bit_matrix(n, dim).sum(axis=0).astype(np.int64)


def atom_pops(bits, p):
    """p=(dim,) 概率 → 逐原子激发 ⟨n_i⟩=(n,)。"""
    return bits @ p                       # (n,dim)@(dim,) → (n,)


def corr_matrix(bits, p):
    """关联 C_ij = ⟨n_i n_j⟩ − ⟨n_i⟩⟨n_j⟩；返回 (n, n)。

    ⟨n_i n_j⟩ = Σ_s bit_i(s)·bit_j(s)·p_s = (bits·p) @ bits.T。
    """
    ni = bits @ p
    ninj = (bits * p) @ bits.T            # (n,n)
    return ninj - np.outer(ni, ni)


def entropy_at_cut(psi, n, cut):
    """双部 von Neumann 熵（自然对数，单位 nat）：前 cut 个原子 vs 其余。

    把 psi 按 (2^cut, 2^(n−cut)) 重塑 → Schmidt 奇异值 → S = −Σ λ² ln λ²。

    **区分"真 0"与"参数错"**（2026-09-22，工单 P2-3）：原实现三种情形一律
    `return 0.0`，于是"算错了"会被伪装成"没纠缠"——
      · `n < 2` → **0.0**：没有双部划分，S 恒 0 是**物理正确**的（不是错）；
      · 态矢长度 ≠ 2^n → **`ValueError`**：调用方维度错；
      · `n ≥ 2` 但 `cut ∉ [1, n−1]` → **`ValueError`**：参数错。
    """
    n = int(n)
    if n < 2:
        return 0.0                       # 无双部划分：S=0 是物理事实
    dim = 1 << n
    v = np.asarray(psi).ravel()
    if v.size != dim:
        raise ValueError('entropy_at_cut：态矢长度 %d ≠ 2^%d=%d（调用方维度错）'
                         % (v.size, n, dim))
    cut = int(cut)
    if not (1 <= cut <= n - 1):
        raise ValueError('entropy_at_cut：cut=%d 越界（n=%d 时须 ∈[1,%d]）'
                         % (cut, n, n - 1))
    a = 1 << cut
    b = dim >> cut
    M = v.reshape(a, b)
    sv = np.linalg.svd(M, compute_uv=False)
    w = sv ** 2
    w = w[w > 1e-15]
    if w.size == 0:
        return 0.0
    return float(-np.sum(w * np.log(w)))


def product_label(n, s):
    """产物态索引 s → 紧凑标签 '|g r g …⟩'（原子 0 在左/MSB）。"""
    bits = ''.join('r' if ((s >> (n - 1 - i)) & 1) else 'g' for i in range(n))
    return '|%s>' % bits if n else '|>'
