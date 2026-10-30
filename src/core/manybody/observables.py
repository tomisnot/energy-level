# -*- coding: utf-8 -*-
"""多体观测量（ED 全态可得）：逐原子布居、关联 C_ij、双部纠缠熵。

基序与 assemble.build_H_sparse 一致：产物态索引 s 的 bit i ↔ 原子 i
（MSB：bit(s,i)=(s>>(n−1−i))&1），bit=1 ↔ |r⟩（激发）、0 ↔ |g⟩。

所有函数都从**概率分布 p=|psi|²**（dim,）或**态矢 psi**（dim,）算，
不持有大数组，供 engine_ed 在每个输出时刻流式归约（大 dim 不物化 psi_all）。
"""
import numpy as np
from .basis import basis_bits

__all__ = ['bit_matrix', 'popcount', 'atom_pops', 'corr_matrix',
           'entropy_at_cut', 'product_label', 'spin_observables',
           'spin_observables_basis', 'loop_observables',
           'link_loop_observables']


def bit_matrix(n, dim):
    """(n, dim) int8：bits[i, s] = 原子 i 在产物态 s 的激发数（1=r / 0=g）。"""
    return basis_bits(n, np.arange(dim, dtype=np.int64))



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


def entropy_in_basis(psi, states, n, cut, max_entries=1 << 20):
    """Physical bipartite entropy using occupied Schmidt row/column identities.

    Returns None when the requested dense SVD exceeds the explicit scratch
    budget; this never turns an uncomputed quantity into physical zero.
    """
    if n < 2:
        return 0.
    if not 1 <= cut < n:
        raise ValueError('entropy_cut 须位于 [1,N-1]')
    states = np.asarray(states, np.int64)
    left, right = states >> (n-cut), states & ((1 << (n-cut))-1)
    _, li = np.unique(left, return_inverse=True)
    _, ri = np.unique(right, return_inverse=True)
    shape = (int(li.max())+1, int(ri.max())+1)
    if shape[0]*shape[1] > max_entries:
        return None
    matrix = np.zeros(shape, complex)
    matrix[li,ri] = psi
    weights = np.linalg.svd(matrix, compute_uv=False)**2
    norm = weights.sum()
    if norm <= 0:
        return None
    weights /= norm
    weights = weights[weights > 0]
    return float(-np.sum(weights*np.log(weights)))


def spin_observables(psi, n):
    """Return local Pauli/ladder expectations and ``<sigma+_i sigma-_j>``.

    The basis is ``(|g>, |r>)`` with ``sigma+ = |r><g|`` and
    ``sigma- = |g><r|``.  Consequently ``phase[i] = arg(<sigma-_i>)`` is
    the phase of the local excited-state amplitude for a product state.
    """
    n = int(n)
    dim = 1 << n
    v = np.asarray(psi, complex).reshape(-1)
    if v.size != dim:
        raise ValueError('spin_observables：态矢长度 %d ≠ 2^%d=%d'
                         % (v.size, n, dim))
    sx = np.zeros(n, float); sy = np.zeros(n, float)
    sz = np.zeros(n, float); sp = np.zeros(n, complex); sm = np.zeros(n, complex)
    s = np.arange(dim, dtype=np.int64)
    for i in range(n):
        mask = np.int64(1) << np.int64(n - 1 - i)
        g = (s & mask) == 0
        r = s[g] | mask
        # <P>=<sigma+>, <M>=<sigma-> in the |g>,|r> basis.
        sp[i] = np.vdot(v[r], v[g])
        sm[i] = np.vdot(v[g], v[r])
        sz[i] = float(np.sum(np.abs(v[g]) ** 2) - np.sum(np.abs(v[r]) ** 2))
        sx[i] = float(2.0 * np.real(sm[i]))
        sy[i] = float(2.0 * np.imag(sm[i]))
    xy = np.zeros((n, n), complex)
            # P_i M_j maps |g_i r_j> -> |r_i g_j>; diagonal i=j is <P M>=<n_r>.
    for i in range(n):
        mi = np.int64(1) << np.int64(n - 1 - i)
        for j in range(n):
            mj = np.int64(1) << np.int64(n - 1 - j)
            if i == j:
                xy[i, j] = np.vdot(v, ((s & mi) != 0) * v)
                continue
            src = s[((s & mi) == 0) & ((s & mj) != 0)]
            dst = src | mi
            dst = dst ^ mj
            xy[i, j] = np.vdot(v[dst], v[src])
    phase = np.angle(sm)
    phase[np.abs(sm) < 1e-14] = np.nan
    return dict(sx=sx, sy=sy, sz=sz, sp=sp, sm=sm,
                xy_corr=xy, phase=phase)


def spin_observables_basis(psi, states, n):
    """Same observables as :func:`spin_observables` in a projected basis."""
    q = np.asarray(psi, complex).reshape(-1)
    states = np.asarray(states, dtype=np.int64)
    if q.size != states.size:
        raise ValueError('spin_observables_basis：态矢与 basis 长度不一致')
    pos = {int(s): k for k, s in enumerate(states)}
    n = int(n)
    sx = np.zeros(n, float); sy = np.zeros(n, float)
    sz = np.zeros(n, float); sp = np.zeros(n, complex); sm = np.zeros(n, complex)
    xy = np.zeros((n, n), complex)
    for i in range(n):
        mi = 1 << (n - 1 - i)
        for k, s in enumerate(states):
            if int(s) & mi:
                sz[i] -= abs(q[k]) ** 2
                g = int(s) ^ mi
                kg = pos.get(g)
                if kg is not None:
                    sm[i] += np.conj(q[kg]) * q[k]
                    sp[i] += np.conj(q[k]) * q[kg]
            else:
                sz[i] += abs(q[k]) ** 2
    sx[:] = 2 * np.real(sm); sy[:] = 2 * np.imag(sm)
    for i in range(n):
        mi = 1 << (n - 1 - i)
        for j in range(n):
            mj = 1 << (n - 1 - j)
            if i == j:
                xy[i, j] = sum(abs(q[k]) ** 2 for k, s in enumerate(states)
                                if int(s) & mi)
                continue
            for k, s in enumerate(states):
                s = int(s)
                if (s & mi) == 0 and (s & mj):
                    dst = (s | mi) ^ mj
                    kd = pos.get(dst)
                    if kd is not None:
                        xy[i, j] += np.conj(q[kd]) * q[k]
    phase = np.angle(sm); phase[np.abs(sm) < 1e-14] = np.nan
    return dict(sx=sx, sy=sy, sz=sz, sp=sp, sm=sm,
                xy_corr=xy, phase=phase)


def loop_observables(xy_corr, loops):
    """Compute gauge-invariant bond phase sums for user-specified closed loops.

    For an edge ``i -> j`` we use ``arg(<sigma+_i sigma-_j>)``.  A loop is
    supplied as atom indices without repeating its first index; the final edge
    back to the first index is added automatically.  The returned winding is
    the wrapped phase sum divided by ``2*pi``.
    """
    if loops is None:
        return [], []
    xy = np.asarray(xy_corr, complex)
    phases = []; wind = []
    for loop in loops:
        q = [int(x) for x in loop]
        if len(q) < 2 or any(x < 0 or x >= xy.shape[0] for x in q):
            raise ValueError('loop 必须包含至少两个合法原子下标')
        vals = []
        for a, b in zip(q, q[1:] + q[:1]):
            z = xy[a, b]
            vals.append(float(np.angle(z)) if abs(z) > 1e-14 else np.nan)
        total = float(np.sum(vals)) if np.all(np.isfinite(vals)) else np.nan
        phases.append(total)
        wind.append(float(total / (2.0 * np.pi)) if np.isfinite(total) else np.nan)
    return phases, wind


def link_loop_observables(link_corr, link_pairs, loops):
    """Evaluate loop phase/winding for a time series of selected links.

    ``link_corr[t, q]`` is ``<sigma+_a sigma-_b>`` for
    ``link_pairs[q] == (a, b)``.  A loop may use either orientation; the
    reverse orientation is supplied by the complex conjugate.  Returning a
    time series here keeps loop/winding diagnostics independent of the ED or
    MPS backend.  Missing edges produce NaN instead of silently changing the
    requested loop.
    """
    if loops is None:
        return np.empty((0, 0), float), np.empty((0, 0), float)
    pairs = [(int(a), int(b)) for a, b in (link_pairs or [])]
    lookup = {}
    for q, (a, b) in enumerate(pairs):
        lookup.setdefault((a, b), []).append((q, 1))
        lookup.setdefault((b, a), []).append((q, -1))
    arr = np.asarray(link_corr, complex)
    if arr.ndim == 1:
        arr = arr[None, :]
    phase = np.full((arr.shape[0], len(loops)), np.nan, float)
    winding = np.full_like(phase, np.nan)
    for ell, raw in enumerate(loops):
        loop = [int(x) for x in raw]
        if len(loop) < 2:
            raise ValueError('loop 必须包含至少两个合法原子下标')
        cols = []
        for a, b in zip(loop, loop[1:] + loop[:1]):
            choices = lookup.get((a, b), [])
            if not choices:
                cols = None
                break
            cols.append(choices[0])
        if cols is None:
            continue
        vals = np.zeros((arr.shape[0], len(cols)), float)
        valid = np.ones(arr.shape[0], bool)
        for j, (q, sign) in enumerate(cols):
            z = arr[:, q]
            z = z if sign > 0 else np.conj(z)
            ok = np.abs(z) > 1e-14
            valid &= ok
            vals[:, j] = np.angle(z)
        total = np.sum(vals, axis=1)
        phase[valid, ell] = total[valid]
        winding[valid, ell] = total[valid] / (2.0 * np.pi)
    return phase, winding
