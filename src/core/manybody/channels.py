# -*- coding: utf-8 -*-
"""多体 Hamiltonian 的**唯一通道语义**：对角项 + XOR 翻转通道。

同一套项语义被三处消费：

* ``assemble.build_H_sparse`` —— 装配 CSR；
* ``matrix_free.hamiltonian_linear_operator`` —— CPU 矩阵自由 matvec；
* ``matrix_free._channel_arrays`` / GPU —— 预先展开转移表。

改这里之前先看 ``checks/manybody_channel_selfcheck.py``。

约定（与全库一致）：bit 0 = 最高位对应原子 0（MSB），bit=1 ↔ |r⟩；
``H[t,s] = <t|H|s>``；J/A 的共轭方向见 ``channel_spec``。
"""
from __future__ import annotations

import numpy as np

from ..evolve_util import CONV
from .couplings import active_terms

__all__ = ['ChannelSpec', 'ChannelKernel', 'channel_spec', 'selector_mask']

#: 默认通道缓存预算（惰性；超预算就退回按需计算，绝不静默改变结果）。
DEFAULT_CACHE_BYTES = 128 * 1024 ** 2


class ChannelSpec:
    """一次调用/运行内不变的通道声明（纯数据，可复用）。"""

    __slots__ = ('n', 'delta', 'gamma', 'use', 'pairs', 'channels')

    def __init__(self, n, delta, gamma, use, pairs, channels):
        self.n = int(n)
        self.delta = delta
        self.gamma = gamma
        self.use = frozenset(use)
        self.pairs = pairs
        self.channels = tuple(channels)

    @property
    def dtype(self):
        """H 的数据类型由**数据**派生，不由调用方选择。

        gamma≠0 ⇒ 对角有虚部；任一通道幅度有虚部 ⇒ 非对角复。
        否则 H 是实对称的，CSR 值数组可以少一半内存。
        """
        if np.any(self.gamma):
            return np.dtype(np.complex128)
        for _selector, _mask, amp in self.channels:
            try:
                if complex(amp).imag != 0.0:
                    return np.dtype(np.complex128)
            except (TypeError, ValueError):
                return np.dtype(np.complex128)
        return np.dtype(np.float64)

    def diagonal(self, bits, dtype=None):
        """逐基矢对角（MHz）：δn + Vnn − iΓn/2（按演化约定缩放）。

        ``dtype`` 缺省用 :attr:`dtype`；实模型走 float64，复模型才分配复数。
        """
        dtype = np.dtype(self.dtype if dtype is None else dtype)
        if np.issubdtype(dtype, np.floating) and np.any(self.gamma):
            raise ValueError('实对角 dtype 不能承载非零 gamma（H 必须有虚部）')
        diag = np.zeros(len(bits[0]), dtype)
        if np.issubdtype(dtype, np.floating):
            for i in range(self.n):
                diag += self.delta[i] * bits[i]
        else:
            for i in range(self.n):
                diag += (self.delta[i]
                         - 0.5j * self.gamma[i] * 1e-3 / CONV) * bits[i]
        if 'I1' in self.use:
            for p in self.pairs:
                diag += p.get('V', 0.0) * (bits[int(p['i'])] * bits[int(p['j'])])
        return diag


def selector_mask(bits, selector):
    """selector=((site, bit), …) → 布尔掩码。"""
    mask = None
    for site, bit in selector:
        cur = (bits[int(site)] == int(bit))
        mask = cur if mask is None else (mask & cur)
    if mask is None:
        return np.ones(len(bits[0]), bool)
    return mask


def channel_spec(calib, Om=None, delta=None, gamma=None, terms=None):
    """calib + 逐原子驱动 → :class:`ChannelSpec`。

    ``Om``/``delta``/``gamma`` 是逐原子数组（缺省 0）；``terms`` 缺省时由
    ``active_terms(calib)`` 从 V/J/A 数值推导。J/A 的共轭方向与
    ``assemble.build_H_sparse`` 的既有约定逐项一致：

    * J：|r_i g_j⟩ → |g_i r_j⟩ 取 conj(J)，反向取 J；
    * A：|g_i g_j⟩ → |r_i r_j⟩ 取 A，反向取 conj(A)；
    * Ω：|g⟩ → |r⟩ 取 Ω/2，反向取 conj(Ω)/2。
    """
    atoms = calib.get('atoms') or []
    n = len(atoms)
    if n == 0:
        raise ValueError('通道声明需要至少一个原子')
    Om = np.zeros(n, complex) if Om is None else np.asarray(Om, complex)
    delta = np.zeros(n) if delta is None else np.asarray(delta, float)
    gamma = np.zeros(n) if gamma is None else np.asarray(gamma, float)
    if any(x.shape != (n,) or not np.isfinite(x).all() for x in (Om, delta, gamma)):
        raise ValueError('Om/delta/gamma 须为有限、长度为 N 的数组')
    use = set(active_terms(calib) if terms is None else terms)
    pairs = list(calib.get('pairs') or [])
    channels = []
    for i, om in enumerate(Om):
        if om == 0:
            continue
        channels.append((((i, 0),), np.int64(1) << (n - 1 - i), om / 2.0))
        channels.append((((i, 1),), np.int64(1) << (n - 1 - i), np.conj(om) / 2.0))
    for p in pairs:
        i, j = int(p['i']), int(p['j'])
        mask = (np.int64(1) << (n - 1 - i)) ^ (np.int64(1) << (n - 1 - j))
        if 'I2' in use:
            J = p.get('J', 0.0)
            if J != 0:
                channels.append((((i, 1), (j, 0)), mask, np.conj(J)))
                channels.append((((i, 0), (j, 1)), mask, J))
        if 'I3' in use:
            A = p.get('A', 0.0)
            if A != 0:
                channels.append((((i, 0), (j, 0)), mask, A))
                channels.append((((i, 1), (j, 1)), mask, np.conj(A)))
    return ChannelSpec(n, delta, gamma, use, pairs, channels)


class ChannelKernel:
    """给定基（全空间或任意编码表）上的 matvec 内核。

    * 全空间：目标索引 = ``code ^ mask``，不需要 searchsorted；
    * 受限基：``pos`` 查表（n ≤ 25）或 searchsorted 回退；
    * 索引表按总预算**全有或全无**地缓存：预算不够就每次按需算，
      不会出现“缓存了一半、结果却一样”的隐性行为差。
    """

    def __init__(self, spec, basis=None, budget_bytes=DEFAULT_CACHE_BYTES):
        self.spec = spec
        self.n = spec.n
        if basis is None:
            self.basis = np.arange(1 << self.n, dtype=np.int64)
            self.full = True
        else:
            self.basis = np.asarray(basis, np.int64).reshape(-1)
            if self.basis.size == 0 or np.any(self.basis < 0) \
                    or np.any(self.basis >= (1 << self.n)):
                raise ValueError('basis 必须是非空有效位编码')
            self.full = bool(self.basis.size == (1 << self.n)
                             and np.array_equal(self.basis,
                                                np.arange(1 << self.n)))
            if np.unique(self.basis).size != self.basis.size:
                raise ValueError('basis 不能重复')
        self.dim = len(self.basis)
        self.bits = [((self.basis >> (self.n - 1 - i)) & 1).astype(np.int8)
                     for i in range(self.n)]
        self.diag = spec.diagonal(self.bits)
        self.pos = None
        if not self.full:
            if self.n <= 25:
                self.pos = np.full(1 << self.n, -1, np.int32)
                self.pos[self.basis] = np.arange(self.dim, dtype=np.int32)
            else:
                self.order = np.argsort(self.basis)
                self.sorted_basis = self.basis[self.order]
        self._cache = {}
        self._budget = int(budget_bytes)
        self._bytes = 0
        self._cache_enabled = None

    def _indices(self, ch):
        selector, mask, _amp = ch
        cols = np.flatnonzero(selector_mask(self.bits, selector)).astype(np.int32)
        targets = self.basis[cols] ^ mask
        if self.full:
            loc = targets.astype(np.int32)
            valid = np.ones(cols.size, bool)
        elif self.pos is not None:
            loc = self.pos[targets]
            valid = loc >= 0
        else:
            loc = np.searchsorted(self.sorted_basis, targets)
            valid = loc < self.dim
            valid[valid] &= self.sorted_basis[loc[valid]] == targets[valid]
            loc = self.order[loc]
        return cols[valid], loc[valid].astype(np.int64)

    def _data(self, idx, ch):
        hit = self._cache.get(idx)
        if hit is not None:
            return hit
        data = self._indices(ch)
        if self._cache_enabled is None:
            # 估算“全缓存”是否放得下；放不下就一颗都不缓存，行为可预测。
            per = max(1, self.dim) * 12
            self._cache_enabled = per * max(1, len(self.spec.channels)) <= self._budget
        if self._cache_enabled:
            need = data[0].nbytes + data[1].nbytes
            if self._bytes + need <= self._budget:
                self._cache[idx] = data
                self._bytes += need
        return data

    def matvec(self, vector):
        v = np.asarray(vector, complex).reshape(-1)
        if v.size != self.dim:
            raise ValueError('通道内核：向量长度 %d ≠ 基维度 %d'
                             % (v.size, self.dim))
        y = self.diag * v
        for idx, ch in enumerate(self.spec.channels):
            cols, loc = self._data(idx, ch)
            if cols.size:
                # XOR 在单通道内是单射：同一通道内 loc 不重复，可直接散加。
                y[loc] += np.complex128(ch[2]) * v[cols]
        return y
