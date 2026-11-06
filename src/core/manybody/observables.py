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
           'link_loop_observables', 'normalize_pauli_strings',
           'pauli_specs_to_mps_order', 'pauli_strings_mps_batch',
           'PauliPlan', 'measure_pauli_strings', 'pauli_strings_factorized',
           'PAULI_LOCAL']


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

    边界语义（区分"真 0"与"参数错"）：
      · `n < 2` → **0.0**：没有双部划分，S 恒 0 是**物理正确**的；
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


# =====================================================================
# 任意 Pauli 字符串：通用高阶观测原语
# =====================================================================
#
# 目的：让"任意站点 × 任意单格点 Pauli 轴的乘积"成为一个**可请求的观测量**，
# 而不是把某个具体物理量（手性、环乘积、Wilson 线……）写进引擎。引擎只算
# ⟨σ^{a1}_{i1} σ^{a2}_{i2} … ⟩，物理含义（怎么组合、要不要乘 1/8、要不要连通减法）
# 一律留在调用方——与 `manybody.source`（"引擎不解释算符的物理含义"）同一纪律。
#
# 单格点矩阵一律用**标准 Pauli**，基为 (|g⟩,|r⟩)=(index 0,index 1)：
#     σˣ=[[0,1],[1,0]]  σʸ=[[0,-i],[i,0]]  σᶻ=[[1,0],[0,-1]]
# ⚠ 本仓的 `sp/sm` 是 ⟨|r⟩⟨g|⟩ / ⟨|g⟩⟨r|⟩（Rydberg 意义的"激发/退激发"），
#   即 σ⁺_本仓 = (σˣ − iσʸ)/2、σ⁻_本仓 = (σˣ + iσʸ)/2，与"按 σᶻ 升降"的教科书
#   编号相反。要用 sp/sm 组合时按这个关系换，别按名字猜。
PAULI_LOCAL = {
    'x': np.array([[0, 1], [1, 0]], dtype=complex),
    'y': np.array([[0, -1j], [1j, 0]], dtype=complex),
    'z': np.array([[1, 0], [0, -1]], dtype=complex),
}

_AXIS_ALIASES = {'x': 'x', 'y': 'y', 'z': 'z',
                 'sx': 'x', 'sy': 'y', 'sz': 'z',
                 'σx': 'x', 'σy': 'y', 'σz': 'z',
                 'sigmax': 'x', 'sigmay': 'y', 'sigmaz': 'z'}


def _axis_name(raw):
    key = str(raw).strip().lower()
    axis = _AXIS_ALIASES.get(key)
    if axis is None:
        raise ValueError('Pauli 字符串的轴须为 x/y/z（可写 σx 等），得到 %r' % (raw,))
    return axis


def normalize_pauli_strings(spec, n_atoms=None, err=None):
    """``manybody.pauli_strings`` → ``[{'label', 'terms'}]``，**当场校验**。

    接受两种写法（可混用）：::

        [[[0, "x"], [1, "y"], [2, "z"]], ...]                 # 纯列表
        [{"label": "chi_A", "terms": [[0,"x"],[1,"y"],[2,"z"]]}]

    规则（任一违反都报错，绝不静默改写）：
      * 一条字符串内**不允许同一站点出现两次**（那不是 Pauli 字符串）；
      * 站点须为 ``[0, n_atoms)`` 的整数；给了 ``n_atoms`` 才检查上界；
      * 空字符串（没有 terms）非法——空积恒为 1，没有观测意义；
      * 未给 ``label`` 时自动生成可读名，如 ``x@0 y@1 z@2``。

    返回的 ``terms`` 保留调用方给的站点顺序（求值只依赖集合，不依赖顺序）。
    """
    if spec is None:
        return None
    err = err or ValueError
    if not isinstance(spec, (list, tuple)):
        raise err('manybody.pauli_strings 须是 list（得到 %s）' % type(spec).__name__)
    if not spec:
        return []
    n = None if n_atoms is None else int(n_atoms)
    out = []
    for k, item in enumerate(spec):
        tag = 'pauli_strings[%d]' % k
        if isinstance(item, dict):
            terms_raw = item.get('terms', item.get('sites'))
            label = item.get('label')
        else:
            terms_raw = item
            label = None
        if terms_raw is None:
            raise err('%s 缺 terms（形如 [[0,"x"],[1,"y"]]）' % tag)
        if not isinstance(terms_raw, (list, tuple)) or not terms_raw:
            raise err('%s.terms 须是非空序列' % tag)
        terms, seen = [], set()
        for t in terms_raw:
            if not isinstance(t, (list, tuple)) or len(t) != 2:
                raise err('%s.terms 的每一项须为 [site, axis]，得到 %r' % (tag, t))
            site = t[0]
            if isinstance(site, bool) or not isinstance(site, (int, np.integer)):
                raise err('%s.terms 的站点须为整数，得到 %r' % (tag, site))
            site = int(site)
            if n is not None and not (0 <= site < n):
                raise err('%s.terms 的站点 %d 越界（原子数 %d）' % (tag, site, n))
            if site in seen:
                raise err('%s 在同一站点 %d 上放了两个算符；'
                          'Pauli 字符串要求站点互不相同' % (tag, site))
            seen.add(site)
            terms.append((site, _axis_name(t[1])))
        if label is None:
            label = ' '.join('%s@%d' % (ax, s) for s, ax in terms)
        out.append({'label': str(label), 'terms': terms})
    return out


# MPS 侧的算符名 → 系数。**必须用簇模型自己注册的算符**（`rydberg_cluster_model`
# 加的是 SZ/P/M，见 dmrg.py），不能用 TeNPy 自带的 Sigmaz/Sp/Sm：
#   · TeNPy 在 conserve='Sz' 下会把局部基**按电荷重排**（index(up)=1），
#     于是 'Sigmaz' 与 EL 的 diag(+1,−1) 差一个符号、'Sp'/'Sm' 互换；
#   · 模型注册的 SZ/P/M 矩阵是**固定**的（与 conserve 无关），与 calib 的
#     |g⟩=index0、|r⟩=index1 严格同序。
# 换算：σˣ = P+M；σʸ = i·P − i·M；σᶻ = SZ。
def _mps_axis_terms(site, axis):
    """单轴 → [(系数, 算符名), …]；优先用模型自注册的算符，缺就明确报错。"""
    names = set(site.opnames)
    if axis == 'z':
        if 'SZ' in names:
            return ((1.0, 'SZ'),)
        if 'Sz' in names:                      # 退化路径：2Sz 与 diag(+1,−1) 的符号
            sign = -1.0 if site.state_index('up') == 1 else 1.0
            return ((2.0 * sign, 'Sz'),)
        raise ValueError('MPS 上的 Pauli 字符串缺少 σᶻ 算符（站点既无 SZ 也无 Sz）')
    if 'SX' in names and axis == 'x':
        return ((1.0, 'SX'),)
    if 'SY' in names and axis == 'y':
        return ((1.0, 'SY'),)
    if 'P' in names and 'M' in names:
        return (((1.0, 'P'), (1.0, 'M')) if axis == 'x'
                else ((1j, 'P'), (-1j, 'M')))
    raise ValueError('MPS 上的 Pauli 字符串缺少 σ%s 算符（站点既无 S%s 也无 P/M）'
                     % (axis, axis.upper()))


def pauli_specs_to_mps_order(specs, order):
    """原子原序的 Pauli specs → MPS 序（``order[k]`` = 第 k 个 MPS 位的原子）。

    只做一次下标映射；调用方不要把已映射的 specs 再传进来。
    """
    pos = {int(a): k for k, a in enumerate(order)}
    out = []
    for spec in specs or []:
        out.append({'label': spec['label'],
                    'terms': [(pos[int(s)], ax) for (s, ax) in spec['terms']]})
    return out


def pauli_strings_mps_batch(psi, specs):
    """一批 Pauli 字符串 → ``[complex, …]``，在有限 MPS 上求值。

    实现：把每个 Pauli 串展开成**局域算符之积的和**（σˣ=P+M、σʸ=iP−iM），
    每项用 TeNPy 的 ``MPS.expectation_value_term`` 收缩。这条路的要点：

    * **对荷区透明**：``conserve='Sz'`` 下 `P`/`M` 是改变电荷的算符，但每个
      乘积项整体荷守恒，逐项收缩可以正常进行（实测 tenpy 1.1.1 支持）。
      不要用 `get_rho_segment`：它返回的轴序随 segment 是否连续而变，不能作为
      稳定契约。
    * **不假设局部基的顺序**：算符名由站点自己提供（簇模型注册的 SZ/P/M），
      与 calib 的 |g⟩/|r⟩ 同序；TeNPy 自带的 Sigmaz/Sp/Sm 在 conserve='Sz'
      下会因按电荷重排而差符号/互换，不能直接用。
    * 代价 ∝ 2^(非 z 轴个数)：三体串 ≤ 8 项，六体串 64 项；每个站点集合都
      直接收缩，不做 2^(2k) 的稠密中间量。
    """
    if not specs:
        return []
    sites = list(psi.sites)
    out = []
    for spec in specs:
        terms = [(int(s), a) for s, a in spec['terms']]
        for site, _axis in terms:
            if not (0 <= site < len(sites)):
                raise ValueError('pauli_strings_mps_batch：站点 %r 越界（MPS 长度 %d）'
                                 % (site, len(sites)))
        stack = [(0, 1.0 + 0j, [])]
        acc = 0j
        while stack:
            idx, coef, cur = stack.pop()
            if idx == len(terms):
                acc += coef * psi.expectation_value_term(cur)
                continue
            site, axis = terms[idx]
            for c, op in _mps_axis_terms(sites[site], axis):
                stack.append((idx + 1, coef * c, cur + [(op, site)]))
        out.append(complex(acc))
    return out


# =====================================================================
# Pauli 串的统一求值入口 + 预编译计划
# =====================================================================
#: PauliPlan 的默认缓存预算（只缓存预编译索引；不改变结果）。
PAULI_PLAN_MAX_BYTES = 256 * 1024 ** 2


def _axis_q(bit, axis):
    """单轴在**输出位** s 上的相位因子 q(s)：σᶻ→∓1、σʸ→±i、σˣ→1。"""
    if axis == 'z':
        return np.where(bit, -1.0, 1.0)
    if axis == 'y':
        return np.where(bit, 1j, -1j)
    return 1.0


def _check_pauli_terms(terms, n):
    out = []
    seen = set()
    for site, axis in terms:
        site = int(site)
        if not 0 <= site < n:
            raise ValueError('Pauli 串站点 %r 越界（n=%d）' % (site, n))
        if site in seen:
            raise ValueError('Pauli 串在同一站点 %d 上放了两个算符' % site)
        if axis not in ('x', 'y', 'z'):
            raise ValueError('Pauli 串轴须为 x/y/z，得到 %r' % (axis,))
        seen.add(site)
        out.append((site, axis))
    return out


class PauliPlan:
    """把 Pauli 串预编译成“索引 + 相位”，跨时刻复用。

    三种态形态各自走一条编译路径，但**入口与结果语义只有一个**：

    * ``full``   ：全空间 2^N 态矢；
    * ``basis``  ：任意位编码基（扇区/投影 Plan）；
    * ``mps``    ：TeNPy 有限 MPS（委托 ``pauli_strings_mps_batch``）。

    编译结果是纯数据（``perm/phase`` 或 ``loc/phase``），与时间无关；同一
    ``PauliPlan`` 在 ED 的每个输出时刻复用，避免重复建 `arange`/掩码。
    """

    def __init__(self, n, specs, *, basis_states=None, kind=None,
                 max_bytes=PAULI_PLAN_MAX_BYTES):
        self.n = int(n)
        self.specs = [dict(s) for s in (specs or [])]
        self.basis_states = (None if basis_states is None
                             else np.asarray(basis_states, np.int64).reshape(-1))
        if basis_states is not None:
            kind = 'basis'
        elif kind is None:
            kind = 'full'
        if kind not in ('full', 'basis', 'mps'):
            raise ValueError("PauliPlan kind 只支持 full/basis/mps")
        self.kind = kind
        self.max_bytes = int(max_bytes)
        self._cache = {}
        self._bytes = 0
        if kind == 'basis':
            self.dim = len(self.basis_states)
            if self.dim == 0:
                raise ValueError('PauliPlan：basis_states 为空')
            self._build_locator()
        elif kind == 'full':
            self.dim = 1 << self.n
        else:
            self.dim = None

    # ---- 受限基的 code → position 映射 --------------------------------
    def _build_locator(self):
        codes = self.basis_states
        if self.n <= 25:
            self._pos = np.full(1 << self.n, -1, np.int64)
            self._pos[codes] = np.arange(self.dim, dtype=np.int64)
            self._sorted = None
        else:
            self._pos = None
            self._order = np.argsort(codes)
            self._sorted = codes[self._order]

    def _locate(self, target):
        if self._pos is not None:
            loc = self._pos[target]
            return loc, loc >= 0
        loc = np.searchsorted(self._sorted, target)
        ok = loc < self.dim
        ok[ok] &= self._sorted[loc[ok]] == target[ok]
        return self._order[loc], ok

    # ---- 编译 ---------------------------------------------------------
    def _compile_full(self, terms):
        """逐算符**从左到右**推进：t ← t^mask、phase ← phase·q(t)。

        这等价于 ``(O1…Ok v)[s] = q1(s)·q2(s^m1)…·v(s^m1^…)``；
        不需要中间态落在哪个子空间，也不做任何截断。
        """
        idx = np.arange(self.dim, dtype=np.int64)
        t = idx.copy()
        phase = np.ones(self.dim, complex)
        for site, axis in terms:
            shift = self.n - 1 - site
            bit = (t >> shift) & 1
            phase = phase * _axis_q(bit, axis)
            if axis != 'z':
                t = t ^ (np.int64(1) << shift)
        return t, phase

    def _compile_basis(self, terms):
        codes = self.basis_states
        t = codes.copy()
        phase = np.ones(self.dim, complex)
        for site, axis in terms:
            shift = self.n - 1 - site
            bit = (t >> shift) & 1
            if axis == 'z':
                phase = phase * np.where(bit, -1.0, 1.0)
            elif axis == 'y':
                # 受限基路按**源位**取矩阵元：<r|σʸ|g>=+i、<g|σʸ|r>=-i。
                phase = phase * np.where(bit, -1j, 1j)
            if axis != 'z':
                t = t ^ (np.int64(1) << shift)
        loc, ok = self._locate(t)
        return loc, ok, phase

    def _compiled(self, i):
        hit = self._cache.get(i)
        if hit is not None:
            return hit
        terms = _check_pauli_terms(self.specs[i]['terms'], self.n)
        if not terms:
            data = ('empty',)
        elif self.kind == 'full':
            data = ('full',) + self._compile_full(terms)
        elif self.kind == 'basis':
            data = ('basis',) + self._compile_basis(terms)
        else:
            data = ('mps', terms)
        need = 0
        for item in data[1:]:
            if isinstance(item, np.ndarray):
                need += item.nbytes
        if need and self._bytes + need <= self.max_bytes:
            self._cache[i] = data
            self._bytes += need
        return data

    # ---- 求值 ---------------------------------------------------------
    def expect(self, state):
        """→ ``list[complex]``，顺序与 ``specs`` 一致。"""
        if self.kind == 'mps':
            return pauli_strings_mps_batch(state, self.specs)
        v = np.asarray(state, complex).reshape(-1)
        if v.size != self.dim:
            raise ValueError('PauliPlan：态矢长度 %d ≠ 维度 %d' % (v.size, self.dim))
        out = []
        for i in range(len(self.specs)):
            data = self._compiled(i)
            if data[0] == 'empty':
                out.append(complex(np.vdot(v, v)))
            elif data[0] == 'full':
                out.append(complex(np.sum(np.conj(v) * data[2] * v[data[1]])))
            else:  # basis
                loc, ok, phase = data[1], data[2], data[3]
                out.append(complex(np.sum(np.conj(v[loc[ok]]) * phase[ok] * v[ok])))
        return out


def measure_pauli_strings(state, specs, *, basis_states=None, n=None, order=None):
    """**统一入口**：按态的形态自动分派到全空间/受限基/MPS/torch 适配器。

    - ``basis_states`` 给出 → 受限基（``state`` 是这些位编码上的向量）；
    - ``state`` 是 TeNPy MPS（有 ``expectation_value_term``/``sites``）→ MPS 路；
    - ``state`` 有 ``pauli_expectation`` 方法 → 委托（CUDA MPS 适配器用，
      不把 torch 拉进 core 顶层）；
    - 否则按 1-D 态矢处理，长度必须是 2^n。

    ``order`` 仅在 MPS 路使用：``order[k]`` = 第 k 个 MPS 位对应的原子下标；
    给了它就把**原子原序**的 specs 映射到 MPS 序。引擎内部已映射时不要重复传。
    """
    specs = list(specs or [])
    if not specs:
        return []
    if hasattr(state, 'pauli_expectation'):
        return [complex(x) for x in state.pauli_expectation(specs)]
    if hasattr(state, 'expectation_value_term') or hasattr(state, 'sites'):
        mapped = specs
        if order is not None:
            pos = {int(a): k for k, a in enumerate(order)}
            mapped = [dict(s, terms=[(pos[int(si)], ax) for si, ax in s['terms']])
                      for s in specs]
        return pauli_strings_mps_batch(state, mapped)
    v = np.asarray(state, complex).reshape(-1)
    if basis_states is not None:
        if n is None:
            raise ValueError('measure_pauli_strings：受限基必须显式给 n（位宽不能从'
                             '最大基矢反推，最高位可能全为 0）')
        return PauliPlan(int(n), specs, basis_states=basis_states,
                         kind='basis').expect(v)
    dim = v.size
    if dim == 0 or dim & (dim - 1):
        raise ValueError('measure_pauli_strings：态矢长度 %d 不是 2 的幂' % dim)
    n = int(round(np.log2(dim))) if n is None else int(n)
    if (1 << n) != dim:
        raise ValueError('measure_pauli_strings：态矢长度 %d ≠ 2^%d' % (dim, n))
    return PauliPlan(n, specs, kind='full').expect(v)


def pauli_strings_factorized(spins, specs):
    """因子化（经典/平均场）Pauli 串：``∏ ⟨σ^{a_i}_i⟩``。

    ``spins`` 形状 ``(n_t, n_atom, 3)``（单位 Bloch 向量）。这是**近似**语义，
    调用方必须在结果里标注 ``approximation='factorized ...'``；不要把它当纯态
    期望使用。
    """
    s = np.asarray(spins, float)
    if s.ndim != 3 or s.shape[2] != 3:
        raise ValueError('pauli_strings_factorized：spins 形状须为 (n_t,n_atom,3)')
    comp = {'x': 0, 'y': 1, 'z': 2}
    out = np.zeros((s.shape[0], len(specs or [])), complex)
    for q, spec in enumerate(specs or []):
        val = np.ones(s.shape[0], complex)
        for site, axis in spec['terms']:
            if axis not in comp:
                raise ValueError('Pauli 串轴须为 x/y/z，得到 %r' % (axis,))
            val = val * s[:, int(site), comp[axis]]
        out[:, q] = val
    return out
