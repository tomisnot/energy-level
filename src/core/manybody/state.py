# -*- coding: utf-8 -*-
"""Generic local two-level state specifications shared by ED and MPS.

The representation is deliberately model-independent: a product state is a
list of normalized local vectors in the local basis ``(|g>, |r>)``.  A vortex
is just one possible choice of their phases; the solver does not know about
that interpretation.
"""
import numpy as np


def make_mps_state(psi, order, charge_sector=None):
    """An independent finite TeNPy MPS plus its atom/basis metadata.

    This is a Python RunSpec payload, not a JSON object.  Use the HDF5
    checkpoint helpers when transporting it between processes or sessions.
    """
    snapshot = psi.copy()
    if snapshot.L > 1:
        snapshot.canonical_form(renormalize=False, cutoff=0.)
    state = dict(kind='mps', backend='tenpy', version=1, psi=snapshot,
                 order=list(map(int, order)), local_basis=['g', 'r'],
                 charge_sector=charge_sector)
    validate_mps_state(state, len(order))
    return state


def _reorder_atom_to_mps(vec, order):
    n = len(order)
    order = [int(a) for a in order]
    out = np.empty(1 << n, complex)
    for atom_idx in range(1 << n):
        mps_idx = 0
        for atom in order:                       # order = atom -> MPS position
            mps_idx = (mps_idx << 1) | ((atom_idx >> (n - 1 - atom)) & 1)
        out[atom_idx] = vec[mps_idx]
    return out


def mps_state_to_vector(state, n, *, max_n=16):
    """Materialize a finite native MPS state in the atom-order full basis.

    Only for spectrum/ED handoff at small N; this is an explicit 2^N
    materialization, never a silent fallback for large systems.
    """
    n = int(n)
    if n > int(max_n):
        raise ValueError('原生 MPS 转全空间向量上限为 N=%d（当前 %d）；'
                         '请改用源态谱支持的向量/模式源或缩减模型' % (max_n, n))
    if not isinstance(state, dict) or state.get('kind') != 'mps':
        raise ValueError('state 必须是 kind=mps 的原生态')
    order = list(map(int, state.get('order') or []))
    if sorted(order) != list(range(n)):
        raise ValueError('MPS state.order 必须是原子下标的完整排列')
    backend = state.get('backend')
    if backend == 'tenpy':
        validate_mps_state(state, n)
        theta = state['psi'].get_theta(0, n)
        vec = np.asarray(theta.to_ndarray(), complex).reshape(-1)
    elif backend == 'tdvp-dense':
        from .tdvp_dense import _load_dense_state
        mps = _load_dense_state(state, n, order)
        vec = np.asarray(mps.to_dense(), complex) * float(mps.norm)
    else:
        raise ValueError('不支持把 backend=%r 的原生态转为全空间向量' % (backend,))
    if vec.size != (1 << n):
        raise ValueError('原生态物化维度 %d ≠ 2^N=%d' % (vec.size, 1 << n))
    vec = _reorder_atom_to_mps(vec, order)
    norm = float(np.linalg.norm(vec))
    if not np.isfinite(norm) or norm <= 0:
        raise ValueError('原生态物化后零范数/非有限')
    return vec / norm


def validate_mps_state(state, n):
    if state.get('backend') != 'tenpy' or state.get('version') != 1:
        raise ValueError('CPU MPS state 需要 backend=tenpy, version=1')
    from tenpy.networks.mps import MPS
    psi = state.get('psi')
    if not isinstance(psi, MPS) or psi.bc != 'finite' or psi.L != n:
        raise ValueError('state.psi 必须是相同站点数的有限 TeNPy MPS')
    if sorted(state.get('order', [])) != list(range(n)):
        raise ValueError('MPS state.order 必须是原子下标的完整排列')
    if state.get('local_basis') != ['g', 'r'] or any(s.dim != 2 for s in psi.sites):
        raise ValueError('MPS state 的局部基必须是 g/r 二能级')
    norm = float(np.real(psi.overlap(psi)))
    if not np.isfinite(norm) or abs(norm-1.) > 1e-8:
        raise ValueError('MPS state 必须归一；不会静默丢弃算符作用态的范数')


def save_mps_checkpoint(state, path):
    """Persist a trusted local finite MPS with charge and ordering metadata."""
    from tenpy.tools import hdf5_io
    validate_mps_state(state, len(state.get('order', [])))
    hdf5_io.save(state, str(path))


def load_mps_checkpoint(path):
    """Read a locally produced TeNPy HDF5 checkpoint; no basis conversion."""
    from tenpy.tools import hdf5_io
    state = hdf5_io.load(str(path))
    validate_mps_state(state, len(state.get('order', [])))
    return state


def _scalar(x):
    if isinstance(x, (list, tuple)) and len(x) == 2:
        return complex(float(x[0]), float(x[1]))
    return complex(x)


def state_vector(spec, n, basis_indices=None):
    """Materialize a normalized state directly in the declared MSB basis.

    Patterns and product states never allocate a full 2**N intermediate when
    a projected basis is supplied. Initial weight outside the basis is an
    error, rather than an implicit physical projection/renormalization.
    """
    from .basis import basis_bits
    n = int(n)
    basis = (np.arange(1 << n, dtype=np.int64) if basis_indices is None
             else np.asarray(basis_indices, np.int64))
    projected = basis_indices is not None
    if isinstance(spec, np.ndarray):
        full = np.asarray(spec, complex)
        if full.shape != (1 << n,):
            raise ValueError('初态向量形状必须是 (2**N,)；受限态需携带完整基身份')
        norm = np.linalg.norm(full)
        vec = full[basis] / norm if norm > 0 else full[basis]
    elif isinstance(spec, (list, tuple)) and spec and isinstance(spec[0], str) and spec[0] == 'sup':
        vec = np.zeros(len(basis), complex)
        for component in spec[1]:
            if (isinstance(component, (list, tuple)) and len(component) == 2
                    and not isinstance(component[0], (list, tuple, dict))):
                pattern, amplitude = component
                vec += _scalar(amplitude) * state_vector(pattern, n, basis_indices)
            else:
                vec += state_vector(component, n, basis_indices)
        norm = np.linalg.norm(vec)
        if not np.isfinite(norm) or norm <= 0:
            raise ValueError('多体初态零范数或包含非有限数值')
        return vec / norm
    elif isinstance(spec, (int, np.integer)) and not isinstance(spec, bool):
        if not 0 <= int(spec) < (1 << n):
            raise ValueError('多体初态索引越界')
        vec = (basis == int(spec)).astype(complex)
    elif (isinstance(spec, (list, tuple)) and len(spec) == (1 << n)
          and not (len(spec) == n and all(isinstance(x, (str, int, np.integer)) for x in spec))):
        return state_vector(np.asarray(spec, complex), n, basis_indices)
    else:
        vectors = product_vectors(spec, n)
        if vectors is None:
            raise ValueError('ground 初态必须由求解器从当前 Hamiltonian 准备')
        bits = basis_bits(n, basis)
        vec = np.ones(len(basis), complex)
        for i, local in enumerate(vectors):
            vec *= np.asarray(local)[bits[i]]
    norm = float(np.linalg.norm(vec))
    if not np.isfinite(norm) or norm <= 0:
        raise ValueError('多体初态零范数或不在受限子空间')
    if projected and abs(norm - 1.) > 1e-10:
        raise ValueError('初态含受限子空间之外的幅度；不会静默投影')
    return vec / norm


def _local_vector(x):
    if isinstance(x, str):
        if x == 'g': return np.array([1., 0.], complex)
        if x == 'r': return np.array([0., 1.], complex)
        raise ValueError('局部二能级态字符串只接受 g 或 r')
    if isinstance(x, (int, np.integer)) and not isinstance(x, bool):
        if int(x) not in (0, 1): raise ValueError('局部二能级态索引只接受 0 或 1')
        return np.eye(2, dtype=complex)[:, int(x)]
    a = np.asarray([_scalar(v) for v in x], complex) if hasattr(x, '__len__') else None
    if a is None or a.shape != (2,):
        raise ValueError('局部态必须是 [a_g, a_r]，形状为 (2,)')
    norm = float(np.linalg.norm(a))
    if not np.isfinite(norm) or norm <= 0.0: raise ValueError('局部态不能是零向量或含非有限数值')
    return a / norm


def product_vectors(spec, n):
    """Return local vectors or ``None`` for the special ``ground`` spec."""
    if spec is None:
        return [np.array([1., 0.], complex) for _ in range(int(n))]
    if isinstance(spec, (int, np.integer)) and not isinstance(spec, bool):
        if not 0 <= int(spec) < (1 << int(n)):
            raise ValueError('多体初态索引越界')
        return [_local_vector((int(spec) >> (int(n)-1-i)) & 1) for i in range(int(n))]
    if isinstance(spec, str):
        if spec == 'ground': return None
        s = spec.replace('|', '').replace('>', '').replace(' ', '')
        if len(s) != int(n) or any(c not in 'gr' for c in s):
            raise ValueError('模式串须由长度为 N 的 g/r 组成')
        return [_local_vector(c) for c in s]
    if isinstance(spec, dict):
        kind = spec.get('kind', 'product')
        if kind != 'product':
            raise ValueError('未知乘积态 kind=%r；请使用 kind=product + amplitudes' % kind)
        raw = spec.get('amplitudes', spec.get('local_states'))
        if raw is None:
            raise ValueError('product 初态需要 amplitudes 或 local_states')
        spec = raw
    if isinstance(spec, (list, tuple)) and len(spec) == int(n):
        # Strings/integers are the legacy pattern form; arrays are general
        # local superpositions.
        return [_local_vector(x) for x in spec]
    raise ValueError('无法解析多体乘积初态')


def product_vector(spec, n):
    vecs = product_vectors(spec, n)
    if vecs is None: return None
    out = np.array([1. + 0j])
    for a in vecs:
        out = np.kron(out, a)
    return out


def validate_product_spec(spec, n):
    """Validate without materializing the (2^N) vector."""
    if isinstance(spec, str) and spec == 'ground': return
    product_vectors(spec, n)


def validate_psi_spec(spec, n):
    """Validate a general RunSpec initial-state description without 2**N allocation.

    This mirrors the public ED grammar (patterns, indices, product states and
    ``['sup', ...]``) but deliberately does not construct a Hilbert-space
    vector.  The execution backend still performs the final dimension check
    when it actually needs the state.
    """
    n = int(n)
    dim = 1 << n
    if spec is None or (isinstance(spec, str) and spec == 'ground'):
        return
    if isinstance(spec, np.ndarray):
        # 任意态矢（Python API 用；如脉冲后 exp(-iεO)|0⟩）。与 state_vector 同一
        # 口径：必须是**全空间** (2^N,)，投影基由执行层按 basis_indices 处理。
        if spec.shape != (dim,):
            raise ValueError('多体初态向量形状须是 (2**N,) = (%d,)，收到 %r'
                             % (dim, spec.shape))
        if not np.isfinite(spec).all() or np.linalg.norm(spec) <= 0:
            raise ValueError('多体初态向量零范数或含非有限数值')
        return
    if isinstance(spec, (int, np.integer)) and not isinstance(spec, bool):
        if not 0 <= int(spec) < dim:
            raise ValueError('多体初态索引 %d 越界（dim=%d）' % (int(spec), dim))
        return
    if isinstance(spec, str):
        s = spec.replace('|', '').replace('>', '').replace(' ', '')
        if len(s) != n or any(c not in 'gr' for c in s):
            raise ValueError('模式串须由长度为 N 的 g/r 组成')
        return
    if isinstance(spec, dict):
        if spec.get('kind') == 'mps':
            if spec.get('backend') == 'tenpy':
                validate_mps_state(spec, n)
            else:
                tensors = [np.asarray(t) for t in spec.get('tensors', [])]
                order = spec.get('order', list(range(n)))
                if (len(tensors) != n or sorted(order) != list(range(n))
                        or any(t.ndim != 3 or t.shape[1] != 2 for t in tensors)
                        or tensors[0].shape[0] != 1 or tensors[-1].shape[2] != 1
                        or any(a.shape[2] != b.shape[0] for a,b in zip(tensors,tensors[1:]))
                        or any(not np.isfinite(t).all() for t in tensors)):
                    raise ValueError('MPS tensors/order 不合法')
            return
        if spec.get('kind') in ('peps', 'tensor_network'):
            ts = spec.get('tensors')
            if not isinstance(ts, (list, tuple)) or len(ts) != n:
                raise ValueError('PEPS 初态 tensors 数量必须等于原子数')
            return
        validate_product_spec(spec, n)
        return
    if isinstance(spec, np.ndarray):
        if spec.ndim != 1 or spec.shape != (dim,):
            raise ValueError('多体初态向量形状应 (%d,)，得 %s' % (dim, spec.shape))
        if not np.isfinite(np.asarray(spec, complex)).all() or np.linalg.norm(np.asarray(spec, complex)) <= 0:
            raise ValueError('多体初态零范数')
        return
    if isinstance(spec, (list, tuple)):
        if spec and isinstance(spec[0], str) and spec[0] == 'sup':
            comps = spec[1] if len(spec) > 1 else []
            if not isinstance(comps, (list, tuple)):
                raise ValueError("多体叠加初态的第二项须是分量列表")
            for comp in comps:
                if (isinstance(comp, (list, tuple)) and len(comp) == 2
                        and not isinstance(comp[0], (list, tuple, dict))):
                    validate_psi_spec(comp[0], n)
                    _scalar(comp[1])
                else:
                    validate_psi_spec(comp, n)
            return
        if len(spec) == n:
            # A list of local vectors is the generic product-state form.
            if all(isinstance(x, (str, int, np.integer)) and not isinstance(x, bool)
                   for x in spec):
                for x in spec:
                    _local_vector(x)
                return
            if any(hasattr(x, '__len__') for x in spec):
                validate_product_spec(spec, n)
                return
        if len(spec) == dim:
            v = np.asarray(spec, complex)
            if not np.isfinite(v).all() or np.linalg.norm(v) <= 0:
                raise ValueError('多体初态零范数')
            return
    raise ValueError('无法解析多体初态 psi0=%r' % (spec,))


def initial_sigma_z(spec, n, *, max_mps_n=16, max_vector_n=18):
    """初态的逐格点 ⟨σᶻᵢ⟩（g=+1，r=−1），不要求直积态。

    乘积态/索引/模式串按局部态直接算，不分配 2^N；叠加态、任意全空间态矢先物
    化成向量再算局域期望；有限 MPS 仅在 ``n <= max_mps_n`` 时物化，任意态矢
    仅在 ``n <= max_vector_n`` 时物化，否则返回 ``None``（调用方应显示"不可用"，
    不要编造）。``ground`` 需要求解器先算基态，这里同样返回 ``None``。
    """
    n = int(n)
    if n <= 0:
        return None
    if spec is None:
        return np.ones(n, dtype=float)
    if isinstance(spec, str) and spec == 'ground':
        return None
    # 乘积态/模式串/索引/局部直积：局部期望即可，避免 2^N。
    try:
        vectors = product_vectors(spec, n)
    except (ValueError, TypeError):
        vectors = None
    if vectors is not None:
        return np.array([abs(v[0]) ** 2 - abs(v[1]) ** 2 for v in vectors], float)
    # 非直积态要物化才能取局域期望；展示层有明确 N 上限，避免把一次运行
    # 的 2^N 向量再复制成 N×2^N 的位矩阵。
    if n > max_vector_n:
        return None
    # 原生 MPS 只在明确的小 N 上限内物化；大 MPS 的局域期望留给执行层。
    if isinstance(spec, dict) and spec.get('kind') == 'mps':
        try:
            vec = mps_state_to_vector(spec, n, max_n=max_mps_n)
        except (ValueError, TypeError):
            return None
        basis = np.arange(1 << n, dtype=np.int64)
    else:
        try:
            vec = state_vector(spec, n)
        except (ValueError, TypeError):
            return None
        basis = np.arange(1 << n, dtype=np.int64)
    from .basis import basis_bits
    bits = np.asarray(basis_bits(n, basis), dtype=np.int8)
    probs = np.abs(vec) ** 2
    return (1.0 - 2.0 * bits) @ probs


__all__ = ['product_vectors', 'product_vector', 'validate_product_spec',
           'validate_psi_spec', 'make_mps_state', 'validate_mps_state',
           'save_mps_checkpoint', 'load_mps_checkpoint', 'mps_state_to_vector',
           'initial_sigma_z']
