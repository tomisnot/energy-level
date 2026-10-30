# -*- coding: utf-8 -*-
"""模型无关的扭转参数 Berry/Chern 诊断原语。

本模块不构造晶格、不决定边界条件，也不假设基矢的物理含义。调用方提供
``state_fn(theta_x, theta_y)``，返回一个归一化态矢或列空间矩阵；后者用于
简并基态的非 Abelian Fukui/Wilson-loop 离散曲率。扭转如何进入 Hamiltonian
必须由模型层实现，通常是环面跨边界链接的 Peierls 相位。
"""
import numpy as np

__all__ = ['fukui_chern', 'loop_flux', 'twist_pairs', 'twist_calibration', 'wilson_loop',
           'spectral_flow', 'entanglement_spectrum', 'topology_result',
           'topology_report']

TOPOLOGY_SCHEMA = 'el.topology.v1'
TOPOLOGY_STATUSES = ('not_run', 'ok', 'finite_size_candidate',
                     'inconclusive', 'failed')


def _frame(value):
    """将态矢/子空间转成正交列框架，并拒绝零向量。"""
    a = np.asarray(value, complex)
    if a.ndim == 1:
        if a.size == 0:
            raise ValueError('state_fn 返回空态矢')
        nrm = np.linalg.norm(a)
        if not np.isfinite(nrm) or nrm <= 1e-14:
            raise ValueError('state_fn 返回零范数或非有限态矢')
        return (a / nrm)[:, None]
    if a.ndim != 2 or a.shape[0] == 0 or a.shape[1] == 0:
        raise ValueError('state_fn 必须返回形如 (dim,) 或 (dim,n_states) 的数组')
    q, r = np.linalg.qr(a, mode='reduced')
    if q.shape[1] != a.shape[1] or np.min(np.abs(np.diag(r))) <= 1e-14:
        raise ValueError('state_fn 返回的简并子空间列不满秩')
    return q


def _link_phase(left, right):
    """两正交框架之间的 Abelian/non-Abelian overlap 相位。"""
    if left.shape != right.shape:
        raise ValueError('所有扭转点必须返回相同维度和相同简并度的态')
    z = np.linalg.det(left.conj().T @ right)
    az = abs(z)
    if not np.isfinite(az) or az <= 1e-14:
        raise ValueError('相邻扭转点的子空间重叠过小，网格或能隙不足')
    return z / az, float(az)


def fukui_chern(state_fn, grid=(8, 8)):
    """在二维扭转环面上计算离散 Chern 数。

    ``state_fn(theta_x, theta_y)`` 的参数范围为 ``[0,2π)``，可返回单个态矢
    或 ``(dim,n_states)`` 的简并子空间。对多列返回非 Abelian 链变量
    ``det(U†V)/|det(U†V)|``。结果保留未取整的 `chern`、最小链接重叠和每个
    网格面曲率，方便调用方判断有限网格/近简并失败；引擎不把结果自动宣称
    为拓扑序。
    """
    nx, ny = (int(grid[0]), int(grid[1]))
    if nx < 2 or ny < 2:
        raise ValueError('Fukui 网格每个方向至少需要 2 个点')
    frames = [[None] * ny for _ in range(nx)]
    for ix in range(nx):
        tx = 2.0 * np.pi * ix / nx
        for iy in range(ny):
            ty = 2.0 * np.pi * iy / ny
            frames[ix][iy] = _frame(state_fn(tx, ty))
    dim, ns = frames[0][0].shape
    ux = np.empty((nx, ny), complex)
    uy = np.empty((nx, ny), complex)
    min_overlap = float('inf')
    for ix in range(nx):
        for iy in range(ny):
            ux[ix, iy], ox = _link_phase(frames[ix][iy], frames[(ix + 1) % nx][iy])
            uy[ix, iy], oy = _link_phase(frames[ix][iy], frames[ix][(iy + 1) % ny])
            min_overlap = min(min_overlap, ox, oy)
    curvature = np.empty((nx, ny), float)
    for ix in range(nx):
        for iy in range(ny):
            plaquette = (ux[ix, iy] * uy[(ix + 1) % nx, iy]
                         * np.conj(ux[ix, (iy + 1) % ny])
                         * np.conj(uy[ix, iy]))
            curvature[ix, iy] = float(np.angle(plaquette))
    ch = float(curvature.sum() / (2.0 * np.pi))
    return dict(chern=ch, chern_rounded=int(np.rint(ch)), grid=(nx, ny),
                dim=int(dim), n_states=int(ns),
                min_link_overlap=float(min_overlap),
                max_abs_curvature=float(np.max(np.abs(curvature))),
                curvature=curvature)


def loop_flux(pairs, loop, key='J'):
    """从 `calibrate` 的逐对记录计算有向链接的规范不变量闭环相位。

    `pairs` 中每条记录含 `i,j,key` 且只存 `i<j`；反向边自动取复共轭。
    返回复数乘积、弧度相位和逐边值。缺边或零链接明确报错，避免静默改环。
    """
    seq = [int(x) for x in loop]
    if len(seq) < 2 or len(set(seq)) != len(seq):
        raise ValueError('loop 至少需要两个不重复的站点')
    table = {(int(p['i']), int(p['j'])): p for p in pairs}
    edges = []
    product = 1.0 + 0.0j
    for a, b in zip(seq, seq[1:] + seq[:1]):
        i, j = min(a, b), max(a, b)
        rec = table.get((i, j))
        if rec is None:
            raise ValueError('loop 缺少链接 (%d,%d)' % (a, b))
        z = complex(rec[key])
        if a > b:
            z = np.conj(z)
        if abs(z) <= 1e-14:
            raise ValueError('loop 链接 (%d,%d) 幅值为零' % (a, b))
        product *= z
        edges.append(dict(i=a, j=b, value=z, phase_rad=float(np.angle(z))))
    return dict(loop=seq, product=product, phase_rad=float(np.angle(product)),
                winding=float(np.angle(product) / (2.0 * np.pi)), edges=edges)


def twist_pairs(pairs, positions, periods, theta=(0.0, 0.0), key='J',
                winding_key='twist_winding'):
    """给跨周期链接施加环面 Peierls 扭转。

    `positions` 是胞元内坐标，`periods=(Lx,Ly)`；对 `i<j` 的记录，先取
    ``w = round((r_j-r_i)/L)``，再将链接乘以
    ``exp(-1j * dot(theta, w))``。这个符号约定对应“正向最小像跨越正边界
    获得 `exp(+i theta)`”；若模型已有明确的跨胞元整数，可把它放在记录的
    `winding_key` 中，优先于几何推断。函数只修改指定的 hopping `key`，保留
    `V/A` 等其他项，并返回每条链接的 winding/phase 因子供审计。

    这一步只负责链接数据。`A` 双激发项是否允许扭转取决于模型的 U(1) 对称性，
    因而不会被静默修改。
    """
    pos = np.asarray(positions, float)
    L = np.asarray(periods, float).reshape(-1)
    th = np.asarray(theta, float).reshape(-1)
    if pos.ndim != 2 or pos.shape[1] < 2:
        raise ValueError('positions 需要形如 (N,2) 或 (N,3)')
    if L.size != 2 or np.any(~np.isfinite(L)) or np.any(L <= 0):
        raise ValueError('periods 需要两个正数 (Lx,Ly)')
    if th.size != 2 or np.any(~np.isfinite(th)):
        raise ValueError('theta 需要两个有限数 (theta_x,theta_y)')
    out, windings, phases = [], [], []
    for raw in pairs:
        p = dict(raw)
        i, j = int(p['i']), int(p['j'])
        if not (0 <= i < len(pos) and 0 <= j < len(pos)):
            raise ValueError('pair 下标超出 positions')
        if key not in p:
            raise ValueError('pair 缺少 hopping key=%r' % key)
        if winding_key in p:
            w = np.asarray(p[winding_key], int).reshape(-1)
            if w.size != 2:
                raise ValueError('%s 需要两个整数' % winding_key)
        else:
            d = pos[j, :2] - pos[i, :2]
            w = np.rint(d / L).astype(int)
        phase = np.exp(-1j * float(np.dot(th, w)))
        p[key] = complex(p[key]) * phase
        p[winding_key] = [int(w[0]), int(w[1])]
        out.append(p)
        windings.append([int(w[0]), int(w[1])])
        phases.append(complex(phase))
    return dict(pairs=out, theta=th.copy(), periods=L.copy(),
                windings=windings, phase_factors=phases,
                convention='hopping *= exp(-i dot(theta, winding))')


def twist_calibration(calib, positions, periods, theta=(0.0, 0.0), key='J'):
    """对 EL `calibrate` 返回的内存对象施加 hopping 扭转并保留其余字段。"""
    if not isinstance(calib, dict) or 'pairs' not in calib:
        raise ValueError('calib 必须是含 pairs 的 EL 标定字典')
    out = dict(calib)
    twisted = twist_pairs(calib['pairs'], positions, periods, theta=theta, key=key)
    out['pairs'] = twisted['pairs']
    out['topology_twist'] = dict(theta=twisted['theta'], periods=twisted['periods'],
                                 windings=twisted['windings'],
                                 convention=twisted['convention'])
    return out


def _unitary_link(left, right):
    """框架重叠的极分解单位阵和最小奇异值。"""
    if left.shape != right.shape:
        raise ValueError('Wilson loop 的相邻框架形状必须一致')
    overlap = left.conj().T @ right
    u, s, vh = np.linalg.svd(overlap, full_matrices=False)
    if s.size == 0 or not np.all(np.isfinite(s)) or float(s.min()) <= 1e-14:
        raise ValueError('Wilson loop 相邻子空间重叠过小')
    return u @ vh, float(s.min())


def wilson_loop(state_fn, path):
    """沿闭合参数路径计算 Abelian/非 Abelian Wilson loop。

    `path` 给出不重复终点的参数列表；函数自动连接最后一点和第一点。
    `state_fn(parameter)` 返回单态或简并子空间列矩阵。返回 Wilson 矩阵、
    本征相位、行列式总相位和最小子空间重叠。对简并基态，矩阵本征相位比
    单个任意规范下的态相位更稳定。
    """
    params = list(path)
    if len(params) < 2:
        raise ValueError('Wilson loop 至少需要两个路径点')
    frames = [_frame(state_fn(p)) for p in params]
    W = np.eye(frames[0].shape[1], dtype=complex)
    min_overlap = float('inf')
    for i, left in enumerate(frames):
        link, ov = _unitary_link(left, frames[(i + 1) % len(frames)])
        W = W @ link
        min_overlap = min(min_overlap, ov)
    vals = np.linalg.eigvals(W)
    phases = np.angle(vals)
    det = np.linalg.det(W)
    return dict(n_points=len(params), n_states=int(W.shape[0]),
                wilson_matrix=W, eigenphases_rad=phases,
                determinant_phase_rad=float(np.angle(det)),
                min_link_overlap=float(min_overlap))


def spectral_flow(solver, path, n_levels=None):
    """沿 flux/twist 路径追踪低能谱流。

    `solver(parameter)` 返回 `{'evals': ..., 'vecs': ...}`，也可只返回本征值数组。
    有态矢时使用相邻点最大重叠的 Hungarian 匹配追踪分支；没有态矢时按能量排序，
    并在结果中标明 `tracking='energy_order'`。函数不假设路径一定闭合。
    """
    params = list(path)
    if len(params) < 1:
        raise ValueError('spectral_flow path 不能为空')
    samples = []
    for p in params:
        raw = solver(p)
        if isinstance(raw, dict):
            vals = raw.get('evals', raw.get('energies'))
            vecs = raw.get('vecs', raw.get('vectors', raw.get('states')))
        else:
            vals, vecs = raw, None
        if vals is None:
            raise ValueError('solver 返回值缺少 evals/energies')
        vals = np.asarray(vals).reshape(-1)
        if vals.size == 0:
            raise ValueError('solver 返回空谱')
        order = np.argsort(np.real(vals))
        vals = np.real(vals[order]).astype(float)
        if n_levels is not None:
            vals = vals[:int(n_levels)]
        if vecs is not None:
            v = np.asarray(vecs, complex)
            if v.ndim == 1:
                v = v[:, None]
            if v.ndim != 2 or v.shape[1] < len(order):
                raise ValueError('solver vecs 列数不足以对应 evals')
            v = v[:, order[:len(vals)]]
            v, _ = np.linalg.qr(v, mode='reduced')
        samples.append((vals, v))
    k = len(samples[0][0])
    if any(len(vals) != k for vals, _ in samples):
        raise ValueError('spectral_flow 各点返回的能级数必须一致')
    tracked = np.empty((len(samples), k), float)
    branch = np.empty((len(samples), k), int)
    tracked[0], branch[0] = samples[0][0], np.arange(k)
    tracking = 'overlap' if samples[0][1] is not None else 'energy_order'
    for t in range(1, len(samples)):
        vals, vecs = samples[t]
        prev_vecs = samples[t - 1][1]
        if prev_vecs is None or vecs is None:
            perm = np.arange(k)
        else:
            from scipy.optimize import linear_sum_assignment
            overlap = np.abs(prev_vecs.conj().T @ vecs) ** 2
            rr, cc = linear_sum_assignment(-overlap)
            perm = np.empty(k, int); perm[rr] = cc
        tracked[t] = vals[perm]
        branch[t] = perm
    gap = tracked[:, 1] - tracked[:, 0] if k > 1 else np.full(len(samples), np.nan)
    return dict(path=np.asarray(params), evals=tracked, energy_order=np.asarray(
        [v for v, _ in samples]), branch_indices=branch, tracking=tracking,
                ground_gap=gap, min_ground_gap=float(np.nanmin(gap)) if k > 1 else float('nan'))


def entanglement_spectrum(psi, n_sites, cut, base=np.e):
    """二分纯态的 Schmidt 权重和纠缠能谱。"""
    n, cut = int(n_sites), int(cut)
    if n < 2 or not (1 <= cut <= n - 1):
        raise ValueError('entanglement_spectrum 要求 n>=2 且 1<=cut<n')
    v = np.asarray(psi, complex).reshape(-1)
    if v.size != (1 << n):
        raise ValueError('态矢长度必须为 2^n')
    norm = np.linalg.norm(v)
    if not np.isfinite(norm) or norm <= 1e-14:
        raise ValueError('态矢零范数或非有限')
    M = (v / norm).reshape(1 << cut, 1 << (n - cut))
    sv = np.linalg.svd(M, compute_uv=False)
    weights = sv ** 2
    weights = weights[weights > 1e-15]
    log_base = float(np.log(base)) if base not in (None, np.e) else 1.0
    energies = -np.log(weights) / log_base
    entropy = float(-np.sum(weights * np.log(weights)))
    return dict(n_sites=n, cut=cut, schmidt_values=sv,
                weights=weights, entanglement_energies=energies,
                entropy=entropy, entropy_base=base, rank=int(weights.size))


def topology_result(diagnostic, status='inconclusive', data=None,
                    evidence=None, warnings=None, error=None):
    """构造统一拓扑诊断结果。"""
    status = str(status)
    if status not in TOPOLOGY_STATUSES:
        raise ValueError('未知 topology status=%r' % status)
    return dict(schema=TOPOLOGY_SCHEMA, diagnostic=str(diagnostic),
                status=status, data=dict(data or {}), evidence=list(evidence or []),
                warnings=list(warnings or []), error=error)


def topology_report(diagnostics, metadata=None):
    """合并多个诊断并给出可审计的总体状态。"""
    items = []
    for d in diagnostics:
        if d.get('schema') != TOPOLOGY_SCHEMA:
            raise ValueError('diagnostic schema 不匹配')
        items.append(dict(d))
    statuses = [d['status'] for d in items]
    if any(s == 'failed' for s in statuses):
        overall = 'failed'
    elif any(s in ('finite_size_candidate', 'inconclusive', 'not_run') for s in statuses):
        overall = 'inconclusive'
    else:
        overall = 'ok'
    return dict(schema=TOPOLOGY_SCHEMA, diagnostic='topology_report',
                status=overall, metadata=dict(metadata or {}), diagnostics=items,
                status_counts={s: statuses.count(s) for s in TOPOLOGY_STATUSES})
