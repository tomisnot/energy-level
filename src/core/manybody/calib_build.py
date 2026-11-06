# -*- coding: utf-8 -*-
"""显式 calib 的构造小工具：几何 + 耦合 → calib（calib 直通路的配套）。

本模块只做**通用**的三件事，不含任何具体物理模型：

1. ``lattice_positions``：任意 2D/3D 格子（给定基矢 + 胞内基元）→ 点云；
2. ``distance_pairs``：点云 → 两两距离与（可选的）周期像求和；
3. ``from_positions``：点云 + 耦合核 → 可直接塞进 ``manybody.calib`` 的 dict。

耦合核由调用方给：``coupling(r)`` 返回 ``J``（标量）或 ``{'V':…, 'J':…, 'A':…}``
或 ``(V, J, A)`` 三元组，单位 MHz，``r`` 单位 µm（与全库一致）。

⚠ 周期像这里是**显式小范围求和**（``images=n`` ⇒ (2n+1)² 个平移）。需要
解析尾部或大范围像求和的高精度模型，调用方自己算好耦合表再传 ``pairs``。
"""
from __future__ import annotations

import numpy as np

__all__ = ['lattice_positions', 'distance_pairs', 'from_positions']


def _as_points(positions):
    pts = np.asarray(positions, float)
    if pts.ndim != 2 or pts.shape[1] != 3:
        raise ValueError('positions 须是 (N,3) 的点云（µm），收到 %r' % (pts.shape,))
    if pts.shape[0] == 0:
        raise ValueError('positions 为空——calib 至少要有一个原子')
    return pts


def _coupling_triplet(value):
    """把耦合核的返回值规范成 (V, J, A)。"""
    if isinstance(value, dict):
        return (float(value.get('V') or 0.0), float(value.get('J') or 0.0),
                float(value.get('A') or 0.0))
    if isinstance(value, (tuple, list)):
        if len(value) == 3:
            return tuple(float(x) for x in value)
        raise ValueError('耦合核返回的元组须是 (V,J,A)，收到 %r' % (value,))
    return (0.0, float(value), 0.0)          # 标量 ⇒ 纯交换 J


def _v3(vec, what='向量'):
    """2D/3D 向量 → 3D（z 缺省补 0）。"""
    v = np.asarray(vec, float)
    if v.size == 2:
        return np.array([v[0], v[1], 0.0])
    if v.size == 3:
        return v
    raise ValueError('%s 须是 2 或 3 个数，收到 %r' % (what, vec))


def lattice_positions(cells, a1, a2, offsets=((0.0, 0.0, 0.0),), a3=None):
    """格子 → 点云：``pos = m*a1 + n*a2 (+ l*a3) + offset_sub``（单位 µm）。

    ``cells``：``(m,n)`` 或 ``(m,n,l)`` 的可迭代（顺序即原子下标顺序）；
    ``offsets``：胞内基元，可给 2 元或 3 元坐标（缺的 z 补 0）；
    ``a3``：三维格子才需要。
    """
    a1 = _v3(a1, 'a1')
    a2 = _v3(a2, 'a2')
    a3 = None if a3 is None else _v3(a3, 'a3')
    basis = [_v3(off, 'offsets 项') for off in offsets]
    out = []
    for cell in cells:
        cell = tuple(float(x) for x in cell)
        if len(cell) == 2:
            if a3 is not None:
                raise ValueError('给了 a3 但 cells 是 (m,n)：请用 (m,n,l)')
            origin = cell[0] * a1 + cell[1] * a2
        elif len(cell) == 3:
            if a3 is None:
                raise ValueError('cells 是 (m,n,l) 但没给 a3')
            origin = cell[0] * a1 + cell[1] * a2 + cell[2] * a3
        else:
            raise ValueError('cells 每项须是 (m,n) 或 (m,n,l)，收到 %r' % (cell,))
        for off in basis:
            p = origin + off
            out.append([float(p[0]), float(p[1]), float(p[2]) if p.size > 2 else 0.0])
    return out


def distance_pairs(positions, *, cutoff_um=None, periods_um=None, images=0):
    """点云 → ``[(i, j, r_eff, dx, dy, dz), …]``（i<j，周期像已求和）。

    ``periods_um=(Lx,Ly)`` + ``images=n`` 时，对每个原子对把 (2n+1)² 个平移的
    贡献都算进来；返回的 ``r_eff`` 是**最近像**的距离（仅作诊断），真正累加的是
    每个像各自的 1/r 依赖，由 ``from_positions`` 逐像调用耦合核完成。
    ↪ 若只需要距离列表（不走 calib），这个函数也直接可用。
    """
    pts = _as_points(positions)
    n = pts.shape[0]
    shifts = [(0.0, 0.0)]
    if periods_um is not None:
        periods = np.asarray(periods_um, float)
        if periods.shape != (2,) or np.any(periods <= 0):
            raise ValueError('periods_um 须是 (Lx,Ly) 两个正数')
        k = int(images)
        if k < 0:
            raise ValueError('images 不能为负')
        shifts = [(ix * periods[0], iy * periods[1])
                  for ix in range(-k, k + 1) for iy in range(-k, k + 1)]
    out = []
    for i in range(n):
        for j in range(i + 1, n):
            best = None
            for sx, sy in shifts:
                d = pts[j] - pts[i] + np.array([sx, sy, 0.0])
                r = float(np.linalg.norm(d))
                if cutoff_um is not None and r > float(cutoff_um):
                    continue
                if best is None or r < best[0]:
                    best = (r, d)
            if best is not None:
                out.append((i, j, best[0], float(best[1][0]), float(best[1][1]),
                            float(best[1][2])))
    return out


def from_positions(positions, coupling, *, terms=('I2',), cutoff_um=None,
                   periods_um=None, images=0, source='calib_build.from_positions'):
    """点云 + 耦合核 → 可直接作为 ``manybody.calib`` 的 dict。

    ``coupling(r)``：r（µm）→ J 标量，或 ``{'V','J','A'}`` / ``(V,J,A)``（MHz）。
    给了 ``periods_um`` 时对每个像分别调用一次耦合核并**累加** V/J/A（周期像求和），
    自像（同一原子、零平移）不计。
    """
    pts = _as_points(positions)
    n = pts.shape[0]
    shifts = [(0.0, 0.0)]
    if periods_um is not None:
        periods = np.asarray(periods_um, float)
        if periods.shape != (2,) or np.any(periods <= 0):
            raise ValueError('periods_um 须是 (Lx,Ly) 两个正数')
        k = int(images)
        if k < 0:
            raise ValueError('images 不能为负')
        shifts = [(ix * periods[0], iy * periods[1])
                  for ix in range(-k, k + 1) for iy in range(-k, k + 1)]
    cutoff = None if cutoff_um is None else float(cutoff_um)
    pairs, dropped = [], 0
    for i in range(n):
        for j in range(i + 1, n):
            acc = np.zeros(3)
            touched = False
            for sx, sy in shifts:
                d = pts[j] - pts[i] + np.array([sx, sy, 0.0])
                r = float(np.linalg.norm(d))
                if r <= 0.0:
                    continue
                if cutoff is not None and r > cutoff:
                    dropped += 1
                    continue
                acc += np.asarray(_coupling_triplet(coupling(r)), float)
                touched = True
            if touched:
                pairs.append({'i': i, 'j': j,
                              'V': float(acc[0]), 'J': float(acc[1]),
                              'A': float(acc[2])})
    return {'atoms': [{'id': i, 'pos': [float(x) for x in p]} for i, p in enumerate(pts)],
            'pairs': pairs, 'terms': list(terms), 'source': str(source),
            'dropped': [], 'dropped_amp': {},
            'notes': {'n_dropped_by_cutoff': dropped,
                      'periods_um': None if periods_um is None else list(map(float, periods_um)),
                      'images': int(0 if periods_um is None else images)}}
