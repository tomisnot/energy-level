# -*- coding: utf-8 -*-
"""多体 HamiltonianPlan：装配层的统一、可序列化中间表示。

Plan 保留物理 calib，同时决定 full / blockade / PXP 的基空间与投影规则。
后端只消费 ``compile_sparse(plan)`` 或 plan 的 basis/transition 信息；它们
不再各自决定是否删除非法 blockade 状态。
"""
import numpy as np
from .basis import independent_states, validate_edges

__all__ = ['build_plan', 'compile_sparse', 'plan_summary']


def _constraint_dict(constraint):
    if constraint is None:
        return dict(kind='none', mode='full', threshold_mhz=0.0, edges=None)
    if isinstance(constraint, str):
        constraint = dict(kind=constraint)
    if not isinstance(constraint, dict):
        raise ValueError('manybody.constraint 须为 dict 或字符串')
    kind = str(constraint.get('kind', 'none')).lower()
    mode = str(constraint.get('mode', 'project' if kind in ('pxp', 'blockade') else 'full')).lower()
    if kind not in ('none', 'blockade', 'pxp'):
        raise ValueError('constraint.kind 只支持 none/blockade/pxp')
    if mode not in ('full', 'project'):
        raise ValueError('constraint.mode 只支持 full/project')
    threshold = float(constraint.get('threshold_mhz', 0.0) or 0.0)
    if not np.isfinite(threshold) or threshold < 0:
        raise ValueError('constraint.threshold_mhz 必须是有限非负数')
    return dict(kind=kind, mode=mode,
                threshold_mhz=threshold,
                edges=constraint.get('edges'))


def _edges(calib, c):
    pairs = calib.get('pairs') or []
    explicit = c.get('edges')
    if explicit is not None and not isinstance(explicit, str):
        return validate_edges(len(calib['atoms']), explicit)
    if explicit not in (None, 'auto'):
        raise ValueError('constraint.edges 只支持 auto 或整数边列表')
    threshold = float(c.get('threshold_mhz', 0.0))
    if threshold <= 0:
        raise ValueError('blockade/PXP project 模式需要 threshold_mhz>0 或显式 edges')
    out = set()
    for p in pairs:
        if abs(complex(p.get('V', 0.0))) >= threshold:
            out.add(tuple(sorted((int(p['i']), int(p['j'])))))
    return sorted(out)


def build_plan(calib, Om=None, delta=None, gamma=None, terms=None, constraint=None,
               max_states=1 << 20):
    """从物理 calib 装配统一 plan；不生成稠密矩阵。"""
    atoms = calib.get('atoms') or []
    n = len(atoms)
    Om = np.zeros(n, complex) if Om is None else np.asarray(Om, complex)
    delta = np.zeros(n) if delta is None else np.asarray(delta, float)
    gamma = np.zeros(n) if gamma is None else np.asarray(gamma, float)
    if any(x.shape != (n,) or not np.isfinite(x).all() for x in (Om, delta, gamma)):
        raise ValueError('HamiltonianPlan 的 Om/delta/gamma 须为有限、长度为原子数的数组')
    c = _constraint_dict(constraint)
    projected = c['kind'] in ('blockade', 'pxp') and c['mode'] == 'project'
    edges = _edges(calib, c) if projected else []
    states = independent_states(n, edges, max_states) if projected else None
    return dict(version=1, kind='HamiltonianPlan', constraint=c,
                n_atoms=n, full_dim=1 << n, dim=(len(states) if states is not None else 1 << n),
                projected=projected, blockade_edges=[list(e) for e in edges],
                states=states, Om=Om.tolist(), delta=delta.tolist(), gamma=gamma.tolist(),
                terms=list(calib.get('terms') if terms is None else terms),
                calib=calib)


def compile_sparse(plan):
    """Compile the same physical terms in the full or declared reduced basis."""
    from .assemble import build_H_sparse
    return build_H_sparse(plan['calib'], Om=plan['Om'], delta=plan['delta'],
                          gamma=plan['gamma'], terms=plan['terms'],
                          basis_indices=plan.get('states'))


def plan_summary(plan):
    return dict(kind=plan['constraint']['kind'], mode=plan['constraint']['mode'],
                n_atoms=plan['n_atoms'], full_dim=plan['full_dim'], dim=plan['dim'],
                reduction=plan['full_dim'] - plan['dim'],
                blockade_edges=plan['blockade_edges'])
