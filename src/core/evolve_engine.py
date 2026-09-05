# -*- coding: utf-8 -*-
"""evolve_engine.py（t63 第 3 层）：统一接口——n_body=1|2 调度。

GUI/上层零改动目标：evolve_engine(states, fields, psi0, ..., method, n_body)
- n_body=1：单原子（fields = 单场列表）→ evolve_mixed / evolve_sambe / evolve_hard
- n_body=2：两体（fields_A/fields_B + psi0_A/psi0_B）→ evolve_mixed_pair /
  evolve_sambe_pair / evolve_hard_pair
内部路由到现有引擎（第 2 层的核心收敛为单原子核心 + tensor_fields 提升在后续
轮次落地——本模块先统一入口，行为委托现有实现保证重构安全）。
"""
import numpy as np


def evolve_engine(method, states, fields=None, fields_A=None, fields_B=None,
                  psi0=0, psi0_A=0, psi0_B=0, t_grid=None, n_body=1,
                  **kwargs):
    """统一引擎入口。method: 'mixed' | 'sambe' | 'hard' | 'ip4'。

    n_body=1：用 fields + psi0；n_body=2：用 fields_A/B + psi0_A/B。
    其余 kwargs（frozen_h_ns/per_freq_M/vdd_matrix/...）透传。
    """
    if t_grid is None:
        raise ValueError('t_grid 必填')
    if n_body == 1:
        if method == 'mixed':
            from .evolve_mixed import evolve_mixed
            return evolve_mixed(states, fields, psi0, t_grid, **kwargs)
        if method == 'sambe':
            from .sambe import evolve_sambe
            return evolve_sambe(states, fields, psi0, t_grid, **kwargs)
        if method == 'hard':
            from .hard import evolve_hard
            return evolve_hard(states, fields, psi0, t_grid, **kwargs)
        raise ValueError(f'未知单原子方法 {method}')
    if n_body == 2:
        if method == 'mixed':
            from .evolve_mixed_pair import evolve_mixed_pair
            return evolve_mixed_pair(states, fields_A, fields_B, psi0_A,
                                     psi0_B, t_grid, **kwargs)
        if method == 'sambe':
            from .sambe import evolve_sambe_pair
            return evolve_sambe_pair(states, fields_A, fields_B, psi0_A,
                                     psi0_B, t_grid, **kwargs)
        if method == 'hard':
            from .hard_pair import evolve_hard_pair
            return evolve_hard_pair(states, fields_A, fields_B, psi0_A,
                                    psi0_B, t_grid, **kwargs)
        raise ValueError(f'未知两体方法 {method}')
    raise ValueError(f'n_body 必须 1 或 2（got {n_body}）')
