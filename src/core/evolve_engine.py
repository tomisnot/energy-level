# -*- coding: utf-8 -*-
"""evolve_engine.py：统一接口——method × n_body 调度。

- n_body=1：单原子（fields = 单场列表）→ evolve_mixed / sambe.evolve_sambe /
  hard.evolve_hard
- n_body=2：两体（fields_A/fields_B + psi0_A/psi0_B）→ evolve_mixed_pair /
  sambe.evolve_sambe_pair / hard.evolve_hard_pair
  两体还可用 **psi0_pair**（两体纠缠初态 spec `['pair_sup', […]]`）取代
  逐原子直积——交换结果 |60P_A,60S_B⟩±|60S_A,60P_B⟩ 必须走这条。
"""
import numpy as np


def evolve_engine(method, states, fields=None, fields_A=None, fields_B=None,
                  psi0=0, psi0_A=0, psi0_B=0, t_grid=None, n_body=1,
                  psi0_pair=None, **kwargs):
    """统一引擎入口。method: 'mixed' | 'sambe' | 'hard' | 'ip4'。

    n_body=1：用 fields + psi0；n_body=2：用 fields_A/B + psi0_A/B
    （psi0_pair 非 None 时**取代** psi0_A/B）。
    其余 kwargs（frozen_h_ns/per_freq_M/vdd_matrix/...）透传。
    """
    if t_grid is None:
        raise ValueError('t_grid 必填')
    if psi0_pair is not None:
        if n_body != 2:
            raise ValueError('psi0_pair（两体纠缠初态）只在 n_body=2 下有效')
        kwargs['psi0_pair'] = psi0_pair
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
            from .evolve_mixed import evolve_mixed_pair
            return evolve_mixed_pair(states, fields_A, fields_B, psi0_A,
                                     psi0_B, t_grid, **kwargs)
        if method == 'sambe':
            from .sambe import evolve_sambe_pair
            return evolve_sambe_pair(states, fields_A, fields_B, psi0_A,
                                     psi0_B, t_grid, **kwargs)
        if method == 'hard':
            from .hard import evolve_hard_pair
            return evolve_hard_pair(states, fields_A, fields_B, psi0_A,
                                    psi0_B, t_grid, **kwargs)
        raise ValueError(f'未知两体方法 {method}')
    raise ValueError(f'n_body 必须 1 或 2（got {n_body}）')
