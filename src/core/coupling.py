# -*- coding: utf-8 -*-
"""coupling.py（t63 第 1 层统一）：RHS 打包公共层——单/两体共用。

从 evolve_mixed_pair._pack_rhs + evolve_mixed._MixedSetup 的波构建抽出泛化：
- pack_rhs(pairs, t_end, n_tab)：pairs 7 元组 → njit 数组（分组键
  (is_mw, w, id(env))——t62 用例 1 教训：光场/微波即使同一 env 对象也必须分属
  不同组——is_mw 决定 cos 载波；env 表预采样 + 线性插值）
- 返回 dict: iis, jjs, vals, dDs, is_mw, wcos, n_pairs_w, env_tab, env_dt
"""
import numpy as np


def pack_rhs(pairs, t_end, n_tab=20001):
    """pairs → numba 数组（按 (is_mw, w, id(env)) 分组 + env 表预采样）。

    pairs 元素：(a, b, v, dD, is_mw, w, env)——分组键含 is_mw/w（t62 用例 1
    教训：光场/微波即使 env 同一对象也须分属不同组——is_mw 决定 cos 载波）。
    """
    grp = {}
    order = []
    for p in pairs:
        a, b, v, dD, is_mw, w, env = p
        key = (int(is_mw), float(w), id(env))
        if key not in grp:
            grp[key] = dict(is_mw=is_mw, w=w, env=env, items=[])
            order.append(key)
        grp[key]['items'].append((a, b, v, dD))
    iis, jjs, vals, dDs = [], [], [], []
    is_mw_a, wcos_a, n_ppw = [], [], []
    env_rows = []
    for key in order:
        g = grp[key]
        for (a, b, v, dD) in g['items']:
            iis.append(a); jjs.append(b); vals.append(v); dDs.append(dD)
        n_ppw.append(len(g['items']))
        is_mw_a.append(1 if g['is_mw'] else 0)
        wcos_a.append(g['w'])
        ts = np.linspace(0.0, t_end, n_tab)
        env_rows.append(np.asarray(g['env'](ts), float))
    env_tab = np.stack(env_rows) if env_rows else np.zeros((0, n_tab))
    env_dt = t_end / (n_tab - 1) if n_tab > 1 else 1.0
    return dict(iis=np.asarray(iis, np.int64),
                jjs=np.asarray(jjs, np.int64),
                vals=np.asarray(vals, np.complex128),
                dDs=np.asarray(dDs, np.float64),
                is_mw=np.asarray(is_mw_a, np.int64),
                wcos=np.asarray(wcos_a, np.float64),
                n_pairs_w=np.asarray(n_ppw, np.int64),
                env_tab=env_tab, env_dt=env_dt)