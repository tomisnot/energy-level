# -*- coding: utf-8 -*-
"""能级构造（t19：evolve.py 拆分）——make_levels / gaussian / square / step。"""
import numpy as np

from . import core


def gaussian(t0, sigma):
    """高斯包络 exp(−(t−t0)²/2σ²)；返回 callable(t_ns array)->array。"""
    def wf(t):
        t = np.asarray(t, float)
        return np.exp(-((t - t0) ** 2) / (2.0 * sigma ** 2))
    return wf




def square(t0, t1):
    """方波包络（t0≤t≤t1 为 1，其余 0）。"""
    def wf(t):
        t = np.asarray(t, float)
        return ((t >= t0) & (t <= t1)).astype(float)
    return wf




def step(t0):
    """阶跃包络（t≥t0 为 1）。"""
    def wf(t):
        t = np.asarray(t, float)
        return (t >= t0).astype(float)
    return wf




def make_levels(cache, specs):
    """能级规格列表 → states（支持超精细基组）。

    specs 每项 dict(n,l,j) + 子能级标识：
      - kind='mJ', m           → 磁子能级（里德堡态）
      - kind='F', F, mF, amps[, E_hf]
                               → 超精细态（amps = {m_J: 复振幅}，来自
                                 core.get_hyperfine_levels；E_hf 为其相对能）
      - 缺省 kind → 按 mJ 处理（向后兼容）
    E_mhz = 组中心（可显式覆盖） + E_hf。
    返回 states: [{n,l,j,kind,F,mF,mj,m,amps,E_mhz,label,index}]。
    """
    states = []
    for i, s in enumerate(specs):
        n, l, j = int(s['n']), int(s['l']), float(s['j'])
        E0 = float(s.get('E_mhz', cache.E_mhz(n, l, j)))
        E_hf = float(s.get('E_hf', 0.0))
        E = E0 + E_hf
        if s.get('kind') == 'F':
            F, mF = float(s['F']), float(s['mF'])
            label = s.get(
                'label',
                f'{core.auto_label(n, l, j)} F={F:.1f}, m_F={mF:+.1f}')
            states.append(dict(
                n=n, l=l, j=j, kind='F', F=F, mF=mF, mj=None, m=mF,
                amps=s.get('amps'), E_mhz=E, label=label, index=i))
        else:
            m = float(s['m'])
            label = s.get('label',
                          f'{core.auto_label(n, l, j)} m_J={m:+.1f}')
            states.append(dict(
                n=n, l=l, j=j, kind='mJ', F=None, mF=None, mj=m, m=m,
                amps=None, E_mhz=E, label=label, index=i))
    return states


