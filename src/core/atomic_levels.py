# -*- coding: utf-8 -*-
"""能级构造（t19：evolve.py 拆分）——make_levels / gaussian / square / step。"""
import numpy as np

from . import static_atom as core


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

    **透传字段（2026-09-22，P1-6）**：specs 里若带 `qstr` / `dc_shift_mhz` /
    `group_label` / `color`，原样带进 states——它们来自 `static_atom.base_sublevels`
    （静态页与动态基组的唯一基准），是"静态量能对回态"的必需身份信息
    （`qstr` 是静态页行身份键的一半；`dc_shift_mhz` 是 DC-Stark 位移）。
    不带这些键时行为与改前**逐位相同**（键不出现）。
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
            st = dict(
                n=n, l=l, j=j, kind='F', F=F, mF=mF, mj=None, m=mF,
                amps=s.get('amps'), E_mhz=E, label=label, index=i)
        else:
            m = float(s['m'])
            label = s.get('label',
                          f'{core.auto_label(n, l, j)} m_J={m:+.1f}')
            st = dict(
                n=n, l=l, j=j, kind='mJ', F=None, mF=None, mj=m, m=m,
                amps=None, E_mhz=E, label=label, index=i)
        for k in ('qstr', 'dc_shift_mhz', 'group_label', 'color'):
            if k in s:
                st[k] = s[k]
        states.append(st)
    return states


# ---------------- 子能级唯一身份键（跨模块共用） ----------------
# 2026-09-15：从 src/ui/ui_waves.py **搬到这里**（无 Qt）。原因：并行扫描的
# runner 跑在独立子进程里，要按身份键解析初态；若继续放 ui_waves，子进程
# 就会被拖进 PyQt5 导入。ui_waves 保留同名再导出，既有调用方零改动。
def state_key(s):
    """子能级身份键：(kind, n, l, j, m, energy_rounded)。

    - kind='F' 时 m = m_F；kind='mJ' 时 m = m_J
    - 末位能量（MHz，6 位小数）使键跨 B 稳定、与 F 标号无关、天然无重复；
      旧存档（无能量位）经 find_state_index 前缀回退兼容。
    """
    en = round(float(s['E_mhz']), 6)
    if s.get('kind') == 'F':
        return ('F', int(s['n']), int(s['l']), float(s['j']),
                float(s['mF']), en)
    return ('mJ', int(s['n']), int(s['l']), float(s['j']),
            float(s['m']), en)


def find_state_index(states, spec):
    """states 中匹配身份键 spec 的第一个索引（0 基）；找不到返回 -1。

    匹配顺序：① 完整键（含能量）精确；② 前 5 元 (kind,n,l,j,m) 精确
    （兼容旧存档无能量位 / 能量微漂）；③ (kind,n,l,j,m) 组内能量最近
    （B 漂移后 F 标号/能量变化）。
    """
    if not spec:
        return -1
    spec = tuple(spec)
    keys = [state_key(s) for s in states]
    # ① 完整键精确（含能量）
    for i, k in enumerate(keys):
        if k == spec:
            return i
    # ② 前 5 元精确（旧存档 / 同 (n,l,j,m) 能量微漂）
    if len(spec) >= 5:
        base = spec[:5]
        for i, k in enumerate(keys):
            if k[:5] == base:
                return i
    # ③ (kind,n,l,j,m) 组内能量最近
    base = spec[:5]
    best, bd = -1, None
    ref_e = spec[5] if len(spec) > 5 else None
    for i, k in enumerate(keys):
        if k[:5] != base:
            continue
        d = abs(k[5] - ref_e) if ref_e is not None else 0.0
        if bd is None or d < bd:
            best, bd = i, d
    return best


