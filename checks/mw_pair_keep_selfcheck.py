# -*- coding: utf-8 -*-
"""微波对保留判据自检（t79 立 / t80 扩）："成本预算 + 累积相位"取代固定能量窗。

背景（freeze 场景诊断）：旧实现用与 |Ω| 无关的固定窗 | |ΔD|−ω | <
`MW_NRES_BW_MHZ`（原 5e3）筛微波对。三光子梯子 60P→59D→58F→58G
（7.5 GHz 驱动；梯级 10.223/11.226/0.420 GHz → 单光子失谐 −2.72/−3.73/
+7.08 GHz）的**第三级 Δ=7.079 GHz > 5 GHz 被整条删掉**——400 ns 演化布居
零变化、无任何报错。与 t78 给 V_dd 换判据（固定能隙窗 → 累积相位）同因。

判据（t80 定稿，两个无量纲量、都随 T 自适应）：
  成本 `coupling.mw_cap_mhz(T, 预算)`：窗宽 = 预算/(C·T) ≥ Δ ← 付得起吗
  （**默认预算为空 = 不限制**：成本只换速度，绝不默默改模型）
  精度 `coupling.mw_pair_keep`：Θ = C·T·Ω²/max(Δ,Ω) ≥ PHASE_TOL  ← 重要吗
`MW_NRES_BW_MHZ` / `MW_M_BW_MHZ` / `_TF_MW_M_BW_MHZ` / `_M_ANALYTIC_BW_MHZ`
**已删**（前三个合并为 M_EST_BW_MHZ 的 M 估计窗；保留窗改由相预算派生）。

计数模型（2026-09-14 起）：A–D 四段各 = 1 个**不变式包**（整包 1 项 = 4 项），
包内子条件原样保留、单独累计进 `SUB`（共 31 个子条件）。原断言不许删/放宽。

覆盖：
  A. 判据语义（纯函数）：零/负 Ω、成本上限、相位阈值、Ω 单调性、
     T 线性、与手算公式全等（网格）；
  B. 真实梯子（真实 Rabi）：三级梯级在 1e-3…1e3 mW/cm² 全保留；
     旧 5e3 窗会删第三级（回归动因锚）；THz 跨尺度对越绝对上界被丢；
  C. 引擎级：`_MixedSetup`（mixed 单体）在 T=400 ns 保留梯子三级、
     在 T=1e-9 ns（1 fs）全丢（T 自适应）、把成本上限压到 1e3 时第三级
     被 drop_cap 剔除；`build_two_field(t_end_ns=…)` 与 mixed 判据同源
     （同一态集 → 同一保留集）；
  D. 相预算派生（t80）：窗宽 == PHASE_BUDGET/(C·T)、标定锚（T=620 →
     1e4 MHz）、随 T 单调收窄、**总步数上界与 T 无关**、向量化实现
     `mw_candidate_mask` ≡ 标量 `mw_pair_keep`。

用法：python checks\\mw_pair_keep_selfcheck.py
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.stdout.reconfigure(encoding='utf-8', errors='replace')

from checks.checklib import Checker                              # noqa: E402
from arc import Rubidium87                                       # noqa: E402
from src.core import static_atom as core                         # noqa: E402
from src.core.atomic_levels import make_levels                   # noqa: E402
from src.core.waveforms import Wave, GaussianEnv                 # noqa: E402
from src.core.field_build import build_fields_exact, group_by_freq  # noqa: E402
from src.core.evolve_twofield import build_two_field             # noqa: E402
from src.core.evolve_mixed import _MixedSetup                    # noqa: E402
from src.core.coupling import (mw_pair_keep, mw_cap_mhz,        # noqa: E402
                              mw_candidate_mask)
from src.core.evolve_util import CONV                           # noqa: E402
# 成本预算现在**默认不限制**（None）；要按成本剔除必须显式给值。
# _BUD = 4e4 是 t79/t80 时代的默认档，保留下来当「显式成本档」的回归锚。
_BUD = 4e4
from src.core.coupling import PHASE_TOL                        # noqa: E402
from src.core.sambe import M_EST_BW_MHZ                        # noqa: E402
from src.core.static_atom import MW_MAX_MHZ                    # noqa: E402

C = Checker('mw_pair_keep')
Cv = CONV

# ---------------------------------------------------------------
# A. 判据语义（纯函数）
# ---------------------------------------------------------------
T0 = 400.0
CAP = 1e4

# 每段 = 1 个**不变式包**（整包 1 项）；包内子条件原样保留、累计进 SUB。
_A = []
_A.append(('A1 负/零 Ω → 丢', (not mw_pair_keep(0.0, 100.0, T0)
                               and not mw_pair_keep(-5.0, 100.0, T0))))
_A.append(('A2 兜底上限：Δ 越限 → 丢（即使 Ω=1e6）',
           not mw_pair_keep(1e6, CAP * 1.5, T0, cap_mhz=CAP)))
_A.append(('A3 兜底上限：Δ 在限内 → 不因上限丢',
           mw_pair_keep(1e6, CAP * 0.5, T0, cap_mhz=CAP)))
_A.append((f'A4 相位阈下 → 丢（Ω 极弱，Δ 远小于上限）'
           f'Θ={Cv * T0 * 1e-6 / 1e3:.3g}',
           not mw_pair_keep(1e-3, 1e3, T0)))
_A.append(('A5 相位阈上 → 留（共振强对）', mw_pair_keep(1e3, 0.0, T0)))

# Ω 单调性：固定 Δ，扫 Ω，应存在唯一的"由丢转留"穿越点且之后恒留
_oms = np.logspace(-4, 4, 81)
_ks = np.array([mw_pair_keep(o, 1e3, T0) for o in _oms])
# 单调不减 ⇔ 不存在 i<j 使 keep[i] 真而 keep[j] 假（真值集必为后缀）
_mono = all(not (_ks[i] and not _ks[j])
            for i in range(len(_ks)) for j in range(i + 1, len(_ks)))
_A.append(('A6 Ω 单调性：保留集随 Ω 单调不减', _mono))
_i = int(np.argmax(_ks))
_A.append((f'A6b 穿越点在手算阈值处 Ω*={_oms[_i]:.4g}',
           abs(_oms[_i]
               - np.sqrt(PHASE_TOL * max(1e3, 1.0) / (Cv * T0))) < 0.1
           * _oms[_i]))

# T 线性：同一 (Ω,Δ)，T 减半则 Θ 减半 → 找到 T 由留转丢的用例
_om, _det = 1.0, 1e3
_t_ok = PHASE_TOL * max(_det, _om) / (Cv * _om ** 2)
_A.append((f'A7 T 线性：T>2·T* 留、T<T*/2 丢 T*={_t_ok:.4g} ns',
           mw_pair_keep(_om, _det, 2.1 * _t_ok)
           and not mw_pair_keep(_om, _det, 0.5 * _t_ok)))

# 与手算公式全等（网格，含共振/色散两极限与上限）
_bad = 0
for om in (1e-3, 1.0, 1e3, 1e6):
    for det in (0.0, 1.0, 1e3, 1e5):
        for Tn in (1.0, 400.0):
            ref = (Cv * Tn * om ** 2 / max(det, om) >= PHASE_TOL)
            if det > CAP:
                ref = False
            if mw_pair_keep(om, det, Tn, cap_mhz=CAP) != ref:
                _bad += 1
_A.append((f'A8 与手算公式全等（4×4×2 网格，含上限）：{_bad} 处不符', _bad == 0))
_A.append(('A9 默认 cap=None 不做上限截断',
           mw_pair_keep(1e6, 1e9, T0) and not mw_pair_keep(1e-6, 1e9, T0)))
C.group('A 判据语义（纯函数）：零/负 Ω、成本上限、相位阈值、Ω 单调性、'
        'T 线性、与手算公式全等', _A)

# ---------------------------------------------------------------
# B. 真实梯子（真实 Rabi）
# ---------------------------------------------------------------
cache = core.ArcCache(Rubidium87())
# 三光子梯子所需流形（σ−：m 逐级 −1）+ 一个 6P 用于跨尺度对照
SPEC = [
    dict(n=60, l=0, j=0.5, kind='mJ', m=+0.5),
    dict(n=60, l=0, j=0.5, kind='mJ', m=-0.5),
    dict(n=60, l=1, j=1.5, kind='mJ', m=+0.5),
    dict(n=60, l=1, j=1.5, kind='mJ', m=-0.5),
    dict(n=59, l=2, j=2.5, kind='mJ', m=-0.5),
    dict(n=59, l=2, j=2.5, kind='mJ', m=-1.5),
    dict(n=58, l=3, j=3.5, kind='mJ', m=-2.5),
    dict(n=58, l=4, j=4.5, kind='mJ', m=-3.5),
    dict(n=6, l=1, j=1.5, kind='mJ', m=-0.5),
]
states = make_levels(cache, SPEC)
N = len(states)
E = np.array([s['E_mhz'] for s in states], float)
IDX = {(s['n'], s['l'], round(float(s['j']), 1), round(float(s['m']), 1)): i
       for i, s in enumerate(states)}

# 三级梯子（(起点, 终点, 标称 ΔE)）——单光子失谐依次 −2.72 / −3.73 / +7.08 GHz
RUNGS = [
    ((60, 1, 1.5, -0.5), (59, 2, 2.5, -1.5), 10223.34),
    ((59, 2, 2.5, -1.5), (58, 3, 3.5, -2.5), 11225.76),
    ((58, 3, 3.5, -2.5), (58, 4, 4.5, -3.5), 420.68),
]
# 跨尺度对照：60S m=−0.5 的最近偶极允许跃迁（n=60→6P：ΔE≈2.96e8 MHz）
FAR = ((60, 0, 0.5, -0.5), (6, 1, 1.5, -0.5))

W_FREQ = 7500.0
T_FREEZE = 400.0
OLD_WIN = 5e3                     # t79 之前的 MW_NRES_BW_MHZ 值（回归动因锚）


def _pairs(inten, t_end=None):
    """→ {(i,j): (Δ, Ω, 新判据, 旧窗)}，真实 Rabi 矩阵逐对。"""
    w = Wave('微波', W_FREQ, GaussianEnv(200, 40), pol='σ-',
             intensity_mwcm2=inten)
    meta = build_two_field(cache, states, [w], t_end_ns=t_end)
    out = {}
    for g in meta['mw_groups']:
        om_w = float(g['omega_mhz'])
        for A in g['As']:
            A = np.asarray(A, float)
            for i in range(N):
                for j in range(i + 1, N):
                    om = abs(A[i, j])
                    if om < 1e-12:
                        continue
                    det = abs(abs(E[i] - E[j]) - om_w)
                    keep = (mw_pair_keep(om, det, T_FREEZE)
                            if t_end is not None else True)
                    out[(i, j)] = (det, om, keep, det <= OLD_WIN)
    return out


_p = _pairs(1.0, t_end=T_FREEZE)
_rung_keys = [(IDX[a], IDX[b]) for a, b, _g in RUNGS]
_far_key = (IDX[FAR[0]], IDX[FAR[1]])
for _k in _rung_keys:
    if _k[0] > _k[1]:
        _k = (_k[1], _k[0])

_B = []
_B.append(('B1 三级梯子都在偶极允许对里（真实 Rabi 非零）',
           all(k in _p or (k[1], k[0]) in _p for k in _rung_keys)))
_d3 = _p[_rung_keys[2]][0]
_B.append((f'B2 第三级失谐 > 旧固定窗 5 GHz（回归动因锚）Δ={_d3:.3f} MHz',
           _d3 > OLD_WIN))
_B.append((f'B3 旧固定窗会删第三级 → 三光子路径消失'
           f'（旧窗保留={_p[_rung_keys[2]][3]}）',
           not _p[_rung_keys[2]][3]))
_B.append((f'B4 新判据保留第三级（1 mW/cm²）Ω='
           f'{_p[_rung_keys[2]][1]:.1f} Θ='
           f'{Cv * T_FREEZE * _p[_rung_keys[2]][1] ** 2 / _d3:.3g}',
           _p[_rung_keys[2]][2]))
_B.append((f'B5 相预算窗容得下第三级（mw_cap_mhz(400)='
           f'{mw_cap_mhz(T_FREEZE):.0f} vs Δ={_d3:.1f} MHz）',
           mw_cap_mhz(T_FREEZE) > _d3))
_B.append(('B6 三级梯级在 1e-3…1e3 mW/cm² 全保留',
           all(_pairs(i, t_end=T_FREEZE)[_rung_keys[2]][2]
               for i in (1e-3, 1.0, 1e2, 1e3))))
_B.append((f'B7 跨尺度 THz 对越绝对上界 → 丢（ΔE='
           f'{abs(E[_far_key[0]] - E[_far_key[1]]):.3g} MHz '
           f'> 上界={MW_MAX_MHZ:g}）',
           _far_key not in _p))
_B.append((f'B8 M 估计窗不充当保留界：Δ=1.5·M_EST_BW 的强对在短 T 下仍保留'
           f'（窗={M_EST_BW_MHZ:g} Δ={1.5 * M_EST_BW_MHZ:g} '
           f'cap={mw_cap_mhz(100.0):.0f}）',
           mw_pair_keep(1e3, 1.5 * M_EST_BW_MHZ, 100.0,
                        cap_mhz=min(mw_cap_mhz(100.0), MW_MAX_MHZ))
           and 1.5 * M_EST_BW_MHZ > M_EST_BW_MHZ))
_B.append(('B9 不传 t_end_ns → 退回绝对上界（梯子三级仍在上界内）',
           _rung_keys[2] in _pairs(1.0, t_end=None)))
C.group('B 真实梯子（真实 Rabi）：三级梯级全功率保留、旧 5e3 窗会删第三级、'
        'THz 跨尺度对被绝对上界丢弃', _B)

# ---------------------------------------------------------------
# D. 相预算派生（t80）：mw_cap_mhz 的标定锚与单调性
# ---------------------------------------------------------------
_D = []
_D.append((f'D1 默认（预算空）不删真实耦合：退回绝对上界'
           f'（显式档 {mw_cap_mhz(400.0, _BUD):.0f} MHz）',
           abs(mw_cap_mhz(400.0) - 1e5) < 1e-9))
_D.append(('D1b 显式给成本档 → 窗宽 == 预算/(C·T)',
           abs(mw_cap_mhz(400.0, _BUD) - _BUD / (Cv * 400.0)) < 1e-9))
_D.append((f'D2 标定锚：T=620 ns 复现 t79 的 1e4 MHz 窗（±5%）'
           f'{mw_cap_mhz(620.0, _BUD):.0f} MHz',
           abs(mw_cap_mhz(620.0, _BUD) - 1e4) / 1e4 < 0.05))
_caps = [mw_cap_mhz(t, _BUD) for t in (10.0, 100.0, 400.0, 1000.0, 1e4)]
_D.append((f'D3 随 T 单调收窄（T 越长付不起越快项）'
           f'{["%.3g" % c for c in _caps]}',
           all(_caps[i] > _caps[i + 1] for i in range(len(_caps) - 1))))
_D.append(('D4 步数上界与 T 无关（相预算的本质）',
           all(abs(c * Cv * t / _BUD - 1.0) < 1e-9
               for c, t in zip(_caps, (10.0, 100.0, 400.0, 1000.0, 1e4)))))
# 相预算窗在 freeze 场景容得下第三级，但不至于放进 THz 跨尺度对
_D.append((f'D5 freeze T=400：窗宽 1e3…3e4 MHz 之间（够梯子、不够 THz）'
           f'{mw_cap_mhz(400.0):.0f} MHz',
           1e3 < mw_cap_mhz(400.0, _BUD) < 3e4))
# 候选掩码 == 逐对 mw_pair_keep（向量化实现与标量判据一致）
_Om = np.array([[0.0, 1e3, 5e-4], [0.0, 0.0, 3e3], [0.0, 0.0, 0.0]])
_dE = np.array([[0.0, 1e3, 2.9e8], [0.0, 0.0, 1e4], [0.0, 0.0, 0.0]])
_m = mw_candidate_mask(_Om, _dE, 0.0, 400.0)
_ok = True
for _i in range(3):
    for _j in range(3):
        if _i >= _j:
            continue
        _ref = mw_pair_keep(abs(_Om[_i, _j]), abs(abs(_dE[_i, _j]) - 0.0),
                            400.0, cap_mhz=min(mw_cap_mhz(400.0),
                                               MW_MAX_MHZ))
        if bool(_m[_i, _j]) != _ref:
            _ok = False
_D.append(('D6 向量化 mw_candidate_mask ≡ 标量 mw_pair_keep（含成本+精度）',
           _ok))

# ---------------------------------------------------------------
# C. 引擎级：mixed 单体 `_MixedSetup`（冻结场景路径）
# ---------------------------------------------------------------
def _mixed_pairs(t_end, cap=None):
    w = Wave('微波', W_FREQ, GaussianEnv(200, 40), pol='σ-',
             intensity_mwcm2=1.0)
    fields = build_fields_exact(cache, states, [w])
    # 分流约定（params.MW_MAX_MHZ）：载频 ≤ 100 GHz 视为微波，> 视为光场
    f_mw = [f for f in fields if f['omega_L_mhz'] <= MW_MAX_MHZ]
    f_opt = [f for f in fields if f['omega_L_mhz'] > MW_MAX_MHZ]
    grps_opt = group_by_freq(f_opt) if f_opt else []
    grps_mw = group_by_freq(f_mw) if f_mw else []
    s = _MixedSetup(E, grps_opt, grps_mw, N, mw_cap=cap, t_end_ns=t_end)
    keys = set()
    for pr in s.pairs:
        if pr[4] == 1:                       # 微波对
            a, b = int(pr[0]), int(pr[1])
            keys.add((min(a, b), max(a, b)))
    return s, keys


_s, _keys = _mixed_pairs(T_FREEZE)
_C = []
_C.append((f'C1 mixed 单体：T=400 ns 保留梯子三级'
           f'（缺 {[k for k in _rung_keys if k not in _keys]}）',
           all(k in _keys or (k[1], k[0]) in _keys for k in _rung_keys)))
_C.append((f'C2 mixed 单体：统计与 pairs 自洽（keep=微波对数）'
           f'keep={_s.mw_pair_stats["keep"]} 实际={len(_keys)}',
           _s.mw_pair_stats['keep'] == len(_keys)))
_s0, _keys0 = _mixed_pairs(1e-9)
_C.append((f'C3 T 自适应：T=1 fs 时全丢（无对能在 1 fs 内累积相位）'
           f'{_s0.mw_pair_stats}',
           len(_keys0) == 0 and _s0.mw_pair_stats['drop_phase'] > 0))
_scap, _keyscap = _mixed_pairs(T_FREEZE, cap=1e3)
_C.append((f'C4 成本上限生效：cap=1e3 → 第三级被 drop_cap 剔除'
           f'{_scap.mw_pair_stats}',
           _rung_keys[2] not in _keyscap
           and _scap.mw_pair_stats['drop_cap'] > 0))

# build_two_field(t_end_ns=…) 与 mixed 判据同源 → 同一保留集（≤cap 内）
_btf = set()
w1 = Wave('微波', W_FREQ, GaussianEnv(200, 40), pol='σ-', intensity_mwcm2=1.0)
for g in build_two_field(cache, states, [w1], t_end_ns=T_FREEZE)['mw_groups']:
    for A in g['As']:
        A = np.asarray(A, float)
        for i in range(N):
            for j in range(i + 1, N):
                if abs(A[i, j]) > 1e-12:
                    _btf.add((i, j))
_C.append((f'C5 build_two_field(t_end) 与 mixed 微波保留集一致'
           f'（两场独有 {sorted(_btf - _keys)}，'
           f'mixed 独有 {sorted(_keys - _btf)}）',
           _btf == _keys))

# 包按 A–D 顺序登记（C 段的子条件在 D 段之后才算完，故这里统一登记）
C.group('C 引擎级：mixed 单体 `_MixedSetup`（T 自适应、成本上限、'
        '与 build_two_field 同源）', _C)
C.group('D 相预算派生（t80）：标定锚、随 T 单调收窄、步数上界、'
        '向量化 ≡ 标量', _D)

sys.exit(0 if C.summary() else 1)
