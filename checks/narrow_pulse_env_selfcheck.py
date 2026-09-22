# -*- coding: utf-8 -*-
"""窄脉冲包络自检（2026-09-22，手册 §8.1 / 工单 P1-1）：**恒常判断不得靠固定采样密度**。

## 事故（本判据要钉死的东西）

`sambe._env_all_constant` / `sambe._pair_propagate` / `field_build.env_samples`
原先都用 `np.ptp(wf(np.linspace(0,T,64))) > 1e-9` 判"包络是否恒常"。**固定 64 点
对窄脉冲是盲的**：σ ≲ T/800 时整个脉冲落在采样点之间。实测 T=620 ns / σ=0.5 ns：
64 点 ptp=9.3e-22（< 1e-9）而**真实 ptp=1.0**。

后果是**两个引擎给出方向相反的错答案，且都静默**：
  · sambe 判恒常 → 走"一次对角化覆盖全时刻"快路径、用**中点**包络
    ⇒ 整窗按**峰值 CW** 算（`error_report['n_segments']=1`，看起来完全正常）；
  · ip4  判恒常 → 取 `env_samples` 的**均值**（≈0）⇒ 场**完全消失**（等于零场）。
同一份输入，一个引擎当满场、一个当无场——连"哪个更可信"都无从判断。

## 判据（致死性说明）

本判据对**同一个窄脉冲**取四个参照并断言两两可分：
    narrow ≠ CW（峰值常数包络）   ← 抓 sambe 的错向
    narrow ≠ ZERO（零场）         ← 抓 ip4 的错向
    narrow(sambe) ≈ narrow(ip4)   ← 两个独立引擎必须给同一个物理答案
    narrow ≈ 解析 Rabi 面积 (Θ/2)² ← 与闭式解对齐（Θ = 2π·1e-3·Ω·σ·√(2π)）
    sambe 的 n_segments > 1        ← 必须真的走自适应分段，而不是一次对角化
另外钉住 `env_span` 的**解析**答案与 `env_samples` 的恒常位，并钉住
`evolve_reference`（全库"权威锚"）——它原先也漏掉窄脉冲（max P_e 恒 0）。

**致死性**：把 `_env_all_constant` / `env_samples` 改回 64 点 `np.ptp`，则
`narrow == CW`（sambe）与 `narrow == ZERO`（ip4）两条同时变红；把
`_seg_decomp` 改回"只看中点"，sambe 侧的解析对齐会变红。
（验证过：修复前 sambe 与 CW 逐位相同、ip4 与零场逐位相同。）

用法：python checks\\narrow_pulse_env_selfcheck.py
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.stdout.reconfigure(encoding='utf-8', errors='replace')

from checks.checklib import Checker                            # noqa: E402
from arc import Rubidium87                                     # noqa: E402
from src.core import static_atom as core                       # noqa: E402
from src.core.atomic_levels import make_levels                 # noqa: E402
from src.core.evolve_mixed import evolve_mixed                 # noqa: E402
from src.core.evolve_reference import evolve_reference         # noqa: E402
from src.core.field_build import (build_fields_exact, env_span,  # noqa: E402
                                  env_is_const, env_samples,
                                  env_feature_ns, feature_max_step)
from src.core.sambe import evolve_sambe                        # noqa: E402
from src.core.waveforms import Wave, GaussianEnv               # noqa: E402

C = Checker('narrow_pulse_env')
cache = core.ArcCache(Rubidium87())
CONV = 2 * np.pi * 1e-3

# ---------------- 场景：T=620 ns 窗口里一个 σ=T/1000 的高斯脉冲 ----------------
T = 620.0
SIG = T / 1000.0                      # 0.62 ns
OM = 1.0                              # Rabi (MHz，周频)；解析面积用
TG = np.linspace(0.0, T, 3)

states = make_levels(cache, [
    dict(n=60, l=0, j=0.5, kind='mJ', m=0.5, E_mhz=0.0, label='g'),
    dict(n=60, l=1, j=0.5, kind='mJ', m=0.5, E_mhz=1000.0, label='e')])
A = np.zeros((2, 2)); A[0, 1] = A[1, 0] = OM


def _fld(wf, tag):
    return {'kind': 'exact', 'A': A, 'omega_L_mhz': 1000.0,
            'waveform': wf, 'label': tag}


def _gauss(t):
    return np.exp(-((np.asarray(t, float) - 0.5 * T) ** 2) / (2.0 * SIG ** 2))


_ones = lambda t: np.ones_like(np.asarray(t, float))           # noqa: E731
_zeros = lambda t: np.zeros_like(np.asarray(t, float))         # noqa: E731

f_narrow = [_fld(_gauss, 'narrow')]
f_cw = [_fld(_ones, 'cw')]
f_zero = [_fld(_zeros, 'zero')]

# ---------------- A. 恒常判断本身（解析 vs 采样） ----------------
lo, hi = env_span(_gauss, 0.0, T)
_, is_const = env_samples(_gauss, T)
# 64 点采样看到的 ptp（= 修复前判据的输入）——用来说明"为什么旧判据会错"
_old_ptp = float(np.ptp(_gauss(np.linspace(0.0, T, 64))))
C.group('A 恒常判断：解析答案正确，且旧 64 点判据确实会误判', [
    ('env_span 给出真实变差 (0,1)', abs(lo) < 1e-12 and abs(hi - 1.0) < 1e-12),
    ('env_is_const=False', env_is_const(_gauss, 0.0, T) is False),
    ('env_samples 恒常位=False（原为 True）', is_const is False),
    ('旧 64 点判据确实看不见该脉冲（ptp<1e-9）', _old_ptp < 1e-9),
    ('真实变差远大于门限', (hi - lo) > 1e-3),
    ('Wave 包络走解析特征时长（=σ）',
     abs((env_feature_ns(Wave('w', 1000.0, GaussianEnv(0.5 * T, SIG),
                              pol='π').waveform) or -1) - SIG) < 1e-12),
    ('深混叠区给出步长保护（非 None）',
     feature_max_step([_gauss], T) is not None),
    ('宽脉冲（σ=60/T=400）不给保护（no-op）',
     feature_max_step([lambda t: np.exp(-((np.asarray(t, float) - 200.0) ** 2)
                                        / (2 * 60.0 ** 2))], 400.0) is None),
])

# ---------------- B. 三引擎的窄脉冲答案 ----------------
r_narrow = evolve_sambe(states, f_narrow, 0, TG, tol=1e-8)
r_cw = evolve_sambe(states, f_cw, 0, TG, tol=1e-8)
r_zero = evolve_sambe(states, f_zero, 0, TG, tol=1e-8)
p_narrow = float(r_narrow['pops'][-1, 1])
p_cw = float(r_cw['pops'][-1, 1])
p_zero = float(r_zero['pops'][-1, 1])

r_mixed = evolve_mixed(states, f_narrow, 0, TG)
p_mixed = float(r_mixed['pops'][-1, 1])

# ip4（RWA 路径，走 field() 格式）
from src.core.field_build import field                         # noqa: E402
from src.core.evolve_rwa import evolve as rwa_evolve           # noqa: E402
f_ip4_n = [field(0, 1, omega0_mhz=OM, waveform=_gauss)]
f_ip4_z = [field(0, 1, omega0_mhz=OM, waveform=_zeros)]
p_ip4 = float(rwa_evolve(states, f_ip4_n, 0, TG, method='ip4')['pops'][-1, 1])
p_ip4_zero = float(rwa_evolve(states, f_ip4_z, 0, TG, method='ip4')
                   ['pops'][-1, 1])

# 解析：Θ = 2π·1e-3·Ω·σ·√(2π)（高斯脉冲面积）⇒ P_e = (Θ/2)²
theta = CONV * OM * SIG * np.sqrt(2.0 * np.pi)
p_ana = (theta / 2.0) ** 2

C.group('B 窄脉冲：sambe 不得等于 CW、ip4 不得等于零场', [
    (f'sambe 有动力学（P_e={p_narrow:.3e} > 1e-9）', p_narrow > 1e-9),
    (f'sambe ≠ CW（{p_narrow:.6e} vs {p_cw:.6e}）',
     abs(p_narrow - p_cw) > 1e-3),
    (f'ip4 ≠ 零场（{p_ip4:.6e} vs {p_ip4_zero:.6e}）',
     abs(p_ip4 - p_ip4_zero) > 1e-9),
    ('零场参照确实为 0（对照有效）', abs(p_zero) < 1e-12),
    (f'CW 参照确实满转移（{p_cw:.4f}）', p_cw > 0.5),
    (f'sambe 走自适应分段（n_segments={r_narrow["error_report"]["n_segments"]}>1）',
     int(r_narrow['error_report']['n_segments']) > 1),
    (f'sambe 未触底（dt_min_hit=False）',
     r_narrow['error_report'].get('dt_min_hit') is False),
])

C.group('C 三个独立引擎给同一个物理答案（且与解析面积对齐）', [
    (f'sambe ≈ ip4（{p_narrow:.6e} vs {p_ip4:.6e}）',
     abs(p_narrow - p_ip4) <= 1e-8 + 1e-2 * abs(p_ip4)),
    (f'sambe ≈ mixed（{p_narrow:.6e} vs {p_mixed:.6e}）',
     abs(p_narrow - p_mixed) <= 1e-8 + 1e-2 * abs(p_mixed)),
    (f'sambe ≈ 解析 (Θ/2)²={p_ana:.6e}（差 {abs(p_narrow - p_ana):.2e}）',
     abs(p_narrow - p_ana) <= 1e-10 + 0.02 * p_ana),
])

# ---------------- D. 权威锚也必须看得见窄脉冲 ----------------
ref = evolve_reference(states, f_narrow, 0, np.linspace(0.0, T, 621),
                       rtol=1e-10, atol=1e-12)
p_ref = float(ref['pops'][:, 1].max())
C.group('D evolve_reference（权威锚）不得漏掉窄脉冲', [
    (f'reference 有动力学（max P_e={p_ref:.3e} > 1e-9）', p_ref > 1e-9),
    (f'reference ≈ 解析（差 {abs(p_ref - p_ana):.2e}）',
     abs(p_ref - p_ana) <= 1e-10 + 0.02 * p_ana),
])

# ---------------- E. 宽脉冲回归：修复不得改变既有场景的判断 ----------------
wide = lambda t: np.exp(-((np.asarray(t, float) - 200.0) ** 2)   # noqa: E731
                        / (2 * 60.0 ** 2))
C.group('E 宽脉冲回归（修复对既有场景 no-op）', [
    ('σ=60/T=400 仍判非常数', env_is_const(wide, 0.0, 400.0) is False),
    ('常数包络仍判常数',
     env_is_const(_ones, 0.0, 400.0) is True),
    ('零包络仍判常数', env_is_const(_zeros, 0.0, 400.0) is True),
    ('超宽高斯（σ=1e9）仍判常数',
     env_is_const(lambda t: np.exp(-(np.asarray(t, float) ** 2) / 2e18),
                  0.0, 120.0) is True),
    ('宽高斯不给步长保护', feature_max_step([wide], 400.0) is None),
    ('零包络不给步长保护', feature_max_step([_zeros], 400.0) is None),
])

# ---------------- F. 包络 span 的解析性（形状无关的正确性） ----------------
from src.core.waveforms import SquareEnv, FileEnv               # noqa: E402
sq = SquareEnv(100.0, 20.0, height=1.0, fall_start_ns=300.0)
# 段 [50,150] 横跨升沿：中点在升沿里 ⇒ 旧 _seg_decomp 会走 K(mid)；
# 段 [50,95] 完全在脉冲前（中点=72.5，env 恰为 0）⇒ 必须判"整段为零"
lo_sq, hi_sq = sq.span(50.0, 95.0)
lo_sq2, hi_sq2 = sq.span(50.0, 150.0)
fe = FileEnv([0.0, 10.0, 20.0, 30.0], [1.0, 0.0, 0.0, 1.0])
C.group('F 各包络 span 的解析正确性（分段/分片形状）', [
    ('方波：脉冲前段整段为 0', abs(lo_sq) < 1e-12 and abs(hi_sq) < 1e-12),
    ('方波：横跨升沿段 sup>0（不再被当中点零）', hi_sq2 > 0.5),
    ('方波：峰位 sup=height',
     abs(sq.span(295.0, 305.0)[1] - 1.0) < 1e-9),
    ('文件包络：样品点间为 0（精确，非采样近似）',
     abs(fe.span(12.0, 18.0)[1]) < 1e-12),
    ('文件包络：端点外钳 0',
     abs(fe.span(40.0, 50.0)[1]) < 1e-12),
])

sys.exit(0 if C.summary() else 1)
