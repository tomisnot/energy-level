# -*- coding: utf-8 -*-
"""场致加热与损失模型自检（Re0 重构 · 第 14 刀 Phase H1）。

守的是 `src/core/field_loss.py` 的**公式与量级**（不含演化器接线，那是 H3）：
  A7 反冲温度 / 散射速率量级；A8 弱场回线性 · 强场饱和上限 Γ/2 · 线性式高估；
  A1 修正旧 F3 取反——里德堡寻址几何（60S↔6P）下 γ(60S)>0（旧实现恒 0），
     5S→6P（短寿命在上）逐位一致；L4 光电离量级；A9 n_crit 阱深散射数；
  双计规则（伙伴 Γ_out=0 ⇒ 该向散射计损归零）与近共振剔除（|Δ|<near_cutoff）。

纯 numpy（无 ARC/numba/引擎），逐条对齐源仓 dissipation_lab/selfcheck 的锚。
运行：D:\\python310\\python.exe checks\\field_loss_selfcheck.py
"""
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from src.core.field_loss import (scatter_rate_mhz, scatter_rate_linear_mhz,
                                 scatter_pair_rates, add_scatter_pair,
                                 FieldLossConfig, TrapModel,
                                 photoion_rate_mhz, recoil_temperature_K)
from src.core.field_build import build_field_loss
from src.core.atomic_levels import make_levels, square
from src.core.evolve_util import decay
from checklib import Checker

ck = Checker('field_loss_selfcheck')

LAM = 1013.8e-9          # m，本仓寻址光
GAM_6P = 8.464           # MHz（τ=118.1 ns）
GAM_60S = 4.35e-3        # MHz（τ=230 µs）


# ---------------- 包 1：A7 量级锚 ----------------
subs = []
Er = recoil_temperature_K(LAM)
subs.append((f'A7 E_r/k_B≈107.2nK (={Er:.4g}K)',
             abs(Er / 107.2e-9 - 1) < 0.01))
g = scatter_rate_mhz(100.0, 5000.0, GAM_6P) * 1e6
subs.append((f'A7 γ_sc(100M,5G)≈846/s (={g:.1f})', abs(g / 846 - 1) < 0.02))
ck.group('A7 反冲温度与散射速率量级', subs)

# ---------------- 包 2：A8 饱和与线性极限 ----------------
subs = []
gl = scatter_rate_linear_mhz(100.0, 5000.0, GAM_6P)
subs.append(('A8 弱场回到线性式（s≪1）',
             abs(g / (gl * 1e6) - 1) < 0.001))
g_sat = scatter_rate_mhz(20000.0, 5000.0, GAM_6P)
g_lin = scatter_rate_linear_mhz(20000.0, 5000.0, GAM_6P)
subs.append((f'A8 强场线性式高估~9x (ratio={g_lin / g_sat:.2f})',
             abs(g_lin / g_sat - 9) < 1.0))
g_inf = scatter_rate_mhz(1e9, 5000.0, GAM_6P)
subs.append((f'A8 饱和上限 Γ/2 (={g_inf:.6g})', abs(g_inf - GAM_6P / 2) < 1e-6))
ck.group('A8 含饱和：弱场回线性 / 强场封顶 Γ÷2', subs)

# ---------------- 包 3：A1 修正 F3 取反（里德堡几何） ----------------
subs = []
gi, gj = scatter_pair_rates(100.0, 5000.0, GAM_60S, GAM_6P)
subs.append((f'A1 γ(60S)=Ω²Γ_6P/4Δ²>0 (={gi * 1e6:.1f}/s)',
             abs(gi * 1e6 / 846 - 1) < 0.02))
subs.append((f'A1 γ(6P) 用 60S 的 Γ（小~1000×, {gj / gi:.1e}）', gj / gi < 0.01))
g5s = scatter_rate_mhz(100.0, 20000.0, GAM_6P)
subs.append(('A1 5S→6P 几何逐位一致（短寿命在上旧实现恰对）',
             abs(g5s / scatter_rate_linear_mhz(100.0, 20000.0, GAM_6P) - 1)
             < 0.001))
ck.group('A1 修正旧 F3：源态/Γ 取反，里德堡几何 γ(60S) 不再恒 0', subs)

# ---------------- 包 4：L4 光电离量级 ----------------
subs = []
r_pi = photoion_rate_mhz(1e7, 1064e-9, 1.0) * 1e6
subs.append((f'L4 光电离量级 5.4e4/s (={r_pi:.3g})', abs(r_pi / 5.4e4 - 1) < 0.05))
subs.append(('L4 σ=0 或 I≤0 → 关闭', photoion_rate_mhz(1e7, 1064e-9, 0.0) == 0.0
             and photoion_rate_mhz(0.0, 1064e-9, 1.0) == 0.0))
ck.group('L4 光电离 R_PI=σ·Φ（线性于光强）', subs)

# ---------------- 包 5：A9 阱深散射数 ----------------
subs = []
tp = TrapModel(LAM, depth_K=1e-3, kappa_geo=2.0)
subs.append((f'A9 N_crit(1mK)≈4664 (={tp.n_crit:.0f})',
             abs(tp.n_crit / 4664 - 1) < 0.01))
subs.append(('A9 depth_K=None → 关闭（n_crit=inf, 不给加热数）',
             TrapModel(LAM, depth_K=None).n_crit == float('inf')))
ck.group('A9 加热逃逸 N_crit=U/(κ·E_r) 与开关', subs)

# ---------------- 包 6：双计规则 + 近共振剔除（add_scatter_pair） ----------------
subs = []
# 双计：伙伴 Γ_out=0 ⇒ 掺入它的源态散射计损归零
gi0, gj0 = scatter_pair_rates(100.0, 5000.0, GAM_60S, 0.0)
subs.append(('双计：伙伴 out=0 → γ(源态)=0（不重复计损）', gi0 == 0.0 and gj0 > 0.0))
# 近共振剔除：|Δ| < near_cutoff → 整对跳过，cfg 不变
env1 = lambda ts: np.ones_like(np.atleast_1d(np.asarray(ts, float)))
cfg_near = FieldLossConfig(2)
add_scatter_pair(cfg_near, 0, 1, Om_mhz=100.0, det_mhz=3000.0,
                 gam_out_i_mhz=GAM_60S, gam_out_j_mhz=GAM_6P, env_fn=env1,
                 near_cutoff_mhz=5000.0)
subs.append(('近共振剔除：|Δ|<near_cutoff → 通道数 0',
             cfg_near.is_empty()))
cfg_far = FieldLossConfig(2)
add_scatter_pair(cfg_far, 0, 1, Om_mhz=100.0, det_mhz=8000.0,
                 gam_out_i_mhz=GAM_60S, gam_out_j_mhz=GAM_6P, env_fn=env1)
pk = cfg_far.peak_loss_rates_mhz()
subs.append((f'远失谐对装配：两端各有损耗（60S={pk[0]:.2e} 6P={pk[1]:.2e}）',
             pk[0] > 0.0 and pk[1] > 0.0 and not cfg_far.is_empty()))
ck.group('add_scatter_pair：双计规则 + 近共振剔除 + 双向装配', subs)


# ---------------- 包 7：build_field_loss（H2 装配器，桩 cache 不依赖 ARC） ----------------
class _StubAtom:
    def __init__(self, tau_map):
        self.tau_map = tau_map

    def getStateLifetime(self, n, l, j, temperature=0.0):
        return self.tau_map[(int(n), int(l), round(float(j), 3))]


class _StubCache:
    def __init__(self, tau_map):
        self.atom = _StubAtom(tau_map)

    def E_mhz(self, n, l, j):
        return 0.0


# idx0 = 60S（长寿命 Γ_60S），idx1 = 5P 状（短寿命 Γ_6P）；E 相差 6000 MHz
tau_map = {(60, 0, 0.5): 1e-6 / GAM_60S, (5, 1, 1.5): 1e-6 / GAM_6P}
stub = _StubCache(tau_map)
bstates = make_levels(stub, [
    dict(n=60, l=0, j=0.5, m=-0.5, E_mhz=0.0),
    dict(n=5, l=1, j=1.5, m=0.5, E_mhz=6000.0)])


def _bfield(omega_L, om=100.0, I=0.0):
    A = np.zeros((2, 2))
    A[0, 1] = A[1, 0] = om
    return [{'kind': 'exact', 'A': A, 'omega_L_mhz': omega_L,
             'waveform': square(0.0, 1e12), 'intensity_mwcm2': I}]


spec_far = {'near_cutoff_mhz': 5000.0}
subs = []
# 远失谐（det=6000>5000）：两端各得一条损耗，与手算逐位一致
cfg = build_field_loss(stub, bstates, _bfield(0.0), [], spec_far)
pk = cfg.peak_loss_rates_mhz()
subs.append((f'H2 far 对装配：γ(60S)={pk[0]:.3e} 用伙伴 Γ_6P',
             abs(pk[0] - scatter_rate_mhz(100.0, 6000.0, GAM_6P)) < 1e-15))
subs.append((f'H2 far 对装配：γ(5P)={pk[1]:.3e} 用伙伴 Γ_60S（小~1000×）',
             abs(pk[1] - scatter_rate_mhz(100.0, 6000.0, GAM_60S)) < 1e-15))
# 双计：伙伴 idx1 已配自发辐射通道（to=None）→ out_rate(1)=0 → γ(60S)=0
cfg_dk = build_field_loss(stub, bstates, _bfield(0.0),
                          [decay(1, GAM_6P, None)], spec_far)
pk_dk = cfg_dk.peak_loss_rates_mhz()
subs.append((f'双计：伙伴配了衰变 → γ(60S) 归零（={pk_dk[0]:.1e}）',
             pk_dk[0] == 0.0 and pk_dk[1] > 0.0))
# 近共振剔除：omega_L=2000 → det=4000<5000 → 整对跳过 → None
subs.append(('近共振剔除：det<near_cutoff → cfg None',
             build_field_loss(stub, bstates, _bfield(2000.0), [],
                              spec_far) is None))
# opt-in：无 spec → None
subs.append(('opt-in：spec=None → None',
             build_field_loss(stub, bstates, _bfield(0.0), [], None) is None))
# 光电离：σ>0 且 I>0 → 多一条 photoion 通道
cfg_pi = build_field_loss(stub, bstates, _bfield(0.0, I=1e7), [],
                          {'near_cutoff_mhz': 5000.0,
                           'photoion': {'sigma_mb_by_idx': {1: 1.0},
                                        'lam_m': 1064e-9}})
subs.append(('L4 光电离：spec.photoion → 有 photoion 通道',
             cfg_pi is not None and any(
                 c['kind'] == 'photoion' for c in cfg_pi.channels)))
ck.group('H2 build_field_loss：远失谐装配 / 双计 / 近共振 / opt-in / 光电离', subs)

ok = ck.summary()
sys.exit(0 if ok else 1)
