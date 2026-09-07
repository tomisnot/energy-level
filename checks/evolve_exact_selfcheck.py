# -*- coding: utf-8 -*-
"""第一性原理含时引擎自检（evolve_exact.py，method='exact'）。

模型：实验室系 TDSE，无 RWA / 无旋转系 / 无近/远失谐分流：
    H(t) = H0 + Σ_k Ω_k·env_k(t)·cos(ω_k t)·A_k
算法：Sambe (t,t') 光子块扩展空间 + 包络自适应分段（冻结中点精确指数
+ Duhamel 缺陷修正，节点数自适应）；误差由 tol 控制并随报告可见。

场景（E 系列）：
  E1 解析基准：共振方波 π 脉冲（P_e(π 时刻)=1、半 π 时刻=0.5）
  E2 交叉验证：高斯失谐驱动 vs 参考积分器（H0 相互作用绘景 DOP853）
  E3 封存权威：常数包络单频 vs src/_archive/evolve_floquet_pwc.evolve_floquet
  E4 物理自洽：远失谐 CW 的 AC 位移（干涉法提取）≈ 解析 −Ω²ν/(2(ν²−ω²))
  E5 收敛性：tol 1e-5 vs 1e-7 误差单调下降；范数守恒
  E6 耗散：纯衰变解析 e^{−Γt}；CW 驱动稳态 ≈ Ω²/(Γ²/4+2Ω²)（转参考路径）
  E7 STIRAP 冒烟：Λ 型反直觉序双脉冲，末态转移 > 0.9

运行：D:\\python310\\python.exe -u checks\\evolve_exact_selfcheck.py
"""
import os
import sys
sys.stdout.reconfigure(encoding='utf-8', errors='replace')
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from src.core.sambe import evolve_sambe as ex_ev
from src.core.evolve_reference import evolve_reference as ex_ref
# t74：evolve shim 已删——按新家直连
from src.core.atomic_levels import gaussian, square, make_levels
from src.core.field_build import field, real_field, _field_geometry, _sublevel_row
from src.core.evolve_util import decay, auto_decays, _as_vector
from src.core.evolve_rwa import (evolve, _CONV, _e_rot, _evolve_full,
                                 _evolve_full_hybrid)

from checklib import Checker

ck = Checker('evolve_exact_selfcheck')


def two_states():
    return [dict(n=60, l=0, j=0.5, kind='mJ', mj=-0.5, m=-0.5, E_mhz=0.0,
                 label='g', index=0, amps=None, F=None, mF=None),
            dict(n=60, l=1, j=0.5, kind='mJ', mj=-0.5, m=-0.5, E_mhz=1000.0,
                 label='e', index=1, amps=None, F=None, mF=None)]


def fld(A, wL, wf, label='f'):
    return {'kind': 'exact', 'A': np.asarray(A, float),
            'omega_L_mhz': float(wL), 'waveform': wf, 'label': label}


# ---------------- E1 解析基准：共振方波 π 脉冲 ----------------
A10 = np.zeros((2, 2)); A10[0, 1] = A10[1, 0] = 10.0
ones = lambda t: np.ones_like(np.asarray(t, float))
t1 = np.linspace(0, 50.0, 51)
r1 = ex_ev(two_states(), [fld(A10, 1000.0, ones)], 0, t1,
                     tol=1e-8)
pe25 = float(r1['pops'][25, 1])
pe50 = float(r1['pops'][50, 1])
ck.check(f'E1a 共振 π 脉冲半程 P_e(25ns)={pe25:.5f}≈0.5',
         abs(pe25 - 0.5) < 2e-3, f'P_e={pe25}')
ck.check(f'E1b 共振 π 脉冲全程 P_e(50ns)={pe50:.6f}≈1',
         abs(pe50 - 1.0) < 2e-3, f'P_e={pe50}')

# ---------------- E2 交叉验证：高斯失谐 vs 参考积分器 ----------------
wf_g = lambda t: np.exp(-((np.asarray(t, float) - 200.0) ** 2)
                        / (2 * 60.0 ** 2))
t2 = np.linspace(0, 400.0, 401)
r2 = ex_ev(two_states(), [fld(A10, 1015.0, wf_g)], 0, t2,
                     tol=1e-6)
ref2 = ex_ref(two_states(), [fld(A10, 1015.0, wf_g)], 0, t2,
                           rtol=1e-10, atol=1e-12)
d2 = float(np.max(np.abs(r2['pops'] - ref2['pops'])))
ck.check(f'E2 高斯失谐 vs DOP853 参考 max|Δpops|={d2:.2e} ≤ 1e-5',
         d2 <= 1e-5, f'Δ={d2}')

# ---------------- E3 封存权威：常数包络单频 vs 精确 Floquet ----------------
from src._archive.evolve_floquet_pwc import evolve_floquet
t3 = np.linspace(0, 300.0, 301)
fld_pair = field(0, 1, omega0_mhz=10.0, detune_mhz=20.0)
r3f = evolve_floquet(two_states(), [fld_pair], 0, t3)
A3 = np.zeros((2, 2)); A3[0, 1] = A3[1, 0] = 10.0
r3 = ex_ev(two_states(), [fld(A3, 1020.0, ones)], 0, t3,
                     tol=1e-8)
d3 = float(np.max(np.abs(r3['pops'] - r3f['pops'])))
ck.check(f'E3 常数包络单频 vs 封存精确 Floquet max|Δpops|={d3:.2e} ≤ 1e-3',
         d3 <= 1e-3, f'Δ={d3}')

# ---------------- E4 物理自洽：远失谐 AC 位移（干涉法） ----------------
# 三态：|g⟩(0) 与 |u⟩(0) 简并且仅 |g⟩ 被远失谐光耦合到 |e⟩(1000)；
# 初态 (|g⟩+|u⟩)/√2 → |u⟩ 布居以 δ 频率振荡（δ = 位移差）。
st4 = two_states() + [dict(n=61, l=0, j=0.5, kind='mJ', mj=-0.5, m=-0.5,
                           E_mhz=0.0, label='u', index=2, amps=None,
                           F=None, mF=None)]
A4 = np.zeros((3, 3)); A4[0, 1] = A4[1, 0] = 10.0      # 仅 g↔e
nu, w4, Om = 1000.0, 950.0, 10.0
delta_th = -Om ** 2 * nu / (2.0 * (nu ** 2 - w4 ** 2))   # MHz（解析二阶）
t4 = np.linspace(0, 2000.0, 2001)
psi0_4 = np.zeros(3, complex); psi0_4[0] = psi0_4[2] = 1.0 / np.sqrt(2)
r4 = ex_ev(st4, [fld(A4, w4, ones)], psi0_4, t4, tol=1e-7)
pu = r4['pops'][:, 2]
# P_u(t) = ½(1−cos(2π·δ·t_µs))：FFT 低频段（<20 MHz，避开 ~50 MHz 快振荡）
# 取主峰频 = |δ|
sig = pu - pu.mean()
sp = np.abs(np.fft.rfft(sig * np.hanning(len(sig))))
freq_mhz = np.fft.rfftfreq(len(sig), d=1.0) * 1000.0   # 1/ns → MHz
band = (freq_mhz > 0.1) & (freq_mhz < 20.0)
delta_fit = float(freq_mhz[band][np.argmax(sp[band])])
err4 = abs(delta_fit - abs(delta_th))
ck.check(f'E4 远失谐位移 拟合|δ|={delta_fit:.4f} vs 解析 {abs(delta_th):.4f} '
         f'MHz（差 {err4:.3f} ≤ {0.08 * abs(delta_th):.3f}）',
         err4 <= 0.08 * abs(delta_th),
         f'fit={delta_fit} th={delta_th}')

# ---------------- E5 收敛性：tol 单调 + 范数守恒 ----------------
r5a = ex_ev(two_states(), [fld(A10, 1015.0, wf_g)], 0, t2,
                      tol=1e-4)
r5b = ex_ev(two_states(), [fld(A10, 1015.0, wf_g)], 0, t2,
                      tol=1e-7)
ea = float(np.max(np.abs(r5a['pops'] - ref2['pops'])))
eb = float(np.max(np.abs(r5b['pops'] - ref2['pops'])))
ck.check(f'E5a tol 收敛：err(1e-4)={ea:.2e} > err(1e-7)={eb:.2e}',
         eb < ea, f'ea={ea} eb={eb}')
nd = float(np.max(np.abs(r5b['pops'].sum(axis=1) - 1.0)))
ck.check(f'E5b 幺正范数守恒：全程 |Σp−1|_max={nd:.2e} ≤ 1e-5',
         nd <= 1e-5, f'norm_dev={nd}')

# ---------------- E6 耗散（转参考路径） ----------------
G6 = 10.0
t6 = np.linspace(0, 400.0, 401)
r6 = ex_ev(two_states(), [], 1, t6, tol=1e-8,
                     decays=[decay(1, G6)])
exp6 = np.exp(-G6 * 1e-3 * t6)
d6 = float(np.max(np.abs(r6['pops'][:, 1] - exp6)))
cons6 = float(np.max(np.abs(r6['pops'].sum(axis=1) + r6['loss'] - 1.0)))
ck.check(f'E6a 解析衰变 e^{{−Γt}} max|Δ|={d6:.2e} ≤ 1e-6',
         d6 <= 1e-6 and cons6 <= 1e-9, f'Δ={d6} cons={cons6}')
# CW 驱动稳态：共振二能级 + 衰变 e→g：
# ρ_ee = c²/(γ²/4 + 2c²)，c = π·1e-3·Ω_MHz（RWA 耦合，rad/ns），γ = Γ·1e-3
c6 = np.pi * 1e-3 * 1.0
g6 = 2.0 * 1e-3
rho_th = c6 ** 2 / (g6 ** 2 / 4.0 + 2.0 * c6 ** 2)
t6b = np.linspace(0, 6000.0, 1201)
A6 = np.zeros((2, 2)); A6[0, 1] = A6[1, 0] = 1.0
r6b = ex_ev(two_states(), [fld(A6, 1000.0, ones)], 1, t6b,
                      tol=1e-6, decays=[decay(1, 2.0, 0)])
pe_ss = float(r6b['pops'][-1, 1])
ck.check(f'E6b CW 驱动稳态 ρ_ee={pe_ss:.4f} ≈ {rho_th:.4f}',
         abs(pe_ss - rho_th) < 1.5e-2, f'ρ_ee={pe_ss} th={rho_th}')

# ---------------- E7 STIRAP 冒烟（Λ 型反直觉序）+ 参考交叉验证 ----------------
st7 = [dict(n=60, l=0, j=0.5, kind='mJ', mj=-0.5, m=-0.5, E_mhz=0.0,
            label='g', index=0, amps=None, F=None, mF=None),
       dict(n=60, l=1, j=0.5, kind='mJ', mj=-0.5, m=-0.5, E_mhz=1000.0,
            label='e', index=1, amps=None, F=None, mF=None),
       dict(n=61, l=0, j=0.5, kind='mJ', mj=-0.5, m=-0.5, E_mhz=30.0,
            label='s', index=2, amps=None, F=None, mF=None)]
Ap = np.zeros((3, 3)); Ap[0, 1] = Ap[1, 0] = 50.0      # pump: g↔e
As_ = np.zeros((3, 3)); As_[2, 1] = As_[1, 2] = 50.0    # stokes: s↔e
wp = lambda t: np.exp(-((np.asarray(t, float) - 240.0) ** 2)
                      / (2 * 60.0 ** 2))
ws_ = lambda t: np.exp(-((np.asarray(t, float) - 120.0) ** 2)
                       / (2 * 60.0 ** 2))
flds7 = [fld(Ap, 1000.0, wp, 'pump'), fld(As_, 970.0, ws_, 'stokes')]
t7 = np.linspace(0, 500.0, 501)
r7 = ex_ev(st7, flds7, 0, t7, tol=1e-6)
ps_end = float(r7['pops'][-1, 2])
pe_max = float(np.max(r7['pops'][:, 1]))
ck.check(f'E7a STIRAP 末态 P_s={ps_end:.4f} > 0.95（中间态峰值 '
         f'{pe_max:.4f} < 0.05）', ps_end > 0.95 and pe_max < 0.05,
         f'P_s={ps_end} P_e_max={pe_max}')
ref7 = ex_ref(st7, flds7, 0, t7, rtol=1e-9, atol=1e-11)
d7 = float(np.max(np.abs(r7['pops'] - ref7['pops'])))
ck.check(f'E7b STIRAP vs DOP853 参考 max|Δpops|={d7:.2e} ≤ 5e-5',
         d7 <= 5e-5, f'Δ={d7}')

ok = ck.summary()
sys.exit(0 if ok else 1)
