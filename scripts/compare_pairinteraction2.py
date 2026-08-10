# -*- coding: utf-8 -*-
"""对比第二部分：Zeeman（mJ 级）、DC-Stark ↔ AC-SOS 交叉验证。

pairinteraction 的 Rb SQDT 模型忽略超精细 → 全部在 fine 结构 (n,l,j,mJ) 级对比。
AC↔DC 交叉验证原理：本项目远失谐 SOS 项 = −(E0²/4ħ)Σ|d|²·2ω_ki/(ω_ki²−ω²)，
当激光频率 ω→0（远低于所有跃迁）时退化为 −(E0²/2ħ)Σ|d|²/ω_ki = ½·DC Stark。
故本项目 AC(ω→0) 应 ≈ pairinteraction DC Stark(同 E0)/2。
"""
import sys
sys.stdout.reconfigure(encoding='utf-8', errors='replace')
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import pairinteraction as pi
from src import core
from arc.alkali_atom_data import Rubidium87

cache = core.ArcCache(Rubidium87())
B = 21.7

print('=' * 78)
print('C. Zeeman 对比（mJ 级，B=21.7 G）')
print('=' * 78)
for n, l, j, label in [(6, 1, 1.5, '6P3/2'), (60, 0, 0.5, '60S')]:
    # 本项目 mJ 子能级（相对最低于能级）
    rows = core._group_sublevel_rows(cache, n, l, j, B)
    e0 = min(r['E_mhz'] for r in rows)
    ours = sorted((r['E_mhz'] - e0) for r in rows)
    # pairinteraction Zeeman
    basis = pi.BasisAtom("Rb", n=(n, n), l=(l, l), j=(j, j))
    sysz = pi.SystemAtom(basis).set_magnetic_field([0, 0, B], unit="gauss")
    sysz.diagonalize()
    es = np.sort(sysz.get_eigenenergies(unit="MHz"))
    es = es - es[0]
    print(f'  {label}: 本项目 {len(ours)} 态, PI {len(es)} 态')
    for i in range(min(len(ours), len(es))):
        print(f'    {ours[i]:12.4f} vs {es[i]:12.4f} MHz   差={ours[i]-es[i]:+9.4f}')

print()
print('=' * 78)
print('D. 60S Zeeman 相邻劈裂')
print('=' * 78)
rows = core._group_sublevel_rows(cache, 60, 0, 0.5, B)
e0 = min(r['E_mhz'] for r in rows)
ours = sorted((r['E_mhz'] - e0) for r in rows)
basis = pi.BasisAtom("Rb", n=(60, 60), l=(0, 0), j=(0.5, 0.5))
sysz = pi.SystemAtom(basis).set_magnetic_field([0, 0, B], unit="gauss").diagonalize()
es = np.sort(sysz.get_eigenenergies(unit="MHz")) - np.sort(sysz.get_eigenenergies(unit="MHz"))[0]
print(f'  本项目劈裂: {ours[1]-ours[0]:.6f} MHz')
print(f'  PI 劈裂:    {es[1]-es[0]:.6f} MHz')

print()
print('=' * 78)
print('E. SOS(ω→0) ↔ DC-Stark 交叉验证（60S mJ=+0.5）')
print('=' * 78)
# E0 取一个较小的场（极化率线性区）：I=1e-3 mW/cm² → E0
I_mw = 1e-3
I_wm2 = I_mw * core.MW_PER_CM2_TO_W_PER_M2
E0 = (2.0 * I_wm2 / (core.epsilon_0 * core.c)) ** 0.5  # V/m
print(f'  E0 = {E0:.4f} V/m (I={I_mw} mW/cm²)')

# 本项目：legacy AC @ 极低频率（ω→0，纯 SOS 区）
groups = [dict(n=60, l=0, j=0.5, label='60S', color='r')]
for f_ghz in (0.1, 1.0):
    cfg = dict(enabled=True, intensity=I_mw,
               frequency_hz=f_ghz * 1e9, polarization='π')
    ac = core.ac_stark_shift_mhz(cache, groups[0], 0.5, groups, cfg, B_gauss=0.0)
    print(f'  本项目 AC @ ω={f_ghz} GHz = {ac*1e3:+.4f} kHz')

# pairinteraction DC Stark：60S 基组（含近邻 Rydberg 态）
basis = pi.BasisAtom("Rb", n=(58, 62), l=(0, 3), m=(0.5, 0.5))
print(f'  PI DC-Stark 基组: {basis.number_of_states} 态')
sysdc = pi.SystemAtom(basis).set_electric_field([0, 0, E0], unit="V/m").diagonalize()
ket = pi.KetAtom("Rb", n=60, l=0, j=0.5, m=0.5)
e_shift = sysdc.get_corresponding_energy(ket, unit="MHz") - ket.get_energy(unit="MHz")
print(f'  PI DC-Stark 位移(60S mJ+0.5) = {e_shift*1e3:+.4f} kHz')
print(f'  PI DC/2 = {e_shift*1e3/2:+.4f} kHz  ← 应与本项目 AC(ω→0) 接近')
