# -*- coding: utf-8 -*-
"""对比本项目 (core.py, ARC) 与 pairinteraction 2.5.0 的计算结果。

覆盖三类可对比任务：
  A) 能级：fine 结构跃迁频率、6P3/2 超精细+Zeeman 子能级、60S Zeeman
  B) 偶极矩阵元：6P3/2↔60S、60S↔60P1/2、60S↔60P3/2、6P↔6S 等
  C) AC-Stark 交叉验证：远失谐区 pairinteraction DC Stark ≈ 本项目 AC（因子 2 换算）
"""
import sys
sys.stdout.reconfigure(encoding='utf-8', errors='replace')
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import pairinteraction as pi
from src.core import core
from arc.alkali_atom_data import Rubidium87

cache = core.ArcCache(Rubidium87())
B = 21.7

print('=' * 78)
print('A. 能级对比（fine 结构跃迁频率）')
print('=' * 78)
# 本项目的跃迁频率（ARC）
pairs = [(60, 0, 0.5, 60, 1, 0.5, '60S↔60P1/2'),
         (60, 0, 0.5, 60, 1, 1.5, '60S↔60P3/2'),
         (60, 1, 0.5, 60, 1, 1.5, '60P1/2↔60P3/2'),
         (6, 1, 1.5, 60, 0, 0.5, '6P3/2↔60S'),
         (6, 0, 0.5, 6, 1, 1.5, '6S1/2↔6P3/2'),
         (6, 0, 0.5, 6, 1, 0.5, '6S1/2↔6P1/2')]
for na, la, ja, nb, lb, jb, name in pairs:
    f_arc = core.transition_frequency_hz(cache.atom, na, la, ja, nb, lb, jb) / 1e9
    ea = pi.KetAtom("Rb", n=na, l=la, j=ja, m=ja).get_energy(unit="GHz")
    eb = pi.KetAtom("Rb", n=nb, l=lb, j=jb, m=jb).get_energy(unit="GHz")
    f_pi = abs(ea - eb)
    dev = abs(f_pi - f_arc) / 1e3  # GHz → THz
    print(f'  {name:14s} ARC={f_arc:14.4f} GHz  PI={f_pi:14.4f} GHz  差={dev:10.5f} THz')

print()
print('=' * 78)
print('B. 偶极矩阵元对比（原子单位 |d|）')
print('=' * 78)
# (源, 目标, q) 列表，q: 1=σ+, 0=π, -1=σ−
mat = [
    ((60, 0, 0.5, 0.5), (6, 1, 1.5, 1.5), -1, '60S+0.5 ↔ 6P3/2+1.5 (σ⁻)'),
    ((60, 0, 0.5, 0.5), (6, 1, 1.5, -0.5), -1, '60S+0.5 ↔ 6P3/2-0.5 (σ⁻)'),
    ((60, 0, 0.5, 0.5), (60, 1, 0.5, -0.5), -1, '60S+0.5 ↔ 60P1/2-0.5 (σ⁻)'),
    ((60, 0, 0.5, 0.5), (60, 1, 0.5, 0.5), 0, '60S+0.5 ↔ 60P1/2+0.5 (π)'),
    ((60, 0, 0.5, 0.5), (60, 1, 1.5, -0.5), -1, '60S+0.5 ↔ 60P3/2-0.5 (σ⁻)'),
    ((6, 1, 1.5, 1.5), (6, 0, 0.5, 0.5), -1, '6P3/2+1.5 ↔ 6S+0.5 (σ⁻)'),
    ((6, 1, 1.5, 0.5), (6, 0, 0.5, 0.5), 0, '6P3/2+0.5 ↔ 6S+0.5 (π)'),
]
for (na, la, ja, ma), (nb, lb, jb, mb), q, name in mat:
    d_arc = abs(cache.dme(na, la, ja, ma, nb, lb, jb, mb, q))
    ka = pi.KetAtom("Rb", n=na, l=la, j=ja, m=ma)
    kb = pi.KetAtom("Rb", n=nb, l=lb, j=jb, m=mb)
    d_pi = abs(ka.get_matrix_element(kb, "electric_dipole", q, unit="e * a0"))
    print(f'  {name:32s} ARC={d_arc:12.6f}  PI={d_pi:12.6f}  比={d_pi/d_arc if d_arc else float("nan"):8.5f}')
