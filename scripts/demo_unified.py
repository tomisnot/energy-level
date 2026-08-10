# -*- coding: utf-8 -*-
"""统一驱动演示（单体，driver 统一接口 build_system + drive）：
寻址光驱动 6P3/2↔60S（布居振荡）+ 微波(60S↔60P)做能移（AC-Stark）。

对比：无微波能移（寻址光共振） vs 有微波能移（60S 被位移 → 寻址光失谐）。
运行：D:\\python310\\python.exe -u scripts\\demo_unified.py
"""
import os
import sys
sys.stdout.reconfigure(encoding='utf-8', errors='replace')
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
from arc import Rubidium87
from src import core, driver

cache = core.ArcCache(Rubidium87())
groups = [dict(n=5, l=0, j=0.5, label='5S'), dict(n=6, l=1, j=1.5, label='6P3/2'),
          dict(n=60, l=0, j=0.5, label='60S'), dict(n=60, l=1, j=0.5, label='60P1/2')]

# 寻址光共振频率（6P3/2 F=3 mF=+3 ↔ 60S mJ=+0.5，σ-）
rows_6p = core._group_sublevel_rows(cache, 6, 1, 1.5, 0.0)
rows_60s = core._group_sublevel_rows(cache, 60, 0, 0.5, 0.0)
b63 = next(r for r in rows_6p if r['qstr'] == 'F=3.0, m_F=+3.0')
a60 = next(r for r in rows_60s if r['qstr'] == 'm_J=+0.5')
f_ad = (a60['E_mhz'] - b63['E_mhz']) * 1e6

ad = dict(kind='ad', group_a=dict(n=6, l=1, j=1.5), group_b=dict(n=60, l=0, j=0.5),
          pol='σ-', om_mhz=8.0, omega_rad_ns=2 * np.pi * 1e-3 * f_ad / 1e6)

# 微波做能移（60S↔60P，远失谐 +20GHz，纯二阶能移区；U2 验证 60S 位移 ~-360 MHz）
f_mw0 = core.transition_frequency_hz(cache.atom, 60, 0, 0.5, 60, 1, 0.5)
mw = dict(frequency_hz=f_mw0 + 20e9, intensity=200.0, polarization='π')


def p60(res):
    i = next(k for k, s in enumerate(res['single'])
             if s['n'] == 60 and s['l'] == 0 and s['qstr'] == 'm_J=+0.5')
    return res['pops'][i]


# 无微波能移
s0 = driver.build_system(cache, groups, 1)
r0 = driver.drive(s0, ad, psi0_ket=(6, 1, 1.5, 3.0))
p0 = p60(r0)
# 有微波能移
s1 = driver.build_system(cache, groups, 1, static_cfg=dict(ac_stark=mw))
r1 = driver.drive(s1, ad, psi0_ket=(6, 1, 1.5, 3.0))
p1 = p60(r1)

print(f'无微波能移:  60S max={p0.max():.3f} 末={p0[-1]:.3f}  (Ω_ad={r0["om_mhz"]:.2f} MHz)')
print(f'有微波能移:  60S max={p1.max():.3f} 末={p1[-1]:.3f}  (微波位移后失谐)')
