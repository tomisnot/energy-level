# -*- coding: utf-8 -*-
"""driver 端到端演示：dressed 投影（含 vdW）微波驱动封锁对比。

R=60μm 无相互作用（PP 自由）vs R=4μm 强相互作用（vdW+交换封锁 PP）。
运行：D:\\python310\\python.exe -u scripts\\demo_pair_drive.py
"""
import os
import sys
sys.stdout.reconfigure(encoding='utf-8', errors='replace')
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
from arc import Rubidium87
from src.core import core, pair_core
from src import driver

cache = core.ArcCache(Rubidium87())
groups = [{'n': 60, 'l': 0, 'j': 0.5}, {'n': 60, 'l': 1, 'j': 0.5}]
w = 2 * np.pi * 1e-3 * 16826.7          # 60S↔60P 微波载频（rad/ns，≈共振）

for R in (60.0, 4.0):
    res = driver.pair_drive_dressed(cache, groups, R, 0.0,
                                    dict(kind='mw', pol='π', om_mhz=30.0,
                                         omega_rad_ns=w), m_total=1.0)
    N = len(res['single'])
    sel = {i * N + j: k for k, (i, j) in enumerate(res['pairs'])}
    ip = pair_core.find_sublevel(res['single'], 60, 1, 0.5, 0.5)
    is_ = pair_core.find_sublevel(res['single'], 60, 0, 0.5, 0.5)
    pp = res['pops'][sel[ip * N + ip]]
    ss = res['pops'][sel[is_ * N + is_]]
    vex = abs(pair_core.exchange_coupling(cache, groups, R))
    dss, _ = pair_core.vdw_shift(cache, groups, R, 0.0, 0.0, 0.0, 1, m_total=1.0)
    print(f'R={R:5.1f}um:  V_ex={vex:6.2f}  dSS={dss:6.2f} MHz  '
          f'max P(PP)={pp.max():.3f}   末态 SS={ss[-1]:.3f} PP={pp[-1]:.3f}')
