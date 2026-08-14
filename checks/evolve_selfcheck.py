# -*- coding: utf-8 -*-
"""单原子动态模拟引擎自检（evolve.py 独立模块）：
- vs solve_ivp 精确参考（直接积分 H(t)，独立于演化实现）
- 布居归一 / 多光场 / 组驱动（core 偶极）/ 高斯脉冲
- 大基组性能（CPU vs GPU）"""
import os, sys, time
sys.stdout.reconfigure(encoding='utf-8', errors='replace')
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from scipy.integrate import solve_ivp
from arc import Rubidium87
from src import core, evolve

cache = core.ArcCache(Rubidium87())

# ---- 能级：6P3/2 (m=∓1/2) + 60S1/2 (m=+1/2) ----
states = evolve.make_levels(cache, [
    dict(n=6, l=1, j=1.5, m=-0.5, label='6P_{3/2} m=-1/2'),
    dict(n=6, l=1, j=1.5, m=+0.5, label='6P_{3/2} m=+1/2'),
    dict(n=60, l=0, j=0.5, m=+0.5, label='60S_{1/2} m=+1/2'),
])
N = len(states)
print(f'能级数 N = {N}')

fields = [evolve.field(1, 2, omega0_mhz=20.0, waveform=evolve.square(0, 1e9))]
t = np.linspace(0, 400.0, 2001)
psi0 = 2                       # 60S m=+0.5

# ---- 参考：solve_ivp 直接积分 H(t)（恒定 → 精确） ----
# 减去共同参考能量（不影响布居，仅消除整体相位 → 避免步长爆炸）
E_ref = states[0]['E_mhz']
I_N = np.eye(N)

def rhs(tv, y):
    Ht = evolve.hamiltonian(states, fields, tv) - E_ref * I_N
    return -1j * evolve._CONV * (Ht @ y)
v0 = np.zeros(N, complex); v0[psi0] = 1.0
sol = solve_ivp(rhs, (t[0], t[-1]), v0, t_eval=t, rtol=1e-10, atol=1e-13)
pops_ref = np.abs(sol.y).T ** 2          # (n_t, N)

print(f'\n恒定方波（Ω=20MHz, Δ=0）演化 400ns：')
ok = True
for method in ('pwc', 'ip4'):
    for backend in ('cpu', 'auto'):
        t0 = time.time()
        r = evolve.evolve(states, fields, psi0, t, method=method,
                          backend=backend)
        dt = time.time() - t0
        err = np.max(np.abs(r['pops'] - pops_ref))
        norm = np.max(np.abs(r['pops'].sum(axis=1) - 1.0))
        print(f'  [{method}/{backend:>4}] max_err={err:.2e}  norm_err={norm:.2e}'
              f'  耗时 {dt:6.2f}s')
        ok &= (err < 1e-5 and norm < 1e-9)

# ---- 多光场（共享下态 60S） ----
r3 = evolve.evolve(states, [evolve.field(2, 1, omega0_mhz=10.0),
                            evolve.field(2, 0, omega0_mhz=10.0)],
                   2, t, method='pwc')
norm3 = np.max(np.abs(r3['pops'].sum(axis=1) - 1.0))
print(f'多光场（两 π 驱动共享下态）：归一误差 {norm3:.2e}')
ok &= (norm3 < 1e-9)

# ---- 组驱动（复用 core 偶极矩阵元） ----
fg = evolve.group_field(cache, states, dict(n=6, l=1, j=1.5),
                        dict(n=60, l=0, j=0.5), pol='π', omega0_mhz=20.0)
r4 = evolve.evolve(states, [fg], 2, t, method='pwc')
norm4 = np.max(np.abs(r4['pops'].sum(axis=1) - 1.0))
print(f'组驱动（core 偶极 π）：归一误差 {norm4:.2e}')
ok &= (norm4 < 1e-9)

# ---- 高斯脉冲 + 布居归一 ----
t2 = np.linspace(0, 800.0, 4001)
fields2 = [evolve.field(1, 2, omega0_mhz=30.0,
                        waveform=evolve.gaussian(400.0, 60.0))]
t0 = time.time()
r2 = evolve.evolve(states, fields2, psi0, t2, method='pwc')
dt2 = time.time() - t0
gp = evolve.group_pops(states, r2['pops'],
                       [dict(n=6, l=1, j=1.5), dict(n=60, l=0, j=0.5)])
print(f'\n高斯脉冲（σ=60ns, Ω0=30MHz）：耗时 {dt2:.2f}s，'
      f'末态 6P 总布居 {gp[0, -1]:.3f}，60S 总布居 {gp[1, -1]:.3f}，'
      f'归一误差 {np.max(np.abs(r2["pops"].sum(axis=1) - 1)):.2e}')
ok &= (np.max(np.abs(r2['pops'].sum(axis=1) - 1.0)) < 1e-9)

# ---- 大基组性能（含高能旁观态，仅测归一 + 计时） ----
big = []
for n, l, j, ms in ((5, 1, 0.5, (-0.5, 0.5)),
                    (5, 1, 1.5, (-1.5, -0.5, 0.5, 1.5)),
                    (6, 0, 0.5, (-0.5, 0.5)),
                    (6, 1, 1.5, (-1.5, -0.5, 0.5, 1.5)),
                    (7, 0, 0.5, (-0.5, 0.5)),
                    (7, 1, 1.5, (-1.5, -0.5, 0.5, 1.5)),
                    (8, 0, 0.5, (-0.5, 0.5)),
                    (60, 0, 0.5, (-0.5, 0.5)),
                    (60, 1, 1.5, (-1.5, -0.5, 0.5, 1.5)),
                    (61, 0, 0.5, (-0.5, 0.5))):
    for m in ms:
        big.append(dict(n=n, l=l, j=j, m=m))
states_big = evolve.make_levels(cache, big)
ib = big.index({'n': 6, 'l': 1, 'j': 1.5, 'm': 0.5})
ie = big.index({'n': 60, 'l': 0, 'j': 0.5, 'm': 0.5})
t_big = np.linspace(0, 400.0, 2001)
t0 = time.time()
r5 = evolve.evolve(states_big, [evolve.field(ib, ie, omega0_mhz=20.0)],
                   ie, t_big, method='pwc')
dt5 = time.time() - t0
norm5 = np.max(np.abs(r5['pops'].sum(axis=1) - 1.0))
print(f'\n大基组 N={len(states_big)}（含高能旁观态）：耗时 {dt5:.2f}s，'
      f'归一误差 {norm5:.2e}')
ok &= (norm5 < 1e-9)

print(f'\n[evolve_selfcheck] PASS={ok}')
sys.exit(0 if ok else 1)
