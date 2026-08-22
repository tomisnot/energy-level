# -*- coding: utf-8 -*-
"""RWA 精度基准（evolve.py）：用户典型工作场景——光频 + 微波同开、共享中间态。

两类场景，均以 **全非 RWA 第一性原理**（evolve._evolve_full，solve_ivp 自适应，
实验室系 H=H0+ΣΩ(t)cos(ωt)A）为参考：

场景 A（纯微波两能级 60S↔60P，无光频）：full 参考快（~7s/20ns），
   扫描 Ω/ωL → 量化微波 RWA / BS 修正误差。
场景 B（阶梯三能级 6P↔60S↔60P，光频 Ω=20MHz 同开）：full 参考含光频载波
   （~1.9e6 rad/ns）很慢，用短时间窗 T_B（默认 3ns）。
   对比 ① 独立 RWA（旧逻辑，共享能级错误）② 级联 RWA（新默认）③ 级联+BS。

报告：各算法 vs 第一性原理的 max|Δpops|（全部时刻/全部态），及耗时。
预算：算法 ≤ 10×（级联 RWA 跑同任务耗时）。参考慢是允许的。

用法：D:\\python310\\python.exe -u checks\\evolve_rwa_bench.py [T_A_ns] [T_B_ns]
默认 T_A=20（纯微波参考 ~2-8s/场景），T_B=20（阶梯参考用 full_fast 混合
参考：光频 RWA（误差 ~1e-7）+ 微波非 RWA，只受微波步长限制，~1s/ns）。

**2025-08**：pwc（RWA 分段常数）已从 evolve.py 封存，本基准改用封存模块
src/_archive/evolve_floquet_pwc.py 的 evolve_pwc（等价旧 method='pwc'）。
"""
import os
import sys
import time
sys.stdout.reconfigure(encoding='utf-8', errors='replace')
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from arc import Rubidium87
from src import core, evolve
from src._archive.evolve_floquet_pwc import _evolve_pwc, evolve_pwc

T_A = float(sys.argv[1]) if len(sys.argv) > 1 else 20.0
T_B = float(sys.argv[2]) if len(sys.argv) > 2 else 20.0

cache = core.ArcCache(Rubidium87())


def old_rwa_evolve(states, fields, psi0, t):
    """旧逻辑：独立 RWA（每驱动 e 态各自减 ω_L）——共享能级会结构性错误。"""
    H0 = np.diag([s['E_mhz'] for s in states]).astype(float)
    geoms = [evolve._field_geometry(states, f) for f in fields]
    for g in geoms:
        for i in g['e_idx']:
            H0[i, i] -= g['omega_L_mhz']
    psi0v = evolve._as_vector(states, psi0)
    _, pops, _ = _evolve_pwc(H0, geoms, psi0v, t, False, False)
    return pops


# ================= 场景 A：纯微波两能级 =================
statesA = evolve.make_levels(cache, [
    dict(n=60, l=0, j=0.5, m=+0.5, label='60S'),
    dict(n=60, l=1, j=0.5, m=+0.5, label='60P'),
])
w_mw = statesA[1]['E_mhz'] - statesA[0]['E_mhz']
tA = np.linspace(0, T_A, 401)
print(f'场景 A：纯微波 60S↔60P（ω_mw={w_mw:.4g} MHz），T={T_A} ns')
print(f'{"Ω_mw(MHz)":>10} | {"Ω/ωL%":>7} | {"参考耗时":>7} | '
      f'{"RWA err":>9} | {"RWA+BS err":>10} | {"算法耗时":>8}')
print('-' * 80)
EA = np.array([s['E_mhz'] for s in statesA])
psiA = evolve._as_vector(statesA, 0)
for om in (30.0, 100.0, 300.0):
    fields = [evolve.field(0, 1, omega0_mhz=om)]
    geoms = [evolve._field_geometry(statesA, f) for f in fields]
    t0 = time.time()
    _, ref, _ = evolve._evolve_full(EA, geoms, psiA, tA, False)
    dt_ref = time.time() - t0
    t0 = time.time()
    r_rwa = evolve_pwc(statesA, fields, 0, tA, bs=False)
    r_bs = evolve_pwc(statesA, fields, 0, tA, bs=True)
    dt_alg = time.time() - t0
    e_rwa = np.max(np.abs(r_rwa['pops'] - ref))
    e_bs = np.max(np.abs(r_bs['pops'] - ref))
    print(f'{om:10.1f} | {om/w_mw*100:6.3f}% | {dt_ref:6.1f}s | '
          f'{e_rwa:9.2e} | {e_bs:10.2e} | {dt_alg:6.2f}s')

# ================= 场景 B：阶梯三能级（光频+微波同开，共享 60S） =================
statesB = evolve.make_levels(cache, [
    dict(n=6, l=1, j=1.5, m=+0.5, label='6P'),
    dict(n=60, l=0, j=0.5, m=+0.5, label='60S'),
    dict(n=60, l=1, j=0.5, m=+0.5, label='60P'),
])
tB = np.linspace(0, T_B, 401)
print(f'\n场景 B：阶梯 6P↔60S↔60P（光频 Ω=20MHz 同开，共享中间态 60S），T={T_B} ns')
print('参考 = full_fast（光频 RWA + 微波非 RWA，光频误差 ~1e-7 可忽略）')
print(f'{"Ω_mw,Δ_mw":>12} | {"参考耗时":>7} | {"独立RWA":>9} | '
      f'{"级联RWA":>9} | {"级联+BS":>9} | {"算法耗时":>8}')
print('-' * 88)
for om_mw, det_mw in ((30.0, 0.0), (100.0, 0.0), (300.0, 0.0)):
    fields = [evolve.field(0, 1, omega0_mhz=20.0, waveform=evolve.square(0, T_B)),
              evolve.field(1, 2, omega0_mhz=om_mw, detune_mhz=det_mw,
                           waveform=evolve.square(0, T_B))]
    EB = np.array([s['E_mhz'] for s in statesB])
    psiB = evolve._as_vector(statesB, 1)
    t0 = time.time()
    _, ref, _ = evolve._evolve_full_hybrid(EB, statesB, fields, psiB, tB, False)
    dt_ref = time.time() - t0
    t0 = time.time()
    r_new = evolve_pwc(statesB, fields, 1, tB, bs=False)
    r_bs = evolve_pwc(statesB, fields, 1, tB, bs=True)
    r_old = old_rwa_evolve(statesB, fields, 1, tB)
    dt_alg = time.time() - t0
    e_old = np.max(np.abs(r_old - ref))
    e_new = np.max(np.abs(r_new['pops'] - ref))
    e_bs = np.max(np.abs(r_bs['pops'] - ref))
    print(f'({om_mw:5.1f},{det_mw:5.1f}) | {dt_ref:6.1f}s | '
          f'{e_old:9.2e} | {e_new:9.2e} | {e_bs:9.2e} | {dt_alg:6.2f}s')

print('\n说明：误差 = 各算法 vs 参考的 max|Δpops|。')
print('场景 A 参考 = 全非 RWA（evolve._evolve_full，solve_ivp 实验室系）；')
print('场景 B 参考 = _evolve_full_hybrid（光频 RWA + 微波非 RWA，快 ~10^4 倍）。')
print('独立RWA = 旧逻辑（共享能级 60S 时微波被错误失谐 ~ω_opt，结构性错误）；')
print('级联RWA = 统一旋转系（pwc）；级联+BS = 加 Bloch–Siegert 一阶修正。')
print('精算推荐：method=auto → floquet（常数包络）/ ip4（时变包络），'
      '均为非 RWA 第一性原理，见 mw_bench_firstprinciples.py。')
print('算法预算 = ≤10× 级联 RWA 耗时；参考（第一性原理）慢是允许的。')
