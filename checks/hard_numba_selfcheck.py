# -*- coding: utf-8 -*-
"""hard 路线自检（t8 验收）：numba 三件套加速的正确性 + 性能。

1. 已知答案 Rabi（A=5MHz 手动场，同 _tf_smoke[A] 场景）：
   fast(驱动) vs 原版 evolve_hard max|Δpops| ≤ 1e-6，且 P1max≈0.5021
2. 22态 620ns：fast vs 原版基线存档 max|Δpops| ≤ 1e-6，且计时 ≤ 5s（验收）
3. 两体 hard：①V_dd=0 直积 ≡ 单原子 hard（≤1e-4）
               ②物理2态 V_dd≠0 交换 ≡ 实验室系稠密参考（≤1e-5）
运行：python -u checks/hard_numba_selfcheck.py（首次含 numba 编译 ~1-2 分钟）
"""
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from arc import Rubidium87  # noqa: E402
from src.core import core, pair_core  # noqa: E402
from src.core.evolve import make_levels  # noqa: E402
from src.core.waveforms import GaussianEnv, Wave  # noqa: E402
from src.core.evolve_two_field import (build_two_field,  # noqa: E402
                                       evolve_hard)
from src.core.evolve_exact import _CONV  # noqa: E402
from src.core.hard_numba import evolve_hard_fast  # noqa: E402
from src.core.hard_pair import evolve_hard_pair  # noqa: E402

CACHE = core.ArcCache(Rubidium87())
PASS, FAIL = [], []


def check(name, cond, msg=''):
    (PASS if cond else FAIL).append(name)
    print(f'  {"✅" if cond else "❌"} [{name}] {msg}')


# ---------- 1. 已知答案 Rabi ----------
print('[1] 已知答案 Rabi（A=5MHz，P1max≈0.5021）：fast vs 原版')
states2 = make_levels(CACHE, [
    dict(n=60, l=0, j=0.5, kind='mJ', m=-0.5, E_mhz=0.0),
    dict(n=60, l=0, j=0.5, kind='mJ', m=0.5, E_mhz=17236.743)])
A2 = np.zeros((2, 2)); A2[0, 1] = A2[1, 0] = 5.0
tw = GaussianEnv(100, 20, 1.0)   # σ=20ns（同 _tf_smoke[A]：P1max≈0.5021）
meta2 = dict(E_rot=np.array([0.0, 17236.743]), N=2, opt_terms=[],
             mw_groups=[dict(omega_mhz=17236.743, As=[A2], envs=[tw],
                             labels=['w1'])],
             disc_times=[])
t = np.linspace(0.0, 300.0, 301)
r0 = evolve_hard(states2, meta2, 0, t, rtol=1e-8, atol=1e-10,
                 log=lambda *a: None)
r1 = evolve_hard_fast(states2, meta2, 0, t, rtol=1e-8, atol=1e-10,
                      log=lambda *a: None, driver='numba')
pk = np.max(r0['pops'][:, 1])
d1 = np.max(np.abs(r0['pops'] - r1['pops']))
check('Rabi 物理（P1max≈0.5021）', abs(pk - 0.5021) < 0.01, f'P1max={pk:.4f}')
check('Rabi fast ≡ 原版', d1 < 1e-6, f'max|Δ|={d1:.2e}')

# ---------- 2. 22态 620ns 验收 ----------
print('[2] 22态 620ns：fast vs 基线 + ≤5s 验收')
from tempsimu._scenario_models import load_scene
cache, states_full, wa, wb, _ = load_scene()
psi0 = next(i for i, s in enumerate(states_full)
            if (s['n'], s['l'], s['j']) == (60, 0, 0.5) and abs(s['m']-0.5) < 1e-9)
meta = build_two_field(cache, states_full, wa, mw_M=2)
t620 = np.linspace(0.0, 620.0, 125)
t0 = time.perf_counter()
rf = evolve_hard_fast(states_full, meta, psi0, t620, rtol=1e-8, atol=1e-10,
                      log=lambda *a: None, driver='numba')
dt620 = time.perf_counter() - t0
t0 = time.perf_counter()
rf2 = evolve_hard_fast(states_full, meta, psi0, t620, rtol=1e-8, atol=1e-10,
                       log=lambda *a: None, driver='numba')
dt620b = time.perf_counter() - t0
base = np.load(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                            '..', 'tempsimu', 'tf_hard_620.npz'))
d620 = np.max(np.abs(base['pops'] - rf['pops']))
check('620ns fast ≡ 原版基线', d620 < 1e-6, f'max|Δ|={d620:.2e}')
check('620ns 验收 ≤5s（缓存热）', dt620b <= 5.0,
      f'{dt620b:.2f}s（首次含缓存加载 {dt620:.2f}s；原版 173.2s）')
print(f'  范数偏差={rf["error_report"]["max_norm_dev"]:.2e}')

# ---------- 3. 两体 hard ----------
print('[3] 两体 hard：V_dd=0 直积 + 物理2态交换')
meta_A = build_two_field(cache, states_full, wa, mw_M=2)
meta_B = build_two_field(cache, states_full, wb, mw_M=2)
t3 = np.linspace(0.0, 300.0, 61)
rA = evolve_hard(states_full, meta_A, psi0, t3, rtol=1e-8, atol=1e-10,
                 log=lambda *a: None, numba=True)
rB = evolve_hard(states_full, meta_B, psi0, t3, rtol=1e-8, atol=1e-10,
                 log=lambda *a: None, numba=True)
rp = evolve_hard_pair(states_full, meta_A, meta_B, psi0, psi0, t3,
                      vdd_matrix=None, rtol=1e-8, atol=1e-10,
                      log=lambda *a: None)
dA = np.max(np.abs(rp['pops_A'] - rA['pops']))
dB = np.max(np.abs(rp['pops_B'] - rB['pops']))
prod = rA['pops'][:, :, None] * rB['pops'][:, None, :]
d2d = np.max(np.abs(rp['pops_2d'] - prod.reshape(len(t3), -1)))
check('V_dd=0 pops_A ≡ 单原子', dA < 1e-4, f'max|Δ|={dA:.2e}')
check('V_dd=0 pops_B ≡ 单原子', dB < 1e-4, f'max|Δ|={dB:.2e}')
check('V_dd=0 pops_2d ≡ 直积', d2d < 1e-4, f'max|Δ|={d2d:.2e}')

# 物理2态交换 vs 实验室系参考
states2p = make_levels(CACHE, [
    dict(n=60, l=0, j=0.5, kind='mJ', m=-0.5, E_mhz=0.0),
    dict(n=60, l=1, j=1.5, kind='mJ', m=-1.5, E_mhz=17236.743)])
N2 = len(states2p)
mwp = [Wave('mw', 17236.743, GaussianEnv(100, 40, 1.0), pol='σ-',
            offset_ns=0.0, intensity_mwcm2=2.0)]
m2A = build_two_field(CACHE, states2p, mwp, mw_M=2)
m2B = build_two_field(CACHE, states2p, mwp, mw_M=2)
A_mw = m2A['mw_groups'][0]['As'][0]
V2 = pair_core.vdd_sparse(CACHE, [dict(s) for s in states2p], 10.0, 45.0, 0.0)
t4 = np.linspace(0.0, 50.0, 101)
r4 = evolve_hard_pair(states2p, m2A, m2B, 0, 0, t4, vdd_matrix=V2,
                      rtol=1e-9, atol=1e-11, log=lambda *a: None)
from scipy.integrate import solve_ivp
E_lab2 = np.array([s['E_mhz'] for s in states2p], float)
E_p2 = np.array([E_lab2[i] + E_lab2[j] for i in range(N2) for j in range(N2)])
AI = np.kron(A_mw, np.eye(N2)) + np.kron(np.eye(N2), A_mw)
psi0_2d = np.zeros(N2 * N2, complex); psi0_2d[0] = 1.0
env = m2A['mw_groups'][0]['envs'][0]

def rhs_lab(tv, y):
    c = env(tv) * np.cos(_CONV * 17236.743 * tv)
    return -1j * _CONV * ((np.diag(E_p2) + c * AI + V2.toarray()) @ y)

sol = solve_ivp(rhs_lab, (0.0, 50.0), psi0_2d, method='DOP853',
                t_eval=t4, rtol=1e-9, atol=1e-11)
pops_ref = np.abs(sol.y).T ** 2
d4 = np.max(np.abs(r4['pops_2d'] - pops_ref))
check('物理2态 V_dd≠0 ≡ 实验室系参考', d4 < 1e-5, f'max|Δpops|={d4:.3e}')
ex = np.max(np.abs(r4['pops_2d'][:, 1] - r4['pops_2d'][:, N2]))
check('交换通道对称 |S,P>↔|P,S>', ex < 1e-6, f'max|Δ|={ex:.2e}')

print(f'\n{"="*50}')
print(f'自检结果: {len(PASS)} pass / {len(PASS)+len(FAIL)} total')
if FAIL:
    print(f'失败项: {FAIL}')
else:
    print('全部通过！')
