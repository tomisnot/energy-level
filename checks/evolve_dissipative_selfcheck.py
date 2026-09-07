# -*- coding: utf-8 -*-
"""Lindblad 耗散引擎自检（evolve.py 算法 4：自发辐射等马尔可夫耗散）。

场景（D 系列，全部第一性原理独立参考）：
  D1 解析衰变 e^{−Γt}（两能级，无场；eig / rk4 / auto 三路）
  D2 CW 驱动 + 衰变 vs 独立 Liouvillian 本征参考（本文件内实现，同 tempsimu）
  D3 时变高斯包络 + 衰变 vs solve_ivp 向量化主方程（独立实现）
  D4 真实交换耦合（pair_core.exchange_coupling）+ 衰变 vs solve_me 参考
  D5 **multi 场 ac 约定回归**：ip4 / lindblad vs 第一性原理 solve_ivp
     （全额 env·ac；ip4 曾把 ac 减半——修复后此场景必须 <1e-3）
  D6 显式落点布居守恒（无 loss 通道，Σpops=1）
  D7 取消（progress_arr 置位 → RuntimeError）
  D8 纯退相（from==to → 相干 e^{−Γt/2}，解析）
  D9 无耗散一致性：lindblad（decays=[]）≡ ip4（RWA 时变场景）
  D10 自动档：ARC 权威寿命自动配置（6P3/2 Γ=1/τ(ARC)，里德堡排除，τ 阈值过滤）
  D11 无耗散一致性补充（另一包络）
  D12 numba 内核 ≡ numpy 回退（非对齐输出网格 → 部分步线性插值路径）

运行：D:\\python310\\python.exe -u checks\\evolve_dissipative_selfcheck.py
"""
import os
import sys
sys.stdout.reconfigure(encoding='utf-8', errors='replace')
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from scipy.integrate import solve_ivp
from arc import Rubidium87

from src import core, waveforms
# t74：evolve shim 已删——按新家直连
from src.core.atomic_levels import gaussian, square, make_levels
from src.core.field_build import (field, real_field, group_field,
                                  _field_geometry, _sublevel_row)
from src.core.evolve_util import (decay, auto_decays, _as_vector,
                                  _CONV)
from src.core.evolve_rwa import (evolve, _e_rot, _evolve_full,
                                 _evolve_full_hybrid, _ip4_reference)

from src import pair_core as pc
from checklib import Checker

cache = core.ArcCache(Rubidium87())
B = 22.0
ck = Checker('evolve_dissipative_selfcheck')


def states_sp(cache, B):
    """60S m=−0.5 / 60P1/2 m=−0.5（真实能量，60S↔60P 微波跃迁）。"""
    return make_levels(cache, [
        dict(n=60, l=0, j=0.5, m=-0.5,
             E_mhz=core._sublevel_energy(cache, 60, 0, 0.5, -0.5, B)),
        dict(n=60, l=1, j=0.5, m=-0.5,
             E_mhz=core._sublevel_energy(cache, 60, 1, 0.5, -0.5, B))])


def solve_me(H_mhz, Llist, rho0, t_ns):
    """常数 Lindblad 精确解（Liouvillian 本征分解；temp-sim solve_me 同款）。
    单位：H MHz → ×_CONV（rad/ns）；L 矩阵元已含 √(Γ×1e-3)。"""
    d = H_mhz.shape[0]
    Hh = np.asarray(H_mhz, complex) * _CONV
    I = np.eye(d)
    Lv = -1j * (np.kron(I, Hh) - np.kron(Hh.conj().T, I))
    for L in Llist:
        LdL = L.conj().T @ L
        Lv = Lv + (np.kron(L.conj(), L)
                   - 0.5 * (np.kron(I, LdL) + np.kron(LdL.conj().T, I)))
    w, V = np.linalg.eig(Lv)
    c0 = np.linalg.solve(V, rho0.ravel(order='F'))
    W = V * c0[None, :]
    diag_idx = np.arange(d) * (d + 1)
    E = np.exp(w[:, None] * t_ns[None, :])
    return (W[diag_idx, :] @ E).real.T          # (n_t, d) 布居（含 sink）


# ---------------- D1 解析衰变 ----------------
states = states_sp(cache, B)
t1 = np.linspace(0, 400.0, 401)
G1 = 10.0
exp1 = np.exp(-G1 * 1e-3 * t1)
for mth in ('auto', 'lindblad', 'lindblad_eig'):
    r = evolve(states, [], 1, t1, method=mth,
                      decays=[decay(1, G1)])
    err = float(np.max(np.abs(r['pops'][:, 1] - exp1)))
    cons = float(np.max(np.abs(r['pops'].sum(axis=1) + r['loss'] - 1.0)))
    ck.check(f'D1 解析衰变 e^{{-Γt}} [{mth}/{r["method"]}] err={err:.1e}',
             err < 1e-6 and cons < 1e-9, f'err={err} cons={cons}')

# ---------------- D2 CW 驱动 + 衰变 vs 独立 Liouvillian 参考 ----------------
G2 = 4.0
fld2 = field(0, 1, omega0_mhz=0.5, waveform=square(0, 1e9))
decs2 = [decay(1, G2)]                     # 60P → loss
ref_arr2, rwa_g2, _ = _ip4_reference(states, [fld2])
E_rot2 = _e_rot(np.array([s['E_mhz'] for s in states]), ref_arr2,
                       states, rwa_g2, [])
d2 = 3
H2 = np.zeros((d2, d2), complex)
H2[:2, :2] = np.diag(E_rot2).astype(float) + 0.25 * np.asarray(rwa_g2[0]['A'])
L2 = np.zeros((d2, d2), complex); L2[2, 1] = np.sqrt(G2 * 1e-3)
rho02 = np.zeros((d2, d2), complex); rho02[1, 1] = 1.0
t2 = np.linspace(0, 1000.0, 501)
p_ref2 = solve_me(H2, [L2], rho02, t2)[:, :2]
for mth in ('lindblad_eig', 'lindblad'):
    r = evolve(states, [fld2], 1, t2, method=mth, decays=decs2)
    err = float(np.max(np.abs(r['pops'] - p_ref2)))
    ck.check(f'D2 CW+衰变 [{mth}] err={err:.1e}', err < 1e-8, f'err={err}')

# ---------------- D3 时变高斯 + 衰变 vs solve_ivp 向量化主方程 ----------------
G3 = 15.0
fld3 = field(0, 1, omega0_mhz=0.5,
                    waveform=gaussian(500.0, 150.0))
decs3 = [decay(1, G3, 0)]                  # 60P → 60S（显式落点）
ref_arr3, rwa_g3, non_g3 = _ip4_reference(states, [fld3])
E_rot3 = _e_rot(np.array([s['E_mhz'] for s in states]), ref_arr3,
                       states, rwa_g3, non_g3)
N3 = 2

def H3_at(tv):
    H = np.diag(E_rot3).astype(float)
    for g in rwa_g3:
        env = float(np.asarray(g['waveform'](np.atleast_1d(tv))).flat[0])
        H = H + (g['omega0_mhz'] * env / 2.0) * g['A']
    return H

def rhs3(tv, rho_v):
    rho = rho_v.reshape((N3, N3), order='F')
    H = H3_at(tv)
    d = -1j * _CONV * (H @ rho - rho @ H)
    g = G3 * 1e-3
    d[0, 0] += g * rho[1, 1].real
    d[1, :] -= 0.5 * g * rho[1, :]
    d[:, 1] -= 0.5 * g * rho[:, 1]
    return d.ravel(order='F')

t3 = np.linspace(0, 1000.0, 501)
rho0v = np.zeros(N3 * N3, complex); rho0v[N3 * 1 + 1] = 1.0     # |1⟩⟨1|
sol3 = solve_ivp(rhs3, (t3[0], t3[-1]), rho0v, t_eval=t3,
                 rtol=1e-9, atol=1e-12)
p_ref3 = np.array([np.real(np.diag(sol3.y[:, m].reshape((N3, N3), order='F')))
                   for m in range(len(t3))])
r3 = evolve(states, [fld3], 1, t3, method='lindblad', decays=decs3)
err3 = float(np.max(np.abs(r3['pops'] - p_ref3)))
cons3 = float(np.max(np.abs(r3['pops'].sum(axis=1) - 1.0)))
ck.check(f'D3 时变高斯+衰变 err={err3:.1e} 守恒={cons3:.1e}',
         err3 < 1e-6 and cons3 < 1e-9, f'err={err3} cons={cons3}')

# ---------------- D4 真实交换耦合 + 衰变 vs solve_me ----------------
groups4 = [dict(n=60, l=0, j=0.5), dict(n=60, l=1, j=0.5)]
V_ex = pc.exchange_coupling(cache, groups4, R_um=30.0)
states4 = make_levels(cache, [
    dict(n=60, l=0, j=0.5, m=0.5,
         E_mhz=core._sublevel_energy(cache, 60, 0, 0.5, 0.5, B)),
    dict(n=60, l=1, j=0.5, m=0.5,
         E_mhz=core._sublevel_energy(cache, 60, 1, 0.5, 0.5, B))])
G4 = 4.0
fld4 = field(0, 1, omega0_mhz=2.0 * abs(V_ex),
                    waveform=square(0, 1e9))
decs4 = [decay(1, G4)]
t4 = np.linspace(0, 600.0, 301)
d4 = 3
H4 = np.zeros((d4, d4), complex)
H4[0, 1] = H4[1, 0] = V_ex                      # 交换 V（旋转系共振，对角 0）
L4 = np.zeros((d4, d4), complex); L4[2, 1] = np.sqrt(G4 * 1e-3)
rho04 = np.zeros((d4, d4), complex); rho04[1, 1] = 1.0
p_ref4 = solve_me(H4, [L4], rho04, t4)[:, :2]
for mth in ('lindblad_eig', 'lindblad'):
    r = evolve(states4, [fld4], 1, t4, method=mth, decays=decs4)
    err = float(np.max(np.abs(r['pops'] - p_ref4)))
    ck.check(f'D4 交换V={V_ex:.3f}MHz+衰变 [{mth}] err={err:.1e}',
             err < 1e-8, f'err={err}')

# ---------------- D5 multi 场 ac 约定回归（第一性原理，全额 env·ac） ----------------
specs5 = []
for (n, l, j) in [(60, 0, 0.5), (60, 1, 1.5)]:
    for m in (-0.5, 0.5):
        E = cache.E_mhz(n, l, j) + cache.gJ(l, j) * core.MU_B_MHZ_PER_G * m * B
        specs5.append(dict(n=n, l=l, j=j, m=float(m), E_mhz=float(E)))
states5 = make_levels(cache, specs5)
N5 = len(states5)
t5 = np.linspace(0, 150.0, 151)
opt5 = waveforms.SquareEnv(60.0, 25.0, height=1.0, fall_start_ns=110.0)
acv5 = np.array([70.0, -46.0, 30.0, -20.0])
mw5 = field(0, 1, omega0_mhz=10.0, waveform=gaussian(80.0, 30.0))
ac5 = {'kind': 'multi', 'M': np.diag(acv5), 'A': np.zeros((N5, N5)),
       'ac_shift': acv5, 'g_idx': [], 'e_idx': [],
       'omega0_mhz': 1.0, 'omega_L_mhz': 295723987.476, 'waveform': opt5}
fields5 = [mw5, ac5]
ref_arr5, rwa_g5, non_g5 = _ip4_reference(states5, fields5)
E5 = np.array([s['E_mhz'] for s in states5])
E_rot5 = E5 - ref_arr5
coupled5 = set()
for g in list(rwa_g5) + list(non_g5):
    coupled5.update(g['g_idx']); coupled5.update(g['e_idx'])
E_rot5 = E_rot5 - float(np.min(E_rot5[list(coupled5)]))
for i in range(N5):
    if i not in coupled5:
        E_rot5[i] = 0.0
H05 = np.diag(E_rot5).astype(float)
for g in rwa_g5:
    H05 = H05 + (g['omega0_mhz'] / 2.0) * g['A']

def H5_at(tv):
    H = H05.copy()
    for g in rwa_g5:
        env = float(np.asarray(g['waveform'](np.atleast_1d(tv))).flat[0])
        H = H + (g['omega0_mhz'] * env / 2.0) * g['A']
        a = g.get('ac_shift')
        if a is not None and np.any(np.asarray(a, float) != 0):
            H = H + env * np.diag(np.asarray(a, float))     # 全额 ac
    for g in non_g5:
        env = float(np.asarray(g['waveform'](np.atleast_1d(tv))).flat[0])
        H = H + (g['omega0_mhz'] * env
                 * np.cos(_CONV * g['omega_L_mhz'] * tv)) * g['A']
    return H

def rhs5(tv, y):
    return -1j * _CONV * (H5_at(tv) @ y)

v05 = np.zeros(N5, complex); v05[0] = 1.0
sol5 = solve_ivp(rhs5, (t5[0], t5[-1]), v05, t_eval=t5, rtol=1e-8, atol=1e-11)
p_ref5 = np.abs(sol5.y).T ** 2
r5a = evolve(states5, fields5, 0, t5, method='ip4')
err5a = float(np.max(np.abs(r5a['pops'] - p_ref5)))
ck.check(f'D5 ac 约定回归 ip4 err={err5a:.1e}（<1e-3；曾因 ac 减半达 0.96）',
         err5a < 1e-3, f'err={err5a}')
r5b = evolve(states5, fields5, 0, t5, method='lindblad', decays=[])
err5b = float(np.max(np.abs(r5b['pops'] - p_ref5)))
ck.check(f'D5 ac 约定回归 lindblad err={err5b:.1e}', err5b < 1e-6,
         f'err={err5b}')

# ---------------- D6 显式落点布居守恒 ----------------
r6 = evolve(states, [fld3], 1, t3, method='lindblad',
                   decays=[decay(1, 15.0, 0)])
cons6 = float(np.max(np.abs(r6['pops'].sum(axis=1) - 1.0)))
ck.check(f'D6 显式落点布居守恒 Σpops−1={cons6:.1e}', cons6 < 1e-9,
         f'cons={cons6}')

# ---------------- D7 取消 ----------------
pa = np.zeros(2); pa[1] = 1.0
try:
    evolve(states, [fld3], 1, t3, method='lindblad',
                  decays=decs3, progress_arr=pa)
    ck.check('D7 取消（progress_arr 置位 → RuntimeError）', False, '未抛错')
except RuntimeError as e:
    ck.check('D7 取消（progress_arr 置位 → RuntimeError）', '取消' in str(e),
             f'{type(e).__name__}: {e}')

# ---------------- D8 纯退相（from==to → 相干 e^{−Γt/2}） ----------------
G8 = 8.0
t8 = np.linspace(0, 500.0, 51)
r8 = evolve(states, [], [1 / np.sqrt(2), 1 / np.sqrt(2)], t8,
                   method='lindblad_eig', decays=[decay(1, G8, 1)],
                   record_amp=True)
coh8 = abs(r8['rho'][0, 1])
coh_exp = 0.5 * np.exp(-G8 * 1e-3 * t8[-1] / 2.0)
pops_ok = float(np.max(np.abs(r8['pops'] - 0.5)))
ck.check(f'D8 纯退相 相干 |ρ01|={coh8:.4f} vs e^{{-Γt/2}}={coh_exp:.4f}'
         f'（布居漂移 {pops_ok:.1e}）',
         abs(coh8 - coh_exp) < 1e-3 and pops_ok < 1e-9,
         f'coh={coh8} exp={coh_exp}')

# ---------------- D9 无耗散一致性：lindblad ≡ ip4 ----------------
r9a = evolve(states, [fld3], 1, t3, method='lindblad', decays=[])
r9b = evolve(states, [fld3], 1, t3, method='ip4')
err9 = float(np.max(np.abs(r9a['pops'] - r9b['pops'])))
ck.check(f'D9 无耗散 lindblad ≡ ip4 err={err9:.1e}', err9 < 1e-5,
         f'err={err9}')

# ---------------- D10 自动档：ARC 权威寿命自动配置 ----------------
# 基组：6P3/2 四个 mJ + 60S（里德堡，τ~230 µs 应被排除）
states10 = make_levels(cache, [
    dict(n=6, l=1, j=1.5, m=float(m),
         E_mhz=core._sublevel_energy(cache, 6, 1, 1.5, float(m), B))
    for m in (-1.5, -0.5, 0.5, 1.5)] + [
    dict(n=60, l=0, j=0.5, m=0.5,
         E_mhz=core._sublevel_energy(cache, 60, 0, 0.5, 0.5, B))])
auto = auto_decays(cache, states10)
i6p = [i for i, s in enumerate(states10) if s['n'] == 6]
tau_6p = cache.atom.getStateLifetime(6, 1, 1.5, temperature=0.0)
gam_6p = 1e-6 / float(tau_6p)
ok10 = (len(auto) == len(i6p)
        and all(a['to'] is None for a in auto)
        and all(a['from'] in i6p for a in auto)
        and all(abs(a['gamma_mhz'] - gam_6p) < 1e-9 * gam_6p
                for a in auto))
ck.check(f'D10 自动档 ARC 寿命：6P3/2 ×{len(i6p)} 通道 Γ={gam_6p:.4f} MHz，'
         f'60S 排除', ok10,
         f'channels={len(auto)} tau_6p={tau_6p}')
# τ 阈值过滤：τ_max=10 ns < 6P(118 ns) → 空
auto0 = auto_decays(cache, states10, tau_max_s=1e-8)
ck.check('D10 自动档 τ 阈值过滤（τ_max=10ns → 0 通道）',
         len(auto0) == 0, f'channels={len(auto0)}')

# ---------------- D11 无耗散一致性补充：lindblad ≡ ip4（RWA 场景，另一包络） ----------------
fld11 = field(0, 1, omega0_mhz=0.3,
                     waveform=gaussian(400.0, 120.0))
t11 = np.linspace(0, 800.0, 401)
r11a = evolve(states, [fld11], 1, t11, method='lindblad', decays=[])
r11b = evolve(states, [fld11], 1, t11, method='ip4')
err11 = float(np.max(np.abs(r11a['pops'] - r11b['pops'])))
ck.check(f'D11 无耗散 lindblad ≡ ip4（另一包络）err={err11:.1e}',
         err11 < 1e-5, f'err={err11}')

# ---------------- D12 numba 内核 ≡ numpy 回退（非对齐输出→部分步插值） ----------------
t12 = np.linspace(0, 100.0, 100)          # 输出网格与积分步长不对齐 → 部分步走插值
fld12 = field(0, 1, omega0_mhz=0.4,
                     waveform=gaussian(45.0, 13.0))
decs12 = [decay(1, 12.0, 0)]
r12a = evolve(states, [fld12], 1, t12, method='lindblad',
                     decays=decs12)                        # numba（默认）
r12b = evolve(states, [fld12], 1, t12, method='lindblad',
                     backend='cpu', decays=decs12)         # numpy 回退
err12 = float(np.max(np.abs(r12a['pops'] - r12b['pops'])))
cons12 = float(np.max(np.abs(r12a['pops'].sum(axis=1) - 1.0)))
ck.check(f'D12 numba ≡ numpy（非对齐输出）err={err12:.1e} 守恒={cons12:.1e}',
         err12 < 1e-4 and cons12 < 1e-9, f'err={err12} cons={cons12}')

ok = ck.summary()
sys.exit(0 if ok else 1)
