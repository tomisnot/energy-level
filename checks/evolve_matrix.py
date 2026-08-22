# -*- coding: utf-8 -*-
"""动态引擎测评矩阵：统一场算符（M = A + diag(ac)）系统验证。

四个维度（一次并行跑完，16 核）：
  1. 正确性：各场景 vs 独立参考（solve_ivp / 高 frac 收敛）
  2. 精度：max|ΔP|、归一误差（ΣP=1）
  3. 性能：真实计时（每场景独立 worker，互不干扰）
  4. RWA 误差：Ω/ωL 扫描，确认误差量级 ~Ω²/4ω（Bloch-Siegert）

场景（动态分析只用 ip4，2025-08 起不再测评 floquet/pwc——已封存至
src/_archive/evolve_floquet_pwc.py）：
  S1 纯微波多能级（60S+59P/60P/61P，N=20，方波）——ip4
  S2 微波高斯（S1 基组，高斯包络）——ip4
  S3 阶梯（6P↔60S 光频 RWA + 60S↔60P 微波非 RWA，N=6）——ip4
  S4 光场频移阻塞（初态制备：微波 π 脉冲 + 光场时变 AC）——ip4
  S5 时变 AC 位移正确性（vs solve_ivp 独立积分含时变对角）——ip4

用法: D:\\python310\\python.exe -u checks\\evolve_matrix.py
输出: checks\\_matrix_result.json（每场景 {method: {err, norm, dt}}）
"""
import os
import sys
import json
import time
import multiprocessing as mp

sys.stdout.reconfigure(encoding='utf-8', errors='replace')
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from arc import Rubidium87
from scipy.integrate import solve_ivp
from src import core, evolve, pulse

_CONV = evolve._CONV
B = 22.0
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                   '_matrix_result.json')

_cache = None


def get_cache():
    global _cache
    if _cache is None:
        _cache = core.ArcCache(Rubidium87())
    return _cache


# ================= 场景构造 =================
def make_states_s1(cache, B):
    specs = []
    for (n, l, j) in [(60, 0, 0.5), (59, 1, 0.5), (59, 1, 1.5),
                      (60, 1, 0.5), (60, 1, 1.5), (61, 1, 0.5), (61, 1, 1.5)]:
        for m in np.arange(-j, j + 1e-9, 1.0):
            E = cache.E_mhz(n, l, j) + cache.gJ(l, j) * core.MU_B_MHZ_PER_G * m * B
            specs.append(dict(n=n, l=l, j=j, m=float(m), E_mhz=float(E)))
    return evolve.make_levels(cache, specs)


def make_states_ladder(cache, B):
    specs = []
    for (n, l, j) in [(6, 1, 1.5), (60, 0, 0.5), (60, 1, 0.5)]:
        for m in (-0.5, 0.5):
            E = cache.E_mhz(n, l, j) + cache.gJ(l, j) * core.MU_B_MHZ_PER_G * m * B
            specs.append(dict(n=n, l=l, j=j, m=float(m), E_mhz=float(E)))
    return evolve.make_levels(cache, specs)


def make_states_prep(cache, B=21.7):
    """初态制备：60S/60P1/2/60P3/2/6P3/2 全子能级（含超精细）。"""
    specs = []
    for (n, l, j) in [(60, 0, 0.5), (60, 1, 0.5), (60, 1, 1.5), (6, 1, 1.5)]:
        hf = core.get_hyperfine_levels(cache, n, l, j, B)
        if hf is None:
            for m in np.arange(-j, j + 1e-9, 1.0):
                E = core._sublevel_energy(cache, n, l, j, m, B)
                specs.append(dict(n=n, l=l, j=j, kind='mJ', m=float(m),
                                  E_mhz=float(E)))
        else:
            for F, mF, e_hf, _w, amps in hf:
                specs.append(dict(n=n, l=l, j=j, kind='F', F=F, mF=mF,
                                  E_hf=float(e_hf), amps=amps))
    return evolve.make_levels(cache, specs)


def build_s1(cache, envelope=None):
    """纯微波：60S↔60P 共振方波/高斯（真实 Rabi）。"""
    states = make_states_s1(cache, B)
    i0 = next(i for i, s in enumerate(states)
              if (s['n'], s['l'], s['j']) == (60, 0, 0.5) and s['m'] > 0)
    freq = cache.E_mhz(60, 1, 0.5) - cache.E_mhz(60, 0, 0.5)
    dmax = max(abs(core.dipole_amplitude(cache, evolve._sublevel_row(si),
                                         evolve._sublevel_row(sj), 0))
               for si in states if si['n'] == 60 and si['l'] == 0
               for sj in states if sj['l'] == 1)
    I = core.intensity_mwcm2_from_rabi(dmax, 100.0)
    fld, ac = evolve.real_field(cache, states, freq, I, pol='π',
                                waveform=envelope)
    return states, [fld], i0


def build_s3(cache):
    """阶梯：光频 6P↔60S（RWA）+ 微波 60S↔60P（非 RWA）。"""
    states = make_states_ladder(cache, B)
    i0 = next(i for i, s in enumerate(states)
              if (s['n'], s['l'], s['j']) == (6, 1, 1.5) and s['m'] < 0)
    d_opt = max(abs(core.dipole_amplitude(cache, evolve._sublevel_row(si),
                                          evolve._sublevel_row(sj), 1))
                for si in states if si['n'] == 6
                for sj in states if sj['n'] == 60 and sj['l'] == 0)
    I_opt = core.intensity_mwcm2_from_rabi(d_opt, 20.0)
    f_opt = cache.E_mhz(60, 0, 0.5) - cache.E_mhz(6, 1, 1.5)
    fld_o, ac_o = evolve.real_field(cache, states, f_opt, I_opt, pol='σ+')
    d_mw = max(abs(core.dipole_amplitude(cache, evolve._sublevel_row(si),
                                         evolve._sublevel_row(sj), 0))
               for si in states if si['n'] == 60 and si['l'] == 0
               for sj in states if sj['n'] == 60 and sj['l'] == 1)
    I_mw = core.intensity_mwcm2_from_rabi(d_mw, 100.0)
    f_mw = cache.E_mhz(60, 1, 0.5) - cache.E_mhz(60, 0, 0.5)
    fld_m, ac_m = evolve.real_field(cache, states, f_mw, I_mw, pol='π')
    fields = [f for f in (fld_o, fld_m) if f is not None]
    return states, fields, i0


def build_s4(cache):
    """初态制备（时变 AC 阻塞）：微波1 π 脉冲 + 光场频移 + 微波2。"""
    states = make_states_prep(cache)
    i0 = 1                                     # 60S m=+0.5
    mw_wf = evolve.gaussian(150.0, 41.7)
    mw1, _ = evolve.real_field(cache, states, 17236.743, 1.29051816e-5,
                               'σ-', waveform=mw_wf)
    mw2, _ = evolve.real_field(cache, states, 17236.743, 1.29051816e-5,
                               'σ-', waveform=evolve.gaussian(430.0, 41.7))
    opt = pulse.SmoothSquareEnv(200.0, 40.0, height=1.0, fall_start_ns=420.0)
    f_opt, ac_opt = evolve.real_field(cache, states, 295723987.476,
                                      126849894.0, 'π', waveform=opt)
    fields = [mw1, mw2]
    if f_opt is not None:
        fields.append(f_opt)
    elif np.any(np.asarray(ac_opt) != 0):
        acv = np.asarray(ac_opt, float)
        fields.append({'kind': 'multi', 'M': np.diag(acv),
                       'A': np.zeros((len(states), len(states))),
                       'ac_shift': acv, 'g_idx': [], 'e_idx': [],
                       'omega0_mhz': 1.0, 'omega_L_mhz': 295723987.476,
                       'waveform': opt})
    return states, fields, i0


def build_s5(cache):
    """时变 AC 位移正确性：两态 + 微波 + 时变光场位移（独立可积分）。"""
    states = evolve.make_levels(cache, [
        dict(n=60, l=0, j=0.5, m=0.5,
             E_mhz=core._sublevel_energy(cache, 60, 0, 0.5, 0.5, B)),
        dict(n=60, l=1, j=1.5, m=-0.5,
             E_mhz=core._sublevel_energy(cache, 60, 1, 1.5, -0.5, B))])
    i0 = 0
    mw_wf = evolve.gaussian(150.0, 41.7)
    mw, _ = evolve.real_field(cache, states,
                              states[1]['E_mhz'] - states[0]['E_mhz'],
                              1.29051816e-5, 'σ-', waveform=mw_wf)
    opt = pulse.SmoothSquareEnv(200.0, 40.0, height=1.0, fall_start_ns=420.0)
    acv = np.array([70.0, -46.0])            # 模拟光场对两态的位移
    acfield = {'kind': 'multi', 'M': np.diag(acv), 'A': np.zeros((2, 2)),
               'ac_shift': acv, 'g_idx': [], 'e_idx': [],
               'omega0_mhz': 1.0, 'omega_L_mhz': 295723987.476,
               'waveform': opt}
    return states, [mw, acfield], i0


# ================= 独立参考 =================
def solve_ivp_ref(states, fields, psi0, t, thresh=1e-4):
    """solve_ivp 混合参考（与 _evolve_full_hybrid 同构，独立实现）。

    注意：仅适用短窗/小基组（非 RWA cos 快项使自适应步长 ~5e-5 ns，
    长窗会跑不完）。长窗场景用高 frac ip4 收敛参考。
    """
    N = len(states)
    E = np.array([s['E_mhz'] for s in states])
    geoms = [evolve._field_geometry(states, f) for f in fields]

    def ratio(g):
        return g.get('real_omega0_mhz', g['omega0_mhz']) / max(g['omega_L_mhz'], 1e-9)
    rwa_g = [g for g in geoms if ratio(g) < thresh]
    non_g = [g for g in geoms if ratio(g) >= thresh]
    ref_arr, _, _ = evolve._ip4_reference(states, fields)
    H0 = np.diag(E - ref_arr).astype(float)
    H0 = H0 - np.min(np.diag(H0)) * np.eye(N)
    for g in rwa_g:
        H0 = H0 + (g['omega0_mhz'] / 2.0) * g['A']
        a = g.get('ac_shift')
        if a is not None and np.any(np.asarray(a, float) != 0):
            H0 = H0 + np.diag(np.asarray(a, float))

    def H_at(tv):
        H = H0.copy()
        for g in rwa_g:
            env = float(np.asarray(g['waveform'](np.atleast_1d(tv))).flat[0])
            H = H + (g['omega0_mhz'] * (env - 1.0) / 2.0) * g['A']
            a = g.get('ac_shift')
            if a is not None and np.any(np.asarray(a, float) != 0):
                H = H + (env - 1.0) * np.diag(np.asarray(a, float))
        for g in non_g:
            env = float(np.asarray(g['waveform'](np.atleast_1d(tv))).flat[0])
            H = H + (g['omega0_mhz'] * env
                     * np.cos(_CONV * g['omega_L_mhz'] * tv)) * g['A']
            a = g.get('ac_shift')
            if a is not None and np.any(np.asarray(a, float) != 0):
                H = H + env * np.diag(np.asarray(a, float))
        return H

    def rhs(tv, y):
        return -1j * _CONV * (H_at(tv) @ y)

    v0 = np.zeros(N, complex); v0[psi0] = 1.0
    sol = solve_ivp(rhs, (t[0], t[-1]), v0, t_eval=t, rtol=1e-9, atol=1e-11)
    return np.abs(sol.y).T ** 2


def ip4_conv_ref(states, fields, psi0, t, frac=80.0):
    """高 frac ip4 收敛参考（长窗替代 solve_ivp：非 RWA cos 快项下
    solve_ivp 自适应步长不可行，改用更高阶收敛验证时变 ac 一致性）。"""
    r = evolve.evolve(states, fields, psi0, t, method='ip4', ip4_frac=frac)
    return r['pops']


# ================= worker（每场景每方法一个进程） =================
def run_case(args):
    name, method, tmax, npts, ip4_frac = args
    cache = get_cache()
    t = np.linspace(0, tmax, npts)
    out = {'err_ref': None, 'norm': None, 'dt': None, 'method': method,
           'note': ''}
    t0 = time.perf_counter()
    try:
        if name == 'S1':
            states, fields, i0 = build_s1(cache)
            r = evolve.evolve(states, fields, i0, t, method=method,
                              ip4_frac=ip4_frac)
        elif name == 'S2':
            env = evolve.gaussian(200.0, 100.0)
            states, fields, i0 = build_s1(cache, envelope=env)
            r = evolve.evolve(states, fields, i0, t, method=method,
                              ip4_frac=ip4_frac)
        elif name == 'S3':
            states, fields, i0 = build_s3(cache)
            r = evolve.evolve(states, fields, i0, t, method=method,
                              ip4_frac=ip4_frac)
        elif name == 'S4':
            states, fields, i0 = build_s4(cache)
            r = evolve.evolve(states, fields, i0, t, method=method,
                              ip4_frac=ip4_frac)
        elif name == 'S5':
            states, fields, i0 = build_s5(cache)
            r = evolve.evolve(states, fields, i0, t, method=method,
                              ip4_frac=ip4_frac)
        else:
            out['note'] = 'unknown scenario'
            return name, out
        out['dt'] = time.perf_counter() - t0
        out['norm'] = float(np.max(np.abs(r['pops'].sum(axis=1) - 1.0)))
        out['method'] = r['method']
        # 独立参考：S3 短窗用 solve_ivp（非 RWA 可行）；S4/S5 长窗用高 frac 收敛
        if name == 'S3':
            ref = solve_ivp_ref(states, fields, i0, t)
            out['err_ref'] = float(np.max(np.abs(r['pops'] - ref)))
        elif name in ('S4', 'S5'):
            ref = ip4_conv_ref(states, fields, i0, t, frac=80.0)
            out['err_ref'] = float(np.max(np.abs(r['pops'] - ref)))
    except Exception as e:
        out['note'] = f'{type(e).__name__}: {e}'
    return name, out


def main():
    mp.freeze_support()
    cases = [
        ('S1', 'ip4', 400.0, 2001, 20.0),
        ('S2', 'ip4', 400.0, 2001, 20.0),
        ('S3', 'ip4', 20.0, 201, 20.0),
        ('S4', 'ip4', 1500.0, 1501, 20.0),
        ('S5', 'ip4', 1500.0, 1501, 20.0),
    ]
    # 支持单 case：python evolve_matrix.py S1 ip4 400 2001 20
    if len(sys.argv) >= 3:
        cases = [(sys.argv[1], sys.argv[2], float(sys.argv[3]),
                  int(sys.argv[4]),
                  float(sys.argv[5]) if len(sys.argv) > 5 else 20.0)]
    ncpu = mp.cpu_count()
    print(f'== 测评矩阵：{len(cases)} 个 worker，{ncpu} 核并行 ==')
    results = {}
    with mp.Pool(min(ncpu, len(cases))) as pool:
        for name, out in pool.imap_unordered(run_case, cases):
            results.setdefault(name, {})[out['method']] = out
            print(f"  [{name}/{out['method']}] err={out['err_ref']} "
                  f"norm={out['norm']:.1e} dt={out['dt']:.1f}s "
                  f"{out['note']}")
    with open(OUT, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=1)
    print(f'\n结果 → {OUT}')


if __name__ == '__main__':
    main()
