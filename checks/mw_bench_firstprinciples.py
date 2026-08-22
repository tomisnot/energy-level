# -*- coding: utf-8 -*-
"""微波多能级驱动：第一性原理候选算法实测基准（真实计时，不估算）。

同一物理场景、同一初态/时间网格下对比：

现有生产方法（evolve.py）:
  ip4      相互作用绘景 RK4（光频 RWA + 微波非 RWA，第一性原理，默认）
  full      solve_ivp 实验室系非 RWA（第一性原理；Python 回调 + 自适应步长，慢）
  full_fast solve_ivp 混合（光频 RWA + 微波非 RWA）
  （pwc/floquet 已封存至 src/_archive/evolve_floquet_pwc.py，2025-08）

候选新方法（第一性原理，误差可收敛到机器精度）:
  fstep     固定步长中点法（实验室系非 RWA；cos/包络预计算为标量数组，
            每步一次 numpy eigh —— 无 Python 回调、无自适应开销）
  floquet   精确 Floquet 传播子（仅常数包络：用 fstep 精确积分**一个周期**
            得传播子 U(T)，本征分解后任意时刻 ψ(t)=U(τ)·U(T)^n·ψ0。
            对周期驱动是 TDSE 的精确解，非光子块截断近似）

场景:
  S1 纯微波多能级：60S + 59P/60P/61P 全部子能级，B=22 G，N=20，方波包络
     （59P/61P 流形距 ω_L 约 ±0.9 GHz，真实地被弱离共振驱动 → 真多能级）
  S2 同 S1 但高斯包络（fstep 收敛表验证包络时变；微波物理同 S1）
  S3 阶梯：光频 6P↔60S（RWA，误差~1e-7）+ 微波 60S↔60P（非 RWA）
     （fstep-hybrid vs full_fast；微波部分第一性原理）

用法: D:\\python310\\python.exe -u checks\\mw_bench_firstprinciples.py [S1|S2|S3|all]
"""
import os
import sys
import time

sys.stdout.reconfigure(encoding='utf-8', errors='replace')
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from arc import Rubidium87
from src import core, evolve

_CONV = 2 * np.pi * 1e-3          # MHz·ns → rad
B_FIELD = 22.0                    # G
T_TOTAL = 400.0                   # ns
N_OUT = 2001


# ---------------- 能级构造（含 Zeeman） ----------------
def make_states(cache, B):
    """60S + 59P/60P/61P 全部 mJ 子能级（B 场 Zeeman 分裂）。N=20。"""
    specs = []
    for (n, l, j) in [(60, 0, 0.5), (59, 1, 0.5), (59, 1, 1.5),
                      (60, 1, 0.5), (60, 1, 1.5), (61, 1, 0.5), (61, 1, 1.5)]:
        gJ = cache.gJ(l, j)
        for m in np.arange(-j, j + 1e-9, 1.0):
            E = cache.E_mhz(n, l, j) + gJ * core.MU_B_MHZ_PER_G * m * B
            specs.append(dict(n=n, l=l, j=j, m=float(m), E_mhz=float(E),
                              label=f'{n}{"SPD"[l]}{j:g} mJ={m:+.1f}'))
    return evolve.make_levels(cache, specs)


def make_states_ladder(cache, B):
    """阶梯 S3：6P3/2(mJ=±1/2) + 60S(±1/2) + 60P1/2(±1/2)。N=6。"""
    specs = []
    for (n, l, j) in [(6, 1, 1.5), (60, 0, 0.5), (60, 1, 0.5)]:
        gJ = cache.gJ(l, j)
        for m in (-0.5, 0.5):
            E = cache.E_mhz(n, l, j) + gJ * core.MU_B_MHZ_PER_G * m * B
            specs.append(dict(n=n, l=l, j=j, m=float(m), E_mhz=float(E),
                              label=f'{n}{"SPD"[l]}{j:g} mJ={m:+.1f}'))
    return evolve.make_levels(cache, specs)


# ---------------- 积分器 1：固定步长中点法（实验室系非 RWA） ----------------
def integrate_midpoint(H0, terms, psi0, t_out, dt):
    """H(t_m) = H0 + Σ_k s_k[m]·A_k；terms = [(A_k, s_k_array)]。

    s_k[m] = Ω_k·env(t_m)·cos(ω_k t_m)（非 RWA）。每步一次 numpy eigh。
    """
    N = len(psi0)
    M = int(np.ceil(t_out[-1] / dt))
    mats = [A for A, _ in terms]
    s = np.array([arr for _, arr in terms])          # (n_terms, M)
    t_out = np.asarray(t_out)
    out_step = np.minimum(np.ceil(t_out / dt).astype(int), M)
    pops = np.zeros((len(t_out), N))
    pops[0] = np.abs(psi0) ** 2
    psi = np.array(psi0, complex)
    rec = 1
    for m in range(M):
        H = H0.copy()
        for k in range(len(mats)):
            H = H + s[k, m] * mats[k]
        e, U = np.linalg.eigh(H)
        ph = np.exp(-1j * _CONV * dt * e)
        psi = U @ (ph * (U.T @ psi))
        while rec < len(t_out) and out_step[rec] == m + 1:
            pops[rec] = np.abs(psi) ** 2
            rec += 1
        if rec >= len(t_out):
            break
    return pops


# ---------------- 积分器 1b：固定步长 RK4（4 阶，纯矩阵乘，无需 eigh） ----------------
def integrate_rk4(H0, terms, psi0, t_out, dt):
    """H(t) = H0 + Σ_k s_k(t)·A_k；terms = [(A_k, s_k_func)]。

    s_k_func(t_ns) → 标量（Ω_k·env(t)·cos(ω_k t) 或 RWA 常数）。
    4 阶龙格库塔：每步 4 次 H(t)@ψ（无矩阵指数）。误差 O(dt⁴)，步长可到
    载波周期 /20~40 仍 ~1e-6（中点法 2 阶需要 /200~400，不可用）。
    """
    N = len(psi0)
    M = int(np.ceil(t_out[-1] / dt))
    mats = [A for A, _ in terms]
    funcs = [f for _, f in terms]
    t_out = np.asarray(t_out)
    out_step = np.minimum(np.ceil(t_out / dt).astype(int), M)
    pops = np.zeros((len(t_out), N))
    pops[0] = np.abs(psi0) ** 2
    psi = np.array(psi0, complex)
    rec = 1

    def H_at(t):
        Hh = H0.copy()
        for k in range(len(mats)):
            Hh = Hh + float(funcs[k](t)) * mats[k]
        return Hh

    c = -1j * _CONV
    for m in range(M):
        tm = m * dt
        k1 = c * (H_at(tm) @ psi)
        k2 = c * (H_at(tm + dt / 2) @ (psi + dt / 2 * k1))
        k3 = c * (H_at(tm + dt / 2) @ (psi + dt / 2 * k2))
        k4 = c * (H_at(tm + dt) @ (psi + dt * k3))
        psi = psi + dt / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
        while rec < len(t_out) and out_step[rec] == m + 1:
            pops[rec] = np.abs(psi) ** 2
            rec += 1
        if rec >= len(t_out):
            break
    return pops


# ---------------- 积分器 1c：相互作用绘景 RK4（第一性原理，消除 H0 相位累积） ----------------
def integrate_rk4_ip(E, terms, psi0, t_out, dt):
    """H0=diag(E)（实验室系对角）。ψ̃ = e^{+iH0t}ψ → ψ̃' = -i Ṽ(t) ψ̃，
    Ṽ_ij(t) = Σ_k s_k(t)·A_k,ij·e^{i(E_i−E_j)t}（s_k = Ω_k·env(t)·cos(ω_k t)）。

    关键：H0 相位被精确除去 → 每步误差 ~ (‖V‖·dt)³ ∝ (Ω·dt)³，与 ‖H0‖ 无关；
    dt = 载波周期 T/20 即可（实验室系 RK4 需要 T/200+，因 H0 相位误差累积）。
    记录：输出时刻从步首做一次部分 RK4 步进到精确 t_out（消除 ~Ω·dt/2 的
    记录时刻映射误差——那是之前收敛只有 1 阶的原因）。
    pops = |ψ|² = |ψ̃|² 不变。完全保留反旋转项 = 第一性原理。
    """
    N = len(psi0)
    M = int(np.ceil(t_out[-1] / dt))
    mats = [A for A, _ in terms]
    funcs = [f for _, f in terms]
    t_out = np.asarray(t_out)
    out_step = np.minimum(np.ceil(t_out / dt).astype(int), M)
    pops = np.zeros((len(t_out), N))
    pops[0] = np.abs(psi0) ** 2
    psi = np.array(psi0, complex)
    psi_m = np.array(psi0, complex)              # 步首状态（t = m·dt）
    rec = 1
    c = -1j * _CONV
    Ec = _CONV * np.asarray(E, float)          # rad/ns

    def Ht(psi_, t, u):
        # Ṽ(t)ψ = Σ_k s_k(t)·D(u)·A_k·D(u*)·ψ，u_i = e^{i E_i t}
        out = np.zeros(N, complex)
        for k in range(len(mats)):
            sk = float(funcs[k](t))
            if abs(sk) < 1e-300:
                continue
            v = u.conj() * psi_
            out = out + sk * (u * (mats[k] @ v))
        return out

    def rk4_step(ps, t0, h):
        """相互作用绘景 RK4 一步：ps(t0) → ps(t0+h)。"""
        ua = np.exp(1j * Ec * t0)
        ub = np.exp(1j * Ec * (t0 + h / 2))
        uc = np.exp(1j * Ec * (t0 + h))
        k1 = c * Ht(ps, t0, ua)
        k2 = c * Ht(ps + h / 2 * k1, t0 + h / 2, ub)
        k3 = c * Ht(ps + h / 2 * k2, t0 + h / 2, ub)
        k4 = c * Ht(ps + h * k3, t0 + h, uc)
        return ps + h / 6 * (k1 + 2 * k2 + 2 * k3 + k4)

    for m in range(M):
        tm = m * dt
        psi = rk4_step(psi_m, tm, dt)
        while rec < len(t_out) and out_step[rec] == m + 1:
            tau = t_out[rec] - tm                # ∈ (0, dt]
            psi_r = rk4_step(psi_m, tm, tau)     # 精确到输出时刻
            pops[rec] = np.abs(psi_r) ** 2
            rec += 1
        psi_m = psi
        if rec >= len(t_out):
            break
    return pops


def floquet_evolve_w(H0, A, omega_mhz, psi0, t_out, n_sub=1000):
    """单驱动（一个 A 矩阵）常数包络周期驱动，T=2π/ω。

    精确积分一个周期得 U(T)（中点法 n_sub 步）→ 本征分解
    U(T)=W·diag(e^{-iεT})·W† → 任意 t=nT+τ：ψ(t)=U(τ)·W·diag(e^{-iεnT})·W†·ψ0。
    对周期驱动 = TDSE 精确解（积分误差可收敛到机器精度；非光子块截断）。
    """
    w = _CONV * omega_mhz                      # rad/ns
    T = 2.0 * np.pi / w
    dt = T / n_sub
    N = len(psi0)
    Us = np.empty((n_sub + 1, N, N), complex)
    Us[0] = np.eye(N)
    U = np.eye(N, dtype=complex)
    tmid = (np.arange(n_sub) + 0.5) * dt
    cw = np.cos(w * tmid)                       # 预计算
    for j in range(n_sub):
        H = H0 + cw[j] * A
        e, Ue = np.linalg.eigh(H)
        D = np.exp(-1j * _CONV * dt * e)
        U = Ue @ (D[:, None] * (Ue.T @ U))
        Us[j + 1] = U
    UT = Us[-1]
    ev, evec = np.linalg.eig(UT)
    eps = -np.angle(ev) / T                     # 准能量 (rad/ns)
    a = evec.conj().T @ psi0
    t_out = np.asarray(t_out)
    pops = np.zeros((len(t_out), N))
    pops[0] = np.abs(psi0) ** 2
    for m in range(1, len(t_out)):
        t = t_out[m]
        n = int(np.floor(t / T))
        tau = t - n * T
        j = min(int(round(tau / dt)), n_sub)
        ph = np.exp(-1j * eps * n * T)
        psi = Us[j] @ (evec @ (a * ph))
        pops[m] = np.abs(psi) ** 2
    return pops


# ---------------- 场景搭建 ----------------
def build_s1(cache, B, envelope=None):
    """纯微波多能级：real_field 自动展开近共振 S↔P 多对（真实 Rabi）。"""
    states = make_states(cache, B)
    i0 = next(i for i, s in enumerate(states)
              if (s['n'], s['l'], s['j']) == (60, 0, 0.5)
              and abs(s['m'] - 0.5) < 1e-9)
    freq_mhz = cache.E_mhz(60, 1, 0.5) - cache.E_mhz(60, 0, 0.5)   # 60S↔60P1/2
    dmax = max(abs(core.dipole_amplitude(cache, si, sj, 0))
               for si in states if si['n'] == 60 and si['l'] == 0
               for sj in states if sj['l'] == 1)
    I = core.intensity_mwcm2_from_rabi(dmax, 100.0)
    fld, ac = evolve.real_field(cache, states, freq_mhz, I, pol='π',
                                waveform=envelope)
    assert fld is not None
    geoms = [evolve._field_geometry(states, fld)]
    E = np.array([s['E_mhz'] for s in states]) + np.asarray(ac)
    om_max = float(fld['A'].max())
    w_mhz = float(fld['omega_L_mhz'])
    nz = int(np.count_nonzero(np.abs(fld['A']) > 1e-12))
    return states, i0, E, fld, geoms, om_max, w_mhz, freq_mhz, I, nz


def build_s3(cache, B):
    """阶梯：光频 6P↔60S（RWA）+ 微波 60S↔60P（非 RWA）。"""
    states = make_states_ladder(cache, B)
    i0 = next(i for i, s in enumerate(states)
              if (s['n'], s['l'], s['j']) == (6, 1, 1.5)
              and abs(s['m'] + 0.5) < 1e-9)
    d_opt = max(abs(core.dipole_amplitude(cache, si, sj, +1))
                for si in states if si['n'] == 6
                for sj in states if sj['n'] == 60 and sj['l'] == 0)
    I_opt = core.intensity_mwcm2_from_rabi(d_opt, 20.0)
    f_opt = cache.E_mhz(60, 0, 0.5) - cache.E_mhz(6, 1, 1.5)
    fld_o, ac_o = evolve.real_field(cache, states, f_opt, I_opt, pol='σ+')
    d_mw = max(abs(core.dipole_amplitude(cache, si, sj, 0))
               for si in states if si['n'] == 60 and si['l'] == 0
               for sj in states if sj['n'] == 60 and sj['l'] == 1)
    I_mw = core.intensity_mwcm2_from_rabi(d_mw, 100.0)
    f_mw = cache.E_mhz(60, 1, 0.5) - cache.E_mhz(60, 0, 0.5)
    fld_m, ac_m = evolve.real_field(cache, states, f_mw, I_mw, pol='π')
    E = (np.array([s['E_mhz'] for s in states])
         + np.asarray(ac_o) + np.asarray(ac_m))
    fields = [f for f in (fld_o, fld_m) if f is not None]
    return states, i0, E, fields


# ---------------- 混合（光频 RWA + 微波非 RWA）固定步长 ----------------
def build_hybrid_terms(states, fields, t, dt):
    """参考频率分配 + RWA 常数项 + 非 RWA cos 项（逻辑移植自 _evolve_full_hybrid）。

    返回 (ref_arr, rwa_terms, non_terms, w_mw_mhz)。
    """
    N = len(states)
    geoms = [evolve._field_geometry(states, f) for f in fields]

    def _key(i):
        s = states[i]
        return (s['n'], s['l'], s['j'])

    def _ratio(g):
        return (g.get('real_omega0_mhz', g['omega0_mhz'])
                / max(g['omega_L_mhz'], 1e-9))
    rwa_g = [g for g in geoms if _ratio(g) < 1e-4]
    non_g = [g for g in geoms if _ratio(g) >= 1e-4]
    Emin = min(s['E_mhz'] for s in states)
    start = next(_key(i) for i, s in enumerate(states) if s['E_mhz'] == Emin)
    ref = {start: 0.0}
    edges = [(_key(g['g_idx'][0]), _key(g['e_idx'][0]), g['omega_L_mhz'])
             for g in rwa_g]
    for _ in range(N + 2):
        changed = False
        for kg, ke, w in edges:
            if kg in ref and ke not in ref:
                ref[ke] = ref[kg] + w
                changed = True
        if not changed:
            break
    for g in non_g:
        kg, ke = _key(g['g_idx'][0]), _key(g['e_idx'][0])
        if kg in ref and ke not in ref:
            ref[ke] = ref[kg]
        elif ke in ref and kg not in ref:
            ref[kg] = ref[ke]
    for i in range(N):
        ref.setdefault(_key(i), 0.0)
    ref_arr = np.array([ref[_key(i)] for i in range(N)])
    rwa_terms = [(g['omega0_mhz'] / 2.0) * g['A'] for g in rwa_g]
    M = int(np.ceil(t[-1] / dt))
    tmid = (np.arange(M) + 0.5) * dt
    non_terms = []
    w_mw = 0.0
    for g in non_g:
        w = _CONV * g['omega_L_mhz']
        w_mw = max(w_mw, g['omega_L_mhz'])
        non_terms.append((g['A'], np.cos(w * tmid)))
    return ref_arr, rwa_terms, non_terms, w_mw


def integrate_midpoint_hybrid(E, ref_arr, rwa_terms, non_terms, psi0,
                              t_out, dt):
    """H(t) = diag(E-ref) + Σ_rwa (Ω/2)A + Σ_non Ω·cos(ω t)·A。"""
    N = len(psi0)
    H0 = np.diag(E - ref_arr).astype(float)
    H0 = H0 - H0.min() * np.eye(N)
    for A in rwa_terms:
        H0 = H0 + A
    return integrate_midpoint(H0, non_terms, psi0, t_out, dt)


# ---------------- 工具 ----------------
def maxdiff(a, b):
    return float(np.max(np.abs(a - b)))


def ac_shift_of(E, states):
    return E - np.array([s['E_mhz'] for s in states])


# ---------------- S1 / S2 ----------------
def run_scenario_s1(cache, envelope, name, dt_list):
    (states, i0, E, fld, geoms, om_max, w_mhz, freq_mhz, I, nz) = \
        build_s1(cache, B_FIELD, envelope)
    N = len(states)
    psi0 = np.zeros(N, complex); psi0[i0] = 1.0
    t = np.linspace(0, T_TOTAL, N_OUT)
    kind = '方波' if envelope is None else '高斯'
    print(f'\n===== {name}：纯微波多能级（{kind}，N={N}，驱动非零耦合对数={nz}，'
          f'Ω_max={om_max:.1f} MHz，ω_L={w_mhz/1e3:.4g} GHz，B={B_FIELD} G，'
          f'T={T_TOTAL} ns）=====')
    fields = [fld]
    acs = ac_shift_of(E, states)
    A = fld['A'].astype(float)
    H0 = np.diag(E).astype(float)
    H0 = H0 - H0.min() * np.eye(N)

    # 参考
    w = _CONV * w_mhz
    if envelope is None:
        s_func = lambda tt: np.cos(w * np.asarray(tt, float))
    else:
        s_func = lambda tt: (np.asarray(envelope(np.asarray(tt, float)), float)
                             * np.cos(w * np.asarray(tt, float)))
    if envelope is None:
        t0 = time.perf_counter()
        ref = floquet_evolve_w(H0, A, w_mhz, psi0, t, n_sub=4000)
        print(f'[参考] Floquet n_sub=4000（TDSE 精确解）: '
              f'{time.perf_counter() - t0:7.2f} s')
    else:
        dt_f = 2 * np.pi / w / 40.0                      # T/40（已收敛，见 debug）
        t0 = time.perf_counter()
        ref = integrate_rk4_ip(E, [(A, s_func)], psi0, t, dt_f)
        print(f'[参考] RK4-IP dt=T/40（收敛参考，T/20 vs T/40 差 <1e-6）: '
              f'{time.perf_counter() - t0:7.2f} s')

    # 独立校验：tight-tol solve_ivp（rtol=1e-12）在 50 ns 短窗 vs 参考
    # （验证 Floquet/RK4 参考本身正确，非自我一致）
    if envelope is None:
        from scipy.integrate import solve_ivp as _sivp
        tv = t[t <= 50.0]
        if len(tv) < 10:
            tv = t[:251]
        H0v = np.diag(E[:len(psi0)]).astype(float)
        H0v = H0v - H0v.min() * np.eye(N)

        def rhs(tv_, y):
            Hh = H0v + float(np.cos(w * tv_)) * A
            return -1j * _CONV * (Hh @ y)
        t0 = time.perf_counter()
        sol = _sivp(rhs, (tv[0], tv[-1]), psi0, t_eval=tv,
                    rtol=1e-12, atol=1e-14)
        dt_chk = time.perf_counter() - t0
        ref_v = floquet_evolve_w(H0v, A, w_mhz, psi0, tv, n_sub=4000)
        print(f'[校验] solve_ivp rtol=1e-12（{tv[-1]:.0f} ns 窗）: '
              f'{dt_chk:7.2f} s   vs 参考 max|Δpops| = '
              f'{maxdiff(np.abs(sol.y).T ** 2, ref_v):.3e}')

    # 生产：full（solve_ivp 非 RWA，旧第一性原理参考）——仅在方波（S1）跑，
    # 高斯（S2）由 RK4 收敛表验证（积分器一致性已在 S1 与 Floquet 交叉验证）
    if envelope is None:
        geoms_r = [evolve._field_geometry(states, f) for f in fields]
        psi0v = evolve._as_vector(states, i0)
        t0 = time.perf_counter()
        _, pops_full, _ = evolve._evolve_full(E, geoms_r, psi0v, t, False)
        dt_full = time.perf_counter() - t0
        print(f'[生产] full solve_ivp: {dt_full:7.2f} s   '
              f'vs 参考 max|Δpops| = {maxdiff(pops_full, ref):.3e}')

    # 候选：RK4-IP（相互作用绘景，第一性原理，消除 H0 相位累积）
    for frac in dt_list:
        dt = 2 * np.pi / w / frac                       # T/frac
        t0 = time.perf_counter()
        pops = integrate_rk4_ip(E, [(A, s_func)], psi0, t, dt)
        dt_ = time.perf_counter() - t0
        print(f'[候选] RK4-IP T/{frac:g} : {dt_:7.2f} s   '
              f'vs 参考 max|Δpops| = {maxdiff(pops, ref):.3e}')

    # 候选：Floquet（仅常数包络）
    if envelope is None:
        t0 = time.perf_counter()
        pops = floquet_evolve_w(H0, A, w_mhz, psi0, t, n_sub=1000)
        dt_ = time.perf_counter() - t0
        print(f'[候选] Floquet n_sub=1000: {dt_:7.2f} s   '
              f'vs 参考 max|Δpops| = {maxdiff(pops, ref):.3e}')
    print('      （参考已含反旋转项 = 第一性原理；pwc 是 RWA → 其误差即 RWA 代价）')


# ---------------- S3 ----------------
def run_scenario_s3(cache):
    states, i0, E, fields = build_s3(cache, B_FIELD)
    N = len(states)
    psi0 = np.zeros(N, complex); psi0[i0] = 1.0
    t = np.linspace(0, T_TOTAL, N_OUT)
    print(f'\n===== S3 阶梯：光频 6P↔60S（RWA）+ 微波 60S↔60P（非 RWA）'
          f'（N={N}，B={B_FIELD} G，T={T_TOTAL} ns）=====')

    # 参考：_evolve_full_hybrid（solve_ivp 混合，光频 RWA + 微波非 RWA）
    psi0v = evolve._as_vector(states, i0)
    t0 = time.perf_counter()
    _, pops_ref, _ = evolve._evolve_full_hybrid(E, states, fields, psi0v, t,
                                                False)
    dt_ff = time.perf_counter() - t0
    print(f'[参考] _evolve_full_hybrid（solve_ivp 混合）: {dt_ff:7.2f} s')

    # 内置 ip4（相互作用绘景 RK4，第一性原理）
    ac3 = E - np.array([s['E_mhz'] for s in states])
    t0 = time.perf_counter()
    r = evolve.evolve(states, fields, i0, t, method='ip4', ac_shift_mhz=ac3)
    dt_ip = time.perf_counter() - t0
    print(f'[内置] ip4（默认 frac=40）: {dt_ip:7.2f} s   '
          f'vs 参考 max|Δpops| = {maxdiff(r["pops"], pops_ref):.3e}')

    # 内置 floquet/pwc：2025-08 起已从 evolve.py 封存（动态分析只用 ip4）——
    # 显式调用应报错并指向封存模块；需要时用封存模块 evolve_floquet/evolve_pwc
    from src._archive.evolve_floquet_pwc import evolve_pwc as _evolve_pwc_arch
    try:
        evolve.evolve(states, fields, i0, t, method='floquet')
        print('[内置] floquet：意外成功（应报错指向封存模块）')
    except ValueError as e:
        print(f'[内置] floquet 显式调用正确拒绝（封存）：{str(e)[:60]}…')

    # 封存 pwc（RWA 对比，仅展示误差量级）
    t0 = time.perf_counter()
    r2 = _evolve_pwc_arch(states, fields, i0, t)
    dt_p = time.perf_counter() - t0
    print(f'[封存] pwc RWA      : {dt_p:7.2f} s   '
          f'vs 参考 max|Δpops| = {maxdiff(r2["pops"], pops_ref):.3e}')


def main():
    which = sys.argv[1] if len(sys.argv) > 1 else 'all'
    cache = core.ArcCache(Rubidium87())
    t0_all = time.perf_counter()
    if which in ('all', 'S1'):
        run_scenario_s1(cache, None, 'S1（方波）', dt_list=(20.0, 40.0))
    if which in ('all', 'S2'):
        env = evolve.gaussian(T_TOTAL / 2.0, 100.0)
        run_scenario_s1(cache, env, 'S2（高斯）', dt_list=(20.0, 40.0, 80.0))
    if which in ('all', 'S3'):
        run_scenario_s3(cache)
    print(f'\n脚本总耗时: {time.perf_counter() - t0_all:.1f} s')


if __name__ == '__main__':
    main()
