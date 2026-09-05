# -*- coding: utf-8 -*-
"""参考积分器（t74 自 evolve_sambe 归位）：H0 相互作用绘景 + 自适应 ODE。

角色：① sambe 引擎含耗散（decays 非空）时的执行路径；② 全库的
第一性原理权威参照（rtol/atol 显式、可选双容差 verify）。
- 幺正（decays 空）：DOP853；
- 耗散：同绘景密度矩阵 + BDF（Lindblad 跳变算符在相互作用绘景下
  时不变，与实验室系严格等价）；显式落点/纯退相经 _decay_ops 支持。
"""
import numpy as np
from scipy.integrate import solve_ivp

from .sambe import _CONV, env_at
from .evolve_util import _as_vector, _decay_ops
from .field_build import _validate_fields
from . import evolve as _evolve  # t74 过渡：_as_vector/_decay_ops 经 shim（P0 后切 evolve_util）


def evolve_reference(states, fields, psi0, t_grid, decays=None,
                     rtol=1e-9, atol=1e-11, verify=False,
                     progress_arr=None, record_amp=False, log=None):
    """实验室系第一性原理参考积分器（零模型近似 + 显式容差）。

    - 幺正（decays 空）：H0 相互作用绘景 + DOP853（自适应、嵌入式误差控制）；
    - 耗散（decays 非空）：同绘景密度矩阵 + Radau（刚性可用）；
      跳变算符在相互作用绘景下时不变（相位抵消），与实验室系严格等价。
    verify=True：容差收紧 10× 重跑，结果附 verify_diff（布居最大差）。
    返回 dict(t, pops, states, om_mhz, method='exact', backend='reference-*',
              loss, [psi_amp | rho, verify_diff])。
    """
    from scipy.integrate import solve_ivp
    from . import evolve as _evolve
    log = log or (lambda *a: None)
    if progress_arr is None:
        progress_arr = np.zeros(2)
    t = np.asarray(t_grid, float)
    T = float(t[-1])
    _validate_fields(fields)
    psi0v = _evolve._as_vector(states, psi0)
    N = len(states)
    E = np.array([s['E_mhz'] for s in states], float)
    E = E - float(E.min())
    terms = [(np.asarray(f['A'], float), float(f['omega_L_mhz']),
              f['waveform']) for f in fields]

    def coupling_at(tv):
        V = np.zeros((N, N))
        for A, wL, wf in terms:
            env = env_at(wf, tv)
            if env != 0.0:
                V = V + (env * np.cos(_CONV * wL * tv)) * A
        return V

    def _run_unitary(rtl, atl):
        def _rhs_sambe(tv, y):
            u = np.exp(1j * _CONV * E * tv)
            return -1j * _CONV * (u * (coupling_at(tv) @ (np.conj(u) * y)))
        sol = solve_ivp(_rhs_sambe, (0.0, T), psi0v, method='DOP853',
                        t_eval=t, rtol=rtl, atol=atl)
        if not sol.success:
            raise RuntimeError(f'reference 积分失败: {sol.message}')
        return sol.y                            # 相互作用绘景振幅

    decays = list(decays or [])
    if not decays:
        from .coupling import stage_progress
        stage_progress(progress_arr, 1, 0.0)  # 阶段 1：参考积分（无内部钩子）
        yI = _run_unitary(rtol, atol)
        pops = np.abs(yI).T ** 2
        out = dict(t=t, pops=pops, states=states,
                   om_mhz=[float(np.abs(A).max()) for A, _, _ in terms],
                   method='exact', backend='reference-dop853',
                   loss=np.zeros(len(t)))
        if record_amp:
            uT = np.exp(1j * _CONV * E * T)     # 实验室系末态振幅
            out['psi_amp'] = np.conj(uT) * yI[:, -1]
        if verify:
            y2 = _run_unitary(rtol / 10.0, atol / 10.0)
            out['verify_diff'] = float(np.max(
                np.abs(np.abs(y2).T ** 2 - pops)))
            log(f'[reference] 双容差验证：Δpops={out["verify_diff"]:.2e}'
                f'（rtol={rtol:.1e}）')
        stage_progress(progress_arr, 2, 1.0)
        return out

    # ---- Lindblad（同绘景密度矩阵；L 时不变） ----
    Ls, N_eff = _evolve._decay_ops(decays, states)
    rho0 = np.zeros((N_eff, N_eff), complex)
    rho0[:N, :N] = np.outer(psi0v, psi0v.conj())

    def rhs_me(tv, rv):
        rho = rv.reshape((N_eff, N_eff), order='F')
        u = np.exp(1j * _CONV * E * tv)
        V = coupling_at(tv)
        H = np.zeros((N_eff, N_eff), complex)
        H[:N, :N] = (u[:, None] * V) * np.conj(u)[None, :]
        d = -1j * _CONV * (H @ rho - rho @ H)
        for L in Ls:
            LdL = L.conj().T @ L
            d = d + (L @ rho @ L.conj().T
                     - 0.5 * (LdL @ rho + rho @ LdL))
        return d.ravel(order='F')

    def _run_mastereq(rtl, atl):
        # BDF：支持复数 y（Radau 不支持；耗散率可致刚性，隐式多步稳定）
        sol = solve_ivp(rhs_me, (0.0, T), rho0.ravel(order='F'),
                        method='BDF', t_eval=t, rtol=rtl, atol=atl)
        if not sol.success:
            raise RuntimeError(f'reference(Lindblad) 积分失败: {sol.message}')
        return sol.y

    from .coupling import stage_progress
    stage_progress(progress_arr, 1, 0.0)      # 阶段 1：主方程积分
    rv = _run_mastereq(rtol, atol)
    pops = np.zeros((len(t), N))
    loss = np.zeros(len(t))
    for k in range(len(t)):
        rho_k = rv[:, k].reshape((N_eff, N_eff), order='F')
        dg = np.real(np.diag(rho_k))
        pops[k] = dg[:N]
        if N_eff > N:
            loss[k] = dg[N]
    out = dict(t=t, pops=np.clip(pops, 0.0, None), states=states,
               om_mhz=[float(np.abs(A).max()) for A, _, _ in terms],
               method='exact', backend='reference-bdf', loss=loss)
    if record_amp:
        out['rho'] = rv[:, -1].reshape((N_eff, N_eff), order='F')
    if verify:
        rv2 = _run_mastereq(rtol / 10.0, atol / 10.0)
        pops2 = np.zeros_like(pops)
        for k in range(len(t)):
            pops2[k] = np.real(np.diag(
                rv2[:, k].reshape((N_eff, N_eff), order='F')))[:N]
        out['verify_diff'] = float(np.max(np.abs(pops2 - pops)))
        log(f'[reference] 双容差验证：Δpops={out["verify_diff"]:.2e}'
            f'（rtol={rtol:.1e}）')
    stage_progress(progress_arr, 2, 1.0)
    return out
