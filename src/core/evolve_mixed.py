# -*- coding: utf-8 -*-
"""混合引擎：光场 Sambe（光子块免 RWA）+ 微波 DOP853 硬积分（t40）。

Strang 分裂：ψ(t+Δt) ≈ U_mw(Δt/2)·U_opt(Δt)·U_mw(Δt/2)
- U_opt(Δt)：光场 Sambe 扩展空间 eigh 冻结精确指数（K0 含 E−n·ω 对角 + 光场
  耦合，复用 evolve_sambe 的 _SambeSetup/photon_geometry；M 由 _m_analytic 定）
- U_mw(τ)：微波 cos 载波 DOP853 硬积分（E 相互作用绘景 + lab 转换；A_mw 过滤
  至近共振对 |能隙−ω|<5GHz——否则 6P 微小耦合带 C·3e8 相位 THz 钉死）
- 大步 Δt 可行（THz 对角与微波块对角对易 4.8e-7 → 分裂误差只来自 MHz 耦合）

**E 相位划分（t38 60ns 验证版）**：E 由 U_mw 的 lab 转换（两半 Δt/2+Δt/2=Δt）
承载；U_opt 的 K 用 _K0_noE = K0 − diag(E_ext)（防双计）——60ns 窗 ≡ 参考 2.4e-5。
⚠️ 300ns+ 长窗：K_noE 丢光场共振（dressed/AC 错误）→ 长窗不正确——完整 φ-帧
（K 含 E + 光场耦合 detuning 相位）为后续补丁（见 docs/t38混合引擎原型报告.md）。
"""
import numpy as np
from numba import njit

from . import evolve as _evolve
from .evolve_sambe import (_group_by_freq, _SambeSetup, _m_analytic, _embed,
                           _reconstruct)

_CONV = 2 * np.pi * 1e-3
MW_NRES_BW_MHZ = 5e3          # 微波近共振带宽（同 build_two_field）


def _split_fields(fields):
    """场列表 → (光场, 微波)。光场 = kind='exact' 载频 > 1e5。"""
    opt = [f for f in fields if f['omega_L_mhz'] > 1e5]
    mw = [f for f in fields if f['omega_L_mhz'] <= 1e5]
    return opt, mw


class _MixedSetup:
    """混合引擎预计算：光场光子块结构 + 微波扩展项打包。"""

    def __init__(self, E, grps_opt, grps_mw, N):
        self.N = N
        self.E = np.asarray(E, float)
        self.E_ext = np.tile(self.E, 1)      # 占位（实际由 setup.P 定）
        self.grps_mw = grps_mw
        # 光场光子块结构（M 逐载频自动）
        Ms = _m_analytic(grps_opt, 1e-6, self.E)
        self.setup = _SambeSetup(self.E, grps_opt, Ms)
        self.P = self.setup.P
        self.E_ext = np.tile(self.E, self.P)
        # K 无 E 版本（t38 60ns 验证分区；300ns+ 需 φ-帧补丁）
        self._K0_noE = self.setup.K0 - np.diag(self.E_ext)
        # 微波扩展项（I⊗A，近共振过滤）——逐波保留 (A_ext, env, ω)
        self.mw_terms = []
        self._pairs = []
        for g in grps_mw:
            w = float(g['omega_mhz'])
            for A, env in zip(g['As'], g['envs']):
                A_ext = np.kron(np.eye(self.P), np.asarray(A, float))
                mask = np.zeros_like(A_ext, bool)
                for a, b in np.argwhere(np.abs(A_ext) > 1e-9):
                    if abs(abs(self.E[a % N] - self.E[b % N]) - w) \
                            < MW_NRES_BW_MHZ:
                        mask[a, b] = True
                A_f = np.where(mask, A_ext, 0.0)
                iis, jjs = np.nonzero(np.abs(A_f) > 1e-9)
                self.mw_terms.append((w, env, A_f,
                                      (iis.astype(np.int64),
                                       jjs.astype(np.int64),
                                       A_f[iis, jjs].astype(np.complex128))))
        self.dim = self.setup.dim


def evolve_mixed(states, fields, psi0, t_grid, dt_ns=2.0, tol=1e-8,
                 progress_arr=None, log=None, record_amp=False):
    """混合引擎主入口（光场 Sambe + 微波 DOP853 硬积分）。

    fields : kind='exact' 场列表（build_fields_exact 输出）
    dt_ns  : Strang 大步长（ns）
    """
    from . import evolve as _ev
    log = log or (lambda *a: None)
    t = np.asarray(t_grid, float)
    if progress_arr is None:
        progress_arr = np.zeros(2)
    N = len(states)
    E = np.array([s['E_mhz'] for s in states], float)
    E = E - float(E.min())
    psi0v = _evolve._as_vector(states, psi0)
    f_opt, f_mw = _split_fields(fields)
    if not f_opt:
        raise ValueError('evolve_mixed 需要光场（载频 > 1e5 MHz）')
    grps_opt = _group_by_freq(f_opt)
    grps_mw = _group_by_freq(f_mw) if f_mw else []
    setup = _MixedSetup(E, grps_opt, grps_mw, N)
    log(f'[mixed] 光场光子块 P={setup.P} dim={setup.dim}；'
        f'微波载频 {[g["omega_mhz"] for g in grps_mw]}；Δt={dt_ns}ns')

    X = _embed(psi0v, setup.setup)
    pops = np.zeros((len(t), N))
    pops[0] = _pops_of_ext(X, setup)
    T = float(t[-1])
    C = _CONV

    # ---- 光场冻结 U_opt（每大步一次 eigh）----
    def U_opt(Xv, ta, tb):
        K = setup._K0_noE.copy()
        for gi, g in enumerate(grps_opt):
            for wf, Kcw in zip(g['envs'], setup.setup.Kc_dense[gi] or []):
                c = float(np.asarray(wf(np.atleast_1d(0.5 * (ta + tb)))).flat[0])
                if Kcw is not None:
                    K = K + c * Kcw
        evals, evecs = np.linalg.eigh(K)
        a0 = evecs.conj().T @ Xv
        return evecs @ (a0 * np.exp(-1j * C * evals * (tb - ta)))

    # ---- 微波 DOP853（E-IP；逐波 (ω, env, A, pairs)）----
    def U_mw(Xv, ta, tb, max_step=0.05):
        dt = tb - ta
        E_ext = setup.E_ext

        def rhs_phi2(tv, phi):
            s = tv - ta
            tot = np.zeros_like(phi)
            for w_mw, env, A_f, (iis, jjs, vals) in setup.mw_terms:
                envv = float(np.asarray(env(np.atleast_1d(tv))).flat[0])
                if envv == 0.0:
                    continue
                cw = envv * np.cos(C * w_mw * tv)
                for a, b, v in zip(iis, jjs, vals):
                    ph = np.exp(1j * C * (E_ext[a] - E_ext[b]) * s)
                    tot[a] += cw * v * ph * phi[b]
            return -1j * C * tot

        from scipy.integrate import solve_ivp
        sol = solve_ivp(rhs_phi2, (ta, tb), Xv, method='DOP853',
                        rtol=tol, atol=1e-10, max_step=max_step)
        if not sol.success:
            raise RuntimeError(f'U_mw 积分失败: {sol.message}')
        return np.exp(-1j * C * E_ext * dt) * sol.y[:, -1]

    tprev = 0.0
    nsteps = 0
    for i in range(1, len(t)):
        progress_arr[0] = min(tprev / T, 0.999)
        tb = float(t[i])
        while tprev < tb - 1e-9:
            hh = min(dt_ns, tb - tprev)
            X = U_mw(X, tprev, tprev + hh / 2)
            X = U_opt(X, tprev, tprev + hh)
            X = U_mw(X, tprev + hh / 2, tprev + hh)
            tprev += hh
            nsteps += 1
            if progress_arr[1] != 0.0:
                raise RuntimeError('用户取消')
        ps = _reconstruct(setup.setup, X, tb)
        pops[i] = _pops_of_ext(X, setup)
        tprev = tb
    progress_arr[0] = 1.0
    norm_dev = float(np.max(np.abs(pops.sum(axis=1) - 1.0)))
    out = dict(t=t, pops=pops, states=states,
               om_mhz=[float(np.abs(A).max())
                       for A in grps_mw[0]['As']] if grps_mw else [],
               method='mixed', backend='mixed-sambe-dop853',
               loss=np.zeros(len(t)),
               error_report=dict(n_segments=nsteps, max_richardson_diff=0.0,
                                 dt_min_hit=False, max_norm_dev=norm_dev,
                                 P=setup.P))
    if record_amp:
        out['psi_amp'] = _reconstruct(setup.setup, X, T)
    return out


def _pops_of_ext(X, setup):
    Xm = np.asarray(X, complex).reshape(setup.P, setup.N)
    return np.abs(Xm.conj().T @ Xm).diagonal().real
