# -*- coding: utf-8 -*-
"""混合引擎：光场 Sambe（光子块免 RWA）+ 微波 DOP853 硬积分（t41 φ-帧一致版）。

Strang 分裂：ψ(t+Δt) ≈ U_mw(Δt/2)·U_opt(Δt)·U_mw(Δt/2)，**全程 φ-帧**：
φ = e^{iC·D_full·t}·X（D_full = 光子块全对角 = diag(K0) = E−n·ω）——
E 单次承载于全局相位（t40 的 K_noE + U_mw lab 转换分区错误已弃）。

- U_opt 的 φ-帧版：K_φ = Σ Kc∘e^{iC(D_a−D_b)·t}（冻结中点）——K0−D_full=0；
  **反旋对过滤** |D_a−D_b| < BW_OPT=2e4 MHz（20 GHz）——±2ω THz 对相位无法
  冻结且振幅 (Ω/2ω)²~1e-14 可弃；**只取上三角 a<b 再写 Hermitian**（防双计）。
- U_mw 的 φ-帧版：cos·A_ext∘e^{iC(D_a−D_b)·t}（**绝对时间 t**——φ 定义是
  绝对的；共振对 cos(ωt)e^{−iωt}=½(1+e^{−2iωt}) 慢项=½ 完美驱动）DOP853。
- 输出时回 X 帧：X = e^{−iC·D_full·t}·φ。

参考：船长最小原型 tempsimu/_t40_phi_c3.py（3 态 300ns DT=2ns 4.83e-4 ✅）。
"""
import numpy as np
from numba import njit

from . import evolve as _evolve
from .evolve_sambe import (_group_by_freq, _SambeSetup, _m_analytic, _embed)

_CONV = 2 * np.pi * 1e-3
MW_NRES_BW_MHZ = 5e3          # 微波近共振带宽
BW_OPT = 2e4                  # 光场 φ-帧反旋过滤带宽（20 GHz）


@njit(cache=True)
def _rhs_mw_phi(t, phi, ta, iis, jjs, vals, dDs, wcos, w_npairs,
                env_tab, env_dt, C):
    """φ-帧微波 RHS：−iC·Σ_w env_w(t)·cos(ω_w·t)·(A_w∘e^{iC·ΔD·t})·φ。

    t 为驱动内部时钟（0 起）；绝对时间 = t + ta（φ=e^{iCDt}X 是绝对的）。
    """
    tv = t + ta
    tot = np.zeros_like(phi)
    ip = 0
    nw = len(wcos)
    for w in range(nw):
        np_w = w_npairs[w]
        if np_w == 0:
            continue
        ei = tv / env_dt
        i0 = int(ei)
        if i0 >= env_tab.shape[1] - 1:
            i0 = env_tab.shape[1] - 2
        f = ei - i0
        envv = env_tab[w, i0] * (1.0 - f) + env_tab[w, i0 + 1] * f
        if envv == 0.0:
            ip += np_w
            continue
        cw = envv * np.cos(C * wcos[w] * tv)
        for k in range(np_w):
            a = iis[ip + k]
            b = jjs[ip + k]
            tot[a] += cw * vals[ip + k] * np.exp(
                1j * C * dDs[ip + k] * tv) * phi[b]
        ip += np_w
    return -1j * C * tot


class _MixedSetup:
    """混合引擎预计算：光场光子块 + φ-帧耦合项打包。"""

    def __init__(self, E, grps_opt, grps_mw, N):
        self.N = N
        self.E = np.asarray(E, float)
        self.grps_opt = grps_opt
        self.grps_mw = grps_mw
        Ms = _m_analytic(grps_opt, 1e-6, self.E)
        self.setup = _SambeSetup(self.E, grps_opt, Ms)
        self.P = self.setup.P
        self.dim = self.setup.dim
        self.D_full = np.real(np.diag(self.setup.K0))   # 光子块全对角

        # ---- 光场 φ-帧耦合项：过滤反旋 + 上三角 + Hermitian ----
        self.opt_terms = []      # (iis, jjs, vals_amp, env)
        for gi, g in enumerate(grps_opt):
            for wf, Kcw in zip(g['envs'], self.setup.Kc_dense[gi] or []):
                if Kcw is None:
                    continue
                iis, jjs = np.nonzero(np.abs(Kcw) > 1e-12)
                dD = np.abs(self.D_full[iis] - self.D_full[jjs])
                keep = dD < BW_OPT
                ii, jj = iis[keep], jjs[keep]
                upper = ii < jj          # 只取上三角（去 (b,a) 双计）
                if upper.any():
                    self.opt_terms.append(
                        (ii[upper].astype(np.int64),
                         jj[upper].astype(np.int64),
                         Kcw[ii[upper], jj[upper]].astype(np.complex128), wf))

        # ---- 微波 φ-帧项：同扇区对 (a, b)，D 差 = E_i−E_j ----
        # 打包 numba 数组：iis/jjs/vals/dDs 全波拼接 + w_npairs 分段 + env 表
        self.mw_terms = []       # 兼容元数据 (ω, env, pairs)
        mw_iis, mw_jjs, mw_vals, mw_dDs = [], [], [], []
        self._mw_wcos = []
        self._mw_w_npairs = []
        self._env_list = []
        for g in grps_mw:
            w = float(g['omega_mhz'])
            for A, env in zip(g['As'], g['envs']):
                A = np.asarray(A, float)
                pairs = []
                np_w = 0
                for sa in range(self.P):
                    for i in range(N):
                        for j in range(N):
                            if abs(A[i, j]) < 1e-12:
                                continue
                            a = sa * N + i
                            b = sa * N + j
                            if b == a:
                                continue
                            # φ-帧近共振过滤：|ΔD|−ω 小（同 t38 E-IP 教训——
                            # 6P 微小耦合带 THz 相位会钉死 DOP853）
                            dD_ab = float(self.D_full[a] - self.D_full[b])
                            if abs(abs(dD_ab) - w) > MW_NRES_BW_MHZ:
                                continue
                            pairs.append((a, b))
                            mw_iis.append(a)
                            mw_jjs.append(b)
                            mw_vals.append(complex(A[i, j]))
                            mw_dDs.append(dD_ab)
                            np_w += 1
                self.mw_terms.append((w, env, pairs))
                self._mw_wcos.append(w)
                self._mw_w_npairs.append(np_w)
                self._env_list.append(env)
        self.mw_iis = np.asarray(mw_iis, np.int64)
        self.mw_jjs = np.asarray(mw_jjs, np.int64)
        self.mw_vals = np.asarray(mw_vals, np.complex128)
        self.mw_dDs = np.asarray(mw_dDs, np.float64)
        self._mw_wcos = np.asarray(self._mw_wcos, np.float64)
        self._mw_w_npairs = np.asarray(self._mw_w_npairs, np.int64)

    def make_env_tabs(self, T, n_tab=31001):
        """预采样包络表（φ-帧微波 numba 查表；网格 ~0.02ns 线性插值）。"""
        self.env_dt = T / (n_tab - 1)
        tab = np.zeros((len(self._env_list), n_tab))
        ts = np.linspace(0.0, T, n_tab)
        for wi, env in enumerate(self._env_list):
            tab[wi] = np.asarray(env(ts), float)
        self.env_tab = tab
        return tab


def evolve_mixed(states, fields, psi0, t_grid, dt_ns=2.0, tol=1e-8,
                 progress_arr=None, log=None, record_amp=False):
    """混合引擎主入口：光场 Sambe（φ-帧）+ 微波 DOP853 硬积分，Strang 分裂。

    fields : kind='exact' 场列表（build_fields_exact 输出；光场 + 微波混合）
    dt_ns  : Strang 大步长（ns）
    """
    log = log or (lambda *a: None)
    t = np.asarray(t_grid, float)
    if progress_arr is None:
        progress_arr = np.zeros(2)
    N = len(states)
    E = np.array([s['E_mhz'] for s in states], float)
    E = E - float(E.min())
    psi0v = _evolve._as_vector(states, psi0)
    C = _CONV

    f_opt = [f for f in fields if f['omega_L_mhz'] > 1e5]
    f_mw = [f for f in fields if f['omega_L_mhz'] <= 1e5]
    grps_opt = _group_by_freq(f_opt)
    grps_mw = _group_by_freq(f_mw) if f_mw else []
    if not grps_opt:
        raise ValueError('evolve_mixed 需要光场（载频 > 1e5 MHz）')
    setup = _MixedSetup(E, grps_opt, grps_mw, N)
    P = setup.P
    T = float(t[-1])
    setup.make_env_tabs(T)
    log(f'[mixed] 光场 P={P} dim={setup.dim}；微波组 {len(grps_mw)}；'
        f'opt_terms={len(setup.opt_terms)} '
        f'mw_pairs={len(setup.mw_iis)}')

    X = _embed(psi0v, setup.setup)

    # ---- U_opt（φ-帧）：K = Σ c·Kc∘e^{iC·ΔD·mid}（上三角 + Hermitian）----
    def U_opt_phi(Xv, ta, tb):
        mid = 0.5 * (ta + tb)
        K = np.zeros((setup.dim, setup.dim), complex)
        any_k = False
        for iis, jjs, vals, wf in setup.opt_terms:
            c = float(np.asarray(wf(np.atleast_1d(mid))).flat[0])
            if c == 0.0:
                continue
            ph = np.exp(1j * C * (setup.D_full[iis] - setup.D_full[jjs]) * mid)
            K[iis, jjs] += c * vals * ph
            K[jjs, iis] += c * np.conj(vals * ph)
            any_k = True
        if not any_k:
            return Xv
        evals, evecs = np.linalg.eigh(K)
        return evecs @ ((evecs.conj().T @ Xv)
                        * np.exp(-1j * C * evals * (tb - ta)))

    # ---- U_mw（φ-帧）：cos·A∘e^{iC·ΔD·t}（绝对 t）numba DOP853 ----
    from .dop853 import _dop853_driver

    def U_mw_phi(Xv, ta, tb, max_step=0.05):
        dt = tb - ta
        if not setup.mw_iis.size:
            return Xv
        prog = np.zeros(2)
        stat = np.zeros(4)
        t_h = np.array([0.0, dt])
        y_out = _dop853_driver(_rhs_mw_phi, Xv, t_h, dt, float(tol), 1e-10,
                               max_step, prog, stat, ta, setup.mw_iis,
                               setup.mw_jjs, setup.mw_vals, setup.mw_dDs,
                               setup._mw_wcos, setup._mw_w_npairs,
                               setup.env_tab, setup.env_dt, float(C))
        return y_out[-1]

    # ---- Strang 全程 φ-帧 ----
    pops = np.zeros((len(t), N))
    pops[0] = _pops_of_ext(X, setup)
    nsteps = 0
    tprev = 0.0
    for i in range(1, len(t)):
        progress_arr[0] = min(tprev / T, 0.999)
        tb = float(t[i])
        while tprev < tb - 1e-9:
            hh = min(dt_ns, tb - tprev)
            ta = tprev
            phi_a = np.exp(1j * C * setup.D_full * ta) * X
            phi_m = U_mw_phi(phi_a, ta, ta + hh / 2)
            phi_m = U_opt_phi(phi_m, ta, ta + hh)
            phi_b = U_mw_phi(phi_m, ta + hh / 2, ta + hh)
            X = np.exp(-1j * C * setup.D_full * (ta + hh)) * phi_b
            tprev = ta + hh
            nsteps += 1
            if progress_arr[1] != 0.0:
                raise RuntimeError('用户取消')
        pops[i] = _pops_of_ext(X, setup)
        tprev = tb
    progress_arr[0] = 1.0
    norm_dev = float(np.max(np.abs(pops.sum(axis=1) - 1.0)))
    out = dict(t=t, pops=pops, states=states,
               om_mhz=[float(np.abs(np.asarray(g['As'][0])).max())
                       for g in grps_mw] if grps_mw else [],
               method='mixed', backend='mixed-phi-sambe-dop853',
               loss=np.zeros(len(t)),
               error_report=dict(n_segments=nsteps, max_richardson_diff=0.0,
                                 dt_min_hit=False, max_norm_dev=norm_dev,
                                 P=P, dim=setup.dim))
    if record_amp:
        out['psi_amp'] = X
    return out


def _pops_of_ext(X, setup):
    Xm = np.asarray(X, complex).reshape(setup.P, setup.N)
    return np.abs(Xm.conj().T @ Xm).diagonal().real
