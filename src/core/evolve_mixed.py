# -*- coding: utf-8 -*-
"""混合引擎：光场 Sambe 结构（光子块免 RWA）+ 微波 cos 硬积分——φ-帧单 DOP853（t42）。

架构简化（船长 t41 关键实验）：φ-帧变换 φ = e^{iC·D_full·t}·X
（D_full = 光子块全对角 = diag(K0) = E−n·ω）已把 THz 对角吸进全局相位——
剩下的 φ-帧方程所有项可控（光场共振对 ΔD≈失谐 MHz 慢；微波同扇区 ΔD=能级差
cos 干涉），**单 DOP853 直接积分**即可；Strang 分裂 + eigh 冻结是多余且引入
结构误差（冻结中点相位对 ΔD≠0 不准 + Strang 对易误差——t41 22 态 1.5e-2
平台的根源），已删除。

dφ/dt = −iC·[Σ_光场 env(t)·Kc∘e^{iC·ΔD·t} + Σ_微波 env(t)·cos(ωt)·A∘e^{iC·ΔD·t}]·φ

- 光场项：共振对 |ΔD|<BW_OPT=20GHz 过滤（反旋 ±2ω 振幅 ~1e-14 可弃）
- 微波项：近共振过滤 |ΔD|−ω<5GHz（6P 微小耦合 THz 相位钉死规避）
- 线性 RHS 双向（Hermitian 天然，无 K 构造双计问题）
- 全程 φ-帧连续积分（driver 一次调用 t_eval=网格）；输出时 X = e^{−iC·D·t}·φ 投影

参考：船长 tempsimu/_t41_single_dop853.py（3 态 300ns max|Δpops|=4.22e-8 ✅）。
"""
import numpy as np
from numba import njit

from . import evolve as _evolve
from .evolve_sambe import (_group_by_freq, _SambeSetup, _m_analytic, _embed)

_CONV = 2 * np.pi * 1e-3
MW_NRES_BW_MHZ = 5e3          # 微波近共振带宽
BW_OPT = 2e4                  # 光场反旋过滤带宽（20 GHz）
AC_MIN_V = 1.0                # 远失谐对 AC 注入的 |Kc| 阈值（MHz）


@njit(cache=True)
def _rhs_phi(t, phi, iis, jjs, vals, dDs, wids, is_mw, wcos,
             n_pairs_w, env_tab, env_dt, ac_diag, C):
    """φ-帧统一 RHS：−iC·[光场 env·Kc + 微波 env·cos·A]∘e^{iC·ΔD·t}·φ
    − iC·ac_diag∘φ（远失谐对 AC 位移解析注入）。

    t 为绝对时间（φ=e^{iCDt}X 定义是绝对的）。wids[k] → 波索引；
    is_mw[w]=1 时该项带 cos(C·w·t)。"""
    tv = t
    tot = np.zeros_like(phi)
    # AC 位移对角项（常数——远失谐对被 BW_OPT 过滤丢失的 dressed 二阶补齐）
    for a in range(len(phi)):
        if ac_diag[a] != 0.0:
            tot[a] += ac_diag[a] * phi[a]
    ip = 0
    nw = len(wcos)
    for w in range(nw):
        np_w = n_pairs_w[w]
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
        if is_mw[w]:
            cw = envv * np.cos(C * wcos[w] * tv)
        else:
            cw = envv
        for k in range(np_w):
            a = iis[ip + k]
            b = jjs[ip + k]
            tot[a] += cw * vals[ip + k] * np.exp(
                1j * C * dDs[ip + k] * tv) * phi[b]
        ip += np_w
    return -1j * C * tot


class _MixedSetup:
    """混合引擎预计算：光场光子块 + φ-帧耦合对统一打包。"""

    def __init__(self, E, grps_opt, grps_mw, N):
        self.N = N
        self.E = np.asarray(E, float)
        self.grps_opt = grps_opt
        self.grps_mw = grps_mw
        Ms = _m_analytic(grps_opt, 1e-6, self.E)
        self.setup = _SambeSetup(self.E, grps_opt, Ms)
        self.P = self.setup.P
        self.dim = self.setup.dim
        self.D_full = np.real(np.diag(self.setup.K0))

        # 波清单（光场在前 + 微波）：is_mw 标志 + ω + env
        self._env_list = []
        self._is_mw = []
        self._wcos = []
        for gi, g in enumerate(grps_opt):
            for wf, Kcw in zip(g['envs'], self.setup.Kc_dense[gi] or []):
                if Kcw is not None:
                    self._env_list.append(wf)
                    self._is_mw.append(0)
                    self._wcos.append(0.0)
        for g in grps_mw:
            w = float(g['omega_mhz'])
            for A, env in zip(g['As'], g['envs']):
                self._env_list.append(env)
                self._is_mw.append(1)
                self._wcos.append(w)

        # 耦合对（按波分段）：光场共振对 |ΔD|<BW_OPT（Kc 对称 → 双向入列
        # 线性 Hermitian 天然）；微波近共振 |ΔD|−ω<5GHz（A 对称 → 双向）
        iis, jjs, vals, dDs, wids = [], [], [], [], []
        n_pairs_w = []
        wi = 0
        for gi, g in enumerate(grps_opt):
            for wf, Kcw in zip(g['envs'], self.setup.Kc_dense[gi] or []):
                if Kcw is None:
                    n_pairs_w.append(0)
                    wi += 1
                    continue
                Kcw = np.asarray(Kcw, complex)
                np_w = 0
                ii, jj = np.nonzero(np.abs(Kcw) > 1e-12)
                for a, b in zip(ii, jj):
                    if abs(self.D_full[a] - self.D_full[b]) < BW_OPT:
                        iis.append(int(a))
                        jjs.append(int(b))
                        vals.append(Kcw[a, b])
                        dDs.append(float(self.D_full[a] - self.D_full[b]))
                        wids.append(wi)
                        np_w += 1
                n_pairs_w.append(np_w)
                wi += 1
        for g in grps_mw:
            w = float(g['omega_mhz'])
            for A, env in zip(g['As'], g['envs']):
                A = np.asarray(A, float)
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
                            dD_ab = float(self.D_full[a] - self.D_full[b])
                            if abs(abs(dD_ab) - w) > MW_NRES_BW_MHZ:
                                continue
                            iis.append(a)
                            jjs.append(b)
                            vals.append(complex(A[i, j]))
                            dDs.append(dD_ab)
                            wids.append(wi)
                            np_w += 1
                n_pairs_w.append(np_w)
                wi += 1
        self.iis = np.asarray(iis, np.int64)
        self.jjs = np.asarray(jjs, np.int64)
        self.vals = np.asarray(vals, np.complex128)
        self.dDs = np.asarray(dDs, np.float64)
        self.wids = np.asarray(wids, np.int64)
        self.n_pairs_w = np.asarray(n_pairs_w, np.int64)
        self._is_mw = np.asarray(self._is_mw, np.int64)
        self._wcos = np.asarray(self._wcos, np.float64)

        # ---- 远失谐光场对 AC 位移解析注入（t43）----
        # BW_OPT 过滤丢掉的远失谐对（|ΔD|≥BW_OPT、|Kc| 显著）的二阶微扰位移：
        # δE_a += |Kc|²/(4ΔD)、δE_b += |Kc|²/(4·(−ΔD))——ref（全 Sambe）的
        # dressed 自然涌现被过滤丢失，用解析二阶补齐（Ω/Δ~0.046 需验证够用）。
        ac = np.zeros(self.dim)
        n_ac = 0
        for gi, g in enumerate(grps_opt):
            for Kcw in self.setup.Kc_dense[gi] or []:
                if Kcw is None:
                    continue
                Kcw = np.asarray(Kcw, complex)
                ii, jj = np.nonzero(np.abs(Kcw) > 1e-12)
                for a, b in zip(ii, jj):
                    dD = float(self.D_full[a] - self.D_full[b])
                    if abs(dD) < BW_OPT:
                        continue
                    v = abs(Kcw[a, b])
                    if v < AC_MIN_V:
                        continue
                    ac[a] += v * v / (4.0 * dD)
                    ac[b] += v * v / (4.0 * (-dD))
                    n_ac += 1
        self.ac_diag = ac
        self.n_ac_pairs = n_ac

    def make_env_tabs(self, T, n_tab=62001):
        self.env_dt = T / (n_tab - 1)
        tab = np.zeros((len(self._env_list), n_tab))
        ts = np.linspace(0.0, T, n_tab)
        for wi, env in enumerate(self._env_list):
            tab[wi] = np.asarray(env(ts), float)
        self.env_tab = tab
        return tab


def evolve_mixed(states, fields, psi0, t_grid, tol=1e-8, max_step=0.05,
                 progress_arr=None, log=None, record_amp=False,
                 driver='numba'):
    """混合引擎主入口：φ-帧单 DOP853（光场 Sambe 结构 + 微波 cos 硬积分）。

    fields : kind='exact' 场列表（build_fields_exact 输出；光场 + 微波混合）
    tol    : DOP853 相对容差
    max_step : DOP853 最大步长（ns；微波 17GHz 载波自动限步，0.05 保守）
    driver : 'numba'（dop853 核心层）| 'scipy'（对照）
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
    T = float(t[-1])
    setup.make_env_tabs(T)
    log(f'[mixed] 光场 P={setup.P} dim={setup.dim}；波 {len(setup._wcos)}'
        f'（mw {int(setup._is_mw.sum())}）；耦合对 {len(setup.iis)}'
        f'；AC 注入远失谐对 {setup.n_ac_pairs}')

    X0 = _embed(psi0v, setup.setup)
    prog = (np.zeros(2) if progress_arr is None
            else np.asarray(progress_arr, float))
    stat = np.zeros(4)

    if driver == 'numba':
        from .dop853 import _dop853_driver
        y_out = _dop853_driver(_rhs_phi, X0, t, T, float(tol), 1e-10,
                               max_step, prog, stat, setup.iis, setup.jjs,
                               setup.vals, setup.dDs, setup.wids,
                               setup._is_mw, setup._wcos, setup.n_pairs_w,
                               setup.env_tab, setup.env_dt, setup.ac_diag,
                               float(C))
    else:
        from scipy.integrate import solve_ivp

        def rhs(tv, phi):
            return _rhs_phi(tv, phi, setup.iis, setup.jjs, setup.vals,
                            setup.dDs, setup.wids, setup._is_mw,
                            setup._wcos, setup.n_pairs_w, setup.env_tab,
                            setup.env_dt, setup.ac_diag, C)
        sol = solve_ivp(rhs, (0.0, T), X0, method='DOP853', t_eval=t,
                        rtol=tol, atol=1e-10, max_step=max_step)
        if not sol.success:
            raise RuntimeError(f'混合引擎积分失败: {sol.message}')
        y_out = sol.y.T       # (len(t), dim) 统一布局

    # 输出：φ → X = e^{−iC·D·t}·φ → 投影物理布居
    # y_out: (len(t), dim)（numba driver 与 scipy 均此布局）
    pops = np.zeros((len(t), N))
    for i, ti in enumerate(t):
        ph = np.exp(-1j * C * setup.D_full * ti)
        Xt = (ph * y_out[i]).reshape(setup.P, N)
        pops[i] = np.abs(Xt.conj().T @ Xt).diagonal().real
    if progress_arr is not None:
        progress_arr[0] = 1.0
    norm_dev = float(np.max(np.abs(pops.sum(axis=1) - 1.0)))
    out = dict(t=t, pops=pops, states=states,
               om_mhz=[float(np.abs(np.asarray(g['As'][0])).max())
                       for g in grps_mw] if grps_mw else [],
               method='mixed', backend='mixed-phi-dop853',
               loss=np.zeros(len(t)),
               error_report=dict(n_segments=int(stat[1]),
                                 nfev=int(stat[0]), n_acc=int(stat[1]),
                                 n_rej=int(stat[2]), dt_min_hit=False,
                                 max_norm_dev=norm_dev,
                                 P=setup.P, dim=setup.dim))
    if record_amp:
        Xf = np.exp(-1j * C * setup.D_full * T) * y_out[-1]
        out['psi_amp'] = Xf
    return out
