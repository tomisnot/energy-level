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
R_CRIT = 0.0135               # t58: M_max=2 的 r 判据（r>r_crit → AC 化）


def ac_ize_physical_E(E, f_opt, r_crit=R_CRIT):
    """t63 统一：公共层 acize.acize_physical（单/两体共用——env² 全量时变
    语义 + K0 纯 E 唯一实现）。返回 (E, A_new_list, shifts_list)。"""
    from .acize import acize_physical
    A_new, shifts_list = acize_physical(E, f_opt, r_crit)
    return E, A_new, shifts_list


@njit(cache=True)
def _rhs_phi(t, phi, iis, jjs, vals, dDs, wids, is_mw, wcos,
             n_pairs_w, env_tab, env_dt, C, gamma=None,
             ac_sh=None, ac_pos=None, ac_N=0):
    """φ-帧统一 RHS：−iC·[光场 env·Kc + 微波 env·cos·A]∘e^{iC·ΔD·t}·φ。

    t 为绝对时间（φ=e^{iCDt}X 定义是绝对的）。wids[k] → 波索引；
    is_mw[w]=1 时该项带 cos(C·w·t)。
    gamma（N，MHz，可 None）：t62 耗散 NonHermitian 对角（落点=丢失，
    与 hard 内核同语义——out[i] −= 0.5·C·gamma[i]·phi[i]）。"""
    tv = t
    tot = np.zeros_like(phi)
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
    res = -1j * C * tot
    # t62e：时变 AC 对角修正 −iC·Σ_f ac_sh[f]·(env_f(t)²−1)∘φ——env 平方
    # （AC ∝ 光强）：env=1 → 0（K0/E2 静态已含）；env=0 → −shift（光关位移归零，
    # 微波共振恢复）。加在 res（不经 tot 的 -1jC——避免双乘）。
    if ac_sh is not None:
        dim = phi.shape[0]
        for f in range(ac_sh.shape[0]):
            w = ac_pos[f]
            ei = tv / env_dt
            i0 = int(ei)
            if i0 >= env_tab.shape[1] - 1:
                i0 = env_tab.shape[1] - 2
            ff = ei - i0
            envv = env_tab[w, i0] * (1.0 - ff) + env_tab[w, i0 + 1] * ff
            dev = envv * envv
            if dev == 0.0:
                continue
            for a in range(dim):
                st = a % ac_N
                res[a] += -1j * C * dev * ac_sh[f, st] * phi[a]
    if gamma is not None:
        # 耗散对角项直接加在返回（不经 -1j·C——否则变虚数增长）；
        # gamma 须为扩展空间长度（N·P，调用侧已按扇区 tile）
        for i in range(phi.shape[0]):
            if gamma[i] != 0.0:
                res[i] -= 0.5 * C * gamma[i] * phi[i]
    return res


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
        # t62e：env 行含全部光学波（直接遍历 g['envs']——Kc_dense 空（AC 化后
        # 零耦合）时 zip 会吞掉 env 行——RHS 时变 AC 对角需其 env 表）
        self._env_list = []
        self._is_mw = []
        self._wcos = []
        for gi, g in enumerate(grps_opt):
            for wf in g['envs']:
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
            for wf in g['envs']:
                Kcw_list = self.setup.Kc_dense[gi] or []
                Kcw = None
                # 第 wi 个波对应该组的第 wi_local 个 Kcw——按序取
                # （简化：env 与 Kcw 同序——直接按组内序 zip）
                # t62e：Kc_dense 空（AC 化零耦合）时 Kcw=None → 波计入 wi
                wi_in_g = 0
                for wf2 in g['envs']:
                    if wf2 is wf:
                        break
                    wi_in_g += 1
                if wi_in_g < len(Kcw_list):
                    Kcw = Kcw_list[wi_in_g]
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
        # t61 清理：t44 的 eigh 法远对净 AC 位移块（K_far 对角化 + ac_diag）
        # 已删除——已被 t62 的 far_sos 注入物理能级 E_i 取代（混合引擎正确组合
        # docs/混合引擎理论推导-光Sambe微波当场.md：AC 化注入 E → K0/D 同步 → 光场
        # 失谐与微波共振自动含位移）。ac_diag 相关 RHS 分支同步移除。
        # t62e：AC 位移时变化——per-field 位移 + 对应光学波索引（RHS 时变对角
        # δE_i(t)=Σ_f shifts_f[i]·env_f(t)²——K0/E2 含静态 Σ，RHS 补 (env²−1)）
        self.ac_shifts = []            # 每光学波的 N 维位移（无位移的光学波=None）
        self.ac_wave_idx = []          # 对应 _env_list 波索引
        self.ac_env_pos = []           # 波在 env_tab 的行号
        self.ac_N = N

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
                 driver='numba', decays=None):
    """混合引擎主入口：φ-帧单 DOP853（光场 Sambe 结构 + 微波 cos 硬积分）。

    fields : kind='exact' 场列表（build_fields_exact 输出；光场 + 微波混合）
    tol    : DOP853 相对容差
    max_step : DOP853 最大步长（ns；微波 17GHz 载波自动限步，0.05 保守）
    driver : 'numba'（dop853 核心层）| 'scipy'（对照）
    decays : t62 耗散——寿命通道列表（落点=丢失，NonHermitian 对角；
            与 hard 引擎同语义，_single_loss_rates；显式落点/纯退相拒绝）
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

    # t62 耗散（落点=丢失）：Gamma_ns（/ns）→ gamma（MHz）传给内核
    gamma = None
    if decays:
        from .hard import _single_loss_rates
        Gamma_ns = _single_loss_rates(states, decays)
        if np.any(Gamma_ns):
            gamma = Gamma_ns / _CONV
            log(f'[mixed] 耗散 {int((Gamma_ns > 0).sum())} 态'
                f'（落点=丢失，NonHermitian）')
    f_opt = [f for f in fields if f['omega_L_mhz'] > 1e5]
    f_mw = [f for f in fields if f['omega_L_mhz'] <= 1e5]
    # t62 AC 化（正确组合）+ t62e 时变化：远强对剔除 + per-field far_sos 位移
    E2, A_opt_new, shifts_list = ac_ize_physical_E(E, f_opt)
    f_opt_ac = [dict(f, A=A) for f, A in zip(f_opt, A_opt_new)]
    grps_opt = _group_by_freq(f_opt_ac)
    grps_mw = _group_by_freq(f_mw) if f_mw else []
    if not grps_opt:
        raise ValueError('evolve_mixed 需要光场（载频 > 1e5 MHz）')
    setup = _MixedSetup(E2, grps_opt, grps_mw, N)
    # t62e：AC 时变数据——光学波在 env 序列的位置（对齐 _env_list 全波行）+
    # per-field 位移（波形对象同一性映射——t62 分组教训）
    ac_sh, ac_pos = [], []
    wi = 0
    for gi, g in enumerate(grps_opt):
        for wf in g['envs']:
            for f_idx, f in enumerate(f_opt):
                if f['waveform'] is wf and np.any(shifts_list[f_idx]):
                    ac_sh.append(shifts_list[f_idx])
                    ac_pos.append(wi)
            wi += 1
    ac_sh = np.asarray(ac_sh, float) if ac_sh else None
    ac_pos = np.asarray(ac_pos, np.int64) if ac_pos else None
    # 耗散 gamma（N）→ 扩展空间（N·P，每扇区同衰减）
    if gamma is not None:
        gamma = np.tile(gamma, setup.P)
    E_shift = E2 - E
    n_ac = int(np.count_nonzero(np.abs(E_shift) > 1e-9))
    log(f'[mixed] AC 化位移范围 {E_shift.min():+.1f}..{E_shift.max():+.1f} MHz'
        f'（{n_ac} 态）；P={setup.P} dim={setup.dim}')
    T = float(t[-1])
    setup.make_env_tabs(T)
    log(f'[mixed] 波 {len(setup._wcos)}'
        f'（mw {int(setup._is_mw.sum())}）；耦合对 {len(setup.iis)}')

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
                               setup.env_tab, setup.env_dt, float(C),
                               gamma, ac_sh, ac_pos, int(N))
    else:
        from scipy.integrate import solve_ivp

        def rhs(tv, phi):
            return _rhs_phi(tv, phi, setup.iis, setup.jjs, setup.vals,
                            setup.dDs, setup.wids, setup._is_mw,
                            setup._wcos, setup.n_pairs_w, setup.env_tab,
                            setup.env_dt, C, gamma, ac_sh, ac_pos, int(N))
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
    # t62 耗散：loss 曲线（梯形积分）+ 含 loss 范数偏差（同 hard）
    from .hard import _loss_and_norm
    Gamma_ns = np.zeros(N)
    if decays:
        from .hard import _single_loss_rates
        Gamma_ns = _single_loss_rates(states, decays)
    loss, norm_dev = _loss_and_norm(pops, Gamma_ns, t)
    out = dict(t=t, pops=pops, states=states,
               om_mhz=[float(np.abs(np.asarray(g['As'][0])).max())
                       for g in grps_mw] if grps_mw else [],
               method='mixed', backend='mixed-phi-dop853',
               loss=loss,
               error_report=dict(n_segments=int(stat[1]),
                                 nfev=int(stat[0]), n_acc=int(stat[1]),
                                 n_rej=int(stat[2]), dt_min_hit=False,
                                 max_norm_dev=norm_dev,
                                 P=setup.P, dim=setup.dim))
    if record_amp:
        Xf = np.exp(-1j * C * setup.D_full * T) * y_out[-1]
        out['psi_amp'] = Xf
    return out
