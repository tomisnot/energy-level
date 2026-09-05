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

from . import evolve as _evolve
from .field_build import _group_by_freq
from .sambe import (_SambeSetup, _m_analytic, _embed)

_CONV = 2 * np.pi * 1e-3
MW_NRES_BW_MHZ = 5e3          # 微波近共振带宽
BW_OPT = 2e4                  # 光场反旋过滤带宽（20 GHz）
R_CRIT = 0.0135               # t58: M_max=2 的 r 判据（r>r_crit → AC 化）


def ac_ize_physical_E(E, f_opt, r_crit=R_CRIT):
    """t63 统一：公共层 acize.acize_physical（单/两体共用——env² 全量时变
    语义 + K0 纯 E 唯一实现）。返回 (E, A_new_list, shifts_list)。"""
    from .acize import acize_physical
    A_new, shifts_list = acize_physical(E, f_opt, r_crit)
    return E, A_new, shifts_list


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
        self.D_full = np.real(np.asarray(self.setup.diag))  # t73: K0→diag

        # 波清单（光场在前 + 微波）——log / AC 映射用
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
        # 耦合对 7 元组（a<b 单侧——反向由共享内核厄米补齐，与两体同契约；
        # t70 三角过滤：Kcw 为 Hermitian 全矩阵，双向打包会把同一耦合驱动两次）
        self.pairs = []
        for gi, g in enumerate(grps_opt):
            Kcw_list = self.setup.Kc_dense[gi] or []
            for wi_local, wf in enumerate(g['envs']):
                if wi_local >= len(Kcw_list):
                    continue            # AC 化后零耦合波（t62e：仅占波行）
                Kcw = np.asarray(Kcw_list[wi_local], complex)
                ii, jj = np.nonzero(np.abs(Kcw) > 1e-12)
                for a, b in zip(ii, jj):
                    if a >= b:
                        continue
                    if abs(self.D_full[a] - self.D_full[b]) < BW_OPT:
                        self.pairs.append(
                            (int(a), int(b), Kcw[a, b],
                             float(self.D_full[a] - self.D_full[b]),
                             0, 0.0, wf))
        for g in grps_mw:
            w = float(g['omega_mhz'])
            for A, env in zip(g['As'], g['envs']):
                A = np.asarray(A, float)
                for sa in range(self.P):
                    for i in range(N):
                        for k in range(i + 1, N):
                            if abs(A[i, k]) < 1e-12:
                                continue
                            a = sa * N + i
                            b = sa * N + k
                            dE = float(self.D_full[a] - self.D_full[b])
                            if abs(abs(dE) - w) > MW_NRES_BW_MHZ:
                                continue
                            self.pairs.append((a, b, complex(A[i, k]), dE,
                                               1, w, env))
        # t61 清理：t44 的 eigh 法远对净 AC 位移块已删——被 t62 far_sos 注入
        # 物理能级取代（见 docs/混合引擎理论推导-光Sambe微波当场.md）。
        self._is_mw = np.asarray(self._is_mw, np.int64)
        self._wcos = np.asarray(self._wcos, np.float64)

    def pack(self, T, n_tab=20001):
        """pairs → coupling.pack_rhs 数组（t73：与两体 mixed 同一打包层）。"""
        from .coupling import pack_rhs
        self.pack = pack_rhs(self.pairs, T, n_tab=n_tab)
        return self.pack


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
    from .coupling import stage_progress
    t = np.asarray(t_grid, float)
    if progress_arr is None:
        progress_arr = np.zeros(3)
    stage_progress(progress_arr, 0, 0.0)     # 阶段 0：装配/JIT 装载
    N = len(states)
    E = np.array([s['E_mhz'] for s in states], float)
    E = E - float(E.min())
    psi0v = _evolve._as_vector(states, psi0)
    C = _CONV

    # t62 耗散（落点=丢失）：Gamma_ns（/ns）→ gamma（MHz）传给内核
    gamma = None
    if decays:
        from .evolve_util import _single_loss_rates
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
    # t62e/t73：AC 时变对角——per-wave (n_ac, dim) + 独立 env 表（与两体
    # mixed 同款；波形对象同一性映射——t62 分组教训）
    ac_waves = []
    for gi, g in enumerate(grps_opt):
        for wf in g['envs']:
            for f_idx, f in enumerate(f_opt):
                if f['waveform'] is wf and np.any(shifts_list[f_idx]):
                    ac_waves.append((shifts_list[f_idx], wf))
    # 耗散 gamma（N）→ 扩展空间（N·P，每扇区同衰减）
    if gamma is not None:
        gamma = np.tile(gamma, setup.P)
    E_shift = E2 - E
    n_ac = int(np.count_nonzero(np.abs(E_shift) > 1e-9))
    log(f'[mixed] AC 化位移范围 {E_shift.min():+.1f}..{E_shift.max():+.1f} MHz'
        f'（{n_ac} 态）；P={setup.P} dim={setup.dim}')
    T = float(t[-1])
    ac_base = None
    ac_env_tab = None
    ac_env_dt = 1.0
    if ac_waves:
        n_acv = len(ac_waves)
        n_ac_tab = 4001
        ac_base = np.zeros((n_acv, setup.dim), float)
        ac_env_tab = np.zeros((n_acv, n_ac_tab), float)
        ts_ac = np.linspace(0.0, T, n_ac_tab)
        for wi, (sh_f, wf) in enumerate(ac_waves):
            for i in range(N):
                v = sh_f[i]
                if v != 0.0:
                    for pn in range(setup.P):
                        ac_base[wi, pn * N + i] = v
            ac_env_tab[wi] = np.asarray(wf(ts_ac), float)
        ac_env_dt = T / (n_ac_tab - 1)
    pack = setup.pack(T)
    log(f'[mixed] 波 {len(setup._wcos)}'
        f'（mw {int(setup._is_mw.sum())}）；耦合对 {len(setup.pairs)}')

    X0 = _embed(psi0v, setup.setup)
    prog = (np.zeros(3) if progress_arr is None
            else np.asarray(progress_arr, float))
    stat = np.zeros(4)
    stage_progress(prog, 0, 0.9)             # 装配完成（JIT 装载在驱动首调）

    if driver == 'numba':
        from .dop853 import _dop853_driver
        from .coupling import rhs_phi_pairs
        # 阶段 1（积分）由驱动在首个接受步置位——调用内的 numba 装载/
        # 编译期保持阶段 0（"准备"标签 + 耗时），GUI 显示不失真
        y_out = _dop853_driver(rhs_phi_pairs, X0, t, T, float(tol), 1e-10,
                               max_step, prog, stat, pack['iis'], pack['jjs'],
                               pack['vals'], pack['dDs'], pack['is_mw'],
                               pack['wcos'], pack['n_pairs_w'],
                               pack['env_tab'], pack['env_dt'], float(C),
                               gamma, ac_base, ac_env_tab, ac_env_dt)
    else:
        from scipy.integrate import solve_ivp
        from .coupling import rhs_phi_pairs
        stage_progress(prog, 1, 0.0)         # 阶段 1：积分（scipy 无内部钩子）

        def rhs(tv, phi):
            return rhs_phi_pairs(tv, phi, pack['iis'], pack['jjs'],
                                 pack['vals'], pack['dDs'], pack['is_mw'],
                                 pack['wcos'], pack['n_pairs_w'],
                                 pack['env_tab'], pack['env_dt'], C,
                                 gamma, ac_base, ac_env_tab, ac_env_dt)
        sol = solve_ivp(rhs, (0.0, T), X0, method='DOP853', t_eval=t,
                        rtol=tol, atol=1e-10, max_step=max_step)
        if not sol.success:
            raise RuntimeError(f'混合引擎积分失败: {sol.message}')
        y_out = sol.y.T       # (len(t), dim) 统一布局

    # 输出：φ → X = e^{−iC·D·t}·φ → 投影物理布居
    # y_out: (len(t), dim)（numba driver 与 scipy 均此布局）
    stage_progress(prog, 2, 0.0)             # 阶段 2：收尾（φ→物理投影）
    from .sambe import sector_sum_pops, sector_sum_amp
    pops = np.zeros((len(t), N))
    for i in range(len(t)):
        # t73b：相干扇和大（解析精确——光子相位恰对消；旧非相干和带
        # O(Ω/ω) 系统误差，22 态场景实测 ~4e-3）
        pops[i] = sector_sum_pops(y_out[i], setup.P, N)
        stage_progress(prog, 2, i / max(len(t) - 1, 1))
    stage_progress(prog, 2, 1.0)
    # t62 耗散：loss 曲线（梯形积分）+ 含 loss 范数偏差（同 hard）
    from .evolve_util import _loss_and_norm
    Gamma_ns = np.zeros(N)
    if decays:
        from .evolve_util import _single_loss_rates
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
        from .sambe import sector_sum_amp
        out['psi_amp'] = sector_sum_amp(y_out[-1], setup.P, N)
    return out
