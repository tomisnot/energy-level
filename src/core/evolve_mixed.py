# -*- coding: utf-8 -*-
"""混合引擎总模块（t74：单原子 + 两体合一，原 evolve_mixed_pair.py 并入；
acize 层亦并入——其唯一消费者家族就是 mixed）。架构：光场 Sambe 结构
（光子块免 RWA）+ 微波 cos 硬积分——φ-帧单 DOP853（t42）。

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

from .evolve_util import _as_vector  # t74 工具归位
from . import tensor as _tensor
from .params import (BW_OPT, MW_NRES_BW_MHZ, R_CRIT,  # t74c 参数单源
                     MW_MAX_MHZ)
from .coupling import (rhs_phi_pairs,  # t74：φ-帧统一内核（单/两体共用）
                       dop853_driver_mixed)  # 薄包装：让磁盘缓存命中
from .field_build import _group_by_freq
from .sambe import (_SambeSetup, _m_analytic, _embed)

_CONV = 2 * np.pi * 1e-3
# ---- AC 化（t74 自 acize.py 并入——唯一消费者家族 = mixed）----


def classify_pairs(A, omega, r_crit=R_CRIT):
    """按 r=Ω/ω 分类：True = 保留（近/弱对——光子块动态）；False = AC 化。"""
    A = np.asarray(A, float)
    r = np.abs(A) / omega
    keep = (r <= r_crit) & (np.abs(A) > 1e-9)
    return keep


def far_sos_pair_shift(A_ij, E_k, E_i, omega):
    """单对 (i,k) 对源态 i 的 A² 等价 far_sos 位移（MHz）。

    = −A_ij²·(E_k−E_i)/(2·((E_k−E_i)²−ω²))——与
    static_atom.far_sos_shift_mhz（d²·w/(w²−ω²) 结构）同语义；A² ∝ d²·E0²
    的 Rabi 形式。t60 仲裁：±18.23 与 M=4 锚一致（C1≈A 6.87e-3）。
    """
    nu = float(E_k) - float(E_i)
    return -(float(A_ij) ** 2) * nu / (nu ** 2 - omega ** 2) / 2.0


def acize_physical(E, fields_opt, r_crit=R_CRIT):
    """远强对 AC 化（唯一实现）：剔除 + per-field 位移。

    fields_opt: kind='exact' 光场列表（A 全偶极矩阵 + omega_L_mhz）。
    返回 (A_new_list, shifts_list)：
      A_new_list[f]：剔除远强对后的光场 A（近对保留）
      shifts_list[f]：第 f 个光场的 N 维位移向量（env² 全量时变语义——
        K0 用纯 E；RHS 时变对角 Σ_f shifts_f·env_f(t)² 由调用侧组装）
    """
    N = len(E)
    shifts_list = []
    A_new = []
    for f in fields_opt:
        A = np.asarray(f['A'], float).copy()
        w = float(f['omega_L_mhz'])
        sh_f = np.zeros(N)
        keep = classify_pairs(A, w, r_crit)
        for i in range(N):
            for k in range(i + 1, N):
                if not keep[i, k]:
                    if abs(A[i, k]) < 1e-9:
                        continue
                    dE = far_sos_pair_shift(A[i, k], E[k], E[i], w)
                    sh_f[i] += dE
                    sh_f[k] -= dE
                    A[i, k] = A[k, i] = 0.0
        A_new.append(A)
        shifts_list.append(sh_f)
    return A_new, shifts_list
def ac_ize_physical_E(E, f_opt, r_crit=R_CRIT):
    """远强对 AC 化入口（t74：acize 层并入本模块——唯一消费者家族）。"""
    A_new, shifts_list = acize_physical(E, f_opt, r_crit)
    return E, A_new, shifts_list


def _assemble_H_snapshot(D_full, pairs, t, C, ac_base=None, ac_waves=None,
                         gamma=None, tags=None):
    """t75 诊断：重组 t 时刻的扩展空间哈密顿 H（MHz，数值显示用）。

    H(t) = diag(K0) + Σ_光场 env·Kc∘e^{iCΔD·t} + Σ_微波 env·cos(Cωt)·A∘e^{iCΔD·t}
           + AC 对角 env² + 耗散 −iΓ/2——与 rhs_phi_pairs 逐项同语义
    （v 已含光子块 ½ 约定；不在 njit 内核里插桩，事后重建即交叉验证）。
    tags：与 pairs 等长的分项名列表（两体用，区分 V_dd/光场/微波）；
    None 时按 is_mw 二分。返回分项 dict（键按实际内容懒建）。
    """
    t = float(t)
    dim = np.asarray(D_full).size
    terms = dict(
        K0=np.diag(np.asarray(D_full, complex)),
        AC对角=np.zeros((dim, dim), complex),
        耗散=np.zeros((dim, dim), complex),
    )
    for k, (a, b, v, dD, is_mw, w, env) in enumerate(pairs):
        e = float(np.atleast_1d(env(np.array([t])))[0])
        if e == 0.0:
            continue
        ph = np.exp(1j * C * dD * t)
        cw = e * (np.cos(C * w * t) if is_mw else 1.0)
        name = tags[k] if tags is not None else ('微波' if is_mw else '光场')
        if name not in terms:
            terms[name] = np.zeros((dim, dim), complex)
        val = cw * v * ph
        terms[name][a, b] += val
        terms[name][b, a] += np.conj(val)
    if ac_base is not None and ac_waves:
        for wi, (sh, wf) in enumerate(ac_waves):
            dev = float(np.atleast_1d(wf(np.array([t])))[0]) ** 2
            if dev != 0.0:
                terms['AC对角'] += np.diag(dev * np.asarray(ac_base[wi], float))
    if gamma is not None:
        g = np.asarray(gamma, float)
        if np.any(g):
            terms['耗散'] = np.diag(-0.5j * g)
    return terms


class _MixedSetup:
    """混合引擎预计算：光场光子块 + φ-帧耦合对统一打包。"""

    def __init__(self, E, grps_opt, grps_mw, N, M_override=None,
                 bw_opt=None, mw_nres_bw=None):
        self.N = N
        self.E = np.asarray(E, float)
        self.grps_opt = grps_opt
        self.grps_mw = grps_mw
        # t75：M 覆写（上帝模式自定义）——int 全组 / list 逐组；None=自动探针
        if M_override is None:
            Ms = _m_analytic(grps_opt, 1e-6, self.E)
        elif np.isscalar(M_override):
            Ms = [int(M_override)] * len(grps_opt)
        else:
            Ms = [int(m) for m in M_override]
            if len(Ms) != len(grps_opt):
                raise ValueError(
                    f'per_freq_M 长度 {len(Ms)} != 载频组数 {len(grps_opt)}')
        self.M_used = list(Ms)
        # t75：过滤带宽覆写（上帝模式）——None=params.py 默认
        self.bw = float(bw_opt) if bw_opt is not None else BW_OPT
        self.mw_nres = (float(mw_nres_bw) if mw_nres_bw is not None
                        else MW_NRES_BW_MHZ)
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
                    if abs(self.D_full[a] - self.D_full[b]) < self.bw:
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
                            if abs(abs(dE) - w) > self.mw_nres:
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
                 driver='numba', decays=None,
                 per_freq_M=None, r_crit=None, diag=False,
                 bw_opt=None, mw_nres_bw=None, mw_max=None):
    """混合引擎主入口：φ-帧单 DOP853（光场 Sambe 结构 + 微波 cos 硬积分）。

    fields : kind='exact' 场列表（build_fields_exact 输出；光场 + 微波混合）
    tol    : DOP853 相对容差
    max_step : DOP853 最大步长（ns；微波 17GHz 载波自动限步，0.05 保守）
    driver : 'numba'（dop853 核心层）| 'scipy'（对照）
    decays : t62 耗散——寿命通道列表（落点=丢失，NonHermitian 对角；
            与 hard 引擎同语义，_single_loss_rates；显式落点/纯退相拒绝）
    per_freq_M : t75 上帝模式覆写——逐载频光子块阶数（int 全组/list 逐组；
            None=自动探针 _m_analytic）
    r_crit : t75 上帝模式覆写——AC 化 r=Ω/ω 阈值（None=默认 R_CRIT）
    diag   : t75 诊断快照——True 时 out['diag'] 附结构化中间产物
    bw_opt / mw_nres_bw / mw_max : t75 上帝模式覆写——φ-帧过滤带宽 /
            微波近共振带宽 / 微波-光场分界（None=params.py 默认）
            （能级/AC 化/光子块几何/耦合对/H(t) 快照/步长历史/扇区轨迹）
    bw_opt / mw_nres_bw / mw_max : t75 上帝模式覆写——φ-帧光场过滤带宽 /
            微波近共振带宽 / 微波-光场分界（None=params.py 默认）
    """
    log = log or (lambda *a: None)
    from .coupling import stage_progress, rhs_phi_pairs
    t = np.asarray(t_grid, float)
    if progress_arr is None:
        progress_arr = np.zeros(3)
    stage_progress(progress_arr, 0, 0.0)     # 阶段 0：装配/JIT 装载
    N = len(states)
    E = np.array([s['E_mhz'] for s in states], float)
    E = E - float(E.min())
    psi0v = _as_vector(states, psi0)
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
    mm = float(mw_max) if mw_max is not None else MW_MAX_MHZ
    f_opt = [f for f in fields if f['omega_L_mhz'] > mm]
    f_mw = [f for f in fields if f['omega_L_mhz'] <= mm]
    # t62 AC 化（正确组合）+ t62e 时变化：远强对剔除 + per-field far_sos 位移
    # t75：r_crit 可由上帝模式覆写（None=R_CRIT 默认）
    rc = float(r_crit) if r_crit is not None else R_CRIT
    E2, A_opt_new, shifts_list = ac_ize_physical_E(E, f_opt, rc)
    f_opt_ac = [dict(f, A=A) for f, A in zip(f_opt, A_opt_new)]
    grps_opt = _group_by_freq(f_opt_ac)
    grps_mw = _group_by_freq(f_mw) if f_mw else []
    if not grps_opt:
        raise ValueError('evolve_mixed 需要光场（载频 > %.0e MHz）' % mm)
    setup = _MixedSetup(E2, grps_opt, grps_mw, N, M_override=per_freq_M,
                        bw_opt=bw_opt, mw_nres_bw=mw_nres_bw)
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
    stat = np.zeros(6) if diag else np.zeros(4)
    dt_hist = (np.zeros((100000, 2)) if diag else None)
    stage_progress(prog, 0, 0.9)             # 装配完成（JIT 装载在驱动首调）

    if driver == 'numba':
        # 阶段 1（积分）由驱动在首个接受步置位——调用内的 numba 装载/
        # 编译期保持阶段 0（"准备"标签 + 耗时），GUI 显示不失真
        y_out = dop853_driver_mixed(X0, t, T, float(tol), 1e-10,
                                    max_step, prog, stat, dt_hist,
                                    pack['iis'], pack['jjs'],
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
    # ---- t75 诊断快照（diag=True 才采集；数值路径零改动）----
    if diag:
        pr = setup.pairs
        diag_pairs = dict(
            a=np.array([p[0] for p in pr], int),
            b=np.array([p[1] for p in pr], int),
            v=np.array([p[2] for p in pr], complex),
            dD=np.array([p[3] for p in pr], float),
            is_mw=np.array([p[4] for p in pr], int),
            w=np.array([p[5] for p in pr], float))
        acize_fields = []
        for f0, f1, sh in zip(f_opt, A_opt_new, shifts_list):
            n0 = int((np.abs(np.asarray(f0['A'], float)) > 1e-9).sum() // 2)
            n1 = int((np.abs(np.asarray(f1, float)) > 1e-9).sum() // 2)
            acize_fields.append(dict(
                omega=float(f0['omega_L_mhz']), n_before=n0, n_kept=n1,
                n_acized=n0 - n1,
                shift_min=float(sh.min()), shift_max=float(sh.max())))
        idx = np.linspace(0, len(t) - 1, min(len(t), 240)).astype(int)
        sector_amps = {p: np.abs(y_out[idx][:, p * N:(p + 1) * N])
                       for p in range(setup.P)}
        ts_snap = sorted(set(float(x) for x in
                             np.linspace(0.0, T, 5)))
        snaps = {}
        for t0 in ts_snap:
            snaps[t0] = _assemble_H_snapshot(
                setup.D_full, pr, t0, C, ac_base=ac_base,
                ac_waves=ac_waves if ac_base is not None else None,
                gamma=gamma)
        diag_dt = (dt_hist[:int(stat[4])] if dt_hist is not None
                   else np.zeros((0, 2)))
        out_diag = dict(
            meta=dict(method='mixed', backend='mixed-phi-dop853', N=N,
                      P=setup.P, dim=setup.dim, t_end=T,
                      M=list(setup.M_used), M_user=(per_freq_M is not None),
                      r_crit=float(rc), n_pairs=len(pr), tol=float(tol),
                      max_step=float(max_step), bw=float(setup.bw),
                      mw_nres=float(setup.mw_nres), mw_max=float(mm),
                      labels=[s['label'] for s in states], stride=N),
            photon=dict(Ms=list(setup.M_used), P=setup.P, dim=setup.dim,
                        strides=list(setup.setup.strides),
                        nvecs=np.asarray(setup.setup.nvecs),
                        D_full=np.asarray(setup.D_full, float).copy()),
            acize=dict(r_crit=float(rc), fields=acize_fields,
                       shift=np.asarray(E2 - E, float).copy()),
            pairs=diag_pairs,
            solver=dict(t_steps=diag_dt[:, 0].copy(),
                        dt=diag_dt[:, 1].copy(),
                        nfev=int(stat[0]), n_acc=int(stat[1]),
                        n_rej=int(stat[2]), norm_dev=norm_dev),
            trajectory=dict(t=t[idx].copy(), sector_amps=sector_amps),
            H_snapshots=snaps,
        )
    out = dict(t=t, pops=pops, states=states,
               om_mhz=[float(np.abs(np.asarray(g['As'][0])).max())
                       for g in grps_mw] if grps_mw else [],
               method='mixed', backend='mixed-phi-dop853',
               loss=loss,
               diag=out_diag if diag else None,
               error_report=dict(n_segments=int(stat[1]),
                                 nfev=int(stat[0]), n_acc=int(stat[1]),
                                 n_rej=int(stat[2]), dt_min_hit=False,
                                 max_norm_dev=norm_dev,
                                 P=setup.P, dim=setup.dim))
    if record_amp:
        Xf = np.exp(-1j * C * setup.D_full * T) * y_out[-1]
        out['psi_amp'] = Xf
    return out


# ============================================================
# 两体混合引擎（t74 自 evolve_mixed_pair.py 并入——mixed 家族一个家）
# ============================================================


def _ac_ize_pair(E, A_o, omega_opt, r_crit=R_CRIT):
    """AC 化（t70 定案）：剔除远强对；**位移不注入**。

    近对（如 60S↔6P）保留在光子块 M=1 动态、AC 位移自然涌现——与 hard 的
    RWA 动态同物理（M=1 对角化涌现 −18.87 ≡ 静态引擎 −16.47）。再注入全量
    静态位移 = 与动态双计（375ns 交换 0.012 vs 0.558 的根因，t70 实验定案；
    旧 inject=True 考古语义已随 t73b 深度清理移除）。
    """
    if A_o is None or not np.any(np.abs(A_o) > 1e-9):
        return None
    A_new, _ = acize_physical(
        E, [dict(kind='exact', A=np.asarray(A_o, float),
                 omega_L_mhz=float(omega_opt))], r_crit)
    return A_new[0]


def _build_pair_geometry(E, optA_ac, optB_ac, omega_opt, bw_opt=None):
    """两体光子块结构（t73b 单次装配）：全部光波（已按侧提升）入同一载频组
    → 一个 _SambeSetup 同时给出 K0（纯 E）与逐波 Kcw（t69 逐波形态：
    交叉项不可能出现；a<b 三角过滤 + |ΔD|<BW_OPT 反旋过滤）。

    optA_ac / optB_ac 元素 (A_new, env)——A 侧 lift_side_A(A⊗I)、B 侧 I⊗A。
    """
    from .tensor_fields import lift_side_A
    bw_opt = BW_OPT if bw_opt is None else float(bw_opt)
    from .sambe import _SambeSetup
    from .field_build import _group_by_freq
    N = len(E)
    N2 = N * N
    P = 3
    dim = N2 * P

    def _one(t):
        return np.ones_like(np.atleast_1d(t), float)

    E_pair = np.array([E[i] + E[j]
                       for i in range(N) for j in range(N)])
    flds = []
    envs_out = []
    for side, lst in (('A', optA_ac), ('B', optB_ac)):
        for A_new, env in lst:
            if A_new is None or not np.any(np.abs(A_new) > 1e-9):
                continue
            if env is None:
                env = _one
            flds.append(dict(kind='exact',
                             A=lift_side_A(np.asarray(A_new, float), N, side),
                             omega_L_mhz=float(omega_opt), waveform=env))
            envs_out.append(env)
    setup = _SambeSetup(E_pair, _group_by_freq(flds), [1])
    K0 = np.real(np.asarray(setup.diag))
    pairs_opt = []
    for Kcw, env in zip(setup.Kc_dense[0] or [], envs_out):
        Kcw = np.asarray(Kcw, complex)
        ii, jj = np.nonzero(np.abs(Kcw) > 1e-12)
        for a, b in zip(ii, jj):
            # a<b 三角过滤（t70）：Kcw 为 Hermitian 全矩阵，双向打包会把
            # 同一耦合驱动两次（Rabi 翻倍 → AC 位移失真 4×）；
            # 反向由 RHS 厄米补齐自动。
            if a >= b:
                continue
            if abs(K0[a] - K0[b]) < bw_opt:
                pairs_opt.append((int(a), int(b), Kcw[a, b],
                                  float(K0[a] - K0[b]), env))
    return K0, pairs_opt, P, dim


def _split_waves(fields, mw_max=MW_MAX_MHZ):
    """fields（kind='exact'）→ (opt_waves, mw_waves)：**逐波列表，不合并**。

    每波形独立 (A, omega, env) 元组——最终形态（t69 船长裁定）：废除 t68 的
    同频合并模型（(ΣA)(Σenv) 展开有交叉项 A₁·env₂+A₂·env₁——顺序脉冲如
    波形1@70+副本@200 在 t=70 给 (A₁+A₂)·1=2A₁ → Rabi 翻倍 → π 提前 + 振荡
    回流——与 sambe T4/T6 修过的同频合并 bug 同类）。逐波后每波独立驱动，
    交叉项不可能出现；pack_rhs 按 (is_mw, w, id(env)) 分组天然支持同频多波。
    """
    opt, mw = [], []
    for f in fields or []:
        A = np.asarray(f['A'], float)
        om = float(f['omega_L_mhz'])
        (opt if om > mw_max else mw).append((A, om, f.get('waveform')))
    return opt, mw


def evolve_mixed_pair(states, fields_A, fields_B, psi0_A, psi0_B, t_grid,
                      vdd_matrix=None, R_um=None, theta_deg=0.0,
                      phi_deg=0.0, cache=None, tol=1e-8, max_step=0.05,
                      progress_arr=None, log=None, record_amp=False,
                      decays_A=None, decays_B=None, driver='numba',
                      r_crit=None, diag=False,
                      bw_opt=None, mw_nres_bw=None, mw_max=None):
    """两体混合引擎（正式版）：光场光子块 + 微波/V_dd 当场，φ-帧单 DOP853。

    fields_A/B : kind='exact' 场列表（build_fields_exact 输出——光场 + 微波）
    vdd_matrix  : V_dd 稀疏/稠密矩阵（N²×N²）；None 且给 R_um/cache 时算
    r_crit : t75 上帝模式覆写——AC 化 r=Ω/ω 阈值（None=默认 R_CRIT）
    diag   : t75 诊断快照——True 时 out['diag'] 附结构化中间产物
    bw_opt / mw_nres_bw / mw_max : t75 上帝模式覆写——φ-帧过滤带宽 /
            微波近共振带宽 / 微波-光场分界（None=params.py 默认）
    返回 dict(t, pops_2d, pops_A, pops_B, states_2d, loss_A, loss_B,
              method='mixed', backend='mixed-phi-dop853-pair', error_report)。

    注意：V_dd 在 φ-帧同扇区 → ΔD=E_pair 差自动涌现（与 n 无关）；
    耗散（t62 实现）：寿命通道（落点=丢失）NonHermitian 对角——与
    hard_pair 同语义（_pair_loss_vectors；显式落点/纯退相拒绝）。
    """
    log = log or (lambda *a: None)
    from .coupling import stage_progress
    if progress_arr is None:
        progress_arr = np.zeros(3)
    stage_progress(progress_arr, 0, 0.0)     # 阶段 0：装配/JIT 装载
    t = np.asarray(t_grid, float)
    T = float(t[-1])
    N = len(states)
    N2 = N * N
    E = np.array([s['E_mhz'] for s in states], float)

    # t62 耗散（落点=丢失）：GammaA/B（/ns）→ gamma_MHz（N²）传给内核
    GammaA = GammaB = None
    gamma = None
    if decays_A or decays_B:
        GammaA, GammaB, has = _tensor._pair_loss_vectors(
            states, decays_A, decays_B, N, N2)
        if has:
            gamma = (GammaA + GammaB) / _CONV
            log(f'[mixed_pair] 耗散 {int((gamma > 0).sum())} 两体态'
                f'（落点=丢失，NonHermitian）')

    # ---- t69 最终形态：逐波拆分（每波形独立 A/ω/env——废除合并模型）----
    mm = float(mw_max) if mw_max is not None else MW_MAX_MHZ
    optA, mwA = _split_waves(fields_A, mm)
    optB, mwB = _split_waves(fields_B, mm)

    # 光场载频一致性（共享光子块要求所有光波同频；异频 → 各自光子块未实现）
    om_o = None
    for _A, om, _env in (optA + optB):
        if om_o is None:
            om_o = om
        elif abs(om_o - om) > 1e-6:
            raise NotImplementedError(
                f'光场载频不同（{om_o:.0f} vs {om:.0f} MHz）——共享光子块'
                '要求同频；异频多光子块为后续扩展')
    if om_o is None:
        raise ValueError('两体混合引擎需要光场（载频 > %.0e MHz）' % mm)

    # ---- V_dd（同扇区当场；预计算/传入）----
    from . import pair_core as _pc
    if vdd_matrix is not None:
        V = vdd_matrix
    elif R_um is not None:
        if cache is None:
            raise ValueError('两体混合需要 cache 算 V_dd 或显式 vdd_matrix')
        V = _pc.vdd_sparse(cache, [dict(s) for s in states],
                           R_um, theta_deg, phi_deg)
    else:
        V = None
    if V is not None:
        import scipy.sparse as sp
        if sp.issparse(V):
            V = V.tocsr()
        else:
            V = sp.csr_matrix(np.asarray(V, complex))
    V_sp = V

    # ---- AC 化逐波（每波独立剔除远强对；位移不注入——t70 定案，见
    # _ac_ize_pair docstring。t73b：shifts 全 0 → 时变 AC 对角路径移除）----
    # t75：r_crit 可由上帝模式覆写
    rc = float(r_crit) if r_crit is not None else R_CRIT
    optA_ac = [(A_new, env) for (A, om, env) in optA
               if (A_new := _ac_ize_pair(E, A, om, rc)) is not None]
    optB_ac = [(A_new, env) for (A, om, env) in optB
               if (A_new := _ac_ize_pair(E, A, om, rc)) is not None]
    if optA_ac or optB_ac:
        log(f'[mixed_pair] AC 化：剔除远强对'
            f'（{len(optA_ac)}+{len(optB_ac)} 光波逐波，位移不注入）')

    # ---- 光子块结构（K0 纯 E——t62f 语义；共享扇区 M=1）----
    stage_progress(progress_arr, 0, 0.2)     # AC 化完成 → 光子块几何
    bw = float(bw_opt) if bw_opt is not None else BW_OPT
    K0, pairs_opt, P, dim = _build_pair_geometry(
        E, optA_ac, optB_ac, om_o, bw_opt=bw)
    # 耗散 gamma（N²）→ 扩展空间（N²·P，每扇区同衰减）
    if gamma is not None:
        gamma = np.tile(gamma, P)
    log(f'[mixed_pair] 光场 P={P} dim={N2}×{P}={dim}'
        f'（5 倍降维参考：全 Sambe N²·15）')

    # ---- pairs 打包（7 元组 a,b,v,dD,is_mw,w,env）+ 分项标签（t75 诊断）----
    mw_nres = (float(mw_nres_bw) if mw_nres_bw is not None
               else MW_NRES_BW_MHZ)
    pairs = []
    pair_tags = []

    def _one(tv):
        return np.ones_like(np.atleast_1d(tv))

    # V_dd：同扇区保留（env=恒 1——修复原型误乘光包络）；
    # **ΔD 过滤（船长 2026-09-06 性能根因）**：vdd_sparse 含跨尺度对
    # （60S/60P↔6P，|ΔD|≈591THz——4812 个）——若全打包进 φ-帧 RHS，
    # DOP853 步长被钉死 ~1e-6ns → 620ns 需 1e9+ 步（半小时+ vs 51s）。
    # 物理上 THz-dE 的 V_dd 是极离共振偶极耦合（净效应 V²/Δ 可忽略，
    # hard_pair 的 vdd_rot_filter 同款剔除——t34 存活元素 |dE|≤34.7GHz）。
    if V_sp is not None:
        coo = V_sp.tocoo()
        for a0, b0, vv in zip(coo.row, coo.col, coo.data):
            if a0 >= b0 or abs(vv) < 1e-12:
                continue
            for pn in range(P):
                a = a0 + pn * N2
                b = b0 + pn * N2
                dD = K0[a] - K0[b]
                if abs(dD) >= bw:
                    continue
                pairs.append((int(a), int(b), complex(vv), dD, 0, 0.0,
                              _one))
                pair_tags.append('V_dd')

    # 微波当场逐波：A 侧 (i,k) 同 B 态 j；B 侧 (j,l) 同 A 态 i（修复残留 j bug）
    # t69：每波形独立 (A_mw, om_m, env_m)——不合并（交叉项不可能出现）
    for side, mw_list in (('A', mwA), ('B', mwB)):
        for A_mw, om_m, env_m in mw_list:
            if A_mw is None or not np.any(np.abs(A_mw) > 1e-9):
                continue
            if env_m is None:
                env_m = _one
            for ia in range(N):
                for ka in range(N):
                    if ia >= ka or abs(A_mw[ia, ka]) < 1e-9:
                        continue
                    for jb in range(N):
                        if side == 'A':
                            # t62f：dE 用纯 E（C 点——位移由 RHS 时变对角承担；
                            # 位移不注入（t70）——配对共振不被静态位移污染）
                            dE = (E[ia] + E[jb]) - (E[ka] + E[jb])
                            a0 = ia * N + jb
                            b0 = ka * N + jb
                        else:
                            dE = (E[jb] + E[ia]) - (E[jb] + E[ka])
                            a0 = jb * N + ia
                            b0 = jb * N + ka
                        if abs(abs(dE) - om_m) > mw_nres:
                            continue
                        for pn in range(P):
                            pairs.append((a0 + pn * N2, b0 + pn * N2,
                                          complex(A_mw[ia, ka]), dE, 1, om_m,
                                          env_m))
                            pair_tags.append('微波')

    # 光场近对逐波并入（pairs_opt 为 5 元组 (a,b,v,dD,env)——各波独立 env）
    for (a, b, v, dD, env) in pairs_opt:
        if env is None:
            env = _one
        pairs.append((int(a), int(b), v, dD, 0, 0.0, env))
        pair_tags.append('光场')

    # ---- 初态嵌入（中心扇区 n=0）+ 积分 ----
    psi0_A_vec = _as_vector(states, psi0_A)
    psi0_B_vec = _as_vector(states, psi0_B)
    psi0_2d = _tensor._as_product_psi(psi0_A_vec, psi0_B_vec, N)
    X0 = np.zeros(dim, complex)
    pc = 1
    X0[pc * N2: (pc + 1) * N2] = psi0_2d

    from .coupling import pack_rhs
    pack = pack_rhs(pairs, T)
    Cv = _CONV
    stat = np.zeros(6) if diag else np.zeros(4)
    dt_hist = (np.zeros((100000, 2)) if diag else None)
    stage_progress(progress_arr, 0, 0.9)     # 打包/env 表完成（JIT 在驱动首调）
    ac_base = None          # t73b：位移不注入（t70 定案）→ AC 对角恒关；
    ac_env_tab = None       # 内核 AC 分支保留供单原子 mixed 使用
    ac_env_dt = 1.0

    def rhs(tv, phi):
        return rhs_phi_pairs(tv, phi, pack['iis'], pack['jjs'],
                               pack['vals'], pack['dDs'], pack['is_mw'],
                               pack['wcos'], pack['n_pairs_w'],
                               pack['env_tab'], pack['env_dt'], Cv, gamma,
                               ac_base, ac_env_tab, ac_env_dt)

    pops = np.zeros((len(t), N2))
    pops[0] = np.abs(psi0_2d) ** 2
    ns = np.array([-1, 0, 1])
    # t61：scipy solve_ivp 逐区间 → dop853 核心层单次贯穿（numba 全编译驱动）
    # 阶段 1（积分）由驱动在首个接受步置位——调用内的 numba 装载/编译期
    # 保持阶段 0（"准备"标签 + 墙钟耗时），GUI 显示不失真
    if driver == 'numba':
        y_out = dop853_driver_mixed(
            X0, t, T, float(tol), 1e-11, max_step,
            progress_arr, stat, dt_hist, pack['iis'], pack['jjs'],
            pack['vals'], pack['dDs'], pack['is_mw'], pack['wcos'],
            pack['n_pairs_w'], pack['env_tab'], pack['env_dt'], Cv, gamma,
            ac_base, ac_env_tab, ac_env_dt)
        stage_progress(progress_arr, 2, 0.0)  # 阶段 2：收尾（跨扇区投影）
        from .sambe import sector_sum_pops
        for i in range(1, len(t)):
            # t73b：与单原子同一重构（= 旧 Xf/ns 相位式——e^{-iC·E·t} 全局
            # 相位后光子相位恰对消，布居相同）
            pops[i] = sector_sum_pops(y_out[i], P, N2)
            stage_progress(progress_arr, 2, i / max(len(t) - 1, 1))
        n_dop = int(stat[0]) if len(stat) > 0 else 0
    else:
        from scipy.integrate import solve_ivp
        X = X0.copy()
        tprev = 0.0
        n_dop = 0
        stage_progress(progress_arr, 1, 0.0)  # 阶段 1：积分（scipy 逐区间）
        for i in range(1, len(t)):
            if progress_arr[1] != 0.0:
                raise RuntimeError('用户取消')
            tb = float(t[i])
            sol = solve_ivp(rhs, (tprev, tb), X, method='DOP853', rtol=tol,
                            atol=1e-11, max_step=max_step)
            if not sol.success:
                raise RuntimeError(f'两体混合积分失败: {sol.message}')
            n_dop += int(sol.nfev)
            X = sol.y[:, -1]
            Xf = np.exp(-1j * Cv * K0 * tb) * X
            psi = np.zeros(N2, complex)
            for pn in range(P):
                psi += np.exp(-1j * Cv * ns[pn] * om_o * tb) \
                    * Xf[pn * N2: (pn + 1) * N2]
            pops[i] = np.abs(psi) ** 2
            tprev = tb
            progress_arr[0] = min(tprev / T, 0.999)
    stage_progress(progress_arr, 2, 1.0)

    # t62 耗散：loss_A/loss_B 曲线 + 含 loss 范数（同 sambe.evolve_sambe_pair）
    loss_A = np.zeros(len(t))
    loss_B = np.zeros(len(t))
    norm_dev = float(np.max(np.abs(pops.sum(axis=1) - 1.0)))
    if GammaA is not None:
        dt_arr = np.diff(t)
        rate_A = pops @ (GammaA * 1.0)          # (n_t,)，GammaA 已 /ns
        rate_B = pops @ GammaB
        loss_A[1:] = np.cumsum(0.5 * (rate_A[1:] + rate_A[:-1]) * dt_arr)
        loss_B[1:] = np.cumsum(0.5 * (rate_B[1:] + rate_B[:-1]) * dt_arr)
        norm_dev = float(np.max(np.abs(
            pops.sum(axis=1) + loss_A + loss_B - 1.0)))
    pops_A = _tensor._reduce_A(pops, N)
    pops_B = _tensor._reduce_B(pops, N)
    E_pair = np.array([E[i] + E[j]
                       for i in range(N) for j in range(N)])
    # ---- t75 诊断快照（两体；diag=True 才采集）----
    if diag:
        diag_pairs = dict(
            a=np.array([p[0] for p in pairs], int),
            b=np.array([p[1] for p in pairs], int),
            v=np.array([p[2] for p in pairs], complex),
            dD=np.array([p[3] for p in pairs], float),
            is_mw=np.array([p[4] for p in pairs], int),
            w=np.array([p[5] for p in pairs], float))
        acize_fields = []
        for (A0, om0, _env0) in (optA + optB):
            A1 = _ac_ize_pair(E, A0, om0, rc)
            n0 = int((np.abs(np.asarray(A0, float)) > 1e-9).sum() // 2)
            n1 = (int((np.abs(np.asarray(A1, float)) > 1e-9).sum() // 2)
                  if A1 is not None else 0)
            acize_fields.append(dict(
                omega=float(om0), n_before=n0, n_kept=n1, n_acized=n0 - n1,
                shift_min=0.0, shift_max=0.0))
        idx = np.linspace(0, len(t) - 1, min(len(t), 240)).astype(int)
        sector_amps = {p: np.abs(y_out[idx][:, p * N2:(p + 1) * N2])
                       for p in range(P)}
        ts_snap = sorted(set(float(x) for x in np.linspace(0.0, T, 5)))
        snaps = {t0: _assemble_H_snapshot(K0, pairs, t0, Cv, gamma=gamma,
                                          tags=pair_tags)
                 for t0 in ts_snap}
        diag_dt = (dt_hist[:int(stat[4])] if dt_hist is not None
                   else np.zeros((0, 2)))
        out_diag = dict(
            meta=dict(method='mixed', backend='mixed-phi-dop853-pair', N=N,
                      P=P, dim=dim, t_end=T, M=[1],
                      r_crit=float(rc), n_pairs=len(pairs), tol=float(tol),
                      max_step=float(max_step), bw=bw, mw_nres=mw_nres,
                      mw_max=float(mm),
                      labels=[f"({states[i]['label']})×({states[j]['label']})"
                              for i in range(N) for j in range(N)],
                      stride=N2),
            photon=dict(Ms=[1], P=P, dim=dim, strides=[1],
                        nvecs=np.arange(P), D_full=np.asarray(K0, float).copy()),
            acize=dict(r_crit=float(rc), fields=acize_fields,
                       shift=np.zeros(N2)),
            pairs=diag_pairs,
            solver=dict(t_steps=diag_dt[:, 0].copy(), dt=diag_dt[:, 1].copy(),
                        nfev=int(stat[0]), n_acc=int(stat[1]),
                        n_rej=int(stat[2]), norm_dev=norm_dev),
            trajectory=dict(t=t[idx].copy(), sector_amps=sector_amps),
            H_snapshots=snaps,
        )
    report = dict(nfev=n_dop, max_norm_dev=norm_dev,
                  P=P, dim=dim, photon_blocks={f'{om_o:.6g}MHz': 3},
                  ac_shifts=None)
    out = dict(
        t=t, pops_2d=pops, pops_A=pops_A, pops_B=pops_B,
        loss_A=loss_A, loss_B=loss_B,
        states_2d=[{
            'label': f"({states[i]['label']}) x ({states[j]['label']})",
            'E_mhz': E_pair[i * N + j],
            'ket_i': i, 'ket_j': j,
        } for i in range(N) for j in range(N)],
        method='mixed', backend='mixed-phi-dop853-pair',
        error_report=report,
        diag=out_diag if diag else None,
    )
    if record_amp:
        from .sambe import sector_sum_amp
        out['psi_amp'] = sector_sum_amp(X, P, N2)
    log(f'[mixed_pair] 完成：norm {norm_dev:.2e}，DOP853 求值 {n_dop}')
    return out
