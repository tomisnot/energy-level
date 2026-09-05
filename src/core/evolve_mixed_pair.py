# -*- coding: utf-8 -*-
"""两体混合引擎（正式版）：φ-帧 + 远对 AC 化注入物理 E + 微波/V_dd 当场。

架构（docs/混合引擎两体理论方案-用例8设计.md）：
- 两体物理基 |i,j⟩（N² 维，E_pair = E_i+E_j）
- 光场共享光子块 M=1（A/B 同频场共享 n → P=3，dim = N²·3 = 1452 @22 态）
- 远失谐强对（r>r_crit）AC 化：剔除 + far_sos 位移注入物理能级
  （A 侧场位移 δA_i 加 E_i 侧、B 侧场位移 δB_j 加 E_j 侧——支持 A/B 异场）
- 微波/V_dd 不进光子块（当场同扇区——ΔD=E_i+E_j−E_k−E_l 与 n 无关）
- φ-帧单 DOP853（RHS 全慢——反旋 |ΔD|<BW_OPT 过滤 + 微波近共振过滤）

接口对齐 evolve_sambe_pair：返回 dict(t, pops_2d, pops_A, pops_B, states_2d,
loss_A, loss_B, method='mixed', backend, error_report)。

修复（相对 tempsimu/_t62b_pair.py 原型）：
1. V_dd 项误乘光场包络 _eo → 修正为恒 1（V_dd 是原子间常数相互作用）
2. 微波 B 侧循环用残留 j 变量（只处理 j=N-1）→ 修正为遍历全部 j
3. A/B 独立场支持（原假设同构单 A_o）
"""
import numpy as np
from scipy.integrate import solve_ivp

from . import evolve as _evolve
from . import tensor as _tensor

_CONV = 2 * np.pi * 1e-3
BW_OPT = 2e4                  # 光场反旋过滤带宽（20 GHz）
R_CRIT = 0.0135               # t58: M_max=2 的 r 判据
MW_BW = 5e3                   # 微波近共振带宽


# t73：φ-帧 RHS 内核上收 coupling.rhs_phi_pairs（单/两体同一实现——
# t66 语义收敛：双方均 a<b 单侧打包 + 内核厄米补齐；AC 对角 pair 风格
# ac_base (n_ac, dim) + 独立 env 表；gamma 扩展空间长度 NonHermitian）
from .coupling import rhs_phi_pairs as _rhs_mixed_pair  # noqa: E402


def _pack_rhs(pairs, t_end, n_tab=20001):
    """t63 统一：公共层 coupling.pack_rhs（单/两体共用）。"""
    from .coupling import pack_rhs as _pk
    return _pk(pairs, t_end, n_tab)


def _ac_ize_pair(E, A_o, omega_opt, r_crit=R_CRIT):
    """AC 化（t70 定案）：剔除远强对；**位移不注入**。

    近对（如 60S↔6P）保留在光子块 M=1 动态、AC 位移自然涌现——与 hard 的
    RWA 动态同物理（M=1 对角化涌现 −18.87 ≡ 静态引擎 −16.47）。再注入全量
    静态位移 = 与动态双计（375ns 交换 0.012 vs 0.558 的根因，t70 实验定案；
    旧 inject=True 考古语义已随 t73b 深度清理移除）。
    """
    from .acize import acize_physical
    if A_o is None or not np.any(np.abs(A_o) > 1e-9):
        return None
    A_new, _ = acize_physical(
        E, [dict(kind='exact', A=np.asarray(A_o, float),
                 omega_L_mhz=float(omega_opt))], r_crit)
    return A_new[0]


def _build_pair_geometry(E, optA_ac, optB_ac, omega_opt):
    """两体光子块结构（t73b 单次装配）：全部光波（已按侧提升）入同一载频组
    → 一个 _SambeSetup 同时给出 K0（纯 E）与逐波 Kcw（t69 逐波形态：
    交叉项不可能出现；a<b 三角过滤 + |ΔD|<BW_OPT 反旋过滤）。

    optA_ac / optB_ac 元素 (A_new, env)——A 侧 lift_side_A(A⊗I)、B 侧 I⊗A。
    """
    from .tensor_fields import lift_side_A
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
            if abs(K0[a] - K0[b]) < BW_OPT:
                pairs_opt.append((int(a), int(b), Kcw[a, b],
                                  float(K0[a] - K0[b]), env))
    return K0, pairs_opt, P, dim


def _split_waves(fields):
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
        (opt if om > 1e5 else mw).append((A, om, f.get('waveform')))
    return opt, mw


def evolve_mixed_pair(states, fields_A, fields_B, psi0_A, psi0_B, t_grid,
                      vdd_matrix=None, R_um=None, theta_deg=0.0,
                      phi_deg=0.0, cache=None, tol=1e-8, max_step=0.05,
                      progress_arr=None, log=None, record_amp=False,
                      decays_A=None, decays_B=None, driver='numba'):
    """两体混合引擎（正式版）：光场光子块 + 微波/V_dd 当场，φ-帧单 DOP853。

    fields_A/B : kind='exact' 场列表（build_fields_exact 输出——光场 + 微波）
    vdd_matrix  : V_dd 稀疏/稠密矩阵（N²×N²）；None 且给 R_um/cache 时算
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
    optA, mwA = _split_waves(fields_A)
    optB, mwB = _split_waves(fields_B)

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
        raise ValueError('两体混合引擎需要光场（载频 > 1e5 MHz）')

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
    optA_ac = [(A_new, env) for (A, om, env) in optA
               if (A_new := _ac_ize_pair(E, A, om)) is not None]
    optB_ac = [(A_new, env) for (A, om, env) in optB
               if (A_new := _ac_ize_pair(E, A, om)) is not None]
    if optA_ac or optB_ac:
        log(f'[mixed_pair] AC 化：剔除远强对'
            f'（{len(optA_ac)}+{len(optB_ac)} 光波逐波，位移不注入）')

    # ---- 光子块结构（K0 纯 E——t62f 语义；共享扇区 M=1）----
    stage_progress(progress_arr, 0, 0.2)     # AC 化完成 → 光子块几何
    K0, pairs_opt, P, dim = _build_pair_geometry(
        E, optA_ac, optB_ac, om_o)
    # 耗散 gamma（N²）→ 扩展空间（N²·P，每扇区同衰减）
    if gamma is not None:
        gamma = np.tile(gamma, P)
    log(f'[mixed_pair] 光场 P={P} dim={N2}×{P}={dim}'
        f'（5 倍降维参考：全 Sambe N²·15）')

    # ---- pairs 打包（7 元组 a,b,v,dD,is_mw,w,env）----
    pairs = []

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
                if abs(dD) >= BW_OPT:
                    continue
                pairs.append((int(a), int(b), complex(vv), dD, 0, 0.0,
                              _one))

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
                        if abs(abs(dE) - om_m) > MW_BW:
                            continue
                        for pn in range(P):
                            pairs.append((a0 + pn * N2, b0 + pn * N2,
                                          complex(A_mw[ia, ka]), dE, 1, om_m,
                                          env_m))

    # 光场近对逐波并入（pairs_opt 为 5 元组 (a,b,v,dD,env)——各波独立 env）
    for (a, b, v, dD, env) in pairs_opt:
        if env is None:
            env = _one
        pairs.append((int(a), int(b), v, dD, 0, 0.0, env))

    # ---- 初态嵌入（中心扇区 n=0）+ 积分 ----
    psi0_A_vec = _evolve._as_vector(states, psi0_A)
    psi0_B_vec = _evolve._as_vector(states, psi0_B)
    psi0_2d = _tensor._as_product_psi(psi0_A_vec, psi0_B_vec, N)
    X0 = np.zeros(dim, complex)
    pc = 1
    X0[pc * N2: (pc + 1) * N2] = psi0_2d

    pack = _pack_rhs(pairs, T)
    Cv = _CONV
    stat = np.zeros(4)
    stage_progress(progress_arr, 0, 0.9)     # 打包/env 表完成（JIT 在驱动首调）
    ac_base = None          # t73b：位移不注入（t70 定案）→ AC 对角恒关；
    ac_env_tab = None       # 内核 AC 分支保留供单原子 mixed 使用
    ac_env_dt = 1.0

    def rhs(tv, phi):
        return _rhs_mixed_pair(tv, phi, pack['iis'], pack['jjs'],
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
        from .dop853 import _dop853_driver
        y_out = _dop853_driver(
            _rhs_mixed_pair, X0, t, T, float(tol), 1e-11, max_step,
            progress_arr, stat, pack['iis'], pack['jjs'], pack['vals'],
            pack['dDs'], pack['is_mw'], pack['wcos'], pack['n_pairs_w'],
            pack['env_tab'], pack['env_dt'], Cv, gamma, ac_base,
            ac_env_tab, ac_env_dt)
        stage_progress(progress_arr, 2, 0.0)  # 阶段 2：收尾（跨扇区投影）
        from .sambe import sector_sum_pops
        for i in range(1, len(t)):
            # t73b：与单原子同一重构（= 旧 Xf/ns 相位式——e^{-iC·E·t} 全局
            # 相位后光子相位恰对消，布居相同）
            pops[i] = sector_sum_pops(y_out[i], P, N2)
            stage_progress(progress_arr, 2, i / max(len(t) - 1, 1))
        n_dop = int(stat[0]) if len(stat) > 0 else 0
    else:
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

    # t62 耗散：loss_A/loss_B 曲线 + 含 loss 范数（同 evolve_sambe_pair）
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
    )
    if record_amp:
        from .sambe import sector_sum_amp
        out['psi_amp'] = sector_sum_amp(X, P, N2)
    log(f'[mixed_pair] 完成：norm {norm_dev:.2e}，DOP853 求值 {n_dop}')
    return out
