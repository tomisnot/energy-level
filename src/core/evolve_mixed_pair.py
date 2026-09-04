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
from numba import njit

from . import evolve as _evolve
from . import tensor as _tensor

_CONV = 2 * np.pi * 1e-3
BW_OPT = 2e4                  # 光场反旋过滤带宽（20 GHz）
R_CRIT = 0.0135               # t58: M_max=2 的 r 判据
MW_BW = 5e3                   # 微波近共振带宽


@njit(cache=True)
def _rhs_mixed_pair(t, phi, iis, jjs, vals, dDs, is_mw, wcos, n_pairs_w,
                    env_tab, env_dt, Cv, gamma=None,
                    ac_base=None, ac_env_tab=None, ac_env_dt=1.0):
    """φ-帧两体 RHS：光场近对 env·Kc∘e^{iC·ΔD·t} + 微波当场 env·cos·A
    + V_dd 当场（is_mw=0 且 wcos=0 → 无 cos 载波，env=1）。
    gamma（N²，MHz，可 None）：t62 耗散 NonHermitian 对角（落点=丢失，
    out[k] −= 0.5·C·gamma[k]·phi[k]——与 hard_pair 同语义）。"""
    tot = np.zeros_like(phi)
    ip = 0
    nw = len(wcos)
    for w in range(nw):
        np_w = n_pairs_w[w]
        if np_w == 0:
            continue
        ei = t / env_dt
        i0 = int(ei)
        if i0 >= env_tab.shape[1] - 1:
            i0 = env_tab.shape[1] - 2
        f = ei - i0
        envv = env_tab[w, i0] * (1.0 - f) + env_tab[w, i0 + 1] * f
        if envv == 0.0:
            ip += np_w
            continue
        if is_mw[w]:
            cw = envv * np.cos(Cv * wcos[w] * t)
        else:
            cw = envv
        for k in range(np_w):
            a = iis[ip + k]
            b = jjs[ip + k]
            v = vals[ip + k]
            ph = np.exp(1j * Cv * dDs[ip + k] * t)
            tot[a] += cw * v * ph * phi[b]
            tot[b] += cw * np.conj(v * ph) * phi[a]
        ip += np_w
    res = -1j * Cv * tot
    # t69 逐波时变 AC 对角 −iC·Σ_w shifts_w·env_w(t)²∘φ（env 平方 = AC∝光强——
    # K0 已是纯 E（设计 C 点），无静态位移抵消 → 修正用 env² 全量（非 env²−1——
    # 单原子 _rhs_phi 同款；env²−1 只在 K0 含静态 E+Σ 时正确，船长 t62f 诊断）。
    # ac_base: (n_ac, dim)，ac_env_tab: (n_ac, n_tab)——每光波独立门控。
    if ac_base is not None and ac_env_tab is not None:
        n_ac = ac_base.shape[0]
        for wi in range(n_ac):
            ei = t / ac_env_dt
            i0 = int(ei)
            if i0 >= ac_env_tab.shape[1] - 1:
                i0 = ac_env_tab.shape[1] - 2
            f = ei - i0
            envv = ac_env_tab[wi, i0] * (1.0 - f) \
                + ac_env_tab[wi, i0 + 1] * f
            dev = envv * envv
            if dev != 0.0:
                for k in range(phi.shape[0]):
                    v = ac_base[wi, k]
                    if v != 0.0:
                        res[k] += -1j * Cv * dev * v * phi[k]
    if gamma is not None:
        # 耗散对角项直接加在返回（不经 -1j·C——否则变虚数增长）；
        # gamma 为扩展空间长度（N²·P，调用侧已 tile）
        for k in range(phi.shape[0]):
            if gamma[k] != 0.0:
                res[k] -= 0.5 * Cv * gamma[k] * phi[k]
    return res


def _pack_rhs(pairs, t_end, n_tab=20001):
    """t63 统一：公共层 coupling.pack_rhs（单/两体共用）。"""
    from .coupling import pack_rhs as _pk
    return _pk(pairs, t_end, n_tab)


def _ac_ize_pair(E, A_o, omega_opt, r_crit=R_CRIT):
    """t63/t69 统一：公共层 acize.acize_physical——**单波** AC 化（每波形独立）。

    返回 (A_new, shifts)：A_new = 剔除远强对后的光场 A（近对保留）；
    shifts = N 维 far_sos 位移（env² 全量时变语义 + K0 纯 E 唯一实现）。
    """
    from .acize import acize_physical
    if A_o is None or not np.any(np.abs(A_o) > 1e-9):
        return (np.zeros_like(A_o, float) if A_o is not None else None,
                np.zeros(len(E)))
    A_new, sh_l = acize_physical(
        E, [dict(kind='exact', A=np.asarray(A_o, float),
                 omega_L_mhz=float(omega_opt))], r_crit)
    return A_new[0], sh_l[0]


def _build_pair_geometry(E, optA_ac, optB_ac, omega_opt):
    """两体光子块结构（t69 逐波最终形态）：K0 共享 + **每波独立 Kcw 提取**。

    optA_ac / optB_ac：A/B 侧 AC 化后光波列表，元素 (A_new, env)——
    每波独立 lift_side_A（A 侧 A⊗I / B 侧 I⊗A）→ _SambeSetup → Kcw 全矩阵
    提取（含 (a>b) 反向对——镜像 t67 修复后基线；|ΔD|<BW_OPT 反旋过滤），
    pairs 携带各自 env（交叉项不可能出现——t69 根因修复）。

    K0_pair[i,j,n] = E_i+E_j − n·ω_opt（dim N²·3，A 无关——共享光子块）。
    """
    from .tensor_fields import lift_side_A
    from .evolve_sambe import _SambeSetup, _group_by_freq
    N = len(E)
    N2 = N * N
    P = 3
    dim = N2 * P

    def _one(t):
        return np.ones_like(np.atleast_1d(t), float)

    E_pair = np.array([E[i] + E[j] for i in range(N) for j in range(N)])
    # K0 共享：A 无关（E_pair − n·ω_opt）——任取一 setup 的 K0
    setup_ref = _SambeSetup(
        E_pair, _group_by_freq([dict(kind='exact',
                                     A=np.zeros((N2, N2)),
                                     omega_L_mhz=float(omega_opt),
                                     waveform=_one)]), [1])
    K0 = np.real(np.diag(setup_ref.K0))

    pairs_opt = []
    for side, lst in (('A', optA_ac), ('B', optB_ac)):
        for A_new, env, _shifts in lst:
            if A_new is None or not np.any(np.abs(A_new) > 1e-9):
                continue
            A_lift = lift_side_A(np.asarray(A_new, float), N, side)
            setup = _SambeSetup(
                E_pair, _group_by_freq([dict(kind='exact', A=A_lift,
                                             omega_L_mhz=float(omega_opt),
                                             waveform=_one)]), [1])
            for Kcw in (setup.Kc_dense[0] or []) if setup.Kc_dense else []:
                if Kcw is None:
                    continue
                Kcw = np.asarray(Kcw, complex)
                ii, jj = np.nonzero(np.abs(Kcw) > 1e-12)
                for a, b in zip(ii, jj):
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
    if progress_arr is None:
        progress_arr = np.zeros(2)
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

    # ---- AC 化逐波（每波独立剔除远强对 + far_sos 位移；env² 全量门控）----
    # optA_ac/optB_ac 元素：(A_new, env, shifts)——shifts 为原始 A 的位移
    # （A_new 已剔除远对——far_sos 为 0——位移必须在剔除前算，此处保留）
    optA_ac = []
    optB_ac = []
    shA = np.zeros(N)
    shB = np.zeros(N)
    for A, om, env in optA:
        A_new, shifts = _ac_ize_pair(E, A, om)
        optA_ac.append((A_new, env, shifts))
        shA = shA + shifts
    for A, om, env in optB:
        A_new, shifts = _ac_ize_pair(E, A, om)
        optB_ac.append((A_new, env, shifts))
        shB = shB + shifts
    if np.any(shA != 0) or np.any(shB != 0):
        log(f'[mixed_pair] AC 化：δA {shA.min():+.2f}..{shA.max():+.2f}'
            f' δB {shB.min():+.2f}..{shB.max():+.2f} MHz'
            f'（{len(optA_ac)}+{len(optB_ac)} 光波逐波）')

    # ---- 光子块结构（K0 纯 E——t62f 语义；共享扇区 M=1）----
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
                            # 含 shA/shB 会在光关时仍污染配对共振）
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
    # t69 逐波 AC 对角：ac_base (n_ac, dim) + ac_env_tab (n_ac, n_tab)——
    # **每光波独立**（shifts_f × env_f(t)²；交叉项不可能出现）。A 侧波位移
    # 作用于 i 侧、B 侧波作用于 j 侧（两体 |i,j⟩ 各取各的）。
    ac_base = None
    ac_env_tab = None
    ac_env_dt = 1.0
    ac_waves = ([(shifts, env_f, 'A') for (_A, env_f, shifts) in optA_ac]
                + [(shifts, env_f, 'B') for (_A, env_f, shifts) in optB_ac])
    ac_waves = [w for w in ac_waves if w[0] is not None
                and np.any(np.abs(w[0]) > 0.0)]
    if ac_waves:
        n_ac = len(ac_waves)
        n_ac_tab = 4001
        ac_base = np.zeros((n_ac, dim), float)
        ac_env_tab = np.zeros((n_ac, n_ac_tab), float)
        ts_ac = np.linspace(0.0, T, n_ac_tab)
        for wi, (sh_f, env_f, side) in enumerate(ac_waves):
            for i in range(N):
                for j in range(N):
                    v = sh_f[i] if side == 'A' else sh_f[j]
                    if v != 0.0:
                        for pn in range(P):
                            ac_base[wi, (i * N + j) + pn * N2] = v
            if env_f is None:
                env_f = _one
            ac_env_tab[wi] = np.asarray(env_f(ts_ac), float)
        ac_env_dt = T / (n_ac_tab - 1)

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
    if driver == 'numba':
        from .dop853 import _dop853_driver
        y_out = _dop853_driver(
            _rhs_mixed_pair, X0, t, T, float(tol), 1e-11, max_step,
            progress_arr, stat, pack['iis'], pack['jjs'], pack['vals'],
            pack['dDs'], pack['is_mw'], pack['wcos'], pack['n_pairs_w'],
            pack['env_tab'], pack['env_dt'], Cv, gamma, ac_base,
            ac_env_tab, ac_env_dt)
        for i in range(1, len(t)):
            tb = float(t[i])
            X = y_out[i]
            Xf = np.exp(-1j * Cv * K0 * tb) * X
            psi = np.zeros(N2, complex)
            for pn in range(P):
                psi += np.exp(-1j * Cv * ns[pn] * om_o * tb) \
                    * Xf[pn * N2: (pn + 1) * N2]
            pops[i] = np.abs(psi) ** 2
        n_dop = int(stat[0]) if len(stat) > 0 else 0
    else:
        X = X0.copy()
        tprev = 0.0
        n_dop = 0
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
    if len(progress_arr) > 0:
        progress_arr[0] = 1.0

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
    E_pair = np.array([(E[i] + shA[i]) + (E[j] + shB[j])
                       for i in range(N) for j in range(N)])
    report = dict(nfev=n_dop, max_norm_dev=norm_dev,
                  P=P, dim=dim, photon_blocks={f'{om_o:.6g}MHz': 3},
                  n_ac_pairs=int((np.abs(shA) > 1e-9).sum()
                                 + (np.abs(shB) > 1e-9).sum()))
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
        Xf = np.exp(-1j * Cv * K0 * T) * X
        psi = np.zeros(N2, complex)
        for pn in range(P):
            psi += np.exp(-1j * Cv * ns[pn] * om_o * T) \
                * Xf[pn * N2: (pn + 1) * N2]
        out['psi_amp'] = psi
    log(f'[mixed_pair] 完成：norm {norm_dev:.2e}，DOP853 求值 {n_dop}')
    return out
