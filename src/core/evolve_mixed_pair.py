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
                    env_tab, env_dt, Cv, gamma=None):
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
    if gamma is not None:
        # 耗散对角项直接加在返回（不经 -1j·C——否则变虚数增长）；
        # gamma 为扩展空间长度（N²·P，调用侧已 tile）
        for k in range(phi.shape[0]):
            if gamma[k] != 0.0:
                res[k] -= 0.5 * Cv * gamma[k] * phi[k]
    return res


def _pack_rhs(pairs, t_end, n_tab=20001):
    """pairs → numba 数组（按 (is_mw, w, id(env)) 分组 + env 表预采样）。

    pairs 元素：(a, b, v, dD, is_mw, w, env)——分组键含 is_mw/w（t62 用例 1
    教训：光场/微波即使 env 同一对象也须分属不同组——is_mw 决定 cos 载波）。
    """
    grp = {}
    order = []
    for p in pairs:
        a, b, v, dD, is_mw, w, env = p
        key = (int(is_mw), float(w), id(env))
        if key not in grp:
            grp[key] = dict(is_mw=is_mw, w=w, env=env, items=[])
            order.append(key)
        grp[key]['items'].append((a, b, v, dD))
    iis, jjs, vals, dDs = [], [], [], []
    is_mw_a, wcos_a, n_ppw = [], [], []
    env_rows = []
    for key in order:
        g = grp[key]
        for (a, b, v, dD) in g['items']:
            iis.append(a); jjs.append(b); vals.append(v); dDs.append(dD)
        n_ppw.append(len(g['items']))
        is_mw_a.append(1 if g['is_mw'] else 0)
        wcos_a.append(g['w'])
        ts = np.linspace(0.0, t_end, n_tab)
        env_rows.append(np.asarray(g['env'](ts), float))
    env_tab = np.stack(env_rows) if env_rows else np.zeros((0, n_tab))
    env_dt = t_end / (n_tab - 1) if n_tab > 1 else 1.0
    return dict(iis=np.asarray(iis, np.int64),
                jjs=np.asarray(jjs, np.int64),
                vals=np.asarray(vals, np.complex128),
                dDs=np.asarray(dDs, np.float64),
                is_mw=np.asarray(is_mw_a, np.int64),
                wcos=np.asarray(wcos_a, np.float64),
                n_pairs_w=np.asarray(n_ppw, np.int64),
                env_tab=env_tab, env_dt=env_dt)


def _ac_ize_pair(E, A_o_A, A_o_B, omega_opt, r_crit=R_CRIT):
    """远强对 AC 化：A/B 侧分别剔除 + far_sos 位移（A→δA、B→δB）。

    返回 (A_oA2, A_oB2, E_shiftA, E_shiftB)——两体 K0_pair[i,j] 用
    (E_i+δA_i)+(E_j+δB_j)（支持 A/B 异场）。
    """
    N = len(E)
    A_a = np.asarray(A_o_A, float).copy()
    A_b = np.asarray(A_o_B, float).copy()
    shA = np.zeros(N)
    shB = np.zeros(N)

    def _ac(A2, sh, tag):
        for i in range(N):
            for k in range(i + 1, N):
                if abs(A2[i, k]) < 1e-9:
                    continue
                if abs(A2[i, k]) <= r_crit * omega_opt:
                    continue
                nu = E[k] - E[i]
                dE = -(A2[i, k] ** 2) * nu / (nu ** 2 - omega_opt ** 2) / 2.0
                sh[i] += dE
                sh[k] -= dE
                A2[i, k] = A2[k, i] = 0.0

    _ac(A_a, shA, 'A')
    _ac(A_b, shB, 'B')
    return A_a, A_b, shA, shB


def _build_pair_geometry(E, A_oA, A_oB, omega_opt, shA, shB):
    """两体光子块结构（共享光场 M=1，P=3）：K0_pair + 近对耦合对。

    K0_pair[i,j,n] = (E_i+δA_i)+(E_j+δB_j) − n·ω_opt
    近对耦合（|ΔD|<BW_OPT 过滤反旋）：A 原子 i→k（B 不动）+ B 原子 j→l。
    """
    N = len(E)
    N2 = N * N
    P = 3
    ns = np.array([-1, 0, 1])
    dim = N2 * P
    K0 = np.zeros(dim)
    for i in range(N):
        for j in range(N):
            base = (E[i] + shA[i]) + (E[j] + shB[j])
            for pn in range(P):
                K0[(i * N + j) + pn * N2] = base - ns[pn] * omega_opt
    pairs_opt = []

    def _add_opt(A2, side):
        for ia in range(N):
            for ka in range(N):
                if ia == ka or abs(A2[ia, ka]) < 1e-9:
                    continue
                v = A2[ia, ka] / 2.0
                if side == 'A':
                    for jb in range(N):
                        for pn in range(P):
                            for dn in (1, -1):
                                p2 = pn + dn
                                if not (0 <= p2 < P):
                                    continue
                                a = (ia * N + jb) + pn * N2
                                b = (ka * N + jb) + p2 * N2
                                dD = K0[a] - K0[b]
                                if abs(dD) < BW_OPT:
                                    pairs_opt.append((a, b, complex(v), dD))
                else:
                    for ib in range(N):
                        for pn in range(P):
                            for dn in (1, -1):
                                p2 = pn + dn
                                if not (0 <= p2 < P):
                                    continue
                                a = (ib * N + ia) + pn * N2
                                b = (ib * N + ka) + p2 * N2
                                dD = K0[a] - K0[b]
                                if abs(dD) < BW_OPT:
                                    pairs_opt.append((a, b, complex(v), dD))

    _add_opt(A_oA, 'A')
    _add_opt(A_oB, 'B')
    return K0, pairs_opt, P, dim


def _extract_fields(fields):
    """fields（kind='exact'）→ (A_opt, A_mw, omega_opt, omega_mw, env_opt, env_mw)。

    同频场 A 累加（物理：同频多波 = 振幅叠加）；env 取第一个（同频场共享
    包络——build_fields_exact 同频组包络一致）；无光场/无微波返回 None。
    """
    A_opt = None
    A_mw = None
    om_o = om_m = None
    env_o = env_m = None
    for f in fields or []:
        A = np.asarray(f['A'], float)
        om = float(f['omega_L_mhz'])
        if om > 1e5:
            if A_opt is None:
                A_opt = np.zeros_like(A)
                om_o = om
                env_o = f['waveform']
            A_opt = A_opt + A
        else:
            if A_mw is None:
                A_mw = np.zeros_like(A)
                om_m = om
                env_m = f['waveform']
            A_mw = A_mw + A
    return A_opt, A_mw, om_o, om_m, env_o, env_m


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

    A_oA, A_mwA, om_oA, om_mA, env_oA, env_mA = _extract_fields(fields_A)
    A_oB, A_mwB, om_oB, om_mB, env_oB, env_mB = _extract_fields(fields_B)

    # 光场载频一致性（共享光子块要求 A/B 光场同频；异频 → 各自光子块未实现）
    if A_oA is not None and A_oB is not None and abs(om_oA - om_oB) > 1e-6:
        raise NotImplementedError(
            f'A/B 光场载频不同（{om_oA:.0f} vs {om_oB:.0f} MHz）——共享光子块'
            '要求同频；异频多光子块为后续扩展')
    if A_oA is None and A_oB is None:
        raise ValueError('两体混合引擎需要光场（载频 > 1e5 MHz）')
    om_o = om_oA if A_oA is not None else om_oB
    # 包络：A/B 光场分别取（共享光子块但包络可不同——RHS 逐波带 env）
    # 简化：同频场取 A 侧包络；A/B 包络不同时逐波（此处同构场景一致）
    env_o = env_oA if env_oA is not None else env_oB
    if A_oA is None:
        A_oA = np.zeros((N, N))
    if A_oB is None:
        A_oB = np.zeros((N, N))
    if env_oA is None:
        env_oA = env_o
    if env_oB is None:
        env_oB = env_o

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

    # ---- AC 化（远强对剔除 + far_sos 位移注入物理 E）----
    A_oA2, A_oB2, shA, shB = _ac_ize_pair(E, A_oA, A_oB, om_o)
    if np.any(shA != 0) or np.any(shB != 0):
        log(f'[mixed_pair] AC 化：δA {shA.min():+.2f}..{shA.max():+.2f}'
            f' δB {shB.min():+.2f}..{shB.max():+.2f} MHz')

    # ---- 光子块结构 + 近对耦合 ----
    K0, pairs_opt, P, dim = _build_pair_geometry(
        E, A_oA2, A_oB2, om_o, shA, shB)
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

    # 微波当场：A 侧 (i,k) 同 B 态 j；B 侧 (j,l) 同 A 态 i（修复残留 j bug）
    for side, A_mw, om_m, env_m in (
            ('A', A_mwA, om_mA, env_mA), ('B', A_mwB, om_mB, env_mB)):
        if A_mw is None:
            continue
        if om_m is None:
            continue
        if env_m is None:
            env_m = _one
        for ia in range(N):
            for ka in range(N):
                if ia >= ka or abs(A_mw[ia, ka]) < 1e-9:
                    continue
                for jb in range(N):
                    if side == 'A':
                        dE = ((E[ia] + shA[ia]) + (E[jb] + shB[jb])) \
                            - ((E[ka] + shA[ka]) + (E[jb] + shB[jb]))
                        a0 = ia * N + jb
                        b0 = ka * N + jb
                    else:
                        dE = ((E[jb] + shA[jb]) + (E[ia] + shB[ia])) \
                            - ((E[jb] + shA[jb]) + (E[ka] + shB[ka]))
                        a0 = jb * N + ia
                        b0 = jb * N + ka
                    if abs(abs(dE) - om_m) > MW_BW:
                        continue
                    for pn in range(P):
                        pairs.append((a0 + pn * N2, b0 + pn * N2,
                                      complex(A_mw[ia, ka]), dE, 1, om_m,
                                      env_m))

    # 光场近对并入（env 用各自 A/B 包络——7 元组）
    # pairs_opt 是 (a,b,v,dD)——按 a 索引归属 A/B 侧包络（近似：同构场景一致）
    for (a, b, v, dD) in pairs_opt:
        pairs.append((int(a), int(b), v, dD, 0, 0.0, env_o))

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

    def rhs(tv, phi):
        return _rhs_mixed_pair(tv, phi, pack['iis'], pack['jjs'],
                               pack['vals'], pack['dDs'], pack['is_mw'],
                               pack['wcos'], pack['n_pairs_w'],
                               pack['env_tab'], pack['env_dt'], Cv, gamma)

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
            pack['env_tab'], pack['env_dt'], Cv, gamma)
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
