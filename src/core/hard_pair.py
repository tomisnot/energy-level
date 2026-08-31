# -*- coding: utf-8 -*-
"""两体 hard 引擎（numba 三件套第 3 件）：k=i·N+j 张量 + V_dd 相位扭曲 + njit DOP853。

物理模型与 evolve_exact_pair 完全一致（无 secular/RWA/旋转系近似）：
    H_2d(t) = [H_A0 + Σ_f Ω_f·env_f(t)·cos(ω_f t)·A_f] ⊗ I
            + I ⊗ [H_B0 + Σ_f Ω_f·env_f(t)·cos(ω_f t)·A_f] + V_dd
实现（hard 身份：DOP853 硬积分，零微波近似，禁止解析化载波）：
  - 光场：每原子旋转系吸收载波（build_two_field 的 E_rot/opt_terms，
    光学 RWA 同单原子 hard——反旋项 Bloch–Siegert ≪ tol 已证）；
  - 微波：cos(C·ω·t) 载波保留（两原子各自 ω），IP 中呈现 C·(ω±ΔE) 振荡；
  - 相互作用绘景 u = exp(iC·E_rot_pair·t)（u 分解为 uA[i]·uB[j]）：
    场项贡献 = −iC·u∘(Ṽ·(conj(u)∘y))；
  - V_dd：实验室系静态 → IP 中相位扭曲 V[k,l]·e^{iC(E_lab[k]−E_lab[l])t}
    （推导：旋转系 Ω-twist 与 IP u-twist 合并 = E_lab 差），按相异 dE 分组
    每组一个 exp。|V| < 1e-4·max 的元素（Forster 型 |60S,60P>↔|6P,6P> 与
    THz 型，实测 5184 中 5040 个、|V| ≤ 1e-4 MHz）省略——离共振净效应
    ≤ V²/Δ ≤ 1e-8 MHz，集体位移 ≤ ~1e-7 MHz ≪ rtol；144 个物理活性元素
    全保留（实测 |dE|·C ≤ 218 rad/ns，与微波载波同尺度，相位精确）。
    性能：每元素逐个 exp 是 RHS 主开销（5184 MACs ≈ 59µs/RHS），过滤后
    V_dd ≈ 9µs，整体 RHS 67→25µs → 1500ns 两体 595s → 81s。

2026-08-30 harddev 首版。
"""
import numpy as np
from numba import njit

from . import evolve as _evolve
from . import pair_evolve as _pe
from .evolve_exact import _CONV
# 核心层（dop853：first-class RHS 驱动 + 包络查表工具；无循环依赖——
# dop853 → hard_numba 系数，hard_pair → dop853）
from .dop853 import _dop853_driver, _env_at, _env_tab, _TAB_N


# ============================================================
# 打包：两体场项（A⊗I / I⊗A 三元组）+ V_dd（按 dE 分组）
# ============================================================
# V_dd 保留阈值（相对 max|V|）：|V| < 此比例的元素（Forster 型 |60S,60P>↔|6P,6P>
# 与 THz 型 60S↔6P/60P↔6P，实测占 5184 中 5040 个、|V| ≤ 1e-4 MHz）其离共振净
# 效应 ≤ V²/Δ ≤ 1e-8 MHz，集体位移 ≤ ~1e-7 MHz ≪ rtol——硬算表示下省略
# （每元素逐个 exp 相位是 RHS 主开销：5184 MACs ≈ 59µs/RHS）。144 个物理活性
# 元素全保留（|dE|·C ≤ 218 rad/ns，精确相位）。
_VDD_MIN_FRAC = 1e-4


def _embed_triplets(A, side, N, mask=None):
    """单原子 A（N×N）→ 两体嵌入三元组 (rows, cols, vals[, signs])。

    side='A'：A⊗I（k=(i,j)→i·N+j，仅 i 变化）；side='B'：I⊗A（仅 j 变化）。
    返回按 row 排序（写顺序化，numba 标量循环更快）。
    mask 提供时（光场拍频）：signs[每元素] = mask[单原子行]−mask[单原子列]
    ∈{±1,0}——旋转系内拍频相位 exp(iC·δ·t·sign) 的符号（t10：exp 非 cos）。
    """
    ii, jj = np.nonzero(np.abs(np.asarray(A, float)) > 1e-15)
    rows, cols, vals = [], [], []
    has_mask = mask is not None
    signs = [] if has_mask else None
    if side == 'A':
        for a, b in zip(ii, jj):
            s = mask[a] - mask[b] if has_mask else 0
            for j in range(N):
                rows.append(a * N + j)
                cols.append(b * N + j)
                vals.append(float(A[a, b]))
                if has_mask:
                    signs.append(s)
    else:
        for a, b in zip(ii, jj):
            s = mask[a] - mask[b] if has_mask else 0
            for i in range(N):
                rows.append(i * N + a)
                cols.append(i * N + b)
                vals.append(float(A[a, b]))
                if has_mask:
                    signs.append(s)
    rows = np.asarray(rows, np.int64)
    cols = np.asarray(cols, np.int64)
    vals = np.asarray(vals, float)
    if rows.size > 1:
        o = np.argsort(rows, kind='stable')
        rows = rows[o]
        cols = cols[o]
        vals = vals[o]
        if has_mask:
            signs = np.asarray(signs, np.int8)[o]
    if has_mask:
        return rows, cols, vals, np.asarray(signs, np.int8)
    return rows, cols, vals


def _pack_pair_hard(fields_meta_A, fields_meta_B, states, T, vdd_sp, n_tab):
    """→ (E_A, E_B, fld_rows, fld_cols, fld_vals, fld_envs, fld_phase_tabs,
         fld_ptr, fld_deltas, fld_sgns,
         vdd_rows, vdd_cols, vdd_vals, vdd_grp_start, vdd_grp_dE, t0, dt, CONV)
    E_A/E_B：每原子 IP 能量（各自减 min）；fld_*：全部场项三元组
    （光场相位表=0 → cos(0)=1，微波相位表=C·ω·t 或 phase_fn——t18 统一
    cos(φ_f(t)) 任意相位）；fld_deltas：光场拍频 δ=ω_f−ω_ref（微波/首场=0）；
    fld_sgns：每三元组相位符号 m[i]−m[k]（δ=0 项为 0）；
    vdd_*：V_dd 按 dE 排序分组。
    """
    N = len(states)
    E_lab = np.array([s['E_mhz'] for s in states], float)
    E_A = np.asarray(fields_meta_A['E_rot'], float)
    E_B = np.asarray(fields_meta_B['E_rot'], float)
    E_A = E_A - float(E_A.min())
    E_B = E_B - float(E_B.min())
    E_lab_pair = np.array([E_lab[i] + E_lab[j]
                           for i in range(N) for j in range(N)])

    # ---- 场项 ----
    fld_rows_l, fld_cols_l, fld_vals_l, fld_env_l = [], [], [], []
    fld_phase_l, fld_delta_l, fld_sgn_l = [], [], []
    for side, meta in (('A', fields_meta_A), ('B', fields_meta_B)):
        for tup in meta['opt_terms']:
            A_opt, wf = tup[0], tup[1]
            omega_f = float(tup[2])
            omega_ref = float(tup[4]) if len(tup) > 4 else omega_f
            mask = tup[5] if len(tup) > 5 else None
            delta = omega_f - omega_ref
            if delta != 0.0 and mask is not None:
                r, c, v, sg = _embed_triplets(A_opt, side, N, mask)
                fld_sgn_l.append(sg)
            else:
                r, c, v = _embed_triplets(A_opt, side, N)
                fld_sgn_l.append(np.zeros(r.size, np.int8))
            fld_rows_l.append(r)
            fld_cols_l.append(c)
            fld_vals_l.append(v)
            fld_env_l.append(_env_tab(wf, T, n_tab))
            fld_phase_l.append(np.zeros(n_tab))        # 光场：cos(0)=1
            fld_delta_l.append(delta)
        for g in meta['mw_groups']:
            w = float(g['omega_mhz'])
            ph_fn = g.get('phase_fn')
            for A, wf in zip(g['As'], g['envs']):
                r, c, v = _embed_triplets(A, side, N)
                fld_rows_l.append(r)
                fld_cols_l.append(c)
                fld_vals_l.append(v)
                fld_env_l.append(_env_tab(wf, T, n_tab))
                if ph_fn is None:
                    ts = np.linspace(0.0, float(T), n_tab)
                    fld_phase_l.append(_CONV * w * ts)   # 定频 φ=C·ω·t
                else:
                    fld_phase_l.append(_env_tab(ph_fn, T, n_tab))
                fld_delta_l.append(0.0)
                fld_sgn_l.append(np.zeros(r.size, np.int8))
    n_terms = len(fld_rows_l)
    if n_terms:
        fld_rows = np.concatenate(fld_rows_l)
        fld_cols = np.concatenate(fld_cols_l)
        fld_vals = np.concatenate(fld_vals_l)
        fld_envs = np.stack(fld_env_l)
        fld_phase_tabs = np.stack(fld_phase_l)
        fld_deltas = np.asarray(fld_delta_l, float)
        fld_sgns = np.concatenate(fld_sgn_l)
        ptr = np.zeros(n_terms + 1, np.int64)
        for k, r in enumerate(fld_rows_l):
            ptr[k + 1] = ptr[k] + r.size
    else:
        fld_rows = np.zeros(0, np.int64)
        fld_cols = np.zeros(0, np.int64)
        fld_vals = np.zeros(0, float)
        fld_envs = np.zeros((0, n_tab))
        fld_phase_tabs = np.zeros((0, n_tab))
        fld_deltas = np.zeros(0)
        fld_sgns = np.zeros(0, np.int8)
        ptr = np.zeros(1, np.int64)

    # ---- V_dd：相位扭曲 + 按 dE 分组 ----
    # 分组：dE 精确量化（δ=1e-6 MHz）——过滤后元素全为物理活性（|V|≥1e-4·max，
    # 实测 |dE|·C ≤ 218 rad/ns），相位须精确；组内按 row 排序（顺序写）。
    # 性能：过滤后 5184→144 元素 → V_dd MAC 从 ~59µs 降到 ~4µs/RHS。
    if vdd_sp is None:
        vdd_rows = np.zeros(0, np.int64)
        vdd_cols = np.zeros(0, np.int64)
        vdd_vals = np.zeros(0, complex)
        grp_start = np.zeros(1, np.int64)
        grp_dE = np.zeros(0, float)
    else:
        V = vdd_sp.tocsr().astype(complex)
        rows, cols = V.nonzero()
        vals = np.asarray(V.data, complex)
        vmax = np.max(np.abs(vals)) if vals.size else 0.0
        if vmax > 0.0:
            keep = np.abs(vals) >= _VDD_MIN_FRAC * vmax
            rows = rows[keep]
            cols = cols[keep]
            vals = vals[keep]
        if rows.size == 0:
            vdd_rows = np.zeros(0, np.int64)
            vdd_cols = np.zeros(0, np.int64)
            vdd_vals = np.zeros(0, complex)
            grp_start = np.zeros(1, np.int64)
            grp_dE = np.zeros(0, float)
        else:
            dE = E_lab_pair[rows] - E_lab_pair[cols]
            key = np.round(dE * 1e6)          # δ=1e-6 MHz
            order = np.lexsort((rows, key))   # 按 (组, row) 排序
            rows = rows[order]
            cols = cols[order]
            vals = vals[order]
            key = key[order]
            bounds = np.flatnonzero(np.diff(key) != 0.0)
            grp_start = np.concatenate([[0], bounds + 1, [rows.size]])
            grp_dE = dE[order][grp_start[:-1]]
            vdd_rows = rows
            vdd_cols = cols
            vdd_vals = vals
    t0 = 0.0
    dt = float(T) / (n_tab - 1) if n_tab > 1 else 1.0
    CONV = float(_CONV)
    return (E_A, E_B, fld_rows, fld_cols, fld_vals, fld_envs,
            fld_phase_tabs, ptr,
            fld_deltas, fld_sgns,
            vdd_rows, vdd_cols, vdd_vals, grp_start, grp_dE, t0, dt, CONV)


# ============================================================
# njit 内核（_env_at 查表来自 dop853 核心层）
# ============================================================
@njit(cache=True)
def _rhs_hard_pair(t, y, E_A, E_B, fld_rows, fld_cols, fld_vals, fld_envs,
                   fld_phase_tabs, fld_ptr, fld_deltas, fld_sgns,
                   vdd_rows, vdd_cols, vdd_vals,
                   vdd_grp_start, vdd_grp_dE, t0t, dtt, CONV):
    """两体 hard RHS：y' = −iC·[u∘(Ṽ·(conj(u)∘y)) + V_dd_twisted·y]。

    光场拍频：δ=ω_f−ω_ref ≠ 0 的项乘 exp(iC·δ·t·sgn)（sgn=每三元组
    相位符号，m[i]−m[k]；δ=0 逐位退化——t10 验证 exp 拍频正确）。
    微波（t18）：cos(φ_f(t)) 相位表插值——定频 φ=C·ω·t（光场相位表=0
    → cos(0)=1），变频/任意相位由 phase_fn 采样。
    """
    N = E_A.size
    N2 = N * N
    uA = np.exp(1j * CONV * E_A * t)
    uB = np.exp(1j * CONV * E_B * t)
    # zu[k] = conj(uA[i])·conj(uB[j])·y[k]
    zu = np.empty(N2, np.complex128)
    for i in range(N):
        ui = uA[i].conjugate()
        for j in range(N):
            zu[i * N + j] = ui * uB[j].conjugate() * y[i * N + j]
    # 场项累加器（后续 ×u_pair）与 V_dd 累加器（相位已含 u·conj(u)，
    # **不再**乘 u_pair）分开——混用会把 V_dd 项多乘一个 u[k]。
    dy_f = np.zeros(N2, np.complex128)
    dy_v = np.zeros(N2, np.complex128)
    # 场项（逐项三元组；光场相位表=0 → cos(0)=1，微波相位表=φ_f(t)）
    n_terms = fld_ptr.size - 1
    for ti in range(n_terms):
        c = _env_at(fld_envs[ti], t, t0t, dtt)
        cw = c * np.cos(_env_at(fld_phase_tabs[ti], t, t0t, dtt))
        if cw != 0.0:
            d = fld_deltas[ti]
            if d != 0.0:
                p1 = np.exp(1j * CONV * d * t)
                p2 = p1.conjugate()
                for m in range(fld_ptr[ti], fld_ptr[ti + 1]):
                    ph = p1 if fld_sgns[m] > 0 else p2
                    dy_f[fld_rows[m]] += fld_vals[m] * cw * ph * zu[fld_cols[m]]
            else:
                for m in range(fld_ptr[ti], fld_ptr[ti + 1]):
                    dy_f[fld_rows[m]] += fld_vals[m] * cw * zu[fld_cols[m]]
    # V_dd（相位扭曲按 dE 分组：每组一个 exp；相位 e^{iC·(E_lab[k]−E_lab[l])t}
    # 已含旋转系 Ω-twist 与 IP u-twist 的合并，无额外 u 因子）
    n_grp = vdd_grp_dE.size
    for g in range(n_grp):
        ph = np.exp(1j * CONV * vdd_grp_dE[g] * t)
        for m in range(vdd_grp_start[g], vdd_grp_start[g + 1]):
            dy_v[vdd_rows[m]] += vdd_vals[m] * ph * y[vdd_cols[m]]
    # out = −iC·[u_pair∘dy_f + dy_v]
    out = np.empty(N2, np.complex128)
    for i in range(N):
        ui = uA[i]
        for j in range(N):
            k = i * N + j
            out[k] = -1j * CONV * (ui * uB[j] * dy_f[k] + dy_v[k])
    return out

# ============================================================
# DOP853 驱动已上收至 src/core/dop853.py（first-class RHS 参数化，
# t11 实验验证与旧两体驱动逐位一致）。本模块只保留内核与打包。
# ============================================================



# ============================================================
# 主入口
# ============================================================
def evolve_hard_pair(states, fields_meta_A, fields_meta_B, psi0_A, psi0_B,
                     t_grid, vdd_matrix=None, R_um=None, theta_deg=0.0,
                     phi_deg=0.0, cache=None, rtol=1e-9, atol=1e-11,
                     progress_arr=None, log=None, record_amp=False,
                     n_tab=_TAB_N, decays_A=None, decays_B=None,
                     driver='numba'):
    """两体 hard（第一性原理 DOP853 硬积分，零微波近似）。

    返回契约与 evolve_exact_pair 同键：t, pops_2d, pops_A, pops_B,
    loss_A, loss_B, states_2d, method='hard', backend='hard-numba-pair',
    error_report。
    """
    from . import pair_core as _pc
    log = log or (lambda *a: None)
    t = np.asarray(t_grid, float)
    T = float(t[-1])
    N = len(states)
    N2 = N * N

    # ---- V_dd ----
    if vdd_matrix is not None:
        V = vdd_matrix
    elif R_um is not None:
        if cache is None:
            raise ValueError('两体 hard 需要 cache（core.ArcCache 实例）'
                             '来计算 V_dd；请传入 cache 或显式 vdd_matrix。')
        V = _pc.vdd_sparse(cache, [dict(s) for s in states],
                           R_um, theta_deg, phi_deg)
    else:
        V = None

    # ---- 耗散（寿命通道 → NonHermitian 对角，同 evolve_exact_pair）----
    GammaA = GammaB = None
    has_decay = bool(decays_A or decays_B)
    if has_decay:
        GammaA, GammaB, _ = _pe._pair_loss_vectors(
            states, decays_A, decays_B, N, N2)
    gamma_MHz = np.zeros(N2)
    if has_decay:
        gamma_MHz = (np.asarray(GammaA, float)
                     + np.asarray(GammaB, float)) / _CONV

    # ---- 初态 ----
    psi0_A_vec = _evolve._as_vector(states, psi0_A)
    psi0_B_vec = _evolve._as_vector(states, psi0_B)
    psi0_2d = _pe._as_product_psi(psi0_A_vec, psi0_B_vec, N)

    pack = _pack_pair_hard(fields_meta_A, fields_meta_B, states, T, V, n_tab)
    E_A, E_B = pack[0], pack[1]
    E_lab = np.array([s['E_mhz'] for s in states], float)
    E_pair = np.array([E_lab[i] + E_lab[j]
                       for i in range(N) for j in range(N)])

    if driver == 'numba':
        prog = (np.zeros(2) if progress_arr is None
                else np.asarray(progress_arr, float))
        stat = np.zeros(4)
        # 复用 dop853 核心层（first-class RHS 参数化，t11 验证逐位一致）
        y_out = _dop853_driver(_rhs_hard_pair, psi0_2d, t, T,
                               float(rtol), float(atol), np.inf,
                               prog, stat, *pack)
        nfev = int(stat[0])
        n_acc = int(stat[1])
        n_rej = int(stat[2])
        if progress_arr is not None:
            progress_arr[0] = 1.0
    else:
        # driver='scipy'：njit RHS + scipy solve_ivp（对照用）
        from scipy.integrate import solve_ivp

        def rhs(tv, y):
            return _rhs_hard_pair(tv, y, *pack)
        sol = solve_ivp(rhs, (0.0, T), psi0_2d, method='DOP853',
                        t_eval=t, rtol=rtol, atol=atol)
        if not sol.success:
            raise RuntimeError(f'hard-pair(scipy) 积分失败: {sol.message}')
        y_out = sol.y.T
        nfev = int(getattr(sol, 'nfev', 0))
        n_acc = -1

    pops_2d = np.abs(y_out) ** 2

    # ---- loss（NonHermitian 速率积分，同 pair_evolve 语义）----
    loss_A = np.zeros(len(t))
    loss_B = np.zeros(len(t))
    if has_decay:
        rate_A = pops_2d @ GammaA
        rate_B = pops_2d @ GammaB
        dt_arr = np.diff(t)
        loss_A[1:] = np.cumsum(0.5 * (rate_A[1:] + rate_A[:-1]) * dt_arr)
        loss_B[1:] = np.cumsum(0.5 * (rate_B[1:] + rate_B[:-1]) * dt_arr)
        norm_dev = float(np.max(np.abs(
            pops_2d.sum(axis=1) + loss_A + loss_B - 1.0)))
    else:
        norm_dev = float(np.max(np.abs(pops_2d.sum(axis=1) - 1.0)))

    pops_A = _pe._reduce_A(pops_2d, N)
    pops_B = _pe._reduce_B(pops_2d, N)
    report = dict(n_segments=n_acc, max_richardson_diff=0.0, dt_min_hit=False,
                  max_norm_dev=norm_dev, nfev=nfev)
    if report['max_norm_dev'] > 1e-6:
        log(f'[hard-pair] ⚠ 范数偏差 {report["max_norm_dev"]:.2e}')
    log(f'[hard-pair] DOP853 完成：{len(t)} 输出点，{nfev} 次 RHS 求值'
        f'（N²={N2} 维，硬积分零微波近似）')
    out = dict(
        t=t, pops_2d=pops_2d, pops_A=pops_A, pops_B=pops_B,
        loss_A=loss_A, loss_B=loss_B,
        states_2d=[{
            'label': f"({states[i]['label']}) x ({states[j]['label']})",
            'E_mhz': E_pair[i * N + j],
            'ket_i': i, 'ket_j': j,
        } for i in range(N) for j in range(N)],
        method='hard', backend='hard-numba-pair',
        error_report=report)
    if record_amp:
        uT = np.exp(1j * _CONV * (E_A[:, None] + E_B[None, :]).ravel() * T)
        out['psi_amp'] = np.conj(uT) * y_out[-1]
    return out
