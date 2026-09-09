# -*- coding: utf-8 -*-
"""hard 路线加速内核（numba 三件套第 1、2 件）：evolve_hard 的 JIT RHS + 包络查表。

背景（docs/AI队友任务书-两场引擎.md §12 hard 路线，harddev 负责）：
- 目标：单原子 22 态 620ns ≤5 s（实测基线 173.2 s @ rtol=1e-8）；
  零近似不变——DOP853 硬积分是 hard 的身份，禁止解析化载波（那是变成 Sambe）。
- 经验迁移（§14.1/§15.3，2026-08-31 已归档参考）：
  ① JIT 内核化（=> numba njit）：RHS 全量编译，消除每次 RHS 求值的 Python
     解释开销（基线 560057 次 RHS / 173.2 s ≈ 309 µs/次 → 内核 ~3-5 µs/次）；
  ② 包络查表化（_pack_pair_terms 模式，pair_evolve.py:290）：均匀网格预采样
     + 线性插值。包络慢变（高斯 σ=40ns / 方波沿 20ns），620ns 上 2^17 点
     → h≈0.0047ns → 插值误差 ~3e-10 ≪ rtol=1e-8；
     —— 载波 cos **不查表**：线性插值表误差 ~h²ω²/8，要达 1e-9 需
        h ≤ sqrt(8e-9)/ω（ω=216.6 rad/ns 反旋项）≈ 8.2e-7 ns → 7.5e8 点/载波，
        不可行；内核直接 cos()（libm ~10ns/次）是唯一正确选择。
- 数学与 evolve_two_field.evolve_hard 逐项一致：
    Ṽ(t)  = Σ_opt c(t)·A_opt + Σ_mw env(t)·cos(C·ω·t)·A_mw
    y'    = −iC·u·(Ṽ·(conj(u)∘y))，u = exp(iC·(E_rot−min)·t)
  （原版 u·Ṽ·u† @ y 的结合律等价分解，浮点舍入差异 ~1e-15 级，
   对 DOP853 轨迹的影响远小于 rtol。）

2026-08-30 harddev 首版。
"""

# ============================================================
# hard 单/两体同一内核（t19 合并文件；t73b 单原子并入两体路径）。
# 单原子：evolve_hard_fast（= evolve_hard）——合成 N_B=1 旁观维走
#         _pack_pair_hard + _rhs_hard_pair（旧单体内核已删）；
# 两体：  evolve_hard_pair（k=i·N+j 张量 + V_dd 相位扭曲 + njit DOP853）。
# 驱动与 DOP853 系数在 dop853.py（t76 上收，消除 dop853 ⇄ hard 循环导入）；
# _env_at/_env_tab 单源在本模块。
# ============================================================

import numpy as np
from numba import njit
from .dop853 import _dop853_driver   # 驱动（系数已随驱动上收 dop853）

from . import tensor as _tensor
from . import static_atom as core  # t74：build_two_field 系列归位所需
from .params import (_TAB_N, _VDD_MIN_FRAC, MW_MAX_MHZ,  # t74c 参数单源
                     OPT_NRES_BW_MHZ, MW_NRES_BW_MHZ)
from .evolve_util import (_single_loss_rates, _loss_and_norm,
                          _as_vector, env_at, _CONV)  # t74 工具归位

# 包络查表点数（620ns → h≈0.0047ns；插值误差 ~3e-10）


@njit(cache=True, nogil=True)
def _env_at(tab, t, t0, dt):
    """均匀网格线性插值（端点 clamp）。"""
    x = (t - t0) / dt
    i0 = int(x)
    n = tab.shape[0]
    if i0 < 0:
        return tab[0]
    if i0 >= n - 1:
        return tab[n - 1]
    f = x - i0
    return tab[i0] * (1.0 - f) + tab[i0 + 1] * f


def _env_tab(_env_sample, T, n_pts):
    """包络在 [0,T] 均匀网格上的采样表（_env_sample 为绝对时间 callable）。"""
    ts = np.linspace(0.0, float(T), n_pts)
    return np.asarray(_env_sample(ts), float)




def _mw_terms_of(fields_meta):
    mw_terms = []
    for g in fields_meta['mw_groups']:
        for A, wf in zip(g['As'], g['envs']):
            mw_terms.append((np.asarray(A, float),
                             float(g['omega_mhz']), wf))
    return mw_terms


def evolve_hard_fast(states, fields_meta, psi0, t_grid, rtol=1e-9, atol=1e-11,
                     progress_arr=None, log=None, record_amp=False,
                     n_tab=_TAB_N, driver='numba', decays=None):
    """单原子 hard（t73b：与两体同一打包/内核/驱动——N_B=1 旁观维）。

    数学与旧单体内核逐项一致：u = exp(iC·E·t) 相互作用绘景、光场旋转系
    RWA（拍频 exp(iC·δ·t·sign)）、微波 cos(φ_f(t)) 硬积分零近似、
    NonHermitian gamma（落点=丢失）；差异仅在 RHS 求和的结合顺序
    （三元组累加 vs 全矩阵装配，~1e-16/步舍入级，DOP853 容差内）。
    driver='numba'（默认，dop853 核心层全编译主循环）| 'scipy'（对照）。
    backend='hard-numba-dop853'。
    """
    log = log or (lambda *a: None)
    from .coupling import stage_progress
    t = np.asarray(t_grid, float)
    T = float(t[-1])
    N = len(states)
    E_rot = np.asarray(fields_meta['E_rot'], float)
    E = E_rot - float(E_rot.min())
    opt_terms = fields_meta['opt_terms']
    mw_terms = _mw_terms_of(fields_meta)
    stage_progress(progress_arr, 0, 0.3)     # 阶段 0：打包/装载
    # t73b：合成 N_B=1 旁观侧（E_B=[0] → uB≡1，B 侧无场项、无 V_dd）——
    # 单原子 hard 与 evolve_hard_pair 共用 _pack_pair_hard + _rhs_hard_pair
    meta_B = dict(E_rot=np.zeros(1), opt_terms=[], mw_groups=[], N=1)
    pack = _pack_pair_hard(fields_meta, meta_B, T, None, n_tab)
    # t24：耗散（落点=丢失）——Gamma_ns（/ns）→ gamma（MHz）传给内核
    has_decay = bool(decays)
    Gamma_ns = (_single_loss_rates(states, decays) if has_decay
                else np.zeros(N))
    gamma = Gamma_ns / _CONV if has_decay else None
    psi0v = _as_vector(states, psi0)
    psi0_2d = np.kron(psi0v, np.ones(1, complex))   # 旁观振幅 1

    if driver == 'numba':
        prog = (np.zeros(3) if progress_arr is None
                else np.asarray(progress_arr, float))
        stat = np.zeros(4)
        stage_progress(prog, 0, 0.9)
        y_out = dop853_driver_hard(psi0_2d, t, T, float(rtol),
                                   float(atol), np.inf, prog, stat, None,
                                   *pack, gamma)
        pops = np.abs(y_out) ** 2
        nfev = int(stat[0])
        n_acc = int(stat[1])
        n_rej = int(stat[2])
        stage_progress(prog, 2, 0.5)         # 阶段 2：收尾（loss/统计）
        log(f'[hard-numba] DOP853 完成：{len(t)} 输出点，{nfev} 次 RHS 求值'
            f'（{n_acc} 接受步 / {n_rej} 拒绝步）'
            + (f'（微波载波 {max((abs(w) for _, w, _ in mw_terms), default=0):.4g}'
               ' MHz 硬积分，零微波近似）'))
    else:
        # ---- driver='scipy'：njit RHS + scipy solve_ivp（对照用）----
        from scipy.integrate import solve_ivp

        def _rhs_pack(tv, y):
            return _rhs_hard_pair(tv, y, *pack, gamma)
        stage_progress(progress_arr, 1, 0.0)
        sol = solve_ivp(_rhs_pack, (0.0, T), psi0_2d, method='DOP853',
                        t_eval=t, rtol=rtol, atol=atol)
        if not sol.success:
            raise RuntimeError(f'hard(numba) 积分失败: {sol.message}')
        y_out = sol.y.T
        nfev = getattr(sol, 'nfev', None)
        n_acc = -1
        n_rej = -1
        pops = np.abs(y_out) ** 2
        stage_progress(progress_arr, 2, 0.5)
        log(f'[hard-numba(scipy)] DOP853 完成：{len(t)} 输出点'
            + (f'，{nfev} 次 RHS 求值' if nfev else '')
            + (f'（微波载波 {max((abs(w) for _, w, _ in mw_terms), default=0):.4g}'
               ' MHz 硬积分，零微波近似）'))
    loss, norm_dev = _loss_and_norm(pops, Gamma_ns, t)
    stage_progress(progress_arr, 2, 1.0)
    report = dict(n_segments=n_acc, max_richardson_diff=0.0,
                  dt_min_hit=False,
                  max_norm_dev=float(norm_dev))
    if report['max_norm_dev'] > 1e-6:
        log(f'[hard-numba] ⚠ 范数偏差 {report["max_norm_dev"]:.2e}'
            '（DOP853 容差不足或 RHS 非厄米）')
    out = dict(t=t, pops=pops, states=states,
               om_mhz=[float(np.abs(A).max()) for A, _, _ in mw_terms]
               + [float(np.abs(A_opt).max()) for A_opt, *_ in opt_terms],
               method='hard', backend='hard-numba-dop853',
               loss=loss, error_report=report)
    if record_amp:
        uT = np.exp(1j * _CONV * E * T)
        out['psi_amp'] = np.conj(uT) * y_out[-1]
    return out


# ============================================================
# 打包：两体场项（A⊗I / I⊗A 三元组）+ V_dd（按 dE 分组）
# ============================================================
# V_dd 保留阈值（相对 max|V|）：|V| < 此比例的元素（Forster 型 |60S,60P>↔|6P,6P>
# 与 THz 型 60S↔6P/60P↔6P，实测占 5184 中 5040 个、|V| ≤ 1e-4 MHz）其离共振净
# 效应 ≤ V²/Δ ≤ 1e-8 MHz，集体位移 ≤ ~1e-7 MHz ≪ rtol——硬算表示下省略
# （每元素逐个 exp 相位是 RHS 主开销：5184 MACs ≈ 59µs/RHS）。144 个物理活性
# 元素全保留（|dE|·C ≤ 218 rad/ns，精确相位）。


def _embed_triplets(A, side, n_self, n_other, mask=None):
    """单原子 A（n_self×n_self）→ 两体嵌入三元组 (rows, cols, vals[, signs])。

    side='A'：A⊗I（k=(i,j)→i·n_other+j）；side='B'：I⊗A（j·n_self+i）。
    旁观维 n_other=1 时退化为单体三元组（rows=a）——t73b 单/两体同一内核。
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
            for j in range(n_other):
                rows.append(a * n_other + j)
                cols.append(b * n_other + j)
                vals.append(float(A[a, b]))
                if has_mask:
                    signs.append(s)
    else:
        for a, b in zip(ii, jj):
            s = mask[a] - mask[b] if has_mask else 0
            for i in range(n_other):
                rows.append(i * n_self + a)
                cols.append(i * n_self + b)
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


def _pack_pair_hard(fields_meta_A, fields_meta_B, T, vdd_sp, n_tab,
                    E_lab=None):
    """→ (E_A, E_B, fld_rows, fld_cols, fld_vals, fld_envs, fld_phase_tabs,
         fld_ptr, fld_deltas, fld_sgns,
         vdd_rows, vdd_cols, vdd_vals, vdd_grp_start, vdd_grp_dE, t0, dt, CONV)
    E_A/E_B：每原子 IP 能量（各自减 min；单体旁观模式 N_B=1、E_B=[0]）；
    fld_*：全部场项三元组（光场相位表=0 → cos(0)=1，微波相位表=C·ω·t 或
    phase_fn——t18 统一 cos(φ_f(t)) 任意相位）；fld_deltas：光场拍频
    δ=ω_f−ω_ref（微波/首场=0）；fld_sgns：每三元组相位符号 m[i]−m[k]；
    vdd_*：V_dd 按 dE 排序分组（E_lab 仅此时需要）。
    t73b：N_A/N_B 由两侧 meta 的 E_rot 长度推断——单/两体同一打包。
    """
    E_A = np.asarray(fields_meta_A['E_rot'], float)
    E_B = np.asarray(fields_meta_B['E_rot'], float)
    E_A = E_A - float(E_A.min())
    E_B = E_B - float(E_B.min())
    N_A = E_A.size
    N_B = E_B.size
    E_lab_pair = (np.array([E_lab[i] + E_lab[j]
                            for i in range(N_A) for j in range(N_B)])
                  if vdd_sp is not None else None)

    # ---- 场项 ----
    fld_rows_l, fld_cols_l, fld_vals_l, fld_env_l = [], [], [], []
    fld_phase_l, fld_delta_l, fld_sgn_l = [], [], []
    for side, meta in (('A', fields_meta_A), ('B', fields_meta_B)):
        n_self, n_other = ((N_A, N_B) if side == 'A' else (N_B, N_A))
        for tup in meta['opt_terms']:
            A_opt, wf = tup[0], tup[1]
            omega_f = float(tup[2])
            omega_ref = float(tup[4]) if len(tup) > 4 else omega_f
            mask = tup[5] if len(tup) > 5 else None
            delta = omega_f - omega_ref
            if delta != 0.0 and mask is not None:
                r, c, v, sg = _embed_triplets(A_opt, side, n_self, n_other,
                                              mask)
                fld_sgn_l.append(sg)
            else:
                r, c, v = _embed_triplets(A_opt, side, n_self, n_other)
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
                r, c, v = _embed_triplets(A, side, n_self, n_other)
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
@njit(cache=True, nogil=True)
def _rhs_hard_pair(t, y, E_A, E_B, fld_rows, fld_cols, fld_vals, fld_envs,
                   fld_phase_tabs, fld_ptr, fld_deltas, fld_sgns,
                   vdd_rows, vdd_cols, vdd_vals,
                   vdd_grp_start, vdd_grp_dE, t0t, dtt, CONV, gamma=None):
    """两体 hard RHS：y' = −iC·[u∘(Ṽ·(conj(u)∘y)) + V_dd_twisted·y]。

    光场拍频：δ=ω_f−ω_ref ≠ 0 的项乘 exp(iC·δ·t·sgn)（sgn=每三元组
    相位符号，m[i]−m[k]；δ=0 逐位退化——t10 验证 exp 拍频正确）。
    微波（t18）：cos(φ_f(t)) 相位表插值——定频 φ=C·ω·t（光场相位表=0
    → cos(0)=1），变频/任意相位由 phase_fn 采样。
    耗散（t24）：gamma（N²，MHz）NonHermitian 对角 −iΓ/2。
    """
    N_A = E_A.size
    N_B = E_B.size
    N2 = N_A * N_B
    uA = np.exp(1j * CONV * E_A * t)
    uB = np.exp(1j * CONV * E_B * t)
    # zu[k] = conj(uA[i])·conj(uB[j])·y[k]（旁观 N_B=1 → uB≡1，单体动力学）
    zu = np.empty(N2, np.complex128)
    for i in range(N_A):
        ui = uA[i].conjugate()
        for j in range(N_B):
            zu[i * N_B + j] = ui * uB[j].conjugate() * y[i * N_B + j]
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
    for i in range(N_A):
        ui = uA[i]
        for j in range(N_B):
            k = i * N_B + j
            out[k] = -1j * CONV * (ui * uB[j] * dy_f[k] + dy_v[k])
    # t24：耗散 NonHermitian 对角 −iΓ_MHz/2 → RHS −(C·Γ)/2·y
    if gamma is not None:
        for k in range(N2):
            out[k] -= 0.5 * CONV * gamma[k] * y[k]
    return out

# ============================================================
# DOP853 驱动已上收至 src/core/dop853.py（first-class RHS 参数化，
# t11 实验验证与旧两体驱动逐位一致）。本模块只保留内核与打包。
# ============================================================


# ============================================================
# hard 引擎的 DOP853 驱动（薄包装 —— 让 numba 磁盘缓存真正命中）
# ============================================================
# 机制与动机同 coupling.dop853_driver_mixed（那边注释更详）：rhs 当参数会让
# 缓存键含 Dispatcher 类型（WeakType，键 ≈ 对象 id）→ 跨进程永不命中，
# 索引无限膨胀、保存时还可能抛 ReferenceError 把整轮演化搞失败。
# 本包装把 rhs 静态绑定成模块级常量；dop853 内核 `inline='always'` 保证
# 包装的编译产物里对 rhs 是直接全局调用（否则会被判 has_dynamic_globals
# 而拒绝缓存）。数值上只是原样转发同一批实参（附加参数走元组），逐位不变。
@njit(cache=True, nogil=True)
def dop853_driver_hard(y0, t_eval, T, rtol, atol, max_step, prog, stat,
                       dt_hist, *rargs):
    """hard 引擎（单/两体共用）：静态绑定 _rhs_hard_pair 的 DOP853 驱动。"""
    return _dop853_driver(_rhs_hard_pair, y0, t_eval, T, rtol, atol,
                          max_step, prog, stat, dt_hist, rargs)


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
        GammaA, GammaB, _ = _tensor._pair_loss_vectors(
            states, decays_A, decays_B, N, N2)
    gamma_MHz = np.zeros(N2)
    if has_decay:
        gamma_MHz = (np.asarray(GammaA, float)
                     + np.asarray(GammaB, float)) / _CONV

    # ---- 初态 ----
    psi0_A_vec = _as_vector(states, psi0_A)
    psi0_B_vec = _as_vector(states, psi0_B)
    psi0_2d = _tensor._as_product_psi(psi0_A_vec, psi0_B_vec, N)

    E_lab = np.array([st['E_mhz'] for st in states], float)
    pack = _pack_pair_hard(fields_meta_A, fields_meta_B, T, V, n_tab,
                           E_lab=E_lab)
    E_A, E_B = pack[0], pack[1]
    E_lab = np.array([s['E_mhz'] for s in states], float)
    E_pair = np.array([E_lab[i] + E_lab[j]
                       for i in range(N) for j in range(N)])

    from .coupling import stage_progress
    stage_progress(progress_arr, 0, 0.9)     # 打包完成（JIT 装载在驱动首调）

    if driver == 'numba':
        prog = (np.zeros(3) if progress_arr is None
                else np.asarray(progress_arr, float))
        stat = np.zeros(4)
        # 阶段 1（积分）由驱动在首个接受步置位（同 evolve_hard_fast）
        # 复用 dop853 核心层（first-class RHS 参数化，t11 验证逐位一致）
        # t24：gamma_MHz 传入内核（落点=丢失 NonHermitian，此前只算不用）
        y_out = dop853_driver_hard(psi0_2d, t, T,
                                   float(rtol), float(atol), np.inf,
                                   prog, stat, None, *pack, gamma_MHz)
        nfev = int(stat[0])
        n_acc = int(stat[1])
        n_rej = int(stat[2])
        stage_progress(prog, 2, 0.5)         # 阶段 2：收尾（loss/约化）
    else:
        # driver='scipy'：njit RHS + scipy solve_ivp（对照用）
        from scipy.integrate import solve_ivp

        def _rhs_hard_pack(tv, y):
            return _rhs_hard_pair(tv, y, *pack, gamma_MHz)
        stage_progress(progress_arr, 1, 0.0)
        sol = solve_ivp(_rhs_hard_pack, (0.0, T), psi0_2d, method='DOP853',
                        t_eval=t, rtol=rtol, atol=atol)
        if not sol.success:
            raise RuntimeError(f'hard-pair(scipy) 积分失败: {sol.message}')
        y_out = sol.y.T
        nfev = int(getattr(sol, 'nfev', 0))
        n_acc = -1
        stage_progress(progress_arr, 2, 0.5)

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

    pops_A = _tensor._reduce_A(pops_2d, N)
    pops_B = _tensor._reduce_B(pops_2d, N)
    stage_progress(progress_arr, 2, 1.0)
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



# ============================================================
# 输入格式构造（t74 自 evolve_twofield 归位——hard 的输入格式属于 hard）
# build_two_field / arbitrary_mw_group / split_by_freq / evolve_hard(scipy 参考)
# ============================================================
MW_MAX_MHZ = 1e5            # 载频 ≤ 100 GHz 视为微波，> 视为光场
OPT_NRES_BW_MHZ = 2e3       # 旋转后近共振对筛选带宽（2 GHz）
MW_NRES_BW_MHZ = 5e3        # 微波近共振对筛选带宽（5 GHz）


def arbitrary_mw_group(cache, states, wave):
    """ArbitraryWave → hard 兼容 mw_group（t18：旧任意波形重构适配 hard）。

    解析信号分解（Hilbert）：amp(t) = env(t)·cos(φ(t)) 精确（env=|z|,
    φ=unwrap arg z，重构误差 ~1e-11）；env/φ 以绝对时间插值函数给出，
    经 hard 打包为查表（cos(φ_f(t)) 任意相位路径，内核插值求值）。
    A 为全偶极实振幅（|d|·rabi）——复杂偶极相位（多通道干涉）暂以 |d|
    吸收，单通道驱动精确（多通道场景为后续扩展）。
    用法：meta = dict(E_rot=…, N=N, opt_terms=[], disc_times=[],
                      mw_groups=[arbitrary_mw_group(cache, states, wave)])。
    """
    from scipy.signal import hilbert
    q = {'σ+': +1, 'π': 0, 'σ-': -1}.get(wave.pol, 0)
    E0_sq = core.field_e0_sq(float(wave.intensity_mwcm2))
    N = len(states)
    A = np.zeros((N, N))
    for i in range(N):
        for j in range(i + 1, N):
            d = core.dipole_amplitude(cache, _crow(states[i]),
                                      _crow(states[j]), q)
            if abs(d) < 1e-12:
                continue
            A[i, j] = A[j, i] = core.rabi_mhz(abs(d), E0_sq)
    if not np.any(A):
        raise ValueError('arbitrary_mw_group：偏振下无任何偶极耦合')
    ts = wave.t_ns + wave.offset_ns          # 绝对时间样品
    z = hilbert(wave.amp)
    env_h, phi_h = np.abs(z), np.unwrap(np.angle(z))
    return dict(omega_mhz=0.0, As=[A],
                envs=[lambda tv: np.interp(np.asarray(tv, float), ts, env_h,
                                           left=0.0, right=0.0)],
                phase_fn=lambda tv: np.interp(
                    np.asarray(tv, float), ts, phi_h,
                    left=phi_h[0], right=phi_h[-1]),
                labels=[wave.name])


def _crow(s):  # t74：与 field_build._sublevel_row 同构（保留旧名以免连带改动）
    """states 元素 → core.dipole_amplitude 子能级行。"""
    return dict(n=s['n'], l=s['l'], j=s['j'], kind=s['kind'],
                mj=s.get('mj'), mF=s.get('mF'), amps=s.get('amps'))


def _build_A(cache, states, freq_mhz, intensity_mwcm2, pol, det_selector):
    """按失谐条件构造 Rabi 矩阵 A[i,j]（峰值，MHz）。

    det_selector(i, j, E) → bool：是否把该对收入（近共振筛选）。
    E 为 states 能量数组（旋转前）。"""
    q = {'σ+': +1, 'π': 0, 'σ-': -1}.get(pol, 0)
    E0_sq = core.field_e0_sq(float(intensity_mwcm2))
    N = len(states)
    E = np.array([s['E_mhz'] for s in states], float)
    A = np.zeros((N, N))
    for i in range(N):
        for j in range(i + 1, N):
            if not det_selector(i, j, E):
                continue
            d = core.dipole_amplitude(cache, _crow(states[i]),
                                      _crow(states[j]), q)
            if abs(d) < 1e-12:
                continue
            A[i, j] = A[j, i] = core.rabi_mhz(abs(d), E0_sq)
    return A


def adaptive_split_threshold(states, default_mhz=MW_MAX_MHZ):
    """能级间隔分布 → 微波/光场自适应分界阈值（MHz）。

    物理：旋转系 RWA 只在载频 ≫ 里德堡尺度跃迁时有效；微波硬积分受
    DOP853 载波可行性约束。能级间隙在 log 尺度成簇（里德堡带 vs 光学
    带），取最大对数间隔比的几何中点作阈值；clamp 到 [1e4, 1e6]
    （DOP853 载波可行性安全带）。鲁棒性：无清晰两簇（单簇/退化）或
    孤立离群点（某侧间隙占比 < 5%）时回退 default_mhz。
    """
    E = np.array([s['E_mhz'] for s in states], float)
    if E.size < 2:
        return default_mhz
    d = np.abs(E[:, None] - E[None, :])
    gaps = np.unique(d[np.triu_indices(E.size, 1)])
    gaps = gaps[gaps > 0]
    if gaps.size < 2:
        return default_mhz
    ratios = gaps[1:] / gaps[:-1]
    k = int(np.argmax(ratios))
    r = ratios[k]
    thr = np.sqrt(gaps[k] * gaps[k + 1])
    # 双侧最小占比（孤立离群点不干扰聚类）：各侧 ≥ 5%（且 ≥ 3 个间隙）
    n = gaps.size
    lo = int(np.searchsorted(gaps, thr, side='right'))
    if (r >= 1e3 and lo >= max(3, 0.05 * n)
            and (n - lo) >= max(3, 0.05 * n)):
        return float(np.clip(thr, 1e4, 1e6))
    return default_mhz


def split_by_freq(waves, states=None):
    """波形列表 → (opt_waves, mw_waves)。

    states 提供时按能级间隔分布自适应分界（adaptive_split_threshold）；
    否则回退固定 MW_MAX_MHZ（向后兼容）。
    """
    thr = (adaptive_split_threshold(states)
           if states is not None else MW_MAX_MHZ)
    opt, mw = [], []
    for w in waves:
        if w.freq_mhz > thr:
            opt.append(w)
        else:
            mw.append(w)
    return opt, mw


def build_two_field(cache, states, waves, mw_M=2):
    """构造两场引擎所需结构。

    返回 dict:
      E_rot      : 旋转后对角能（MHz）——6P 等被光场耦合的低能流形 +ω_opt
      opt_terms  : [(A_opt(N×N), waveform, omega_opt)] 光场（不占光子块）
      mw_groups  : _group_by_freq 格式的微波组（每载频一个，As 为其驱动矩阵）
      N          : 物理态数
    """
    opt_w, mw_w = split_by_freq(waves, states)   # t20：自适应阈值（states 驱动）
    N = len(states)
    E = np.array([s['E_mhz'] for s in states], float)

    # ---- 光场：确定低能流形与旋转参考 ----
    # 对每个光场，先按"旋转前载频近共振"找耦合对（|ΔE−ω|<2GHz），
    # 再取其中能量较低的一侧流形作为旋转对象（该流形整体 +ω）。
    rot_pick = {}            # 流形 (n,l,j) → 是否已定参考
    ref_add = np.zeros(N)
    opt_terms = []
    for w in opt_w:
        # 近共振对（旋转前，以跃迁频率 ≈ ω 判）
        def _sel(i, j, E, w=w):
            return abs(abs(E[i] - E[j]) - w.freq_mhz) < OPT_NRES_BW_MHZ
        A = _build_A(cache, states, w.freq_mhz, w.intensity_mwcm2, w.pol,
                     _sel)
        if not np.any(A):
            continue
        # 该场耦合的所有流形对中的低能流形
        ii, jj = np.nonzero(A)
        keys = [(states[i]['n'], states[i]['l'], states[i]['j'])
                for i in set(ii.tolist() + jj.tolist())]
        # 按平均能量排序，取最低流形作为旋转对象（6P 场景）
        eavg = {}
        for k in keys:
            idxs = [i for i, s in enumerate(states)
                    if (s['n'], s['l'], s['j']) == k]
            eavg[k] = float(np.mean([E[i] for i in idxs]))
        low_k = min(keys, key=lambda k: eavg[k])
        if low_k not in rot_pick:
            rot_pick[low_k] = w.freq_mhz
        omega_ref = rot_pick[low_k]
        # 旋转参考 = 第一个耦合该流形的光场载频（多光场共享流形取首个）
        for i, s in enumerate(states):
            if (s['n'], s['l'], s['j']) == low_k:
                ref_add[i] = omega_ref
        # 拍频信息（任务书 §2.3 / §8 多光场；t10 验证：exp 非 cos）：
        # 第 2+ 个光场相对共享参考 ω_ref 失谐 δ=ω_f−ω_ref，其耦合项在旋转
        # 系内带 exp(i·C·δ·t) 相位（本场自身 δ=0 → 逐位退化为旧行为）。
        # 元组扩展为 (A, wf, omega, name, omega_ref, mask)；mask 标记被本
        # 参考旋转的流形（用于相位符号 m[i]−m[k]）。路径 A（Sambe）复用
        # omega_ref，路径 B（hard）用 δ。
        mask = np.array([(s['n'], s['l'], s['j']) == low_k
                         for s in states], int)
        # RWA 振幅 ×0.5（t17 船长裁决：两场引擎光场项此前用全 Rabi（rabi_mhz），
        # 物理错误 2×——cos(ωt)=½(e^{iωt}+e^{−iωt}) 旋转系只留共振项 → 耦合
        # = ½·rabi_mhz。与 exact 引擎 SambeSetup 的 0.5·A 对齐（evolve_exact.py:
        # 226）。此前 AC-Stark 位移偏 4×（Ω²/Δ）——"AC 位移调交换共振"是本场景
        # 核心物理。微波项保持 1.0（实验室系全 cos，硬积分无 RWA，正确）。
        # 单点修复：所有消费方（Sambe opt_ext / defect / hard coupling_at / numba /
        # pair 打包）自动获得 ½。om_mhz 报告随之变为 RWA 耦合振幅。
        A = 0.5 * A
        opt_terms.append((A, w.waveform, w.freq_mhz, w.name,
                          float(omega_ref), mask))
    E_rot = E + ref_add

    # ---- 收集光学包络不连续点（SquareEnv 上升沿/下降沿边界） ----
    # C¹ 连续的缓变方波在二阶导数处有 kink，Gauss-Legendre 求积收敛慢；
    # 必须将不连续点作为段边界，避免段内包络突变导致缺陷修正非幺正。
    disc_times = []
    for w in opt_w:
        env = w.envelope
        if hasattr(env, 'start_ns'):       # SquareEnv（有 start_ns；GaussianEnv 没有）
            abs_off = float(w.offset_ns)
            s0 = abs_off + float(env.start_ns)
            s1 = abs_off + float(env.start_ns) + float(env.edge_ns)
            s2 = abs_off + float(env.fall_start_ns)
            s3 = abs_off + float(env.fall_start_ns) + float(env.edge_ns)
            for v in (s0, s1, s2, s3):
                if v not in disc_times:      # dedup via list (small N)
                    disc_times.append(v)
    # 去重（相邻 < 1μs 视为同一点）并排序
    if len(disc_times) > 1:
        disc_times.sort()
        filtered = [disc_times[0]]
        for dt in disc_times[1:]:
            if dt - filtered[-1] > 1e-6:
                filtered.append(dt)
        disc_times = filtered

    # ---- 微波场：近共振对筛选（60S↔60P 等，|ΔE−ω|<5GHz）----
    mw_groups = []
    for w in mw_w:
        def _selm(i, j, E, w=w):
            return abs(abs(E[i] - E[j]) - w.freq_mhz) < MW_NRES_BW_MHZ
        A = _build_A(cache, states, w.freq_mhz, w.intensity_mwcm2, w.pol,
                     _selm)
        if not np.any(A):
            continue
        hit = None
        for g in mw_groups:
            if abs(g['omega_mhz'] - w.freq_mhz) <= 1e-9 * max(abs(w.freq_mhz), 1.0):
                hit = g
                break
        if hit is None:
            hit = dict(omega_mhz=w.freq_mhz, As=[], envs=[], labels=[])
            mw_groups.append(hit)
        hit['As'].append(A)
        hit['envs'].append(w.waveform)
        hit['labels'].append(w.name)
    return dict(E_rot=E_rot, opt_terms=opt_terms, mw_groups=mw_groups,
                N=N, disc_times=disc_times)


def evolve_hard(states, fields_meta, psi0, t_grid, rtol=1e-9, atol=1e-11,
                progress_arr=None, log=None, record_amp=False, numba=False,
                decays=None):
    """微波硬算 + 光场旋转系（第一性原理，**零微波近似**）——method='hard'。

    numba=True 时委托 hard_numba.evolve_hard_fast（同数学、同 DOP853、
    同返回契约，620ns 22 态 173s → ~4s；包络查表误差 ~3e-10 ≪ rtol）。

    核心思想（用户提议，2026-08-29）：
    - **光场**：旋转系解析吸收载波（6P 流形 ref=ω_opt，耦合对停在失谐），
      光学项为常数矩阵 c(t)·A_opt——唯一近似是光学 RWA（反旋项全场
      Bloch–Siegert 相位 4.4e-5 rad ≪ tol，可忽略；光学多光子即电离无落点）。
    - **微波**：**不做任何变换**。H(t) 保留完整 cos(ω_mw·t) 载波，
      solve_ivp DOP853 自适应硬积分：
        H(t) = diag(E_rot) + Σ_opt c(t)·A_opt + Σ_mw c(t)·cos(C·ω_mw·t)·A_mw
      → 微波多光子、Bloch–Siegert、任意谐波**天然全含**，零近似零截断；
      → 步长由 17 GHz 载波（周期 58 ps）决定：1500 ns ≈ 2.6 万周期 × ~15 步
        ≈ 4×10⁵ 步 × 22 维 ≈ 10⁹ flop → 秒级；两体 484 维 ~10¹¹ flop → 分钟级；
      → **支持任意波动的微波场**（不只包络：任意时域信号、多频、啁啾都行，
        因为 RHS 直接逐个时刻求值，无周期结构假设）。
    - 误差：DOP853 自适应（rtol/atol 显式）；RHS 厄米 → 范数机器精度守恒
      （对比 Sambe 路径的重构相位 bug——两种变换拼合容易出错，硬算从结构上免疫）。

    返回与 evolve_exact 同构 dict（method='hard'）。
    """
    if numba:
        from .hard import evolve_hard_fast
        return evolve_hard_fast(states, fields_meta, psi0, t_grid,
                                rtol=rtol, atol=atol,
                                progress_arr=progress_arr, log=log,
                                record_amp=record_amp, decays=decays)
    from scipy.integrate import solve_ivp
    log = log or (lambda *a: None)
    t = np.asarray(t_grid, float)
    N = fields_meta['N']
    E_rot = np.asarray(fields_meta['E_rot'], float)
    opt_terms = fields_meta['opt_terms']
    mw_groups = fields_meta['mw_groups']
    psi0v = _as_vector(states, psi0)
    T = float(t[-1])
    # t24：耗散（落点=丢失）——Gamma_ns（/ns），RHS 加 −Γ/2·y
    has_decay = bool(decays)
    if has_decay:
        Gamma_ns = _single_loss_rates(states, decays)
    else:
        Gamma_ns = None

    E = E_rot - float(E_rot.min())          # 公共偏移=全局相位（精确等价）

    # 微波：A 与 (ω, env, 相位函数) 列表；t18 统一 cos(φ_f(t)) 任意相位——
    # 定频 φ=C·ω·t（默认），变频/任意相位由 mw_groups['phase_fn'] 提供。
    mw_terms = []
    for g in mw_groups:
        w = float(g['omega_mhz'])
        ph_fn = g.get('phase_fn')
        for A, wf in zip(g['As'], g['envs']):
            mw_terms.append((np.asarray(A, float), w, wf, ph_fn))

    def coupling_at(tv):
        """旋转系相互作用绘景：H0=diag(E) 精确，Ṽ = u·(光+微波 cos(φ))·u†。

        光场拍频（§2.3/§8）：第 2+ 光场相对共享参考失谐 δ=ω_f−ω_ref，
        耦合项乘 exp(i·C·δ·t·(m[i]−m[k]))（m=旋转流形掩码；t10 验证：
        exp 拍频正确、cos 错误）。首场 δ=0 → 乘 1，逐位退化为旧行为。
        微波（t18）：cos(φ_f(t))，φ 任意（定频=chirp 特例）。
        """
        V = np.zeros((N, N), complex)
        for tup in opt_terms:
            A_opt, wf = tup[0], tup[1]
            omega_f = float(tup[2])
            omega_ref = float(tup[4]) if len(tup) > 4 else omega_f
            m = tup[5] if len(tup) > 5 else None
            c = env_at(wf, tv)
            if c != 0.0:
                d = omega_f - omega_ref
                if d != 0.0 and m is not None:
                    ph = np.exp(1j * _CONV * d * tv
                                * (m[:, None] - m[None, :]))
                    V = V + c * ph * A_opt
                else:
                    V = V + c * A_opt
        for A, wL, wf, ph_fn in mw_terms:
            env = env_at(wf, tv)
            if env != 0.0:
                ph = (ph_fn(tv) if ph_fn is not None
                      else _CONV * wL * tv)
                V = V + env * np.cos(float(np.asarray(ph).flat[0])) * A
        u = np.exp(1j * _CONV * E * tv)
        return u[:, None] * V * np.conj(u)[None, :]

    def _rhs_twofield(tv, y):
        out = -1j * _CONV * (coupling_at(tv) @ y)
        if Gamma_ns is not None:                 # 落点=丢失 NonHermitian
            out = out - 0.5 * Gamma_ns * y
        return out

    if progress_arr is not None:
        if len(progress_arr) > 2:
            progress_arr[2] = 1.0            # 阶段 1：积分（scipy 无内部钩子）
        progress_arr[0] = 0.0
    sol = solve_ivp(_rhs_twofield, (0.0, T), psi0v, method='DOP853',
                    t_eval=t, rtol=rtol, atol=atol)
    if not sol.success:
        raise RuntimeError(f'hard 积分失败: {sol.message}')
    pops = np.abs(sol.y).T ** 2
    if progress_arr is not None:
        if len(progress_arr) > 2:
            progress_arr[2] = 2.0
        progress_arr[0] = 1.0
    nfev = getattr(sol, 'nfev', None)
    log(f'[hard] DOP853 完成：{len(t)} 输出点'
        + (f'，{nfev} 次 RHS 求值' if nfev else '')
        + f'（微波载波 {max((abs(w) for _, w, _, _ in mw_terms), default=0):.4g} MHz'
        f' 硬积分，零微波近似）')
    report = dict(n_segments=1, max_richardson_diff=0.0, dt_min_hit=False,
                  max_norm_dev=float(np.max(np.abs(pops.sum(axis=1) - 1.0))))
    if has_decay:
        from .hard import _loss_and_norm
        loss, norm_dev = _loss_and_norm(pops, Gamma_ns, t)
        report['max_norm_dev'] = norm_dev
    else:
        loss = np.zeros(len(t))
    if report['max_norm_dev'] > 1e-6:
        log(f'[hard] ⚠ 范数偏差 {report["max_norm_dev"]:.2e}'
            '（DOP853 容差不足或 RHS 非厄米）')
    out = dict(t=t, pops=pops, states=states,
               om_mhz=[float(np.abs(A).max()) for A, _, _, _ in mw_terms]
               + [float(np.abs(A_opt).max()) for A_opt, *_ in opt_terms],
               method='hard', backend='hard-dop853',
               loss=loss, error_report=report)
    if record_amp:
        uT = np.exp(1j * _CONV * E * T)
        out['psi_amp'] = np.conj(uT) * sol.y[:, -1]
    return out

