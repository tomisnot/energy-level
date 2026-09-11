# -*- coding: utf-8 -*-
"""RHS 打包公共层（单/两体共用）。

- `pack_rhs(pairs, t_end)`：pairs 7/8 元组 → njit 数组。分组键 (is_mw, w, id(env))
  ——**is_mw 决定 cos 载波，故光场/微波即使共用同一 env 对象也必须分属不同组**。
  env/相位预采样成表 + 线性插值。
- `static_phase_keep(vals, dE, T)`：V_dd 累积相位判据（hard / mixed 共用）。
- `mw_cap_mhz(T)` / `mw_candidate_mask(...)`：微波候选窗（成本 + 精度）。
- `mw_pair_keep(Ω, Δ, T)`：单个微波对的精度判据。

判据推导与历史动因见 `docs/参数与判据决策史.md`。"""
import numpy as np

from .params import (PHASE_TOL, PHASE_BUDGET, MW_MAX_MHZ,
                     _RHS_ENV_TAB_N)
from .evolve_util import _CONV


def stage_progress(arr, st, frac):
    """三槽进度协议助手（t72，GUI ui_dynamic._poll_progress 消费）。

    arr 槽位：[0]=**阶段内**进度 0..1、[1]=取消标志（只读）、[2]=阶段
    （0 准备/JIT 装载、1 积分、2 收尾）。engines 在各阶段边界调用本助手；
    积分阶段由 _dop853_driver 逐步写 arr[0]=t/T。arr 短于 3（旧脚本
    len-2 用法）时只写 arr[0]、跳过阶段槽——旧消费者语义不变。
    """
    if arr is None:
        return
    arr[0] = min(max(float(frac), 0.0), 1.0)
    if len(arr) > 2:
        arr[2] = float(st)


def pack_rhs(pairs, t_end, n_tab=_RHS_ENV_TAB_N):
    """pairs → numba 数组（按 (is_mw, w, id(env), id(ph_fn)) 分组 + 查表预采样）。

    pairs 元素：(a, b, v, dD, is_mw, w, env[, ph_fn])——分组键含 is_mw/w
    （t62 用例 1 教训：光场/微波即使 env 同一对象也须分属不同组——is_mw 决定
    cos 载波）。

    **任意相位（t79）**：pairs 可选第 8 元 `ph_fn`（t→弧度，可为任意 callable）。
    给了它则该波的载波取 `cos(ph_fn(t))` 而非解析的 `cos(Cv·w·t)`：
      · 单调线性相位 φ=ωt 时两者**逐位相同**（线性函数线性插值零误差）；
      · 任意波形（无单一载频）用它表达，与 hard 的 cos(φ_f(t)) 同构。
    相位与包络共用同一时间基（linspace(0,t_end,n_tab)）与 env_dt。
    """
    grp = {}
    order = []
    for p in pairs:
        a, b, v, dD, is_mw, w, env = p[:7]
        ph_fn = p[7] if len(p) > 7 else None
        key = (int(is_mw), float(w), id(env), id(ph_fn))
        if key not in grp:
            grp[key] = dict(is_mw=is_mw, w=w, env=env, ph_fn=ph_fn, items=[])
            order.append(key)
        grp[key]['items'].append((a, b, v, dD))
    iis, jjs, vals, dDs = [], [], [], []
    is_mw_a, wcos_a, n_ppw, has_ph_a = [], [], [], []
    env_rows, ph_rows = [], []
    ts = np.linspace(0.0, t_end, n_tab)
    for key in order:
        g = grp[key]
        for (a, b, v, dD) in g['items']:
            iis.append(a); jjs.append(b); vals.append(v); dDs.append(dD)
        n_ppw.append(len(g['items']))
        is_mw_a.append(1 if g['is_mw'] else 0)
        wcos_a.append(g['w'])
        env_rows.append(np.asarray(g['env'](ts), float))
        if g['ph_fn'] is not None:
            has_ph_a.append(1)
            ph_rows.append(np.asarray(g['ph_fn'](ts), float))
        else:
            has_ph_a.append(0)
            ph_rows.append(np.zeros(n_tab))
    env_tab = np.stack(env_rows) if env_rows else np.zeros((0, n_tab))
    ph_tab = np.stack(ph_rows) if ph_rows else np.zeros((0, n_tab))
    env_dt = t_end / (n_tab - 1) if n_tab > 1 else 1.0
    return dict(iis=np.asarray(iis, np.int64),
                jjs=np.asarray(jjs, np.int64),
                vals=np.asarray(vals, np.complex128),
                dDs=np.asarray(dDs, np.float64),
                is_mw=np.asarray(is_mw_a, np.int64),
                wcos=np.asarray(wcos_a, np.float64),
                n_pairs_w=np.asarray(n_ppw, np.int64),
                env_tab=env_tab, env_dt=env_dt,
                has_ph=np.asarray(has_ph_a, np.int64), ph_tab=ph_tab)


# ============================================================
# 微波候选窗：相预算派生（成本侧）
# ============================================================
def mw_cap_mhz(T_ns, phase_budget=None, Cv=_CONV):
    """微波候选窗上限（MHz）：ΔD_max = PHASE_BUDGET / (C · T)。

    φ-帧微波项振荡在 (|ΔD| ± ω)，DOP853 步长按最快项走；把"愿意付多少步"
    写成无量纲相预算，即得只随 T 自适应、与体系无关的窗。最坏总步数
    ≈ PHASE_BUDGET/0.5，与 T、态数无关。

    与 `mw_pair_keep`（PHASE_TOL，精度侧）配成一对。
    phase_budget=None → params.PHASE_BUDGET。
    """
    b = PHASE_BUDGET if phase_budget is None else float(phase_budget)
    T = max(float(T_ns), 1e-12)            # T→0 时由调用方的绝对上界兜
    return b / (Cv * T)


def mw_candidate_mask(Omega, dE, omega_mhz, T_ns, phase_budget=None,
                      cap_abs_mhz=MW_MAX_MHZ, Cv=_CONV):
    """向量化「微波候选对」筛选（成本 + 精度，与 mixed 同构 → 两引擎保留集一致）。

      Ω : Rabi 矩阵（MHz）    dE: 两端能量差（MHz，实验室系）
    返回布尔掩码（同 Ω 形状）。判定顺序：
      ① Ω ≤ 0 → 丢
      ② Δ = | |dE| − ω | > min(mw_cap_mhz(T), cap_abs_mhz) → 丢（成本）
      ③ Θ = C·T·Ω²/max(Ω, Δ) < PHASE_TOL → 丢（精度）
    """
    Om = np.abs(np.asarray(Omega, float))
    dE = np.abs(np.asarray(dE, float))
    det = np.abs(dE - float(omega_mhz))
    cap = min(mw_cap_mhz(T_ns, phase_budget, Cv), float(cap_abs_mhz))
    keep = (Om > 0.0) & (det <= cap)
    den = np.maximum(Om, det)
    with np.errstate(divide='ignore', invalid='ignore'):
        theta = Cv * float(T_ns) * Om ** 2 / den
    keep &= (theta >= PHASE_TOL)
    return keep


# ============================================================
# 微波对精度判据（累积相位）
# ============================================================
def mw_pair_keep(Omega_mhz, det_mhz, T_ns, cap_mhz=None,
                 theta_tol=PHASE_TOL, Cv=_CONV):
    """单个微波耦合对是否保留（标量版；向量化见 `mw_candidate_mask`）。

      Θ = Cv · T · Ω² / max(Δ, Ω) ≥ θ_tol  且  Δ ≤ cap_mhz
      Ω = 该对 Rabi（MHz）；Δ = | |ΔD| − ω |（相对光子的失谐，MHz）

    与 `static_phase_keep`（V_dd）同一判据，依赖 (Ω, Δ, T) 而非固定能量窗：
    强场保留远失谐多光子通道（多光子路径自然涌现的前提），弱场自动丢弃
    无贡献对，跨尺度 THz 对 Ω 极小 → Θ→0 必被丢。

    `cap_mhz` 是成本预算（见 `mw_cap_mhz`），不是物理近似；None = 不设上限。
    """
    if Omega_mhz <= 0.0:
        return False
    if cap_mhz is not None and abs(det_mhz) > float(cap_mhz):
        return False
    den = max(abs(float(det_mhz)), abs(float(Omega_mhz)))   # Ω>0 且 Δ≥0 → den>0
    return (Cv * float(T_ns) * float(Omega_mhz) ** 2 / den) >= theta_tol


# ============================================================
# 静态耦合（V_dd）丢弃判据：累积相位 Θ —— hard / mixed 共用
# ============================================================
def static_phase_keep(vals, dE, T, theta_tol=PHASE_TOL, Cv=_CONV):
    """V_dd 累积相位判据：Θ = C·T·|V|²/max(|ΔE|, |V|) ≥ theta_tol → 保留。

    参数
      vals     : 逐元素耦合矩阵元（MHz，复数取模；任意形状）
      dE       : 逐元素两端能量差（MHz，与 vals 同长）
      T        : 演化窗口（ns）——效应是累积量，窗口越长判据越严
      theta_tol: 无量纲容差（rad），默认 params.PHASE_TOL
    返回
      (keep 布尔掩码, info dict)——info 供日志/自检：
        n_keep/n_drop、max_v_drop（被剔项最大 |V|）、max_theta_drop、
        max_rate_keep（保留项最大 C·|ΔE|，rad/ns——越大说明留下的相位越快）

    两个极限自动切换：共振（|ΔE|≲|V|）→ Θ=C·T·|V|；远失谐 → Θ=C·T·|V|²/Δ。
    判据只用 (|V|,|ΔE|,T)，**与基组标度、引擎、参考系无关**——故可跨算法共用。
    |V|=|ΔE|=0 的元素 Θ 记 0（丢弃）。
    """
    v = np.abs(np.asarray(vals, complex)).ravel()
    d = np.abs(np.asarray(dE, float)).ravel()
    if v.shape != d.shape:
        raise ValueError(f'vals 与 dE 长度不一致（{v.shape} vs {d.shape}）')
    den = np.maximum(d, v)
    th = np.zeros(v.shape[0], float)
    np.divide((Cv * float(T)) * v * v, den, out=th, where=(den > 0.0))
    keep = th >= float(theta_tol)
    dr = ~keep
    info = dict(n_keep=int(keep.sum()), n_drop=int(dr.sum()),
                max_v_drop=float(v[dr].max()) if dr.any() else 0.0,
                max_theta_drop=float(th[dr].max()) if dr.any() else 0.0,
                max_rate_keep=float((Cv * d[keep]).max()) if keep.any() else 0.0)
    return keep, info


# ============================================================
# φ-帧统一 RHS 内核（t73：mixed 单原子/两体同一实现——t66 语义收敛）
# 输入 = pack_rhs 的输出数组（iis/jjs/vals/dDs 按 a<b 单侧打包，内核厄米
# 补齐反向）+ 逐波时变 AC 对角（ac_base (n_ac, dim) + ac_env_tab）。
# is_mw[w]=1 的波带 cos(C·w·t) 载波；is_mw=0 且 wcos=0 为 V_dd 类恒耦合
# （env=恒 1）。gamma 为扩展空间长度 NonHermitian 对角（MHz）。
# ============================================================
from numba import njit as _njit

@_njit(cache=True, nogil=True)
def rhs_phi_pairs(t, phi, iis, jjs, vals, dDs, is_mw, wcos, n_pairs_w,
                  env_tab, env_dt, has_ph, ph_tab, Cv, gamma=None,
                  ac_base=None, ac_env_tab=None, ac_env_dt=1.0):
    """φ-帧 RHS：−iC·[光场 env·Kc∘e^{iC·ΔD·t} + 微波 env·cos(φ)·A∘e^{iC·ΔD·t}
    + V_dd 当场]·φ + 逐波时变 AC 对角 −iC·Σ shifts·env²·φ + 耗散对角。

    微波载波（t79）：`has_ph[w]=1` 时取查表插值的 `cos(φ(t))`（任意相位，
    与 hard 的 cos(φ_f(t)) 同构）；否则取解析的 `cos(Cv·wcos[w]·t)`
    （逐位保持旧行为——线性相位下两者等价）。相位与包络同时间基，故沿用
    同一个 i0/f。
    """
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
            if has_ph[w]:
                phv = ph_tab[w, i0] * (1.0 - f) + ph_tab[w, i0 + 1] * f
                cw = envv * np.cos(phv)
            else:
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
    # 逐波时变 AC 对角 −iC·Σ_w shifts_w·env_w(t)²∘φ（env 平方 = AC∝光强；
    # K0 为纯 E——位移全量由 RHS 承担，t62e/t62f 定案）
    if ac_base is not None and ac_env_tab is not None:
        n_ac = ac_base.shape[0]
        for wi in range(n_ac):
            ei = t / ac_env_dt
            i0 = int(ei)
            if i0 >= ac_env_tab.shape[1] - 1:
                i0 = ac_env_tab.shape[1] - 2
            f = ei - i0
            envv = (ac_env_tab[wi, i0] * (1.0 - f)
                    + ac_env_tab[wi, i0 + 1] * f)
            dev = envv * envv
            if dev != 0.0:
                for k in range(phi.shape[0]):
                    v = ac_base[wi, k]
                    if v != 0.0:
                        res[k] += -1j * Cv * dev * v * phi[k]
    if gamma is not None:
        for k in range(phi.shape[0]):
            if gamma[k] != 0.0:
                res[k] -= 0.5 * Cv * gamma[k] * phi[k]
    return res


# ============================================================
# mixed 引擎的 DOP853 驱动（薄包装 —— 唯一目的：让磁盘缓存真正命中）
# ============================================================
# 背景：`_dop853_driver` 把 rhs 当参数，签名里因此带 numba 的 Dispatcher
# 类型；那是 WeakType，缓存键取的是 weakref（等价于对象 id），**每个进程
# 都不同** → 磁盘缓存永远查不中：每次启动重编 ~15s，还把新条目写进索引
# （实测 375 个文件 / 100 MB），且保存索引时若旧条目的引用对象已被 GC，
# 会抛 `ReferenceError: underlying object has vanished`，把整轮演化搞失败。
#
# 本包装把 rhs 静态绑定为模块级常量（不再出现在签名里）→ 键稳定。
# 同时 dop853 三个内核都标了 `inline='always'`：内联后包装的编译产物里
# 对 rhs 是**直接全局调用**而不是"把 jit 函数当实参传递"——后者会把
# 包装标记为 has_dynamic_globals 而**完全拒绝缓存**（numba 实测警告）。
# 三个内核因此不再产生自己的缓存条目，全部并入本包装的条目。
# 数值上只是把同一批实参原样转发（附加参数走元组），逐位不变。
from .dop853 import _dop853_driver          # noqa: E402  （系数已上收 dop853，无环）


@_njit(cache=True, nogil=True)
def dop853_driver_mixed(y0, t_eval, T, rtol, atol, max_step, prog, stat,
                        dt_hist, *rargs):
    """mixed 引擎（单/两体共用）：静态绑定 rhs_phi_pairs 的 DOP853 驱动。"""
    return _dop853_driver(rhs_phi_pairs, y0, t_eval, T, rtol, atol,
                          max_step, prog, stat, dt_hist, rargs)
