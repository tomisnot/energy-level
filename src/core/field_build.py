# -*- coding: utf-8 -*-
"""场构造：field / build_fields_exact + 助手。

任意波形（ArbitraryWave，无载频）的场构造**不在本模块**：见
`evolve_twofield.arbitrary_mw_group()`（Hilbert 分解 env/φ → 两场 meta →
hard / mixed 的 cos(φ_f(t)) 任意相位通道）。"""
import numpy as np

from . import static_atom as core
from .atomic_levels import square

from .evolve_util import _POL_Q, resolve_state_idx

from .static_atom import MW_MAX_MHZ

from .field_loss import (FieldLossConfig, add_scatter_pair, photoion_rate_mhz,
                         TrapModel)
from .param import Param   # 就地参数声明（值+元数据同处；mecha.registry 扫出全表）


# ==== 本模块的参数（就地定义；Param 声明值+元数据同处，透明子类读点免改）====
_ENV_SAMPLES = Param(64, 'field_build 包络探针数；恒常判断的**均值**采样数（恒常与否由 env_span 决定，不由采样密度决定）', tier='internal')
_ENV_CONST_TOL = Param(1e-9, '包络恒常门；sup−inf ≤ 本值 → 判恒常（沿用原 np.ptp<1e-9 口径）', tier='internal')
_ENV_SPAN_FALLBACK_N = Param(2001, '包络变差兜底采样点数；**仅**用于没有解析 span 的裸 callable（Wave 包络走各自的精确 span）', tier='internal')
# （原 AC_NEAR_CUTOFF_MHZ 的近/远分流随 real_field 一并删除）
# 上态总衰变率缓存（(n,l,j) → Γ MHz；远失谐散射损失用）
_lifetime_cache = {}


def env_span(wf, t0, t1):
    """包络在**绝对时间** [t0,t1] 上的 (inf, sup)——判"是否恒常"的唯一入口。

    ## 为什么不能用固定点数采样（2026-09-22 修，见手册 §8.1）

    原先消费方（`sambe._env_all_constant` / `sambe._pair_propagate` /
    `env_samples`）用 `np.ptp(wf(np.linspace(0,T,64))) > 1e-9` 判恒常。**固定
    采样点对窄脉冲是盲的**：σ ≲ T/800 时整个脉冲落在采样点之间——实测
    T=620 ns / σ=0.5 ns 时 64 点 ptp=9.3e-22（< 1e-9）而**真实 ptp=1.0**。
    后果是两个引擎给出**方向相反**的错答案，且都静默：
      · sambe 判恒常 → 走"一次对角化"快路径、用**中点**包络 ⇒ 整窗按峰值 CW 算；
      · ip4  判恒常 → 取**均值**（≈0）⇒ 场完全消失。
    （`error_report['n_segments']=1` 看起来完全正常。）

    ## 修法取向：让波形数据层自己回答（而非加密采样）

    "是否恒常"是包络**自身**的性质，可以解析给出、与采样密度无关：
      · `GaussianEnv` 单峰 ⇒ 极值在两端或峰位（O(1)，精确）；
      · `SquareEnv` 分段单调 ⇒ 极值在折点（精确；升降沿重叠时补中点）；
      · `FileEnv` 分段线性 + 区外钳 0 ⇒ 极值在样品点或端点（精确）。
    取向 a（探针密度挂钩最窄特征宽度）仍需选一个"多密才够"的启发式阈值，且
    永远可能被更窄的特征骗过；故采用**取向 b**：`span()` 由各包络类实现（单一
    定义点），本函数只做"解析 → 兜底"的分发。

    分发顺序：① `wf.__self__` 是 `Wave` → `Wave.span`（带 offset，绝对时间）；
    ② `wf` 自带 `span`；③ 兜底密集采样（**不是判据**，只是没有解析答案时的近似）。
    """
    t0, t1 = float(t0), float(t1)
    self_obj = getattr(wf, '__self__', None)          # Wave.waveform 是绑定方法
    fn = getattr(self_obj, 'span', None)
    if callable(fn):
        r = fn(t0, t1)
        if r is not None:
            return float(r[0]), float(r[1])
    fn = getattr(wf, 'span', None)
    if callable(fn):
        r = fn(t0, t1)
        if r is not None:
            return float(r[0]), float(r[1])
    ts = np.linspace(t0, t1, int(_ENV_SPAN_FALLBACK_N))
    vs = np.asarray(wf(ts), float)
    if vs.size == 0:
        return 0.0, 0.0
    return float(vs.min()), float(vs.max())


def env_is_const(wf, t0, t1, tol=None):
    """包络在 [t0,t1] 上是否恒常（sup−inf ≤ tol，默认 `_ENV_CONST_TOL`）。"""
    tol = float(_ENV_CONST_TOL) if tol is None else float(tol)
    lo, hi = env_span(wf, t0, t1)
    return bool((hi - lo) <= tol)


def env_feature_ns(wf, T=None):
    """包络最窄变化特征时长（ns）——"积分器必须分辨的时间尺度"；不可得 ⇒ None。

    ① 解析（首选）：`Wave` 的包络自带 `dt_feature_ns()`（高斯 ⇒ σ；方波 ⇒ 沿时长；
       文件 ⇒ 最小样品间隔）。
    ② 兜底（裸 callable，如自检/脚本里直接传 lambda）：按 `_ENV_SPAN_FALLBACK_N`
       点采样求**非零支撑宽度** W（|env| > 1e-3·sup 的最窄连续段），返回 W/8
       ——对高斯 W≈7.4σ ⇒ W/8≈0.93σ，与解析值同量级。
       分辨率上限 = T/(N−1)：**窄于该分辨率的特征估不出来**（返回 None），此时
       请显式给积分器的 max_step，或改用 `Wave`（走解析路径）。
    """
    for obj in (getattr(wf, '__self__', None), wf):
        fn = getattr(obj, 'dt_feature_ns', None)
        if callable(fn):
            v = fn()
            if v is not None:
                return float(v)
    if T is None or float(T) <= 0.0:
        return None
    ts = np.linspace(0.0, float(T), int(_ENV_SPAN_FALLBACK_N))
    vs = np.abs(np.asarray(wf(ts), float))
    sup = float(vs.max()) if vs.size else 0.0
    if sup <= 0.0:
        return None
    m = vs > 1e-3 * sup
    if not m.any():
        return None
    idx = np.flatnonzero(m)
    brk = np.flatnonzero(np.diff(idx) > 1)
    starts = np.concatenate([[0], brk + 1])
    ends = np.concatenate([brk, [idx.size - 1]])
    dt = float(T) / max(len(ts) - 1, 1)
    widths = [(int(idx[e]) - int(idx[s]) + 1) * dt
              for s, e in zip(starts, ends)]
    return min(widths) / 8.0


def env_samples(waveform, t_end):
    """采样包络判断恒常性。返回 (mean, is_constant)。

    `mean` 仅供**已判恒常**的消费方（ip4 的常数包络快路径）当常数值，按
    `_ENV_SAMPLES` 点采样；`is_constant` 由 `env_span` **解析**决定（不靠采样
    密度）——原实现用同一批 64 点的 `np.ptp` 兼任两者，正是 §8.1 的事故点。
    """
    ts = np.linspace(0.0, float(t_end), _ENV_SAMPLES)
    vs = np.asarray(waveform(ts), float)
    return float(vs.mean()), env_is_const(waveform, 0.0, float(t_end))


_FEATURE_MAX_STEP_FRAC = Param(64, '窄特征步长保护门；T/最窄特征 > 本值 时才限制积分步长（深混叠区；既有宽脉冲场景为 no-op）', tier='internal')
_FEATURE_MAX_STEP_MULT = Param(4.0, '窄特征步长上限 = 本值 × 最窄特征时长（自适应积分器看得见该特征所需）', tier='internal')


def feature_max_step(waveforms, T):
    """由包络最窄特征时长导出的**积分步长上限**（ns）；无需保护 ⇒ None。

    动机（2026-09-22，与 `env_span` 同族、同一份事故）：自适应积分器（DOP853）
    的步长由**它看得见的**最快振荡决定。窄脉冲（σ ≪ 窗口）可能整个落在两步
    之间——更糟的是 `exp(−(t−c)²/2σ²)` 在 `|t−c| > ~19σ` 时**下溢到 0.0**，
    于是每次 RHS 求值包络都**恰为 0**，结果**逐位等于零场**且无任何报错。
    实测：`evolve_reference`（全库"权威锚"）在 T=620 ns / σ=0.5 ns 高斯下
    `max P_e = 0.000000e+00`（解析值 1.550e-5）。

    只在 `T/特征 > _FEATURE_MAX_STEP_FRAC`（深混叠区）时生效 ⇒ 对既有宽脉冲
    场景（如 σ=60 / T=400）是 **no-op**，不改变任何已验收数值。
    """
    T = float(T)
    if T <= 0.0:
        return None
    feats = [f for f in (env_feature_ns(wf, T) for wf in (waveforms or []))
             if f is not None and f > 0.0]
    if not feats:
        return None
    fmin = min(feats)
    if T / fmin <= float(_FEATURE_MAX_STEP_FRAC):
        return None
    return float(_FEATURE_MAX_STEP_MULT) * fmin


# ---------------- 能级 ----------------


# ---------------- 能级 ----------------


def sublevel_row(s):
    """states → core.dipole_amplitude 的子能级行（mJ/F 混合耦合矩阵元）。"""
    return dict(n=s['n'], l=s['l'], j=s['j'], kind=s['kind'],
                mj=s.get('mj'), mF=s.get('mF'), amps=s.get('amps'))


# ---------------- 光场 ----------------


def field_geometry(states, f):
    """解析光场 → dict(A, g_idx, e_idx, omega0_mhz, omega_L_mhz, waveform)。

    （ip4 族场格式的归一化与场构造同源。）
    """
    N = len(states)
    if f['kind'] == 'multi':
        # 多对驱动 + 对角位移统一算符：M = A + diag(ac)（Pulser 范式）。
        M = f.get('M')
        if M is None:                      # 旧格式兼容（只有 A 无 M）
            M = f['A'] + np.diag(np.asarray(f.get('ac_shift', np.zeros(N)), float))
        A = f.get('A')
        if A is None:
            A = M.copy()
            np.fill_diagonal(A, 0.0)
        ac = np.asarray(f.get('ac_shift', np.diag(M).real), float)
        return dict(M=M, A=A, ac_shift=ac,
                    g_idx=f['g_idx'], e_idx=f['e_idx'],
                    omega0_mhz=1.0, real_omega0_mhz=float(np.abs(A).max()),
                    omega_L_mhz=f['omega_L_mhz'],
                    waveform=f['waveform'])
    if f['kind'] == 'pair':
        g, e = int(f['g']), int(f['e'])
        Eg, Ee = states[g]['E_mhz'], states[e]['E_mhz']
        if Eg <= Ee:
            g_idx, e_idx = [g], [e]
        else:
            g_idx, e_idx = [e], [g]
            Eg, Ee = Ee, Eg
        A = np.zeros((N, N))
        A[g_idx[0], e_idx[0]] = A[e_idx[0], g_idx[0]] = 1.0
    else:
        cache = f['cache']
        ga, gb = f['group_a'], f['group_b']
        q = _POL_Q.get(f['pol'], 0)
        idx_a = [i for i, s in enumerate(states)
                 if (s['n'], s['l'], s['j']) == (ga['n'], ga['l'], ga['j'])]
        idx_b = [i for i, s in enumerate(states)
                 if (s['n'], s['l'], s['j']) == (gb['n'], gb['l'], gb['j'])]
        Ea = cache.E_mhz(int(ga['n']), int(ga['l']), float(ga['j']))
        Eb = cache.E_mhz(int(gb['n']), int(gb['l']), float(gb['j']))
        if Ea <= Eb:
            g_idx, e_idx, Eg, Ee = idx_a, idx_b, Ea, Eb
        else:
            g_idx, e_idx, Eg, Ee = idx_b, idx_a, Eb, Ea
        M = np.zeros((len(g_idx), len(e_idx)))
        for ia, i in enumerate(g_idx):
            for ib, j in enumerate(e_idx):
                M[ia, ib] = abs(core.dipole_amplitude(
                    cache, sublevel_row(states[i]), sublevel_row(states[j]),
                    q))
        mx = M.max()
        if mx > 0:
            M = M / mx
        A = np.zeros((N, N))
        for ia, i in enumerate(g_idx):
            for ib, j in enumerate(e_idx):
                A[i, j] = A[j, i] = M[ia, ib]
    omega_L = (Ee - Eg) + float(f['detune_mhz'])
    return dict(M=A.copy(), A=A, ac_shift=np.zeros(N),
                g_idx=g_idx, e_idx=e_idx, omega0_mhz=f['omega0_mhz'],
                omega_L_mhz=omega_L, waveform=f['waveform'])


def build_fields_exact(cache, states, waves):
    """波形列表 → 第一性原理场列表（每波形一个全偶极耦合算符）。

    每个波形 (freq, intensity, pol, envelope) → A[i,j] = 峰值光强下该对的
    Rabi 频率 Ω_ij = core.rabi_mhz（MHz，Ω/2π 周频约定：Sambe 子空间耦合取
    Ω/2，共振布居振荡 sin²(π·Ω_MHz·t_µs)，π 脉冲 = 1/(2Ω) µs，与实验界
    面 Rabi 定义一致），含选择定则；**全部**偶极允许对都进耦合矩阵——
    远失谐对的效应由精确传播自动给出，不存在近/远分流边界，也不注入
    任何静态能移。
    """
    out = []
    N = len(states)
    for w in waves:
        if getattr(w, 'kind', None) == 'arb':
            raise ValueError(
                'build_fields_exact（sambe / mixed 的 kind="exact" 场）不支持'
                '任意波形——其扩展空间需要**单一载频**，而任意波形无载频概念。'
                '请把该文件按「光强包络」导入（Wave + 载频，三引擎全支持），'
                '或改用「完整驱动」（经 arbitrary_mw_group 走 hard 任意相位路径）。')
        # 偏振 → 球基复系数。旧版只取单个 q，于是 '椭圆偏' 落进
        # `_POL_Q.get(..., 0)` 的默认值 0 = **静默按 π 算**（错的）；现在
        # 椭圆偏走 (c₊d₊ + c₀d₀ + c₋d₋) 的相干幅度，与光场行同一套系数。
        ell = getattr(w, 'elliptical', None)
        if w.pol == '椭圆偏':
            if not ell:
                # 缺参数时补齐为 (0,0,0) = x 线偏。旧版是 q 默认 0 → 静默按
                # **π** 算（错的）；这里走真正的椭偏系数路径。**不 raise**：
                # 本函数会在 Qt 槽内被调用，异常 = qFatal = 进程崩溃。
                ell = dict(alpha_deg=0.0, theta_deg=0.0, eps_deg=0.0)
            ell = dict(ell)
            cp, c0, cm = core.elliptical_coeffs(
                float(ell.get('alpha_deg', 0.0)),
                float(ell.get('theta_deg', 0.0)),
                float(ell.get('eps_deg', 0.0)))
            _qs = ((1, cp), (0, c0), (-1, cm))
        else:
            _qs = ((_POL_Q.get(w.pol, 0), 1.0 + 0.0j),)
        E0_sq = core.field_e0_sq(float(w.intensity_mwcm2))
        A = np.zeros((N, N))
        for i in range(N):
            ri = sublevel_row(states[i])
            for j in range(i + 1, N):
                rj = sublevel_row(states[j])
                d = 0.0 + 0.0j
                for _q, _c in _qs:
                    if abs(_c) > 1e-12:
                        d += _c * core.dipole_amplitude(cache, ri, rj, _q)
                if abs(d) < 1e-12:
                    continue
                A[i, j] = A[j, i] = core.rabi_mhz(abs(d), E0_sq)
        if not np.any(A != 0.0):
            continue                       # 该偏振下无任何耦合 → 无效场
        out.append({'kind': 'exact', 'A': A,
                    'omega_L_mhz': float(w.freq_mhz),
                    'waveform': w.waveform,
                    'label': getattr(w, 'name', '波'),
                    'intensity_mwcm2': float(w.intensity_mwcm2),
                    'pol': w.pol, 'elliptical': ell})
    return out


# =====================================================================
# 场致损耗装配（占被删 real_field 的 scatter 位；修正旧 F3 取反）
# =====================================================================
def _state_gamma_leaving_mhz(cache, s):
    """态的**总**离开基组率（MHz）= ARC 1/τ（对全部下能级求和）。

    τ 查不到 / 无限（基态）→ 0（无散射损失可引）。与 auto_decays 同一换算。
    """
    try:
        tau = cache.atom.getStateLifetime(int(s['n']), int(s['l']), float(s['j']))
    except Exception:                              # noqa: BLE001
        return 0.0
    if tau is None or not np.isfinite(float(tau)) or float(tau) <= 0.0:
        return 0.0
    return 1e-6 / float(tau)                       # s⁻¹ → MHz


def build_field_loss(cache, states, fields, decays, spec):
    """远失谐光 → 逐波 env²×逐态时变损耗（`FieldLossConfig`）。opt-in：无 spec → None。

    自动逐 far-detuned 对 (i,j)（由 `fields` 的耦合矩阵 A 与载频推 Ω/Δ）：
      • 近共振剔除：|Δ| < near_cutoff → 走相干动力学，`add_scatter_pair` 整对跳过；
      • 双计规则：伙伴已在 `decays` 中配了自发辐射通道 ⇒ 其 out 记 0
        （该流失已由相干 + 伙伴衰变承担），否则取 ARC 1/τ；
      • 可选光电离：`spec['photoion'] = {'sigma_mb_by_idx': {i: σ}, 'lam_m': ...}`。

    返回 `FieldLossConfig`（无有效通道 → None）。`spec['trap']` 由调用方自行
    交给 `field_loss.TrapModel`（加热逃逸，引擎侧 H3 才消费）。
    """
    if not spec:
        return None
    spec = dict(spec)
    near = float(spec.get('near_cutoff_mhz', 5000.0))
    N = len(states)
    cfg = FieldLossConfig(N)
    dec_from = {int(resolve_state_idx(states, d['from']))
                for d in (decays or []) if d.get('to') is None}

    def out_rate(k):                              # 伙伴离基组率（双计后）
        return 0.0 if k in dec_from else _state_gamma_leaving_mhz(cache, states[k])

    for f in (fields or []):
        A = np.asarray(f.get('A'), float)
        if A.shape != (N, N):
            continue
        wL = float(f.get('omega_L_mhz', 0.0))
        env = f.get('waveform')
        if env is None:
            continue
        for i in range(N):
            for j in range(i + 1, N):
                Om = abs(float(A[i, j]))
                if Om <= 0.0:
                    continue
                det = abs(float(states[j]['E_mhz'])
                          - float(states[i]['E_mhz'])) - wL
                add_scatter_pair(cfg, i, j, Om_mhz=Om, det_mhz=det,
                                 gam_out_i_mhz=out_rate(i),
                                 gam_out_j_mhz=out_rate(j),
                                 env_fn=env, near_cutoff_mhz=near)
    pi = spec.get('photoion')
    if pi:
        lam = float(pi.get('lam_m', 1064e-9))
        by_idx = pi.get('sigma_mb_by_idx') or {}
        for f in (fields or []):
            I = float(f.get('intensity_mwcm2', 0.0))
            env = f.get('waveform')
            if env is None or I <= 0.0:
                continue
            for k, sigma in by_idx.items():
                rate = photoion_rate_mhz(I, lam, float(sigma))
                if rate > 0.0:
                    cfg.add_photoion(int(k), env, rate)
    if cfg.is_empty():
        return None
    tp = spec.get('trap')
    if tp:
        dk = tp.get('depth_K')
        cfg.trap = TrapModel(
            float(tp.get('lam_m', 1013.8e-9)),
            None if dk is None else float(dk),
            kappa_geo=float(tp.get('kappa_geo', 2.0)),
            T0_K=float(tp.get('T0_K', 0.0)),
            cool_rate_ns=float(tp.get('cool_rate_ns', 0.0)),
            mode=str(tp.get('mode', 'budget')))
    return cfg


def arb_applicability(cache, states, wave, t_end_ns=None):
    """任意波形（ArbitraryWave）对当前基组的适用性体检 → 警告文本列表。

    任意波形经 arbitrary_mw_group 转成微波组（Hilbert 分解 env·cosφ，
    无载频），三个已知陷阱：
      ① 文件带宽 f_max 必须覆盖目标跃迁频率——否则**静默零效果**（实测：基带
         包络驱动 17 GHz 跃迁，转移布居恒为 0；采样不足同样归零，不是轻微失真）；
      ② 基组若含光学流形（跃迁 ≫ 微波分界），该路径**不参与光场旋转** → RHS 带
         THz 相位 → DOP853 步长被钉死（实测 2 ns 演化 >60 s 未完成）；
      ③ 超出引擎查表带宽的部分会被折叠（`f_max_mhz(t_end)` 已含该上限）。
    返回 [] = 无警告。供 GUI 运行前提示 / 脚本自检复用。
    """
    warns = []
    if getattr(wave, 'kind', None) != 'arb':
        return warns
    q = _POL_Q.get(wave.pol, 0)
    nus = []
    n = len(states)
    for i in range(n):
        for j in range(i + 1, n):
            d = core.dipole_amplitude(cache, sublevel_row(states[i]),
                                      sublevel_row(states[j]), q)
            if abs(d) > 1e-12:
                nus.append(abs(float(states[j]['E_mhz'])
                               - float(states[i]['E_mhz'])))
    if not nus:
        warns.append('该偏振下基组内无任何偶极允许跃迁——'
                     '任意波形不会驱动任何态')
        return warns
    fmax = float(wave.f_max_mhz(t_end_ns))
    nu_min, nu_max = min(nus), max(nus)
    if fmax > 0.0 and fmax < nu_min:
        warns.append(
            f'文件带宽 f_max≈{fmax:.3g} MHz 低于基组最小偶极跃迁 '
            f'{nu_min:.3g} MHz → 驱动不到任何跃迁（效果为零）。'
            '若这份文件其实是"光强形状"，请按「光强包络」导入'
            '（保留载频，sambe/mixed/hard 全支持）。')
    if nu_max > MW_MAX_MHZ:
        warns.append(
            f'基组含光学流形（最高偶极跃迁 {nu_max / 1e6:.1f} THz）：'
            '任意波形不参与光场旋转 → 积分器会被 THz 相位钉死'
            '（实测 2 ns 演化 >60 s 未完成）。光学驱动请改用光场波形。')
    elif fmax > 0.0 and fmax < nu_max:
        warns.append(f'文件带宽 f_max≈{fmax:.3g} MHz 不足以驱动 '
                     f'{nu_max:.3g} MHz 的跃迁（该跃迁将不被驱动）。')
    return warns


def _validate_fields(fields):
    N = None
    for f in fields:
        if (f or {}).get('kind') != 'exact':
            raise ValueError(
                "method='exact' 要求场为 kind='exact'（由 "
                'build_fields_exact 构造）——不消费 multi 场（其近/远分流'
                '与静态 AC 注入与第一性模型冲突）。')
        A = np.asarray(f['A'], float)
        if N is None:
            N = A.shape[0]
        if A.shape != (N, N):
            raise ValueError('exact 场 A 矩阵维度不一致')
    return N


def group_by_freq(fields):
    """相异载频分组（相对差 ≤1e-9 视为同频，共享一个光子维）。"""
    groups = []
    for f in fields:
        w = float(f['omega_L_mhz'])
        hit = None
        for g in groups:
            if abs(g['omega_mhz'] - w) <= 1e-9 * max(abs(w), 1.0):
                hit = g
                break
        if hit is None:
            hit = dict(omega_mhz=w, As=[], envs=[], labels=[])
            groups.append(hit)
        hit['As'].append(np.asarray(f['A'], float))
        hit['envs'].append(f['waveform'])
        hit['labels'].append(f.get('label', ''))
    return groups


def field(g, e, omega0_mhz=1.0, waveform=None, detune_mhz=0.0):
    """单对驱动（最灵活）：g/e 为 states 索引，A 为 0/1 单耦合。

    **这是 `evolve_rwa`（ip4）与 `evolve_floquet`（封存权威）的输入格式**：
    产品路径（`runspec` → `build_fields_exact`）不用它，但**判据用它构造参考输入**
    ——不是死代码。`sambe` 的 exact 路径要求 `kind="exact"` 是**那个引擎的入参要求**，
    与这里的 `kind="pair"` 是两种并列的场表示。

    omega0_mhz：峰值 Rabi（MHz）；waveform：t→[0,1] 包络（缺省方波全程）。
    detune_mhz：失谐（相对 E_e−E_g，正=蓝）。
    """
    return dict(kind='pair', g=int(g), e=int(e), omega0_mhz=float(omega0_mhz),
                waveform=waveform or square(0.0, 1e12),
                detune_mhz=float(detune_mhz))




