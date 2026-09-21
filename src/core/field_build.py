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
_ENV_SAMPLES = Param(64, 'field_build 包络探针数；包络恒常判断采样数（粗粒度）', tier='internal')
# （原 AC_NEAR_CUTOFF_MHZ 的近/远分流随 real_field 一并删除）
# 上态总衰变率缓存（(n,l,j) → Γ MHz；远失谐散射损失用）
_lifetime_cache = {}


def env_samples(waveform, t_end):
    """采样包络判断恒常性。返回 (mean, is_constant)。"""
    ts = np.linspace(0.0, float(t_end), _ENV_SAMPLES)
    vs = np.asarray(waveform(ts), float)
    return float(vs.mean()), bool(np.ptp(vs) < 1e-9)


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




