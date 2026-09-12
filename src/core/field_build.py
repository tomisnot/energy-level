# -*- coding: utf-8 -*-
"""场构造：field / group_field / real_field / build_fields_exact + 助手。

任意波形（ArbitraryWave，无载频）的场构造**不在本模块**：见
`evolve_twofield.arbitrary_mw_group()`（Hilbert 分解 env/φ → 两场 meta →
hard / mixed 的 cos(φ_f(t)) 任意相位通道）。"""
import numpy as np

from . import static_atom as core
from .atomic_levels import square

from .evolve_util import _POL_Q
from . import params as _P
# AC_NEAR_CUTOFF_MHZ 直连 params 单源（勿经 static_atom 转发）
# 上态总衰变率缓存（(n,l,j) → Γ MHz；远失谐散射损失用）
_lifetime_cache = {}


def _env_samples(waveform, t_end):
    """采样包络判断恒常性。返回 (mean, is_constant)。"""
    ts = np.linspace(0.0, float(t_end), _P.get('_ENV_SAMPLES'))
    vs = np.asarray(waveform(ts), float)
    return float(vs.mean()), bool(np.ptp(vs) < 1e-9)


# ---------------- 能级 ----------------


def _floquet_ac_static(cache, states, freq_mhz, intensity_mwcm2, pol,
                       B_gauss, n_photon=None, sos_n_window=None):
    """用静态页同款 Floquet 引擎（core.direct_ac_stark_shifts）计算
    动态 states 每个能级的 AC 位移（MHz，峰值强度）。

    - 基组：由 states 推断的组 + 全部偶极伙伴（window 由面板/全局配置）；
    - 不分近/远失谐：近共振 dressed 与远失谐反旋项统一对角化；
    - mJ 态直接取对应键的位移；F 态（低激发超精细）= Σ|amps|²·shift（mJ 分量加权）。
    n_photon / sos_n_window 缺省（None）时读取面板全局配置（与静态页一致）。
    """
    from . import static_atom as _c
    groups, seen = [], set()
    for s in states:
        k = (s['n'], s['l'], s['j'])
        if k not in seen:
            seen.add(k)
            groups.append(dict(n=k[0], l=k[1], j=k[2]))
    cfg = dict(enabled=True, frequency_hz=float(freq_mhz) * 1e6,
               intensity=float(intensity_mwcm2), polarization=str(pol),
               elliptical={})
    shifts = _c.direct_ac_stark_shifts(cache, groups, cfg, float(B_gauss),
                                       n_photon=n_photon,
                                       sos_n_window=sos_n_window)
    mlut = {}
    for (n, l, j, qstr, _occ), v in shifts.items():
        mlut.setdefault((n, l, j), {})[qstr] = v
    ac = np.zeros(len(states))
    for i, s in enumerate(states):
        lut = mlut.get((s['n'], s['l'], s['j']), {})
        if s.get('kind') == 'F':
            amps = s.get('amps') or {}
            tot = 0.0
            for mj, am in amps.items():
                v = lut.get(f'm_J={float(mj):+.1f}')
                if v is not None:
                    tot += float(abs(am) ** 2) * v
            ac[i] = tot
        else:
            m = float(s.get('m', s.get('mj', 0.0)))
            ac[i] = lut.get(f'm_J={m:+.1f}', 0.0)
    return ac


# ---------------- 能级 ----------------


def _sublevel_row(s):
    """states → core.dipole_amplitude 的子能级行（mJ/F 混合耦合矩阵元）。"""
    return dict(n=s['n'], l=s['l'], j=s['j'], kind=s['kind'],
                mj=s.get('mj'), mF=s.get('mF'), amps=s.get('amps'))


# ---------------- 光场 ----------------


def _field_geometry(states, f):
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
                    cache, _sublevel_row(states[i]), _sublevel_row(states[j]),
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
            ri = _sublevel_row(states[i])
            for j in range(i + 1, N):
                rj = _sublevel_row(states[j])
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
            d = core.dipole_amplitude(cache, _sublevel_row(states[i]),
                                      _sublevel_row(states[j]), q)
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
    if nu_max > _P.get('MW_MAX_MHZ'):
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
                'waveforms.build_fields_exact 构造）——不消费 real_field/'
                'multi 场（其近/远分流与静态 AC 注入与第一性模型冲突）。')
        A = np.asarray(f['A'], float)
        if N is None:
            N = A.shape[0]
        if A.shape != (N, N):
            raise ValueError('exact 场 A 矩阵维度不一致')
    return N


def _group_by_freq(fields):
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

    omega0_mhz：峰值 Rabi（MHz）；waveform：t→[0,1] 包络（缺省方波全程）。
    detune_mhz：失谐（相对 E_e−E_g，正=蓝）。
    """
    return dict(kind='pair', g=int(g), e=int(e), omega0_mhz=float(omega0_mhz),
                waveform=waveform or square(0.0, 1e12),
                detune_mhz=float(detune_mhz))




def group_field(cache, states, group_a, group_b, pol='π', omega0_mhz=1.0,
                waveform=None, detune_mhz=0.0):
    """组驱动：复用 core 偶极矩阵元构造耦合（选择定则 + 相对 Rabi 强度）。

    group_a/group_b: dict(n,l,j)。A 各 (m_g,m_e) 耦合 = |dme|/max|dme|，
    即最强耦合处 Rabi=Ω0（相对强度来自 core 跃迁矩阵元，与静态无关）。
    """
    return dict(kind='group', cache=cache, states=states,
                group_a=dict(group_a), group_b=dict(group_b), pol=pol,
                omega0_mhz=float(omega0_mhz),
                waveform=waveform or square(0.0, 1e12),
                detune_mhz=float(detune_mhz))




def real_field(cache, states, freq_mhz, intensity_mwcm2, pol='π',
               waveform=None, near_cutoff_mhz=None, return_scatter=False,
               B_gauss=None, n_photon=None, sos_n_window=None):
    """真实光场（频率+光强+偏振）→ (field_dict 或 None, ac_shift_mhz)。

    以单个能级为单位遍历基组内所有态对，按自适应失谐判据分流：
      - 近共振（|Δ| < max(near_cutoff, Ω)）→ 合并进**单个多对
        驱动场**（A 直接含各对真实 Rabi Ω_ij，统一 ω_L = 光频 f；
        RWA 下等价于多个 pair，但 H 组装一次完成，性能快 ~N_对 倍）；
      - 远失谐 →（仅 B_gauss=None 的回退路径）SOS 二阶 AC 位移
        （加进 H0 对角），驱动对绝不重复计 AC。
    失谐基准 = 裸能级差（超精细+Zeeman，不含 AC），无歧义。

    **AC 位移后端（B_gauss 提供时）**：统一由 core.direct_ac_stark_shifts
    （Floquet 直接对角化，含全部偶极伙伴基组、近共振 dressed 与远失谐
    反旋项、n_photon 阶光子块）计算，**不分近/远**（优于 far_sos 逐对求和——
    后者受基组范围剧烈影响）。F 态（低激发超精细）
    位移 = 按 mJ 分量 |amps|² 加权投影。近共振 Rabi 驱动 A 矩阵仍按
    near_cutoff 筛选（RK4 步长的 RWA 硬约束，与位移无关）。

    near_cutoff_mhz：近/远分流边界（MHz），缺省 = params.AC_NEAR_CUTOFF_MHZ
    （5 GHz 保守地板：覆盖超精细/精细结构流形，宁多算相干）。协议场景
    （如 STIRAP 想只相干驱动目标对、把 ~150 MHz 外的流形其余态作远失谐
    AC 处理）可显式调小（如 25 MHz）；边界仍随 Rabi 走 max(cutoff, Ω)。

    **统一场算符**（Pulser 范式）：M = A + diag(ac_shift) 一个矩阵——
    Rabi 耦合元与对角位移元同属该场的效应，H(t) += env(t)·M。
    返回 (field, ac_shift_mhz[N])：field['M'] 为统一算符，
    field['A']/field['ac_shift'] 保留元数据（诊断/ip4 稀疏优化用）；
    无近共振对时 field=None（ac 仍返回，由调用方包装成纯位移场）。

    **远失谐散射损失**（return_scatter=True 时第 3 返回 scatter[N]，MHz）：
    失谐光虽远仍以散射率 σ=Ω²Γ_up/(4Δ²) 把源态布居打到上态，上态
    （短寿命低激发态如 6P）自发辐射"回不来"→ 源态 depump/loss；
    Γ_up=1/τ（ARC 寿命，缓存）。演化引擎在 loss 输出中按
    ∫σ(t)·env²(t)·P_source(t)dt 计入（见 pair_evolve）。
    field['scatter'] 亦随场返回（无近共振时同样有效）。
    """
    from . import static_atom as _c
    q = {'π': 0, 'σ+': 1, 'σ-': -1}.get(pol, 0)
    N = len(states)
    E0_sq = _c.field_e0_sq(intensity_mwcm2)
    omega_rad = 2.0 * np.pi * freq_mhz * 1e6
    _env_sample = waveform if waveform is not None else square(0.0, 1e12)
    A = np.zeros((N, N))
    g_idx, e_idx = [], []
    ac = np.zeros(N)
    scatter = np.zeros(N)      # 远失谐光散射损失率（MHz，源态）
    lifetime_cache = _lifetime_cache
    for i in range(N):
        for j in range(i + 1, N):
            d = _c.dipole_amplitude(
                cache, _sublevel_row(states[i]), _sublevel_row(states[j]), q)
            if abs(d) < 1e-12:
                continue
            lo, hi = (i, j) if states[i]['E_mhz'] <= states[j]['E_mhz'] \
                else (j, i)
            nu_mhz = states[hi]['E_mhz'] - states[lo]['E_mhz']   # 裸差
            Delta = freq_mhz - nu_mhz
            Om = _c.rabi_mhz(abs(d), E0_sq)
            if abs(Delta) < max(near_cutoff_mhz or _P.get('AC_NEAR_CUTOFF_MHZ'), Om):
                # 近共振 → 多对驱动（A 存真实 Rabi，ω_L = 光频）
                A[lo, hi] = A[hi, lo] = Om
                g_idx.append(lo); e_idx.append(hi)
            else:
                # 远失谐：B_gauss is None（旧回退）→ SOS 累加进 ac；
                # 否则 ac 由循环后的 Floquet 统一计算（不分近/远）。
                if B_gauss is None:
                    nu_rad = 2.0 * np.pi * nu_mhz * 1e6
                    ac[lo] += _c.far_sos_shift_mhz(
                        d, E0_sq, omega_rad, nu_rad)
                    ac[hi] += _c.far_sos_shift_mhz(
                        d, E0_sq, omega_rad, -nu_rad)
                # 补充：**off-resonant 散射损失**（用户报 exchange depump 缺失）。
                # 失谐光虽远，仍以散射率 σ=Ω²Γ_up/(4Δ²) 把源态布居打到上态，
                # 上态（如 6P）短寿命自发辐射"回不来"→ 源态 depump/loss。
                # Γ_up = 1/τ（ARC 寿命，MHz）；仅 Δ≫Γ 的洛伦兹尾（这里 Ω²/Δ² 可控）。
                keyh = (int(states[hi]['n']), int(states[hi]['l']),
                        float(states[hi]['j']))
                gam = lifetime_cache.get(keyh)
                if gam is None:
                    try:
                        tau = cache.atom.getStateLifetime(
                            keyh[0], keyh[1], keyh[2], temperature=0.0)
                        gam = (1e-6 / float(tau)) if tau else 0.0
                    except Exception:
                        gam = 0.0
                    lifetime_cache[keyh] = gam
                if gam > 0.0:
                    scatter[lo] += (Om * Om * gam
                                    / (4.0 * max(Delta * Delta, 1e-6)))
    # ---- AC 位移：统一 Floquet 后端（B_gauss 提供时，不分近/远）----
    if B_gauss is not None:
        try:
            ac = np.asarray(_floquet_ac_static(
                cache, states, freq_mhz, float(intensity_mwcm2), pol,
                float(B_gauss), n_photon=n_photon,
                sos_n_window=sos_n_window), float)
        except Exception:
            # Floquet 失败 → 回退 far_sos 路径（递归 B=None，保稳健）
            _, ac = real_field(cache, states, freq_mhz, intensity_mwcm2,
                               pol, waveform, near_cutoff_mhz) or (None, ac)
            ac = np.asarray(ac or np.zeros(len(states)), float)
    if not g_idx:
        if return_scatter:
            return None, ac, scatter
        return None, ac
    M = A + np.diag(ac)
    fld = {'kind': 'multi', 'M': M, 'A': A, 'ac_shift': ac,
           'g_idx': g_idx, 'e_idx': e_idx,
           'omega0_mhz': 1.0, 'omega_L_mhz': freq_mhz, 'waveform': _env_sample,
           'scatter': scatter}
    if return_scatter:
        return fld, ac, scatter
    return fld, ac





