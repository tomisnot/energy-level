# -*- coding: utf-8 -*-
"""单原子动态模拟引擎（evolve.py）：纯含时薛定谔方程演化，独立模块，高性能。

设计原则（与静态计算 / driver 完全解耦）：
- 输入：所有考虑的能级（能量 + 量子数）、任意多组随时间变化的光场、初态
- 输出：各态布居（及末态复振幅）随时间演化
- **不算斯塔克能移**：所有光都是动态的，不存在精确能移；H(t) 里只有用户给的
  能级能量 + 时变光耦合项（RWA 旋转系修正除外，那是坐标变换不是能移）
- 复用 core 的偶极跃迁矩阵元（dme / dipole_amplitude）构造光耦合
  （选择定则 + 相对 Rabi 强度），与静态与否无关
- 不依赖 driver / pair_core（无 build_system / _single_operator）

物理模型（单原子，多驱动）：
    H(t) = H0 − Σ_i ref_i·P_i + Σ_k [ (Ω_k(t)/2)·A_k ] + BS        （RWA，pwc）
    H(t) = H0 + Σ_k Ω_k(t)·cos(ω_{L,k} t)·A_k                       （非 RWA 精确）
- H0       ：对角 = 各能级能量 E_mhz（用户输入，可含 Zeeman 等任意静态）
- ref_i    ：每个态的旋转参考频率（统一级联旋转系）——从最低能态 ref=0 沿
  驱动图链式传播 ref[e]=ref[g]+ω_L，正确处理**共享能级**的多驱动
- 光场 k   ：驱动 |g_k⟩↔|e_k⟩，耦合矩阵 A_k（core 偶极构造 或 用户直给 0/1 单对）
- Ω_k(t)   ：Rabi = Ω0_k · waveform_k(t)（MHz；waveform 为 0~1 包络 callable）
- ω_{L,k}  ：光频率 = (E_e − E_g) + Δ_k；Δ_k 失谐，正 = 蓝失谐

============================================================
演化算法（三个各有特色，精度/时间实测见 checks/mw_bench_firstprinciples.py）：
------------------------------------------------------------
1) method='floquet'（默认 auto 首选）：精确 Floquet 传播子。
   适用：**常数包络 + 单一频率**（CW / 方波脉冲 / real_field 多对驱动）。
   只精确积分**一个周期**（中点法 n_sub 步）→ U(T) 本征分解（准能量 ε_k）
   → 任意时刻 ψ(t)=U(τ)·W·diag(e^{-iεnT})·W†·ψ0。
   对周期驱动是 **TDSE 精确解**（无 RWA、无光子块截断，反旋转项/Bloch–
   Siegert 全部自动包含），**成本不随模拟时长增长**。
   实测：N=20、ω=16.8 GHz、400 ns → 0.26 s，机器精度
   （对比：solve_ivp 非 RWA 334 s 且误差 5.6e-4）。

2) method='ip4'：相互作用绘景 RK4（第一性原理，任意时变包络）。
   光频（Ω/ωL < 1e-4）→ RWA（旋转系耦合，反旋转项误差 ~1e-7 忽略）；
   微波等强驱动（Ω/ωL ≥ 1e-4）→ **非 RWA**（保留 cos(ω t) 反旋转项）。
   旋转参考：RWA 场级联传播；非 RWA 场 e 跟随 g 共享参考（同 full_fast）。
   相互作用绘景（H0=diag(E−ref) 相位精确除去）→ 每步误差 ~(Ω·dt)³，
   dt = 最快非 RWA 载波周期/ip4_frac（默认 20 → 误差 ~3e-5，~20-30 s；
   frac=40 → ~1e-6，~50-60 s）。
   全 RWA 且包络恒常 → 一次对角化精确解（秒出）。
   实测：N=20 高斯脉冲 400 ns → 29 s（T/20）/ 54 s（T/40），误差 3e-5 / 1e-6。

3) method='pwc'：RWA 分段常数（**最快预览**，精度 ~Ω/ωL）。
   每段 [t_m,t_{m+1}] H 取中点 → exp(−iHΔt)（eigh），CPU 逐段 /
   torch GPU 批量（pops 全程 GPU 累积，仅一次同步）。
   微波强驱动下 RWA 误差可达 ~3%（实测），交互预览够用，精算用 floquet/ip4。

method='auto'（默认）：常数包络 + 单频 → 'floquet'；否则 → 'ip4'。
   （pwc 是唯一 RWA 方法，供显式快速预览 / 对比用，不自动选。）

参考积分器（内部保留，仅 checks 交叉验证用，不作为可选 method）：
  _evolve_full / _evolve_full_hybrid —— solve_ivp 非 RWA 第一性原理（慢）。

单位：能级/耦合 MHz，时间 ns，角频率 rad/ns。
"""

import numpy as np
import scipy.linalg

from . import core

# numba JIT（可选）：加速 ip4 主循环（强耦合/长时窗时步数可达 10^6~10^7）
try:
    from numba import njit as _njit
except Exception:
    _njit = None

_CONV = 2 * np.pi * 1e-3          # MHz·ns → rad
_POL_Q = {'π': 0, 'σ+': +1, 'σ-': -1}

# 光频/微波分界：Ω/ωL < 此值 → RWA（反旋转误差 ~1e-7）；≥ → 非 RWA
IP4_RWA_THRESH = 1e-4
# 包络恒常判定采样点数
_ENV_SAMPLES = 64


# ---------------- 波形（0~1 包络工厂；用户 UI 可自定义 waveform callable） ----------------
def gaussian(t0, sigma):
    """高斯包络 exp(−(t−t0)²/2σ²)；返回 callable(t_ns array)->array。"""
    def wf(t):
        t = np.asarray(t, float)
        return np.exp(-((t - t0) ** 2) / (2.0 * sigma ** 2))
    return wf


def square(t0, t1):
    """方波包络（t0≤t≤t1 为 1，其余 0）。"""
    def wf(t):
        t = np.asarray(t, float)
        return ((t >= t0) & (t <= t1)).astype(float)
    return wf


def step(t0):
    """阶跃包络（t≥t0 为 1）。"""
    def wf(t):
        t = np.asarray(t, float)
        return (t >= t0).astype(float)
    return wf


def _env_samples(waveform, t_end):
    """采样包络判断恒常性。返回 (mean, is_constant)。"""
    ts = np.linspace(0.0, float(t_end), _ENV_SAMPLES)
    vs = np.asarray(waveform(ts), float)
    return float(vs.mean()), bool(np.ptp(vs) < 1e-9)


# ---------------- 能级 ----------------
def make_levels(cache, specs):
    """能级规格列表 → states（支持超精细基组）。

    specs 每项 dict(n,l,j) + 子能级标识：
      - kind='mJ', m           → 磁子能级（里德堡态）
      - kind='F', F, mF, amps[, E_hf]
                               → 超精细态（amps = {m_J: 复振幅}，来自
                                 core.get_hyperfine_levels；E_hf 为其相对能）
      - 缺省 kind → 按 mJ 处理（向后兼容）
    E_mhz = 组中心（可显式覆盖） + E_hf。
    返回 states: [{n,l,j,kind,F,mF,mj,m,amps,E_mhz,label,index}]。
    """
    states = []
    for i, s in enumerate(specs):
        n, l, j = int(s['n']), int(s['l']), float(s['j'])
        E0 = float(s.get('E_mhz', cache.E_mhz(n, l, j)))
        E_hf = float(s.get('E_hf', 0.0))
        E = E0 + E_hf
        if s.get('kind') == 'F':
            F, mF = float(s['F']), float(s['mF'])
            label = s.get(
                'label',
                f'{core.auto_label(n, l, j)} F={F:.1f}, m_F={mF:+.1f}')
            states.append(dict(
                n=n, l=l, j=j, kind='F', F=F, mF=mF, mj=None, m=mF,
                amps=s.get('amps'), E_mhz=E, label=label, index=i))
        else:
            m = float(s['m'])
            label = s.get('label',
                          f'{core.auto_label(n, l, j)} m_J={m:+.1f}')
            states.append(dict(
                n=n, l=l, j=j, kind='mJ', F=None, mF=None, mj=m, m=m,
                amps=None, E_mhz=E, label=label, index=i))
    return states


def _sublevel_row(s):
    """states → core.dipole_amplitude 的子能级行（mJ/F 混合耦合矩阵元）。"""
    return dict(n=s['n'], l=s['l'], j=s['j'], kind=s['kind'],
                mj=s.get('mj'), mF=s.get('mF'), amps=s.get('amps'))


# ---------------- 光场 ----------------
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
               waveform=None):
    """真实光场（频率+光强+偏振）→ (field_dict 或 None, ac_shift_mhz)。

    以单个能级为单位遍历基组内所有态对，按自适应失谐判据分流：
      - 近共振（|Δ| < max(AC_NEAR_CUTOFF_MHZ, Ω)）→ 合并进**单个多对
        驱动场**（A 直接含各对真实 Rabi Ω_ij，统一 ω_L = 光频 f；
        RWA 下等价于多个 pair，但 H 组装一次完成，性能快 ~N_对 倍）；
      - 远失谐 → SOS 二阶 AC 位移（加进 H0 对角），驱动对绝不重复计 AC。
    失谐基准 = 裸能级差（超精细+Zeeman，不含 AC），无歧义。

    **统一场算符**（Pulser 范式）：M = A + diag(ac_shift) 一个矩阵——
    Rabi 耦合元与对角位移元同属该场的效应，H(t) += env(t)·M。
    返回 (field, ac_shift_mhz[N])：field['M'] 为统一算符，
    field['A']/field['ac_shift'] 保留元数据（诊断/ip4 稀疏优化用）；
    无近共振对时 field=None（ac 仍返回，由调用方包装成纯位移场）。
    """
    from . import core as _c
    q = {'π': 0, 'σ+': 1, 'σ-': -1}.get(pol, 0)
    N = len(states)
    E0_sq = _c.field_e0_sq(intensity_mwcm2)
    omega_rad = 2.0 * np.pi * freq_mhz * 1e6
    wf = waveform if waveform is not None else square(0.0, 1e12)
    A = np.zeros((N, N))
    g_idx, e_idx = [], []
    ac = np.zeros(N)
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
            if abs(Delta) < max(_c.AC_NEAR_CUTOFF_MHZ, Om):
                # 近共振 → 多对驱动（A 存真实 Rabi，ω_L = 光频）
                A[lo, hi] = A[hi, lo] = Om
                g_idx.append(lo); e_idx.append(hi)
            else:
                # 远失谐 → SOS（低态负移、高态正移）
                nu_rad = 2.0 * np.pi * nu_mhz * 1e6
                ac[lo] += _c.far_sos_shift_mhz(d, E0_sq, omega_rad, nu_rad)
                ac[hi] += _c.far_sos_shift_mhz(d, E0_sq, omega_rad, -nu_rad)
    if not g_idx:
        return None, ac
    M = A + np.diag(ac)
    return {'kind': 'multi', 'M': M, 'A': A, 'ac_shift': ac,
            'g_idx': g_idx, 'e_idx': e_idx,
            'omega0_mhz': 1.0, 'omega_L_mhz': freq_mhz, 'waveform': wf}, ac


def _field_geometry(states, f):
    """解析光场 → dict(A, g_idx, e_idx, omega0_mhz, omega_L_mhz, waveform)。"""
    N = len(states)
    if f['kind'] == 'multi':
        # 多对驱动 + 对角位移统一算符：M = A + diag(ac)（Pulser 范式）。
        # omega0=1 → 组装 H += env·M；A/ac_shift 保留供 ip4 稀疏优化与诊断。
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


def _assign_reference(states, fields):
    """统一旋转系（级联 RWA）：给每个态分配旋转参考频率 ref_mhz。

    以 (n,l,j) 组为单位：从能量最低的组 ref=0 出发，沿每个光场
    (g组→e组) 传播 ref[e组] = ref[g组] + ω_L,k（ω_L 含失谐）——链式/树状
    驱动图使每个驱动对在旋转系里共振（耦合为常数）。这正确处理
    **共享能级**的多驱动（如光频 6P↔60S 与微波 60S↔60P 同开时 60S 是
    中间态，级联旋转 ref[60P]=ref[60S]+ω_mw）。
    闭环不一致的驱动留下残留拍频 δ（返回 deltas，调用方按 RWA 丢弃）。

    返回 (ref_mhz 每态数组, deltas 每 field 列表)。
    """
    def _key(i):
        s = states[i]
        return (s['n'], s['l'], s['j'])
    keys = []
    for i in range(len(states)):
        k = _key(i)
        if k not in keys:
            keys.append(k)
    ref = {k: np.nan for k in keys}
    Emin = min(s['E_mhz'] for s in states)
    start = next(_key(i) for i, s in enumerate(states) if s['E_mhz'] == Emin)
    ref[start] = 0.0
    edges = []
    for f in fields:
        g = _field_geometry(states, f)
        for gi, ei in zip(g['g_idx'], g['e_idx']):
            edges.append((_key(gi), _key(ei), g['omega_L_mhz']))
    for _ in range(len(keys) + 2):
        changed = False
        for kg, ke, w in edges:
            if np.isnan(ref[kg]) and not np.isnan(ref[ke]):
                ref[kg] = ref[ke] - w
                changed = True
            elif np.isnan(ref[ke]) and not np.isnan(ref[kg]):
                ref[ke] = ref[kg] + w
                changed = True
        if not changed:
            break
    for kg, ke, w in edges:
        if np.isnan(ref[kg]) and np.isnan(ref[ke]):
            ref[kg] = 0.0
            ref[ke] = w
        elif np.isnan(ref[ke]):
            ref[ke] = ref[kg] + w
        elif np.isnan(ref[kg]):
            ref[kg] = ref[ke] - w
    for k in keys:
        if np.isnan(ref[k]):
            ref[k] = 0.0
    ref_arr = np.array([ref[_key(i)] for i in range(len(states))])
    deltas = [w - (ref[ke] - ref[kg]) for kg, ke, w in edges]
    return ref_arr, deltas


def hamiltonian(states, fields, t_ns, bs=False):
    """t 时刻旋转系 RWA 哈密顿（MHz，N×N 实对称）——供调试/参考。"""
    E = np.array([s['E_mhz'] for s in states])
    ref, _ = _assign_reference(states, fields)
    H = np.diag(E - ref)
    for f in fields:
        g = _field_geometry(states, f)
        if bs and g['omega_L_mhz'] > 1e-6:
            bs_shift = g['omega0_mhz'] ** 2 / (4.0 * g['omega_L_mhz'])
            for i in g['g_idx']:
                H[i, i] -= bs_shift
            for i in g['e_idx']:
                H[i, i] += bs_shift
        env = float(np.asarray(g['waveform'](np.atleast_1d(t_ns))).flat[0])
        # 统一场算符：Rabi 耦合 (env/2)·A + 对角位移 env·diag(ac)（时变）
        H = H + (g['omega0_mhz'] * env / 2.0) * g['A']
        a = g.get('ac_shift')
        if a is not None and np.any(np.asarray(a, float) != 0.0):
            H = H + env * np.diag(np.asarray(a, float))
    return H


# ---------------- 后端选择（仅 pwc 用） ----------------
def _pick_backend(backend):
    """返回是否用 torch。'auto' → numpy（实测 torch 导入 ~4 s 且小矩阵批量
    eigh 无优势；pwc 的 numpy 路径对典型 N≤30 仅 0.1~0.3 s）。大基组/GPU
    用户显式选 'gpu'（torch 批量 eigh）。"""
    if backend == 'cpu':
        return False
    if backend == 'gpu':
        try:
            import torch        # noqa: F401
        except ImportError:
            raise RuntimeError('GPU 后端需要 torch（pip install torch）')
        return True
    return False


# ============================================================
# 算法 1：精确 Floquet 传播子（常数包络 + 单频，第一性原理，成本不随时长增长）
# ============================================================
def _evolve_floquet(E, mats, omega_mhz, psi0, t, n_sub=2000, record_amp=False,
                    progress_arr=None):
    """H(t) = diag(E) + cos(_CONV·ω·t)·Σ_k mats_k（单频 ω、常数包络）。

    精确积分一个周期 T=2π/(_CONV·ω)（中点法 n_sub 步，存 U(τ) 网格）
    → U(T) 本征分解（准能量 ε_k = −arg(λ_k)/T）
    → 任意 t = nT + τ：ψ(t) = U(τ)·W·diag(e^{−iεnT})·W†·ψ0。

    对周期驱动 = TDSE 精确解（无 RWA、无光子块截断）；误差仅来自单周期
    数值积分（2 阶收敛，n_sub 加密到机器精度）。成本 = 一个周期积分 +
    一次对角化 + 每输出时刻 O(N²)，**不随模拟时长增长**。
    progress_arr（可选）：np.float64[2]，[0]=进度 0..1，[1]≠0 请求取消。
    """
    w = _CONV * omega_mhz
    T = 2.0 * np.pi / w
    N = len(psi0)
    # 内存自适应：n_sub 过大时降档（(n_sub+1)·N²·16B ≤ ~256 MB）
    n_sub = max(200, min(int(n_sub), int(256e6 / (N * N * 16.0))))
    dt = T / n_sub
    H0 = np.diag(E).astype(float)
    Asum = np.sum(mats, axis=0).astype(float)          # 单频：Σ_k cos·A_k
    # 单周期积分（中点法），存 U 于子步网格
    Us = np.empty((n_sub + 1, N, N), complex)
    Us[0] = np.eye(N)
    U = np.eye(N, dtype=complex)
    cw = np.cos(w * ((np.arange(n_sub) + 0.5) * dt))
    for j in range(n_sub):
        H = H0 + cw[j] * Asum
        e, Ue = np.linalg.eigh(H)
        D = np.exp(-1j * _CONV * dt * e)
        U = Ue @ (D[:, None] * (Ue.T @ U))
        Us[j + 1] = U
        if progress_arr is not None and (j & 127) == 0:
            progress_arr[0] = 0.9 * (j + 1) / n_sub
            if progress_arr[1] != 0.0:
                raise RuntimeError('用户取消')
    UT = Us[-1]
    ev, evec = np.linalg.eig(UT)
    eps = -np.angle(ev) / T
    a = evec.conj().T @ psi0
    t_out = np.asarray(t, float)
    pops = np.zeros((len(t_out), N))
    pops[0] = np.abs(psi0) ** 2

    def _psi_at(tt):
        n = int(np.floor(tt / T))
        tau = tt - n * T
        j = min(int(round(tau / dt)), n_sub)
        return Us[j] @ (evec @ (a * np.exp(-1j * eps * n * T)))

    for m in range(1, len(t_out)):
        pops[m] = np.abs(_psi_at(t_out[m])) ** 2
        if progress_arr is not None and (m & 127) == 0:
            progress_arr[0] = 0.9 + 0.1 * m / len(t_out)
    psi_f = _psi_at(t_out[-1]) if record_amp else None
    if progress_arr is not None:
        progress_arr[0] = 1.0
    return t_out, pops, psi_f


# ============================================================
# 算法 2：相互作用绘景 RK4（第一性原理，任意时变包络；光频 RWA + 微波非 RWA）
# ============================================================
def _ip4_reference(states, fields):
    """ip4 的旋转参考：RWA 场级联传播 + 非 RWA 场 e 跟随 g（共享参考）。

    返回 (ref_arr, rwa_geoms, non_geoms, E_rot)。
    """
    N = len(states)
    geoms = [_field_geometry(states, f) for f in fields]

    def _ratio(g):
        return (g.get('real_omega0_mhz', g['omega0_mhz'])
                / max(g['omega_L_mhz'], 1e-9))
    rwa_g = [g for g in geoms if _ratio(g) < IP4_RWA_THRESH]
    non_g = [g for g in geoms if _ratio(g) >= IP4_RWA_THRESH]

    def _key(i):
        s = states[i]
        return (s['n'], s['l'], s['j'])
    Emin = min(s['E_mhz'] for s in states)
    start = next(_key(i) for i, s in enumerate(states) if s['E_mhz'] == Emin)
    ref = {start: 0.0}
    # 纯位移场（无驱动对 g_idx 空）不参与参考传播。**全部**耦合对都要取边
    # ——multi 场可耦合多个 e 组（如光场 6P↔60S 耦合全部 6P F 态），若只按
    # g_idx[0] 传播，其余 6P 态 ref 留 0 → E_rot 伪劈裂 ~2.96e8 MHz → u 相位
    # 爆炸（曾致含 6P 组的多场场景微波驱动完全失效）。
    edges = [(_key(gi), _key(ei), g['omega_L_mhz'])
             for g in rwa_g if g.get('g_idx')
             for gi, ei in zip(g['g_idx'], g['e_idx'])]
    for _ in range(N + 2):
        changed = False
        for kg, ke, w in edges:
            if kg in ref and ke not in ref:
                ref[ke] = ref[kg] + w
                changed = True
        if not changed:
            break
    # 非 RWA 场：**全部**耦合组共享同一参考（e 跟随 g）。
    # 注意：multi 场可耦合多个 e 组（如 60S↔60P1/2 与 60S↔60P3/2），
    # 若只按第一对传播，其余 e 组 ref 留 0 → E_rot 出现 ~2.96e8 MHz 伪劈裂
    # → w_max 爆炸 → dt 被压到 ~1e-7 ns（内存/时间灾难）。必须逐对传播。
    for g in non_g:
        if not g.get('g_idx'):
            continue                     # 纯位移场：无驱动对，跳过
        gs = set()
        for gi, ei in zip(g['g_idx'], g['e_idx']):
            gs.add(_key(gi))
            gs.add(_key(ei))
        known = {k: ref[k] for k in gs if k in ref}
        vals = set(known.values())
        if len(vals) == 1:
            v = next(iter(vals))
            for k in gs:
                if k not in ref:
                    ref[k] = v
        # len(vals) > 1：场两端分属不同参考系（驱动有拍频）——罕见，保持原状
    for i in range(N):
        ref.setdefault(_key(i), 0.0)
    ref_arr = np.array([ref[_key(i)] for i in range(N)])
    return ref_arr, rwa_g, non_g


# ---------------- ip4 核心循环（numba JIT；无 numba 时回退 numpy） ----------------
def _ip4_vtilde_impl(B, mats, S, n_terms, N, sidx, u, ps):
    """out = u ⊙ (B@(u* ⊙ ps))，B = Σ_k S[k,sidx]·mats[k]（就地写 B）。

    统一场算符 M=A+diag(ac)：对角位移在 u 变换下相位抵消（|u|²=1），
    随包络时变自动正确，无需独立 ac 处理。"""
    for i in range(N):
        for j in range(N):
            B[i, j] = 0.0
    for k in range(n_terms):
        sk = S[k, sidx]
        if sk != 0.0:
            for i in range(N):
                for j in range(N):
                    B[i, j] += sk * mats[k, i, j]
    out = np.zeros(N, dtype=np.complex128)
    for i in range(N):
        s = 0.0j
        for j in range(N):
            s += B[i, j] * (np.conj(u[j]) * ps[j])
        out[i] = u[i] * s
    return out


if _njit is not None:
    _ip4_vtilde = _njit(cache=True)(_ip4_vtilde_impl)
else:
    _ip4_vtilde = _ip4_vtilde_impl


@_njit(cache=True)
def _ip4_vtilde_sp(B, mats, S, n_terms, N, sidx, u, ps, nz_i, nz_j,
                   row_ptr, n_nz):
    """稀疏版：只处理耦合非零元（CSR）。out = u ⊙ (B@(u* ⊙ ps))。

    统一场算符 M=A+diag(ac)：对角位移在 u 变换下相位抵消（|u|²=1），
    随包络时变自动正确，无需独立 ac 处理。"""
    for p in range(n_nz):
        B[nz_i[p], nz_j[p]] = 0.0
    for k in range(n_terms):
        sk = S[k, sidx]
        if sk != 0.0:
            for p in range(n_nz):
                i = nz_i[p]
                j = nz_j[p]
                B[i, j] += sk * mats[k, i, j]
    out = np.zeros(N, dtype=np.complex128)
    for i in range(N):
        s = 0.0j
        for p in range(row_ptr[i], row_ptr[i + 1]):
            j = nz_j[p]
            s += B[i, j] * (np.conj(u[j]) * ps[j])
        out[i] = u[i] * s
    return out


def _ip4_rk4_core_numpy(mats, S, Ec, dt, psi0, out_step, n_out, N,
                        n_terms, record_amp, t_end, progress_arr):
    """numpy 回退版（无 numba）。返回 (psi_starts, psi_last, div)。"""
    c = -1j * _CONV
    w05 = np.exp(1j * Ec * (dt / 2.0))
    w1 = np.exp(1j * Ec * dt)
    psi = np.array(psi0, complex)
    u_m = np.ones(N, complex)
    B = np.zeros((N, N), complex)
    M = int(out_step[n_out - 1])
    psi_starts = np.zeros((n_out, N), complex)
    psi_last = np.array(psi0, complex)
    rec = 1
    div = 0

    def rk4(ps, u0, i0):
        u05 = u0 * w05
        u1 = u0 * w1
        k1 = c * _ip4_vtilde(B, mats, S, n_terms, N, i0, u0, ps)
        k2 = c * _ip4_vtilde(B, mats, S, n_terms, N, i0 + 2, u05,
                             ps + dt / 2 * k1)
        k3 = c * _ip4_vtilde(B, mats, S, n_terms, N, i0 + 2, u05,
                             ps + dt / 2 * k2)
        k4 = c * _ip4_vtilde(B, mats, S, n_terms, N, i0 + 4, u1,
                             ps + dt * k3)
        return ps + dt / 6 * (k1 + 2 * k2 + 2 * k3 + k4), u1

    for m in range(M):
        while rec < n_out and out_step[rec] == m + 1:
            psi_starts[rec] = psi
            rec += 1
        if m == M - 1:
            psi_last = psi.copy()
        psi, u_m = rk4(psi, u_m, 4 * m)
        if (m & 511) == 0:
            progress_arr[0] = m / M
            if progress_arr[1] != 0.0:
                div = 2
                break
        if not np.isfinite(psi).all() or np.abs(psi).max() > 100.0:
            div = 1
            break
    return psi_starts, psi_last, div


if _njit is not None:

    @_njit(cache=True, nogil=True)
    def _ip4_rk4_core(mats, S, Ec, dt, psi0, out_step, n_out, N, n_terms,
                      record_amp, t_end, nz_i, nz_j, row_ptr, n_nz,
                      progress_arr):
        """相互作用绘景 RK4 主循环（numba，稀疏耦合，nogil 供进度轮询）。

        mats: (n_terms, N, N) float64 统一场算符（M=A+diag(ac)，对角位移
        随包络时变、相互作用绘景相位自动抵消）；S: (n_terms, ng) float64
        （1/4 步子网格）；out_step: (n_out,) int64。返回 (psi_starts,
        psi_last, div)：psi_starts[rec] = 输出时刻所在步的**步首**状态
        （供 numpy 做精确部分步进）；div：0=正常，1=|ψ|发散，2=用户取消。
        progress_arr: np.float64[2]，[0]=进度 0..1（每 512 步写），
        [1]≠0 请求取消。
        """
        c = -1j * (2.0 * np.pi * 1e-3)
        psi = psi0.copy()
        u = np.ones(N, dtype=np.complex128)
        w05 = np.exp(1j * Ec * (dt / 2.0))
        w1 = np.exp(1j * Ec * dt)
        B = np.zeros((N, N), dtype=np.complex128)
        k1 = np.zeros(N, dtype=np.complex128)
        k2 = np.zeros(N, dtype=np.complex128)
        k3 = np.zeros(N, dtype=np.complex128)
        k4 = np.zeros(N, dtype=np.complex128)
        ps = np.zeros(N, dtype=np.complex128)
        u05 = np.zeros(N, dtype=np.complex128)
        u1 = np.zeros(N, dtype=np.complex128)
        psi_starts = np.zeros((n_out, N), dtype=np.complex128)
        M = int(out_step[n_out - 1])
        rec = 1
        div = 0
        psi_last = psi0.copy()
        for m in range(M):
            i0 = 4 * m
            while rec < n_out and out_step[rec] == m + 1:
                for i in range(N):
                    psi_starts[rec, i] = psi[i]
                rec += 1
            if m == M - 1:
                for i in range(N):
                    psi_last[i] = psi[i]      # 最后步的步首（t=(M-1)dt）
            # ---- RK4 阶段 ----
            for i in range(N):
                u05[i] = u[i] * w05[i]
                u1[i] = u[i] * w1[i]
            # k1（t, u）
            k1 = c * _ip4_vtilde_sp(B, mats, S, n_terms, N, i0, u, psi,
                                    nz_i, nz_j, row_ptr, n_nz)
            # k2（t+dt/2, u05）
            for i in range(N):
                ps[i] = psi[i] + dt / 2.0 * k1[i]
            k2 = c * _ip4_vtilde_sp(B, mats, S, n_terms, N, i0 + 2, u05, ps,
                                    nz_i, nz_j, row_ptr, n_nz)
            # k3（t+dt/2, u05）
            for i in range(N):
                ps[i] = psi[i] + dt / 2.0 * k2[i]
            k3 = c * _ip4_vtilde_sp(B, mats, S, n_terms, N, i0 + 2, u05, ps,
                                    nz_i, nz_j, row_ptr, n_nz)
            # k4（t+dt, u1）
            for i in range(N):
                ps[i] = psi[i] + dt * k3[i]
            k4 = c * _ip4_vtilde_sp(B, mats, S, n_terms, N, i0 + 4, u1, ps,
                                    nz_i, nz_j, row_ptr, n_nz)
            # 更新
            for i in range(N):
                psi[i] = psi[i] + dt / 6.0 * (k1[i] + 2.0 * k2[i]
                                              + 2.0 * k3[i] + k4[i])
                u[i] = u1[i]
            # 进度 + 取消（每 512 步一次，nogil 下主线程可轮询）
            if (m & 511) == 0:
                progress_arr[0] = m / M
                if progress_arr[1] != 0.0:
                    div = 2
                    break
            # 发散防护
            for i in range(N):
                if not np.isfinite(psi[i]) or np.abs(psi[i]) > 100.0:
                    div = 1
                    break
            if div:
                break
        return psi_starts, psi_last, div


def _evolve_ip4(E, states, fields, psi0, t, frac=40.0, record_amp=False,
                progress_arr=None):
    """相互作用绘景 RK4（第一性原理）。

    - 光频（Ω/ωL < IP4_RWA_THRESH）→ RWA 旋转系耦合（反旋转误差 ~1e-7）；
    - 微波等强驱动 → 非 RWA：保留 cos(ω t) 反旋转项（第一性原理）。
    - 参考系：RWA 级联 + 非 RWA 共享（_ip4_reference）。
    - 相互作用绘景 H0=diag(E−ref)：相位精确除去 → 每步误差 ~(Ω·dt)³，
      dt = 最快非 RWA 载波周期/frac（默认 40 → 误差 ~1e-6，~1 min 内）。
    - 全 RWA 且包络恒常 → 一次对角化精确解（RWA 内精确，秒出）。
    """
    N = len(states)
    if progress_arr is None:
        progress_arr = np.zeros(2)
    ref_arr, rwa_g, non_g = _ip4_reference(states, fields)
    E_rot = np.asarray(E, float) - ref_arr
    # 旁观态（不在任何耦合对里）无相位演化需求（ψ=0），但其 E_rot 绝对值
    # 可达光频量级（如 6P↔60S 光场把 60S 参考抬到 ~2.96e8 MHz），会让
    # u(t)=e^{iE_rot t} 每步旋转数千 rad → exp 数值精度灾难。把它们设为 0
    # （相对任意，因 ψ≡0 不参与任何项）；耦合态再减共同参考（整体相位
    # 不影响布居，消除跨组 ~2.96e8 的大偏移）。
    coupled = set()
    for g in list(rwa_g) + list(non_g):
        coupled.update(g['g_idx'])
        coupled.update(g['e_idx'])
    E_rot = E_rot - float(np.min(E_rot[list(coupled)])) if coupled else E_rot
    for i in range(N):
        if i not in coupled:
            E_rot[i] = 0.0
    t_out = np.asarray(t, float)

    # ---- 全 RWA 且包络恒常 → 一次对角化精确解 ----
    if not non_g and rwa_g:
        consts = [_env_samples(g['waveform'], t_out[-1]) for g in rwa_g]
        if all(c for _, c in consts):
            H0 = np.diag(E_rot).astype(float)
            for g, (c, _) in zip(rwa_g, consts):
                H0 = H0 + (g['omega0_mhz'] * c / 2.0) * g['A']
                a = g.get('ac_shift')
                if a is not None and np.any(np.asarray(a, float) != 0.0):
                    H0 = H0 + c * np.diag(np.asarray(a, float))
            evals, evecs = np.linalg.eigh(H0)
            coef = evecs.T @ psi0
            ph = np.exp(-1j * _CONV * evals[:, None] * t_out[None, :])
            psi_t = evecs @ (coef[:, None] * ph)
            pops = np.abs(psi_t).T ** 2
            return t_out, pops, (psi_t[:, -1] if record_amp else None)

    # ---- 步长：由 Ṽ(t) 内最快的真实振荡频率决定 ----
    # Ṽ(t) = Σ s_k(t)·D(u)A_kD(u*)：矩阵元振荡在
    #   非 RWA 项：|E_rot_i−E_rot_j| ± ω_L（最快 = 劈裂 + 载波）
    #   RWA 项   ：|E_rot_i−E_rot_j|（旋转系失谐，通常 ≪ 载波）
    # 远失谐对（|Δ| 大）的劈裂 ν ≫ ω_L，若只按 ω_L 限步会漏解析 → 失稳。
    # 另加 RK4 稳定性约束 dt·‖Ṽ‖<2（低频率强驱动 ω_L≲Ω 时载波约束失效）、
    # 分辨率约束（全程至少 500 步）与 0.05 ns 绝对上限。
    w_max = 0.0
    cmax = 0.0
    for g in rwa_g:
        A = np.abs(g['A'])
        cmax = max(cmax, (g['omega0_mhz'] / 2.0) * A.max())
        for i in range(N):
            for j in range(i + 1, N):
                if A[i, j] > 1e-12:
                    w_max = max(w_max, abs(E_rot[i] - E_rot[j]))
    for g in non_g:
        A = np.abs(g['A'])
        cmax = max(cmax, g['omega0_mhz'] * A.max())
        wL = abs(g['omega_L_mhz'])
        for i in range(N):
            for j in range(i + 1, N):
                if A[i, j] > 1e-12:
                    w_max = max(w_max, abs(E_rot[i] - E_rot[j]) + wL)
    if w_max > 0:
        dt_carrier = 2.0 * np.pi / (_CONV * w_max) / frac
    else:
        dt_carrier = 0.01                        # 无耦合（慢变包络兜底）
    # RK4 精度约束：dt·‖Ṽ‖ ≲ 0.05（经验值 → 布居误差 ~1e-4~1e-5）。
    # 正常场景（Ω ≪ ω_L）dt_carrier 已满足；超强耦合（Ω ≳ ω_L，如
    # 100 mW/cm² 里德堡微波 → Ω~19 GHz）时载波约束不足以保证精度，
    # 必须按耦合幅度加密步长，否则 |ψ| 漂移 ~1e-2。
    dt_acc = 0.06 / (_CONV * max(cmax, 1e-6))
    dt_stab = 2.0 / (_CONV * max(cmax, 1e-6))    # RK4 稳定性上界
    dt_res = t_out[-1] / 500.0                   # 至少 500 步
    dt = min(dt_carrier, dt_acc, dt_stab, dt_res, 0.05)
    if dt <= 0 or not np.isfinite(dt):
        raise RuntimeError('ip4 步长非法：请检查驱动频率/光强配置。')

    # ---- 耦合项 s(t)·A 与相位 u(t) 预计算到 1/4 步长子网格 ----
    # RK4 各阶段时刻 = m·dt + c·dt（c∈{0,1/2,1/2,1}）→ 索引 4m, 4m+2, 4m+4
    M = int(np.ceil(t_out[-1] / dt))
    ng = 4 * M + 1
    tg = np.arange(ng) * (dt / 4.0)
    sgrid = []                          # 每项 (M, s 数组(ng,))
    for g in rwa_g:
        sgrid.append((g['M'],
                      (g['omega0_mhz'] / 2.0)
                      * np.asarray(g['waveform'](tg), float)))
    for g in non_g:
        sgrid.append((g['M'],
                      g['omega0_mhz'] * np.asarray(g['waveform'](tg), float)
                      * np.cos(_CONV * g['omega_L_mhz'] * tg)))
    # 统一场算符 M = A + diag(ac_shift)：对角位移随包络 env(t) 时变，
    # 相互作用绘景里对角元相位自动抵消（|u|²=1），无需独立 ac 机制。
    mats = [M for M, _ in sgrid]
    S = np.stack([s for _, s in sgrid]) if sgrid else np.zeros((0, ng))

    # ---- 无任何耦合项（空场）→ 布居不变，直接返回 ----
    if not mats:
        pops = np.zeros((len(t_out), N))
        pops[:] = np.abs(psi0) ** 2
        return t_out, pops, (np.array(psi0, complex) if record_amp else None)

    # ---- 相互作用绘景 RK4（numba 内核 + numpy 精确部分步进记录） ----
    out_step = np.minimum(np.ceil(t_out / dt).astype(int), M)
    pops = np.zeros((len(t_out), N))
    pops[0] = np.abs(psi0) ** 2
    Ec = _CONV * E_rot
    n_terms = len(mats)
    c = -1j * _CONV

    def rk4_step_arb(ps, t0, h):
        """任意子步（输出时刻精确记录用）：callable 求 s。"""
        def env_of(g, tt):
            return float(np.asarray(g['waveform'](np.atleast_1d(tt))).flat[0])
        ua = np.exp(1j * Ec * t0)
        ub = np.exp(1j * Ec * (t0 + h / 2))
        uc = np.exp(1j * Ec * (t0 + h))

        def Ht_arb(ps, t, u):
            out = np.zeros(N, complex)
            for g in rwa_g:
                out = out + (g['omega0_mhz'] / 2.0
                             * env_of(g, t)) * (u * (g['M'] @ (u.conj() * ps)))
            for g in non_g:
                s = g['omega0_mhz'] * env_of(g, t) * np.cos(_CONV * g['omega_L_mhz'] * t)
                out = out + s * (u * (g['M'] @ (u.conj() * ps)))
            return out

        k1 = c * Ht_arb(ps, t0, ua)
        k2 = c * Ht_arb(ps + h / 2 * k1, t0 + h / 2, ub)
        k3 = c * Ht_arb(ps + h / 2 * k2, t0 + h / 2, ub)
        k4 = c * Ht_arb(ps + h * k3, t0 + h, uc)
        return ps + h / 6 * (k1 + 2 * k2 + 2 * k3 + k4)

    mats_arr = np.asarray(mats, float)
    psi0c = np.asarray(psi0, complex)
    # 稀疏结构（CSR）：耦合非零元并集，供 numba 内核加速
    nz = set()
    for k in range(n_terms):
        A = mats_arr[k]
        for i in range(N):
            for j in range(N):
                if abs(A[i, j]) > 1e-12:
                    nz.add((i, j))
    nz = sorted(nz)
    n_nz = len(nz)
    if n_nz > 0 and _njit is not None:
        nz_i = np.array([p[0] for p in nz], np.int64)
        nz_j = np.array([p[1] for p in nz], np.int64)
        row_ptr = np.zeros(N + 1, np.int64)
        for i in range(N):
            row_ptr[i + 1] = row_ptr[i] + sum(1 for p in nz if p[0] == i)
        psi_starts, psi_last, div = _ip4_rk4_core(
            mats_arr, S, Ec, dt, psi0c, out_step, len(t_out), N, n_terms,
            record_amp, t_out[-1], nz_i, nz_j, row_ptr, n_nz, progress_arr)
    else:
        psi_starts, psi_last, div = _ip4_rk4_core_numpy(
            mats_arr, S, Ec, dt, psi0c, out_step, len(t_out), N, n_terms,
            record_amp, t_out[-1], progress_arr)
    if div == 1:
        raise RuntimeError(
            f'ip4 数值发散（dt={dt:.3g} ns）。可能原因：驱动频率过低'
            '（ω_L ≲ Ω，如 MHz 级超精细/Zeeman 驱动）或耦合过强。'
            '可尝试：减小 ip4_frac（如 5）、降低光强，或先用 '
            "method='pwc' 预览。")
    if div == 2:
        raise RuntimeError('用户取消')
    # 输出记录：从各输出时刻所在步的步首做精确部分步进
    for rec in range(1, len(t_out)):
        sm = int(out_step[rec])
        if sm <= 0:
            continue
        tm = (sm - 1) * dt
        psi_r = rk4_step_arb(psi_starts[rec], tm, t_out[rec] - tm)
        pops[rec] = np.abs(psi_r) ** 2
        if progress_arr[1] != 0.0:
            raise RuntimeError('用户取消')
    if progress_arr is not None:
        progress_arr[0] = 1.0
    # 末态振幅（精确到 t_end）
    if record_amp:
        M_last = int(out_step[-1])
        tm = (M_last - 1) * dt
        psi_f = rk4_step_arb(psi_last, tm, t_out[-1] - tm)
    else:
        psi_f = None
    return t_out, pops, psi_f


# ============================================================
# 算法 3：RWA 分段常数（最快预览；torch GPU 批量，pops 全程 GPU 累积）
# ============================================================
def _evolve_pwc(H0, geoms, psi0, t, use_torch, record_amp, progress_arr=None):
    N = len(psi0)
    dt = np.diff(t)
    M = len(dt)
    H0 = np.array(H0, dtype=float)
    H0 = H0 - np.min(np.diag(H0)) * np.eye(N)   # 减共同参考（不影响布居）
    const = H0.copy()
    for g in geoms:
        bs = g.get('bs_shift', 0.0)
        for i in g['g_idx']:
            const[i, i] -= bs
        for i in g['e_idx']:
            const[i, i] += bs
    # 每段 H_m（中点采样波形）
    tm = (t[:-1] + t[1:]) / 2.0
    Hm = np.repeat(const[None], M, axis=0)
    for g in geoms:
        env = np.asarray(g['waveform'](tm), float)
        Hm = Hm + (g['omega0_mhz'] * env / 2.0)[:, None, None] * g['A'][None]
        # 时变 AC 位移（远失谐对）：随包络开关加 H0 对角（pwc 同 ip4 语义）
        a = g.get('ac_shift')
        if a is not None and np.any(np.asarray(a, float) != 0.0):
            Hm = Hm + env[:, None, None] * np.diag(np.asarray(a, float))[None]

    n_t = len(t)
    psi = np.array(psi0, complex)
    if progress_arr is None:
        progress_arr = np.zeros(2)

    def _tick(m):
        if (m & 127) == 0:
            progress_arr[0] = m / M
            if progress_arr[1] != 0.0:
                raise RuntimeError('用户取消')

    if use_torch:
        import torch
        Ht = torch.from_numpy(Hm)
        ev, U = torch.linalg.eigh(Ht)       # (M,N), (M,N,N) 实
        U = U.to(torch.complex128)
        psit = torch.from_numpy(psi)
        dts = torch.from_numpy(dt)
        pops_t = torch.zeros((n_t, N), dtype=torch.float64)
        pops_t[0] = psit.abs() ** 2
        for m in range(M):
            _tick(m)
            ph = torch.exp(-1j * _CONV * dts[m] * ev[m])
            psit = U[m] @ (ph * (U[m].conj().T @ psit))
            pops_t[m + 1] = psit.abs() ** 2
        pops = pops_t.numpy()               # 仅一次 GPU→CPU 同步
        psi = psit.numpy()
    else:
        pops = np.zeros((n_t, N))
        pops[0] = np.abs(psi0) ** 2
        for m in range(M):
            _tick(m)
            e, U = np.linalg.eigh(Hm[m])
            ph = np.exp(-1j * _CONV * dt[m] * e)
            psi = U @ (ph * (U.T @ psi))
            pops[m + 1] = np.abs(psi) ** 2
    progress_arr[0] = 1.0
    if record_amp:
        return t, pops, psi
    return t, pops, None


# ============================================================
# 参考积分器（内部保留，仅 checks 交叉验证用；不作为可选 method）
# ============================================================
def _evolve_full(E, geoms, psi0, t, record_amp):
    """非 RWA 零近似演化（实验室系 solve_ivp，慢）——第一性原理参考。"""
    from scipy.integrate import solve_ivp
    N = len(psi0)
    H0 = np.diag(E).astype(float)
    H0 = H0 - np.min(np.diag(H0)) * np.eye(N)

    def H_at(tv):
        H = H0.copy()
        for g in geoms:
            env = float(np.asarray(g['waveform'](np.atleast_1d(tv))).flat[0])
            H = H + (g['omega0_mhz'] * env
                     * np.cos(_CONV * g['omega_L_mhz'] * tv)) * g['A']
        return H

    def rhs(tv, y):
        return -1j * _CONV * (H_at(tv) @ y)

    sol = solve_ivp(rhs, (t[0], t[-1]), np.asarray(psi0, complex),
                    t_eval=t, rtol=1e-8, atol=1e-10)
    if not sol.success:
        raise RuntimeError(f'_evolve_full 积分失败: {sol.message}')
    pops = np.abs(sol.y).T ** 2
    if record_amp:
        return t, pops, sol.y[:, -1]
    return t, pops, None


def _evolve_full_hybrid(E, states, fields, psi0, t, record_amp, thresh=1e-4):
    """混合参考（光频 RWA + 其余非 RWA，solve_ivp）——验证 ip4 的参考。

    非 RWA 驱动的 g/e 共享同一旋转参考（e 跟随 g），使微波耦合为纯 cos 快项
    ——步长只受微波载波限制，比全非 RWA 快 ~10^4 倍；光频 RWA 误差 ~1e-7。
    """
    from scipy.integrate import solve_ivp
    N = len(states)
    geoms = [_field_geometry(states, f) for f in fields]

    def _key(i):
        s = states[i]
        return (s['n'], s['l'], s['j'])

    def _ratio(g):
        return (g.get('real_omega0_mhz', g['omega0_mhz'])
                / max(g['omega_L_mhz'], 1e-9))
    rwa_g = [g for g in geoms if _ratio(g) < thresh]
    non_g = [g for g in geoms if _ratio(g) >= thresh]
    ref_arr, _, _ = _ip4_reference(states, fields)
    H0 = np.diag(E - ref_arr).astype(float)
    H0 = H0 - np.min(np.diag(H0)) * np.eye(N)
    for g in rwa_g:
        H0 = H0 + (g['omega0_mhz'] / 2.0) * g['A']

    def H_at(tv):
        H = H0.copy()
        for g in rwa_g:
            env = float(np.asarray(g['waveform'](np.atleast_1d(tv))).flat[0])
            H = H + (g['omega0_mhz'] * (env - 1.0) / 2.0) * g['A']
        for g in non_g:
            env = float(np.asarray(g['waveform'](np.atleast_1d(tv))).flat[0])
            H = H + (g['omega0_mhz'] * env
                     * np.cos(_CONV * g['omega_L_mhz'] * tv)) * g['A']
        return H

    def rhs(tv, y):
        return -1j * _CONV * (H_at(tv) @ y)

    sol = solve_ivp(rhs, (t[0], t[-1]), np.asarray(psi0, complex),
                    t_eval=t, rtol=1e-9, atol=1e-11)
    if not sol.success:
        raise RuntimeError(f'_evolve_full_hybrid 积分失败: {sol.message}')
    pops = np.abs(sol.y).T ** 2
    if record_amp:
        return t, pops, sol.y[:, -1]
    return t, pops, None


# ---------------- 一站式入口 ----------------
def _onehot(N, i):
    v = np.zeros(N, complex)
    v[i] = 1.0
    return v


def _as_vector(states, psi0):
    """初态 → 复向量。psi0 为：索引 | (n,l,j,m) | dict(n,l,j,m) | 复数组。"""
    N = len(states)
    if isinstance(psi0, np.ndarray):
        v = np.asarray(psi0, complex)
        if v.shape != (N,):
            raise ValueError(f'初态向量形状应为 ({N},)，实际 {v.shape}')
        return v
    if isinstance(psi0, (int, np.integer)):
        return _onehot(N, int(psi0))
    if isinstance(psi0, dict):
        psi0 = (psi0['n'], psi0['l'], psi0['j'], psi0['m'])
    if isinstance(psi0, (tuple, list)):
        if len(psi0) == N:
            return np.asarray(psi0, complex)
        if len(psi0) == 4:
            n, l, j, m = psi0
            for i, s in enumerate(states):
                if (abs(s['n'] - n) < 1e-9 and abs(s['l'] - l) < 1e-9
                        and abs(s['j'] - j) < 1e-9 and abs(s['m'] - m) < 1e-9):
                    return _onehot(N, i)
            raise ValueError(f'初态 (n,l,j,m)={psi0} 不在能级列表中')
        raise ValueError(f'无法解析初态 psi0: {psi0}')
    raise ValueError(f'无法解析初态 psi0: {psi0}')


def _auto_pick(states, fields, t_end):
    """auto 调度：常数包络 + 单频 → 'floquet'；否则 → 'ip4'。"""
    if not fields:
        return 'ip4'
    geoms = [_field_geometry(states, f) for f in fields]
    w0 = geoms[0]['omega_L_mhz']
    single = all(abs(g['omega_L_mhz'] - w0)
                 <= 1e-6 * max(abs(w0), 1.0) for g in geoms)
    const = all(_env_samples(g['waveform'], t_end)[1] for g in geoms)
    return 'floquet' if (single and const) else 'ip4'


def evolve(states, fields, psi0, t_grid, method='auto', backend='auto',
           record_amp=False, bs=False, ac_shift_mhz=None, ip4_frac=20.0,
           floquet_n_sub=2000, progress_arr=None, log=None):
    """单原子含时演化（独立模块，不含任何静态能移计算）。

    states : make_levels 输出（全部考虑的能级）
    fields : field()/group_field()/real_field() 输出列表（任意多组时变光场）
    psi0   : 初态——states 索引 | (n,l,j,m) | dict(n,l,j,m) | 复向量
    t_grid : 时间数组（ns）
    method : 'auto'（默认）自动调度
               → 常数包络 + 单频 → 'floquet'（精确，机器精度，秒级）
               → 否则            → 'ip4'（第一性原理，~20-30 s，误差 ~3e-5）
             'floquet'：精确 Floquet 传播子（仅限常数包络 + 单频；越界报错）
             'ip4'    ：相互作用绘景 RK4（任意包络；光频 RWA + 微波非 RWA）
             'pwc'    ：RWA 分段常数（最快预览，精度 ~Ω/ωL；微波可差 ~3%）
    backend: 'auto'（默认）/ 'cpu' / 'gpu'（仅 pwc 用；floquet/ip4 恒 numpy）
    record_amp: 是否保留末态复振幅
    bs     : Bloch–Siegert 一阶修正（仅 pwc 有意义；floquet/ip4 非 RWA
             已自动包含，传入忽略）
    ip4_frac / floquet_n_sub：ip4 步长 = 最快非 RWA 载波周期/ip4_frac
             （默认 20 → ~3e-5 误差、~20-30 s；40 → ~1e-6、~50-60 s）；
             floquet 单周期子步数（默认 2000）。
    progress_arr: 可选 np.float64[2] 共享进度数组——[0]=进度 0..1（计算中
             定期写入），[1]≠0 请求取消（抛 RuntimeError('用户取消')）。
             适合放在工作线程里由 UI 轮询（numba 内核 nogil，主线程可读）。
    log      : 可选 callable(str)——算法提示/警告（auto 选型、RWA 误差等）
             转发给 UI（如 QStatusBar）；缺省静默。

    返回 dict(t, pops, states, om_mhz, method, backend[, psi_amp])。
      pops: (n_t, n_states) 各态布居（Σ 每行=1）。
    """
    t = np.asarray(t_grid, float)
    psi0v = _as_vector(states, psi0)
    E = np.array([s['E_mhz'] for s in states])       # 裸能（ref/ω_L 基准，不含 AC）
    if ac_shift_mhz is not None:
        # 显式常数 ac（旧接口，测试用）：优先，field 内嵌时变 ac 忽略（防双重）
        E = E + np.asarray(ac_shift_mhz, float)      # H0 对角加远失谐 AC
        for f in fields:
            f.pop('ac_shift', None)
            if 'M' in f:                             # M = A + diag(ac) 统一算符：
                ac0 = np.diag(f['M']).copy()         # 清掉 M 对角，保持 A 部分
                f['M'] = f['M'] - np.diag(ac0)
    if progress_arr is None:
        progress_arr = np.zeros(2)
    log = log or (lambda *a: None)                   # 可选信息回调（GUI 转发状态栏）

    if method == 'auto':
        method = _auto_pick(states, fields, t[-1])
        log(f'[evolve] auto → {method}'
            + ('（常数包络+单频，精确 Floquet）' if method == 'floquet'
               else '（相互作用绘景 RK4，第一性原理）'))

    if method == 'floquet':
        geoms = [_field_geometry(states, f) for f in fields]
        if not geoms:
            t, pops, _ = _evolve_ip4(E, states, fields, psi0v, t,
                                     progress_arr=progress_arr)
            return dict(t=t, pops=pops, states=states, om_mhz=[],
                        method='ip4', backend='numpy')
        w0 = geoms[0]['omega_L_mhz']
        ok_freq = all(abs(g['omega_L_mhz'] - w0)
                      <= 1e-6 * max(abs(w0), 1.0) for g in geoms)
        ok_env = all(_env_samples(g['waveform'], t[-1])[1] for g in geoms)
        if not (ok_freq and ok_env):
            raise ValueError(
                'method=floquet 要求全部光场常数包络且单一频率'
                f'（单频={ok_freq}，常数包络={ok_env}）；'
                "时变包络/多频请用 method='ip4' 或 'auto'")
        mats = []
        E_f = np.array(E, float)
        for g in geoms:
            c = _env_samples(g['waveform'], t[-1])[0]
            mats.append(g['omega0_mhz'] * c * g['A'])
            # floquet 仅常数包络：ac·env = ac·常数 → 并入 H0 对角
            a = g.get('ac_shift')
            if a is not None and np.any(np.asarray(a, float) != 0.0):
                E_f = E_f + c * np.asarray(a, float)
        t, pops, psi = _evolve_floquet(E_f, mats, w0, psi0v, t,
                                       n_sub=floquet_n_sub,
                                       record_amp=record_amp,
                                       progress_arr=progress_arr)
        out = dict(t=t, pops=pops, states=states,
                   om_mhz=[g['omega0_mhz'] for g in geoms],
                   method='floquet', backend='numpy')
        if record_amp:
            out['psi_amp'] = psi
        return out

    if method == 'ip4':
        t, pops, psi = _evolve_ip4(E, states, fields, psi0v, t,
                                   frac=ip4_frac, record_amp=record_amp,
                                   progress_arr=progress_arr)
        out = dict(t=t, pops=pops, states=states,
                   om_mhz=[g['omega0_mhz']
                           for g in [_field_geometry(states, f) for f in fields]],
                   method='ip4', backend='numpy')
        if record_amp:
            out['psi_amp'] = psi
        return out

    if method == 'pwc':
        ref, deltas = _assign_reference(states, fields)
        H0 = np.diag(E - ref).astype(float)
        geoms = [_field_geometry(states, f) for f in fields]
        for g, d in zip(geoms, deltas):
            wL = g['omega_L_mhz']
            g['residual_mhz'] = d
            if abs(d) > 1e-6:
                log(f'[evolve] 警告：光场驱动对残留拍频 {d:.3g} MHz'
                    '（驱动图闭环不一致），按 RWA 丢弃该项（误差 ~Ω/|δ|）。')
            g['bs_shift'] = (g['omega0_mhz'] ** 2 / (4.0 * wL)
                             if (bs and wL > 1e-6) else 0.0)
            r0 = g.get('real_omega0_mhz', g['omega0_mhz'])
            ratio = r0 / max(wL, 1e-9)
            if ratio >= 0.005:
                log(f'[evolve] 提示：驱动 Ω/ωL={ratio:.3%}（>=0.5%），'
                    f'RWA 误差估计 ~{0.25 * ratio:.1e}；'
                    "精算请用 method='auto'（floquet/ip4，非 RWA 精确）。")
        use_torch = _pick_backend(backend)
        t, pops, psi = _evolve_pwc(H0, geoms, psi0v, t, use_torch,
                                   record_amp, progress_arr)
        out = dict(t=t, pops=pops, states=states,
                   om_mhz=[g['omega0_mhz'] for g in geoms],
                   method='pwc', backend='torch' if use_torch else 'numpy')
        if record_amp:
            out['psi_amp'] = psi
        return out

    raise ValueError(f"未知 method: {method}（可选 'auto'/'floquet'/'ip4'/'pwc'）")


# ---------------- 观测辅助 ----------------
def rwa_error_estimate(states, fields):
    """每个驱动的 RWA 误差估计（供 method 选择 / 精度审计）。

    返回列表，每项 dict(omega0_mhz, omega_L_mhz, ratio=Ω/ωL, est_err)。
    est_err = max|Δpops| 估计 ≈ Ω/(4ωL)（基准实测：30/100/300MHz 微波
    在 60S↔60P（ωL=1.68e4MHz）下误差 4.4e-4/1.5e-3/4.5e-3，系数 ~1/4；
    注意：多能级 + 长时（400ns）实测 RWA 误差可比单对估计大 ~20 倍，达 ~3%，
    精算勿依赖 RWA。光频驱动 ~1e-7 可忽略）。
    """
    out = []
    for f in fields:
        g = _field_geometry(states, f)
        wL = max(g['omega_L_mhz'], 1e-9)
        r = g['omega0_mhz'] / wL
        out.append(dict(omega0_mhz=g['omega0_mhz'],
                        omega_L_mhz=g['omega_L_mhz'], ratio=r,
                        est_err=min(0.25 * r, 1.0)))
    return out


def group_pops(states, pops, groups):
    """每组（dict(n,l,j)）总布居随时间：(n_groups, n_t)。"""
    gp = []
    for g in groups:
        idx = [i for i, s in enumerate(states)
               if (s['n'], s['l'], s['j']) == (g['n'], g['l'], g['j'])]
        gp.append(pops[:, idx].sum(axis=1))
    return np.array(gp)


def level_pops(states, pops, n, l, j, m=None):
    """指定子能级布居随时间；m=None 取整组 (n,l,j)。"""
    idx = [i for i, s in enumerate(states)
           if (s['n'], s['l'], s['j']) == (n, l, j)
           and (m is None or abs(s['m'] - m) < 1e-9)]
    if not idx:
        return np.zeros(pops.shape[0])
    return pops[:, idx].sum(axis=1)
