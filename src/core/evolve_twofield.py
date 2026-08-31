# -*- coding: utf-8 -*-
"""两场引擎（evolve_two_field）：光场旋转系 + 微波 Sambe 光子块 —— 单原子版。

物理背景（docs/论文级方案调研-交换全过程场景.md，2026-08-29 方案收敛）：
- 用户指正：depump 光的 AC 位移是交换共振的开关（6P 流形必要，不能砍）；
- 数值教训：① 光场 THz 载波若进 Sambe 光子块，缺陷求积面对 ±1.86e6 rad/ns
  振荡必然爆炸（620 ns 卡死根因）；② 统一 M 被离共振 60S↔60P 巨矩阵元污染。
- 本模块的核心思想（"分开光/微波场"）：
  1. **光场**：旋转系解析吸收载波——把被光场耦合的**低能流形**（6P₃/₂ 全部 16 态）
     整体平移参考 ref = ω_opt（取第一个耦合该流形的光场载频），使 60S↔6P 耦合对
     停在失谐 Δ=420/620 MHz（GHz 级慢变）。光场因此**不占光子块**：其耦合项
     Ω/2·A_opt 为常数矩阵（载波已吸收），包络 c(t) 随段冻结——只有包络尺度振荡。
     - 光学多光子可忽略（用户确认：第二个 295.7 THz 光子即电离连续态，离散基组
       内无落点；反旋项全场 Bloch–Siegert 相位 4.4e-5 rad ≪ tol）→ RWA 丢弃；
     - 顺带消灭"19.3 GHz 离共振 60S↔60P 污染"：该对在旋转系里快振荡，RWA 丢弃。
  2. **微波场**：Sambe 光子块 M=2（5 块，k=±2）——**多光子显式保留**（用户确认
     60S→60P→更高级联是真实通道，driver.py 光子数 BFS 已证）。光子偏移仅
     ±34.5 GHz（216 rad/ns），缺陷求积可解析，无 THz。
  3. 冻结段传播 + 缺陷修正（与 evolve_exact 同构）：K(t) = K_static + Σ c_opt(t)·(A_opt⊗I_P)
     + Σ c_mw(t)·Kc_mw；只有微波光子偏移参与求积振荡（≤216 rad/ns）。

单位与约定：与 evolve_exact 完全一致（MHz/ns，_CONV=2π×1e-3，Rabi 周频，
kind='exact' 场 A 为峰值 Rabi 矩阵）。返回 dict(t, pops, states, om_mhz,
method='two_field', backend, loss, error_report)。

2026-08-29：单原子版（两体扩展在 evolve_exact_pair 同框架下二期做）。
"""
import numpy as np
import scipy.sparse as sp

from . import static_atom as core
from . import evolve as _evolve
from .evolve_sambe import (_CONV, _M_MAX, _M_ANALYTIC_BW_MHZ, _group_by_freq,
                           _gl_nodes, _GL_CACHE, setup_P)

# ---- 场分类 ----
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


def _crow(s):
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


class _TwoFieldSetup:
    """旋转光场 + 微波光子块扩展空间。

    扩展基 |i⟩⊗|n⟩（n=0..2M_mw），索引 idx = i + N·n。
    微波 M_mw 统一（多微波载频各自光子维 → 混合进制，与本类一致）。
    """

    def __init__(self, E_rot, N, mw_groups, M, opt_terms):
        self.N = N
        self.E_rot = np.asarray(E_rot, float)
        self.mw_groups = mw_groups
        self.M = M
        G = len(mw_groups)
        if isinstance(M, (int, np.integer)):
            self.Ms = [int(M)] * G
        else:
            self.Ms = [int(m) for m in M]
            if len(self.Ms) != G:
                raise ValueError(f'M 长度 {len(self.Ms)} ≠ 微波组数 {G}')
        self.Ds = [2 * m + 1 for m in self.Ms]
        P = 1
        for d in self.Ds:
            P *= d
        self.P = P
        self.dim = N * P
        idx = np.arange(P)
        self.strides, self.nvecs = [], []
        base = np.zeros(P)
        stride = 1
        for gi, g in enumerate(mw_groups):
            Dg, Mg = self.Ds[gi], self.Ms[gi]
            nvec = (idx // stride) % Dg
            self.strides.append(stride)
            self.nvecs.append(nvec)
            base = base - (nvec - Mg) * g['omega_mhz']
            stride *= Dg
        # 对角：E_rot + 微波光子偏移（无光场偏移——载波已旋转吸收！）
        self.diag = np.tile(self.E_rot, P) + np.repeat(base, N)
        nphase = np.zeros(P)
        for nvec, g in zip(self.nvecs, mw_groups):
            nphase = nphase + nvec * (_CONV * g['omega_mhz'])
        self.nphase = nphase
        self.p_center = sum(Mg * st for Mg, st in zip(self.Ms, self.strides))
        # 微波光子邻接耦合结构（k=±1..±M 显式 = 多光子保留）
        self.coup = []
        for gi, g in enumerate(mw_groups):
            Dg = self.Ds[gi]
            Vpat = np.zeros((N, N))
            for A in g['As']:
                Vpat = Vpat + (np.abs(A) > 1e-12)
            ii, jj = np.nonzero(Vpat)
            stride = self.strides[gi]
            grid = np.meshgrid(*[np.arange(d) for d in self.Ds],
                               indexing='ij')
            pflat = np.zeros(P, int)
            for gg in range(G):
                pflat = pflat + grid[gg].ravel() * self.strides[gg]
            ng_g = grid[gi].ravel()
            rows, cols = [], []
            for s in range(Dg - 1):
                p_src = pflat[ng_g == s]
                p_dst = p_src + stride
                for a, b in zip(ii, jj):
                    rows.append(a + N * p_src)
                    cols.append(b + N * p_dst)
            rows = np.concatenate(rows) if rows else np.zeros(0, int)
            cols = np.concatenate(cols) if cols else np.zeros(0, int)
            self.coup.append(dict(r=rows, c=cols, nz_i=ii, nz_j=jj,
                                  n_p=P // Dg))
        self.K0 = np.diag(self.diag)
        self.dense = self.dim <= 2200
        # 微波单位包络耦合结构 Kc_g_wf（稠密）——每波形一个矩阵，与包络配对；
        # 修复：同频多波形合并 bug（原代码将 ΣA 合并后乘 Σenv，丢失逐波配对）。
        # Kc_dense[gi][w_idx] = 0.5·A_w 的扩展空间映射矩阵（与 envs[w_idx] 一一配对）
        self.Kc_dense = []
        for gi, cp in enumerate(self.coup):
            if cp['r'].size and self.dense:
                wf_list = []
                for A in mw_groups[gi]['As']:
                    vals = 0.5 * np.asarray(A[cp['nz_i'], cp['nz_j']])
                    dtype = float if np.isrealobj(vals) else complex
                    Kcw = np.zeros((self.dim, self.dim), dtype)
                    seq = np.tile(np.repeat(vals, cp['n_p']), self.Ds[gi] - 1)
                    Kcw[cp['r'], cp['c']] += seq
                    Kcw[cp['c'], cp['r']] += np.conj(seq)
                    wf_list.append(Kcw)
                self.Kc_dense.append(wf_list)
            else:
                self.Kc_dense.append(None)
        # 光场单位矩阵（A_opt ⊗ I_P，块对角重复）
        self.opt_ext = []
        for A_opt, _, _, _ in opt_terms:
            if self.dense:
                self.opt_ext.append(
                    np.kron(np.eye(P), np.asarray(A_opt, float)))
            else:
                self.opt_ext.append(
                    sp.kron(sp.eye(P, format='csr'),
                            sp.csr_matrix(A_opt)).tocsr())

    def c_mid(self, groups, opt_terms, mid):
        """段中点：微波各包络和 + 光场各包络和。"""
        c_mw = [sum(float(np.asarray(wf(np.atleast_1d(mid))).flat[0])
                    for wf in g['envs']) for g in groups]
        c_opt = [float(np.asarray(wf(np.atleast_1d(mid))).flat[0])
                 for _, wf, _, _ in opt_terms]
        return c_mw, c_opt

    def hamiltonian_seg(self, mw_groups, opt_terms, mid):
        """段哈密顿（稠密）：K0 + Σ_w env_w(mid)·Kcw_w + Σ o env_o(mid)·Ke_o。
        修复同频合并bug：逐波形配对 env_w × Kcw 而非 (Σenv)×Kc_merged。"""
        if self.dense:
            any_c = False
            K = self.K0.copy()
            for gi, g in enumerate(mw_groups):
                Kcw_list = self.Kc_dense[gi]
                if Kcw_list is None:
                    continue
                for wf, Kcw in zip(g['envs'], Kcw_list):
                    c_w = float(np.asarray(wf(np.atleast_1d(mid))).flat[0])
                    if c_w != 0.0:
                        K = K + c_w * Kcw
                        any_c = True
            for oi, (A_opt, wf, _, _) in enumerate(opt_terms):
                c_o = float(np.asarray(wf(np.atleast_1d(mid))).flat[0])
                if c_o != 0.0:
                    K = K + c_o * self.opt_ext[oi]
                    any_c = True
            if not any_c:
                return ('diag', None)
            return ('dense', K)
        raise NotImplementedError(
            '两场引擎稀疏路径待实现（当前维度应走稠密）')


def _pops_of(ps):
    return np.abs(ps) ** 2


def _embed(psi, setup):
    X = np.zeros(setup.dim, complex)
    off = setup.N * setup.p_center
    X[off:off + setup.N] = psi
    return X


def _reconstruct(setup, X, tq):
    Xm = np.asarray(X, complex).reshape(setup.P, setup.N)
    return np.exp(-1j * setup.nphase * float(tq)) @ Xm


def evolve_two_field(states, fields_meta, psi0, t_grid, tol=1e-6,
                     progress_arr=None, log=None, record_amp=False):
    """两场引擎主入口（method='two_field'）。

    fields_meta : build_two_field 的输出（E_rot, opt_terms, mw_groups, N）。
    psi0         : 物理态索引/(n,l,j,m)/复向量。
    返回与 evolve_exact 同构 dict（method='two_field'）。
    """
    log = log or (lambda *a: None)
    t = np.asarray(t_grid, float)
    N = fields_meta['N']
    E_rot = fields_meta['E_rot']
    opt_terms = fields_meta['opt_terms']
    mw_groups = fields_meta['mw_groups']
    psi0v = _evolve._as_vector(states, psi0)
    T = float(t[-1])
    tol = float(tol)

    # 微波光子块阶数：解析界（近共振 r），至少 M=2（多光子显式）
    E = np.array([s['E_mhz'] for s in states], float)

    def _peak_r(g):
        vmax = 0.0
        for A in g['As']:
            Anz = np.abs(A) > 1e-12
            rows, cols = np.nonzero(Anz)
            if rows.size == 0:
                continue
            det = np.abs((E[rows] - E[cols]) - g['omega_mhz'])
            m = det < MW_NRES_BW_MHZ
            if m.any():
                vmax = max(vmax, float(np.abs(A[rows[m], cols[m]]).max()))
        return vmax / max(abs(g['omega_mhz']), 1e-9)

    Ms = []
    for g in mw_groups:
        r = _peak_r(g)
        # 多光子显式：M ≥ 2（k=±2）；若 r 大到需要更高阶则升
        M = 2
        while (r / 4.0) ** (2 * (M - 1)) > tol and M < _M_MAX:
            M += 1
        Ms.append(M)
    log(f'[two_field] 微波光子块 M={Ms}（扇区 {setup_P(Ms)}'
        + (f'，k=±{max(Ms)-1} 多光子显式' if Ms else '，无微波）'))
    if not mw_groups:
        # 无微波：纯光场旋转系演化（无光子块）
        return _evolve_opt_only(states, fields_meta, psi0v, t, tol, log)

    setup = _TwoFieldSetup(E_rot, N, mw_groups, Ms, opt_terms)
    log(f'[two_field] 扩展维 = {N} × {setup.P} = {setup.dim}')
    # 读取光学不连续点（由 build_two_field 收集）
    disc_times = list(fields_meta.get('disc_times', []))

    pops = np.zeros((len(t), N))
    pops[0] = _pops_of(psi0v)
    report = dict(n_segments=0, max_richardson_diff=0.0, dt_min_hit=False,
                  max_norm_dev=0.0)
    h_floor = max(T / 2e5, 1e-9)

    def seg_decomp(ta, tb):
        mid = 0.5 * (ta + tb)
        # 计算各波形单独包络（用于 hamiltonian_seg 逐波配对，修复合并 bug）
        c_mw = [sum(float(np.asarray(wf(np.atleast_1d(mid))).flat[0])
                    for wf in g['envs']) for g in mw_groups]
        c_opt = [float(np.asarray(wf(np.atleast_1d(mid))).flat[0])
                 for _, wf, _, _ in opt_terms]
        kind, K = setup.hamiltonian_seg(mw_groups, opt_terms, mid)
        if kind == 'diag':
            return ('diag', None, c_mw, c_opt)
        evals, evecs = np.linalg.eigh(K)
        return ('dense', (evals, evecs), c_mw, c_opt)

    def defect_sweep(dec, ta, tb, X, nodes, ws):
        """ΔK = Σ_w Δc_w·Kcw_w + Σ_o Δc_o·(A_opt⊗I) 的本征基缺陷。
        修复：逐波形配对 env_w(nodes)-env_w(mid) × Kcw_w 而非 (Σenv-Σmid)×Kc_merged。"""
        _, (evals, evecs), c_mw, c_opt = dec
        a0 = evecs.conj().T @ X
        Psi = (evecs @ (a0[:, None] * np.exp(
            -1j * _CONV * np.outer(evals, nodes - ta)))).T   # (n, dim)
        V = np.zeros_like(Psi)
        mid = 0.5 * (ta + tb)
        # 微波包络差（逐波配对）
        for gi, g in enumerate(mw_groups):
            Kcw_list = setup.Kc_dense[gi]
            if Kcw_list is None:
                continue
            V_g = np.zeros_like(Psi)
            has_active = False
            for wf, Kcw in zip(g['envs'], Kcw_list):
                dc_w = np.asarray(wf(nodes), float) - float(np.asarray(wf(np.atleast_1d(mid)), float).flat[0])
                if np.max(np.abs(dc_w)) > 1e-14:
                    V_g = V_g + dc_w[:, None] * (Psi @ Kcw.T)
                    has_active = True
            if has_active:
                V = V + V_g
        # 光场包络差（单波形组，不变）
        for oi, (A_opt, wf, _, _) in enumerate(opt_terms):
            Ke = setup.opt_ext[oi]
            dc = np.asarray(wf(nodes), float) - c_opt[oi]
            if np.max(np.abs(dc)) > 1e-14:
                V = V + dc[:, None] * (Psi @ Ke.T)
        G = V @ evecs.conj()
        Ph = np.exp(1j * _CONV * np.outer(evals, nodes))
        Csum = np.cumsum(Ph * (G * ws[:, None]).T, axis=1)
        S = Csum[:, -1]
        return S

    def defect_at(S, evals, evecs, tq):
        return evecs @ (-1j * _CONV
                        * np.exp(-1j * _CONV * evals * tq) * S)

    X = _embed(psi0v, setup)
    tprev = 0.0
    for iout in range(1, len(t)):
        if progress_arr is not None:
            progress_arr[0] = min(tprev / T, 0.999)
        tb = float(t[iout])

        def _propagate_seg(ta, tb, X):
            if progress_arr is not None and progress_arr[1] != 0.0:
                raise RuntimeError('用户取消')
            dec = seg_decomp(ta, tb)
            if dec[0] == 'diag':
                report['n_segments'] += 1
                return X * np.exp(-1j * _CONV * (tb - ta) * setup.diag)
            evals, evecs = dec[1]
            a0 = evecs.conj().T @ X
            psi_frozen = evecs @ (a0 * np.exp(-1j * _CONV * (tb - ta) * evals))
            est = 0.0
            psi_out = psi_frozen
            prev = None
            for n_nodes in (32, 64, 128):
                glx, glw = _gl_nodes(n_nodes)
                nodes = ta + 0.5 * (tb - ta) * (glx + 1.0)
                ws = 0.5 * (tb - ta) * glw
                S = defect_sweep(dec, ta, tb, X, nodes, ws)
                cur = psi_frozen + defect_at(S, evals, evecs, tb)
                if prev is not None:
                    est = float(np.linalg.norm(cur - prev))
                    psi_out = cur
                    if est <= tol:
                        break
                else:
                    psi_out = cur
                prev = cur
            if est > tol and (tb - ta) > h_floor:
                tm = 0.5 * (ta + tb)
                Xm = _propagate_seg(ta, tm, X)
                return _propagate_seg(tm, tb, Xm)
            report['max_richardson_diff'] = max(
                report['max_richardson_diff'], est)
            report['n_segments'] += 1
            if est > tol:
                report['dt_min_hit'] = True
            return psi_out

        X = _propagate_seg(tprev, tb, X)
        ps = _reconstruct(setup, X, tb)
        p = _pops_of(ps)
        pops[iout] = p
        report['max_norm_dev'] = max(report['max_norm_dev'],
                                     float(abs(p.sum() - 1.0)))
        tprev = tb
    if progress_arr is not None:
        progress_arr[0] = 1.0
    report['photon_blocks'] = {
        f'{g["omega_mhz"]:.6g}MHz': setup.Ds[gi]
        for gi, g in enumerate(mw_groups)}
    report['tol'] = tol
    out = dict(t=t, pops=pops, states=states,
               om_mhz=[float(max(np.abs(A).max() for A in g['As']))
                       for g in mw_groups],
               method='two_field',
               backend='tf-sambe-dense' if setup.dense else 'tf-sambe-sparse',
               loss=np.zeros(len(t)), error_report=report)
    if record_amp:
        out['psi_amp'] = _reconstruct(setup, X, T)
    return out


def _evolve_opt_only(states, fields_meta, psi0v, t, tol, log):
    """无微波：纯光场旋转系（无光子块），直接自适应 RK 风格（简单实现：solve_ivp 参考）。"""
    from scipy.integrate import solve_ivp
    N = fields_meta['N']
    E_rot = fields_meta['E_rot']
    opt_terms = fields_meta['opt_terms']
    E_rot = E_rot - float(E_rot.min())

    def _rhs_twofield(tv, y):
        V = np.zeros((N, N), complex)
        for A_opt, wf, _, _ in opt_terms:
            c = float(np.asarray(wf(np.atleast_1d(tv))).flat[0])
            if c != 0.0:
                V = V + c * A_opt
        u = np.exp(1j * _CONV * E_rot * tv)
        return -1j * _CONV * (u * (V @ (np.conj(u) * y)))

    sol = solve_ivp(_rhs_twofield, (0.0, float(t[-1])), psi0v, method='DOP853',
                    t_eval=t, rtol=1e-9, atol=1e-11)
    if not sol.success:
        raise RuntimeError(f'two_field(opt-only) 积分失败: {sol.message}')
    pops = np.abs(sol.y).T ** 2
    log(f'[two_field] 纯光场（无微波）：DOP853 参考路径 {len(t)} 点')
    report = dict(n_segments=1, max_richardson_diff=0.0, dt_min_hit=False,
                  max_norm_dev=float(np.max(np.abs(pops.sum(axis=1) - 1.0))))
    return dict(t=t, pops=pops, states=states,
                om_mhz=[float(np.abs(A_opt).max()) for A_opt, *_ in opt_terms],
                method='two_field', backend='tf-optonly-dop853',
                loss=np.zeros(len(t)), error_report=report)


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
    psi0v = _evolve._as_vector(states, psi0)
    T = float(t[-1])
    # t24：耗散（落点=丢失）——Gamma_ns（/ns），RHS 加 −Γ/2·y
    has_decay = bool(decays)
    if has_decay:
        from .hard import _single_loss_rates
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
            c = float(np.asarray(wf(np.atleast_1d(tv))).flat[0])
            if c != 0.0:
                d = omega_f - omega_ref
                if d != 0.0 and m is not None:
                    ph = np.exp(1j * _CONV * d * tv
                                * (m[:, None] - m[None, :]))
                    V = V + c * ph * A_opt
                else:
                    V = V + c * A_opt
        for A, wL, wf, ph_fn in mw_terms:
            env = float(np.asarray(wf(np.atleast_1d(tv))).flat[0])
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
        progress_arr[0] = 0.15
    sol = solve_ivp(_rhs_twofield, (0.0, T), psi0v, method='DOP853',
                    t_eval=t, rtol=rtol, atol=atol)
    if not sol.success:
        raise RuntimeError(f'hard 积分失败: {sol.message}')
    pops = np.abs(sol.y).T ** 2
    if progress_arr is not None:
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


# ---- t19 函数级改名兼容别名（旧名 = 新名，外部零改动）----
_TwoFieldSetup.K_at = _TwoFieldSetup.hamiltonian_seg
