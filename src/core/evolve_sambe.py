# -*- coding: utf-8 -*-
"""第一性原理含时求解引擎（evolve_exact.py）：零模型近似的单原子动态模拟。

物理模型（唯一定义——无 RWA、无旋转系、无近/远失谐分流、无静态 AC 注入）：

    H(t) = H0 + Σ_k Ω_k·env_k(t)·cos(ω_k t)·A_k                （实验室系，MHz）

AC 位移、Bloch–Siegert 位移、远失谐效应全部从精确传播中自然涌现，
不需要（也不允许）任何外部能移注入。

数值算法（两条互补路径，均第一性原理）：

1) 主力路径 `evolve_exact`：包络自适应分段 + Sambe 光子块扩展空间传播。
   - 时间轴按包络曲率自适应分段（与载波周期无关——光频载波不产生步数）；
     段内包络冻结于中点值，Sambe 哈密顿（含块对角载波相位）精确指数传播；
     链上状态用 Richardson 外推 (4·双子步−单步)/3 提阶至 O(dt⁴)，输出取双子步。
     误差判据 = 单步 vs 双子步的扩展态差范数（≈ 真误差界，无重构噪声）≤ tol。
   - 全程在扩展空间连续传播（边界不重嵌入，无微运动泄漏；幺正守恒）。
   - Sambe (t,t') 扩展空间 K_{n⃗,m⃗} = H_{n⃗−m⃗} − Σ_g n_g ω_g δ（时不变），
     载波被精确解析，段传播成本与段内载波周期数无关。
   - 光子块阶数 M：全传播 M vs M+1 对比递增收敛（布居差 ≤ tol）。
   - 两处误差（包络离散、光子块截断）均收敛可验；Σ|ψ|² 偏差 ≈ 截断误差量级，
     随结果 error_report 可见。
   - 全常数包络 → 单次扩展空间对角化，成本与总时长无关。

2) 参考路径 `evolve_reference`：H0 相互作用绘景 + 自适应高精度 ODE
   （幺正 DOP853 / 含耗散 Radau），rtol/atol 显式暴露，可选双容差验证。
   角色：回归层权威参考；method='exact' 含耗散时的第一期执行路径
   （段内耗散超算符留待第二期）。

场格式（kind='exact'，由 build_fields_exact 或手工构造）：
    {'kind': 'exact', 'A': N×N 实对称（峰值光强下各对真实 Rabi，MHz），
     'omega_L_mhz': 载频, 'waveform': t→[0,1] 包络 callable, 'label': str}

单位：与 evolve.py 一致（MHz, ns, rad/ns；_CONV = 2π×1e-3）。
"""
import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import expm_multiply

from . import static_atom as core
from . import sambe

_CONV = 2 * np.pi * 1e-3          # MHz·ns → rad
_POL_Q = {'π': 0, 'σ+': +1, 'σ-': -1}

# 光子块阶数：起始/上限（自适应收敛于两者之间）
_M_START = 2
_M_MAX = 8
# M 判定的近共振带宽（MHz）：离共振 Rydberg-Rydberg 巨矩阵元（如光学频下
# 60S↔60P 的 Ω~19 GHz）不参与 r 判定——其物理贡献是二阶微扰（~1e-9 相对），
# 光子块阶数由与载频近共振的对决定（Mizuta 2025 / Kuwahara–Mori–Saito 2016
# 收敛界针对共振耦合算子范数；离共振对的 Sambe 收敛在 M=1 已足够）。
_M_ANALYTIC_BW_MHZ = 2e4   # 20 GHz
# 扩展维 ≤ 此值走稠密 eigh，否则稀疏 expm_multiply
# （2026-08-29 提到 2200：逐载频 M 后 22 态 3 载频 = 45 扇区×22 = 990 维，
#  M+1 探针 setup 75 扇区×22 = 1650 维——仍须稠密，否则掉进 expm_multiply
#  THz 对角爆炸。2200 维实对称 eigh ~0.5 s，可接受。）
_DENSE_MAX = 2200
# 包络采样（活性判断/恒常判断）
_ENV_PROBE_N = 2001
# Richardson 递归深度保护
_DT_MIN_FRAC = 2e5
# 临时调试日志（非 None 时记录 (ta, dt, rms)）
_DBG_LOG = None


def _eigh_opt(K):
    """eigh 加速（t37）：Sambe 段 eigh 是小矩阵（~234-990 维），OpenBLAS
    多线程在此尺度是净损失（24 线程 16ms vs 1 线程 6ms @234 维——线程同步
    开销主导）。调用方应在**整个演化运行**外包 threadpool_limits(1)
    （逐段开关上下文的开销 ~15ms/段，反而更慢）；本 helper 保留为参考。
    数值结果与多线程一致（~1e-13 舍入内）。"""
    try:
        import threadpoolctl
        with threadpoolctl.threadpool_limits(limits=1):
            return np.linalg.eigh(K)
    except Exception:
        return np.linalg.eigh(K)


# ============================================================
# 场构造（无 near_cutoff / 无 AC 注入 / 无 scatter——全部补丁机制切割）
# ============================================================
def _sublevel_row(s):
    """states → core.dipole_amplitude 的子能级行（与 evolve 同款）。"""
    return dict(n=s['n'], l=s['l'], j=s['j'], kind=s['kind'],
                mj=s.get('mj'), mF=s.get('mF'), amps=s.get('amps'))


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
                '第一性引擎（method=exact）暂不支持任意波形'
                '（ArbitraryWave，无单载频概念）；请用 method=auto'
                '（自动转 arbitrary 路径）。')
        q = _POL_Q.get(w.pol, 0)
        E0_sq = core.field_e0_sq(float(w.intensity_mwcm2))
        A = np.zeros((N, N))
        for i in range(N):
            for j in range(i + 1, N):
                d = core.dipole_amplitude(
                    cache, _sublevel_row(states[i]), _sublevel_row(states[j]),
                    q)
                if abs(d) < 1e-12:
                    continue
                A[i, j] = A[j, i] = core.rabi_mhz(abs(d), E0_sq)
        if not np.any(A != 0.0):
            continue                       # 该偏振下无任何耦合 → 无效场
        out.append({'kind': 'exact', 'A': A,
                    'omega_L_mhz': float(w.freq_mhz),
                    'waveform': w.waveform,
                    'label': getattr(w, 'name', '波')})
    return out


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


# ============================================================
# Sambe 扩展空间（结构预计算 + 按段装配）
# ============================================================
class _SambeSetup:
    """光子块扩展空间结构（与段无关的部分只算一次）。

    基组 |i⟩⊗|n_1..n_G⟩，扁平索引 i + N·p，p = Σ_g n_g·stride_g
    （n_g ∈ 0..2M_g，对角用中心化量子数 n_g−M_g——与重构相位中的绝对
    n_g 恰好抵消各自的 M_g 偏移，规范自洽）。

    **逐载频 M**（2026-08-29）：M 可为 int（全部载频同阶，向后兼容）或
    list（每载频独立阶数 M_g）。混合进制张量积：P = Π(2M_g+1)，
    stride_g = Π_{g'<g}(2M_g'+1)。理论依据：Ho–Chu 多模 Floquet（各模
    独立收敛判据）+ Mizuta 2025 截断 Sambe 误差上界；光学载频
    r=Ω/ω~1e-7 → M=1 即单光子阶（误差 ~1e-13），微波 r~3e-4 → M=1..2
    （误差 ~1e-8..1e-11）——统一 M 会在 125 扇区里浪费 ~3 个数量级。
    """

    def __init__(self, E, groups, M):
        geo = sambe.photon_geometry(E, groups, M)
        self.E = np.asarray(E, float)
        self.N = len(E)
        self.groups = groups
        self.Ms = geo['Ms']
        self.Ds = geo['Ds']
        self.P = geo['P']
        self.dim = geo['dim']
        self.strides = geo['strides']
        self.nvecs = geo['nvecs']
        self.diag = geo['diag']
        self.nphase = geo['nphase']
        self.p_center = geo['p_center']
        # 耦合结构：每组 (r, c) 索引对 ↔ V_seg[nz_i, nz_j]·0.5
        self.coup = []
        G = len(groups)
        for gi, g in enumerate(groups):
            Dg = self.Ds[gi]
            Mg = self.Ms[gi]
            Vpat = np.zeros((self.N, self.N))
            for A in g['As']:
                Vpat = Vpat + (np.abs(A) > 1e-12)
            # 全部有序对（(a,b) 与 (b,a) 分属不同光子方向，都须正向加）；
            # 光子数减少方向 = 已加条目的转置，K_at 中统一对称化补齐
            ii, jj = np.nonzero(Vpat)
            stride = self.strides[gi]
            # 光子格点扁平索引与各组量子数（一次性构造，**混合进制**）
            grid = np.meshgrid(*[np.arange(d) for d in self.Ds],
                               indexing='ij')
            pflat = np.zeros(self.P, int)
            for gg in range(G):
                pflat = pflat + grid[gg].ravel() * self.strides[gg]
            ng_g = grid[gi].ravel()
            rows, cols = [], []
            for s in range(Dg - 1):
                p_src = pflat[ng_g == s]       # n_g == s 的全部格点
                p_dst = p_src + stride
                for a, b in zip(ii, jj):
                    # 正向：a 在 n=s → b 在 n+1（转置方向由对称化补齐）
                    rows.append(a + self.N * p_src)
                    cols.append(b + self.N * p_dst)
            rows = np.concatenate(rows) if rows else np.zeros(0, int)
            cols = np.concatenate(cols) if cols else np.zeros(0, int)
            # 值序列与行序（s 层 × 有序对 × p_src 格点）对齐：
            # 每层 = repeat(vals, n_p)，tile Dg−1 层
            self.coup.append(dict(r=rows, c=cols, nz_i=ii, nz_j=jj,
                                  n_p=self.P // Dg))
        self.K0 = np.diag(self.diag)
        self.dense = self.dim <= _DENSE_MAX
        # 单位包络耦合结构 Kc_g_wf（稠密）——每波形一个单位矩阵，与包络配对；
        # 修复：同频多波形合并 bug（原代码将 ΣA 合并后乘 Σenv，丢失逐波配对）。
        # Kc_dense[gi][w_idx] = 0.5·A_w 的扩展空间映射矩阵（与 envs[w_idx] 一一配对）
        self.Kc_dense = []
        for gi, cp in enumerate(self.coup):
            if cp['r'].size and self.dense:
                wf_list = []
                for A in groups[gi]['As']:
                    vals = 0.5 * np.asarray(A[cp['nz_i'], cp['nz_j']])
                    dtype = float if np.isrealobj(vals) else complex
                    Kc_w = np.zeros((self.dim, self.dim), dtype)
                    seq = np.tile(np.repeat(vals, cp['n_p']), self.Ds[gi] - 1)
                    Kc_w[cp['r'], cp['c']] += seq
                    Kc_w[cp['c'], cp['r']] += np.conj(seq)
                    wf_list.append(Kc_w)
                self.Kc_dense.append(wf_list)
            else:
                self.Kc_dense.append(None)
        # 全部耦合实数 → 段哈密顿实对称（eigh 提速）
        self.real_K = all(
            Kc_list is None or all(np.isrealobj(kw) for kw in Kc_list
                                   if kw is not None)
            for Kc_list in self.Kc_dense
        )

    def V_segs(self, groups, mid):
        """各组在段中点的冻结耦合矩阵 Σ_w env_w(mid)·A_w。"""
        Vs = []
        active = False
        for g in groups:
            V = np.zeros((self.N, self.N))
            for A, wf in zip(g['As'], g['envs']):
                c = float(np.asarray(wf(np.atleast_1d(mid))).flat[0])
                if c != 0.0:
                    V = V + c * A
            if np.any(V != 0.0):
                active = True
            Vs.append(V)
        return Vs, active

    def hamiltonian_seg(self, Vs):
        """段哈密顿（稠密或稀疏）。Vs 全零时返回 ('diag', None)。"""
        if not any(np.any(V != 0.0) for V in Vs):
            return ('diag', None)
        if self.dense:
            K = self.K0.copy()
            for gi, (V, cp) in enumerate(zip(Vs, self.coup)):
                if cp['r'].size:
                    vals = 0.5 * V[cp['nz_i'], cp['nz_j']]
                    seq = np.tile(np.repeat(vals, cp['n_p']),
                                  self.Ds[gi] - 1)
                    K[cp['r'], cp['c']] += seq
                    # 转置方向（光子数减少）：同值厄米补齐，不可对称化（会减半）
                    K[cp['c'], cp['r']] += np.conj(seq)
            return ('dense', K)
        rows = [np.arange(self.dim)]
        cols = [np.arange(self.dim)]
        data = [self.diag]
        for gi, (V, cp) in enumerate(zip(Vs, self.coup)):
            if cp['r'].size:
                rows.append(cp['r'])
                cols.append(cp['c'])
                vals = 0.5 * V[cp['nz_i'], cp['nz_j']]
                data.append(np.tile(np.repeat(vals, cp['n_p']),
                                    self.Ds[gi] - 1))
                # 转置方向（光子数减少）同样入 COO（同值，厄米）
                rows.append(cp['c'])
                cols.append(cp['r'])
                data.append(np.tile(np.repeat(np.conj(vals), cp['n_p']),
                                    self.Ds[gi] - 1))
        K = sp.coo_matrix((np.concatenate(data),
                           (np.concatenate(rows), np.concatenate(cols))),
                          shape=(self.dim, self.dim)).tocsr()
        return ('sparse', K)


def _embed(psi, setup):
    """物理态 → 扩展空间中央光子块（量子数 n=0；嵌入自由度取单块非零，
    重构求和 Σ_n c_n e^{−inωt} 在嵌入时刻精确还原 ψ）。"""
    X = np.zeros(setup.dim, complex)
    off = setup.N * setup.p_center
    X[off:off + setup.N] = psi
    return X


def _apply_prop(prop, setup, psi, ta, tq):
    """段传播 + (t,t') 重构：ψ(ta) → ψ(tq[])（(nq, N) 复矩阵）。

    重构 ψ_i(t) = Σ_p e^{i·nphase[p]·t}·φ_{p,i}(t)——Sambe 展开的物理投影。
    """
    tq = np.asarray(tq, float)
    if prop[0] == 'diag':
        ph = np.exp(-1j * _CONV * np.outer(tq - ta, setup.E))   # (nq, N)
        return psi[None, :] * ph
    if prop[0] == 'dense':
        evals, evecs = prop[1]
        a0 = evecs.conj().T @ _embed(psi, setup)
        dt = tq - ta
        A = np.exp(-1j * _CONV * np.outer(evals, dt))           # (dim, nq)
        X = evecs @ (a0[:, None] * A)                           # (dim, nq)
        Xm = X.reshape(setup.P, setup.N, len(tq))
        # 重构 ψ_i(t) = Σ_n c_n(t)·e^{−inωt}（Sambe 展开，负相位）
        PH = np.exp(-1j * np.outer(setup.nphase, tq))           # (P, nq)
        return np.einsum('pq,piq->iq', PH, Xm).T
    # 稀疏路径：逐查询 expm_multiply（后备；大基组/多频场景）
    out = np.zeros((len(tq), setup.N), complex)
    X0 = _embed(psi, setup)
    for k, t in enumerate(tq):
        X = expm_multiply(-1j * _CONV * (t - ta) * prop[1], X0)
        Xm = X.reshape(setup.P, setup.N)
        out[k] = (np.exp(-1j * setup.nphase * t) @ Xm)
    return out


def _pops_of(ps):
    return np.abs(ps) ** 2


# ============================================================
# 光子块阶数：常数包络探针（单次对角化，截断差随时变包络的上界）
# ============================================================
def _peak_const_groups(groups, T):
    """全部包络 → 峰值常数的镜像组（探针用）。"""
    out = []
    tg = np.linspace(0.0, T, _ENV_PROBE_N)
    for g in groups:
        peaks = [float(np.max(np.abs(np.asarray(wf(tg), float))))
                 for wf in g['envs']]
        out.append(dict(
            omega_mhz=g['omega_mhz'], As=list(g['As']),
            envs=[(lambda c: (lambda t: np.full_like(
                np.asarray(t, float), c)))(pk) for pk in peaks],
            labels=g['labels']))
    return out


def _probe_pops(E, probe_groups, M, psi0, T):
    """探针场景单次对角化布居（五个采样时刻）。"""
    st = _SambeSetup(E, probe_groups, M)
    Vs, active = st.V_segs(probe_groups, 0.5 * T)
    if not active:
        return np.tile(_pops_of(np.asarray(psi0, complex)), (5, 1))
    kind, K = st.hamiltonian_seg(Vs)
    if kind == 'dense':
        prop = ('dense', np.linalg.eigh(K))
    else:
        prop = ('sparse', K)
    ps = _apply_prop(prop, st, np.asarray(psi0, complex), 0.0,
                     T * np.array([0.1, 0.3, 0.5, 0.7, 1.0]))
    return np.abs(ps) ** 2


def _group_res_peak(g, E_mhz, bw_mhz):
    """组内与载频**近共振**（|ΔE−ω|<bw）的对的最大 |Ω|；E_mhz None → 全局峰值。

    离共振 Rydberg-Rydberg 巨矩阵元（60S↔60P 在光学频 Ω~19 GHz，|ΔE−ω|~295 THz）
    被排除——其物理贡献为二阶微扰，光子块阶数只由共振对决定。"""
    if E_mhz is None:
        return max(float(np.abs(A).max()) for A in g['As'])
    E = np.asarray(E_mhz, float)
    peak = 0.0
    for A in g['As']:
        Anz = np.abs(A) > 1e-12
        rows, cols = np.nonzero(Anz)
        if rows.size == 0:
            continue
        det = np.abs((E[rows] - E[cols]) - g['omega_mhz'])
        m = det < bw_mhz
        if m.any():
            peak = max(peak, float(np.abs(A[rows[m], cols[m]]).max()))
    return peak


def _m_analytic(groups, tol, E_mhz=None):
    """逐载频解析下界：(r_g/4)^{2(M_g−1)} ≤ tol（r_g = 近共振对最大 Ω/ω）。

    返回与 groups 等长的 M 列表——**每组独立**：光学载频（近共振 Ω~104 MHz，
    r~3.5e-7）→ M=1（3 块，单光子阶即收敛）；微波 r~5e-4 → M=2（5 块）。
    离共振巨矩阵元（光学频 60S↔60P，Ω~19 GHz）不参与判定（_group_res_peak）。
    统一 M 会把光学拖到微波阶数（5³=125 扇区），逐载频 M=(1,1,2) → 45 扇区。
    """
    Ms = []
    for g in groups:
        vmax = _group_res_peak(g, E_mhz, _M_ANALYTIC_BW_MHZ)
        r = vmax / max(abs(g['omega_mhz']), 1e-9)
        M = _M_START - 1                      # 从 M=1（3 块）起步
        if r > 0.0:
            # M=1 的截断误差 ~ (r/4)²（单光子阶，|k|≥2 谐波）；
            # 若已 ≤ tol 停留 M=1；否则按 (r/4)^{2(M−1)} 升阶。
            if (r / 4.0) ** 2 > tol:
                M = 2
                while (r / 4.0) ** (2 * (M - 1)) > tol and M < _M_MAX:
                    M += 1
        Ms.append(max(1, M))
    return Ms


def setup_P(Ms):
    """逐载频 M 列表 → 扇区数 Π(2M_g+1)。"""
    p = 1
    for m in Ms:
        p *= (2 * int(m) + 1)
    return p


# ============================================================
# 包络自适应分段主驱动（冻结中点精确指数 + Richardson 外推 + 减半控制）
# ============================================================
# Gauss-Legendre 节点/权重缓存（缺陷修正求积；节点数自适应倍增）
_GL_CACHE = {}


def _gl_nodes(n):
    """n 点 Gauss-Legendre 节点/权重（[−1,1]，缓存）。"""
    gl = _GL_CACHE.get(n)
    if gl is None:
        from numpy.polynomial.legendre import leggauss
        gl = leggauss(n)
        _GL_CACHE[n] = gl
    return gl


def _seg_decomp(setup, groups, ta, tb):
    """段冻结中点哈密顿 K(c_mid) 的对角分解。
    返回 ('diag', None) 或 ('dense', evals, evecs, c_mids)。"""
    mid = 0.5 * (ta + tb)
    if setup.dense:
        active = False
        K = setup.K0.copy()
        c_mids = []
        for g, Kc_list in zip(groups, setup.Kc_dense):
            if Kc_list is None:
                c_mids.append(0.0)
                continue
            # 逐波配对：c_w = env_w(mid)，K_cw = Kc_dense[gi][w]（修复合并 bug）
            Kc_sum = np.zeros_like(Kc_list[0]) if Kc_list[0] is not None else None
            c_total = 0.0
            active_g = False
            for wf, Kcw in zip(g['envs'], Kc_list):
                c_w = float(np.asarray(wf(np.atleast_1d(mid))).flat[0])
                c_total += c_w
                if c_w != 0.0 and Kcw is not None:
                    Kc_sum = Kc_sum + c_w * Kcw if Kc_sum is not None else c_w * Kcw
                    active_g = True
            c_mids.append(c_total)
            if active_g:
                K = K + Kc_sum
                active = True
        if not active:
            return ('diag', None)
        evals, evecs = np.linalg.eigh(K)
        return ('dense', evals, evecs, c_mids)
    Vs, active = setup.V_segs(groups, mid)
    if not active:
        return ('diag', None)
    kind, K = setup.hamiltonian_seg(Vs)
    if kind == 'dense':
        evals, evecs = np.linalg.eigh(K)
        return ('dense', evals, evecs, None)
    return ('sparse', K, None, None)


def _frozen_apply(evals, evecs, a0, ta, tq):
    """冻结传播子作用：e^{−iC K(c_mid)(tq−ta)}·X(ta)。"""
    return evecs @ (a0 * np.exp(-1j * _CONV * evals * (tq - ta)))


def _defect_sweep(setup, groups, ta, tb, nodes, ws, evals, evecs, c_mids,
                  psi_nodes):
    """单次缺陷扫描（本征基可分离形式，O(dim·n)）：

    δ(t) = −iC∫_{ta}^{t} U_f(t,s)·ΔK(s)·ψ(s) ds，
    本征基分量 δ_r(t) = −iC·e^{−iCε_r t}·S_r，
    S_r = Σ_q w_q·e^{+iCε_r t_q}·(evecs†·ΔK(t_q)·ψ(t_q))_r——对固定节点组
    一次求和即得任意目标时刻的 δ（无嵌套）；节点级 δ 满足 Volterra 上限
    （仅用 ≤ 该节点的求积点，累积和实现）。返回 (S, delta_nodes)：
    S = 全区间和（供目标时刻求值）。
    """
    nodes = np.asarray(nodes, float)
    ws = np.asarray(ws, float)
    Psi = np.asarray(psi_nodes, complex)                 # (n, dim)
    V = np.zeros_like(Psi)
    mid = 0.5 * (ta + tb)                                  # 段中点（用于逐波修正）
    for g, Kc_list, c_mids_g in zip(groups, setup.Kc_dense, c_mids):
        if Kc_list is None:
            continue
        # 逐波配对：dc_w(q) = env_w(nodes[q]) − env_w(mid)，乘对应 Kcw（修复合并bug）
        V_g = np.zeros_like(Psi)
        has_active = False
        for wf, Kcw in zip(g['envs'], Kc_list):
            dc_w = np.asarray(wf(nodes), float) - float(np.asarray(wf(np.atleast_1d(mid)), float).flat[0])
            if Kcw is not None and np.max(np.abs(dc_w)) > 1e-14:
                V_g = V_g + dc_w[:, None] * (Psi @ Kcw.T)
                has_active = True
        if has_active:
            V = V + V_g
    G = V @ evecs.conj()                                 # (n, dim) evecs†·v_q
    Ph = np.exp(1j * _CONV * np.outer(evals, nodes))     # (dim, n)
    Csum = np.cumsum(Ph * (G * ws[:, None]).T, axis=1)   # (dim, n) 累积和
    S = Csum[:, -1]                                      # 全区间和（本征基）
    Phn = np.exp(-1j * _CONV * np.outer(evals, nodes))   # (dim, n)
    # 节点级 δ（物理基，满足 Volterra 上限）
    delta_nodes = (evecs @ (-1j * _CONV * (Phn * Csum))).T   # (n, dim)
    return S, delta_nodes


def _defect_at(setup, S, evals, evecs, tq):
    """由缺陷扫描结果 S（本征基）求物理基 δ(tq)。"""
    return evecs @ (-1j * _CONV * np.exp(-1j * _CONV * evals * tq) * S)


def _propagate_adaptive(setup, groups, psi0, t_grid, tol, progress_arr, log,
                        frozen_h_ns=None):
    """自适应分段演化：全程在 Sambe 扩展空间连续传播（边界不重嵌入，
    无微运动泄漏；幺正守恒），仅在输出时刻投影物理态。

    段传播 = 冻结中点精确指数 + Duhamel 缺陷修正（真实包络差在段内 Gauss
    求积，块对角载波相位由冻结指数精确承载）；求积节点数自适应倍增（被积
    函数含本征拍频振荡，节点须解析），节点加密差即误差估计：≤ tol 接受，
    否则真中点细分（细分同时降低所需节点数）——可细分到输出网格以下，
    直到残差 ≤ tol 或触及绝对下限。

    **frozen_h_ns（t36 方案 1）**：>0 时切换"纯冻结"模式——固定 h 内部细分，
    每段只做中点对角化 + 精确指数（无缺陷修正/无 Gauss 节点/无 Richardson）。
    缺陷修正的 Gauss 核含 THz 光子块本征跨度 → 在 THz 场景段被细分到
    h_floor 仍不收敛；纯冻结误差 = 段内包络变化（σ 缓变场景 h=1ns 布居
    偏差 ~1e-7），与 THz 解耦。默认 None = 原缺陷路径（行为不变）。
    """
    T = float(t_grid[-1])
    t = np.asarray(t_grid, float)
    N = setup.N
    psi0 = np.asarray(psi0, complex)
    pops = np.zeros((len(t), N))
    pops[0] = _pops_of(psi0)
    report = dict(n_segments=0, max_richardson_diff=0.0, dt_min_hit=False,
                  max_norm_dev=0.0)
    if frozen_h_ns is not None and frozen_h_ns > 0:
        report['frozen_h_ns'] = float(frozen_h_ns)
    h_floor = max(T / _DT_MIN_FRAC, 1e-9)

    def _frozen_seg(ta, tb, X):
        """纯冻结段：中点 K 一次对角化 + 精确指数（无缺陷修正）。"""
        if progress_arr[1] != 0.0:
            raise RuntimeError('用户取消')
        dec = _seg_decomp(setup, groups, ta, tb)
        if dec[0] == 'diag':
            return X * np.exp(-1j * _CONV * (tb - ta) * setup.diag)
        if dec[0] == 'sparse':
            _, K, _, _ = dec
            return expm_multiply(-1j * _CONV * (tb - ta) * K, X)
        _, evals, evecs, _ = dec
        a0 = evecs.conj().T @ X
        return evecs @ (a0 * np.exp(-1j * _CONV * evals * (tb - ta)))

    def _propagate_seg(ta, tb, X):
        """自适应传播 X(ta) → X(tb)（无输出；返回修正后链状态）。"""
        if progress_arr[1] != 0.0:
            raise RuntimeError('用户取消')
        if frozen_h_ns is not None and frozen_h_ns > 0:
            # t36：纯冻结——固定 h 内部细分（不依赖缺陷修正收敛；段长由
            # 包络缓变决定，与 THz 本征跨度解耦）
            nsub = max(1, int(np.ceil((tb - ta) / float(frozen_h_ns))))
            hh = (tb - ta) / nsub
            report['n_segments'] += nsub
            for s in range(nsub):
                X = _frozen_seg(ta + s * hh, ta + (s + 1) * hh, X)
            return X
        dec = _seg_decomp(setup, groups, ta, tb)
        if dec[0] == 'diag':
            report['n_segments'] += 1
            return X * np.exp(-1j * _CONV * (tb - ta) * setup.diag)
        if dec[0] == 'sparse':
            _, K, _, _ = dec
            return expm_multiply(-1j * _CONV * (tb - ta) * K, X)
        _, evals, evecs, c_mids = dec
        a0 = evecs.conj().T @ X

        def frozen_at(tq):
            return _frozen_apply(evals, evecs, a0, ta, tq)

        psi_prev = None
        psi_out = frozen_at(tb)
        est = 0.0
        for n_nodes in (32, 64, 128):
            glx, glw = _gl_nodes(n_nodes)
            nodes = ta + 0.5 * (tb - ta) * (glx + 1.0)
            ws = 0.5 * (tb - ta) * glw
            # 节点冻结态（批量）
            F = evecs @ (a0[:, None] * np.exp(
                -1j * _CONV * np.outer(evals, nodes - ta)))   # (dim, n)
            psi_nodes = F.T
            # 两级缺陷扫描：一级用冻结被积态，二级用修正后节点态（Volterra）
            S1, d_nodes1 = _defect_sweep(
                setup, groups, ta, tb, nodes, ws, evals, evecs, c_mids,
                psi_nodes)
            S2, _ = _defect_sweep(
                setup, groups, ta, tb, nodes, ws, evals, evecs, c_mids,
                psi_nodes + d_nodes1)
            psi_out = frozen_at(tb) + _defect_at(setup, S2, evals, evecs, tb)
            if psi_prev is not None:
                est = float(np.linalg.norm(psi_out - psi_prev))
                if est <= tol:
                    break
            psi_prev = psi_out
        if est > tol and (tb - ta) > h_floor:
            tm = 0.5 * (ta + tb)
            Xm = _propagate_seg(ta, tm, X)
            return _propagate_seg(tm, tb, Xm)
        report['max_richardson_diff'] = max(
            report['max_richardson_diff'], est)
        report['n_segments'] += 1
        if est > tol:
            report['dt_min_hit'] = True
        if _DBG_LOG is not None:
            _DBG_LOG.append((ta, tb - ta, est))
        return psi_out

    X = _embed(psi0, setup)
    tprev = 0.0
    for i in range(1, len(t)):
        progress_arr[0] = min(tprev / T, 0.999)
        X = _propagate_seg(tprev, float(t[i]), X)
        ps = _reconstruct(setup, X, t[i])
        p = _pops_of(ps)
        pops[i] = p
        report['max_norm_dev'] = max(report['max_norm_dev'],
                                    float(abs(p.sum() - 1.0)))
        tprev = float(t[i])
    psi_f = _reconstruct(setup, X, T)
    progress_arr[0] = 1.0
    return pops, psi_f, report


def _reconstruct(setup, X, tq):
    """扩展态 → 物理态（(t,t') 投影 ψ_i = Σ_n X_{n,i} e^{−inω t}）。"""
    Xm = np.asarray(X, complex).reshape(setup.P, setup.N)
    return np.exp(-1j * setup.nphase * float(tq)) @ Xm


def _env_all_constant(groups, T):
    for g in groups:
        for wf in g['envs']:
            v = np.asarray(wf(np.linspace(0.0, T, 64)), float)
            if np.ptp(v) > 1e-9:
                return False
    return True


# ============================================================
# 单次全传播（给定光子块阶数；供主入口与 M 收敛对比）
# ============================================================
def _run_sambe(E, groups, M, psi0, t, tol, progress_arr, log,
               frozen_h_ns=None):
    """返回 (pops, psi_f, report, setup)。frozen_h_ns：t36 纯冻结模式段长。"""
    setup = _SambeSetup(E, groups, M)
    T = float(t[-1])
    if _env_all_constant(groups, T):
        # 全常数包络 → 单次扩展空间对角化（成本与时长无关）
        Vs, active = setup.V_segs(groups, 0.5 * T)
        if active:
            kind, K = setup.hamiltonian_seg(Vs)      # 常数包络：冻结即精确
        else:
            kind = 'diag'
        if kind == 'dense':
            prop = ('dense', np.linalg.eigh(K))
        elif kind == 'sparse':
            prop = ('sparse', K)
        else:
            prop = ('diag', None)
        ps = _apply_prop(prop, setup, psi0, 0.0, t)
        pops = np.abs(ps) ** 2
        report = dict(n_segments=1, max_richardson_diff=0.0,
                      dt_min_hit=False,
                      max_norm_dev=float(np.max(
                          np.abs(pops.sum(axis=1) - 1.0))))
        return pops, ps[-1], report, setup
    pops, psi_f, report = _propagate_adaptive(
        setup, groups, psi0, t, tol, progress_arr, log, frozen_h_ns)
    return pops, psi_f, report, setup


# ============================================================
# 主入口：method='exact'（无耗散；含耗散转参考路径）
# ============================================================
def evolve_sambe(states, fields, psi0, t_grid, tol=1e-6, decays=None,
                 progress_arr=None, log=None, record_amp=False,
                 E_shift=None, frozen_h_ns=None):
    """第一性原理单原子含时演化（无模型近似）。

    fields : kind='exact' 场列表（build_fields_exact 输出或手工同构）
    tol    : 布居容差（包络冻结 + 光子块截断两处共用）
    decays : 非空 → 转 evolve_reference（Lindblad，第一期执行路径）
    frozen_h_ns : t36 方案 1——>0 时启用"纯冻结分段"（砍缺陷修正，固定
        段长 h；适合 THz 光子块 × 时变包络场景如 STIRAP，缺陷修正的
        Gauss 核在 THz 本征跨度下死局）。None = 原缺陷路径（默认不变）。
    返回 dict(t, pops, states, om_mhz, method='exact', backend, loss,
              error_report, [psi_amp | rho])——与 evolve.evolve 同构。
    """
    from . import evolve as _evolve
    log = log or (lambda *a: None)
    t = np.asarray(t_grid, float)
    if progress_arr is None:
        progress_arr = np.zeros(2)
    _validate_fields(fields)
    psi0v = _evolve._as_vector(states, psi0)

    if decays:
        log(f'[exact] 含 {len(decays)} 个衰变通道 → 转参考路径'
            '（H0 相互作用绘景 + 自适应主方程积分，同为第一性原理）')
        return evolve_reference(states, fields, psi0v, t, decays=decays,
                                progress_arr=progress_arr,
                                record_amp=record_amp)

    E = np.array([s['E_mhz'] for s in states], float)
    if E_shift is not None:
        E = E + np.asarray(E_shift, float)
    E = E - float(E.min())                    # 公共偏移 = 全局相位（精确等价）

    if not fields:
        pops = np.zeros((len(t), len(states)))
        pops[:] = _pops_of(psi0v)
        out = dict(t=t, pops=pops, states=states, om_mhz=[],
                   method='exact', backend='none', loss=np.zeros(len(t)),
                   error_report={'n_segments': 0, 'photon_blocks': {},
                                 'max_richardson_diff': 0.0})
        if record_amp:
            out['psi_amp'] = psi0v.copy()
        return out

    groups = _group_by_freq(fields)
    T = float(t[-1])
    tol = float(tol)

    # ---- 光子块收敛：逐载频常数包络探针 ----
    # 每载频独立递增（固定其他组）：光学 r~1e-7 → M=1 即收敛；
    # 微波 r~3e-4 → M=1..2。避免统一 M 把光学拖到微波阶数（125 扇区）。
    Ms = _m_analytic(groups, tol, E)
    probe_groups = _peak_const_groups(groups, T)
    G = len(groups)
    for gi in range(G):
        # 近共振峰 Ω（去离共振 Rydberg 巨矩阵元污染）；r 极小（光学量级
        # <1e-5）→ 解析界已证单光子阶收敛，探针验证纯属浪费——跳过。
        vmax = _group_res_peak(groups[gi], E, _M_ANALYTIC_BW_MHZ)
        r_g = vmax / max(abs(groups[gi]['omega_mhz']), 1e-9)
        if r_g < 1e-5:
            continue
        # 微波类组：本组做 M vs M+1 验证（其余组固定）
        while Ms[gi] < _M_MAX:
            M_cur = list(Ms)
            M_nxt = list(Ms)
            M_nxt[gi] = M_cur[gi] + 1
            cur = _probe_pops(E, probe_groups, M_cur, psi0v, T)
            nxt = _probe_pops(E, probe_groups, M_nxt, psi0v, T)
            d = float(np.max(np.abs(nxt - cur)))
            if d <= tol:
                break                            # 本组 M_g 已收敛
            Ms[gi] = M_nxt[gi]
            log(f'[exact] 光子块 M[{gi}]={M_cur[gi]}→{M_nxt[gi]}：'
                f'探针截断差 {d:.2e}，递增')
            if Ms[gi] >= _M_MAX:
                log(f'[exact] 警告：载频 {groups[gi]["omega_mhz"]:.6g}MHz 光子块'
                    f'达上限 M={_M_MAX}，截断误差可能 > tol')
                break
    log(f'[exact] 光子块（逐载频）M={Ms}'
        f'（扇区数 Π(2M_g+1)={setup_P(Ms)}）')
    if frozen_h_ns is not None and frozen_h_ns > 0:
        # t37：frozen 段 eigh 是小矩阵——全程单线程（OpenBLAS 24 线程在此
        # 尺度是净损失 16ms→6ms @234 维；逐段开关上下文开销更大，故包全程）
        import threadpoolctl
        with threadpoolctl.threadpool_limits(limits=1):
            pops, psi_f, report, setup = _run_sambe(
                E, groups, Ms, psi0v, t, tol, progress_arr, log, frozen_h_ns)
    else:
        pops, psi_f, report, setup = _run_sambe(
            E, groups, Ms, psi0v, t, tol, progress_arr, log, frozen_h_ns)
    if frozen_h_ns is not None and frozen_h_ns > 0:
        log(f'[exact] 纯冻结模式完成：{report["n_segments"]} 段'
            f'（h={frozen_h_ns}ns），范数偏差 {report["max_norm_dev"]:.2e}')
    else:
        log(f'[exact] 完成：{report["n_segments"]} 段，'
            f'最大 Richardson 差 {report["max_richardson_diff"]:.2e}'
            f'（tol={tol:.1e}），范数偏差 '
            f'{report["max_norm_dev"]:.2e}')

    report['photon_blocks'] = {f'{g["omega_mhz"]:.6g}MHz': setup.Ds[gi]
                               for gi, g in enumerate(groups)}
    report['tol'] = float(tol)
    out = dict(t=t, pops=pops, states=states,
               om_mhz=[float(max(np.abs(A).max() for A in g['As']))
                       for g in groups],
               method='exact',
               backend='sambe-dense' if setup.dense else 'sambe-sparse',
               loss=np.zeros(len(t)), error_report=report)
    if record_amp:
        out['psi_amp'] = psi_f
    return out


# ============================================================
# 参考路径：H0 相互作用绘景 + 自适应高精度 ODE（权威参考/耗散路径）
# ============================================================
def evolve_reference(states, fields, psi0, t_grid, decays=None,
                     rtol=1e-9, atol=1e-11, verify=False,
                     progress_arr=None, record_amp=False, log=None):
    """实验室系第一性原理参考积分器（零模型近似 + 显式容差）。

    - 幺正（decays 空）：H0 相互作用绘景 + DOP853（自适应、嵌入式误差控制）；
    - 耗散（decays 非空）：同绘景密度矩阵 + Radau（刚性可用）；
      跳变算符在相互作用绘景下时不变（相位抵消），与实验室系严格等价。
    verify=True：容差收紧 10× 重跑，结果附 verify_diff（布居最大差）。
    返回 dict(t, pops, states, om_mhz, method='exact', backend='reference-*',
              loss, [psi_amp | rho, verify_diff])。
    """
    from scipy.integrate import solve_ivp
    from . import evolve as _evolve
    log = log or (lambda *a: None)
    if progress_arr is None:
        progress_arr = np.zeros(2)
    t = np.asarray(t_grid, float)
    T = float(t[-1])
    _validate_fields(fields)
    psi0v = _evolve._as_vector(states, psi0)
    N = len(states)
    E = np.array([s['E_mhz'] for s in states], float)
    E = E - float(E.min())
    terms = [(np.asarray(f['A'], float), float(f['omega_L_mhz']),
              f['waveform']) for f in fields]

    def coupling_at(tv):
        V = np.zeros((N, N))
        for A, wL, wf in terms:
            env = float(np.asarray(wf(np.atleast_1d(tv))).flat[0])
            if env != 0.0:
                V = V + (env * np.cos(_CONV * wL * tv)) * A
        return V

    def _run_unitary(rtl, atl):
        def _rhs_sambe(tv, y):
            u = np.exp(1j * _CONV * E * tv)
            return -1j * _CONV * (u * (coupling_at(tv) @ (np.conj(u) * y)))
        sol = solve_ivp(_rhs_sambe, (0.0, T), psi0v, method='DOP853',
                        t_eval=t, rtol=rtl, atol=atl)
        if not sol.success:
            raise RuntimeError(f'reference 积分失败: {sol.message}')
        return sol.y                            # 相互作用绘景振幅

    decays = list(decays or [])
    if not decays:
        progress_arr[0] = 0.2
        yI = _run_unitary(rtol, atol)
        pops = np.abs(yI).T ** 2
        out = dict(t=t, pops=pops, states=states,
                   om_mhz=[float(np.abs(A).max()) for A, _, _ in terms],
                   method='exact', backend='reference-dop853',
                   loss=np.zeros(len(t)))
        if record_amp:
            uT = np.exp(1j * _CONV * E * T)     # 实验室系末态振幅
            out['psi_amp'] = np.conj(uT) * yI[:, -1]
        if verify:
            y2 = _run_unitary(rtol / 10.0, atol / 10.0)
            out['verify_diff'] = float(np.max(
                np.abs(np.abs(y2).T ** 2 - pops)))
            log(f'[reference] 双容差验证：Δpops={out["verify_diff"]:.2e}'
                f'（rtol={rtol:.1e}）')
        progress_arr[0] = 1.0
        return out

    # ---- Lindblad（同绘景密度矩阵；L 时不变） ----
    Ls, N_eff = _evolve._decay_ops(decays, states)
    rho0 = np.zeros((N_eff, N_eff), complex)
    rho0[:N, :N] = np.outer(psi0v, psi0v.conj())

    def rhs_me(tv, rv):
        rho = rv.reshape((N_eff, N_eff), order='F')
        u = np.exp(1j * _CONV * E * tv)
        V = coupling_at(tv)
        H = np.zeros((N_eff, N_eff), complex)
        H[:N, :N] = (u[:, None] * V) * np.conj(u)[None, :]
        d = -1j * _CONV * (H @ rho - rho @ H)
        for L in Ls:
            LdL = L.conj().T @ L
            d = d + (L @ rho @ L.conj().T
                     - 0.5 * (LdL @ rho + rho @ LdL))
        return d.ravel(order='F')

    def _run_mastereq(rtl, atl):
        # BDF：支持复数 y（Radau 不支持；耗散率可致刚性，隐式多步稳定）
        sol = solve_ivp(rhs_me, (0.0, T), rho0.ravel(order='F'),
                        method='BDF', t_eval=t, rtol=rtl, atol=atl)
        if not sol.success:
            raise RuntimeError(f'reference(Lindblad) 积分失败: {sol.message}')
        return sol.y

    rv = _run_mastereq(rtol, atol)
    pops = np.zeros((len(t), N))
    loss = np.zeros(len(t))
    for k in range(len(t)):
        rho_k = rv[:, k].reshape((N_eff, N_eff), order='F')
        dg = np.real(np.diag(rho_k))
        pops[k] = dg[:N]
        if N_eff > N:
            loss[k] = dg[N]
    out = dict(t=t, pops=np.clip(pops, 0.0, None), states=states,
               om_mhz=[float(np.abs(A).max()) for A, _, _ in terms],
               method='exact', backend='reference-bdf', loss=loss)
    if record_amp:
        out['rho'] = rv[:, -1].reshape((N_eff, N_eff), order='F')
    if verify:
        rv2 = _run_mastereq(rtol / 10.0, atol / 10.0)
        pops2 = np.zeros_like(pops)
        for k in range(len(t)):
            pops2[k] = np.real(np.diag(
                rv2[:, k].reshape((N_eff, N_eff), order='F')))[:N]
        out['verify_diff'] = float(np.max(np.abs(pops2 - pops)))
        log(f'[reference] 双容差验证：Δpops={out["verify_diff"]:.2e}'
            f'（rtol={rtol:.1e}）')
    progress_arr[0] = 1.0
    return out
    return out



# ---- t19 函数级改名兼容别名（旧名 = 新名，外部零改动）----
_SambeSetup.K_at = _SambeSetup.hamiltonian_seg
_run_once = _run_sambe
# _run_me → _run_mastereq 为嵌套运行助手（无模块级旧名引用，不设别名）
evolve_exact = evolve_sambe
