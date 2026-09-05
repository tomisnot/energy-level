# -*- coding: utf-8 -*-
"""单原子 RWA 含时演化（t19：evolve.py 拆分）——ip4/Lindblad + evolve 调度。"""
import numpy as np
import scipy.linalg

from . import static_atom as core
from .field_build import _env_samples, _sublevel_row
from .evolve_arbitrary import _evolve_arbitrary

# numba JIT（可选）：加速 ip4 主循环（强耦合/长时窗时步数可达 10^6~10^7）
try:
    from numba import njit as _njit
except Exception:
    _njit = None

from .evolve_util import _CONV             # t74：常量单源
from .evolve_util import (_POL_Q, _onehot, _as_vector, decay, auto_decays,
                          _resolve_state_idx, _decay_ops)  # t74 工具归位
from .field_build import _field_geometry  # t74：ip4 场格式归一归位场层
# 光频/微波分界：Ω/ωL < 此值 → RWA（反旋转误差 ~1e-7）；≥ → 非 RWA
IP4_RWA_THRESH = 1e-4


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


# ============================================================
# 算法 2：相互作用绘景 RK4（第一性原理，任意时变包络；光频 RWA + 微波非 RWA）
# ============================================================
def _rwa_reference(states, fields):
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
    # 纯位移场（无 g_idx）不参与参考传播。**全部**耦合对都要取边——
    # multi 场可耦合多个 e 组，只按 g_idx[0] 传播会让其余组 ref 留 0 →
    # E_rot 伪劈裂 → u 相位爆炸。
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
    # 非 RWA 场：**全部**耦合组共享同一参考（e 跟随 g）——multi 场耦合多个
    # e 组必须逐对传播，否则其余组 ref 留 0 → E_rot 伪劈裂 → 步长灾难。
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
def _rwa_vtilde_impl(B, mats, S, n_terms, N, sidx, u, ps):
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
    _rwa_vtilde = _njit(cache=True)(_rwa_vtilde_impl)
else:
    _rwa_vtilde = _rwa_vtilde_impl


@_njit(cache=True)
def _rwa_vtilde_sp(B, mats, S, n_terms, N, sidx, u, ps, nz_i, nz_j,
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


def _rwa_rk4_core_numpy(mats, S, Ec, dt, psi0, out_step, n_out, N,
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
        k1 = c * _rwa_vtilde(B, mats, S, n_terms, N, i0, u0, ps)
        k2 = c * _rwa_vtilde(B, mats, S, n_terms, N, i0 + 2, u05,
                             ps + dt / 2 * k1)
        k3 = c * _rwa_vtilde(B, mats, S, n_terms, N, i0 + 2, u05,
                             ps + dt / 2 * k2)
        k4 = c * _rwa_vtilde(B, mats, S, n_terms, N, i0 + 4, u1,
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
    def _rwa_rk4_core(mats, S, Ec, dt, psi0, out_step, n_out, N, n_terms,
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
            k1 = c * _rwa_vtilde_sp(B, mats, S, n_terms, N, i0, u, psi,
                                    nz_i, nz_j, row_ptr, n_nz)
            # k2（t+dt/2, u05）
            for i in range(N):
                ps[i] = psi[i] + dt / 2.0 * k1[i]
            k2 = c * _rwa_vtilde_sp(B, mats, S, n_terms, N, i0 + 2, u05, ps,
                                    nz_i, nz_j, row_ptr, n_nz)
            # k3（t+dt/2, u05）
            for i in range(N):
                ps[i] = psi[i] + dt / 2.0 * k2[i]
            k3 = c * _rwa_vtilde_sp(B, mats, S, n_terms, N, i0 + 2, u05, ps,
                                    nz_i, nz_j, row_ptr, n_nz)
            # k4（t+dt, u1）
            for i in range(N):
                ps[i] = psi[i] + dt * k3[i]
            k4 = c * _rwa_vtilde_sp(B, mats, S, n_terms, N, i0 + 4, u1, ps,
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


def _evolve_rwa(E, states, fields, psi0, t, frac=40.0, record_amp=False,
                progress_arr=None):
    """相互作用绘景 RK4（第一性原理）。

    - 光频（Ω/ωL < IP4_RWA_THRESH）→ RWA 旋转系耦合（反旋转误差 ~1e-7）；
    - 微波等强驱动 → 非 RWA：保留 cos(ω t) 反旋转项（第一性原理）。
    - 参考系：RWA 级联 + 非 RWA 共享（_rwa_reference）。
    - 相互作用绘景 H0=diag(E−ref)：相位精确除去 → 每步误差 ~(Ω·dt)³，
      dt = 最快非 RWA 载波周期/frac（默认 40 → 误差 ~1e-6，~1 min 内）。
    - 全 RWA 且包络恒常 → 一次对角化精确解（RWA 内精确，秒出）。
    """
    N = len(states)
    if progress_arr is None:
        progress_arr = np.zeros(2)
    ref_arr, rwa_g, non_g = _rwa_reference(states, fields)
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
        # 统一场算符 M = A + diag(ac)：Rabi 耦合取半（RWA (Ω/2)env·A），
        # 对角 AC 位移取**全值** env·diag(ac)（D5 回归钉死）。
        sgrid.append((g['A'],
                      (g['omega0_mhz'] / 2.0)
                      * np.asarray(g['waveform'](tg), float)))
        a = g.get('ac_shift')
        if a is not None and np.any(np.asarray(a, float) != 0.0):
            sgrid.append((np.diag(np.asarray(a, float)),
                          np.asarray(g['waveform'](tg), float)))
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
        psi_starts, psi_last, div = _rwa_rk4_core(
            mats_arr, S, Ec, dt, psi0c, out_step, len(t_out), N, n_terms,
            record_amp, t_out[-1], nz_i, nz_j, row_ptr, n_nz, progress_arr)
    else:
        psi_starts, psi_last, div = _rwa_rk4_core_numpy(
            mats_arr, S, Ec, dt, psi0c, out_step, len(t_out), N, n_terms,
            record_amp, t_out[-1], progress_arr)
    if div == 1:
        raise RuntimeError(
            f'ip4 数值发散（dt={dt:.3g} ns）。可能原因：驱动频率过低'
            '（ω_L ≲ Ω，如 MHz 级超精细/Zeeman 驱动）或耦合过强。'
            '可尝试：减小 ip4_frac（如 5）或降低光强。')
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
# 算法 3：Lindblad 主方程（自发辐射等马尔可夫耗散，第一性原理）
# ============================================================
def _e_rot(E, ref_arr, states, rwa_g, non_g):
    """旋转系对角能（与 ip4 同款归一）：旁观态（无耦合）归零、
    耦合态减共同参考——避免光频量级 E_rot 使相位/步长爆炸。"""
    N = len(states)
    E_rot = np.asarray(E, float) - ref_arr
    coupled = set()
    for g in list(rwa_g) + list(non_g):
        coupled.update(g['g_idx'])
        coupled.update(g['e_idx'])
    if coupled:
        E_rot = E_rot - float(np.min(E_rot[list(coupled)]))
    for i in range(N):
        if i not in coupled:
            E_rot[i] = 0.0
    return E_rot


def _decay_ops(decays, states):
    """衰变通道 → (Ls, N_eff)。L = √Γ|to⟩⟨from|（Γ MHz → √(Γ×1e-3)/ns）。

    to=None → 汇入虚拟 loss 态（下标 N）：H 不耦合它，trace 守恒，
    布居只进不出 → loss = ρ[N,N]。返回 (Ls, N_eff)。
    """
    N = len(states)
    has_loss = any(d.get('to') is None for d in decays)
    N_eff = N + (1 if has_loss else 0)
    Ls = []
    for d in decays:
        fi = _resolve_state_idx(states, d['from'])
        if fi is None:
            raise ValueError('衰变通道 from 不能为 None（必须指定衰变态）')
        ti = _resolve_state_idx(states, d['to'])
        gam = float(d.get('gamma_mhz', 0.0))
        if gam < 0 or not np.isfinite(gam):
            raise ValueError(f'衰变率非法: {gam}（须 ≥0 有限值）')
        if ti is None:
            ti = N
        L = np.zeros((N_eff, N_eff), complex)
        L[ti, fi] = np.sqrt(gam * 1e-3)
        Ls.append(L)
    return Ls, N_eff


def _lindblad_eligible(states, fields, t_end):
    """lindblad_eig 前提：旋转系下 H 恒定——
    全部场 RWA 可容（Ω/ωL < 阈值）+ 包络恒常 + 无残留拍频（闭环一致）。"""
    if not fields:
        return True
    geoms = [_field_geometry(states, f) for f in fields]
    for g in geoms:
        ratio = (g.get('real_omega0_mhz', g['omega0_mhz'])
                 / max(g['omega_L_mhz'], 1e-9))
        if ratio >= IP4_RWA_THRESH:
            return False
        if not _env_samples(g['waveform'], t_end)[1]:
            return False
    _, deltas = _assign_reference(states, fields)
    return all(abs(d) <= 1e-6 for d in deltas)


def _evolve_lindblad_eig(E_rot, rwa_g, consts, Ls, psi0, t, record_amp=False,
                         progress_arr=None):
    """常数 H（旋转系）+ 常数 L：Liouvillian 本征分解精确解（第一性原理）。

    ρ_vec(t) = Σ_k e^{λ_k t} v_k (w_k† ρ0)——机器精度、成本与时长无关
    （同 tempsimu 的 solve_me，单位约定已修正：H 乘 2π×1e-3，L 只乘 1e-3）。
    H = diag(E_rot) + Σ (Ω/2)·c·A + c·diag(ac)；c 为常数包络值。
    返回 (t, pops, loss, rho_f)；pops 为 (n_t, N)，loss 为虚拟汇布居。
    """
    N = len(psi0)
    N_eff = Ls[0].shape[0] if Ls else N
    H0 = np.zeros((N_eff, N_eff), complex)
    H0[:N, :N] = np.diag(E_rot).astype(float)
    for g, (c, _) in zip(rwa_g, consts):
        H0[:N, :N] += (g['omega0_mhz'] * c / 2.0) * g['A']
        a = g.get('ac_shift')
        if a is not None and np.any(np.asarray(a, float) != 0.0):
            H0[:N, :N] += c * np.diag(np.asarray(a, float))
    I = np.eye(N_eff)
    Lv = -1j * _CONV * (np.kron(I, H0) - np.kron(H0.conj().T, I))
    for L in Ls:
        LdL = L.conj().T @ L
        Lv = Lv + (np.kron(L.conj(), L)
                   - 0.5 * (np.kron(I, LdL) + np.kron(LdL.conj().T, I)))
    if progress_arr is not None:
        progress_arr[0] = 0.5
    w, V = np.linalg.eig(Lv)
    rho0 = np.zeros((N_eff, N_eff), complex)
    rho0[:N, :N] = np.outer(psi0, psi0.conj())
    c0 = np.linalg.solve(V, rho0.ravel(order='F'))
    t_out = np.asarray(t, float)
    E_exp = np.exp(w[:, None] * t_out[None, :])       # (neig, n_t)
    W = V * c0[None, :]
    diag_idx = np.arange(N_eff) * (N_eff + 1)
    rho_diag = (W[diag_idx, :] @ E_exp).real          # (N_eff, n_t)
    if progress_arr is not None:
        progress_arr[0] = 1.0
    pops = np.clip(rho_diag.T, 0.0, None)             # (n_t, N_eff)
    loss = pops[:, N] if N_eff > N else np.zeros(len(t_out))
    if record_amp:
        rho_f = (V @ (c0 * np.exp(w * t_out[-1]))).reshape(
            (N_eff, N_eff), order='F')
    else:
        rho_f = None
    return t_out, np.ascontiguousarray(pops[:, :N]), loss, rho_f


# ============================================================
# lindblad RK4 numba 内核（旋转系，任意时变包络；S 预计算到 dt/2 子网格）
# ============================================================
def _lindblad_rhs_impl(S, W, nnz_i, nnz_j, Ec, rho, sidx, sk, X, d,
                       N_eff, n_terms, n_nz, n_jumps, ti, fi, gam, ng):
    """d = −i(X−X†) + 跳变；X = Hρ，H = diag(Ec) + Σ_k s_k·W_k（稀疏，rad/ns）。
    ρ 厄米 + H 实 → ρH = X†：对易子只需**一次**稀疏点乘（第一性原理精确）。
    sidx 可为分数（输出部分步用线性插值；整数时刻精确）。"""
    i0 = int(sidx)
    fr = sidx - i0
    if i0 > ng - 2:
        i0 = ng - 2
        fr = sidx - i0
    for k in range(n_terms):
        if fr == 0.0:
            sk[k] = S[k, i0]
        else:
            sk[k] = S[k, i0] * (1.0 - fr) + S[k, i0 + 1] * fr
    # X = Hρ：对角项 + 稀疏项（O(nnz·N)）
    for i in range(N_eff):
        ei = Ec[i]
        for j in range(N_eff):
            X[i, j] = ei * rho[i, j]
    for p in range(n_nz):
        ii = nnz_i[p]
        kk = nnz_j[p]
        v = 0.0
        for k in range(n_terms):
            v += sk[k] * W[k, p]
        if v != 0.0:
            for j in range(N_eff):
                X[ii, j] += v * rho[kk, j]
    # d = −i(X − X†) + 跳变
    for i in range(N_eff):
        for j in range(N_eff):
            d[i, j] = -1j * (X[i, j] - np.conj(X[j, i]))
    for q in range(n_jumps):
        g = gam[q]
        i_t = ti[q]
        i_f = fi[q]
        d[i_t, i_t] += g * rho[i_f, i_f].real
        for j in range(N_eff):
            d[i_f, j] -= 0.5 * g * rho[i_f, j]
            d[j, i_f] -= 0.5 * g * rho[j, i_f]


def _lindblad_rk4_step_impl(S, W, nnz_i, nnz_j, Ec, rho, sidx0, h, dt, sk, X,
                            d, k1, k2, k3, k4, rho2, N_eff, n_terms, n_nz,
                            n_jumps, ti, fi, gam, ng):
    """RK4 一步（物理步长 h，起点子网格 sidx0）。结果写 rho2（不覆盖 rho）。
    阶段偏移（子网格单位）：0, h/dt, h/dt, 2h/dt——全步 h=dt 时恰为整数
    （0,1,1,2，精确采样）；部分步为分数（S 线性插值）。"""
    q1 = h / dt
    hh = 0.5 * h
    _lindblad_rhs(S, W, nnz_i, nnz_j, Ec, rho, sidx0, sk, X, k1,
                  N_eff, n_terms, n_nz, n_jumps, ti, fi, gam, ng)
    for i in range(N_eff):
        for j in range(N_eff):
            rho2[i, j] = rho[i, j] + hh * k1[i, j]
    _lindblad_rhs(S, W, nnz_i, nnz_j, Ec, rho2, sidx0 + q1, sk, X, k2,
                  N_eff, n_terms, n_nz, n_jumps, ti, fi, gam, ng)
    for i in range(N_eff):
        for j in range(N_eff):
            rho2[i, j] = rho[i, j] + hh * k2[i, j]
    _lindblad_rhs(S, W, nnz_i, nnz_j, Ec, rho2, sidx0 + q1, sk, X, k3,
                  N_eff, n_terms, n_nz, n_jumps, ti, fi, gam, ng)
    for i in range(N_eff):
        for j in range(N_eff):
            rho2[i, j] = rho[i, j] + h * k3[i, j]
    _lindblad_rhs(S, W, nnz_i, nnz_j, Ec, rho2, sidx0 + 2.0 * q1, sk, X, k4,
                  N_eff, n_terms, n_nz, n_jumps, ti, fi, gam, ng)
    for i in range(N_eff):
        for j in range(N_eff):
            rho2[i, j] = (rho[i, j]
                          + (h / 6.0) * (k1[i, j] + 2.0 * k2[i, j]
                                         + 2.0 * k3[i, j] + k4[i, j]))


def _lindblad_rk4_core_impl(S, W, nnz_i, nnz_j, Ec, dt, rho0, t_out, out_step,
                            M, t_end, N_eff, n_terms, n_nz, n_jumps, ti, fi,
                            gam, progress_arr, pops, ng):
    """旋转系 Lindblad RK4 主循环（numba，nogil，可取消/进度）。

    rho0 就地演化；输出时刻在步内做部分步（S 线性插值）精确写 pops；
    末步部分推进到 t_end（rho0 就地=ρ(t_end)）。返回 div：0 正常 / 1 发散 / 2 取消。
    """
    rho = rho0
    n_out = t_out.shape[0]
    sk = np.zeros(n_terms, dtype=np.float64)
    X = np.zeros((N_eff, N_eff), dtype=np.complex128)
    d = np.zeros((N_eff, N_eff), dtype=np.complex128)
    k1 = np.zeros((N_eff, N_eff), dtype=np.complex128)
    k2 = np.zeros((N_eff, N_eff), dtype=np.complex128)
    k3 = np.zeros((N_eff, N_eff), dtype=np.complex128)
    k4 = np.zeros((N_eff, N_eff), dtype=np.complex128)
    rho2 = np.zeros((N_eff, N_eff), dtype=np.complex128)
    ri = 1
    for m in range(1, M + 1):
        t_m = (m - 1) * dt
        sidx0 = 2.0 * (m - 1)
        # 输出部分步（用步首状态 rho；末点=ρ(t_end) 就地保存）
        while ri < n_out and out_step[ri] == m:
            hh = t_out[ri] - t_m
            _lindblad_rk4_step(S, W, nnz_i, nnz_j, Ec, rho, sidx0, hh, dt,
                               sk, X, d, k1, k2, k3, k4, rho2, N_eff,
                               n_terms, n_nz, n_jumps, ti, fi, gam, ng)
            for i in range(N_eff):
                pops[ri, i] = rho2[i, i].real
            if ri == n_out - 1:
                for i in range(N_eff):
                    for j in range(N_eff):
                        rho[i, j] = rho2[i, j]        # 末输出点 = ρ(t_end)
            ri += 1
        if ri >= n_out:
            break                                       # 全部输出已记录
        _lindblad_rk4_step(S, W, nnz_i, nnz_j, Ec, rho, sidx0, dt, dt,
                           sk, X, d, k1, k2, k3, k4, rho2, N_eff,
                           n_terms, n_nz, n_jumps, ti, fi, gam, ng)
        for i in range(N_eff):
            for j in range(N_eff):
                rho[i, j] = rho2[i, j]
        if (m & 127) == 0:
            progress_arr[0] = m / M
            if progress_arr[1] != 0.0:
                return 2
            tr = 0.0
            for i in range(N_eff):
                tr += rho[i, i].real
            if not np.isfinite(tr):
                return 1
    return 0


if _njit is not None:
    _lindblad_rhs = _njit(cache=True)(_lindblad_rhs_impl)
    _lindblad_rk4_step = _njit(cache=True)(_lindblad_rk4_step_impl)
    _lindblad_rk4_core = _njit(cache=True, nogil=True)(_lindblad_rk4_core_impl)
else:
    _lindblad_rhs = _lindblad_rhs_impl
    _lindblad_rk4_step = _lindblad_rk4_step_impl
    _lindblad_rk4_core = _lindblad_rk4_core_impl


def _evolve_lindblad(E, states, fields, decays, psi0, t, frac=20.0,
                     backend='auto', progress_arr=None):
    """Lindblad 主方程矩阵 RK4（第一性原理，任意时变包络）。

    旋转系（同 ip4）：
        H(t) = diag(E_rot) + Σ_rwa (Ω/2)env(t)·A + env(t)·diag(ac)
                            + Σ_non Ω·env(t)·cos(ω_L t)·A + env(t)·diag(ac)
    跳变算符 L=√Γ|to⟩⟨from| 在旋转系下**时不变**（LρL†、L†L 中相位自动
    抵消）→ 耗散项与旋转参考无关，精确。rank-1 跳变按 O(N²) 快速路径。
    步长三重约束（同 ip4）：载波分辨率（最快振荡/frac）、RK4 精度
    （dt·‖H‖≲0.06）、稳定性（dt·‖H‖<2），外加 500 步下限与 0.05 ns 上限。
    返回 (t, pops, loss, rho_f)。
    """
    N = len(states)
    if progress_arr is None:
        progress_arr = np.zeros(2)
    ref_arr, rwa_g, non_g = _rwa_reference(states, fields)
    E_rot = _e_rot(np.asarray(E, float), ref_arr, states, rwa_g, non_g)
    Ls, N_eff = _decay_ops(decays, states)
    t_out = np.asarray(t, float)

    # ---- 步长（ip4 同款启发；耗散率并入耦合幅度约束） ----
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
    gmax = max((abs(L).max() ** 2 for L in Ls), default=0.0)   # Γ(1/ns)
    cmax = max(cmax, 0.5 * gmax)
    if w_max > 0:
        dt_carrier = 2.0 * np.pi / (_CONV * w_max) / frac
    else:
        dt_carrier = 0.01
    dt_acc = 0.06 / (_CONV * max(cmax, 1e-6))
    dt_stab = 2.0 / (_CONV * max(cmax, 1e-6))
    dt_res = t_out[-1] / 500.0
    dt = min(dt_carrier, dt_acc, dt_stab, dt_res, 0.05)
    if dt <= 0 or not np.isfinite(dt):
        raise RuntimeError('lindblad 步长非法：请检查驱动频率/光强/衰变配置。')

    # ---- H(t)（旋转系，统一场算符：Rabi 半耦合 + ac 全位移） ----
    H0 = np.zeros((N_eff, N_eff), complex)
    H0[:N, :N] = np.diag(E_rot).astype(float)

    def H_at(tv):
        H = H0.copy()
        for g in rwa_g:
            env = float(np.asarray(g['waveform'](np.atleast_1d(tv))).flat[0])
            H[:N, :N] += (g['omega0_mhz'] * env / 2.0) * g['A']
            a = g.get('ac_shift')
            if a is not None and np.any(np.asarray(a, float) != 0.0):
                H[:N, :N] += env * np.diag(np.asarray(a, float))
        for g in non_g:
            env = float(np.asarray(g['waveform'](np.atleast_1d(tv))).flat[0])
            H[:N, :N] += (g['omega0_mhz'] * env
                          * np.cos(_CONV * g['omega_L_mhz'] * tv)) * g['A']
            a = g.get('ac_shift')
            if a is not None and np.any(np.asarray(a, float) != 0.0):
                H[:N, :N] += env * np.diag(np.asarray(a, float))
        return H

    # 跳变元数据：rank-1 快速路径（O(N²)）；一般 L 走通用路径（防御）
    jumps = []
    generic = []
    for L in Ls:
        nz = np.argwhere(np.abs(L) > 1e-15)
        if len(nz) == 1:
            ti, fi = int(nz[0][0]), int(nz[0][1])
            jumps.append((ti, fi, L[ti, fi]))
        else:
            generic.append(L)

    def _rhs_lindblad(rho, tv):
        H = H_at(tv)
        d = -1j * _CONV * (H @ rho - rho @ H)
        for ti, fi, val in jumps:
            g = (val * val.conj()).real
            rff = rho[fi, fi].real
            d[ti, ti] += g * rff                       # LρL†（仅对角受）
            d[fi, :] -= 0.5 * g * rho[fi, :]           # −½{L†L, ρ}
            d[:, fi] -= 0.5 * g * rho[:, fi]
        for L in generic:
            LdL = L.conj().T @ L
            d = d + (L @ rho @ L.conj().T
                     - 0.5 * (LdL @ rho + rho @ LdL))
        return d

    def rk4(rho, tv, h):
        k1 = _rhs_lindblad(rho, tv)
        k2 = _rhs_lindblad(rho + (h / 2.0) * k1, tv + h / 2.0)
        k3 = _rhs_lindblad(rho + (h / 2.0) * k2, tv + h / 2.0)
        k4 = _rhs_lindblad(rho + h * k3, tv + h)
        return rho + (h / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)

    # ---- 积分：numba 内核（默认） / numpy 回退 ----
    M = int(np.ceil(t_out[-1] / dt))
    out_step = np.minimum(np.ceil(t_out / dt).astype(int), M)

    if _njit is not None and backend != 'cpu' and not generic:
        # ---- numba 内核路径：S 预计算到 dt/2 子网格（RK4 各阶段时刻均为
        # dt/2 整数倍 → 全步精确采样；仅输出部分步线性插值） ----
        ng = 2 * M + 1
        tg = np.arange(ng) * (dt / 2.0)
        sgrid = []
        for g in rwa_g:
            sgrid.append((g['A'],
                          (g['omega0_mhz'] / 2.0)
                          * np.asarray(g['waveform'](tg), float)))
            a = g.get('ac_shift')
            if a is not None and np.any(np.asarray(a, float) != 0.0):
                sgrid.append((np.diag(np.asarray(a, float)),
                              np.asarray(g['waveform'](tg), float)))
        for g in non_g:
            sgrid.append((g['A'],
                          g['omega0_mhz'] * np.asarray(g['waveform'](tg), float)
                          * np.cos(_CONV * g['omega_L_mhz'] * tg)))
            a = g.get('ac_shift')
            if a is not None and np.any(np.asarray(a, float) != 0.0):
                sgrid.append((np.diag(np.asarray(a, float)),
                              np.asarray(g['waveform'](tg), float)))
        n_terms = len(sgrid)
        # 稀疏结构：合并所有项的 (i,j) 非零并集；W[k,p] = 项 k 在 p 处的系数
        pairs = []
        seen = {}
        for k, (Mk, _) in enumerate(sgrid):
            for i in range(N):
                for j in range(N):
                    if Mk[i, j] != 0.0 and (i, j) not in seen:
                        seen[(i, j)] = len(pairs)
                        pairs.append((i, j))
        n_nz = len(pairs)
        W = np.zeros((n_terms, n_nz), float)
        for k, (Mk, _) in enumerate(sgrid):
            for p, (i, j) in enumerate(pairs):
                W[k, p] = Mk[i, j]
        W *= _CONV                                   # MHz → rad/ns
        nnz_i = np.array([i for i, _ in pairs], np.int64)
        nnz_j = np.array([j for _, j in pairs], np.int64)
        S = np.stack([s for _, s in sgrid]) if sgrid \
            else np.zeros((0, ng))
        Ec = np.zeros(N_eff, complex)
        Ec[:N] = _CONV * E_rot
        ti = np.array([t for t, _, _ in jumps], np.int64)
        fi = np.array([f for _, f, _ in jumps], np.int64)
        gam = np.array([(v * v.conj()).real for _, _, v in jumps], float)
        rho0n = np.zeros((N_eff, N_eff), complex)
        rho0n[:N, :N] = np.outer(psi0, psi0.conj())
        pops_n = np.zeros((len(t_out), N_eff))
        pops_n[0] = np.real(np.diag(rho0n))
        div = _lindblad_rk4_core(S, W, nnz_i, nnz_j, Ec, dt, rho0n, t_out,
                                 out_step, M, t_out[-1], N_eff, n_terms,
                                 n_nz, len(jumps), ti, fi, gam, progress_arr,
                                 pops_n, ng)
        if div == 2:
            raise RuntimeError('用户取消')
        if div == 1:
            raise RuntimeError('lindblad 数值发散（numba 内核）：请检查驱动'
                               '频率/光强/衰变配置。')
        progress_arr[0] = 1.0
        pops = np.clip(pops_n, 0.0, None)
        loss = pops[:, N] if N_eff > N else np.zeros(len(t_out))
        return (t_out, np.ascontiguousarray(pops[:, :N]), loss, rho0n)

    # ---- numpy 回退（backend='cpu' / 无 numba / 通用 L） ----
    rec_by_step = {}
    for rec in range(1, len(t_out)):
        rec_by_step.setdefault(int(out_step[rec]), []).append(rec)
    rho = np.zeros((N_eff, N_eff), complex)
    rho[:N, :N] = np.outer(psi0, psi0.conj())
    pops = np.zeros((len(t_out), N_eff))
    pops[0] = np.real(np.diag(rho))
    for m in range(1, M + 1):
        t_m = (m - 1) * dt
        rho_m = rho
        if m in rec_by_step:
            for rec in rec_by_step[m]:
                pops[rec] = np.real(np.diag(rk4(rho_m, t_m,
                                                t_out[rec] - t_m)))
        rho = rk4(rho_m, t_m, dt)
        if (m & 127) == 0:
            progress_arr[0] = m / M
            if progress_arr[1] != 0.0:
                raise RuntimeError('用户取消')
    progress_arr[0] = 1.0
    rho_f = rho
    if M * dt > t_out[-1] + 1e-12:      # 末步部分推进到精确 t_end（record_amp）
        rho_f = rk4(rho, (M - 1) * dt, t_out[-1] - (M - 1) * dt)
    pops = np.clip(pops.real, 0.0, None)
    loss = pops[:, N] if N_eff > N else np.zeros(len(t_out))
    return (t_out, np.ascontiguousarray(pops[:, :N]), loss, rho_f)


# ============================================================
# 参考积分器（内部保留，仅 checks 交叉验证用；不作为可选 method）
# ============================================================
def _evolve_reference_lab(E, geoms, psi0, t, record_amp):
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

    def _rhs_ref(tv, y):
        return -1j * _CONV * (H_at(tv) @ y)

    sol = solve_ivp(_rhs_ref, (t[0], t[-1]), np.asarray(psi0, complex),
                    t_eval=t, rtol=1e-8, atol=1e-10)
    if not sol.success:
        raise RuntimeError(f'_evolve_reference_lab 积分失败: {sol.message}')
    pops = np.abs(sol.y).T ** 2
    if record_amp:
        return t, pops, sol.y[:, -1]
    return t, pops, None


def _evolve_reference_hybrid(E, states, fields, psi0, t, record_amp, thresh=1e-4):
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
    ref_arr, _, _ = _rwa_reference(states, fields)
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

    def _rhs_ref(tv, y):
        return -1j * _CONV * (H_at(tv) @ y)

    sol = solve_ivp(_rhs_ref, (t[0], t[-1]), np.asarray(psi0, complex),
                    t_eval=t, rtol=1e-9, atol=1e-11)
    if not sol.success:
        raise RuntimeError(f'_evolve_reference_hybrid 积分失败: {sol.message}')
    pops = np.abs(sol.y).T ** 2
    if record_amp:
        return t, pops, sol.y[:, -1]
    return t, pops, None


# ============================================================
# 初态与向量工具
# ============================================================
def evolve(states, fields, psi0, t_grid, method='auto', backend='auto',
           record_amp=False, ac_shift_mhz=None, ip4_frac=20.0,
           progress_arr=None, log=None, decays=None, tol=None):
    """单原子含时演化（独立模块，不含任何静态能移计算）。

    states : make_levels 输出（全部考虑的能级）
    fields : field()/group_field()/real_field() 输出列表（任意多组时变光场）；
             method='exact' 时为 kind='exact' 场（waveforms.build_fields_exact）
    psi0   : 初态——states 索引 | (n,l,j,m) | dict(n,l,j,m) | 复向量
    t_grid : 时间数组（ns）
    method : 'auto'（默认）自动调度
               → 无耗散：恒 'ip4'（相互作用绘景 RK4，第一性原理）
               → 有耗散：旋转系 H 恒定 → 'lindblad_eig'（本征分解，机器精度）
                         否则            → 'lindblad'（矩阵 RK4，第一性原理）
             'exact'       ：第一性原理引擎（实验室系无 RWA/无旋转系/无近远分流：
               Sambe 光子块扩展空间 + 包络自适应分段 + Duhamel 缺陷修正，
               误差由 tol 控制并随结果 error_report 可见；含耗散时转参考路径）
             'ip4'         ：相互作用绘景 RK4（任意包络；光频 RWA + 微波非 RWA）
             'lindblad'    ：Lindblad 主方程矩阵 RK4（任意时变包络 + 耗散）
             'lindblad_eig'：常数 H + 耗散 → Liouvillian 本征分解（精确）
             （显式 'ip4' 搭配 decays 时自动切换 Lindblad 路径并 log 提示——
               它是纯幺正方法，不能含耗散）
             **2025-08 清理**：'floquet'/'pwc' 已从本模块移除（动态分析只用
             ip4）。需要精确 Floquet / RWA 分段常数时用封存模块。
    decays   : decay() 输出列表（自发辐射等马尔可夫耗散；空 = 纯幺正）
    tol      : method='exact' 布居容差（默认 1e-6；包络离散 + 光子块截断共用）
    backend: 'auto'（默认）/ 'cpu'（lindblad 用 numba 内核，'cpu' 强制纯
              numpy 回退；ip4 恒 numpy）
    record_amp: 是否保留末态复振幅（纯幺正 → psi_amp；lindblad → rho 末密度阵）
    ip4_frac: ip4 步长 = 最快非 RWA 载波周期/ip4_frac
             （默认 20 → ~3e-5 误差、~20-30 s；40 → ~1e-6、~50-60 s）
    progress_arr: 可选 np.float64[2] 共享进度数组——[0]=进度 0..1（计算中
             定期写入），[1]≠0 请求取消（抛 RuntimeError('用户取消')）。
             适合放在工作线程里由 UI 轮询（numba 内核 nogil，主线程可读）。
    log      : 可选 callable(str)——算法提示/警告（auto 选型、RWA 误差等）
             转发给 UI（如 QStatusBar）；缺省静默。

    返回 dict(t, pops, states, om_mhz, method, backend, loss[, psi_amp | rho])。
      pops: (n_t, n_states) 各态布居；loss: (n_t,) 离开建模空间的布居
      （decays 全为显式落点或未配置时恒 0）；有耗散时 Σpops + loss = 1。
    """
    t = np.asarray(t_grid, float)
    psi0v = _as_vector(states, psi0)
    E = np.array([s['E_mhz'] for s in states])       # 裸能（ref/ω_L 基准，不含 AC）
    decays = list(decays or [])
    if decays:
        _decay_ops(decays, states)                   # 提前校验（解析错误尽早报）
    if method == 'exact':
        # 第一性原理引擎：实验室系无 RWA/无旋转系/无近远分流；ac_shift 作为
        # H0 常数对角修正传入（旧接口兼容）；场须为 kind='exact'
        from . import evolve_sambe as _exact
        return _exact.evolve_exact(
            states, fields, psi0v, t, tol=(1e-6 if tol is None else tol),
            decays=decays, progress_arr=progress_arr, log=log,
            record_amp=record_amp, E_shift=ac_shift_mhz)
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

    if method == 'floquet' or method == 'pwc':
        # 2025-08 清理：两种算法已封存（动态分析只用 ip4，RK4 足够快）
        raise ValueError(
            f"method={method!r} 已从 evolve.py 移除（动态分析只用 ip4，"
            'RK4 足够快）。需要精确 Floquet 传播子 / RWA 分段常数（pwc）时，'
            '请用封存模块：'
            'from src._archive.evolve_floquet_pwc import '
            'evolve_floquet, evolve_pwc')

    if method == 'auto':
        if decays:
            method = ('lindblad_eig' if _lindblad_eligible(states, fields,
                                                           t[-1])
                      else 'lindblad')
            log(f'[evolve] auto → {method}（含 {len(decays)} 个衰变通道，'
                'Lindblad 主方程，第一性原理）')
        else:
            method = 'ip4'
            log('[evolve] auto → ip4（相互作用绘景 RK4，第一性原理）')
    elif decays and method == 'ip4':
        # 纯幺正方法不能含耗散：自动切换 Lindblad 路径（保兼容，不静默改行为）
        log(f'[evolve] 检测到 {len(decays)} 个衰变通道：ip4 为纯幺正'
            '方法，已自动切换 Lindblad 路径')
        method = ('lindblad_eig' if _lindblad_eligible(states, fields, t[-1])
                  else 'lindblad')

    # ---- 任意波形（第一性原理）：auto 检测/显式指定 ----
    has_arb = any((f or {}).get('kind') == 'arb' for f in fields)
    if has_arb and method == 'auto':
        if decays:
            raise NotImplementedError(
                '任意波形求解（arbitrary）为纯幺正方法，暂不支持衰减通道；'
                '请先关闭自发辐射。')
        method = 'arbitrary'
        log('[evolve] 检测到任意波形场 → arbitrary'
            '（实验室系相互作用绘景 RK4，第一性原理，无 RWA/旋转系）')
    if method == 'arbitrary':
        if not has_arb:
            raise ValueError(
                "method='arbitrary' 要求至少一个任意波形场（kind='arb'，"
                '由 waveforms.build_fields 生成）')
        if decays:
            raise NotImplementedError(
                '任意波形求解（arbitrary）为纯幺正方法，暂不支持衰减通道。')
        t, pops, psi = _evolve_arbitrary(
            E, states, fields, psi0v, t,
            frac=max(int(ip4_frac), 80),      # 任意波形默认精度门（RK4 4 阶）
            progress_arr=progress_arr)
        log(f'[evolve] arbitrary frac={max(int(ip4_frac), 80)}'
            '（RK4 4 阶；信号采样请保证快载波 ≥4~8 点/周期）')
        out = dict(t=t, pops=pops, states=states,
                   om_mhz=[f.get('omega0_mhz', 0.0) for f in fields],
                   method='arbitrary', backend='numpy',
                   loss=np.zeros(len(t)))
        if record_amp:
            out['psi_amp'] = psi
        return out

    if method == 'ip4':
        t, pops, psi = _evolve_rwa(E, states, fields, psi0v, t,
                                   frac=ip4_frac, record_amp=record_amp,
                                   progress_arr=progress_arr)
        out = dict(t=t, pops=pops, states=states,
                   om_mhz=[g['omega0_mhz']
                           for g in [_field_geometry(states, f) for f in fields]],
                   method='ip4', backend='numpy',
                   loss=np.zeros(len(t)))
        if record_amp:
            out['psi_amp'] = psi
        return out

    if method == 'lindblad_eig':
        Ls, N_eff = _decay_ops(decays, states)
        if not _lindblad_eligible(states, fields, t[-1]):
            raise ValueError(
                "method='lindblad_eig' 要求旋转系下 H 恒定：全部光场 RWA 可容"
                '且包络恒常、无残留拍频；时变包络/强驱动请用'
                " method='lindblad' 或 'auto'")
        if N_eff > 48:
            log(f'[evolve] N_eff={N_eff}>48：lindblad_eig 本征分解过大，'
                '自动改用 lindblad（RK4）')
            method = 'lindblad'
        else:
            ref_arr, rwa_g, non_g = _rwa_reference(states, fields)
            E_rot = _e_rot(E, ref_arr, states, rwa_g, non_g)
            consts = [_env_samples(g['waveform'], t[-1]) for g in rwa_g]
            t, pops, loss, rho_f = _evolve_lindblad_eig(
                E_rot, rwa_g, consts, Ls, psi0v, t, record_amp=record_amp,
                progress_arr=progress_arr)
            out = dict(t=t, pops=pops, states=states,
                       om_mhz=[g['omega0_mhz']
                               for g in [_field_geometry(states, f)
                                         for f in fields]],
                       method='lindblad_eig', backend='numpy', loss=loss)
            if record_amp:
                out['rho'] = rho_f
            return out

    if method == 'lindblad':
        t, pops, loss, rho_f = _evolve_lindblad(
            E, states, fields, decays, psi0v, t, frac=ip4_frac,
            backend=backend, progress_arr=progress_arr)
        out = dict(t=t, pops=pops, states=states,
                   om_mhz=[g['omega0_mhz']
                           for g in [_field_geometry(states, f) for f in fields]],
                   method='lindblad', backend='numpy', loss=loss)
        if record_amp:
            out['rho'] = rho_f
        return out

    raise ValueError(
        f"未知 method: {method}"
        "（可选 'auto'/'ip4'/'lindblad'/'lindblad_eig'；'floquet'/'pwc' "
        '已封存至 src/_archive/evolve_floquet_pwc.py）')


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



# ---- t19 函数级改名兼容别名（旧名 = 新名，外部零改动）----
_ip4_reference = _rwa_reference
_ip4_vtilde_impl = _rwa_vtilde_impl
_ip4_vtilde_sp = _rwa_vtilde_sp
_ip4_vtilde = _rwa_vtilde
_ip4_rk4_core_numpy = _rwa_rk4_core_numpy
_ip4_rk4_core = _rwa_rk4_core
_evolve_ip4 = _evolve_rwa
_evolve_full = _evolve_reference_lab
_evolve_full_hybrid = _evolve_reference_hybrid
