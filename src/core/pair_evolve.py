# -*- coding: utf-8 -*-
"""两原子含时演化引擎（pair_evolve.py）。

在单原子 evolve.py 基础上扩展两体物理：
  H(t) = [H_A(t) ⊗ I_B] + [I_A ⊗ H_B(t)] + V_dd

V_dd 在旋转系下做缓变近似（secular RWA）：仅保留旋转系振荡频率
|E_rot_β − E_rot_α| ≪ |V_dd| 的矩阵元（对角位移 + 近简并交换耦合），
其余高振荡项（∝光频/微波频/大失谐）平均为零。
**判据必须用旋转系能量差（含 V_dd 对角位移），而非旋转参考之差**——
旧实现误用 ref 差：无光场时 ref≡0 → 全部元素被当"secular"保留，且步长
估计不解析其振荡 → 动力学错误（详见 _secular_filter 注释）。

核心设计：
  1. 两体基组 = 单体 states 的笛卡尔积 |i⟩⊗|j⟩（N² 维）
  2. 单体场 M_A⊗I_B / I_A⊗M_B → 两体场算符
  3. V_dd secular → 预计算静态矩阵，并入 H0 或作为常数耦合项
  4. 单原子旋转参考 ref_A/ref_B → 两体 ref = ref_A⊗I + I⊗ref_B
  5. 自包含 RK4 主循环（复用 _ip4_vtilde 内核，但控制流独立于 evolve.py）

与单原子引擎一致的约定/归一：
  - RWA 场步长只计旋转系失谐 |ΔE|；非 RWA 场加载波 ω_L；ω_L=0 常数场
    （V_dd secular）计 |ΔE|（保留项振荡必须被步长解析）
  - 全 RWA + 常数包络 + 无振荡非 RWA 场 → 一次对角化精确解
    （非 RWA 微波驱动即使包络恒常也含 cos(ω_L t) 载波，哈密顿量时变，
    不能走对角化路径）
  - 旁观态 E_rot 归零、耦合态减公共参考（同 evolve._e_rot，防相位爆炸）

单位：与 evolve.py 一致（MHz, ns, rad/ns）。
"""
import numpy as np
import scipy.sparse as sp

from . import evolve
from . import core

_CONV = 2 * np.pi * 1e-3  # MHz·ns → rad


# ============================================================
# 两体基组
# ============================================================

def tensor_states(states):
    """单原子 states → 两体 product 基组（N² 维）。

    states = [s0, s1, ..., s_{N-1}]（evolve.make_levels 输出）
    product basis: |s_i⟩⊗|s_j⟩，k = i*N + j

    返回：
    states_2d : list of dict，每个含 ket_i, ket_j, label, E_mhz, index
      N : 单体维度
    """
    N = len(states)
    states_2d = []
    for i in range(N):
        for j in range(N):
            si, sj = states[i], states[j]
            E = si['E_mhz'] + sj['E_mhz']
            # 使用普通 ASCII 连接符，避免部分 Matplotlib 字体无法显示 ⊗。
            # 两体图例仍保留两个单体的完整能级标签。
            label = f"({si['label']}) x ({sj['label']})"
            states_2d.append(dict(
                ket_i=i, ket_j=j, label=label, E_mhz=E,
                index=i * N + j))
    return states_2d, N


# ============================================================
# 两体场算符
# ============================================================

def _tensor_field_atom(field_raw, states, N, atom='A'):
    """单体场 → 两体场（M_A⊗I_B 或 I_A⊗M_B）。

    field_raw: evolve.field() / real_field() / group_field() 输出
    atom='A'：场作用在原子 A 上
    atom='B'：场作用在原子 B 上
    """
    # 先归一化：pair/group 场 → multi 格式（含 M, A, ac_shift, g_idx, e_idx）
    g = evolve._field_geometry(states, field_raw)
    M = g['M']
    A = g['A']
    ac = g['ac_shift']
    N1 = M.shape[0]
    assert N1 == N, f'场维度 {N1} ≠ 基组维度 {N}'

    I_other = np.eye(N, dtype=complex)
    M_2d = np.kron(M, I_other) if atom == 'A' else np.kron(I_other, M)
    A_2d = np.kron(A, I_other) if atom == 'A' else np.kron(I_other, A)
    # 对角 AC 位移：ac[i] 加到两体 |i,j⟩ 或 |j,i⟩
    ac_2d = np.zeros(N * N)
    if atom == 'A':
        for i in range(N):
            for j in range(N):
                ac_2d[i * N + j] = ac[i]
    else:
        for i in range(N):
            for j in range(N):
                ac_2d[i * N + j] = ac[j]

    g_idx_2d, e_idx_2d = [], []
    for gi, ei in zip(g.get('g_idx', []), g.get('e_idx', [])):
        if atom == 'A':
            for j in range(N):
                g_idx_2d.append(gi * N + j)
                e_idx_2d.append(ei * N + j)
        else:
            for i in range(N):
                g_idx_2d.append(i * N + gi)
                e_idx_2d.append(i * N + ei)

    return {
        'kind': 'multi',
        'M': M_2d,
        'A': A_2d,
        'ac_shift': ac_2d,
        'g_idx': g_idx_2d,
        'e_idx': e_idx_2d,
        'omega0_mhz': g['omega0_mhz'],
        'real_omega0_mhz': g.get('real_omega0_mhz', g['omega0_mhz']),
        'omega_L_mhz': g['omega_L_mhz'],
        'waveform': g['waveform'],
    }


def tensor_fields(fields_A, fields_B, states, N):
    """两组单体场 → 两体场列表（A⊗I + I⊗B）。

    fields_A/B: 各自的 evolve field 列表
    states: 单原子能级列表（供 _field_geometry 归一化）
    N: 单体维度

    返回两体场列表，直接喂给 pair_evolve。
    """
    fields_2d = []
    for f in fields_A:
        fields_2d.append(_tensor_field_atom(f, states, N, atom='A'))
    for f in fields_B:
        fields_2d.append(_tensor_field_atom(f, states, N, atom='B'))
    return fields_2d


# ============================================================
# V_dd 旋转系 secular RWA
# ============================================================

# secular 判据比例：|旋转系失谐| < SECULAR_RATIO × |V| 的矩阵元保留
# （保留项的振荡频率 ≤ ~10|V|，RK4 步长约束 dt_acc 可解析；远失谐项
# 振荡平均为零 → 丢弃，即 RWA）
SECULAR_RATIO = 10.0


def _secular_filter(V_dense, E_rot_2d, ratio=SECULAR_RATIO):
    """V_dd 的旋转系 secular 筛选（唯一判据，两处调用共用）。

    判据：V_{αβ} 在相互作用绘景（旋转系）中的振荡频率 =
    |E_rot_β − E_rot_α|，其中 E_rot 含 V_dd 对角位移（与演化用的
    u(t)=e^{iE_rot t} 相位自洽）。|Δ| < ratio·|V| → 保留（缓变，RWA）；
    否则高振荡项平均为零 → 丢弃。

    **修复说明**：旧实现误用旋转参考之差（ref_β−ref_α）作判据——无光场
    时 ref≡0，判据恒成立 → 全部 V_dd 元素（含 33 GHz 级失谐的 vdW/Förster
    耦合）被当作"secular"保留，且步长估计不解析其振荡 → 动力学既非精确
    也非 RWA，物理错误。现在用真实旋转系失谐。

    输入可为稠密 ndarray 或 scipy 稀疏矩阵（稀疏时只遍历非零元，大基组
    不会因 toarray 爆内存）。

    返回 (V_diag, V_sec, n_kept)：V_diag 为对角位移，V_sec 为保留的
    非对角稀疏矩阵，n_kept 为保留的非对角元数。
    """
    N2 = len(E_rot_2d)
    E_rot_full = np.asarray(E_rot_2d, float)
    V_sec = sp.lil_matrix((N2, N2), dtype=complex)
    n_kept = 0

    if sp.issparse(V_dense):
        V_c = V_dense.tocsr()
        V_diag = np.real(np.asarray(V_c.diagonal()))
        E_rot_full = E_rot_full + V_diag
        coo = V_c.tocoo()
        for a, b, v in zip(coo.row, coo.col, coo.data):
            if a >= b or abs(v) < 1e-15:
                continue
            if abs(E_rot_full[b] - E_rot_full[a]) < ratio * abs(v):
                V_sec[a, b] = v
                V_sec[b, a] = np.conj(v)
                n_kept += 1
        return V_diag, V_sec.tocsr(), n_kept

    V_dense = np.asarray(V_dense, complex)
    V_diag = np.real(np.diag(V_dense))
    E_rot_full = E_rot_full + V_diag
    for alpha in range(N2):
        for beta in range(alpha + 1, N2):
            v = V_dense[alpha, beta]
            if abs(v) < 1e-15:
                continue
            if abs(E_rot_full[beta] - E_rot_full[alpha]) < ratio * abs(v):
                V_sec[alpha, beta] = v
                V_sec[beta, alpha] = np.conj(v)
                n_kept += 1
    return V_diag, V_sec.tocsr(), n_kept


def vdd_secular(cache, single, R_um, theta_deg, phi_deg, ref_A, ref_B, N,
                E_rot_2d=None):
    """V_dd 的旋转系 secular 近似（cache 路径，pair_evolve 内部用）。

    E_rot_2d: 两体旋转系能量（N²，未加 V_dd 对角）；缺省时从 single 的
    E_mhz 与 ref_A/ref_B 重建（向后兼容）。

    返回：
      V_diag : (N²,) 实数数组——对角位移
      V_sec  : (N², N²) 稀疏矩阵——secular 非对角项
      n_kept : 保留的非对角矩阵元数
    """
    from .pair_core import vdd_sparse
    V_sparse = vdd_sparse(cache, single, R_um, theta_deg, phi_deg)
    V_dense = V_sparse.toarray() if sp.issparse(V_sparse) else np.asarray(V_sparse)
    if E_rot_2d is None:
        N = len(single)
        E_rot_2d = np.zeros(N * N)
        for i in range(N):
            for j in range(N):
                E_rot_2d[i * N + j] = (single[i]['E_mhz'] - ref_A[i]) \
                    + (single[j]['E_mhz'] - ref_B[j])
    return _secular_filter(V_dense, E_rot_2d)


# ============================================================
# 两体旋转参考
# ============================================================

def pair_rotate_frame(states, fields_A, fields_B):
    """分别为原子 A/B 计算旋转参考（复用 evolve._ip4_reference）。

    返回 ref_A, ref_B（各 N 元数组）。
    """
    if fields_A:
        ref_A, _, _ = evolve._ip4_reference(states, fields_A)
    else:
        ref_A = np.zeros(len(states))
    if fields_B:
        ref_B, _, _ = evolve._ip4_reference(states, fields_B)
    else:
        ref_B = np.zeros(len(states))
    return ref_A, ref_B


# ============================================================
# 两体初态
# ============================================================

def _as_product_psi(psi0_A, psi0_B, N):
    """单体初态 → 两体积态 |ψ_A⟩⊗|ψ_B⟩（N² 元向量）。"""
    a = np.asarray(psi0_A, complex).ravel()
    b = np.asarray(psi0_B, complex).ravel()
    assert len(a) == N and len(b) == N, \
        f'初态维度不匹配：A={len(a)}, B={len(b)}, N={N}'
    return np.kron(a, b)


# ============================================================
# 自包含 RK4 主循环（独立于 evolve._evolve_ip4）
# ============================================================

def _pair_ip4(E_rot_2d, fields_2d, psi0, t, frac=20.0,
              progress_arr=None, decays=None):
    """两体相互作用绘景 RK4（自包含，不调用 evolve._evolve_ip4）。

    E_rot_2d: 两体旋转系能量（N²），已含 V_dd 对角位移
    fields_2d: 两体场列表（omega_L_mhz 为旋转频率；V_dd 常数场 omega_L=0）
    psi0: 两体积态初态（N²）
    t: 时间数组（ns）

    返回 (t_out, pops_2d, psi_f)。
    """
    N2 = len(E_rot_2d)
    t_out = np.asarray(t, float)
    if progress_arr is None:
        progress_arr = np.zeros(2)

    # 旋转系能量相位（与 V_dd 对角位移、旁观态归一后的 E_rot_2d 自洽）
    Ec = _CONV * E_rot_2d

    # ---- 每场 RWA 分类（步长估计与项组装共用同一判据） ----
    def _is_rwa(f):
        """光频（Ω/ωL < 阈值）且 ω_L>0 → RWA；ω_L=0 的常数场（V_dd
        secular 等）按非 RWA 处理（cos(0)=1，无载波振荡）。"""
        wL = f['omega_L_mhz']
        ratio = (f.get('real_omega0_mhz', f['omega0_mhz'])
                 / max(wL, 1e-9))
        return wL > 0 and ratio < evolve.IP4_RWA_THRESH

    # ---- 全 RWA 且包络恒常且无振荡非 RWA 场 → 对角化精确解 ----
    # 注意：非 RWA 场（微波等）即使包络恒常也含 cos(ω_L t) 载波，
    # 哈密顿量仍时变，不能走对角化路径（旧实现漏掉此条件 → 错误地把
    # 微波驱动当静态 → 动力学错误）。
    all_const = all(evolve._env_samples(f['waveform'], t_out[-1])[1]
                    for f in fields_2d)
    has_osc = any(f['omega_L_mhz'] > 0 and not _is_rwa(f)
                  for f in fields_2d)

    if all_const and fields_2d and decays is None and not has_osc:
        # 常数哈密顿量 → 一次对角化
        H0 = np.diag(E_rot_2d).astype(complex)
        for f in fields_2d:
            if _is_rwa(f):
                c_val, _ = evolve._env_samples(f['waveform'], t[-1])
                H0 += (f['omega0_mhz'] / 2.0 * c_val) * f['A']
                ac = np.asarray(f.get('ac_shift', np.zeros(N2)), float)
                if np.any(ac != 0.0):
                    H0 += c_val * np.diag(ac)
            else:
                H0 += f['omega0_mhz'] * f['M']
        evals, evecs = np.linalg.eigh(H0)
        coef = evecs.T.conj() @ psi0
        ph = np.exp(-1j * _CONV * evals[:, None] * t_out[None, :])
        psi_t = evecs @ (coef[:, None] * ph)
        pops = np.abs(psi_t).T ** 2
        return t_out, pops, psi_t[:, -1]

    # ---- 一般情况：RK4 步进 ----
    # 步长：Ṽ(t) 内最快真实振荡频率。
    #   RWA 场   ：|E_rot_i − E_rot_j|（旋转系失谐）
    #   非 RWA 场：|E_rot_i − E_rot_j| + ω_L（载波）
    #   ω_L=0 常数场（V_dd secular）：|E_rot_i − E_rot_j|——保留项的
    #     振荡频率必须解析（旧实现跳过 ω_L≤0 场 → 保留的非共振元振荡
    #     未被步长覆盖 → 混叠，动力学错误）。
    w_max = 0.0
    cmax = 0.0
    for f in fields_2d:
        A_abs = np.abs(f['M'])
        cmax = max(cmax, f['omega0_mhz'] * A_abs.max())
        wL = abs(f['omega_L_mhz']) if not _is_rwa(f) else 0.0
        for i in range(N2):
            for j in range(i + 1, N2):
                if A_abs[i, j] > 1e-12:
                    w_max = max(w_max, abs(Ec[i] - Ec[j]) / _CONV + wL)

    if w_max > 0:
        dt_carrier = 2.0 * np.pi / (_CONV * w_max) / frac
    else:
        dt_carrier = t_out[-1] / 500.0 if t_out[-1] > 0 else 0.01
    dt_acc = 0.06 / (_CONV * max(cmax, 1e-6))
    dt_stab = 2.0 / (_CONV * max(cmax, 1e-6))
    dt_res = t_out[-1] / 500.0 if t_out[-1] > 0 else 0.05
    dt = min(dt_carrier, dt_acc, dt_stab, dt_res, 0.05)
    if dt <= 0 or not np.isfinite(dt):
        raise RuntimeError('pair_evolve 步长非法：请检查驱动频率/耦合强度配置。')

    M_steps = max(1, int(np.ceil(t_out[-1] / dt)))
    dt = t_out[-1] / M_steps

    terms = []
    for f in fields_2d:
        if _is_rwa(f):
            terms.append((np.asarray(f['A'], complex),
                          lambda tt, field=f:
                          field['omega0_mhz'] / 2.0
                          * float(np.asarray(field['waveform'](
                              np.atleast_1d(tt))).flat[0])))
            ac = np.asarray(f.get('ac_shift', np.zeros(N2)), float)
            if np.any(ac != 0.0):
                terms.append((np.diag(ac).astype(complex), lambda tt, field=f:
                              float(np.asarray(field['waveform'](
                                  np.atleast_1d(tt))).flat[0])))
        else:
            terms.append((np.asarray(f['M'], complex),
                          lambda tt, field=f:
                          field['omega0_mhz']
                          * float(np.asarray(field['waveform'](
                              np.atleast_1d(tt))).flat[0])
                          * np.cos(_CONV * field['omega_L_mhz'] * tt)))

    c = -1j * _CONV

    def rk4_step(state, t0, h):
        """在 [t0, t0+h] 上执行一次 RK4，支持输出时刻的部分步。"""
        def interaction(state_value, time_value):
            u = np.exp(1j * Ec * time_value)
            out = np.zeros(N2, complex)
            for matrix, coefficient in terms:
                out += coefficient(time_value) * (
                    u * (matrix @ (u.conj() * state_value)))
            return c * out

        k1 = interaction(state, t0)
        k2 = interaction(state + h / 2.0 * k1, t0 + h / 2.0)
        k3 = interaction(state + h / 2.0 * k2, t0 + h / 2.0)
        k4 = interaction(state + h * k3, t0 + h)
        return state + h / 6.0 * (k1 + 2.0 * k2 + 2.0 * k3 + k4)

    psi = np.array(psi0, complex)
    pops = np.zeros((len(t_out), N2))
    pops[0] = np.abs(psi) ** 2
    rec = 1
    eps = max(1e-12, abs(dt) * 1e-10)
    for m in range(M_steps):
        t0 = m * dt
        t1 = min((m + 1) * dt, t_out[-1])
        step_state = psi
        full_state = None
        while rec < len(t_out) and t_out[rec] <= t1 + eps:
            h = max(0.0, float(t_out[rec] - t0))
            if h >= dt - eps:
                if full_state is None:
                    full_state = rk4_step(step_state, t0, dt)
                sample = full_state
            else:
                sample = rk4_step(step_state, t0, h)
            pops[rec] = np.abs(sample) ** 2
            rec += 1
        if full_state is None:
            full_state = rk4_step(step_state, t0, t1 - t0)
        psi = full_state
        if not np.isfinite(psi).all() or np.abs(psi).max() > 100.0:
            raise RuntimeError('pair_evolve 数值发散')
        if (m & 511) == 0:
            progress_arr[0] = min(1.0, t1 / t_out[-1])
            if progress_arr[1] != 0.0:
                raise RuntimeError('用户取消')

    while rec < len(t_out):
        pops[rec] = np.abs(psi) ** 2
        rec += 1
    progress_arr[0] = 1.0
    return t_out, pops, psi


# ============================================================
# 约化布居
# ============================================================

def _reduce_A(pops_2d, N):
    """两体布居 → 原子 A 约化布居（trace out B）。"""
    n_t = pops_2d.shape[0]
    pops_A = np.zeros((n_t, N))
    for i in range(N):
        for j in range(N):
            pops_A[:, i] += pops_2d[:, i * N + j]
    return pops_A


def _reduce_B(pops_2d, N):
    """两体布居 → 原子 B 约化布居（trace out A）。"""
    n_t = pops_2d.shape[0]
    pops_B = np.zeros((n_t, N))
    for i in range(N):
        for j in range(N):
            pops_B[:, j] += pops_2d[:, i * N + j]
    return pops_B


# ============================================================
# 辅助
# ============================================================

def states_to_single(states):
    """evolve states → pair_core 兼容的 groups 列表。"""
    seen = set()
    groups = []
    for s in states:
        key = (s['n'], s['l'], s['j'])
        if key not in seen:
            seen.add(key)
            groups.append(dict(n=s['n'], l=s['l'], j=s['j']))
    return groups


# ============================================================
# 两体演化主入口
# ============================================================

def pair_evolve(states, fields_A, fields_B, psi0_A, psi0_B,
                vdd_matrix=None,
                t_grid=None, method='ip4', backend='auto',
                decays_A=None, decays_B=None,
                progress_arr=None, log=None,
                record_amp=False, ip4_frac=20.0,
                R_um=None, theta_deg=0.0, phi_deg=0.0,
                cache=None):
    """两原子含时演化。

    参数：
      states      : 单原子能级列表（evolve.make_levels 输出），两原子共用
      fields_A/B  : 各自的 evolve field 列表
      psi0_A/B    : 各自初态（states 索引 / (n,l,j,m) dict / 复向量）
      vdd_matrix  : V_dd 矩阵（N²×N²，complex）——若提供则直接使用
      t_grid      : 时间数组（ns）
      method      : 'ip4'（默认）
      decays_A/B  : 各自的 Lindblad 衰变通道列表
      R_um/theta/phi/cache: V_dd 自动计算参数

    返回 dict：
      t, pops_2d, pops_A, pops_B, states_2d, method, backend
    """
    if t_grid is None:
        raise ValueError('t_grid 必须提供')
    if method == 'auto':
        method = 'ip4'
    if method != 'ip4':
        raise ValueError("两体演化当前只支持 method='ip4'")
    if backend not in ('auto', 'cpu', 'numpy'):
        raise ValueError("两体演化当前只支持 numpy backend")
    if decays_A or decays_B:
        # 旧实现静默丢弃衰变通道（_build_two_body_decays 计算后从未使用），
        # 会造成"开了衰变但实际无耗散"的假象——宁可显式拒绝。
        raise NotImplementedError(
            '两体演化暂不支持 Lindblad 衰变（decays_A/decays_B 须为空）。'
            '单原子耗散请用 evolve.evolve(method="lindblad")。')
    t = np.asarray(t_grid, float)
    N = len(states)
    N2 = N * N

    # ---- 两体旋转参考 ----
    ref_A, ref_B = pair_rotate_frame(states, fields_A, fields_B)

    # ---- 旋转系单体能量 ----
    E_A = np.array([s['E_mhz'] for s in states]) - ref_A
    E_B = np.array([s['E_mhz'] for s in states]) - ref_B

    # 两体旋转系能量
    E_rot_2d = np.zeros(N2)
    for i in range(N):
        for j in range(N):
            E_rot_2d[i * N + j] = E_A[i] + E_B[j]

    # ---- V_dd（旋转系 secular RWA，判据统一走 _secular_filter） ----
    V_diag = np.zeros(N2)
    V_sec = sp.csr_matrix((N2, N2), dtype=complex)
    n_kept_vdd = 0

    if vdd_matrix is not None:
        V_diag, V_sec, n_kept_vdd = _secular_filter(vdd_matrix, E_rot_2d)
    elif R_um is not None:
        if cache is None:
            raise ValueError(
                '两体演化需要 cache（core.ArcCache 实例）来计算 V_dd；'
                '请传入 cache，或显式提供 vdd_matrix。'
                '（缺失 cache 时旧实现会静默以 V_dd=0 演化——两体退化为'
                '两独立单体，物理错误。）')
        # 动态页的 states 已经是实际演化所用的单体基组；V_dd 必须在
        # 同一顺序、同一维度的基组上构造，否则矩阵元会错位到别的态。
        single = [dict(s) for s in states]
        V_diag, V_sec, n_kept_vdd = vdd_secular(
            cache, single, R_um, theta_deg, phi_deg, ref_A, ref_B, N,
            E_rot_2d=E_rot_2d)

    if log and n_kept_vdd > 0:
        log(f'[pair_evolve] V_dd secular: {n_kept_vdd} 保留'
            f'（共 {N2*(N2-1)//2} 对）')

    # ---- V_dd 对角 → 并入 E_rot ----
    E_rot_2d += V_diag

    # ---- 构建两体场 ----
    fields_2d = tensor_fields(fields_A, fields_B, states, N)

    # ---- V_dd 非对角 secular → 常数场（omega_L=0） ----
    if V_sec.nnz > 0:
        V_sec_dense = V_sec.toarray() if sp.issparse(V_sec) else np.asarray(V_sec)
        V_field = {
            'kind': 'multi',
            'M': V_sec_dense,
            'A': np.zeros((N2, N2)),
            'ac_shift': np.zeros(N2),
            'g_idx': [],
            'e_idx': [],
            'omega0_mhz': 1.0,
            'omega_L_mhz': 0.0,
            'waveform': evolve.square(0.0, 1e12),
        }
        fields_2d.append(V_field)

    # ---- 旁观态归一（同单原子 _e_rot）：避免光频量级 E_rot 使
    # u(t)=e^{iE_rot t} 相位爆炸、步长估计被旁观态污染 ----
    # 动力学相关集合 = 驱动场耦合态 ∪ V_dd 耦合态。
    coupled = set()
    for f in fields_2d:
        coupled.update(f.get('g_idx', []))
        coupled.update(f.get('e_idx', []))
    if V_sec.nnz > 0:
        nz_i, nz_j = V_sec.nonzero()
        coupled.update(nz_i.tolist())
        coupled.update(nz_j.tolist())
    if coupled:
        E_rot_2d = E_rot_2d - float(np.min(E_rot_2d[list(coupled)]))
    for k in range(N2):
        if k not in coupled:
            E_rot_2d[k] = 0.0

    # ---- 两体初态 ----
    psi0_A_vec = evolve._as_vector(states, psi0_A)
    psi0_B_vec = evolve._as_vector(states, psi0_B)
    psi0_2d = _as_product_psi(psi0_A_vec, psi0_B_vec, N)

    if progress_arr is None:
        progress_arr = np.zeros(2)

    # ---- 自包含 RK4 演化 ----
    t_out, pops_2d, psi_f = _pair_ip4(
        E_rot_2d, fields_2d, psi0_2d, t,
        frac=ip4_frac, progress_arr=progress_arr)

    # ---- 约化布居 ----
    pops_A = _reduce_A(pops_2d, N)
    pops_B = _reduce_B(pops_2d, N)

    return dict(
        t=t_out,
        pops_2d=pops_2d,
        pops_A=pops_A,
        pops_B=pops_B,
        states_2d=[{
            'label': f"({states[i]['label']}) x ({states[j]['label']})",
            'E_mhz': E_rot_2d[i * N + j],
            'ket_i': i,
            'ket_j': j,
        } for i in range(N) for j in range(N)],
        method='ip4',
        backend='numpy',
    )
