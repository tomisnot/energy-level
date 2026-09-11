# -*- coding: utf-8 -*-
"""两原子含时演化引擎（pair_evolve.py）。

在单原子 evolve.py 基础上扩展两体物理：
  H(t) = [H_A(t) ⊗ I_B] + [I_A ⊗ H_B(t)] + V_dd

V_dd 在旋转系下做缓变近似（secular RWA）：仅保留旋转系振荡频率
|E_rot_β − E_rot_α| ≪ |V_dd| 的矩阵元（对角位移 + 近简并交换耦合），
其余高振荡项平均为零。
**判据必须用旋转系能量差（含 V_dd 对角位移），而非旋转参考之差**
（后者无光场时恒 0 → 全保留，见 _secular_filter）。

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
  - 旁观态 E_rot 归零、耦合态减公共参考（同 _e_rot，防相位爆炸）

单位：与 evolve.py 一致（MHz, ns, rad/ns）。
"""
import numpy as np
import scipy.sparse as sp

from src.core.atomic_levels import gaussian, square, step
from src.core.evolve_util import _as_vector, _decay_ops, _resolve_state_idx
from src.core.sambe import env_at
from src.core.field_build import _env_samples
from src.core.evolve_rwa import (_evolve_rwa, _rwa_reference, _e_rot,
                                 _assign_reference, _field_geometry,
                                 RWA_TOL)
from src.core import static_atom as core
from src.core import tensor as _tensor  # noqa: E402  （t14张量化调度层搬移）

_CONV = 2 * np.pi * 1e-3  # MHz·ns → rad

# numba JIT（可选）：并行化 RK4 主循环（CSR matvec 行并行 + 消除 Python/numpy
# 调度开销）。无 numba 时自动回退纯 Python 路径（功能等价，仅慢）。
try:
    from numba import njit, prange
    from numba.typed import List as _NList
    _HAS_NUMBA = True
except Exception:
    njit = prange = None
    _NList = None
    _HAS_NUMBA = False


# ============================================================
# 两体基组
# ============================================================


tensor_states = _tensor.tensor_states

# ============================================================
# 两体场算符（tensor 模块 alias）
# ============================================================


tensor_fields = _tensor.tensor_fields
# ============================================================
# V_dd 旋转系 secular RWA（含二阶 vdW 并入对角）
# ============================================================

# secular 判据比例：|旋转系失谐| < SECULAR_RATIO × |V| 的矩阵元保留
# （保留项振荡频率 ≤ ~10|V|，RK4 步长约束 dt_acc 可解析）；
# 被丢弃的远失谐项（|Δ| ≥ 10|V|，微扰安全区 |V/Δ| ≤ 0.1）不再平均为零——
# 其二阶净效应 |V|²/Δ（∝1/R⁶ 的 vdW 能移）并入对角位移。
SECULAR_RATIO = 10.0


def _secular_filter(V_dense, E_rot_2d, ratio=SECULAR_RATIO):
    """V_dd 的旋转系 secular 筛选（唯一判据，两处调用共用）。

    判据：V_{αβ} 在相互作用绘景（旋转系）中的振荡频率 =
    |E_rot_β − E_rot_α|（E_rot 含裸 V_dd 对角，与演化 u(t)=e^{iE_rot t}
    相位自洽）。|Δ| < ratio·|V| → 保留（缓变，RWA 相干耦合）；
    否则 → 丢弃，但其**二阶 vdW 位移 |V|²/Δ 并入对角**（近简并通道保留、
    远失谐通道贡献位移，两者正交——缺此则 ∝1/R⁶ 的 vdW 能移完全丢失）。
    注意：判据须用旋转系能量差而非旋转参考之差（后者无光场时恒 0）。

    输入可为稠密 ndarray 或 scipy 稀疏矩阵（稀疏时只遍历非零元，大基组
    不因 toarray 爆内存）。

    返回 (V_diag, V_sec, n_kept)：V_diag 为对角位移（裸对角 + 二阶 vdW），
    V_sec 为保留的非对角稀疏矩阵，n_kept 为保留的非对角元数。
    """
    N2 = len(E_rot_2d)
    E_rot_full = np.asarray(E_rot_2d, float)
    V_sec = sp.lil_matrix((N2, N2), dtype=complex)
    n_kept = 0

    def _drop(a, b, v):
        """远失谐元：二阶 vdW 并入对角（反对称：α 与 β 位移一正一负）。"""
        delta = E_rot_full[a] - E_rot_full[b]
        if abs(delta) > 1e-12:
            shift = abs(v) ** 2 / delta
            V_diag[a] += shift
            V_diag[b] -= shift

    if sp.issparse(V_dense):
        V_c = V_dense.tocsr()
        V_diag = np.real(np.asarray(V_c.diagonal())).copy()
        E_rot_full = E_rot_full + V_diag
        coo = V_c.tocoo()
        for a, b, v in zip(coo.row, coo.col, coo.data):
            if a >= b or abs(v) < 1e-15:
                continue
            if abs(E_rot_full[b] - E_rot_full[a]) < ratio * abs(v):
                V_sec[a, b] = v
                V_sec[b, a] = np.conj(v)
                n_kept += 1
            else:
                _drop(a, b, v)
        return V_diag, V_sec.tocsr(), n_kept

    V_dense = np.asarray(V_dense, complex)
    V_diag = np.real(np.diag(V_dense)).copy()
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
            else:
                _drop(alpha, beta, v)
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
    from src.core.static_pair import vdd_sparse
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
    """分别为原子 A/B 计算旋转参考（复用 _rwa_reference）。

    返回 ref_A, ref_B（各 N 元数组）。
    """
    if fields_A:
        ref_A, _, _ = _rwa_reference(states, fields_A)
    else:
        ref_A = np.zeros(len(states))
    if fields_B:
        ref_B, _, _ = _rwa_reference(states, fields_B)
    else:
        ref_B = np.zeros(len(states))
    return ref_A, ref_B


# ============================================================
# 两体初态
# ============================================================


_as_product_psi = _tensor._as_product_psi
# ============================================================
# 自包含 RK4 主循环（独立于 _evolve_rwa）
# ============================================================

# ============================================================
# numba 并行 RK4 内核（CPU 多核加速；无 numba 时回退纯 Python）
# ============================================================
def _pack_pair_terms(fields_2d, N2, G, dt, t_end):
    """两体场列表 → numba 内核输入（扁平 CSR + 偏移，避免 typed-list 慢索引）。

    与旧 terms 展开逐项等价：
      - RWA 场 → (A, ω0/2·env) + 若 ac 非零 (diag(ac), env)
      - 非 RWA/常数场 → (M, ω0·env·cos(ω_L t))；ω_L=0 → cos=1
    包络在 [0, t_end] 均匀 G 点子网格采样（线性插值查表）。

    返回 (DATA_all, IDX_all, PTR2D[n,K+1], OFF[n], NT, COEF, OMG, ENV[n,G])。
    """
    ttype, tcoef, tomg, tenv = [], [], [], []
    data_all, idx_all, ptr2d_rows, offs = [], [], [], []
    tgrid = np.linspace(0.0, float(t_end), G)

    def _env_samples(wf):
        """包络在 tgrid 上一次向量化采样（各 Env 的 __call__ 均支持数组）。"""
        return np.asarray(wf(tgrid), float).copy()

    data_off = 0
    for f in fields_2d:
        wL = float(f['omega_L_mhz'])
        ratio = (f.get('real_omega0_mhz', f['omega0_mhz'])
                 / max(wL, 1e-9))
        is_rwa = wL > 0 and ratio < RWA_TOL
        env_arr = _env_samples(f['waveform'])

        def _push(csr, tt, coef, omg, env):
            nonlocal data_off
            csr = csr.tocsr().astype(complex)
            d = np.ascontiguousarray(csr.data)
            data_all.append(d)
            idx_all.append(np.ascontiguousarray(csr.indices.astype(np.int64)))
            ptr2d_rows.append(np.ascontiguousarray(csr.indptr.astype(
                np.int64)))
            offs.append(data_off)
            data_off += d.size
            ttype.append(tt); tcoef.append(float(coef))
            tomg.append(float(omg)); tenv.append(env)

        if is_rwa:
            A = np.asarray(f['A'], complex)
            _push(sp.csr_matrix(A), 0, f['omega0_mhz'] / 2.0, 0.0, env_arr)
            ac = np.asarray(f.get('ac_shift', np.zeros(N2)), float)
            if np.any(ac != 0.0):
                _push(sp.csr_matrix(np.diag(ac).astype(complex)), 0,
                      1.0, 0.0, env_arr)
        else:
            M = f['M']
            spM = (M.tocsr() if sp.issparse(M)
                   else sp.csr_matrix(np.asarray(M, complex)))
            _push(spM, 1, f['omega0_mhz'], wL, env_arr)
    DATA = np.concatenate(data_all) if data_all else np.zeros(0, complex)
    IDX = np.concatenate(idx_all) if idx_all else np.zeros(0, np.int64)
    n_terms = len(ttype)
    PTR2D = np.zeros((max(n_terms, 1), N2 + 1), np.int64)
    for k, p in enumerate(ptr2d_rows):
        PTR2D[k, :p.size] = p
    OFF = np.asarray(offs, np.int64)
    NT = np.asarray(ttype, np.int8)
    COEF = np.asarray(tcoef, float)
    OMG = np.asarray(tomg, float)
    ENV = np.zeros((max(n_terms, 1), G), float)
    if tenv:
        ENV[:len(tenv)] = np.asarray(tenv, float)
    return DATA, IDX, PTR2D, OFF, NT, COEF, OMG, ENV


if _HAS_NUMBA:
    @njit(cache=True, nogil=True)
    def _pair_ht(ps, uu, ts, Ec, DATA, IDX, PTR2D, OFF, NT, COEF, OMG,
                 ENV, dt, gam, out):
        """out[i] = Σ_k coef_k(ts)·uu_i·Σ_j M_k[i,j]·(conj(uu_j)·ps_j)
        − 0.5i·(gam[i]/_CONV)·ps[i]（两体寿命耗散，落点=丢失）。"""
        N2 = Ec.shape[0]
        n_terms = NT.shape[0]
        G = ENV.shape[1]
        dh = dt * 0.5
        for i in range(N2):
            a = 0.0j
            ui = uu[i]
            for k in range(n_terms):
                gi = int(ts / dh)
                if gi < 0:
                    gi = 0
                if gi > G - 2:
                    gi = G - 2
                w = ts / dh - gi
                e = ENV[k, gi] * (1.0 - w) + ENV[k, gi + 1] * w
                tt = NT[k]
                if tt == 0:
                    cf = COEF[k] * e
                elif tt == 1:
                    cf = COEF[k] * e * np.cos(_CONV * OMG[k] * ts)
                else:
                    cf = COEF[k] * e
                if cf == 0.0:
                    continue
                lo = OFF[k] + PTR2D[k, i]
                hi = OFF[k] + PTR2D[k, i + 1]
                s = 0.0j
                for p in range(lo, hi):
                    j = IDX[p]
                    s += DATA[p] * (np.conj(uu[j]) * ps[j])
                a += cf * ui * s
            if gam[i] != 0.0:
                a -= 0.5j * (gam[i] / _CONV) * ps[i]
            out[i] = a

    @njit(cache=True, nogil=True, parallel=True)
    def _pair_ht_par(ps, uu, ts, Ec, DATA, IDX, PTR2D, OFF, NT, COEF, OMG,
                     ENV, dt, gam):
        """大基组并行版（含两体寿命耗散对角项）。"""
        N2 = Ec.shape[0]
        n_terms = NT.shape[0]
        G = ENV.shape[1]
        dh = dt * 0.5
        out = np.zeros(N2, np.complex128)
        for i in prange(N2):
            a = 0.0j
            ui = uu[i]
            for k in range(n_terms):
                gi = int(ts / dh)
                if gi < 0:
                    gi = 0
                if gi > G - 2:
                    gi = G - 2
                w = ts / dh - gi
                e = ENV[k, gi] * (1.0 - w) + ENV[k, gi + 1] * w
                tt = NT[k]
                if tt == 0:
                    cf = COEF[k] * e
                elif tt == 1:
                    cf = COEF[k] * e * np.cos(_CONV * OMG[k] * ts)
                else:
                    cf = COEF[k] * e
                if cf == 0.0:
                    continue
                lo = OFF[k] + PTR2D[k, i]
                hi = OFF[k] + PTR2D[k, i + 1]
                s = 0.0j
                for p in range(lo, hi):
                    j = IDX[p]
                    s += DATA[p] * (np.conj(uu[j]) * ps[j])
                a += cf * ui * s
            if gam[i] != 0.0:
                a -= 0.5j * (gam[i] / _CONV) * ps[i]
            out[i] = a
        return out

    @njit(cache=True, nogil=True)
    def _pair_rwa_rk4_jit(Ec, DATA, IDX, PTR2D, OFF, NT, COEF, OMG, ENV,
                      w05, w1, psi0, t_out, dt, use_par, gam,
                      gammaA, gammaB, progress_arr=None):
        """顺序时间步进 + 每步 4 阶段（小基组串行复用缓冲 / 大基组 prange）。

        gam: 两体总耗散率（/ns，K）；gammaA/gammaB: 分原子耗散，用于 loss 累计。
        返回 (pops, psi, lossA, lossB)；lossA/B 为累计曲线（与 t_out 同长）。
        """
        N2 = Ec.shape[0]
        n_out = t_out.shape[0]
        T = t_out[-1]
        M = int(np.ceil(T / dt))
        dh = dt * 0.5
        eps = max(1e-12, abs(dt) * 1e-10)
        u = np.ones(N2, np.complex128)
        psi = psi0.copy()
        b1 = np.zeros(N2, np.complex128)
        b2 = np.zeros(N2, np.complex128)
        b3 = np.zeros(N2, np.complex128)
        b4 = np.zeros(N2, np.complex128)
        full = np.zeros(N2, np.complex128)
        pops = np.zeros((n_out, N2))
        sA = np.zeros(n_out)
        sB = np.zeros(n_out)
        IA = 0.0
        IB = 0.0
        pops[0] = np.abs(psi) ** 2
        rec = 1
        for m in range(M):
            tm = m * dt
            if progress_arr is not None:
                progress_arr[0] = min(1.0, tm / T)
                if progress_arr[1] != 0.0:
                    raise RuntimeError('用户取消')
            ub = u * w05
            uc = u * w1
            if use_par:
                k1 = -1j * _CONV * _pair_ht_par(
                    psi, u, tm, Ec, DATA, IDX, PTR2D, OFF, NT, COEF, OMG,
                    ENV, dt, gam)
                k2 = -1j * _CONV * _pair_ht_par(
                    psi + 0.5 * dt * k1, ub, tm + dh, Ec, DATA, IDX,
                    PTR2D, OFF, NT, COEF, OMG, ENV, dt, gam)
                k3 = -1j * _CONV * _pair_ht_par(
                    psi + 0.5 * dt * k2, ub, tm + dh, Ec, DATA, IDX,
                    PTR2D, OFF, NT, COEF, OMG, ENV, dt, gam)
                k4 = -1j * _CONV * _pair_ht_par(
                    psi + dt * k3, uc, tm + dt, Ec, DATA, IDX, PTR2D,
                    OFF, NT, COEF, OMG, ENV, dt, gam)
            else:
                _pair_ht(psi, u, tm, Ec, DATA, IDX, PTR2D, OFF, NT, COEF,
                         OMG, ENV, dt, gam, b1)
                k1 = -1j * _CONV * b1
                _pair_ht(psi + 0.5 * dt * k1, ub, tm + dh, Ec, DATA, IDX,
                         PTR2D, OFF, NT, COEF, OMG, ENV, dt, gam, b2)
                k2 = -1j * _CONV * b2
                _pair_ht(psi + 0.5 * dt * k2, ub, tm + dh, Ec, DATA, IDX,
                         PTR2D, OFF, NT, COEF, OMG, ENV, dt, gam, b3)
                k3 = -1j * _CONV * b3
                _pair_ht(psi + dt * k3, uc, tm + dt, Ec, DATA, IDX,
                         PTR2D, OFF, NT, COEF, OMG, ENV, dt, gam, b4)
                k4 = -1j * _CONV * b4
            full = psi + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
            t1 = min(tm + dt, T)
            while rec < n_out and t_out[rec] <= t1 + eps:
                h = max(0.0, float(t_out[rec] - tm))
                if h >= dt - eps:
                    sample = full
                else:
                    w05h = np.exp(1j * Ec * (h * 0.5))
                    w1h = np.exp(1j * Ec * h)
                    uhb = u * w05h
                    uhc = u * w1h
                    s0 = psi
                    if use_par:
                        kk1 = -1j * _CONV * _pair_ht_par(
                            s0, u, tm, Ec, DATA, IDX, PTR2D, OFF, NT,
                            COEF, OMG, ENV, dt, gam)
                        kk2 = -1j * _CONV * _pair_ht_par(
                            s0 + 0.5 * h * kk1, uhb, tm + 0.5 * h, Ec,
                            DATA, IDX, PTR2D, OFF, NT, COEF, OMG, ENV,
                            dt, gam)
                        kk3 = -1j * _CONV * _pair_ht_par(
                            s0 + 0.5 * h * kk2, uhb, tm + 0.5 * h, Ec,
                            DATA, IDX, PTR2D, OFF, NT, COEF, OMG, ENV,
                            dt, gam)
                        kk4 = -1j * _CONV * _pair_ht_par(
                            s0 + h * kk3, uhc, tm + h, Ec, DATA, IDX,
                            PTR2D, OFF, NT, COEF, OMG, ENV, dt, gam)
                    else:
                        _pair_ht(s0, u, tm, Ec, DATA, IDX, PTR2D, OFF,
                                 NT, COEF, OMG, ENV, dt, gam, b1)
                        kk1 = -1j * _CONV * b1
                        _pair_ht(s0 + 0.5 * h * kk1, uhb, tm + 0.5 * h,
                                 Ec, DATA, IDX, PTR2D, OFF, NT, COEF,
                                 OMG, ENV, dt, gam, b2)
                        kk2 = -1j * _CONV * b2
                        _pair_ht(s0 + 0.5 * h * kk2, uhb, tm + 0.5 * h,
                                 Ec, DATA, IDX, PTR2D, OFF, NT, COEF,
                                 OMG, ENV, dt, gam, b3)
                        kk3 = -1j * _CONV * b3
                        _pair_ht(s0 + h * kk3, uhc, tm + h, Ec, DATA,
                                 IDX, PTR2D, OFF, NT, COEF, OMG, ENV,
                                 dt, gam, b4)
                        kk4 = -1j * _CONV * b4
                    sample = s0 + (h / 6.0) * (kk1 + 2.0 * kk2
                                               + 2.0 * kk3 + kk4)
                pops[rec] = np.abs(sample) ** 2
                st = IA + IB
                ltot = max(1.0 - pops[rec].sum(), 0.0)
                if st > 0.0:
                    sA[rec] = ltot * (IA / st)
                    sB[rec] = ltot * (IB / st)
                rec += 1
            if not np.isfinite(full).all() or np.abs(full).max() > 100.0:
                raise RuntimeError('pair_evolve 数值发散')
            psi = full
            u = u * w1
            # 速率积分（两原子分别）——loss 份额用 IA/(IA+IB)
            pw1 = np.abs(psi) ** 2
            IA += dt * (gammaA * pw1).sum()
            IB += dt * (gammaB * pw1).sum()
        while rec < n_out:
            pops[rec] = np.abs(psi) ** 2
            st = IA + IB
            ltot = max(1.0 - pops[rec].sum(), 0.0)
            if st > 0.0:
                sA[rec] = ltot * (IA / st)
                sB[rec] = ltot * (IB / st)
            rec += 1
        if progress_arr is not None:
            progress_arr[0] = 1.0
        return pops, psi, sA, sB


def _pair_rwa_jit(E_rot_2d, fields_2d, psi0, t_out, dt, M_steps,
                  progress_arr, gamma_A=None, gamma_B=None):
    """numba 版 _pair_rwa 步进（扁平 CSR + 可选两体寿命耗散）。

    默认串行内核：两体场行内非零少（kron 结构），prange 行并行的区域调度
    开销通常大于收益（实测 K=576、60ns 下并行 27 s vs 串行 4.8 s）。
    _pair_ht_par 保留作参数（use_par 置 True 可强制），供行内非零密集、
    基组极大的场景调参。
    """
    N2 = len(E_rot_2d)
    Ec = _CONV * E_rot_2d
    G = 2 * M_steps + 3
    DATA, IDX, PTR2D, OFF, NT, COEF, OMG, ENV = _pack_pair_terms(
        fields_2d, N2, G, dt, t_out[-1])
    w05 = np.exp(1j * Ec * (dt * 0.5))
    w1 = np.exp(1j * Ec * dt)
    gam = np.zeros(N2, float)
    gA = np.zeros(N2, float)
    gB = np.zeros(N2, float)
    if gamma_A is not None:
        gA = np.asarray(gamma_A, float)
        gam = gam + gA
    if gamma_B is not None:
        gB = np.asarray(gamma_B, float)
        gam = gam + gB
    pops, psi, lossA, lossB = _pair_rwa_rk4_jit(
        Ec, DATA, IDX, PTR2D, OFF, NT, COEF, OMG, ENV, w05, w1,
        np.asarray(psi0, complex), np.asarray(t_out, float), dt, False,
        gam, gA, gB, progress_arr=progress_arr)
    return pops, psi, lossA, lossB


# ============================================================
# 两体 Lindblad 耗散（v1：寿命通道，落点=丢失 → NonHermitian 确定性演化）
# ============================================================

_pair_loss_vectors = _tensor._pair_loss_vectors
def _pair_rwa(E_rot_2d, fields_2d, psi0, t, frac=20.0,
              progress_arr=None, decays=None,
              gamma_A=None, gamma_B=None, decay_out=False):
    """两体相互作用绘景 RK4（自包含，不调用 _evolve_rwa）。

    E_rot_2d: 两体旋转系能量（N²），已含 V_dd 对角位移
    fields_2d: 两体场列表（omega_L_mhz 为旋转频率；V_dd 常数场 omega_L=0）
    psi0: 两体积态初态（N²）
    t: 时间数组（ns）
    gamma_A/gamma_B: 两体每基元耗散率（/ns，落点=丢失通道）——
      非 Hermitian 对角演化（等效 L=√Γ|loss⟩⟨from|），loss 累计返回。
    decay_out=True 时返回 (t, pops, psi, lossA, lossB)，否则 (t, pops, psi)。

    返回 (t_out, pops_2d, psi_f[, lossA, lossB])。
    """
    N2 = len(E_rot_2d)
    t_out = np.asarray(t, float)
    if progress_arr is None:
        progress_arr = np.zeros(2)

    has_decay = (gamma_A is not None and np.any(gamma_A)) or \
                (gamma_B is not None and np.any(gamma_B))
    gam = np.zeros(N2, float)
    if gamma_A is not None:
        gam = gam + np.asarray(gamma_A, float)
    if gamma_B is not None:
        gam = gam + np.asarray(gamma_B, float)

    # 旋转系能量相位（与 V_dd 对角位移、旁观态归一后的 E_rot_2d 自洽）
    Ec = _CONV * E_rot_2d

    # ---- 每场 RWA 分类（步长估计与项组装共用同一判据） ----
    def _is_rwa(f):
        """光频（Ω/ωL < 阈值）且 ω_L>0 → RWA；ω_L=0 的常数场（V_dd
        secular 等）按非 RWA 处理（cos(0)=1，无载波振荡）。"""
        wL = f['omega_L_mhz']
        ratio = (f.get('real_omega0_mhz', f['omega0_mhz'])
                 / max(wL, 1e-9))
        return wL > 0 and ratio < RWA_TOL

    # ---- 全 RWA 且包络恒常且无振荡非 RWA 场 → 对角化精确解 ----
    # 非 RWA 场（微波）即使包络恒常也含 cos(ω_L t) 载波，哈密顿量时变，
    # 不能走对角化路径。
    all_const = all(_env_samples(f['waveform'], t_out[-1])[1]
                    for f in fields_2d)
    has_osc = any(f['omega_L_mhz'] > 0 and not _is_rwa(f)
                  for f in fields_2d)

    if all_const and fields_2d and decays is None and not has_osc \
            and not has_decay:
        # 常数哈密顿量 → 一次对角化
        H0 = np.diag(E_rot_2d).astype(complex)
        for f in fields_2d:
            if _is_rwa(f):
                c_val, _ = _env_samples(f['waveform'], t[-1])
                H0 += (f['omega0_mhz'] / 2.0 * c_val) * f['A']
                ac = np.asarray(f.get('ac_shift', np.zeros(N2)), float)
                if np.any(ac != 0.0):
                    H0 += c_val * np.diag(ac)
            else:
                M = f['M']
                H0 = H0 + f['omega0_mhz'] * (M.toarray() if sp.issparse(M) else M)
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
    #   ω_L=0 常数场（V_dd secular）：|E_rot_i − E_rot_j|（保留项振荡须解析）
    # 稀疏矩阵只遍历非零元（避免 O(N⁴) 全对扫描）。
    w_max = 0.0
    cmax = 0.0
    for f in fields_2d:
        M = f['M']
        cmax = max(cmax, f['omega0_mhz'] * np.abs(M).max())
        wL = abs(f['omega_L_mhz']) if not _is_rwa(f) else 0.0
        if sp.issparse(M):
            coo = M.tocoo()
            for i, j, a in zip(coo.row, coo.col, coo.data):
                if i < j and abs(a) > 1e-12:
                    w_max = max(w_max, abs(Ec[i] - Ec[j]) / _CONV + wL)
        else:
            A_abs = np.abs(M)
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

    # ---- numba 并行内核（CPU 多核；无 numba 时下方 Python 循环回退）----
    if _HAS_NUMBA:
        try:
            if has_decay:
                _pops, _psi, _lA, _lB = _pair_rwa_jit(
                    E_rot_2d, fields_2d, psi0, t_out, dt, M_steps,
                    progress_arr, gamma_A=gamma_A, gamma_B=gamma_B)
                if decay_out:
                    return t_out, _pops, _psi, _lA, _lB
                return t_out, _pops, _psi
            _pops, _psi, _lA, _lB = _pair_rwa_jit(
                E_rot_2d, fields_2d, psi0, t_out, dt, M_steps,
                progress_arr)
            if decay_out:
                return t_out, _pops, _psi, _lA, _lB
            return t_out, _pops, _psi
        except Exception as ex:
            # jit 编译/执行异常：回退纯 Python 路径（功能等价）
            raise ex

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
            M = f['M']
            terms.append(((M if sp.issparse(M) else np.asarray(M, complex)),
                          lambda tt, field=f:
                          field['omega0_mhz']
                          * float(np.asarray(field['waveform'](
                              np.atleast_1d(tt))).flat[0])
                          * np.cos(_CONV * field['omega_L_mhz'] * tt)))
    if has_decay:
        # 两体寿命耗散（落点=丢失）：H 对角加 −i·Γ/2（/ns，折回 MHz）
        terms.append((sp.diags(-0.5j * gam / _CONV).astype(complex),
                      lambda tt: 1.0))

    c = -1j * _CONV

    # ---- u 相位增量：全步只做向量逐元乘，不再每阶段现算 exp ----
    # u(t) = e^{iEc·t}；步进 u0 *= w1_full；RK4 阶段时刻的 u 由
    # u0（步首）乘预计算 w05_full/w1_full 得到。部分步（h≠dt）现算。
    # （耗散已作为对角项进 terms，u 保持纯相位。）
    w05_full = np.exp(1j * Ec * (dt / 2.0))
    w1_full = np.exp(1j * Ec * dt)

    def rk4_step(state, t0, h, ua):
        """在 [t0, t0+h] 上执行一次 RK4（ua = e^{iEc·t0}，步首相位）。"""
        if abs(h - dt) < 1e-15:
            ub, uc = ua * w05_full, ua * w1_full
        else:
            ub = ua * np.exp(1j * Ec * (h / 2.0))
            uc = ua * np.exp(1j * Ec * h)

        def Ht(ps, uu, tt):
            out = np.zeros(N2, complex)
            for matrix, coefficient in terms:
                # 稀疏矩阵（V_dd secular）走稀疏 matvec，O(nnz) 而非 O(N⁴)
                out += coefficient(tt) * (uu * (matrix @ (uu.conj() * ps)))
            return out

        k1 = c * Ht(state, ua, t0)
        k2 = c * Ht(state + h / 2.0 * k1, ub, t0 + h / 2.0)
        k3 = c * Ht(state + h / 2.0 * k2, ub, t0 + h / 2.0)
        k4 = c * Ht(state + h * k3, uc, t0 + h)
        return state + h / 6.0 * (k1 + 2.0 * k2 + 2.0 * k3 + k4)

    psi = np.array(psi0, complex)
    pops = np.zeros((len(t_out), N2))
    sA = np.zeros(len(t_out))
    sB = np.zeros(len(t_out))
    IA = 0.0
    IB = 0.0
    pops[0] = np.abs(psi) ** 2
    rec = 1
    eps = max(1e-12, abs(dt) * 1e-10)
    u0 = np.ones(N2, complex)               # e^{iEc·0} = 1
    _gA = np.asarray(gamma_A, float) if gamma_A is not None else np.zeros(N2)
    _gB = np.asarray(gamma_B, float) if gamma_B is not None else np.zeros(N2)
    for m in range(M_steps):
        t0 = m * dt
        t1 = min((m + 1) * dt, t_out[-1])
        step_state = psi
        full_state = None
        while rec < len(t_out) and t_out[rec] <= t1 + eps:
            h = max(0.0, float(t_out[rec] - t0))
            if h >= dt - eps:
                if full_state is None:
                    full_state = rk4_step(step_state, t0, dt, u0)
                sample = full_state
            else:
                sample = rk4_step(step_state, t0, h, u0)
            pops[rec] = np.abs(sample) ** 2
            if has_decay:
                st = IA + IB
                ltot = max(1.0 - pops[rec].sum(), 0.0)
                if st > 0.0:
                    sA[rec] = ltot * (IA / st)
                    sB[rec] = ltot * (IB / st)
            rec += 1
        if full_state is None:
            full_state = rk4_step(step_state, t0, t1 - t0, u0)
        psi = full_state
        u0 = u0 * w1_full                    # 步进相位（逐元乘）
        if not np.isfinite(psi).all() or np.abs(psi).max() > 100.0:
            raise RuntimeError('pair_evolve 数值发散')
        if (m & 511) == 0:
            progress_arr[0] = min(1.0, t1 / t_out[-1])
            if progress_arr[1] != 0.0:
                raise RuntimeError('用户取消')
        if has_decay:
            pw1 = np.abs(psi) ** 2
            IA += dt * float((_gA * pw1).sum())
            IB += dt * float((_gB * pw1).sum())

    while rec < len(t_out):
        pops[rec] = np.abs(psi) ** 2
        if has_decay:
            st = IA + IB
            ltot = max(1.0 - pops[rec].sum(), 0.0)
            if st > 0.0:
                sA[rec] = ltot * (IA / st)
                sB[rec] = ltot * (IB / st)
        rec += 1
    progress_arr[0] = 1.0
    if decay_out:
        return t_out, pops, psi, sA, sB
    return t_out, pops, psi


# ============================================================
# 约化布居
# ============================================================


_reduce_A = _tensor._reduce_A

_reduce_B = _tensor._reduce_B
def _scatter_loss_curve(fields, pops_atom, t, N):
    """远失谐光的 off-resonant 散射损失曲线（report-only 后处理）。

    对每束光：散射率 σ（MHz，源态 i）由 real_field 给出（≈Ω²Γ_up/(4Δ²)），
    源布居随时间被散射到短寿命上态 → depump/loss：
        loss_i(t) = ∫₀ᵗ σ_i·1e-3·env(t')²·P_atom[i](t') dt'
    P_atom = 该原子约化布居（n_t, N，输出网格）。不反馈演化（弱散射二阶小）。
    """
    n_t = len(t)
    curve = np.zeros(n_t)
    if not fields:
        return curve
    dt_arr = np.diff(t)
    for f in fields:
        _sc = f.get('scatter')
        sc = np.asarray(_sc, float) if _sc is not None else np.zeros(N)
        if not np.any(sc > 0.0):
            continue
        wf = f.get('waveform')
        if wf is None:
            continue
        env2 = np.array([env_at(wf, tt)
                         ** 2 for tt in t], float)
        for i in range(N):
            rate = float(sc[i])
            if rate <= 0.0:
                continue
            rate_ns = rate * 1e-3                # MHz → /ns
            prod = env2 * pops_atom[:, i]
            integ = np.zeros(n_t)
            integ[1:] = np.cumsum(0.5 * (prod[1:] + prod[:-1]) * dt_arr)
            curve = curve + rate_ns * integ
    return curve


# ============================================================
# 辅助
# ============================================================


states_to_single = _tensor.states_to_single
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
      states      : 单原子能级列表（make_levels 输出），两原子共用
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
    t = np.asarray(t_grid, float)
    N = len(states)
    N2 = N * N

    # ---- 两体 Lindblad 耗散（v1：寿命通道，落点=丢失，NonHermitian）----
    gamma_A = gamma_B = None
    has_decay = bool(decays_A or decays_B)
    if has_decay:
        gamma_A, gamma_B, _ = _pair_loss_vectors(
            states, decays_A, decays_B, N, N2)
        if log:
            log('[pair_evolve] 两体耗散：' +
                (f'A {int((np.asarray(gamma_A) > 0).sum() // N)} 条通道, '
                 if np.any(gamma_A) else 'A 无, ') +
                (f'B {int((np.asarray(gamma_B) > 0).sum() // N)} 条通道'
                 if np.any(gamma_B) else 'B 无') +
                '（落点=丢失，NonHermitian + loss 累计）')

    # ---- 两体旋转参考 ----
    ref_A, ref_B = pair_rotate_frame(states, fields_A, fields_B)

    # ---- 常数包络光的 AC 位移 → 并入旋转参考（演化与判据一致）----
    # 常数位移是可调的工作点（如 exchange 共振）：并入 E_rot 后，演化
    # 相互作用绘景相位 u=e^{iE_rot t} 与 V_dd 交换项在旋转系里都"看见"
    # 位移 → 判据保留的对真正共振、发生交换；同时把该常数位移从光场
    # M 对角扣除，避免双重计入（时变包络的位移保留在 M 中，不入参考）。
    # 通过修改 fields_A/fields_B 的 M，使后续 tensor_fields 使用修正后场。
    shift_A = np.zeros(N, float)
    shift_B = np.zeros(N, float)
    for fields, shift in ((fields_A or [], shift_A), (fields_B or [], shift_B)):
        for fld in fields:
            acv = np.asarray(fld.get('ac_shift', np.zeros(N)), float)
            if not np.any(acv):
                continue
            # 常数判定用中段平台（排除起始/结束边沿，如长方波脉冲）：
            # 只有平台段恒常的光才是稳定的频移工作点。
            _t0 = float(t[0]); _t1 = float(t[-1])
            ts = np.linspace(_t0 + 0.1 * (_t1 - _t0),
                             _t0 + 0.9 * (_t1 - _t0), 24)
            vs = np.asarray(fld['waveform'](ts), float)
            if not np.any(vs > 0.0) or np.ptp(vs) > 1e-6:
                continue
            c_mean = float(vs.mean())
            delta = acv * c_mean
            shift += delta
            M = fld.get('M')
            if M is not None:
                Md = np.asarray(M)
                M = M - np.diag(delta)          # 扣常数对角（防双计）
                fld['M'] = M
                if log:
                    log(f'[pair_evolve] absorb：扣除常数对角 '
                        f'{np.abs(delta).max():.3g} MHz（剩余 '
                        f'|M|max={np.abs(np.asarray(fld["M"])).max():.3g}）')
    if np.any(shift_A) or np.any(shift_B):
        if log:
            log('[pair_evolve] 常数光 AC 位移并入旋转参考：'
                f'A 峰 {np.abs(shift_A).max():.3g} / '
                f'B 峰 {np.abs(shift_B).max():.3g} MHz')

    # ---- 旋转系单体能量 ----
    E_A = np.array([s['E_mhz'] for s in states]) - ref_A + shift_A
    E_B = np.array([s['E_mhz'] for s in states]) - ref_B + shift_B

    # 两体旋转系能量
    E_rot_2d = np.zeros(N2)
    for i in range(N):
        for j in range(N):
            E_rot_2d[i * N + j] = E_A[i] + E_B[j]

    # ---- V_dd（旋转系 secular RWA，判据统一走 _secular_filter）----
    # 判据直接使用演化参考 E_rot_2d（已含常数光位移）——判据与演化一致。
    E_rot_filter = E_rot_2d.copy()

    V_diag = np.zeros(N2)
    V_sec = sp.csr_matrix((N2, N2), dtype=complex)
    n_kept_vdd = 0

    if vdd_matrix is not None:
        V_diag, V_sec, n_kept_vdd = _secular_filter(vdd_matrix,
                                                     E_rot_filter)
    elif R_um is not None:
        if cache is None:
            raise ValueError(
                '两体演化需要 cache（core.ArcCache 实例）来计算 V_dd；'
                '请传入 cache，或显式提供 vdd_matrix。'
                '（缺 cache 时 V_dd 会静默为零，两体退化为两独立单体。）')
        # V_dd 须在实际演化所用 states 的顺序/维度上构造，否则矩阵元错位。
        single = [dict(s) for s in states]
        V_diag, V_sec, n_kept_vdd = vdd_secular(
            cache, single, R_um, theta_deg, phi_deg, ref_A, ref_B, N,
            E_rot_2d=E_rot_filter)

    if log and n_kept_vdd > 0:
        log(f'[pair_evolve] V_dd secular: {n_kept_vdd} 保留'
            f'（共 {N2*(N2-1)//2} 对）')

    # ---- V_dd 对角 → 并入 E_rot ----
    E_rot_2d += V_diag

    # ---- 构建两体场 ----
    fields_2d = tensor_fields(fields_A, fields_B, states, N)

    # ---- V_dd 非对角 secular → 常数场（omega_L=0，保持稀疏：RK4 稀疏 matvec） ----
    if V_sec.nnz > 0:
        V_field = {
            'kind': 'multi',
            'M': V_sec.tocsr(),          # 稀疏（近简并元极少，matvec O(nnz)）
            'A': sp.csr_matrix((N2, N2)),
            'ac_shift': np.zeros(N2),
            'g_idx': [],
            'e_idx': [],
            'omega0_mhz': 1.0,
            'omega_L_mhz': 0.0,
            'waveform': square(0.0, 1e12),
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
    psi0_A_vec = _as_vector(states, psi0_A)
    psi0_B_vec = _as_vector(states, psi0_B)
    psi0_2d = _as_product_psi(psi0_A_vec, psi0_B_vec, N)

    if progress_arr is None:
        progress_arr = np.zeros(2)

    # ---- 自包含 RK4 演化（可选两体寿命耗散）----
    res = _pair_rwa(
        E_rot_2d, fields_2d, psi0_2d, t,
        frac=ip4_frac, progress_arr=progress_arr,
        gamma_A=gamma_A, gamma_B=gamma_B, decay_out=has_decay)
    if has_decay:
        t_out, pops_2d, psi_f, loss_A, loss_B = res
    else:
        t_out, pops_2d, psi_f = res
        loss_A = np.zeros(len(t_out))
        loss_B = np.zeros(len(t_out))

    # ---- 约化布居 ----
    pops_A = _reduce_A(pops_2d, N)
    pops_B = _reduce_B(pops_2d, N)

    # ---- 远失谐光 off-resonant 散射 depump（report-only 后处理）----
    # 光频移光（如 6S↔6P3/2）虽远失谐，仍以 σ=Ω²Γ_up/(4Δ²) 把源态布居
    # 打到短寿命上态（6P 自发辐射回不来）→ 长时间累积 depump。
    try:
        sc_A = _scatter_loss_curve(fields_A, pops_A, t_out, N)
        sc_B = _scatter_loss_curve(fields_B, pops_B, t_out, N)
        if np.any(sc_A) or np.any(sc_B):
            loss_A = loss_A + sc_A
            loss_B = loss_B + sc_B
            if log:
                log(f'[pair_evolve] 远失谐散射 depump：A '
                    f'{sc_A[-1]:.3e} / B {sc_B[-1]:.3e}')
    except Exception as e:
        if log:
            log(f'[pair_evolve] 散射 depump 补丁失败（不致命）：{e}')

    return dict(
        t=t_out,
        pops_2d=pops_2d,
        pops_A=pops_A,
        pops_B=pops_B,
        loss_A=loss_A,
        loss_B=loss_B,
        states_2d=[{
            'label': f"({states[i]['label']}) x ({states[j]['label']})",
            'E_mhz': E_rot_2d[i * N + j],
            'ket_i': i,
            'ket_j': j,
        } for i in range(N) for j in range(N)],
        method='ip4',
        backend='numpy',
    )



# ---- t19 函数级改名兼容别名（旧名 = 新名，外部零改动）----
_pair_ip4 = _pair_rwa
