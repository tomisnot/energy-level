# -*- coding: utf-8 -*-
"""封存：动态演化的 Floquet 传播子 与 RWA 分段常数（pwc）算法（独立可用）。

**封存背景（2025-08）**：动态分析实际**从来只用相互作用绘景 RK4（ip4）**——
单体动态分析相当快速（典型 ~秒级），RK4 足够快，Floquet 没有适用场景；
pwc（RWA 分段常数，最快预览）同样不用于日常分析。为减少 `evolve.py`
阅读成本，两者已从 `src/core/evolve.py` 移除，按 ttl_pulse 的方式封存待用。

**什么时候需要本模块**：
- 需要「精确 Floquet 传播子」的周期驱动研究（常数包络 + 单频，TDSE 精确解，
  成本不随时长增长、无光子块截断）→ `evolve_floquet`；
- 需要 RWA 快速预览 / RWA 误差量化（级联 RWA 参考、BS 一阶修正回归）
  → `evolve_pwc`（或底层 `_evolve_pwc`）。

**用法**（与旧 `evolve.evolve(method='floquet'/'pwc')` 等价）：

    from src._archive.evolve_floquet_pwc import evolve_floquet, evolve_pwc
    r = evolve_floquet(states, fields, psi0, t_grid)          # 常数包络 + 单频
    r = evolve_pwc(states, fields, psi0, t_grid, bs=True)     # RWA 分段常数 + BS

返回 dict 结构与 `evolve.evolve` 一致（t/pops/states/om_mhz/method/backend/loss）。

单位：与 evolve.py 一致（MHz, ns, rad/ns）。
"""
import numpy as np

from ..core.evolve_rwa import _assign_reference
from ..core.evolve_util import as_vector, decay_ops
from ..core.evolve_rwa import RWA_TOL  # t80: 原 IP4_RWA_THRESH 更名
from ..core.field_build import env_samples, field_geometry

CONV = 2 * np.pi * 1e-3          # MHz·ns → rad


# ============================================================
# 后端选择（仅 pwc 用）
# ============================================================
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
# 精确 Floquet 传播子（常数包络 + 单频，第一性原理，成本不随时长增长）
# ============================================================
def _evolve_floquet(E, mats, omega_mhz, psi0, t, n_sub=2000, record_amp=False,
                    progress_arr=None):
    """H(t) = diag(E) + cos(CONV·ω·t)·Σ_k mats_k（单频 ω、常数包络）。

    精确积分一个周期 T=2π/(CONV·ω)（中点法 n_sub 步，存 U(τ) 网格）
    → U(T) 本征分解（准能量 ε_k = −arg(λ_k)/T）
    → 任意 t = nT + τ：ψ(t) = U(τ)·W·diag(e^{−iεnT})·W†·ψ0。

    对周期驱动 = TDSE 精确解（无 RWA、无光子块截断）；误差仅来自单周期
    数值积分（2 阶收敛，n_sub 加密到机器精度）。成本 = 一个周期积分 +
    一次对角化 + 每输出时刻 O(N²)，**不随模拟时长增长**。
    progress_arr（可选）：np.float64[2]，[0]=进度 0..1，[1]≠0 请求取消。
    """
    w = CONV * omega_mhz
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
        D = np.exp(-1j * CONV * dt * e)
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


def evolve_floquet(states, fields, psi0, t_grid, ac_shift_mhz=None,
                   record_amp=False, n_sub=2000, progress_arr=None):
    """等价旧 `evolve.evolve(method='floquet')`（封存后独立入口）。

    要求：全部光场常数包络且单一频率；空场或无场时抛 ValueError。
    返回 dict(t, pops, states, om_mhz, method='floquet', backend='numpy',
    loss, [psi_amp])，与 evolve.evolve 同构。
    """
    t = np.asarray(t_grid, float)
    psi0v = as_vector(states, psi0)
    E = np.array([s['E_mhz'] for s in states])
    fields = list(fields)
    if ac_shift_mhz is not None:
        E = E + np.asarray(ac_shift_mhz, float)
        for f in fields:
            f.pop('ac_shift', None)
            if 'M' in f:
                ac0 = np.diag(f['M']).copy()
                f['M'] = f['M'] - np.diag(ac0)
    if progress_arr is None:
        progress_arr = np.zeros(2)

    geoms = [field_geometry(states, f) for f in fields]
    if not geoms:
        raise ValueError('evolve_floquet 需要至少一个光场（常数包络 + 单频）。')
    w0 = geoms[0]['omega_L_mhz']
    ok_freq = all(abs(g['omega_L_mhz'] - w0)
                  <= 1e-6 * max(abs(w0), 1.0) for g in geoms)
    ok_env = all(env_samples(g['waveform'], t[-1])[1] for g in geoms)
    if not (ok_freq and ok_env):
        raise ValueError(
            'evolve_floquet 要求全部光场常数包络且单一频率'
            f'（单频={ok_freq}，常数包络={ok_env}）；时变包络/多频请用'
            " evolve.evolve(method='ip4')。")
    mats = []
    E_f = np.array(E, float)
    for g in geoms:
        c = env_samples(g['waveform'], t[-1])[0]
        mats.append(g['omega0_mhz'] * c * g['A'])
        a = g.get('ac_shift')
        if a is not None and np.any(np.asarray(a, float) != 0.0):
            E_f = E_f + c * np.asarray(a, float)
    t, pops, psi = _evolve_floquet(E_f, mats, w0, psi0v, t,
                                   n_sub=n_sub, record_amp=record_amp,
                                   progress_arr=progress_arr)
    out = dict(t=t, pops=pops, states=states,
               om_mhz=[g['omega0_mhz'] for g in geoms],
               method='floquet', backend='numpy',
               loss=np.zeros(len(t)))
    if record_amp:
        out['psi_amp'] = psi
    return out


# ============================================================
# RWA 分段常数（最快预览；torch GPU 批量，pops 全程 GPU 累积）
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
            ph = torch.exp(-1j * CONV * dts[m] * ev[m])
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
            ph = np.exp(-1j * CONV * dt[m] * e)
            psi = U @ (ph * (U.T @ psi))
            pops[m + 1] = np.abs(psi) ** 2
    progress_arr[0] = 1.0
    if record_amp:
        return t, pops, psi
    return t, pops, None


def evolve_pwc(states, fields, psi0, t_grid, ac_shift_mhz=None,
               backend='auto', record_amp=False, bs=False, progress_arr=None):
    """等价旧 `evolve.evolve(method='pwc')`（封存后独立入口）。

    bs：Bloch–Siegert 一阶修正（Ω²/4ω_L 对角）。返回 dict(t, pops, states,
    om_mhz, method='pwc', backend='torch'|'numpy', loss, [psi_amp])。
    """
    t = np.asarray(t_grid, float)
    psi0v = as_vector(states, psi0)
    E = np.array([s['E_mhz'] for s in states])
    fields = list(fields)
    if ac_shift_mhz is not None:
        E = E + np.asarray(ac_shift_mhz, float)
        for f in fields:
            f.pop('ac_shift', None)
            if 'M' in f:
                ac0 = np.diag(f['M']).copy()
                f['M'] = f['M'] - np.diag(ac0)
    if progress_arr is None:
        progress_arr = np.zeros(2)
    ref, deltas = _assign_reference(states, fields)
    H0 = np.diag(E - ref).astype(float)
    geoms = [field_geometry(states, f) for f in fields]
    for g, d in zip(geoms, deltas):
        wL = g['omega_L_mhz']
        g['residual_mhz'] = d
        if abs(d) > 1e-6:
            # 驱动图闭环不一致：残留拍频按 RWA 丢弃（与旧行为一致）
            pass
        g['bs_shift'] = (g['omega0_mhz'] ** 2 / (4.0 * wL)
                         if (bs and wL > 1e-6) else 0.0)
        r0 = g.get('real_omega0_mhz', g['omega0_mhz'])
        ratio = r0 / max(wL, 1e-9)
        if ratio >= RWA_TOL:
            # 强驱动（Ω/ωL ≥ 1e-4）：RWA 误差非可忽略，提示用 ip4
            pass
    use_torch = _pick_backend(backend)
    t, pops, psi = _evolve_pwc(H0, geoms, psi0v, t, use_torch,
                               record_amp, progress_arr)
    out = dict(t=t, pops=pops, states=states,
               om_mhz=[g['omega0_mhz'] for g in geoms],
               method='pwc', backend='torch' if use_torch else 'numpy',
               loss=np.zeros(len(t)))
    if record_amp:
        out['psi_amp'] = psi
    return out
