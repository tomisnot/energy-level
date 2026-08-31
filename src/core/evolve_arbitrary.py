# -*- coding: utf-8 -*-
"""任意波形演化（t19：evolve.py 拆分）——实验室系 RK4，无 RWA/旋转系。"""
import numpy as np

try:
    from numba import njit as _njit
except Exception:
    _njit = None

_CONV = 2 * np.pi * 1e-3            # MHz·ns → rad


@_njit(cache=True, nogil=True) if _njit is not None else (lambda f: f)
def _arb_interp(tt, t_samp, a_samp):
    """信号幅值插值（越界取 0；样本升序）。Catmull-Rom 三次：C¹ 连续、
    4 阶局部精度——对快速载波（≥2~4 点/周期）远优于线性（线性插值的
    每周期相位漂移会让载频悄悄失谐，长时演化崩坏）。"""
    n = t_samp.size
    if tt < t_samp[0] or tt > t_samp[-1]:
        return 0.0
    i = np.searchsorted(t_samp, tt)
    if i == 0:
        i = 1
    elif i == n:
        i = n - 1
    t1, t2 = t_samp[i - 1], t_samp[i]
    h = t2 - t1
    if h <= 0.0:
        return a_samp[i - 1]
    i0 = i - 2 if i - 2 >= 0 else 0
    i3 = i + 1 if i + 1 < n else n - 1
    m1 = (a_samp[i] - a_samp[i0]) / (t_samp[i] - t_samp[i0]) \
        if t_samp[i] - t_samp[i0] > 0 else 0.0
    m2 = (a_samp[i3] - a_samp[i - 1]) / (t_samp[i3] - t_samp[i - 1]) \
        if t_samp[i3] - t_samp[i - 1] > 0 else 0.0
    x = (tt - t1) / h
    x2, x3 = x * x, x * x * x
    p1, p2 = a_samp[i - 1], a_samp[i]
    return (p1 * (2.0 * x3 - 3.0 * x2 + 1.0)
            + h * m1 * (x3 - 2.0 * x2 + x)
            + p2 * (-2.0 * x3 + 3.0 * x2)
            + h * m2 * (x3 - x2))


@_njit(cache=True, nogil=True) if _njit is not None else (lambda f: f)


def _arb_apply(E, Vs, t_samp_k, a_samp_k, ts, dt, ps_in, Vscr, kout):
    """kout = dt·(−iC)·Ṽ(ts)·ps_in；Ṽ=U·(Σ a_k(ts)·V_k)·U†，U=diag(e^{iC E ts})。"""
    C = 2.0 * np.pi * 1e-3
    N = E.shape[0]
    nk = Vs.shape[0]
    Vscr.fill(0.0)
    for k in range(nk):
        a = _arb_interp(ts, t_samp_k[k], a_samp_k[k])
        if a != 0.0:
            for i in range(N):
                for j in range(N):
                    Vscr[i, j] = Vscr[i, j] + a * Vs[k, i, j]
    for i in range(N):
        ui_re = np.cos(C * E[i] * ts)
        ui_im = np.sin(C * E[i] * ts)
        for j in range(N):
            uj_re = np.cos(C * E[j] * ts)
            uj_im = -np.sin(C * E[j] * ts)
            re = Vscr[i, j].real * (ui_re * uj_re - ui_im * uj_im) \
                - Vscr[i, j].imag * (ui_re * uj_im + ui_im * uj_re)
            im = Vscr[i, j].real * (ui_re * uj_im + ui_im * uj_re) \
                + Vscr[i, j].imag * (ui_re * uj_re - ui_im * uj_im)
            Vscr[i, j] = re + 1j * im
    for i in range(N):
        acc = 0.0j
        for j in range(N):
            acc = acc + Vscr[i, j] * ps_in[j]
        kout[i] = dt * (-1j * C * acc)


@_njit(cache=True, nogil=True) if _njit is not None else (lambda f: f)


def _arb_rk4_core(E, Vs, t_samp_k, a_samp_k, psi0, t_out, dt,
                  progress_arr=None):
    """相互作用绘景 RK4（任意波形第一性原理）：

        ψ_I = e^{iC H0 t} ψ；Ṽ(t) = U(t)·(Σ a_k(t)·V_k)·U(t)†
    输出按 t_out 逐点对相邻步线性内插记录。
    """
    C = 2.0 * np.pi * 1e-3
    N = E.shape[0]
    n_out = t_out.shape[0]
    T = t_out[-1]
    n_steps = int(np.ceil(T / dt))
    Vscr = np.zeros((N, N), np.complex128)
    k1 = np.zeros(N, np.complex128)
    k2 = np.zeros(N, np.complex128)
    k3 = np.zeros(N, np.complex128)
    k4 = np.zeros(N, np.complex128)
    psi_prev = psi0.copy()
    psi = psi0.copy()
    out = np.zeros((n_out, N), np.complex128)
    out[0] = psi0
    oi = 1
    for m in range(n_steps):
        tm = m * dt
        if progress_arr is not None:
            progress_arr[0] = tm / T
            if progress_arr[1] != 0.0:
                raise RuntimeError('用户取消')
        _arb_apply(E, Vs, t_samp_k, a_samp_k, tm, dt, psi, Vscr, k1)
        _arb_apply(E, Vs, t_samp_k, a_samp_k, tm + 0.5 * dt, dt,
                   psi + 0.5 * k1, Vscr, k2)
        _arb_apply(E, Vs, t_samp_k, a_samp_k, tm + 0.5 * dt, dt,
                   psi + 0.5 * k2, Vscr, k3)
        _arb_apply(E, Vs, t_samp_k, a_samp_k, tm + dt, dt,
                   psi + k3, Vscr, k4)
        psi_prev[:] = psi[:]
        for i in range(N):
            psi[i] = psi[i] + (k1[i] + 2.0 * k2[i] + 2.0 * k3[i] + k4[i]) / 6.0
        while oi < n_out and t_out[oi] <= tm + dt:
            w = (t_out[oi] - tm) / dt
            for i in range(N):
                out[oi, i] = psi_prev[i] + w * (psi[i] - psi_prev[i])
            oi += 1
        if oi >= n_out:
            break
    return out, psi




def _evolve_arbitrary(E, states, fields, psi0, t, frac=80.0,
                      progress_arr=None):
    """任意波形第一性原理演化（实验室系，相对 H0 相互作用绘景 RK4）。

    - 不 RWA、不旋转系、不近/远失谐分流：信号的全部频率成分由数值积分
      原样还原（含远失谐对在相互作用绘景中的贡献——真正的第一性原理）；
    - 步长约束：① 残余跃迁频率（耦合对 |E_i−E_j|）② 信号 Nyquist
      （最小采样间隔）③ RK4 稳定/精度——取最小，慢是必要代价；
      frac 默认 80（RK4 4 阶收敛，20ns/Ω~10⁴MHz 场景达 ~3e-5）；
    - 信号还原用 Catmull-Rom 三次插值（线性插值对快载波有相位漂移）；
    - 只接受 kind='arb' 场（混入普通载波场会 ValueError）。
    """
    N = len(states)
    arbs = [f for f in fields if (f or {}).get('kind') == 'arb']
    if len(arbs) != len(fields):
        raise ValueError(
            "任意波形求解（method='arbitrary'）要求全部光场都是任意波形"
            '场；普通载波场请在波形面板用高斯/方波与 ip4。')
    E = np.asarray(E, float) - np.min(np.asarray(E, float))   # 公共偏移（精确等价）
    Vs, t_samp_k, a_samp_k = [], [], []
    w_sig = 0.0
    w_tran = 0.0
    vmax_total = 0.0
    for f in arbs:
        Vs.append(np.asarray(f['V'], complex))
        tt = np.asarray(f['t_ns'], float) + float(f['offset_ns'])
        aa = np.asarray(f['amp'], float)
        order = np.argsort(tt)
        tt, aa = tt[order], aa[order]
        t_samp_k.append(tt)
        a_samp_k.append(aa)
        dts = np.diff(tt)
        dts = dts[dts > 0]
        if dts.size:
            rs = dts.min()
            w_sig = max(w_sig, np.pi / rs)          # 信号 Nyquist（rad/ns）
        vmax_total += f.get('V_max',
                            float(np.abs(np.asarray(f['V'], complex)).max()))
    for i in range(N):
        for j in range(i + 1, N):
            if any(abs(Vs[k][i, j]) > 1e-12 for k in range(len(Vs))):
                w_tran = max(w_tran, _CONV * abs(E[i] - E[j]))  # rad/ns
    w_max = w_tran + w_sig
    dt = min(
        2.0 * np.pi / (max(w_max, 1e-9) * frac),       # 残余振荡/信号解析
        0.004 / (_CONV * max(vmax_total, 1e-6)),       # dt·‖Ṽ‖ ≲ 0.004
        # （相互作用绘景 RK4 误差 ≈ 步数×(C·Ω·dt)⁵ 线性累积；0.06（ip4）在
        #  强耦合下 ~1e-3，0.004 达到 ~1e-5，代价步数 ×15——必要精度成本）
        2.0 / (_CONV * max(vmax_total, 1e-6)),         # RK4 稳定上界
        t[-1] / 500.0,
        0.05)
    t = np.asarray(t, float)
    Vs_arr = np.asarray(Vs, complex)
    out, psi = _arb_rk4_core(
        np.asarray(E, float), Vs_arr,
        [np.asarray(x, float) for x in t_samp_k],
        [np.asarray(x, float) for x in a_samp_k],
        np.asarray(psi0, complex), t, dt, progress_arr=progress_arr)
    pops = np.abs(out) ** 2
    return t, pops, psi


# ---------------- 一站式入口 ----------------
