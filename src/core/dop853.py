# -*- coding: utf-8 -*-
"""DOP853 核心层（与态空间维度 N 无关）——重构第 2a 步。

从 hard_numba.py 抽取共享内核：两处驱动（单体/两体）的步进/误差/步长/
稠密输出逻辑逐行同构，仅 RHS 调用不同。numba 支持 njit 函数作为 first-class
参数（_numba_funcarg_test.py 已验证），故统一为 _dop853_driver(rhs, ...)，
rhs 为任意 njit 编译的 RHS 回调（单体/两体通用）。

- 系数（_A/_C/_B/_E3/_E5/_D）：**直接复用 hard**（scipy 1.15.2 逐行
  移植、2.9e-10 基线验证，t19：hard_numba→hard）——禁止手写重建
  （曾因手写系数 54 处错误）。
- _rms_norm                 ：范数（纯工具；_env_at/_env_tab 单源在 hard.py）
- _dop853_step / _dop853_dense_interp：单步与稠密插值（与 RHS 参数化解耦）
- _dop853_driver                       ：主循环（rhs 参数化）

使用方：hard.evolve_hard_fast（单体）、hard.evolve_hard_pair（两体）。
"""
import numpy as np
from numba import njit

from .hard import _A, _C, _B, _E3, _E5, _D  # 复用已验证系数
from .params import _TAB_N  # t74c 参数单源

_np = np


@njit(cache=True, nogil=True)
def _rms_norm(x):
    """RMS 范数 sqrt(Σ|x|²/n)（scipy norm 移植）。"""
    n = x.size
    s = 0.0
    for i in range(n):
        s += x[i].real * x[i].real + x[i].imag * x[i].imag
    return _np.sqrt(s / n)


@njit(cache=True, nogil=True)
def _dop853_step(t, y, f, h, K, rhs, *rargs):
    """单步：12 级显式 RK（rhs 为 njit 回调，rargs 为其附加参数）。"""
    n = y.size
    for i in range(n):
        K[0, i] = f[i]
    for s in range(1, 13):
        dy = _np.zeros(n, _np.complex128)
        for j in range(s):
            a = _A[s, j]
            if a != 0.0:
                for i in range(n):
                    dy[i] += a * K[j, i]
        for i in range(n):
            dy[i] *= h
        K[s] = rhs(t + _C[s] * h, y + dy, *rargs)
    y_new = _np.empty(n, _np.complex128)
    for i in range(n):
        acc = y[i]
        for j in range(12):
            b = _B[j]
            if b != 0.0:
                acc += h * b * K[j, i]
        y_new[i] = acc
    f_new = rhs(t + h, y_new, *rargs)
    for i in range(n):
        K[12, i] = f_new[i]
    return y_new, f_new


@njit(cache=True, nogil=True)
def _dop853_dense_interp(y_old, y_new, f_old, f_new, K, t_old, h, x,
                         rhs, *rargs):
    """Hairer contd8 稠密输出：补算 13..15 级 + 7 次多项式插值（scipy 移植，
    与 hard_numba 原版逐行等价；rhs 参数化）。"""
    n = y_old.size
    for s in range(13, 16):
        dy = _np.zeros(n, _np.complex128)
        for j in range(s):
            a = _A[s, j]
            if a != 0.0:
                for i in range(n):
                    dy[i] += a * K[j, i]
        for i in range(n):
            dy[i] *= h
        K[s] = rhs(t_old + _C[s] * h, y_old + dy, *rargs)
    F = _np.zeros((7, n), _np.complex128)
    for i in range(n):
        d = y_new[i] - y_old[i]
        F[0, i] = d
        F[1, i] = h * f_old[i] - d
        F[2, i] = 2.0 * d - h * (f_new[i] + f_old[i])
    for r in range(4):
        for j in range(16):
            dij = _D[r, j]
            if dij != 0.0:
                for i in range(n):
                    F[3 + r, i] += h * dij * K[j, i]
    y = _np.zeros(n, _np.complex128)
    for i in range(6, -1, -1):
        for k in range(n):
            y[k] += F[i, k]
        if (6 - i) % 2 == 0:
            for k in range(n):
                y[k] *= x
        else:
            for k in range(n):
                y[k] *= (1.0 - x)
    for k in range(n):
        y[k] += y_old[k]
    return y


@njit(cache=True, nogil=True)
def _dop853_driver(rhs, y0, t_eval, T, rtol, atol, max_step, prog, stat,
                   *rargs):
    """scipy solve_ivp(DOP853) 主循环的 numba 移植（rhs 参数化，任意 N）。

    rhs(t, y, *rargs) -> dydt（njit 编译）。rargs 承载场矩阵/查表等附加参数。
    返回 y_out (n_eval, n) 复数；stat[0..3] = nfev/n_acc/n_rej/n_dense。
    步进/误差/步长控制与 scipy rk.py 逐行一致（SAFETY=0.9 等）。
    """
    n = y0.size
    n_eval = t_eval.size
    SAFETY = 0.9
    MIN_FACTOR = 0.2
    MAX_FACTOR = 10.0
    EXP = -1.0 / 8.0
    t0 = 0.0
    K = _np.zeros((16, n), _np.complex128)
    y_out = _np.zeros((n_eval, n), _np.complex128)
    nfev = 0
    n_acc = 0
    n_rej = 0
    n_dense = 0

    y = y0.copy()
    f = rhs(t0, y, *rargs)
    nfev += 1
    i_eval = 0
    while i_eval < n_eval and t_eval[i_eval] <= t0:
        for k in range(n):
            y_out[i_eval, k] = y[k]
        i_eval += 1
    if i_eval >= n_eval or t0 >= T:
        stat[0] = nfev
        stat[1] = n_acc
        stat[2] = n_rej
        stat[3] = n_dense
        return y_out

    # ---- select_initial_step（scipy common.py，order=7）----
    scale = _np.empty(n)
    for i in range(n):
        scale[i] = atol + _np.abs(y[i]) * rtol
    d0 = _rms_norm(y / scale)
    d1 = _rms_norm(f / scale)
    interval = T - t0
    if d0 < 1e-5 or d1 < 1e-5:
        h0 = 1e-6
    else:
        h0 = 0.01 * d0 / d1
    if h0 > interval:
        h0 = interval
    y1 = y + h0 * f
    f1 = rhs(t0 + h0, y1, *rargs)
    nfev += 1
    d2 = _rms_norm((f1 - f) / scale) / h0
    if d1 <= 1e-15 and d2 <= 1e-15:
        h1 = max(1e-6, h0 * 1e-3)
    else:
        h1 = (0.01 / max(d1, d2)) ** (1.0 / 8.0)
    h_abs = min(100.0 * h0, h1, interval, max_step)

    # ---- 主循环 ----
    t = t0
    # t72 进度映射：自适应步长下 t/T 与墙钟严重不符（stiff 脉冲区步长比
    # 自由区小 1-2 个量级 → 条在 stiff 区爬、末段瞬间冲满）。nogil 内核
    # 中每次 RHS 求值开销恒定 → nfev ∝ 墙钟。用平滑的局部求值密度 rate
    # （RHS 次数/模拟时间，EMA 0.7/0.3）外推剩余求值数：
    #     p = nfev / (nfev + rate·(T−t))
    # 条的移动速度跟随真实墙钟速度，未来变贵/变便宜时自动减速/加速；
    # p 单调不减（钳制）、封顶 0.999（收尾阶段接管最后一段）。
    rate_est = 0.0
    prev_t = 0.0
    prev_nfev = 1
    p_max = 0.0
    stage_set = False
    while t < T and i_eval < n_eval:
        step_accepted = False
        step_rejected = False
        while not step_accepted:
            h = h_abs
            t_new = t + h
            if t_new > T:
                t_new = T
                h = t_new - t
            y_new, f_new = _dop853_step(t, y, f, h, K, rhs, *rargs)
            nfev += 12
            # 误差估计（scipy _estimate_error_norm）
            err5_n2 = 0.0
            err3_n2 = 0.0
            for i in range(n):
                sc = atol + max(_np.abs(y[i]), _np.abs(y_new[i])) * rtol
                e5 = 0.0j
                e3 = 0.0j
                for j in range(13):
                    e5 += _E5[j] * K[j, i]
                    e3 += _E3[j] * K[j, i]
                e5 = e5 / sc
                e3 = e3 / sc
                err5_n2 += e5.real * e5.real + e5.imag * e5.imag
                err3_n2 += e3.real * e3.real + e3.imag * e3.imag
            if err5_n2 == 0.0 and err3_n2 == 0.0:
                error_norm = 0.0
            else:
                error_norm = (_np.abs(h) * err5_n2
                              / _np.sqrt((err5_n2 + 0.01 * err3_n2) * n))
            if error_norm < 1.0:
                if error_norm == 0.0:
                    factor = MAX_FACTOR
                else:
                    factor = min(MAX_FACTOR, SAFETY * error_norm ** EXP)
                if step_rejected:
                    factor = min(1.0, factor)
                h_abs *= factor
                step_accepted = True
            else:
                h_abs *= max(MIN_FACTOR, SAFETY * error_norm ** EXP)
                step_rejected = True
                n_rej += 1
        h_step = h
        n_acc += 1
        # t72：首个接受步后置阶段=1（积分）——驱动调用内的 numba 装载/
        # 编译期保持调用侧的阶段 0（"准备"标签 + 墙钟耗时），GUI 不失真
        if (not stage_set) and prog.shape[0] > 2:
            prog[2] = 1.0
            stage_set = True
        # t72：预测剩余工作量的进度映射（见循环前注释）
        dt_loc = t_new - prev_t
        if dt_loc > 0.0:
            inst = (nfev - prev_nfev) / dt_loc
            if rate_est == 0.0:
                rate_est = inst
            else:
                rate_est = 0.7 * rate_est + 0.3 * inst
        prev_t = t_new
        prev_nfev = nfev
        denom = nfev + rate_est * (T - t_new)
        p_now = nfev / denom if denom > 0.0 else 0.0
        if p_now > 0.999:
            p_now = 0.999
        if p_now < p_max:
            p_now = p_max
        p_max = p_now
        prog[0] = p_now
        # t21：取消支持——GUI 置 prog[1]≠0 时抛 RuntimeError（numba njit 内
        # raise 已验证兼容），上层（worker）按 progress_arr[1]≠0 → 'cancelled'
        if prog[1] != 0.0:
            raise RuntimeError('cancelled')
        # 稠密输出：记录 (t, t_new] 内的 t_eval 点
        if i_eval < n_eval and t_eval[i_eval] <= t_new:
            dense_done = False
            while i_eval < n_eval and t_eval[i_eval] <= t_new:
                x = (t_eval[i_eval] - t) / h_step
                if x <= 0.0:
                    yq = y
                elif x >= 1.0:
                    yq = y_new
                else:
                    if not dense_done:
                        dense_done = True
                        n_dense += 1
                        nfev += 3
                    yq = _dop853_dense_interp(y, y_new, f, f_new, K, t,
                                              h_step, x, rhs, *rargs)
                for k in range(n):
                    y_out[i_eval, k] = yq[k]
                i_eval += 1
        t = t_new
        y = y_new
        f = f_new
    stat[0] = nfev
    stat[1] = n_acc
    stat[2] = n_rej
    stat[3] = n_dense
    return y_out
