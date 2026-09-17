# -*- coding: utf-8 -*-
"""下降沿判决实验：紧凑波形构造函数（**可被 CodeAct 直接 import**）。

为什么单独成模块：上一批扫描用的是"把整张包络折线写成 Python 字面量"的载荷
（11 点 × ~100 个折点，约 12 KB 代码），提交时软件进程静默消失。
本模块把折线**按参数生成**，调用方只需传几十字节 ⇒ 载荷降两个量级。

用法（CodeAct / 离线均可）：
    import sys; sys.path.insert(0, ROOT + '/scripts/research')
    from _edge_profiles import fs, profiles, MW
"""
import numpy as np

IRES = 0.42          # I_res / I3
I3 = 2.0e8           # mW/cm²
Q_FS = 295724187.476
Q_MW = 17236.743
I_MW = 5.5e-5
TMAX = 2000.0

# 两支微波 π（与存档一致）
MW = [
    {'name': 'p70', 'freq_mhz': Q_MW, 'intensity': I_MW, 'pol': 'σ-',
     'offset_ns': 0.0, 'env': ['gauss', 70.0, 20.0, 1.0]},
    {'name': 'p200', 'freq_mhz': Q_MW, 'intensity': I_MW, 'pol': 'σ-',
     'offset_ns': 0.0, 'env': ['gauss', 200.0, 20.0, 1.0]},
]


def _fs(breaks, name='频移光'):
    """折线 (t_ns, I/I3) → 波形 dict（env 是**幅度** ⇒ 开方；时间严格递增）。"""
    t, v = [], []
    for x, y in breaks:
        if t and x <= t[-1]:
            x = t[-1] + 1e-3
        t.append(float(x))
        v.append(float(y))
    return {'name': name, 'freq_mhz': Q_FS, 'intensity': I3, 'pol': 'π',
            'offset_ns': 0.0,
            'env': ['file', t, [float(np.sqrt(max(y, 0.0))) for y in v]]}


def tri(depth, tau, n):
    """I_res → depth → I_res 对称三角 excursion（**末态 = I_res**）。"""
    br = [(0.0, 0.0), (119.999, 0.0), (120.0, 1.0), (254.999, 1.0), (255.0, IRES)]
    for k in range(1, n + 1):
        br.append((255.0 + tau * k / n, IRES + (depth - IRES) * k / n))
    for k in range(1, n + 1):
        br.append((255.0 + tau + tau * k / n, depth - (depth - IRES) * k / n))
    br.append((TMAX, IRES))
    return br


def hold(depth, dw):
    """跳到 depth 停 dw，再跳回 I_res（E1 式）——穿越一次、瞬时。"""
    return [(0.0, 0.0), (119.999, 0.0), (120.0, 1.0), (254.999, 1.0),
            (255.0, depth), (255.0 + dw, depth), (255.0 + dw + 0.001, IRES),
            (TMAX, IRES)]


def down(tau):
    """单向下降（前人 F 族定义）。tau=0 为严格阶跃。"""
    return [(0.0, 0.0), (119.999, 0.0), (120.0, 1.0), (255.0, 1.0),
            (255.0 + tau, IRES), (TMAX, IRES)]


def step_at(t_jump):
    """在 t_jump 处从 I3 阶跃到 I_res（边沿时刻可控）。

    用途：判决"同样回到 I_res 但边沿时刻不同"是否有影响。
    """
    return [(0.0, 0.0), (119.999, 0.0), (120.0, 1.0), (t_jump, 1.0),
            (t_jump + 0.001, IRES), (TMAX, IRES)]


def ramp_at(t_start, tau):
    """在 [t_start, t_start+tau] 内线性下坡 I3 → I_res。"""
    return [(0.0, 0.0), (119.999, 0.0), (120.0, 1.0), (t_start, 1.0),
            (t_start + tau, IRES), (TMAX, IRES)]


def normal(tau, t_start=280.0):
    """**正常下降沿**（前人 F 族定义）：平台 I3 → 在 t_start 处线性下坡 τ 到 I_res。

    与前人 F/tf 族逐条同构，便于与离线数据对拍。
    """
    if tau <= 0:
        return [(0.0, 0.0), (119.999, 0.0), (120.0, 1.0),
                (t_start - 0.001, 1.0), (t_start, IRES), (TMAX, IRES)]
    # 用多点逼近线性（避免 FileEnv 折点间插值带来的分段感）
    n = max(8, int(round(tau)))
    br = [(0.0, 0.0), (119.999, 0.0), (120.0, 1.0), (t_start, 1.0)]
    for k in range(1, n + 1):
        br.append((t_start + tau * k / n, 1.0 - (1.0 - IRES) * k / n))
    br.append((TMAX, IRES))
    out, last = [], None
    for t, v in br:
        if last is not None and t <= last:
            continue
        out.append((t, v))
        last = t
    return out


def one(tag, br, npts=1001, R=14.1, theta=31.1):
    """一条轨迹 → submit_sweep 的一个 point。"""
    return {'dynamic.t_max': TMAX, 'dynamic.npts': npts,
            'dynamic.pair.R_um': R, 'dynamic.pair.theta_deg': theta,
            'dynamic.method': 'mixed',
            'dynamic.waves_A': MW + [_fs(br)]}


def profiles(kind):
    """按族取 (tag, breaks, npts) 列表。点序 = tag 顺序。"""
    if kind == 'A':                       # 对称往返（判决"往返是否恢复幅度"）
        return [('A/tri25_d1.00', tri(1.00, 25.0, 25), 1001),
                ('A/tri50_d1.00', tri(1.00, 50.0, 50), 1001),
                ('A/tri100_d1.00', tri(1.00, 100.0, 100), 1001)]
    if kind == 'B':                       # 停留式
        return [('B/hold10@0.71', hold(0.71, 10.0), 1001),
                ('B/hold25@0.71', hold(0.71, 25.0), 1001),
                ('B/hold50@0.71', hold(0.71, 50.0), 1001)]
    if kind == 'C':                       # 单向下降（参照 + 交叉校验）
        return [('C/step', down(0.0), 1001),
                ('C/down25', down(25.0), 1001),
                ('C/down50', down(50.0), 1001)]
    if kind == 'fine':                    # 收敛检查：细时间网格
        return [('X/tri50_d1.00_n2001', tri(1.00, 50.0, 50), 2001),
                ('X/tri50_d1.00_n501', tri(1.00, 50.0, 50), 501)]
    if kind == 'ctrl':                    # 时序对照：同样回到 I_res，边沿时刻不同
        return [('T/step@255', step_at(255.0), 1001),
                ('T/step@305', step_at(305.0), 1001),
                ('T/step@400', step_at(400.0), 1001),
                ('T/step@600', step_at(600.0), 1001),
                ('T/ramp50@255', ramp_at(255.0, 50.0), 1001),
                ('T/ramp50@305', ramp_at(305.0, 50.0), 1001)]
    if kind == 'smooth':                  # 平滑度：同一轨迹不同折线采样点数
        return [('S/ramp_n%d' % n, _ramp_seg(n), 1001)
                for n in (1, 2, 4, 8, 16, 32, 64, 128, 256, 1000)]
    if kind == 'dense':                   # 正常下降沿密扫 25→125 ns（每 5 ns）
        return [('D/tf%d' % tf, normal(float(tf)), 1001)
                for tf in range(25, 126, 5)]
    raise ValueError('unknown kind %r' % kind)


def _ramp_seg(nseg):
    """I3 → I_res 的 50 ns 下坡，用 nseg 段折线逼近（与提交时同一条代码路径）。"""
    br = [(0.0, 0.0), (119.999, 0.0), (120.0, 1.0)]
    for k in range(nseg + 1):
        br.append((255.0 + 50.0 * k / nseg, 1.0 - (1.0 - IRES) * k / nseg))
    br.append((TMAX, IRES))
    out, last = [], None
    for t, v in br:
        if last is not None and t <= last:
            continue
        out.append((t, v))
        last = t
    return out