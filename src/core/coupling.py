# -*- coding: utf-8 -*-
"""coupling.py（t63 第 1 层统一）：RHS 打包公共层——单/两体共用。

从 evolve_mixed_pair._pack_rhs + evolve_mixed._MixedSetup 的波构建抽出泛化：
- pack_rhs(pairs, t_end, n_tab)：pairs 7 元组 → njit 数组（分组键
  (is_mw, w, id(env))——t62 用例 1 教训：光场/微波即使同一 env 对象也必须分属
  不同组——is_mw 决定 cos 载波；env 表预采样 + 线性插值）
- 返回 dict: iis, jjs, vals, dDs, is_mw, wcos, n_pairs_w, env_tab, env_dt
"""
import numpy as np


def stage_progress(arr, st, frac):
    """三槽进度协议助手（t72，GUI ui_dynamic._poll_progress 消费）。

    arr 槽位：[0]=**阶段内**进度 0..1、[1]=取消标志（只读）、[2]=阶段
    （0 准备/JIT 装载、1 积分、2 收尾）。engines 在各阶段边界调用本助手；
    积分阶段由 _dop853_driver 逐步写 arr[0]=t/T。arr 短于 3（旧脚本
    len-2 用法）时只写 arr[0]、跳过阶段槽——旧消费者语义不变。
    """
    if arr is None:
        return
    arr[0] = min(max(float(frac), 0.0), 1.0)
    if len(arr) > 2:
        arr[2] = float(st)


def pack_rhs(pairs, t_end, n_tab=20001):
    """pairs → numba 数组（按 (is_mw, w, id(env)) 分组 + env 表预采样）。

    pairs 元素：(a, b, v, dD, is_mw, w, env)——分组键含 is_mw/w（t62 用例 1
    教训：光场/微波即使 env 同一对象也须分属不同组——is_mw 决定 cos 载波）。
    """
    grp = {}
    order = []
    for p in pairs:
        a, b, v, dD, is_mw, w, env = p
        key = (int(is_mw), float(w), id(env))
        if key not in grp:
            grp[key] = dict(is_mw=is_mw, w=w, env=env, items=[])
            order.append(key)
        grp[key]['items'].append((a, b, v, dD))
    iis, jjs, vals, dDs = [], [], [], []
    is_mw_a, wcos_a, n_ppw = [], [], []
    env_rows = []
    for key in order:
        g = grp[key]
        for (a, b, v, dD) in g['items']:
            iis.append(a); jjs.append(b); vals.append(v); dDs.append(dD)
        n_ppw.append(len(g['items']))
        is_mw_a.append(1 if g['is_mw'] else 0)
        wcos_a.append(g['w'])
        ts = np.linspace(0.0, t_end, n_tab)
        env_rows.append(np.asarray(g['env'](ts), float))
    env_tab = np.stack(env_rows) if env_rows else np.zeros((0, n_tab))
    env_dt = t_end / (n_tab - 1) if n_tab > 1 else 1.0
    return dict(iis=np.asarray(iis, np.int64),
                jjs=np.asarray(jjs, np.int64),
                vals=np.asarray(vals, np.complex128),
                dDs=np.asarray(dDs, np.float64),
                is_mw=np.asarray(is_mw_a, np.int64),
                wcos=np.asarray(wcos_a, np.float64),
                n_pairs_w=np.asarray(n_ppw, np.int64),
                env_tab=env_tab, env_dt=env_dt)

# ============================================================
# φ-帧统一 RHS 内核（t73：mixed 单原子/两体同一实现——t66 语义收敛）
# 输入 = pack_rhs 的输出数组（iis/jjs/vals/dDs 按 a<b 单侧打包，内核厄米
# 补齐反向）+ 逐波时变 AC 对角（ac_base (n_ac, dim) + ac_env_tab）。
# is_mw[w]=1 的波带 cos(C·w·t) 载波；is_mw=0 且 wcos=0 为 V_dd 类恒耦合
# （env=恒 1）。gamma 为扩展空间长度 NonHermitian 对角（MHz）。
# ============================================================
from numba import njit as _njit

@_njit(cache=True, nogil=True)
def rhs_phi_pairs(t, phi, iis, jjs, vals, dDs, is_mw, wcos, n_pairs_w,
                  env_tab, env_dt, Cv, gamma=None,
                  ac_base=None, ac_env_tab=None, ac_env_dt=1.0):
    """φ-帧 RHS：−iC·[光场 env·Kc∘e^{iC·ΔD·t} + 微波 env·cos·A∘e^{iC·ΔD·t}
    + V_dd 当场]·φ + 逐波时变 AC 对角 −iC·Σ shifts·env²·φ + 耗散对角。"""
    tot = np.zeros_like(phi)
    ip = 0
    nw = len(wcos)
    for w in range(nw):
        np_w = n_pairs_w[w]
        if np_w == 0:
            continue
        ei = t / env_dt
        i0 = int(ei)
        if i0 >= env_tab.shape[1] - 1:
            i0 = env_tab.shape[1] - 2
        f = ei - i0
        envv = env_tab[w, i0] * (1.0 - f) + env_tab[w, i0 + 1] * f
        if envv == 0.0:
            ip += np_w
            continue
        if is_mw[w]:
            cw = envv * np.cos(Cv * wcos[w] * t)
        else:
            cw = envv
        for k in range(np_w):
            a = iis[ip + k]
            b = jjs[ip + k]
            v = vals[ip + k]
            ph = np.exp(1j * Cv * dDs[ip + k] * t)
            tot[a] += cw * v * ph * phi[b]
            tot[b] += cw * np.conj(v * ph) * phi[a]
        ip += np_w
    res = -1j * Cv * tot
    # 逐波时变 AC 对角 −iC·Σ_w shifts_w·env_w(t)²∘φ（env 平方 = AC∝光强；
    # K0 为纯 E——位移全量由 RHS 承担，t62e/t62f 定案）
    if ac_base is not None and ac_env_tab is not None:
        n_ac = ac_base.shape[0]
        for wi in range(n_ac):
            ei = t / ac_env_dt
            i0 = int(ei)
            if i0 >= ac_env_tab.shape[1] - 1:
                i0 = ac_env_tab.shape[1] - 2
            f = ei - i0
            envv = ac_env_tab[wi, i0] * (1.0 - f)                 + ac_env_tab[wi, i0 + 1] * f
            dev = envv * envv
            if dev != 0.0:
                for k in range(phi.shape[0]):
                    v = ac_base[wi, k]
                    if v != 0.0:
                        res[k] += -1j * Cv * dev * v * phi[k]
    if gamma is not None:
        for k in range(phi.shape[0]):
            if gamma[k] != 0.0:
                res[k] -= 0.5 * Cv * gamma[k] * phi[k]
    return res
