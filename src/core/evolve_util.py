# -*- coding: utf-8 -*-
"""演化公用工具层（t74）：全引擎共享的衰变/初态/耗散率/包络取值工具的唯一家园。

历史：原散落在 evolve_rwa.py 并经已删除的 evolve.py shim 被全库借用，
t74 归位本模块。

单位约定（与各引擎一致）：Γ 用 MHz；时间 ns；速率 = Γ×1e-3 /ns
（Lindblad 跳变算符不乘 2π，与 tempsimu 的 2π 修复一致）。
"""
import numpy as np

_CONV = 2 * np.pi * 1e-3

_POL_Q = {'π': 0, 'σ+': +1, 'σ-': -1}


def env_at(wf, t):
    """标量包络取值——收敛各引擎反复出现的
    float(np.asarray(wf(np.atleast_1d(t))[, float]).flat[0]) 惯用式。"""
    return float(np.asarray(wf(np.atleast_1d(t)), float).flat[0])


def _onehot(N, i):
    v = np.zeros(N, complex)
    v[i] = 1.0
    return v


def _as_vector(states, psi0):
    """初态 → 复向量。psi0 为：索引 | (n,l,j,m) | dict(n,l,j,m) | 复数组。"""
    N = len(states)
    if isinstance(psi0, np.ndarray):
        v = np.asarray(psi0, complex)
        if v.shape != (N,):
            raise ValueError(f'初态向量形状应为 ({N},)，实际 {v.shape}')
        return v
    if isinstance(psi0, (int, np.integer)):
        return _onehot(N, int(psi0))
    if isinstance(psi0, dict):
        psi0 = (psi0['n'], psi0['l'], psi0['j'], psi0['m'])
    if isinstance(psi0, (tuple, list)):
        if len(psi0) == N:
            return np.asarray(psi0, complex)
        if len(psi0) == 4:
            n, l, j, m = psi0
            for i, s in enumerate(states):
                if (abs(s['n'] - n) < 1e-9 and abs(s['l'] - l) < 1e-9
                        and abs(s['j'] - j) < 1e-9 and abs(s['m'] - m) < 1e-9):
                    return _onehot(N, i)
            raise ValueError(f'初态 (n,l,j,m)={psi0} 不在能级列表中')
        raise ValueError(f'无法解析初态 psi0: {psi0}')
    raise ValueError(f'无法解析初态 psi0: {psi0}')


def decay(g, gamma_mhz, e=None):
    """衰变通道工厂 → {'from': g, 'to': e, 'gamma_mhz': Γ}。

    物理：跳变算符 L = √Γ |e⟩⟨g|（Lindblad），Γ 单位 MHz；时间单位 ns，
    速率 = Γ×1e-3 /ns（与 tempsimu 的 2π 修复一致，L 不需乘 2π）。
    g  ：衰变态（states 索引 | (n,l,j,m) | dict(n,l,j,m)）
    e  ：落点（同上；None → 丢失：布居离开建模空间，累积为结果 loss）
    g==e：纯退相（|g⟩⟨g| 型测量算符，相干衰减速率 Γ/2，布居不变）
    """
    return {'from': g, 'to': e, 'gamma_mhz': float(gamma_mhz)}


def auto_decays(cache, states, tau_max_s=1e-4, temperature=0.0):
    """按 ARC 权威寿命自动配置自发辐射（Γ = 1/τ，落点 = 丢失）。

    对基组里每个态查 ARC `getStateLifetime(n,l,j,temperature)`——权威数据
    （由偶极矩阵元对全部下能级求和得到），比逐通道算 A 系数更精准便捷。
    只含寿命 τ ≤ tau_max_s 的态（默认 100 µs：5P/6S/6P/4D 等低激发态；
    排除里德堡 ~0.1 ms+ 的慢自发衰变——需要里德堡/黑体辐射时请用自定义）。
    每组 (n,l,j) 的每个子能级各一条通道（自发辐射率对 F/mJ 子能级相同）。
    temperature=0 → 纯自发辐射（不含黑体）；返回 decay() 列表。
    """
    out = []
    for i, s in enumerate(states):
        try:
            tau = cache.atom.getStateLifetime(int(s['n']), int(s['l']),
                                              float(s['j']),
                                              temperature=float(temperature))
        except Exception:
            continue                                   # 查不到 → 跳过
        if tau is None or not np.isfinite(float(tau)) \
                or float(tau) > tau_max_s:
            continue
        gam = 1e-6 / float(tau)                        # s⁻¹ → MHz
        if gam >= 1e-6:
            out.append(decay(i, gam, None))
    return out


def _resolve_state_idx(states, key):
    """能级定位：索引 | (n,l,j,m) | dict(n,l,j,m) → 下标；None → None。"""
    if key is None:
        return None
    if isinstance(key, (int, np.integer)):
        i = int(key)
        if not (0 <= i < len(states)):
            raise ValueError(f'能级索引 {i} 越界（N={len(states)}）')
        return i
    if isinstance(key, dict):
        key = (key['n'], key['l'], key['j'], key['m'])
    n, l, j, m = key
    for i, s in enumerate(states):
        if (abs(s['n'] - n) < 1e-9 and abs(s['l'] - l) < 1e-9
                and abs(s['j'] - j) < 1e-9 and abs(s['m'] - m) < 1e-9):
            return i
    raise ValueError(f'衰变端点 {key} 不在能级列表中')


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


def _single_loss_rates(states, decays):
    """单原子寿命通道 → 每态耗散率向量 Gamma_ns（/ns，落点=丢失）。

    与两体 _pair_loss_vectors 同语义：from 解析到索引，Γ（MHz）×1e-3
    = 速率 /ns；显式落点（to 在空间）/纯退相 → NotImplementedError
    （落点=丢失的确定性 NonHermitian 演化；其他请用 lindblad 引擎）。
    """
    Gamma = np.zeros(len(states))
    for ch in decays or []:
        g_idx = _resolve_state_idx(states, ch['from'])
        if g_idx is None:
            raise ValueError(
                f'hard 耗散：通道 from 无法解析到基组 ({ch["from"]!r})')
        to = ch.get('to')
        if to is not None and _resolve_state_idx(states, to) is not None:
            raise NotImplementedError(
                'hard 耗散仅支持落点=丢失（寿命通道）；显式落点 / 纯退相'
                '请用单原子 lindblad 引擎')
        gam_ns = float(ch['gamma_mhz']) * 1e-3          # MHz → /ns
        if gam_ns <= 0:
            continue
        Gamma[g_idx] += gam_ns
    return Gamma


def _loss_and_norm(pops, Gamma_ns, t):
    """从布居 + 耗散率（/ns）× 时间数组 → loss 数组 + 含 loss 范数偏差。

    loss(t) = ∫₀ᵗ pops(τ)·Gamma_ns dτ（梯形积分），
    norm_dev = max|Σpops(t) + loss(t) − 1|（无耗散时 Σpops≡1）。
    """
    if not np.any(Gamma_ns):
        return np.zeros(len(t)), float(np.max(np.abs(pops.sum(axis=1) - 1.0)))
    rate = pops @ Gamma_ns
    dt = np.diff(t)
    loss = np.zeros(len(t))
    loss[1:] = np.cumsum(0.5 * (rate[1:] + rate[:-1]) * dt)
    norm_dev = float(np.max(np.abs(pops.sum(axis=1) + loss - 1.0)))
    return loss, norm_dev
